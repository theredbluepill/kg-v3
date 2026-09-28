//! Minimal in-process C ABI for batched Python/Rust rollout integration.
//!
//! Legacy entry points transport JSON while retaining native `Game` objects.
//! Myolie entry points use checked caller-owned arrays and reusable native staging.
//! Every mutating batch call is transactional across all games.

use crate::joint_matching::{
    JointBenchRequest, JointRequest, benchmark_joint_rows, compile_joint_rows,
};
use crate::myolie_features::{self, FEATURE_COUNT};
use crate::native_agents::{NativeAgent, NativeAgentKind};
use crate::policy_rows::{
    CachedRow, MARKET_FACTORS, compact_policy_row_v1, pack_compact_rows_v1, pack_rows,
    policy_action, policy_row,
};
use crate::{
    ATTRIB_FIELDS, ATTRIB_VERSION, ECON_FIELDS, Game, MarketStepMetrics, StepSnapshot, TraceHeader,
    UnitStepMetrics,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

#[derive(Debug)]
pub struct ReBatch {
    games: Vec<Game>,
    /// Raw explicit-header clock validation, before Game derives day/hour.
    /// Consulted only by the opt-in observation schema; never changes old games.
    myolie_invest_errors: Vec<Option<String>>,
    /// Rows of the last `re_batch_policy_rows` call, in request order.
    policy_rows: Vec<CachedRow>,
    /// Stateful native external controllers, isolated per game and seat.
    native_agents: Vec<[NativeAgentSlot; 2]>,
    /// Reused staging keeps caller outputs unchanged if encoding or stepping fails.
    myolie_output: MyolieOutput,
    /// Explicit per-batch environment parallelism; None preserves serial execution.
    /// Never use Rayon's global pool or change a game's internal operation order.
    myolie_pool: Option<rayon::ThreadPool>,
}

#[derive(Debug, Default)]
struct MyolieOutput {
    features: Vec<f32>,
    context: Vec<i64>,
    banks: Vec<f64>,
    done: Vec<u8>,
}

#[derive(Clone, Debug)]
struct NativeAgentSlot {
    r04: NativeAgent,
    e776: NativeAgent,
    flex: NativeAgent,
    evgen: NativeAgent,
    ecobot: NativeAgent,
    starter: NativeAgent,
    tetsutani: NativeAgent,
    shoprouter: NativeAgent,
    v39: NativeAgent,
    v43: NativeAgent,
    v47: NativeAgent,
    v48: NativeAgent,
    farm2945: NativeAgent,
    tetsu65: NativeAgent,
    smaller: NativeAgent,
    tetsu342: NativeAgent,
    metav4: NativeAgent,
    pipe16: NativeAgent,
    v56: NativeAgent,
    cha22: NativeAgent,
    /// Myolie array collection: the controller that plays this seat inside
    /// `re_batch_myolie_step_frames` (None = the caller's frames) and the action
    /// it emitted on the last committed step (for trace/diagnostics).
    myolie_native: Option<NativeAgentKind>,
    myolie_last: Option<Value>,
    /// Myolie anchor-teacher collection: a controller that computes an action for
    /// this (learner) seat on every step from the same pre-step state, advancing
    /// its own memory; the action is executed only when `myolie_teacher_execute`
    /// is set for the step (the caller then passes length 0), otherwise the
    /// caller's frames are executed and the teacher action is only a label.
    myolie_teacher: Option<NativeAgentKind>,
    /// Opt-in executed-history memory; currently qualified only for Cha22.
    myolie_teacher_execution_aware: bool,
    myolie_teacher_execute: bool,
    myolie_teacher_last: Option<Value>,
}

impl Default for NativeAgentSlot {
    fn default() -> Self {
        Self {
            r04: NativeAgent::new(NativeAgentKind::R04),
            e776: NativeAgent::new(NativeAgentKind::E776),
            flex: NativeAgent::new(NativeAgentKind::Flex),
            evgen: NativeAgent::new(NativeAgentKind::Evgen),
            ecobot: NativeAgent::new(NativeAgentKind::Ecobot),
            starter: NativeAgent::new(NativeAgentKind::Starter),
            tetsutani: NativeAgent::new(NativeAgentKind::Tetsutani),
            shoprouter: NativeAgent::new(NativeAgentKind::ShopRouter),
            v39: NativeAgent::new(NativeAgentKind::V39),
            v43: NativeAgent::new(NativeAgentKind::V43),
            v47: NativeAgent::new(NativeAgentKind::V47),
            v48: NativeAgent::new(NativeAgentKind::V48),
            farm2945: NativeAgent::new(NativeAgentKind::Farm2945),
            tetsu65: NativeAgent::new(NativeAgentKind::Tetsu65),
            smaller: NativeAgent::new(NativeAgentKind::Smaller),
            tetsu342: NativeAgent::new(NativeAgentKind::Tetsu342),
            metav4: NativeAgent::new(NativeAgentKind::Metav4),
            pipe16: NativeAgent::new(NativeAgentKind::Pipe16),
            v56: NativeAgent::new(NativeAgentKind::V56),
            cha22: NativeAgent::new(NativeAgentKind::Cha22),
            myolie_native: None,
            myolie_last: None,
            myolie_teacher: None,
            myolie_teacher_execution_aware: false,
            myolie_teacher_execute: false,
            myolie_teacher_last: None,
        }
    }
}

impl NativeAgentSlot {
    fn get(&self, kind: NativeAgentKind) -> &NativeAgent {
        match kind {
            NativeAgentKind::R04 => &self.r04,
            NativeAgentKind::E776 => &self.e776,
            NativeAgentKind::Flex => &self.flex,
            NativeAgentKind::Evgen => &self.evgen,
            NativeAgentKind::Ecobot => &self.ecobot,
            NativeAgentKind::Starter => &self.starter,
            NativeAgentKind::Tetsutani => &self.tetsutani,
            NativeAgentKind::ShopRouter => &self.shoprouter,
            NativeAgentKind::V39 => &self.v39,
            NativeAgentKind::V43 => &self.v43,
            NativeAgentKind::V47 => &self.v47,
            NativeAgentKind::V48 => &self.v48,
            NativeAgentKind::Farm2945 => &self.farm2945,
            NativeAgentKind::Tetsu65 => &self.tetsu65,
            NativeAgentKind::Smaller => &self.smaller,
            NativeAgentKind::Tetsu342 => &self.tetsu342,
            NativeAgentKind::Metav4 => &self.metav4,
            NativeAgentKind::Pipe16 => &self.pipe16,
            NativeAgentKind::V56 => &self.v56,
            NativeAgentKind::Cha22 => &self.cha22,
        }
    }

    fn set(&mut self, kind: NativeAgentKind, agent: NativeAgent) {
        match kind {
            NativeAgentKind::R04 => self.r04 = agent,
            NativeAgentKind::E776 => self.e776 = agent,
            NativeAgentKind::Flex => self.flex = agent,
            NativeAgentKind::Evgen => self.evgen = agent,
            NativeAgentKind::Ecobot => self.ecobot = agent,
            NativeAgentKind::Starter => self.starter = agent,
            NativeAgentKind::Tetsutani => self.tetsutani = agent,
            NativeAgentKind::ShopRouter => self.shoprouter = agent,
            NativeAgentKind::V39 => self.v39 = agent,
            NativeAgentKind::V43 => self.v43 = agent,
            NativeAgentKind::V47 => self.v47 = agent,
            NativeAgentKind::V48 => self.v48 = agent,
            NativeAgentKind::Farm2945 => self.farm2945 = agent,
            NativeAgentKind::Tetsu65 => self.tetsu65 = agent,
            NativeAgentKind::Smaller => self.smaller = agent,
            NativeAgentKind::Tetsu342 => self.tetsu342 = agent,
            NativeAgentKind::Metav4 => self.metav4 = agent,
            NativeAgentKind::Pipe16 => self.pipe16 = agent,
            NativeAgentKind::V56 => self.v56 = agent,
            NativeAgentKind::Cha22 => self.cha22 = agent,
        }
    }
}

fn native_agent_slots(count: usize) -> Vec<[NativeAgentSlot; 2]> {
    (0..count)
        .map(|_| [NativeAgentSlot::default(), NativeAgentSlot::default()])
        .collect()
}

#[derive(Deserialize)]
struct MyolieNativeAssignment {
    game: usize,
    seat: usize,
    kind: Option<NativeAgentKind>,
}

#[derive(Deserialize)]
struct MyolieTeacherAssignment {
    game: usize,
    seat: usize,
    kind: Option<NativeAgentKind>,
    #[serde(default)]
    execution_aware: bool,
}

#[derive(Deserialize)]
struct NativeActionRequest {
    game: usize,
    seat: usize,
    kind: NativeAgentKind,
}

#[derive(Serialize)]
struct BatchStepDoneMetrics {
    done: Vec<bool>,
    market: MarketStepMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    market_live: Option<MarketStepMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    market_opponent: Option<MarketStepMetrics>,
    unit: UnitStepMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_live: Option<UnitStepMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit_opponent: Option<UnitStepMetrics>,
    /// Per game, per seat: cash after the step, conservative mark-to-market wealth
    /// after the step, and committed production actions during the step.  Read-only
    /// telemetry for JA26 potential-based credit; never part of the transition.
    money: Vec<Vec<f64>>,
    wealth: Vec<Vec<f64>>,
    production: Vec<Vec<u64>>,
}

#[derive(Serialize)]
struct BatchSeatWealth {
    money: Vec<Vec<f64>>,
    wealth: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BatchStepMetricsRequest {
    Actions(Vec<Vec<Value>>),
    WithLiveSeats {
        actions: Vec<Vec<Value>>,
        live_seats: Vec<usize>,
    },
}

/// Owned byte buffer returned across the C ABI. Pass it exactly once to
/// `re_buffer_free`; most exports return UTF-8 JSON or diagnostics. The sampler
/// plan export explicitly returns a versioned little-endian binary payload.
#[repr(C)]
pub struct ReBuffer {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

impl ReBuffer {
    fn empty() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }

    fn from_vec(mut bytes: Vec<u8>) -> Self {
        let buffer = Self {
            data: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
        std::mem::forget(bytes);
        buffer
    }
}

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into());
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "Rust panic crossed the rollout boundary".to_string()
    }
}

fn ffi_result<T>(fallback: T, operation: impl FnOnce() -> Result<T, String>) -> T {
    clear_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            set_error(error);
            fallback
        },
        Err(payload) => {
            set_error(panic_message(payload));
            fallback
        },
    }
}

unsafe fn input_bytes<'a>(data: *const u8, len: usize) -> Result<&'a [u8], String> {
    if data.is_null() {
        if len == 0 {
            return Ok(&[]);
        }
        return Err("null input pointer with nonzero length".to_string());
    }
    // SAFETY: the caller promises that `data` points to `len` readable bytes for the
    // duration of this call. Python's ctypes wrapper holds the source buffer alive.
    Ok(unsafe { slice::from_raw_parts(data, len) })
}

unsafe fn batch_mut<'a>(batch: *mut ReBatch) -> Result<&'a mut ReBatch, String> {
    // SAFETY: constructors return a unique heap allocation and the wrapper serializes
    // calls, retaining ownership until `re_batch_free`.
    unsafe { batch.as_mut() }.ok_or_else(|| "null ReBatch pointer".to_string())
}

unsafe fn batch_ref<'a>(batch: *const ReBatch) -> Result<&'a ReBatch, String> {
    // SAFETY: see `batch_mut`; this call only borrows the allocation immutably.
    unsafe { batch.as_ref() }.ok_or_else(|| "null ReBatch pointer".to_string())
}

fn games_from_headers(headers: &[TraceHeader]) -> Result<Vec<Game>, String> {
    if headers.is_empty() {
        return Err("a rollout batch must contain at least one header".to_string());
    }
    headers.iter().map(Game::from_seed_header).collect()
}

fn games_from_explicit_headers(headers: &[TraceHeader]) -> Result<Vec<Game>, String> {
    if headers.is_empty() {
        return Err("an explicit rollout batch must contain at least one header".to_string());
    }
    headers.iter().map(Game::from_header).collect()
}

fn encode_snapshots(games: &[Game]) -> Result<ReBuffer, String> {
    let snapshots: Vec<StepSnapshot> = games.iter().map(Game::snapshot).collect();
    serde_json::to_vec(&snapshots)
        .map(ReBuffer::from_vec)
        .map_err(|error| format!("serialize batch snapshots: {error}"))
}

const MYOLIE_MAX_ACTORS: usize = 241;
const MYOLIE_MAX_FRAMES: usize = 252;
const MYOLIE_SLOTS: usize = 12;
const MYOLIE_WIDTHS: [i16; MYOLIE_SLOTS] = [241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2];
const MYOLIE_UNIT_NAMES: [&str; 19] = [
    "NONE",
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "DROP",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "FERTILIZE",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
];
const MYOLIE_MARKET_NAMES: [&str; 8] = [
    "NONE",
    "HIRE",
    "BUY_LAND",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "EMPTY",
];
const MYOLIE_ITEMS: [&str; 13] = [
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
];

/// Version 1 fixes all dimensions, dtypes and canonical frame semantics below.
#[unsafe(no_mangle)]
pub extern "C" fn re_myolie_array_abi_version() -> u32 {
    1
}

/// Build a real native-reset trace header from configuration and an integer seed.
/// This cold-path adapter avoids a Python engine dependency or fabricated initial
/// states when using the existing transactional batch/reset API.
///
/// # Safety
/// `data` must identify `len` readable bytes for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_native_seed_header(data: *const u8, len: usize) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        let bytes = unsafe { input_bytes(data, len) }?;
        let request: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let config: crate::Config =
            serde_json::from_value(request["configuration"].clone()).map_err(|e| e.to_string())?;
        let seed = request["seed"]
            .as_number()
            .ok_or("seed must be an integer")?
            .clone();
        let decimal = seed.to_string();
        if decimal.contains(['.', 'e', 'E']) {
            return Err("seed must be an integer".into());
        }
        let game = Game::new_with_seed_decimal(config.clone(), &decimal, 2)?;
        let snapshot = game.snapshot();
        let header = TraceHeader {
            format: crate::TRACE_FORMAT.into(),
            seed,
            configuration: config,
            shop_schedule: Vec::new(),
            rng_schedule: Vec::new(),
            initial: crate::InitialState {
                public: snapshot.public,
                privates: snapshot.privates,
            },
            terminal_banks: vec![0.0, 0.0],
            transitions: 0,
        };
        serde_json::to_vec(&header)
            .map(ReBuffer::from_vec)
            .map_err(|e| e.to_string())
    })
}

/// Explicit opt-in suffix capability. Existing array ABI1 remains unchanged.
#[unsafe(no_mangle)]
pub extern "C" fn re_myolie_invest_observation_version() -> u32 {
    1
}

/// Native Myolie grammar DFA. Little-endian header: magic 0x4D534731, ABI 1,
/// node count, start node 0, edge count (five u32). Each node is four u32:
/// slot, pending HIRE requests before this frame, edge offset, vocabulary width.
/// Then edge_count u8 mask entries followed by edge_count i32 next-node entries;
/// -1 is terminal STOP and -2 is an invalid token. Free with re_buffer_free.
#[unsafe(no_mangle)]
pub extern "C" fn re_myolie_sampler_plan(actors: u32, limit: u32, hire_limit: u32) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        crate::myolie_sampler::plan(actors, limit, hire_limit).map(ReBuffer::from_vec)
    })
}

/// Validate and decode a complete canonical frame program into JSON action.
/// frames_len counts i16 tokens, not frames/bytes; every frame has 12 tokens.
/// The caller retains readable aligned storage for the duration of this call.
/// Checkpoint HIRE support is validated in addition to the syntax/shape grammar.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_myolie_sampler_decode(
    actors: u32,
    limit: u32,
    hire_limit: u32,
    frames: *const i16,
    frames_len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        crate::myolie_sampler::validate_decode_length(actors, limit, hire_limit, frames_len)?;
        array_region(frames, frames_len, frames_len, "sampler frames")?;
        // SAFETY: scalar/length/alignment checks precede slice construction;
        // readable allocation and lifetime are the documented caller contract.
        let frames = unsafe { slice::from_raw_parts(frames, frames_len) };
        let action = crate::myolie_sampler::decode(actors, limit, hire_limit, frames)?;
        serde_json::to_vec(&action)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize Myolie sampler action: {error}"))
    })
}

/// Check a raw buffer before making a Rust slice. The caller still owns the
/// allocation/lifetime contract: an address cannot establish that memory exists.
fn array_region<T>(
    pointer: *const T,
    len: usize,
    expected: usize,
    name: &str,
) -> Result<(usize, usize), String> {
    if len != expected {
        return Err(format!("{name}: expected {expected} elements, got {len}"));
    }
    if pointer.is_null() && len != 0 {
        return Err(format!("{name}: null pointer"));
    }
    if !(pointer as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(format!("{name}: misaligned pointer"));
    }
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| format!("{name}: byte length overflows"))?;
    let end = (pointer as usize)
        .checked_add(bytes)
        .ok_or_else(|| format!("{name}: address overflows"))?;
    Ok((pointer as usize, end))
}

fn disjoint_regions(regions: &[(usize, usize)]) -> Result<(), String> {
    for (i, &(start, end)) in regions.iter().enumerate() {
        for &(other_start, other_end) in &regions[..i] {
            if start < other_end && other_start < end {
                return Err("Myolie array buffers must not overlap".to_string());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct MyoliePointers {
    features: *mut f32,
    context: *mut i64,
    banks: *mut f64,
    done: *mut u8,
}

impl MyoliePointers {
    fn validate(self, games: usize, lengths: [usize; 4]) -> Result<Vec<(usize, usize)>, String> {
        self.validate_version(games, lengths, 1)
    }

    fn validate_version(
        self,
        games: usize,
        lengths: [usize; 4],
        version: u32,
    ) -> Result<Vec<(usize, usize)>, String> {
        let width = myolie_feature_width(version)?;
        let features = games
            .checked_mul(2 * width)
            .ok_or("Myolie feature count overflow")?;
        let context = games
            .checked_mul(4)
            .ok_or("Myolie context count overflow")?;
        let banks = games.checked_mul(2).ok_or("Myolie bank count overflow")?;
        let regions = vec![
            array_region(self.features, lengths[0], features, "features")?,
            array_region(self.context, lengths[1], context, "context")?,
            array_region(self.banks, lengths[2], banks, "banks")?,
            array_region(self.done, lengths[3], games, "done")?,
        ];
        disjoint_regions(&regions)?;
        Ok(regions)
    }

    unsafe fn publish(self, staged: &MyolieOutput) {
        // SAFETY: caller supplies valid writable disjoint arrays; all sizes and
        // alignment were checked before staging. No fallible operation follows.
        unsafe {
            ptr::copy_nonoverlapping(
                staged.features.as_ptr(),
                self.features,
                staged.features.len(),
            );
            ptr::copy_nonoverlapping(staged.context.as_ptr(), self.context, staged.context.len());
            ptr::copy_nonoverlapping(staged.banks.as_ptr(), self.banks, staged.banks.len());
            ptr::copy_nonoverlapping(staged.done.as_ptr(), self.done, staged.done.len());
        }
    }
}

// Cold-path provenance: explicit Game import derives day/hour from step and
// configuration. Preserve consistency rejection before those raw fields are
// discarded. Config's existing strict integer parser already rejects fractional
// multipliers, strings and booleans. No JSON is parsed in per-turn collection.
fn investment_header_errors(bytes: &[u8], explicit: bool) -> Result<Vec<Option<String>>, String> {
    let headers: Vec<Value> = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    Ok(headers
        .iter()
        .map(|header| {
            if !explicit {
                return None;
            }
            let public = &header["initial"]["public"];
            let step = public["step"].as_u64();
            let day = public["day"].as_u64();
            let hour = public["hour"].as_u64();
            let turns = header["configuration"]
                .get("turnsPerDay")
                .and_then(Value::as_f64)
                .unwrap_or(24.0);
            match (step, day, hour) {
                (Some(step), Some(day), Some(hour))
                    if day as f64 == (step as f64 / turns).floor()
                        && hour as f64 == step as f64 % turns =>
                {
                    None
                },
                _ => Some(
                    "investment raw day/hour are inconsistent with step and turnsPerDay".into(),
                ),
            }
        })
        .collect())
}

fn validate_investment_headers(errors: &[Option<String>], version: u32) -> Result<(), String> {
    if version == 2 {
        if let Some((index, error)) = errors
            .iter()
            .enumerate()
            .find_map(|(i, e)| e.as_ref().map(|e| (i, e)))
        {
            return Err(format!("game {index}: {error}"));
        }
    }
    Ok(())
}

fn myolie_feature_width(version: u32) -> Result<usize, String> {
    match version {
        1 => Ok(FEATURE_COUNT),
        2 => Ok(myolie_features::INVEST_FEATURE_COUNT),
        _ => Err(format!("unsupported Myolie observation version {version}")),
    }
}

impl MyolieOutput {
    #[cfg(test)]
    fn encode(&mut self, games: &[Game]) -> Result<(), String> {
        self.encode_version(games, 1, None)
    }

    fn encode_version(
        &mut self,
        games: &[Game],
        version: u32,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<(), String> {
        let width = myolie_feature_width(version)?;
        self.features.resize(games.len() * 2 * width, 0.0);
        self.context.resize(games.len() * 4, 0);
        self.banks.resize(games.len() * 2, 0.0);
        self.done.resize(games.len(), 0);
        if let Some(pool) = pool {
            // Indexed collection joins every task and retains serial error order.
            // Writes touch disjoint native staging slices, never caller buffers.
            let results: Vec<Result<(), String>> = pool.install(|| {
                games
                    .par_iter()
                    .zip(self.features.par_chunks_mut(2 * width))
                    .zip(self.context.par_chunks_mut(4))
                    .zip(self.banks.par_chunks_mut(2))
                    .zip(self.done.par_iter_mut())
                    .enumerate()
                    .map(|(index, ((((game, features), context), banks), done))| {
                        encode_myolie_game(
                            game, index, width, version, features, context, banks, done,
                        )
                    })
                    .collect()
            });
            for result in results {
                result?;
            }
        } else {
            for (index, game) in games.iter().enumerate() {
                encode_myolie_game(
                    game,
                    index,
                    width,
                    version,
                    &mut self.features[index * 2 * width..(index + 1) * 2 * width],
                    &mut self.context[index * 4..(index + 1) * 4],
                    &mut self.banks[index * 2..(index + 1) * 2],
                    &mut self.done[index],
                )?;
            }
        }
        Ok(())
    }
}

// The serial and parallel routes call the same per-game arithmetic in the same order.
#[allow(clippy::too_many_arguments)]
fn encode_myolie_game(
    game: &Game,
    index: usize,
    width: usize,
    version: u32,
    features: &mut [f32],
    context: &mut [i64],
    banks: &mut [f64],
    done: &mut u8,
) -> Result<(), String> {
    let limit = game.config.max_market_orders_per_turn.capped_usize();
    if !(1..=10).contains(&limit) || game.farms.len() != 2 {
        return Err("Myolie arrays require two seats and at most 10 market orders".to_string());
    }
    context[0] = i64::try_from(game.step).map_err(|_| "Myolie step exceeds int64")?;
    context[3] = limit as i64;
    for seat in 0..2 {
        let actors = 1 + game.farms[seat].hands.len();
        if actors > MYOLIE_MAX_ACTORS {
            return Err(format!(
                "Myolie arrays represent at most {MYOLIE_MAX_ACTORS} actors"
            ));
        }
        context[1 + seat] = actors as i64;
        banks[seat] = game.farms[seat].money;
        if !banks[seat].is_finite() {
            return Err(format!("game {index}, seat {seat}: nonfinite Myolie bank"));
        }
        let start = seat * width;
        let encode = if version == 1 {
            myolie_features::encode
        } else {
            myolie_features::encode_invest
        };
        encode(game, seat, &mut features[start..start + width])
            .map_err(|error| format!("game {index}, seat {seat}: {error}"))?;
        if let Some(feature) = features[start..start + width]
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(format!(
                "game {index}, seat {seat}: nonfinite Myolie feature {feature}"
            ));
        }
    }
    *done = u8::from(game.done);
    Ok(())
}

fn step_myolie_candidates(
    games: &mut [Game],
    actions: &[Option<Vec<Value>>],
    pool: Option<&rayon::ThreadPool>,
) -> Result<(), String> {
    if let Some(pool) = pool {
        let results: Vec<Result<(), String>> = pool.install(|| {
            games
                .par_iter_mut()
                .zip(actions.par_iter())
                .map(|(game, action)| {
                    if let Some(joint) = action {
                        game.step(joint)
                    } else {
                        Ok(())
                    }
                })
                .collect()
        });
        for result in results {
            result?;
        }
    } else {
        for (game, action) in games.iter_mut().zip(actions) {
            if let Some(joint) = action {
                game.step(joint)?;
            }
        }
    }
    Ok(())
}

/// Decode the canonical actor241 frame layout without JSON serialization. This
/// checks syntax/ordering, never affordability or checkpoint-owned HIRE support.
fn myolie_action(game: &Game, seat: usize, program: &[i16], length: i32) -> Result<Value, String> {
    let actors = 1 + game.farms[seat].hands.len();
    let limit = game.config.max_market_orders_per_turn.capped_usize();
    if actors > MYOLIE_MAX_ACTORS || !(1..=10).contains(&limit) {
        return Err("Myolie action exceeds actor/order representation".to_string());
    }
    let length = usize::try_from(length).map_err(|_| "negative Myolie frame length")?;
    if length < actors + 1 || length > actors + limit + 1 || length > MYOLIE_MAX_FRAMES {
        return Err(format!(
            "invalid Myolie frame length {length} for {actors} actors and {limit} orders"
        ));
    }
    let mut units = Vec::with_capacity(actors);
    let mut markets = Vec::with_capacity(length - actors - 1);
    for (index, frame) in program[..length * MYOLIE_SLOTS]
        .chunks_exact(MYOLIE_SLOTS)
        .enumerate()
    {
        for (slot, (&token, &width)) in frame.iter().zip(&MYOLIE_WIDTHS).enumerate() {
            if token < 0 || token >= width {
                return Err(format!(
                    "frame {index} slot {slot} token {token} outside vocabulary"
                ));
            }
        }
        if index < actors {
            let kind = frame[1] as usize;
            let item = frame[3] as usize;
            let transfer = kind == 6 || kind == 7;
            let allowed_item = if transfer {
                (1..=12).contains(&item)
            } else if kind == 8 {
                (1..=5).contains(&item)
            } else {
                item == 0
            };
            if frame[0] as usize != index
                || !(1..=18).contains(&kind)
                || frame[2] != 0
                || !allowed_item
                || frame[7..].iter().any(|&v| v != 0)
                || (!transfer && frame[4..7].iter().any(|&v| v != 0))
                || (frame[4] == 0 && (frame[5] != 0 || frame[6] != 0))
                || (frame[4] == 1 && frame[5] == 0 && frame[6] == 0)
            {
                return Err(format!("frame {index} violates canonical unit syntax"));
            }
            let mut command = vec![Value::String(MYOLIE_UNIT_NAMES[kind].to_string())];
            if transfer || kind == 8 {
                command.push(Value::String(MYOLIE_ITEMS[item].to_string()));
            }
            if transfer && frame[4] == 1 {
                command.push(Value::from(i64::from(frame[5]) * 32 + i64::from(frame[6])));
            }
            units.push(Value::Array(command));
        } else if index == length - 1 {
            if frame[..11].iter().any(|&v| v != 0) || frame[11] != 1 {
                return Err("Myolie program lacks distinct final STOP".to_string());
            }
        } else {
            let kind = frame[7] as usize;
            let item = frame[8] as usize;
            let quantity = (3..=6).contains(&kind);
            let allowed_item = match kind {
                3 => (1..=5).contains(&item),
                4 => item == 1 || item == 9,
                5 => (10..=12).contains(&item),
                6 => (1..=9).contains(&item),
                _ => item == 0,
            };
            if frame[..7].iter().any(|&v| v != 0)
                || !(1..=7).contains(&kind)
                || frame[11] != 0
                || !allowed_item
                || (!quantity && (frame[9] != 0 || frame[10] != 0))
            {
                return Err(format!("frame {index} violates canonical market syntax"));
            }
            let mut command = if kind == 7 {
                Vec::new()
            } else {
                vec![Value::String(MYOLIE_MARKET_NAMES[kind].to_string())]
            };
            if quantity {
                command.push(Value::String(MYOLIE_ITEMS[item].to_string()));
                command.push(Value::from(i64::from(frame[9]) * 32 + i64::from(frame[10])));
            }
            markets.push(Value::Array(command));
        }
    }
    let mut raw = serde_json::Map::new();
    raw.insert("farmer".to_string(), units.remove(0));
    raw.insert("hands".to_string(), Value::Array(units));
    raw.insert("market".to_string(), Value::Array(markets));
    Ok(Value::Object(raw))
}

/// Fill caller-owned float32 features[n,2,8165], int64 context[n,4], float64
/// banks[n,2] and uint8 done[n]. Lengths are element counts, not byte counts.
/// Context columns are step, seat0 actor count, seat1 actor count, order limit.
/// Buffers must be aligned, writable and nonoverlapping; retain them until return.
/// Errors leave every caller buffer unchanged. Serialize calls on each batch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_observe(
    batch: *mut ReBatch,
    features: *mut f32,
    features_len: usize,
    context: *mut i64,
    context_len: usize,
    banks: *mut f64,
    banks_len: usize,
    done: *mut u8,
    done_len: usize,
) -> bool {
    // SAFETY: forwards the documented checked array contract.
    unsafe {
        re_batch_myolie_observe_versioned(
            batch,
            features,
            features_len,
            context,
            context_len,
            banks,
            banks_len,
            done,
            done_len,
            1,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_observe_v2(
    batch: *mut ReBatch,
    features: *mut f32,
    features_len: usize,
    context: *mut i64,
    context_len: usize,
    banks: *mut f64,
    banks_len: usize,
    done: *mut u8,
    done_len: usize,
) -> bool {
    // SAFETY: forwards the documented checked array contract.
    unsafe {
        re_batch_myolie_observe_versioned(
            batch,
            features,
            features_len,
            context,
            context_len,
            banks,
            banks_len,
            done,
            done_len,
            2,
        )
    }
}

unsafe fn re_batch_myolie_observe_versioned(
    batch: *mut ReBatch,
    features: *mut f32,
    features_len: usize,
    context: *mut i64,
    context_len: usize,
    banks: *mut f64,
    banks_len: usize,
    done: *mut u8,
    done_len: usize,
    observation_version: u32,
) -> bool {
    ffi_result(false, || {
        // SAFETY: batch ownership and serialization are the existing ABI contract.
        let batch = unsafe { batch_mut(batch)? };
        validate_investment_headers(&batch.myolie_invest_errors, observation_version)?;
        let pointers = MyoliePointers {
            features,
            context,
            banks,
            done,
        };
        if observation_version == 1 {
            pointers.validate(
                batch.games.len(),
                [features_len, context_len, banks_len, done_len],
            )?;
        } else {
            pointers.validate_version(
                batch.games.len(),
                [features_len, context_len, banks_len, done_len],
                observation_version,
            )?;
        }
        batch.myolie_output.encode_version(
            &batch.games,
            observation_version,
            batch.myolie_pool.as_ref(),
        )?;
        // SAFETY: checked output lengths and alignment; caller owns allocation validity.
        unsafe { pointers.publish(&batch.myolie_output) };
        Ok(true)
    })
}

/// Transactional canonical-frame step followed by array observation. Input is
/// int16 frames[n,2,252,12] and int32 lengths[n,2]. Output contract is observe's.
/// No inputs/outputs may overlap. Padding after each length is ignored. A done
/// game permits both lengths=0 as a no-op; live games always require valid frames.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_step_frames(
    batch: *mut ReBatch,
    frames: *const i16,
    frames_len: usize,
    lengths: *const i32,
    lengths_len: usize,
    features: *mut f32,
    features_len: usize,
    context: *mut i64,
    context_len: usize,
    banks: *mut f64,
    banks_len: usize,
    done: *mut u8,
    done_len: usize,
) -> bool {
    // SAFETY: forwards the documented checked array contract.
    unsafe {
        re_batch_myolie_step_frames_versioned(
            batch,
            frames,
            frames_len,
            lengths,
            lengths_len,
            features,
            features_len,
            context,
            context_len,
            banks,
            banks_len,
            done,
            done_len,
            1,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_step_frames_v2(
    batch: *mut ReBatch,
    frames: *const i16,
    frames_len: usize,
    lengths: *const i32,
    lengths_len: usize,
    features: *mut f32,
    features_len: usize,
    context: *mut i64,
    context_len: usize,
    banks: *mut f64,
    banks_len: usize,
    done: *mut u8,
    done_len: usize,
) -> bool {
    // SAFETY: forwards the documented checked array contract.
    unsafe {
        re_batch_myolie_step_frames_versioned(
            batch,
            frames,
            frames_len,
            lengths,
            lengths_len,
            features,
            features_len,
            context,
            context_len,
            banks,
            banks_len,
            done,
            done_len,
            2,
        )
    }
}

unsafe fn re_batch_myolie_step_frames_versioned(
    batch: *mut ReBatch,
    frames: *const i16,
    frames_len: usize,
    lengths: *const i32,
    lengths_len: usize,
    features: *mut f32,
    features_len: usize,
    context: *mut i64,
    context_len: usize,
    banks: *mut f64,
    banks_len: usize,
    done: *mut u8,
    done_len: usize,
    observation_version: u32,
) -> bool {
    ffi_result(false, || {
        // SAFETY: batch ownership and serialization follow the existing ABI.
        let batch = unsafe { batch_mut(batch)? };
        validate_investment_headers(&batch.myolie_invest_errors, observation_version)?;
        let n = batch.games.len();
        let pointers = MyoliePointers {
            features,
            context,
            banks,
            done,
        };
        let mut regions = pointers.validate_version(
            n,
            [features_len, context_len, banks_len, done_len],
            observation_version,
        )?;
        let expected_frames = n
            .checked_mul(2 * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS)
            .ok_or("Myolie frame count overflow")?;
        regions.push(array_region(frames, frames_len, expected_frames, "frames")?);
        regions.push(array_region(lengths, lengths_len, n * 2, "lengths")?);
        disjoint_regions(&regions)?;
        // SAFETY: pointers are aligned, non-null and lengths were checked; caller
        // promises readable input memory held alive throughout this call.
        let frames = unsafe { slice::from_raw_parts(frames, frames_len) };
        let lengths = unsafe { slice::from_raw_parts(lengths, lengths_len) };
        let mut actions = Vec::with_capacity(n);
        // Seats assigned a native controller act from it (their length must be 0);
        // advanced controllers are committed only after every game steps.
        let mut natives: Vec<(usize, usize, NativeAgentKind, NativeAgent)> = Vec::new();
        let mut teachers: Vec<(usize, usize, NativeAgentKind, NativeAgent, Value)> = Vec::new();
        for (index, game) in batch.games.iter().enumerate() {
            if game.done && lengths[index * 2] == 0 && lengths[index * 2 + 1] == 0 {
                actions.push(None);
                continue;
            }
            if game.done {
                return Err(format!(
                    "game {index} is done; use zero lengths or reset it"
                ));
            }
            let mut joint = Vec::with_capacity(2);
            for seat in 0..2 {
                let slot = &batch.native_agents[index][seat];
                if let Some(kind) = slot.myolie_native {
                    if lengths[index * 2 + seat] != 0 {
                        return Err(format!(
                            "game {index}, seat {seat}: a native seat requires length 0"
                        ));
                    }
                    let mut agent = slot.get(kind).clone();
                    joint.push(
                        agent
                            .action(game, seat)
                            .map_err(|error| format!("game {index}, seat {seat}: {error}"))?,
                    );
                    natives.push((index, seat, kind, agent));
                    continue;
                }
                if let Some(kind) = slot.myolie_teacher {
                    let mut agent = slot.get(kind).clone();
                    let label = agent
                        .action(game, seat)
                        .map_err(|error| format!("game {index}, seat {seat} teacher: {error}"))?;
                    if slot.myolie_teacher_execute {
                        if lengths[index * 2 + seat] != 0 {
                            return Err(format!(
                                "game {index}, seat {seat}: an executing teacher seat requires length 0"
                            ));
                        }
                        joint.push(label.clone());
                        teachers.push((index, seat, kind, agent, label));
                        continue;
                    }
                    teachers.push((index, seat, kind, agent, label));
                }
                let start = (index * 2 + seat) * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS;
                joint.push(
                    myolie_action(
                        game,
                        seat,
                        &frames[start..start + MYOLIE_MAX_FRAMES * MYOLIE_SLOTS],
                        lengths[index * 2 + seat],
                    )
                    .map_err(|error| format!("game {index}, seat {seat}: {error}"))?,
                );
            }
            actions.push(Some(joint));
        }
        let mut candidates = batch.games.clone();
        step_myolie_candidates(&mut candidates, &actions, batch.myolie_pool.as_ref())?;
        // Reconcile only tentative controllers after the whole joint step has
        // succeeded. Use the submitted action, not a predicted successful
        // effect: failures/partial settlement are observed on the next turn.
        // No controller or game is published until output encoding also passes.
        for (index, seat, kind, agent, label) in &mut teachers {
            let slot = &batch.native_agents[*index][*seat];
            if slot.myolie_teacher_execution_aware {
                let executed = &actions[*index].as_ref().expect("live teacher game")[*seat];
                if executed != label {
                    let game = &batch.games[*index];
                    let mut obs = serde_json::to_value(game.snapshot().public)
                        .map_err(|error| error.to_string())?;
                    obs["player"] = serde_json::json!(*seat);
                    obs["private"] = serde_json::to_value(&game.privates[*seat])
                        .map_err(|error| error.to_string())?;
                    match (agent, slot.get(*kind)) {
                        (NativeAgent::Cha22(candidate), NativeAgent::Cha22(before)) => {
                            candidate.commit_executed(before, &obs, label, executed)
                        },
                        _ => return Err("unsupported execution-aware teacher".into()),
                    }
                }
            }
        }
        batch.myolie_output.encode_version(
            &candidates,
            observation_version,
            batch.myolie_pool.as_ref(),
        )?;
        batch.games = candidates;
        batch.policy_rows.clear();
        for slots in batch.native_agents.iter_mut() {
            for slot in slots.iter_mut() {
                slot.myolie_last = None;
                slot.myolie_teacher_last = None;
            }
        }
        for (index, seat, kind, agent, label) in teachers {
            let slot = &mut batch.native_agents[index][seat];
            slot.set(kind, agent);
            slot.myolie_teacher_last = Some(label);
        }
        for (index, seat, kind, agent) in natives {
            let slot = &mut batch.native_agents[index][seat];
            slot.set(kind, agent);
            if let Some(joint) = &actions[index] {
                slot.myolie_last = Some(joint[seat].clone());
            }
        }
        // SAFETY: all checks succeeded before committing; publishing cannot fail.
        unsafe { pointers.publish(&batch.myolie_output) };
        Ok(true)
    })
}

/// Assign (kind) or clear (null) the native controller that plays a seat inside
/// `re_batch_myolie_step_frames`; JSON `[{"game", "seat", "kind"}]`, a cold path
/// used after resets. Controller state is untouched: a reset slot starts fresh.
/// All requests are validated before any assignment changes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_assign_native(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: ownership/readability follow the batch and input contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<MyolieNativeAssignment> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse native assignments: {error}"))?;
        for request in &requests {
            if request.seat >= 2 || request.game >= batch.games.len() {
                return Err(format!(
                    "native assignment game {} seat {} outside batch of {}",
                    request.game,
                    request.seat,
                    batch.games.len()
                ));
            }
        }
        for request in &requests {
            if request.kind.is_some()
                && batch.native_agents[request.game][request.seat]
                    .myolie_teacher
                    .is_some()
            {
                return Err(format!(
                    "game {} seat {} has a teacher; it cannot also be a native seat",
                    request.game, request.seat
                ));
            }
        }
        for request in requests {
            batch.native_agents[request.game][request.seat].myolie_native = request.kind;
        }
        Ok(true)
    })
}

/// Capability marker: old libraries ignore unknown assignment JSON fields, so
/// wrappers must require this symbol before requesting executed-history memory.
#[unsafe(no_mangle)]
pub extern "C" fn re_myolie_execution_aware_teacher_version() -> u32 {
    1
}

/// Assign (kind) or clear (null) the teacher controller of a learner seat; JSON
/// `[{"game", "seat", "kind"}]` (cold path, after resets). A seat cannot have both
/// a native controller and a teacher. All requests are validated first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_assign_teacher(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: ownership/readability follow the batch and input contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<MyolieTeacherAssignment> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse teacher assignments: {error}"))?;
        for request in &requests {
            if request.execution_aware
                && request.kind.is_some()
                && request.kind != Some(NativeAgentKind::Cha22)
            {
                return Err("execution-aware teacher memory is supported only for cha22".into());
            }
            if request.seat >= 2 || request.game >= batch.games.len() {
                return Err(format!(
                    "teacher assignment game {} seat {} outside batch of {}",
                    request.game,
                    request.seat,
                    batch.games.len()
                ));
            }
            if request.kind.is_some()
                && batch.native_agents[request.game][request.seat]
                    .myolie_native
                    .is_some()
            {
                return Err(format!(
                    "game {} seat {} is a native seat; it cannot also have a teacher",
                    request.game, request.seat
                ));
            }
        }
        for request in requests {
            let slot = &mut batch.native_agents[request.game][request.seat];
            slot.myolie_teacher = request.kind;
            slot.myolie_teacher_execution_aware = request.kind.is_some() && request.execution_aware;
            slot.myolie_teacher_execute = false;
            slot.myolie_teacher_last = None;
        }
        Ok(true)
    })
}

/// Per-step execute flags for teacher seats: uint8 flags[n, 2] (1 = the teacher's
/// action is executed this step; the seat must then be given length 0). A flag on
/// a seat without a teacher is rejected; flags persist until set again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_teacher_execute(
    batch: *mut ReBatch,
    flags: *const u8,
    flags_len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: ownership follows the batch contract; the region is validated.
        let batch = unsafe { batch_mut(batch)? };
        let n = batch.games.len();
        array_region(flags, flags_len, n * 2, "teacher execute flags")?;
        let flags = unsafe { slice::from_raw_parts(flags, flags_len) };
        for index in 0..n {
            for seat in 0..2 {
                let flag = flags[index * 2 + seat];
                if flag > 1 {
                    return Err(format!(
                        "game {index} seat {seat}: teacher execute flag {flag} is not 0/1"
                    ));
                }
                let execute = flag == 1;
                if execute && batch.native_agents[index][seat].myolie_teacher.is_none() {
                    return Err(format!(
                        "game {index} seat {seat}: execute flag without a teacher"
                    ));
                }
            }
        }
        for index in 0..n {
            for seat in 0..2 {
                batch.native_agents[index][seat].myolie_teacher_execute =
                    flags[index * 2 + seat] != 0;
            }
        }
        Ok(true)
    })
}

/// JSON `[[seat0, seat1], ...]` of the actions teacher controllers computed on the
/// last committed `re_batch_myolie_step_frames` (null for seats without a teacher).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_teacher_last(batch: *const ReBatch) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        let batch = unsafe { batch_ref(batch)? };
        let rows: Vec<[&Option<Value>; 2]> = batch
            .native_agents
            .iter()
            .map(|slots| [&slots[0].myolie_teacher_last, &slots[1].myolie_teacher_last])
            .collect();
        serde_json::to_vec(&rows)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize teacher last actions: {error}"))
    })
}

/// JSON `[[seat0, seat1], ...]` of the actions native seats emitted on the last
/// committed `re_batch_myolie_step_frames` (null for caller seats and unstepped games).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_native_last(batch: *const ReBatch) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        let batch = unsafe { batch_ref(batch)? };
        let rows: Vec<[&Option<Value>; 2]> = batch
            .native_agents
            .iter()
            .map(|slots| [&slots[0].myolie_last, &slots[1].myolie_last])
            .collect();
        serde_json::to_vec(&rows)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize native last actions: {error}"))
    })
}

/// Version of the opt-in economic-shaping counter read (`re_batch_myolie_econ`),
/// separate from the unchanged Myolie array ABI version. Version 4: out[n,2,32].
/// Version 3: out[n,2,10]
/// (deaths, ineffective commands and telemetry); version 2 was out[n,2,3] and
/// version 1 out[n,2,2].
#[unsafe(no_mangle)]
pub extern "C" fn re_myolie_econ_version() -> u32 {
    4
}

/// Version 4 (econ loss counters, owner 2026-09-27) is out[n,2,32]: version 3 fields 0–9;
/// expiry, overflow, care lost, fertilizer wasted, weeds and terminal goods at 10–15;
/// destroyed holdings, clipped/missed crop growth, DIG, redundant fertilizer, wasted CARE,
/// terminal flags/seeds/unused land, wasted wages, idle hands, unfilled orders, malformed
/// unit slots and sale shortfall/floor units/purchase premium at 16–31. See `crate::ECON_*`.
/// Fill caller-owned uint64 out[n,2,ECON_FIELDS] with every game's cumulative counters per
/// seat since that game was constructed, in `crate::ECON_*` order: animal starvation,
/// crop drought (the step metrics' `upstream_*_deaths`), ineffective unit commands
/// (submitted minus committed over every upstream verb; PASS excluded), unit commands
/// of existing units, PASS commands, committed HARVEST, WATER and FEED, committed
/// SELL units and their cash (integer prices). The
/// counters live in each `Game`: create/reset/reset_indices start them at zero,
/// fork/clone_from copy them with the game state, retire keeps them, and every
/// step path (including native and teacher seats) advances them. Read-only; the
/// length is an element count; errors leave `out` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_econ(
    batch: *const ReBatch,
    out: *mut u64,
    out_len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        let batch = unsafe { batch_ref(batch)? };
        let expected = batch
            .games
            .len()
            .checked_mul(2 * ECON_FIELDS)
            .ok_or("Myolie econ count overflow")?;
        array_region(out as *const u64, out_len, expected, "econ")?;
        let counters = batch
            .games
            .iter()
            .enumerate()
            .map(|(index, game)| {
                game.econ_counters()
                    .ok_or_else(|| format!("game {index}: Myolie econ counters require two seats"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if expected == 0 {
            return Ok(true);
        }
        // SAFETY: aligned, non-null and exactly `expected` elements (checked above);
        // the caller owns a writable allocation for the duration of this call.
        let out = unsafe { slice::from_raw_parts_mut(out, out_len) };
        for (seats, row) in counters.iter().zip(out.chunks_exact_mut(2 * ECON_FIELDS)) {
            row[..ECON_FIELDS].copy_from_slice(&seats[0]);
            row[ECON_FIELDS..].copy_from_slice(&seats[1]);
        }
        Ok(true)
    })
}

/// Version of the economic attribution ledger read (`re_batch_myolie_attrib`, crate::econ_attrib).
#[unsafe(no_mangle)]
pub extern "C" fn re_myolie_attrib_version() -> u32 {
    ATTRIB_VERSION
}

/// Fill caller-owned uint64 out[n,2,ATTRIB_FIELDS] with every game's cumulative attribution ledger per seat
/// (crate::econ_attrib `A_*` layout: per-item sales/purchases and cash, hires/land, produced units, effective and
/// ineffective commands per verb and actor slot, idle slots, market and shed-transfer partial fills, day-end
/// service state). Telemetry only, same lifetime as the econ counters; read-only; errors leave `out` unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_attrib(
    batch: *const ReBatch,
    out: *mut u64,
    out_len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        let batch = unsafe { batch_ref(batch)? };
        let expected = batch
            .games
            .len()
            .checked_mul(2 * ATTRIB_FIELDS)
            .ok_or("Myolie attribution count overflow")?;
        array_region(out as *const u64, out_len, expected, "attrib")?;
        let counters = batch
            .games
            .iter()
            .enumerate()
            .map(|(index, game)| {
                game.attrib_counters().ok_or_else(|| {
                    format!("game {index}: Myolie attribution ledger requires two seats")
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if expected == 0 {
            return Ok(true);
        }
        // SAFETY: aligned, non-null and exactly `expected` elements (checked above).
        let out = unsafe { slice::from_raw_parts_mut(out, out_len) };
        for (seats, row) in counters.iter().zip(out.chunks_exact_mut(2 * ATTRIB_FIELDS)) {
            row[..ATTRIB_FIELDS].copy_from_slice(&seats[0]);
            row[ATTRIB_FIELDS..].copy_from_slice(&seats[1]);
        }
        Ok(true)
    })
}

/// Reset only selected slots, preserving order and untouched games/controllers.
/// JSON is limited to reset headers, outside the per-turn array path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_reset_indices(
    batch: *mut ReBatch,
    indices: *const u32,
    indices_len: usize,
    data: *const u8,
    len: usize,
) -> bool {
    // SAFETY: forwards the documented checked array contract.
    unsafe { re_batch_myolie_reset_indices_versioned(batch, indices, indices_len, data, len, 1) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_reset_indices_v2(
    batch: *mut ReBatch,
    indices: *const u32,
    indices_len: usize,
    data: *const u8,
    len: usize,
) -> bool {
    // SAFETY: forwards the documented checked array contract.
    unsafe { re_batch_myolie_reset_indices_versioned(batch, indices, indices_len, data, len, 2) }
}

unsafe fn re_batch_myolie_reset_indices_versioned(
    batch: *mut ReBatch,
    indices: *const u32,
    indices_len: usize,
    data: *const u8,
    len: usize,
    observation_version: u32,
) -> bool {
    ffi_result(false, || {
        // SAFETY: ownership/readability follow the same batch and input contracts.
        let batch = unsafe { batch_mut(batch)? };
        array_region(indices, indices_len, indices_len, "reset indices")?;
        let indices = if indices_len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(indices, indices_len) }
        };
        let bytes = unsafe { input_bytes(data, len)? };
        let headers: Vec<TraceHeader> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse reset headers: {error}"))?;
        if headers.len() != indices.len() {
            return Err("reset header/index counts differ".to_string());
        }
        let mut seen = std::collections::HashSet::with_capacity(indices.len());
        for &index in indices {
            if index as usize >= batch.games.len() || !seen.insert(index) {
                return Err(format!("duplicate or out-of-range reset index {index}"));
            }
        }
        let invest_errors = investment_header_errors(bytes, false)?;
        validate_investment_headers(&invest_errors, observation_version)?;
        let replacements = headers
            .iter()
            .map(Game::from_seed_header)
            .collect::<Result<Vec<_>, String>>()?;
        // An array reset must leave every replacement usable by observe; reject
        // unsupported feature/actor/configuration shapes before changing slots.
        batch.myolie_output.encode_version(
            &replacements,
            observation_version,
            batch.myolie_pool.as_ref(),
        )?;
        let agents = native_agent_slots(indices.len());
        for (((&index, game), agents), error) in indices
            .iter()
            .zip(replacements)
            .zip(agents)
            .zip(invest_errors)
        {
            batch.games[index as usize] = game;
            batch.myolie_invest_errors[index as usize] = error;
            batch.native_agents[index as usize] = agents;
        }
        batch.policy_rows.clear();
        Ok(true)
    })
}

/// Configure explicit native environment parallelism for this batch only.
/// Calls must be serialized with every other operation on the same batch.
/// A failed configuration leaves the existing pool and all game state unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_set_threads(batch: *mut ReBatch, threads: usize) -> bool {
    ffi_result(false, || {
        // SAFETY: live exclusively owned batch under the usual serialization contract.
        let batch = unsafe { batch_mut(batch)? };
        if threads == 0 || threads > batch.games.len().max(1) {
            return Err(
                "native thread count must be between 1 and the batch game count".to_string(),
            );
        }
        let current = batch
            .myolie_pool
            .as_ref()
            .map_or(1, rayon::ThreadPool::current_num_threads);
        if threads == current {
            return Ok(true);
        }
        let pool = if threads == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("myolie-native-{index}"))
                    .build()
                    .map_err(|error| format!("create native batch thread pool: {error}"))?,
            )
        };
        batch.myolie_pool = pool;
        Ok(true)
    })
}

/// Return configured native threads, or zero on an invalid batch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_threads(batch: *const ReBatch) -> usize {
    ffi_result(0, || {
        // SAFETY: live batch serialized with mutation/free by its owner.
        let batch = unsafe { batch_ref(batch)? };
        Ok(batch
            .myolie_pool
            .as_ref()
            .map_or(1, rayon::ThreadPool::current_num_threads))
    })
}

/// Create a native-reset batch from a JSON array of `TraceHeader` objects.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_create(data: *const u8, len: usize) -> *mut ReBatch {
    ffi_result(ptr::null_mut(), || {
        // SAFETY: forwarded caller contract; the slice is consumed before return.
        let bytes = unsafe { input_bytes(data, len)? };
        let headers: Vec<TraceHeader> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse batch headers: {error}"))?;
        let games = games_from_headers(&headers)?;
        Ok(Box::into_raw(Box::new(ReBatch {
            myolie_invest_errors: investment_header_errors(bytes, false)?,
            native_agents: native_agent_slots(games.len()),
            myolie_output: MyolieOutput::default(),
            myolie_pool: None,
            games,
            policy_rows: Vec::new(),
        })))
    })
}

/// Create a batch from the exact public/private state carried by each header.
/// This is deliberately separate from native reset so ordinary RQ callers cannot
/// accidentally trust an exported state oracle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_explicit_batch_create(data: *const u8, len: usize) -> *mut ReBatch {
    ffi_result(ptr::null_mut(), || {
        // SAFETY: forwarded caller contract; the slice is consumed before return.
        let bytes = unsafe { input_bytes(data, len)? };
        let headers: Vec<TraceHeader> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse explicit batch headers: {error}"))?;
        let games = games_from_explicit_headers(&headers)?;
        Ok(Box::into_raw(Box::new(ReBatch {
            myolie_invest_errors: investment_header_errors(bytes, true)?,
            native_agents: native_agent_slots(games.len()),
            myolie_output: MyolieOutput::default(),
            myolie_pool: None,
            games,
            policy_rows: Vec::new(),
        })))
    })
}

/// Replace every game transactionally with a fresh native reset.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_reset(batch: *mut ReBatch, data: *const u8, len: usize) -> bool {
    ffi_result(false, || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let headers: Vec<TraceHeader> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse batch headers: {error}"))?;
        let games = games_from_headers(&headers)?;
        let invest_errors = investment_header_errors(bytes, false)?;
        batch.native_agents = native_agent_slots(games.len());
        batch.myolie_invest_errors = invest_errors;
        batch.games = games;
        batch.policy_rows.clear();
        Ok(true)
    })
}

/// Replace every game transactionally with the exact state in each header.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_explicit_batch_reset(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let headers: Vec<TraceHeader> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse explicit batch headers: {error}"))?;
        let games = games_from_explicit_headers(&headers)?;
        let invest_errors = investment_header_errors(bytes, true)?;
        batch.native_agents = native_agent_slots(games.len());
        batch.myolie_invest_errors = invest_errors;
        batch.games = games;
        batch.policy_rows.clear();
        Ok(true)
    })
}

/// Replace the batch with clones selected by index, enabling common-prefix forks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_fork(batch: *mut ReBatch, data: *const u8, len: usize) -> bool {
    ffi_result(false, || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let indices: Vec<usize> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse fork indices: {error}"))?;
        if indices.is_empty() {
            return Err("fork must retain at least one game".to_string());
        }
        let mut games = Vec::with_capacity(indices.len());
        let mut native_agents = Vec::with_capacity(indices.len());
        let mut invest_errors = Vec::with_capacity(indices.len());
        for index in indices {
            let game = batch.games.get(index).ok_or_else(|| {
                format!("fork index {index} outside batch of {}", batch.games.len())
            })?;
            games.push(game.clone());
            invest_errors.push(batch.myolie_invest_errors[index].clone());
            native_agents.push(batch.native_agents[index].clone());
        }
        batch.games = games;
        batch.myolie_invest_errors = invest_errors;
        batch.native_agents = native_agents;
        batch.policy_rows.clear();
        Ok(true)
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CloneRequest {
    src: usize,
    dst: usize,
    #[serde(default)]
    seed: Option<serde_json::Number>,
    #[serde(default)]
    clear_teacher: bool,
}

/// Copy games from `src` into selected slots of `dst` (Expert Iteration lanes, owner 2026-09-26):
/// the engine state, its episode seed and both seats' native-controller slots (anchor memory
/// and assignments) are cloned. JSON `[{"src", "dst", "seed"?, "clear_teacher"?}]`. `seed`
/// replaces the clone's integer episode seed, which only drives future day-end draws (weeds,
/// shop unlocks); `clear_teacher` removes teacher assignments so a lane computes no labels.
/// Every request is validated before any slot changes; `dst` must not alias `src`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_clone_from(
    dst: *mut ReBatch,
    src: *const ReBatch,
    data: *const u8,
    len: usize,
) -> bool {
    ffi_result(false, || {
        if ptr::eq(dst as *const ReBatch, src) {
            return Err("clone_from requires two distinct batches".to_string());
        }
        // SAFETY: both pointers follow the constructor/free ownership contract and do not alias.
        let source = unsafe { batch_ref(src)? };
        let target = unsafe { batch_mut(dst)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<CloneRequest> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse clone requests: {error}"))?;
        let mut targets = std::collections::HashSet::new();
        let mut staged = Vec::with_capacity(requests.len());
        for request in &requests {
            let game = source.games.get(request.src).ok_or_else(|| {
                format!(
                    "clone source {} outside batch of {}",
                    request.src,
                    source.games.len()
                )
            })?;
            if request.dst >= target.games.len() {
                return Err(format!(
                    "clone target {} outside batch of {}",
                    request.dst,
                    target.games.len()
                ));
            }
            if !targets.insert(request.dst) {
                return Err(format!("clone target {} requested twice", request.dst));
            }
            let mut game = game.clone();
            if let Some(seed) = &request.seed {
                let seed = crate::resolved_seed(seed)?;
                if !matches!(seed, crate::ResolvedSeed::Integer(_)) {
                    return Err("clone reseed requires an integer seed".to_string());
                }
                game.seed = seed;
            }
            let mut slots = source.native_agents[request.src].clone();
            if request.clear_teacher {
                for slot in slots.iter_mut() {
                    slot.myolie_teacher = None;
                    slot.myolie_teacher_execution_aware = false;
                    slot.myolie_teacher_execute = false;
                    slot.myolie_teacher_last = None;
                }
            }
            staged.push((
                request.dst,
                game,
                slots,
                source.myolie_invest_errors[request.src].clone(),
            ));
        }
        for (index, game, slots, error) in staged {
            target.games[index] = game;
            target.myolie_invest_errors[index] = error;
            target.native_agents[index] = slots;
        }
        target.policy_rows.clear();
        Ok(true)
    })
}

/// Retire selected games (idle Expert Iteration lanes): each becomes done, so
/// `re_batch_myolie_step_frames` treats it as a zero-length no-op, and loses its native
/// and teacher assignments. JSON `[index, ...]`; validated before any change.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_myolie_retire(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> bool {
    ffi_result(false, || {
        // SAFETY: ownership/readability follow the batch and input contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let indices: Vec<usize> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse retire indices: {error}"))?;
        for &index in &indices {
            if index >= batch.games.len() {
                return Err(format!(
                    "retire index {index} outside batch of {}",
                    batch.games.len()
                ));
            }
        }
        for index in indices {
            batch.games[index].done = true;
            for slot in batch.native_agents[index].iter_mut() {
                slot.myolie_native = None;
                slot.myolie_last = None;
                slot.myolie_teacher = None;
                slot.myolie_teacher_execution_aware = false;
                slot.myolie_teacher_execute = false;
                slot.myolie_teacher_last = None;
            }
        }
        batch.policy_rows.clear();
        Ok(true)
    })
}

/// Apply one joint action to every game transactionally and return all snapshots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_step(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let actions: Vec<Vec<Value>> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse batch actions: {error}"))?;
        if actions.len() != batch.games.len() {
            return Err(format!(
                "got {} joint actions for batch of {} games",
                actions.len(),
                batch.games.len()
            ));
        }
        let mut candidates = batch.games.clone();
        for (game, joint_action) in candidates.iter_mut().zip(actions) {
            game.step(&joint_action)?;
        }
        let encoded = encode_snapshots(&candidates)?;
        batch.games = candidates;
        batch.policy_rows.clear();
        Ok(encoded)
    })
}

/// Compile one RQ60/RQ61 crop-modified matching graph. The request and response
/// are JSON so this bounded kernel can be parity-integrated before a packed graph
/// ABI is justified.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_joint_rows(data: *const u8, len: usize) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: forwarded caller contract; the slice is consumed before return.
        let bytes = unsafe { input_bytes(data, len)? };
        let request: JointRequest = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse joint matching request: {error}"))?;
        let rows = compile_joint_rows(&request)?;
        serde_json::to_vec(&rows)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize joint matching rows: {error}"))
    })
}

/// Compile several independent crop-modified graphs behind one Python/Rust
/// boundary. Candidate semantics are identical to `re_joint_rows` per graph.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_joint_rows_batch(data: *const u8, len: usize) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: forwarded caller contract; the slice is consumed before return.
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<JointRequest> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse joint matching batch: {error}"))?;
        let rows = requests
            .iter()
            .map(compile_joint_rows)
            .collect::<Result<Vec<_>, _>>()?;
        serde_json::to_vec(&rows)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize joint matching batch: {error}"))
    })
}

/// Benchmark only repeated native compilation after one JSON parse. The result
/// includes a checksum so the optimizer cannot discard the work.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_joint_bench(data: *const u8, len: usize) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: forwarded caller contract; the slice is consumed before return.
        let bytes = unsafe { input_bytes(data, len)? };
        let request: JointBenchRequest = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse joint benchmark request: {error}"))?;
        let result = benchmark_joint_rows(&request)?;
        serde_json::to_vec(&result)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize joint benchmark result: {error}"))
    })
}

/// Return current snapshots without advancing the games.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_snapshots(batch: *const ReBatch) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        let batch = unsafe { batch_ref(batch)? };
        encode_snapshots(&batch.games)
    })
}

/// Return current cash and conservative mark-to-market wealth per game and seat
/// without advancing the games (the JA26 potential's starting point).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_seat_wealth(batch: *const ReBatch) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        let batch = unsafe { batch_ref(batch)? };
        serde_json::to_vec(&BatchSeatWealth {
            money: batch.games.iter().map(|game| game.seat_money()).collect(),
            wealth: batch.games.iter().map(|game| game.seat_wealth()).collect(),
        })
        .map(ReBuffer::from_vec)
        .map_err(|error| format!("serialize seat wealth: {error}"))
    })
}

#[derive(Deserialize)]
struct CycleRequest {
    game: usize,
    seat: usize,
    x: i64,
    y: i64,
    kind: String,
    /// JAT-05: the crop or animal of a `crop` / `animal` request
    #[serde(default)]
    item: Option<String>,
    /// JAT-05: the acting unit's index and position, for the animal rule's shed trips
    #[serde(default)]
    unit: Option<usize>,
    #[serde(default)]
    at: Option<(i64, i64)>,
    /// JAT-06: the products the seat's sale rule must keep (a `sweep` request)
    #[serde(default)]
    hold: Vec<String>,
}

/// JA28 cycle executor: for each `{game, seat, x, y, kind}` return
/// `{"verb": [...], "orders": [[...], ...]}` from the starter's rule at that tile.
/// JAT-05 requests add `item` (crop or animal) and, for the animal rule, `unit` and
/// `at`.  Read-only; nothing advances and no controller state exists.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_cycle_actions(
    batch: *const ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_ref(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<CycleRequest> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse cycle requests: {error}"))?;
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            if request.seat >= 2 {
                return Err(format!(
                    "cycle seat {} outside two-player game",
                    request.seat
                ));
            }
            let game = batch.games.get(request.game).ok_or_else(|| {
                format!(
                    "cycle game {} outside batch of {}",
                    request.game,
                    batch.games.len()
                )
            })?;
            let value = match request.kind.as_str() {
                "carrot" => crate::native_agents::starter::carrot_cycle_step(
                    game,
                    request.seat,
                    request.x,
                    request.y,
                )?,
                // JA29: staffing request of a tile job with no hand (x, y unused)
                "hire" => crate::native_agents::starter::hire_step(game, request.seat)?,
                // JAT-05 option library: crop and animal lines keyed by item, and the
                // seat's shed-sale rule (x, y unused)
                "crop" => crate::native_agents::library::crop_cycle_step(
                    game,
                    request.seat,
                    request.x,
                    request.y,
                    request
                        .item
                        .as_deref()
                        .ok_or("crop cycle request needs an item")?,
                )?,
                "animal" => crate::native_agents::library::animal_cycle_step(
                    game,
                    request.seat,
                    request.x,
                    request.y,
                    request
                        .item
                        .as_deref()
                        .ok_or("animal cycle request needs an item")?,
                    request
                        .unit
                        .ok_or("animal cycle request needs a unit index")?,
                    request
                        .at
                        .ok_or("animal cycle request needs the unit position")?,
                )?,
                "sweep" => {
                    crate::native_agents::library::sweep_step(game, request.seat, &request.hold)?
                },
                // JAT-06: the crop line that also fertilizes (shed trips like the animal rule)
                "fertilized" => crate::native_agents::library::fertilized_crop_cycle_step(
                    game,
                    request.seat,
                    request.x,
                    request.y,
                    request
                        .item
                        .as_deref()
                        .ok_or("fertilized crop cycle request needs an item")?,
                    request
                        .unit
                        .ok_or("fertilized crop cycle request needs a unit index")?,
                    request
                        .at
                        .ok_or("fertilized crop cycle request needs the unit position")?,
                )?,
                other => return Err(format!("unknown cycle kind {other:?}")),
            };
            results.push(value);
        }
        serde_json::to_vec(&results)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize cycle actions: {error}"))
    })
}

/// Return actions from complete native scripted controllers without advancing games.
/// Controller state is retained per game/seat and reset/forked transactionally with the
/// environment batch. The request order is preserved.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_native_actions(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<NativeActionRequest> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse native action requests: {error}"))?;
        for request in &requests {
            if request.seat >= 2 {
                return Err(format!(
                    "native agent seat {} outside two-player game",
                    request.seat
                ));
            }
            if request.game >= batch.games.len() {
                return Err(format!(
                    "native agent game {} outside batch of {}",
                    request.game,
                    batch.games.len()
                ));
            }
            let slot = &batch.native_agents[request.game][request.seat];
            if slot.myolie_teacher_execution_aware && slot.myolie_teacher == Some(request.kind) {
                return Err("native_actions would advance execution-aware teacher memory without an executed action; use myolie_step_frames".into());
            }
        }
        let mut actions = Vec::with_capacity(requests.len());
        let mut candidates: HashMap<(usize, usize, NativeAgentKind), NativeAgent> =
            HashMap::with_capacity(requests.len());
        for request in requests {
            let key = (request.game, request.seat, request.kind);
            let agent = candidates.entry(key).or_insert_with(|| {
                let slot = &batch.native_agents[request.game][request.seat];
                slot.get(request.kind).clone()
            });
            let game = &batch.games[request.game];
            actions.push(agent.action(game, request.seat)?);
        }
        let encoded = serde_json::to_vec(&actions)
            .map_err(|error| format!("serialize native agent actions: {error}"))?;
        for ((game, seat, kind), candidate) in candidates {
            let slot = &mut batch.native_agents[game][seat];
            slot.set(kind, candidate);
        }
        Ok(ReBuffer::from_vec(encoded))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_native_debug(
    batch: *const ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        let batch = unsafe { batch_ref(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let requests: Vec<NativeActionRequest> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse native debug requests: {error}"))?;
        let mut rows = Vec::with_capacity(requests.len());
        for request in requests {
            if request.seat >= 2 || request.game >= batch.native_agents.len() {
                return Err("native debug request outside batch".into());
            }
            let slot = &batch.native_agents[request.game][request.seat];
            let agent = slot.get(request.kind);
            rows.push(agent.debug());
        }
        serde_json::to_vec(&rows)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize native debug: {error}"))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_len(batch: *const ReBatch) -> usize {
    ffi_result(0, || {
        // SAFETY: the pointer follows the constructor/free ownership contract.
        Ok(unsafe { batch_ref(batch)? }.games.len())
    })
}

/// Return and retain the current thread's last diagnostic as an owned buffer.
/// Like `re_batch_step` but returns only `[done, ...]` as JSON, skipping snapshot
/// serialization for rollouts whose seats are all policy-driven.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_step_done(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let actions: Vec<Vec<Value>> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse batch actions: {error}"))?;
        if actions.len() != batch.games.len() {
            return Err(format!(
                "got {} joint actions for batch of {} games",
                actions.len(),
                batch.games.len()
            ));
        }
        let mut candidates = batch.games.clone();
        for (game, joint_action) in candidates.iter_mut().zip(actions) {
            game.step(&joint_action)?;
        }
        let done: Vec<bool> = candidates.iter().map(|game| game.snapshot().done).collect();
        batch.games = candidates;
        batch.policy_rows.clear();
        serde_json::to_vec(&done)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize done flags: {error}"))
    })
}

/// Like `re_batch_step_done`, with aggregate market intent/effect counters for
/// observability. The transition path and transactional commit remain identical.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_step_done_metrics(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let request: BatchStepMetricsRequest = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse batch actions: {error}"))?;
        let (actions, live_seats) = match request {
            BatchStepMetricsRequest::Actions(actions) => (actions, None),
            BatchStepMetricsRequest::WithLiveSeats {
                actions,
                live_seats,
            } => (actions, Some(live_seats)),
        };
        if actions.len() != batch.games.len() {
            return Err(format!(
                "got {} joint actions for batch of {} games",
                actions.len(),
                batch.games.len()
            ));
        }
        if let Some(seats) = live_seats.as_ref() {
            if seats.len() != batch.games.len() {
                return Err(format!(
                    "got {} live seats for batch of {} games",
                    seats.len(),
                    batch.games.len()
                ));
            }
            if let Some(seat) = seats.iter().find(|seat| **seat >= 2) {
                return Err(format!("live seat {seat} outside two-player game"));
            }
        }
        let mut candidates = batch.games.clone();
        let mut market = MarketStepMetrics::default();
        let mut market_live = live_seats.as_ref().map(|_| MarketStepMetrics::default());
        let mut market_opponent = live_seats.as_ref().map(|_| MarketStepMetrics::default());
        let mut unit = UnitStepMetrics::default();
        let mut unit_live = live_seats.as_ref().map(|_| UnitStepMetrics::default());
        let mut unit_opponent = live_seats.as_ref().map(|_| UnitStepMetrics::default());
        let mut money = Vec::with_capacity(candidates.len());
        let mut wealth = Vec::with_capacity(candidates.len());
        let mut production = Vec::with_capacity(candidates.len());
        for (game_index, (game, joint_action)) in candidates.iter_mut().zip(actions).enumerate() {
            let step_metrics = game.step_with_market_metrics(&joint_action)?;
            money.push(game.seat_money());
            wealth.push(game.seat_wealth());
            production.push(
                step_metrics
                    .unit
                    .by_seat
                    .iter()
                    .map(|seat| seat.committed_production_actions)
                    .collect(),
            );
            if let Some(seats) = live_seats.as_ref() {
                let live_seat = seats[game_index];
                market_live
                    .as_mut()
                    .unwrap()
                    .accumulate_seat(&step_metrics.market.by_seat[live_seat]);
                market_opponent
                    .as_mut()
                    .unwrap()
                    .accumulate_seat(&step_metrics.market.by_seat[1 - live_seat]);
                unit_live
                    .as_mut()
                    .unwrap()
                    .accumulate_seat(&step_metrics.unit.by_seat[live_seat]);
                unit_opponent
                    .as_mut()
                    .unwrap()
                    .accumulate_seat(&step_metrics.unit.by_seat[1 - live_seat]);
            }
            market.accumulate(&step_metrics.market);
            unit.accumulate(&step_metrics.unit);
        }
        let done = candidates.iter().map(|game| game.snapshot().done).collect();
        batch.games = candidates;
        batch.policy_rows.clear();
        serde_json::to_vec(&BatchStepDoneMetrics {
            done,
            market,
            market_live,
            market_opponent,
            unit,
            unit_live,
            unit_opponent,
            money,
            wealth,
            production,
        })
        .map(ReBuffer::from_vec)
        .map_err(|error| format!("serialize done flags and market metrics: {error}"))
    })
}

/// ET-F policy rows for `[[game, seat], ...]` (JSON input): one packed buffer of
/// features, edges and market masks, computed from native state.  The jobs and edges
/// are cached on the batch for `re_batch_policy_actions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_policy_rows(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let pairs: Vec<(usize, usize)> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse policy row pairs: {error}"))?;
        let mut packed = Vec::with_capacity(pairs.len());
        let mut cached = Vec::with_capacity(pairs.len());
        for (game_index, seat) in pairs {
            let game = batch.games.get(game_index).ok_or_else(|| {
                format!(
                    "policy row game {game_index} outside batch of {}",
                    batch.games.len()
                )
            })?;
            let (row, cache) = policy_row(game, seat)?;
            packed.push(row);
            cached.push(cache);
        }
        batch.policy_rows = cached;
        Ok(ReBuffer::from_vec(pack_rows(&packed)))
    })
}

/// Compact-v1 ET-F rows for `[[game, seat], ...]`.  This endpoint deliberately sits
/// beside (rather than changing) `re_batch_policy_rows`, so deployment and callers
/// that do not opt in keep the original dense wire.  Cached jobs/edges use the same
/// action ABI consumed by `re_batch_policy_actions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_policy_rows_compact_v1(
    batch: *mut ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_mut(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let pairs: Vec<(usize, usize)> = serde_json::from_slice(bytes)
            .map_err(|error| format!("parse compact-v1 policy row pairs: {error}"))?;
        let mut packed = Vec::with_capacity(pairs.len());
        let mut cached = Vec::with_capacity(pairs.len());
        for (game_index, seat) in pairs {
            let game = batch.games.get(game_index).ok_or_else(|| {
                format!(
                    "compact-v1 policy row game {game_index} outside batch of {}",
                    batch.games.len()
                )
            })?;
            let (row, cache) = compact_policy_row_v1(game, seat)?;
            packed.push(row);
            cached.push(cache);
        }
        batch.policy_rows = cached;
        Ok(ReBuffer::from_vec(pack_compact_rows_v1(&packed)))
    })
}

/// Compile sampled choices into engine actions for rows of the last
/// `re_batch_policy_rows` call.  Input is packed little-endian: u32 count, then per row
/// u32 row index, u32 units, i32[units] edge choices (-1 idle), i8[21] market choices.
/// Output is a JSON array of actions in input order.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_policy_actions(
    batch: *const ReBatch,
    data: *const u8,
    len: usize,
) -> ReBuffer {
    ffi_result(ReBuffer::empty(), || {
        // SAFETY: both pointers follow their documented ownership contracts.
        let batch = unsafe { batch_ref(batch)? };
        let bytes = unsafe { input_bytes(data, len)? };
        let mut cursor = 0usize;
        let read_u32 = |cursor: &mut usize| -> Result<u32, String> {
            let end = *cursor + 4;
            let slice = bytes.get(*cursor..end).ok_or("truncated policy choices")?;
            *cursor = end;
            Ok(u32::from_le_bytes(slice.try_into().unwrap()))
        };
        let count = read_u32(&mut cursor)? as usize;
        // every row carries at least 8 header bytes + MARKET_FACTORS bytes
        if count > bytes.len() / (8 + MARKET_FACTORS) {
            return Err(format!(
                "policy choices declare {count} rows for {} bytes",
                bytes.len()
            ));
        }
        let mut actions = Vec::with_capacity(count);
        for _ in 0..count {
            let row_index = read_u32(&mut cursor)? as usize;
            let units = read_u32(&mut cursor)? as usize;
            let cached = batch.policy_rows.get(row_index).ok_or_else(|| {
                format!(
                    "policy row {row_index} outside the {} cached rows",
                    batch.policy_rows.len()
                )
            })?;
            if units != cached.world.units.len() {
                return Err(format!(
                    "row {row_index} sent {units} unit choices for {} units",
                    cached.world.units.len()
                ));
            }
            let mut assignment = Vec::with_capacity(units);
            for _ in 0..units {
                assignment.push(read_u32(&mut cursor)? as i32);
            }
            let end = cursor + MARKET_FACTORS;
            let market: Vec<i8> = bytes
                .get(cursor..end)
                .ok_or("truncated market choices")?
                .iter()
                .map(|b| *b as i8)
                .collect();
            cursor = end;
            actions.push(policy_action(cached, &assignment, &market)?);
        }
        if cursor != bytes.len() {
            return Err(format!(
                "policy choices packet has {} trailing bytes",
                bytes.len() - cursor
            ));
        }
        serde_json::to_vec(&actions)
            .map(ReBuffer::from_vec)
            .map_err(|error| format!("serialize policy actions: {error}"))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn re_batch_last_error() -> ReBuffer {
    LAST_ERROR.with(|slot| ReBuffer::from_vec(slot.borrow().as_bytes().to_vec()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_batch_free(batch: *mut ReBatch) {
    if !batch.is_null() {
        // SAFETY: the pointer came from one of the batch constructors via
        // `Box::into_raw`, and the wrapper calls this exactly once.
        drop(unsafe { Box::from_raw(batch) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn re_buffer_free(buffer: ReBuffer) {
    if !buffer.data.is_null() {
        // SAFETY: the buffer came from `ReBuffer::from_vec` and is freed exactly once.
        drop(unsafe { Vec::from_raw_parts(buffer.data, buffer.len, buffer.capacity) });
    }
}

#[cfg(test)]
mod myolie_sampler_ffi_tests {
    use super::*;
    use serde_json::json;

    fn take(buffer: ReBuffer) -> Vec<u8> {
        assert!(!buffer.data.is_null());
        // SAFETY: these buffers come from the tested exports and are freed once.
        let bytes = unsafe { slice::from_raw_parts(buffer.data, buffer.len) }.to_vec();
        unsafe { re_buffer_free(buffer) };
        bytes
    }

    #[test]
    fn sampler_ffi_rejects_bad_pointers_shapes_and_lengths_before_slice() {
        let frames = [0_i16; 24];
        let misaligned = (frames.as_ptr() as *const u8).wrapping_add(1) as *const i16;
        for (actors, length, pointer) in [
            (1, 24, ptr::null()),
            (1, 24, misaligned),
            (1, 0, ptr::null()),
            (1, 23, frames.as_ptr()),
            (1, usize::MAX, frames.as_ptr()),
            (242, 24, frames.as_ptr()),
        ] {
            // SAFETY: deliberately invalid pointers must fail admission before
            // dereferencing; valid addresses refer to the local fixed array.
            let buffer = unsafe { re_myolie_sampler_decode(actors, 10, 16, pointer, length) };
            assert!(buffer.data.is_null());
            assert_eq!(buffer.len, 0);
            assert!(!take(re_batch_last_error()).is_empty());
        }
        let invalid_plan = re_myolie_sampler_plan(1, 11, 16);
        assert!(invalid_plan.data.is_null());
        assert!(!take(re_batch_last_error()).is_empty());
    }

    #[test]
    fn sampler_ffi_success_owns_binary_or_json_without_mutating_input() {
        let bytes = take(re_myolie_sampler_plan(1, 10, 16));
        assert_eq!(&bytes[..4], &crate::myolie_sampler::MAGIC.to_le_bytes());
        let mut frames = [0_i16; 24];
        frames[1] = 1;
        frames[23] = 1;
        let before = frames;
        // SAFETY: the complete aligned token array remains alive during decode.
        let bytes =
            take(unsafe { re_myolie_sampler_decode(1, 10, 16, frames.as_ptr(), frames.len()) });
        assert_eq!(frames, before);
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"farmer": ["PASS"], "hands": [], "market": []})
        );
        assert!(take(re_batch_last_error()).is_empty());
    }
}

#[cfg(test)]
mod myolie_array_tests {
    use super::*;
    use serde_json::json;

    fn program(rows: &[[i16; MYOLIE_SLOTS]]) -> Vec<i16> {
        let mut values = vec![0; MYOLIE_MAX_FRAMES * MYOLIE_SLOTS];
        for (index, row) in rows.iter().enumerate() {
            values[index * MYOLIE_SLOTS..(index + 1) * MYOLIE_SLOTS].copy_from_slice(row);
        }
        values
    }

    #[test]
    fn myolie_nonfinite_banks_fail_admission() {
        let mut game = Game::new(crate::Config::default(), 73, 2).unwrap();
        let mut staged = MyolieOutput::default();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            game.farms[0].money = value;
            assert!(
                staged
                    .encode(&[game.clone()])
                    .unwrap_err()
                    .contains("nonfinite Myolie bank")
            );
        }
    }

    #[test]
    fn myolie_unit_decode_retains_exact_commands_and_quantity_omission() {
        let game = Game::new(crate::Config::default(), 73, 2).unwrap();
        let mut stop = [0; MYOLIE_SLOTS];
        stop[11] = 1;
        let mut count = 0;
        for kind in 1..=18 {
            let items: Vec<usize> = match kind {
                6 | 7 => (1..=12).collect(),
                8 => (1..=5).collect(),
                _ => vec![0],
            };
            let quantities = if kind == 6 || kind == 7 {
                vec![None, Some(1), Some(31), Some(32), Some(1023)]
            } else {
                vec![None]
            };
            for &item in &items {
                for &quantity in &quantities {
                    let mut frame = [0; MYOLIE_SLOTS];
                    frame[1] = kind as i16;
                    frame[3] = item as i16;
                    let mut expected = vec![json!(MYOLIE_UNIT_NAMES[kind])];
                    if item > 0 {
                        expected.push(json!(MYOLIE_ITEMS[item]));
                    }
                    if let Some(quantity) = quantity {
                        frame[4] = 1;
                        frame[5] = quantity / 32;
                        frame[6] = quantity % 32;
                        expected.push(json!(quantity));
                    }
                    assert_eq!(
                        myolie_action(&game, 0, &program(&[frame, stop]), 2).unwrap(),
                        json!({"farmer": expected, "hands": [], "market": []})
                    );
                    count += 1;
                }
            }
        }
        assert_eq!(count, 140);
    }

    #[test]
    fn myolie_market_decode_retains_empty_zero_and_ordered_slots() {
        let game = Game::new(crate::Config::default(), 73, 2).unwrap();
        let mut farmer = [0; MYOLIE_SLOTS];
        farmer[1] = 1;
        let mut empty = [0; MYOLIE_SLOTS];
        empty[7] = 7;
        let mut stop = [0; MYOLIE_SLOTS];
        stop[11] = 1;
        let mut count = 0;
        for kind in 1..=7 {
            let items: Vec<usize> = match kind {
                3 => (1..=5).collect(),
                4 => vec![1, 9],
                5 => (10..=12).collect(),
                6 => (1..=9).collect(),
                _ => vec![0],
            };
            let quantities = if (3..=6).contains(&kind) {
                vec![0, 1, 31, 32, 1023]
            } else {
                vec![0]
            };
            for &item in &items {
                for &quantity in &quantities {
                    let mut frame = [0; MYOLIE_SLOTS];
                    frame[7] = kind as i16;
                    frame[8] = item as i16;
                    frame[9] = quantity / 32;
                    frame[10] = quantity % 32;
                    let mut expected = if kind == 7 {
                        vec![]
                    } else {
                        vec![json!(MYOLIE_MARKET_NAMES[kind])]
                    };
                    if item > 0 {
                        expected.extend([json!(MYOLIE_ITEMS[item]), json!(quantity)]);
                    }
                    assert_eq!(
                        myolie_action(&game, 1, &program(&[farmer, empty, frame, empty, stop]), 5)
                            .unwrap(),
                        json!({"farmer": ["PASS"], "hands": [], "market": [[], expected, []]})
                    );
                    count += 1;
                }
            }
        }
        assert_eq!(count, 98);
    }
}

#[cfg(test)]
mod exit_lane_tests {
    use super::*;
    use serde_json::json;

    fn batch(seeds: &[i64]) -> ReBatch {
        let games: Vec<Game> = seeds
            .iter()
            .map(|&seed| Game::new(crate::Config::default(), seed, 2).unwrap())
            .collect();
        ReBatch {
            myolie_invest_errors: vec![None; seeds.len()],
            native_agents: native_agent_slots(games.len()),
            myolie_output: MyolieOutput::default(),
            myolie_pool: None,
            games,
            policy_rows: Vec::new(),
        }
    }

    fn pass() -> Value {
        json!({"farmer": ["PASS"], "hands": [], "market": []})
    }

    fn state(game: &Game) -> Value {
        serde_json::to_value(game.snapshot()).unwrap()
    }

    fn clone_from(dst: &mut ReBatch, src: &ReBatch, requests: Value) -> bool {
        let bytes = serde_json::to_vec(&requests).unwrap();
        // SAFETY: distinct live batches and a byte buffer alive for the call.
        unsafe { re_batch_clone_from(dst, src, bytes.as_ptr(), bytes.len()) }
    }

    #[test]
    fn clone_copies_state_seed_and_controllers_and_replays_identically() {
        let mut src = batch(&[61, 62]);
        for _ in 0..30 {
            for game in src.games.iter_mut() {
                game.step(&[pass(), pass()]).unwrap();
            }
        }
        src.native_agents[1][1].myolie_native = Some(NativeAgentKind::Cha22);
        src.native_agents[0][0].myolie_teacher = Some(NativeAgentKind::Cha22);
        src.native_agents[0][0].myolie_teacher_execution_aware = true;
        let mut dst = batch(&[7, 8, 9]);
        let untouched = state(&dst.games[1]);
        assert!(clone_from(
            &mut dst,
            &src,
            json!([{"src": 1, "dst": 2}, {"src": 0, "dst": 0, "clear_teacher": true}])
        ));
        assert_eq!(state(&dst.games[2]), state(&src.games[1]));
        assert_eq!(state(&dst.games[0]), state(&src.games[0]));
        assert_eq!(state(&dst.games[1]), untouched);
        assert_eq!(
            format!("{:?}", dst.native_agents[2]),
            format!("{:?}", src.native_agents[1])
        );
        assert!(
            dst.native_agents[0]
                .iter()
                .all(|slot| slot.myolie_teacher.is_none() && !slot.myolie_teacher_execution_aware)
        );
        assert!(src.native_agents[0][0].myolie_teacher.is_some());
        // The episode seed travels with the clone: identical actions give identical futures.
        for _ in 0..300 {
            src.games[1].step(&[pass(), pass()]).unwrap();
            dst.games[2].step(&[pass(), pass()]).unwrap();
        }
        assert_eq!(state(&dst.games[2]), state(&src.games[1]));
    }

    #[test]
    fn reseed_keeps_the_present_and_changes_future_day_end_draws() {
        let mut src = batch(&[61]);
        for _ in 0..30 {
            src.games[0].step(&[pass(), pass()]).unwrap();
        }
        let mut dst = batch(&[1, 2, 3, 4]);
        assert!(clone_from(
            &mut dst,
            &src,
            json!([{"src": 0, "dst": 0}, {"src": 0, "dst": 1, "seed": 1001},
                   {"src": 0, "dst": 2, "seed": 1002}, {"src": 0, "dst": 3, "seed": 1003}])
        ));
        for index in 0..4 {
            assert_eq!(state(&dst.games[index]), state(&src.games[0]));
        }
        for _ in 0..240 {
            for game in dst.games.iter_mut() {
                game.step(&[pass(), pass()]).unwrap();
            }
        }
        let original = state(&dst.games[0]);
        assert!((1..4).any(|index| state(&dst.games[index]) != original));
    }

    #[test]
    fn clone_rejects_bad_requests_without_changing_the_target() {
        let src = batch(&[61, 62]);
        let mut dst = batch(&[7, 8]);
        let before: Vec<Value> = dst.games.iter().map(state).collect();
        for requests in [
            json!([{"src": 2, "dst": 0}]),
            json!([{"src": 0, "dst": 2}]),
            json!([{"src": 0, "dst": 1}, {"src": 1, "dst": 1}]),
            json!([{"src": 0, "dst": 0}, {"src": 1, "dst": 1, "seed": 1.5}]),
            json!([{"src": 0, "dst": 0, "unknown": 1}]),
        ] {
            assert!(!clone_from(&mut dst, &src, requests));
            assert_eq!(dst.games.iter().map(state).collect::<Vec<_>>(), before);
        }
        let bytes = serde_json::to_vec(&json!([{"src": 0, "dst": 1}])).unwrap();
        let alias: *mut ReBatch = &mut dst;
        // SAFETY: the alias is rejected before either reference is formed.
        assert!(!unsafe { re_batch_clone_from(alias, alias, bytes.as_ptr(), bytes.len()) });
        assert_eq!(dst.games.iter().map(state).collect::<Vec<_>>(), before);
    }

    fn econ_all(batch: &ReBatch) -> Vec<u64> {
        let mut out = vec![u64::MAX; batch.games.len() * 2 * ECON_FIELDS];
        // SAFETY: live batch; `out` has the documented [n,2,10] length and outlives the call.
        let ok = unsafe { re_batch_myolie_econ(batch, out.as_mut_ptr(), out.len()) };
        let error = LAST_ERROR.with(|cell| cell.borrow().clone());
        assert!(ok, "{error}");
        out
    }

    /// The shaping counters [starvation, drought, ineffective] of every game and seat.
    fn econ(batch: &ReBatch) -> Vec<u64> {
        econ_all(batch)
            .chunks_exact(ECON_FIELDS)
            .flat_map(|seat| seat[..3].to_vec())
            .collect()
    }

    /// PASS for every seat of every live game through the array step (caller frames).
    fn step_frames_pass(batch: &mut ReBatch) {
        let n = batch.games.len();
        let mut frames = vec![0_i16; n * 2 * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS];
        let mut lengths = vec![0_i32; n * 2];
        for index in 0..n {
            for seat in 0..2 {
                if batch.games[index].done
                    || batch.native_agents[index][seat].myolie_native.is_some()
                {
                    continue;
                }
                let start = (index * 2 + seat) * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS;
                frames[start + 1] = 1;
                frames[start + MYOLIE_SLOTS + 11] = 1;
                lengths[index * 2 + seat] = 2;
            }
        }
        let mut features = vec![0_f32; n * 2 * FEATURE_COUNT];
        let mut context = vec![0_i64; n * 4];
        let mut banks = vec![0_f64; n * 2];
        let mut done = vec![0_u8; n];
        // SAFETY: every array is allocated with the documented shape and outlives the call.
        let ok = unsafe {
            re_batch_myolie_step_frames(
                batch,
                frames.as_ptr(),
                frames.len(),
                lengths.as_ptr(),
                lengths.len(),
                features.as_mut_ptr(),
                features.len(),
                context.as_mut_ptr(),
                context.len(),
                banks.as_mut_ptr(),
                banks.len(),
                done.as_mut_ptr(),
                done.len(),
            )
        };
        let error = LAST_ERROR.with(|cell| cell.borrow().clone());
        assert!(ok, "{error}");
    }

    #[test]
    fn econ_counters_accumulate_in_step_frames_and_follow_the_game() {
        let mut lanes = batch(&[61, 62, 63]);
        assert_eq!(econ(&lanes), vec![0; 18]);
        // Game 0 closes day 0 on its next step: seat 0 loses an unwatered wheat (drought),
        // seat 1 a goose unfed for a second day (starvation). Game 1 (a native seat) and
        // game 2 lose nothing.
        lanes.games[0].step = 23;
        let turns_per_day = lanes.games[0].config.turns_per_day.clone();
        lanes.games[0].farms[0].tiles[0][0] = crate::new_plant("WHEAT", 0, &turns_per_day);
        let mut starved = crate::new_animal("GOOSE", 0);
        starved["consecutive_unfed"] = json!(1);
        lanes.games[0].farms[1].tiles[0][0] = starved;
        lanes.native_agents[1][1].myolie_native = Some(NativeAgentKind::Cha22);
        step_frames_pass(&mut lanes);
        // PASS is never an ineffective command
        assert_eq!(
            econ(&lanes),
            vec![0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        step_frames_pass(&mut lanes); // no further deaths: counters are cumulative, not step-local
        assert_eq!(&econ(&lanes)[..6], &[0, 1, 0, 1, 0, 0]);
        // telemetry: two PASS steps of game 2's caller seats (one farmer each), no commits or sales
        let all = econ_all(&lanes);
        let seat = &all[2 * 2 * ECON_FIELDS..2 * 2 * ECON_FIELDS + ECON_FIELDS];
        let mut expected = [0; ECON_FIELDS];
        expected[3] = 2;
        expected[4] = 2;
        assert_eq!(seat, &expected);
        // fork and clone copy the counters with the state; retire keeps them.
        let mut dst = batch(&[7, 8]);
        assert!(clone_from(&mut dst, &lanes, json!([{"src": 0, "dst": 1}])));
        assert_eq!(econ(&dst), vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]);
        let bytes = serde_json::to_vec(&json!([0])).unwrap();
        // SAFETY: live batch and byte buffer alive for the call.
        assert!(unsafe { re_batch_myolie_retire(&mut lanes, bytes.as_ptr(), bytes.len()) });
        assert_eq!(&econ(&lanes)[..6], &[0, 1, 0, 1, 0, 0]);
        let bytes = serde_json::to_vec(&json!([0, 2, 0])).unwrap();
        // SAFETY: live batch and byte buffer alive for the call.
        assert!(unsafe { re_batch_fork(&mut lanes, bytes.as_ptr(), bytes.len()) });
        assert_eq!(
            econ(&lanes),
            vec![0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
        );
    }

    #[test]
    fn econ_read_rejects_bad_buffers_without_writing() {
        let lanes = batch(&[61, 62]);
        let mut out = vec![7_u64; 41];
        let misaligned = (out.as_mut_ptr() as *mut u8).wrapping_add(1) as *mut u64;
        for (pointer, len) in [
            (out.as_mut_ptr(), 8),
            (out.as_mut_ptr(), 12),
            (out.as_mut_ptr(), 39),
            (out.as_mut_ptr(), 41),
            (ptr::null_mut(), 40),
            (misaligned, 40),
        ] {
            // SAFETY: invalid shapes/pointers must fail admission before any write.
            assert!(!unsafe { re_batch_myolie_econ(&lanes, pointer, len) });
            assert!(!LAST_ERROR.with(|cell| cell.borrow().clone()).is_empty());
        }
        assert_eq!(out, vec![7; 41]);
        // SAFETY: a null batch is rejected.
        assert!(!unsafe { re_batch_myolie_econ(ptr::null(), out.as_mut_ptr(), 0) });
        assert_eq!(re_myolie_econ_version(), 4);
        assert_eq!(re_myolie_array_abi_version(), 1);
    }

    #[test]
    fn retired_games_are_zero_length_noops_in_step_frames() {
        let mut lanes = batch(&[61, 62]);
        lanes.native_agents[0][1].myolie_native = Some(NativeAgentKind::Cha22);
        let bytes = serde_json::to_vec(&json!([0])).unwrap();
        // SAFETY: live batch and byte buffer alive for the call.
        assert!(unsafe { re_batch_myolie_retire(&mut lanes, bytes.as_ptr(), bytes.len()) });
        assert!(lanes.games[0].done && !lanes.games[1].done);
        assert!(
            lanes.native_agents[0]
                .iter()
                .all(|slot| slot.myolie_native.is_none())
        );
        let bad = serde_json::to_vec(&json!([2])).unwrap();
        // SAFETY: as above; the out-of-range index must fail before any change.
        assert!(!unsafe { re_batch_myolie_retire(&mut lanes, bad.as_ptr(), bad.len()) });
        let retired = state(&lanes.games[0]);
        let live_before = lanes.games[1].step;
        let n = 2;
        let mut frames = vec![0_i16; n * 2 * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS];
        let mut lengths = vec![0_i32; n * 2];
        for seat in 0..2 {
            let start = (2 + seat) * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS;
            frames[start + 1] = 1;
            frames[start + MYOLIE_SLOTS + 11] = 1;
            lengths[2 + seat] = 2;
        }
        let mut features = vec![0_f32; n * 2 * FEATURE_COUNT];
        let mut context = vec![0_i64; n * 4];
        let mut banks = vec![0_f64; n * 2];
        let mut done = vec![0_u8; n];
        // SAFETY: every array is allocated with the documented shape and outlives the call.
        let ok = unsafe {
            re_batch_myolie_step_frames(
                &mut lanes,
                frames.as_ptr(),
                frames.len(),
                lengths.as_ptr(),
                lengths.len(),
                features.as_mut_ptr(),
                features.len(),
                context.as_mut_ptr(),
                context.len(),
                banks.as_mut_ptr(),
                banks.len(),
                done.as_mut_ptr(),
                done.len(),
            )
        };
        let error = LAST_ERROR.with(|cell| cell.borrow().clone());
        assert!(ok, "{error}");
        assert_eq!(state(&lanes.games[0]), retired);
        assert_eq!(lanes.games[1].step, live_before + 1);
        assert_eq!(done, vec![1, 0]);
    }
}

#[cfg(test)]
mod native_parallel_tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn batch(threads: usize) -> ReBatch {
        let games = (61..69)
            .map(|seed| Game::new(crate::Config::default(), seed, 2).unwrap())
            .collect();
        let mut value = ReBatch {
            games,
            myolie_invest_errors: vec![None; 8],
            policy_rows: Vec::new(),
            native_agents: native_agent_slots(8),
            myolie_output: MyolieOutput::default(),
            myolie_pool: None,
        };
        assert!(unsafe { re_batch_set_threads(&mut value, threads) });
        value
    }

    struct Buffers {
        features: Vec<f32>,
        context: Vec<i64>,
        banks: Vec<f64>,
        done: Vec<u8>,
    }
    impl Buffers {
        fn new(version: u32) -> Self {
            Self {
                features: vec![42.; 16 * myolie_feature_width(version).unwrap()],
                context: vec![42; 32],
                banks: vec![42.; 16],
                done: vec![42; 8],
            }
        }
        fn bytes(&self) -> (Vec<u32>, Vec<i64>, Vec<u64>, Vec<u8>) {
            (
                self.features.iter().map(|v| v.to_bits()).collect(),
                self.context.clone(),
                self.banks.iter().map(|v| v.to_bits()).collect(),
                self.done.clone(),
            )
        }
        fn observe(&mut self, batch: &mut ReBatch, version: u32) -> bool {
            unsafe {
                re_batch_myolie_observe_versioned(
                    batch,
                    self.features.as_mut_ptr(),
                    self.features.len(),
                    self.context.as_mut_ptr(),
                    self.context.len(),
                    self.banks.as_mut_ptr(),
                    self.banks.len(),
                    self.done.as_mut_ptr(),
                    self.done.len(),
                    version,
                )
            }
        }
        fn step(&mut self, batch: &mut ReBatch, version: u32, bad_last: bool) -> bool {
            let mut frames = vec![0_i16; 8 * 2 * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS];
            let mut lengths = [0_i32; 16];
            for game in 0..8 {
                for seat in 0..2 {
                    if batch.games[game].done
                        || batch.native_agents[game][seat].myolie_native.is_some()
                    {
                        continue;
                    }
                    let start = (game * 2 + seat) * MYOLIE_MAX_FRAMES * MYOLIE_SLOTS;
                    frames[start + 1] = 1; // farmer PASS
                    frames[start + MYOLIE_SLOTS + 11] = 1; // STOP
                    lengths[game * 2 + seat] = 2;
                }
            }
            if bad_last {
                lengths[15] = -1;
            }
            unsafe {
                re_batch_myolie_step_frames_versioned(
                    batch,
                    frames.as_ptr(),
                    frames.len(),
                    lengths.as_ptr(),
                    lengths.len(),
                    self.features.as_mut_ptr(),
                    self.features.len(),
                    self.context.as_mut_ptr(),
                    self.context.len(),
                    self.banks.as_mut_ptr(),
                    self.banks.len(),
                    self.done.as_mut_ptr(),
                    self.done.len(),
                    version,
                )
            }
        }
    }

    fn states(batch: &ReBatch) -> Vec<Value> {
        batch
            .games
            .iter()
            .map(|g| serde_json::to_value(g.snapshot()).unwrap())
            .collect()
    }

    #[test]
    fn native_parallel_both_schemas_match_serial_through_complete_games() {
        for version in [1, 2] {
            let mut serial = batch(1);
            let mut two = batch(2);
            let mut eight = batch(8);
            let mut a = Buffers::new(version);
            let mut b = Buffers::new(version);
            let mut c = Buffers::new(version);
            assert!(a.observe(&mut serial, version));
            assert!(b.observe(&mut two, version));
            assert!(c.observe(&mut eight, version));
            assert_eq!(a.bytes(), b.bytes());
            assert_eq!(a.bytes(), c.bytes());
            for turn in 0..720 {
                assert!(a.step(&mut serial, version, false));
                assert!(b.step(&mut two, version, false));
                assert!(c.step(&mut eight, version, false));
                assert_eq!(a.bytes(), b.bytes(), "version {version}, turn {turn}");
                assert_eq!(a.bytes(), c.bytes(), "version {version}, turn {turn}");
                if turn % 24 == 0 || turn == 719 {
                    assert_eq!(states(&serial), states(&two));
                    assert_eq!(states(&serial), states(&eight));
                    assert_eq!(format!("{:?}", serial.games), format!("{:?}", eight.games));
                }
            }
        }
    }

    #[test]
    fn native_parallel_mixed_failure_never_publishes_games_controllers_or_arrays() {
        for threads in [1, 2, 8] {
            let mut value = batch(threads);
            value.native_agents[0][1].myolie_native = Some(NativeAgentKind::Cha22);
            let mut output = Buffers::new(2);
            assert!(output.observe(&mut value, 2));
            let before = states(&value);
            let controllers = format!("{:?}", value.native_agents);
            let bytes = output.bytes();
            assert!(!output.step(&mut value, 2, true));
            assert_eq!(states(&value), before);
            assert_eq!(format!("{:?}", value.native_agents), controllers);
            assert_eq!(output.bytes(), bytes);
            // Failure after earlier candidates have successfully stepped: observe cannot publish a NaN bank.
            value.games[7].farms[1].money = f64::NAN;
            let before = states(&value);
            assert!(!output.step(&mut value, 2, false));
            assert_eq!(states(&value), before);
            assert!(value.games[7].farms[1].money.is_nan());
            assert_eq!(format!("{:?}", value.native_agents), controllers);
            assert_eq!(output.bytes(), bytes);
            value.games[7].farms[1].money = 100.;
            assert!(output.step(&mut value, 2, false));
            assert_eq!(value.games[0].step, 1);
        }
    }

    #[test]
    fn native_parallel_error_order_is_original_game_order() {
        for threads in [1, 2, 8] {
            let mut value = batch(threads);
            value.games[1].farms[1].money = f64::NAN;
            value.games[7].farms[0].money = f64::NAN;
            let mut output = Buffers::new(1);
            let before = output.bytes();
            for _ in 0..4 {
                assert!(!output.observe(&mut value, 1));
                assert_eq!(
                    LAST_ERROR.with(|x| x.borrow().clone()),
                    "game 1, seat 0: nonfinite Myolie feature 4"
                );
                assert_eq!(output.bytes(), before);
            }
        }
    }

    #[test]
    fn native_parallel_pool_is_private_explicit_bounded_and_transactional() {
        let mut value = batch(8);
        let before = states(&value);
        let pool = value.myolie_pool.as_ref().unwrap();
        let observed = pool.broadcast(|ctx| {
            (
                ctx.index(),
                rayon::current_num_threads(),
                std::thread::current().name().unwrap().to_string(),
            )
        });
        assert_eq!(observed.len(), 8);
        for (index, threads, name) in observed {
            assert_eq!(threads, 8);
            assert_eq!(name, format!("myolie-native-{index}"));
        }
        assert_eq!(unsafe { re_batch_threads(&value) }, 8);
        for bad in [0, 9, usize::MAX] {
            assert!(!unsafe { re_batch_set_threads(&mut value, bad) });
            assert_eq!(unsafe { re_batch_threads(&value) }, 8);
            assert_eq!(states(&value), before);
        }
        assert!(unsafe { re_batch_set_threads(&mut value, 1) });
        assert!(value.myolie_pool.is_none());
        assert_eq!(unsafe { re_batch_threads(&value) }, 1);
        assert_eq!(unsafe { re_batch_threads(ptr::null()) }, 0);
        assert!(!unsafe { re_batch_set_threads(ptr::null_mut(), 2) });
    }

    #[test]
    fn native_parallel_pool_panic_joins_before_ffi_error_and_recovers() {
        let mut value = batch(8);
        let completed = Arc::new(AtomicUsize::new(0));
        let pool = value.myolie_pool.as_ref().unwrap();
        assert!(!ffi_result(false, || {
            pool.scope(|scope| {
                for _ in 0..3 {
                    let done = completed.clone();
                    scope.spawn(move |_| {
                        done.fetch_add(1, Ordering::SeqCst);
                    });
                }
                scope.spawn(|_| panic!("injected private pool failure"));
            });
            Ok(true)
        }));
        assert_eq!(completed.load(Ordering::SeqCst), 3);
        assert!(LAST_ERROR.with(|x| x.borrow().contains("injected private pool failure")));
        let mut output = Buffers::new(2);
        assert!(output.step(&mut value, 2, false));
        assert!(value.games.iter().all(|game| game.step == 1));
    }
}
