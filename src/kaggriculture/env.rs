//! Transactional seed stream and native environment lifecycle.
use super::ObserveError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    Value(String),
    Overflow(String),
    Panic(String),
}
impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(s) | Self::Overflow(s) | Self::Panic(s) => f.write_str(s),
        }
    }
}
impl From<ObserveError> for EnvError {
    fn from(error: ObserveError) -> Self {
        Self::Value(error.to_string())
    }
}
impl From<String> for EnvError {
    fn from(error: String) -> Self {
        Self::Value(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedStream {
    pub next: i64,
    pub stride: i64,
}
impl SeedStream {
    pub fn new(seed: i64, stride: i64) -> Result<Self, EnvError> {
        if seed < 0 || stride < 1 {
            return Err(EnvError::Value(
                "seed must be nonnegative and seed_stride positive".into(),
            ));
        }
        Ok(Self { next: seed, stride })
    }
    pub fn reserve(&self, selected: &[bool]) -> Result<(Vec<Option<i64>>, Self), EnvError> {
        let mut next = self.next;
        let mut seeds = Vec::with_capacity(selected.len());
        for (env, selected) in selected.iter().enumerate() {
            seeds.push(if *selected {
                let seed = next;
                next = next.checked_add(self.stride).ok_or_else(|| {
                    EnvError::Overflow(format!("env={env} seed counter overflow"))
                })?;
                Some(seed)
            } else {
                None
            });
        }
        Ok((seeds, Self { next, ..*self }))
    }
}

use super::buffers::checked_lengths;
use super::reward::RewardConfig;
use super::{
    grammar, write_env, ObsRowMut, ObsStaging, ObservationConfig, ObservationGame,
    PreparedObservation, ValidatedObsBuffersMut,
};
use kaggriculture_engine::Config;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(super) fn caught<T>(work: impl FnOnce() -> Result<T, EnvError>) -> Result<T, EnvError> {
    catch_unwind(AssertUnwindSafe(work)).map_err(|payload| {
        let detail = if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s
        } else {
            "non-string worker panic"
        };
        EnvError::Panic(format!("native lifecycle panic: {detail}"))
    })?
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerminalRecord {
    pub banks: [f64; 2],
    pub margin: f64,
    pub episode_steps: i64,
    pub winner: i64,
    pub econ: [[i64; 32]; 2],
}
struct EnvSlot {
    game: ObservationGame,
    prepared: PreparedObservation,
    seed: i64,
    actors: [i64; 2],
    orders: i64,
    steps: i64,
    hires: [usize; 2],
    terminal: Option<TerminalRecord>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionCache {
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub transition_banks_before: Vec<f64>,
    pub transition_banks_after: Vec<f64>,
    pub transition_econ_before: Vec<i64>,
    pub transition_econ_after: Vec<i64>,
}
pub struct TransitionBuffersMut<'a> {
    pub rewards: &'a mut [f32],
    pub dones: &'a mut [bool],
    pub transition_banks_before: &'a mut [f64],
    pub transition_banks_after: &'a mut [f64],
    pub transition_econ_before: &'a mut [i64],
    pub transition_econ_after: &'a mut [i64],
}
impl TransitionCache {
    fn zeros(n: usize) -> Self {
        Self {
            rewards: vec![0.; n * 2],
            dones: vec![false; n * 2],
            transition_banks_before: vec![0.; n * 2],
            transition_banks_after: vec![0.; n * 2],
            transition_econ_before: vec![0; n * 64],
            transition_econ_after: vec![0; n * 64],
        }
    }
    fn publish(&self, out: TransitionBuffersMut<'_>) {
        out.rewards.copy_from_slice(&self.rewards);
        out.dones.copy_from_slice(&self.dones);
        out.transition_banks_before
            .copy_from_slice(&self.transition_banks_before);
        out.transition_banks_after
            .copy_from_slice(&self.transition_banks_after);
        out.transition_econ_before
            .copy_from_slice(&self.transition_econ_before);
        out.transition_econ_after
            .copy_from_slice(&self.transition_econ_after);
    }
}
impl TransitionBuffersMut<'_> {
    fn validate(&self, n: usize) -> Result<(), EnvError> {
        for (name, actual, expected) in [
            ("rewards", self.rewards.len(), n * 2),
            ("dones", self.dones.len(), n * 2),
            (
                "transition_banks_before",
                self.transition_banks_before.len(),
                n * 2,
            ),
            (
                "transition_banks_after",
                self.transition_banks_after.len(),
                n * 2,
            ),
            (
                "transition_econ_before",
                self.transition_econ_before.len(),
                n * 64,
            ),
            (
                "transition_econ_after",
                self.transition_econ_after.len(),
                n * 64,
            ),
        ] {
            if actual != expected {
                return Err(EnvError::Value(format!(
                    "{name}: expected {expected} values, got {actual}"
                )));
            }
        }
        Ok(())
    }
}

pub struct PendingBatch {
    staging: ObsStaging,
    candidates: Vec<Option<EnvSlot>>,
    stream: SeedStream,
    selected: Option<Vec<bool>>,
    transition: Option<TransitionCache>,
    metrics: Vec<(f64, f64, f64)>,
}
impl PendingBatch {
    pub fn metrics(&self) -> &[(f64, f64, f64)] {
        &self.metrics
    }
}

pub struct NativeEnv {
    config: Config,
    episode_steps: i64,
    hire_multiplier: i64,
    reward: RewardConfig,
    hire_limit: i64,
    pool: ThreadPool,
    slots: Vec<EnvSlot>,
    stream: SeedStream,
    transition: TransitionCache,
    #[cfg(test)]
    fault: Option<Fault>,
}
impl NativeEnv {
    pub fn new(
        n: usize,
        seed: i64,
        stride: i64,
        config: Config,
        reward: RewardConfig,
        threads: usize,
        hire_limit: i64,
    ) -> Result<Self, EnvError> {
        caught(|| {
            checked_lengths(n)?;
            if threads == 0 {
                return Err(EnvError::Value("native_threads must be positive".into()));
            }
            grammar::plan(1, 1, hire_limit)?;
            reward.validate()?;
            let admitted = ObservationConfig::new(&config)?;
            let episode_steps = admitted.episode_steps;
            let hire_multiplier = admitted.farm_hand_cost_mult;
            let (seeds, stream) = SeedStream::new(seed, stride)?.reserve(&vec![true; n])?;
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| EnvError::Value(format!("native_threads: {e}")))?;
            let mut staging = ObsStaging::new(n)?;
            let results: Vec<Result<EnvSlot, EnvError>> = pool.install(|| {
                staging
                    .buffers_mut()
                    .par_envs_mut()
                    .zip(seeds.par_iter())
                    .enumerate()
                    .map(|(i, (mut rows, seed))| {
                        caught(|| {
                            let seed = seed.expect("all construction rows selected");
                            let game =
                                ObservationGame::from_seed(config.clone(), &seed.to_string())?;
                            let prepared = game.prepare()?;
                            write_env(&prepared, &mut rows);
                            Ok(EnvSlot {
                                game,
                                prepared,
                                seed,
                                actors: [
                                    rows.seats[0].globals_int[14],
                                    rows.seats[1].globals_int[14],
                                ],
                                orders: *rows.seats[0].order_limits,
                                steps: 0,
                                hires: [0, 0],
                                terminal: None,
                            })
                        })
                        .map_err(|e| at_env(i, e))
                    })
                    .collect()
            });
            let slots = results.into_iter().collect::<Result<Vec<_>, _>>()?;
            let mut transition = TransitionCache::zeros(n);
            for (i, rows) in staging.buffers_mut().envs_mut().enumerate() {
                transition.transition_banks_before[i * 2..i * 2 + 2]
                    .copy_from_slice(rows.seats[0].banks);
                transition.transition_banks_after[i * 2..i * 2 + 2]
                    .copy_from_slice(rows.seats[0].banks);
            }
            Ok(Self {
                config,
                episode_steps,
                hire_multiplier,
                reward,
                hire_limit,
                pool,
                slots,
                stream,
                transition,
                #[cfg(test)]
                fault: None,
            })
        })
    }
    pub fn n_envs(&self) -> usize {
        self.slots.len()
    }
    pub fn seed_state(&self) -> (i64, Vec<i64>) {
        (
            self.stream.next,
            self.slots.iter().map(|s| s.seed).collect(),
        )
    }
    pub fn state_snapshot(&self, index: usize) -> Result<String, EnvError> {
        let slot = self
            .slots
            .get(index)
            .ok_or_else(|| EnvError::Value(format!("env_index={index} outside batch")))?;
        serde_json::to_string(&slot.game.game().snapshot())
            .map_err(|e| EnvError::Value(e.to_string()))
    }
    pub fn terminal_metrics(&self, index: usize) -> Result<Option<&TerminalRecord>, EnvError> {
        Ok(self
            .slots
            .get(index)
            .ok_or_else(|| EnvError::Value(format!("env_index={index} outside batch")))?
            .terminal
            .as_ref())
    }
    pub fn prepare_observe(&self) -> Result<PendingBatch, EnvError> {
        caught(|| {
            let mut staging = ObsStaging::new(self.n_envs())?;
            self.pool.install(|| {
                staging
                    .buffers_mut()
                    .par_envs_mut()
                    .zip(self.slots.par_iter())
                    .for_each(|(mut rows, slot)| write_env(&slot.prepared, &mut rows))
            });
            Ok(PendingBatch {
                staging,
                candidates: (0..self.n_envs()).map(|_| None).collect(),
                stream: self.stream,
                selected: None,
                transition: Some(self.transition.clone()),
                metrics: vec![],
            })
        })
    }
    pub fn publish_observe(
        &self,
        mut pending: PendingBatch,
        out: &mut ValidatedObsBuffersMut<'_>,
        transition: TransitionBuffersMut<'_>,
    ) -> Result<(), EnvError> {
        let n = self.n_envs();
        transition.validate(n)?;
        if out.envs_mut().count() != n || pending.staging.buffers_mut().envs_mut().count() != n {
            return Err(EnvError::Value(
                "observation publication n_envs mismatch".into(),
            ));
        }
        let cached = pending
            .transition
            .as_ref()
            .ok_or_else(|| EnvError::Value("missing observation transition cache".into()))?;
        pending.staging.publish(out)?;
        cached.publish(transition);
        Ok(())
    }
    pub fn prepare_reset(
        &mut self,
        mask: &[bool],
        truncate: bool,
    ) -> Result<PendingBatch, EnvError> {
        caught(|| {
            if mask.len() != self.n_envs() || (!truncate && mask.iter().any(|x| !x)) {
                return Err(EnvError::Value(
                    "reset mask: batch mismatch or partial full reset".into(),
                ));
            }
            let (seeds, stream) = self.stream.reserve(mask)?;
            let mut staging = ObsStaging::new(self.n_envs())?;
            let results: Vec<Result<Option<EnvSlot>, EnvError>> = self.pool.install(|| {
                staging
                    .buffers_mut()
                    .par_envs_mut()
                    .zip(seeds.par_iter())
                    .enumerate()
                    .map(|(i, (mut rows, seed))| {
                        caught(|| {
                            let Some(seed) = seed else { return Ok(None) };
                            #[cfg(test)]
                            self.inject(FaultPoint::ResetConstruct, i)?;
                            let game =
                                ObservationGame::from_seed(self.config.clone(), &seed.to_string())?;
                            #[cfg(test)]
                            {
                                self.inject(FaultPoint::ResetPrepare, i)?;
                                self.inject(FaultPoint::ResetPanic, i)?;
                            }
                            let prepared = game.prepare()?;
                            write_env(&prepared, &mut rows);
                            Ok(Some(EnvSlot {
                                game,
                                prepared,
                                seed: *seed,
                                actors: [
                                    rows.seats[0].globals_int[14],
                                    rows.seats[1].globals_int[14],
                                ],
                                orders: *rows.seats[0].order_limits,
                                steps: 0,
                                hires: [0, 0],
                                terminal: None,
                            }))
                        })
                        .map_err(|e| at_env(i, e))
                    })
                    .collect()
            });
            let candidates = results.into_iter().collect::<Result<Vec<_>, _>>()?;
            let transition = if truncate {
                None
            } else {
                let mut t = TransitionCache::zeros(self.n_envs());
                for (i, rows) in staging.buffers_mut().envs_mut().enumerate() {
                    t.transition_banks_before[i * 2..i * 2 + 2]
                        .copy_from_slice(rows.seats[0].banks);
                    t.transition_banks_after[i * 2..i * 2 + 2].copy_from_slice(rows.seats[0].banks);
                }
                Some(t)
            };
            Ok(PendingBatch {
                staging,
                candidates,
                stream,
                selected: truncate.then(|| mask.to_vec()),
                transition,
                metrics: vec![],
            })
        })
    }
    pub fn prepare_step(
        &mut self,
        tokens: &[i64],
        lengths: &[i64],
    ) -> Result<PendingBatch, EnvError> {
        caught(|| {
            let n = self.n_envs();
            if tokens.len() != n * 2 * grammar::TOKENS_PER_SEAT || lengths.len() != n * 2 {
                return Err(EnvError::Value(
                    "tokens/lengths: batch shape mismatch".into(),
                ));
            }
            // Raw signed transport is admitted before seed reservation or any
            // narrowing; complete State decoding follows reservation below.
            for (row, (tokens, length)) in tokens
                .chunks_exact(grammar::TOKENS_PER_SEAT)
                .zip(lengths)
                .enumerate()
            {
                if !(1..=grammar::MAX_FRAMES as i64).contains(length) {
                    return Err(EnvError::Value(format!(
                        "env={} seat={} length outside 1..=252",
                        row / 2,
                        row % 2
                    )));
                }
                for (i, token) in tokens.iter().enumerate() {
                    if *token < 0
                        || *token >= grammar::SLOT_WIDTHS[i % grammar::SLOTS] as i64
                        || (i >= *length as usize * grammar::SLOTS && *token != 0)
                    {
                        return Err(EnvError::Value(format!(
                            "env={} seat={} frame={} slot={} invalid token/padding",
                            row / 2,
                            row % 2,
                            i / grammar::SLOTS,
                            grammar::SLOT_NAMES[i % grammar::SLOTS]
                        )));
                    }
                }
            }
            let terminal: Vec<bool> = self
                .slots
                .iter()
                .map(|s| i128::from(s.steps) >= i128::from(self.episode_steps) - 2)
                .collect();
            let (seeds, stream) = self.stream.reserve(&terminal)?;
            let mut actions = Vec::with_capacity(n);
            for (env, slot) in self.slots.iter().enumerate() {
                let mut pair = Vec::with_capacity(2);
                for seat in 0..2 {
                    let offset = (env * 2 + seat) * grammar::TOKENS_PER_SEAT;
                    let plan = grammar::plan(slot.actors[seat], slot.orders, self.hire_limit)?;
                    pair.push(
                        grammar::decode(
                            &plan,
                            &tokens[offset..offset + grammar::TOKENS_PER_SEAT],
                            lengths[env * 2 + seat],
                        )
                        .map_err(|e| EnvError::Value(format!("env={env} seat={seat} {e}")))?,
                    );
                }
                actions.push(pair);
            }
            let candidates: Vec<_> = self.slots.iter().map(|s| s.game.clone()).collect();
            let mut staging = ObsStaging::new(n)?;
            let results: Vec<Result<(EnvSlot, TransitionRow), EnvError>> =
                self.pool.install(|| {
                    candidates
                        .into_par_iter()
                        .zip(staging.buffers_mut().par_envs_mut())
                        .enumerate()
                        .map(|(i, (mut game, mut rows))| {
                            caught(|| {
                                let slot = &self.slots[i];
                                let before_banks = game_banks(&game)?;
                                let before_econ = game_econ(&game)?;
                                let before_hires = executed_hires(&game)?;
                                game.step_with_market_metrics(&actions[i])?;
                                #[cfg(test)]
                                {
                                    self.inject(FaultPoint::StepResult, i)?;
                                    self.inject(FaultPoint::StepPanic, i)?;
                                }
                                let after_banks = game_banks(&game)?;
                                let after_econ = game_econ(&game)?;
                                let after_hires = executed_hires(&game)?;
                                let mut hires = [0; 2];
                                for seat in 0..2 {
                                    hires[seat] = after_hires[seat]
                                        .checked_sub(before_hires[seat])
                                        .ok_or_else(|| {
                                            EnvError::Value(
                                                "HIRE attribution counter decreased".into(),
                                            )
                                        })?;
                                }
                                super::admission::validate_hire_cash(
                                    self.hire_multiplier,
                                    slot.hires,
                                    hires,
                                )?;
                                let done = game.game().terminal_banks().is_some();
                                if done != terminal[i] {
                                    return Err(EnvError::Value(
                                        "terminal seed prediction disagrees with kernel".into(),
                                    ));
                                }
                                let rewards = self.reward.transition(
                                    &before_econ,
                                    &after_econ,
                                    after_banks,
                                    done,
                                )?;
                                let steps = slot.steps.checked_add(1).ok_or_else(|| {
                                    EnvError::Overflow("episode steps outside int64".into())
                                })?;
                                let record = if done {
                                    let margin = after_banks[0] - after_banks[1];
                                    if !margin.is_finite() {
                                        return Err(EnvError::Value(
                                            "terminal margin must be finite".into(),
                                        ));
                                    }
                                    Some(TerminalRecord {
                                        banks: after_banks,
                                        margin,
                                        episode_steps: steps,
                                        winner: if margin > 0. {
                                            0
                                        } else if margin < 0. {
                                            1
                                        } else {
                                            -1
                                        },
                                        econ: after_econ,
                                    })
                                } else {
                                    None
                                };
                                let seed = if let Some(seed) = seeds[i] {
                                    #[cfg(test)]
                                    self.inject(FaultPoint::AutoReset, i)?;
                                    game = ObservationGame::from_seed(
                                        self.config.clone(),
                                        &seed.to_string(),
                                    )?;
                                    seed
                                } else {
                                    slot.seed
                                };
                                #[cfg(test)]
                                self.inject(FaultPoint::StepPrepare, i)?;
                                let prepared = game.prepare()?;
                                write_env(&prepared, &mut rows);
                                let candidate = EnvSlot {
                                    game,
                                    prepared,
                                    seed,
                                    actors: [
                                        rows.seats[0].globals_int[14],
                                        rows.seats[1].globals_int[14],
                                    ],
                                    orders: *rows.seats[0].order_limits,
                                    steps: rows.seats[0].globals_int[0],
                                    hires: [
                                        rows.seats[0].globals_int[12] as usize,
                                        rows.seats[1].globals_int[12] as usize,
                                    ],
                                    terminal: record,
                                };
                                Ok((
                                    candidate,
                                    TransitionRow {
                                        before_banks,
                                        after_banks,
                                        before_econ,
                                        after_econ,
                                        rewards,
                                        done,
                                    },
                                ))
                            })
                            .map_err(|e| at_env(i, e))
                        })
                        .collect()
                });
            // Indexed collection joins every worker before choosing the first
            // error in env order. Nothing published by successful peers.
            let results = results.into_iter().collect::<Result<Vec<_>, _>>()?;
            let mut transition = TransitionCache::zeros(n);
            let mut candidates = Vec::with_capacity(n);
            let mut metrics = Vec::new();
            for (i, (slot, row)) in results.into_iter().enumerate() {
                transition.rewards[i * 2..i * 2 + 2].copy_from_slice(&row.rewards);
                transition.dones[i * 2..i * 2 + 2].fill(row.done);
                transition.transition_banks_before[i * 2..i * 2 + 2]
                    .copy_from_slice(&row.before_banks);
                transition.transition_banks_after[i * 2..i * 2 + 2]
                    .copy_from_slice(&row.after_banks);
                for seat in 0..2 {
                    let range = (i * 2 + seat) * 32..(i * 2 + seat + 1) * 32;
                    transition.transition_econ_before[range.clone()]
                        .copy_from_slice(&row.before_econ[seat]);
                    transition.transition_econ_after[range].copy_from_slice(&row.after_econ[seat]);
                }
                if let Some(record) = &slot.terminal {
                    metrics.push((record.banks[0], record.banks[1], record.margin));
                }
                candidates.push(Some(slot));
            }
            Ok(PendingBatch {
                staging,
                candidates,
                stream,
                selected: None,
                transition: Some(transition),
                metrics,
            })
        })
    }
    pub fn commit(
        &mut self,
        mut pending: PendingBatch,
        out: &mut ValidatedObsBuffersMut<'_>,
        transition: TransitionBuffersMut<'_>,
    ) -> Result<(), EnvError> {
        // Complete preflight before either full or selected publication. Observe
        // never installs replacement slots; truncate never publishes transitions.
        let n = self.n_envs();
        transition.validate(n)?;
        if out.envs_mut().count() != n
            || pending.staging.buffers_mut().envs_mut().count() != n
            || pending.candidates.len() != n
            || pending.selected.as_ref().is_some_and(|m| m.len() != n)
        {
            return Err(EnvError::Value("publication n_envs mismatch".into()));
        }
        if let Some(selected) = &pending.selected {
            commit_selected_rows(&mut pending.staging.buffers_mut(), out, selected);
        } else {
            pending.staging.publish(out)?;
        }
        // COMMIT: all remaining operations are admitted copies or swaps.
        if let Some(mut next) = pending.transition {
            next.publish(transition);
            std::mem::swap(&mut self.transition, &mut next);
        }
        for (slot, candidate) in self.slots.iter_mut().zip(pending.candidates.iter_mut()) {
            if let Some(candidate) = candidate {
                std::mem::swap(slot, candidate);
            }
        }
        std::mem::swap(&mut self.stream, &mut pending.stream);
        Ok(())
    }
}
fn at_env(i: usize, error: EnvError) -> EnvError {
    match error {
        EnvError::Value(s) => EnvError::Value(format!("env={i} {s}")),
        EnvError::Overflow(s) => EnvError::Overflow(format!("env={i} {s}")),
        EnvError::Panic(s) => EnvError::Panic(format!("env={i} {s}")),
    }
}

fn commit_selected_rows(
    staging: &mut ValidatedObsBuffersMut<'_>,
    out: &mut ValidatedObsBuffersMut<'_>,
    selected: &[bool],
) {
    for ((src, dst), selected) in staging.envs_mut().zip(out.envs_mut()).zip(selected) {
        if *selected {
            for (src, dst) in src.seats.into_iter().zip(dst.seats) {
                copy_row(src, dst);
            }
        }
    }
}
fn copy_row(src: ObsRowMut<'_>, dst: ObsRowMut<'_>) {
    let ObsRowMut {
        tile_kind,
        tile_crop,
        tile_animal,
        tile_cell,
        tile_role,
        tiles_int,
        tiles_float,
        actor_slot,
        actor_cell,
        actor_role,
        actor_mask,
        actor_inventory,
        actor_inventory_rank,
        actors_float,
        player_features,
        storage_counts,
        storage_rank,
        banks,
        shop_type,
        shop_slot,
        shop_mask,
        market_product,
        market_float,
        market_int,
        global_features,
        globals_int,
        still_playing,
        order_limits,
        can_act,
    } = src;
    *dst.tile_kind = *tile_kind;
    *dst.tile_crop = *tile_crop;
    *dst.tile_animal = *tile_animal;
    *dst.tile_cell = *tile_cell;
    *dst.tile_role = *tile_role;
    *dst.tiles_int = *tiles_int;
    *dst.tiles_float = *tiles_float;
    *dst.actor_slot = *actor_slot;
    *dst.actor_cell = *actor_cell;
    *dst.actor_role = *actor_role;
    *dst.actor_mask = *actor_mask;
    *dst.actor_inventory = *actor_inventory;
    *dst.actor_inventory_rank = *actor_inventory_rank;
    *dst.actors_float = *actors_float;
    *dst.player_features = *player_features;
    *dst.storage_counts = *storage_counts;
    *dst.storage_rank = *storage_rank;
    *dst.banks = *banks;
    *dst.shop_type = *shop_type;
    *dst.shop_slot = *shop_slot;
    *dst.shop_mask = *shop_mask;
    *dst.market_product = *market_product;
    *dst.market_float = *market_float;
    *dst.market_int = *market_int;
    *dst.global_features = *global_features;
    *dst.globals_int = *globals_int;
    *dst.still_playing = *still_playing;
    *dst.order_limits = *order_limits;
    *dst.can_act = *can_act;
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FaultPoint {
    StepResult,
    StepPanic,
    AutoReset,
    StepPrepare,
    ResetConstruct,
    ResetPrepare,
    ResetPanic,
}
#[cfg(test)]
struct Fault {
    point: FaultPoint,
    env: usize,
    hits: std::sync::atomic::AtomicUsize,
}
#[cfg(test)]
impl NativeEnv {
    pub(super) fn set_fault(&mut self, point: FaultPoint, env: usize) {
        self.fault = Some(Fault {
            point,
            env,
            hits: std::sync::atomic::AtomicUsize::new(0),
        });
    }
    pub(super) fn fault_hits(&self) -> usize {
        self.fault
            .as_ref()
            .unwrap()
            .hits
            .load(std::sync::atomic::Ordering::SeqCst)
    }
    pub(super) fn clear_fault(&mut self) {
        self.fault = None;
    }
    fn inject(&self, point: FaultPoint, env: usize) -> Result<(), EnvError> {
        if let Some(f) = &self.fault {
            if f.point == point && f.env == env {
                f.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if matches!(point, FaultPoint::StepPanic | FaultPoint::ResetPanic) {
                    panic!("injected {point:?} env={env}");
                }
                return Err(EnvError::Value(format!("injected {point:?} env={env}")));
            }
        }
        Ok(())
    }
    pub(super) fn replace_game(&mut self, i: usize, game: ObservationGame) {
        let prepared = game.prepare().unwrap();
        let mut staging = ObsStaging::new(1).unwrap();
        let mut buffers = staging.buffers_mut();
        let mut rows = buffers.envs_mut().next().unwrap();
        write_env(&prepared, &mut rows);
        self.slots[i] = EnvSlot {
            game,
            prepared,
            seed: self.slots[i].seed,
            actors: [rows.seats[0].globals_int[14], rows.seats[1].globals_int[14]],
            orders: *rows.seats[0].order_limits,
            steps: rows.seats[0].globals_int[0],
            hires: [
                rows.seats[0].globals_int[12] as usize,
                rows.seats[1].globals_int[12] as usize,
            ],
            terminal: None,
        };
    }
}
#[cfg(test)]
impl TransitionCache {
    pub(super) fn buffers_mut(&mut self) -> TransitionBuffersMut<'_> {
        TransitionBuffersMut {
            rewards: &mut self.rewards,
            dones: &mut self.dones,
            transition_banks_before: &mut self.transition_banks_before,
            transition_banks_after: &mut self.transition_banks_after,
            transition_econ_before: &mut self.transition_econ_before,
            transition_econ_after: &mut self.transition_econ_after,
        }
    }
    pub(super) fn test_zeros(n: usize) -> Self {
        Self::zeros(n)
    }
}

struct TransitionRow {
    before_banks: [f64; 2],
    after_banks: [f64; 2],
    before_econ: [[i64; 32]; 2],
    after_econ: [[i64; 32]; 2],
    rewards: [f32; 2],
    done: bool,
}
fn game_banks(game: &ObservationGame) -> Result<[f64; 2], EnvError> {
    let values: [f64; 2] = game
        .game()
        .seat_money()
        .try_into()
        .map_err(|_| EnvError::Value("exactly two banks required".into()))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(EnvError::Value("banks must be finite".into()));
    }
    Ok(values)
}
fn game_econ(game: &ObservationGame) -> Result<[[i64; 32]; 2], EnvError> {
    let source = game
        .game()
        .econ_counters()
        .ok_or_else(|| EnvError::Value("exactly two economic counter rows required".into()))?;
    let mut result = [[0; 32]; 2];
    for seat in 0..2 {
        for (index, value) in source[seat].iter().enumerate() {
            result[seat][index] = i64::try_from(*value).map_err(|_| {
                EnvError::Value(format!(
                    "seat={seat} economic counter {index} outside int64"
                ))
            })?;
        }
    }
    Ok(result)
}
fn executed_hires(game: &ObservationGame) -> Result<[u64; 2], EnvError> {
    // Pinned kernel's private-module A_HIRES=52, econ_attrib.rs:29. This
    // diagnostic admission counter never enters actor/critic observations.
    const A_HIRES: usize = 52;
    let source = game
        .game()
        .attrib_counters()
        .ok_or_else(|| EnvError::Value("two HIRE attribution rows required".into()))?;
    Ok([source[0][A_HIRES], source[1][A_HIRES]])
}
