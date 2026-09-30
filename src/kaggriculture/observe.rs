//! Configuration-bound snapshots and checked public rule context.
//!
//! All observation fields are written from one admitted snapshot; diagnostic
//! row checks are separate from the release write path.
use kaggriculture_engine::{
    Config, Counts, Farm, Game, PrivateState, PublicState, StepMetrics, StepSnapshot, TraceHeader,
};
use serde_json::{Map, Value};
use std::fmt;
use std::sync::Arc;

use super::{ObsEnvMut, ObsRowMut, ObservationConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seat {
    Zero,
    One,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObserveErrorKind {
    Config,
    Shape,
    Dtype,
    NonContiguous,
    Aliased,
    State,
    IntegerRange,
    EnumRange,
    Sentinel,
    NonFinite,
    Engine,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObserveError {
    pub kind: ObserveErrorKind,
    pub env: Option<usize>,
    pub seat: Option<Seat>,
    pub field: String,
    pub detail: String,
}

impl ObserveError {
    pub(super) fn new(
        kind: ObserveErrorKind,
        field: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            env: None,
            seat: None,
            field: field.into(),
            detail: detail.into(),
        }
    }

    pub(super) fn at_seat(mut self, seat: Seat) -> Self {
        self.seat = Some(seat);
        self
    }
}

impl fmt::Display for ObserveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(env) = self.env {
            write!(f, "env={env} ")?;
        }
        if let Some(seat) = self.seat {
            write!(
                f,
                "seat={} ",
                match seat {
                    Seat::Zero => 0,
                    Seat::One => 1,
                }
            )?;
        }
        write!(f, "{}: {}", self.field, self.detail)
    }
}

impl std::error::Error for ObserveError {}

impl Seat {
    pub(super) fn index(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}

pub(super) fn checked_f32(
    value: f64,
    field: &str,
    seat: Option<Seat>,
) -> Result<f32, ObserveError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() {
        return Err(ObserveError {
            seat,
            ..ObserveError::new(
                ObserveErrorKind::NonFinite,
                field,
                "derived value must be finite in float64 and float32",
            )
        });
    }
    Ok(narrowed)
}

fn exact_usize(value: usize, field: &str) -> Result<i64, ObserveError> {
    i64::try_from(value).map_err(|_| {
        ObserveError::new(
            ObserveErrorKind::IntegerRange,
            field,
            "integer outside int64",
        )
    })
}

fn engine_error(field: &str, detail: String) -> ObserveError {
    ObserveError::new(ObserveErrorKind::Engine, field, detail)
}

#[derive(Clone)]
pub struct ObservationGame {
    game: Game,
    config: Arc<ObservationConfig>,
}

#[cfg(test)]
thread_local! {
    static SNAPSHOT_ACQUISITIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn snapshot_acquisitions() -> usize {
    SNAPSHOT_ACQUISITIONS.with(std::cell::Cell::get)
}

impl ObservationGame {
    pub fn from_seed(config: Config, seed_decimal: &str) -> Result<Self, ObserveError> {
        let checked = Arc::new(ObservationConfig::new(&config)?);
        let game = Game::new_with_seed_decimal(config, seed_decimal, 2)
            .map_err(|error| engine_error("seed", error))?;
        Ok(Self {
            game,
            config: checked,
        })
    }

    pub fn from_header(header: &TraceHeader) -> Result<Self, ObserveError> {
        let config = Arc::new(ObservationConfig::new(&header.configuration)?);
        // The kernel recomputes day/hour. Check the imported clock before it can
        // erase evidence of an inconsistent input header.
        validate_public_context(&header.initial.public, &config)?;
        validate_private_count(header.initial.privates.len())?;
        let game = Game::from_header(header).map_err(|error| engine_error("header", error))?;
        Ok(Self { game, config })
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn step_with_market_metrics(
        &mut self,
        actions: &[serde_json::Value],
    ) -> Result<StepMetrics, ObserveError> {
        self.game
            .step_with_market_metrics(actions)
            .map_err(|error| engine_error("actions", error))
    }

    pub fn prepare(&self) -> Result<PreparedObservation, ObserveError> {
        let snapshot = self.acquire_snapshot();
        validate_snapshot(&snapshot, &self.config)?;
        Ok(PreparedObservation {
            snapshot,
            config: Arc::clone(&self.config),
        })
    }

    pub(super) fn acquire_snapshot(&self) -> StepSnapshot {
        #[cfg(test)]
        SNAPSHOT_ACQUISITIONS.with(|count| count.set(count.get() + 1));
        self.game.snapshot()
    }

    #[cfg(test)]
    pub(super) fn validate_existing_snapshot(
        &self,
        snapshot: &StepSnapshot,
    ) -> Result<(), ObserveError> {
        validate_snapshot(snapshot, &self.config)
    }
}

pub struct PreparedObservation {
    snapshot: StepSnapshot,
    config: Arc<ObservationConfig>,
}

impl PreparedObservation {
    /// The validated pre-step snapshot a hosted fixed opponent acts on.
    pub fn snapshot(&self) -> &StepSnapshot {
        &self.snapshot
    }
}

fn validate_private_count(count: usize) -> Result<(), ObserveError> {
    if count != 2 {
        return Err(ObserveError::new(
            ObserveErrorKind::State,
            "privates",
            "must contain exactly two private states",
        ));
    }
    Ok(())
}

fn global_values(public: &PublicState, config: &ObservationConfig) -> [f64; 15] {
    let episode = config.episode_steps as f64;
    let day_length = config.turns_per_day as f64;
    let remaining = (i128::from((config.episode_steps - 1).max(1)) - public.step as i128).max(0);
    [
        public.step as f64 / episode,
        public.hour as f64 / day_length,
        public.day as f64 / (episode / day_length),
        remaining as f64 / episode,
        day_length / 24.0,
        config.orders as f64 / 10.0,
        config.shed_capacity as f64 / 1000.0,
        config.farm_hand_cost_mult as f64 / 100.0,
        episode / 1000.0,
        config.weed_spawn_chance,
        config.town_shop_unlock_interval as f64 / day_length,
        config.town_shop_sell_interval as f64 / day_length,
        config.town_center_sell_interval as f64 / day_length,
        config.starting_money as f64 / 200000.0,
        public.town.unlocked_shops.len() as f64 / 8.0,
    ]
}

fn validate_public_context(
    public: &PublicState,
    config: &ObservationConfig,
) -> Result<(), ObserveError> {
    if public.farms.len() != 2 {
        return Err(ObserveError::new(
            ObserveErrorKind::State,
            "public.farms",
            "must contain exactly two farms",
        ));
    }
    if public
        .market
        .params
        .as_ref()
        .is_some_and(|params| !params.is_empty())
    {
        return Err(ObserveError::new(
            ObserveErrorKind::State,
            "public.market.params",
            "must be absent or empty",
        ));
    }
    let step = exact_usize(public.step, "public.step")?;
    let day = exact_usize(public.day, "public.day")?;
    let hour = exact_usize(public.hour, "public.hour")?;
    if day != step / config.turns_per_day {
        return Err(ObserveError::new(
            ObserveErrorKind::State,
            "public.day",
            "must equal step / turnsPerDay",
        ));
    }
    if hour != step % config.turns_per_day {
        return Err(ObserveError::new(
            ObserveErrorKind::State,
            "public.hour",
            "must equal step % turnsPerDay",
        ));
    }
    if public.town.unlocked_shops.len() > 8 {
        return Err(ObserveError::new(
            ObserveErrorKind::State,
            "public.town.unlocked_shops",
            "at most eight shops are representable",
        ));
    }
    for seat in [Seat::Zero, Seat::One] {
        for (role, farm_index) in [seat.index(), 1 - seat.index()].into_iter().enumerate() {
            let farm = &public.farms[farm_index];
            let hire_field = ["globals_int[12]", "globals_int[13]"][role];
            exact_usize(farm.hires_today, hire_field).map_err(|error| error.at_seat(seat))?;
            if farm.hands.len() > 240 {
                return Err(ObserveError::new(
                    ObserveErrorKind::State,
                    ["globals_int[14]", "globals_int[15]"][role],
                    "at most 241 actors per farm are representable",
                )
                .at_seat(seat));
            }
            checked_f32(
                farm.money / 200000.0,
                ["player_features[0,0]", "player_features[1,0]"][role],
                Some(seat),
            )?;
            checked_f32(
                (farm.hands.len() + 1) as f64 / 241.0,
                ["player_features[0,1]", "player_features[1,1]"][role],
                Some(seat),
            )?;
            checked_f32(
                farm.hires_today as f64 / 240.0,
                ["player_features[0,9]", "player_features[1,9]"][role],
                Some(seat),
            )?;
            config.validate_hire_cost(farm.hires_today, seat, role)?;
            if potential_frames(farm.hands.len() + 1, config.orders).is_none() {
                return Err(ObserveError::new(
                    ObserveErrorKind::State,
                    "action_mask.can_act",
                    "potential frame count outside 1..=252",
                )
                .at_seat(seat));
            }
        }
    }
    for (channel, value) in global_values(public, config).into_iter().enumerate() {
        // The formatting allocation is limited to the exceptional path.
        if !value.is_finite() || !(value as f32).is_finite() {
            return Err(ObserveError::new(
                ObserveErrorKind::NonFinite,
                format!("global_features[{channel}]"),
                "derived value must be finite in float64 and float32",
            ));
        }
    }
    Ok(())
}

fn validate_snapshot(
    snapshot: &StepSnapshot,
    config: &ObservationConfig,
) -> Result<(), ObserveError> {
    validate_public_context(&snapshot.public, config)?;
    validate_private_count(snapshot.privates.len())?;
    validate_tiles(&snapshot.public, config)?;
    validate_actors_and_storage(snapshot, config)?;
    validate_shops_and_market(&snapshot.public)?;
    Ok(())
}

// These sets are pinned to the kernel's new_plant/new_animal constructors and
// WEED/build/animal-death objects. The fixture test independently pins literals.
const PLANT_KEYS: [&str; 8] = [
    "kind",
    "crop",
    "planted_day",
    "watered_today",
    "consecutive_unwatered",
    "yield_units",
    "max_lifespan_step",
    "fertilized_until_day",
];
const ANIMAL_KEYS: [&str; 9] = [
    "kind",
    "animal",
    "placed_day",
    "yield_units",
    "consecutive_unfed",
    "fed_today",
    "cared_today",
    "fertilizer_available",
    "pending_care_bonus",
];

#[derive(Clone, Copy)]
struct TileLocation {
    seat: Seat,
    token: usize,
}

impl TileLocation {
    fn error(self, kind: ObserveErrorKind, field: &str, detail: &str) -> ObserveError {
        ObserveError::new(kind, format!("tiles[{}].{field}", self.token), detail).at_seat(self.seat)
    }

    fn keys(self, tile: &Map<String, Value>, keys: &[&str]) -> Result<(), ObserveError> {
        if tile.len() != keys.len() || keys.iter().any(|key| !tile.contains_key(*key)) {
            return Err(self.error(
                ObserveErrorKind::State,
                "keys",
                &format!(
                    "expected exactly {keys:?}, received {:?}",
                    tile.keys().collect::<Vec<_>>()
                ),
            ));
        }
        Ok(())
    }

    fn string<'a>(
        self,
        tile: &'a Map<String, Value>,
        field: &str,
    ) -> Result<&'a str, ObserveError> {
        tile.get(field).and_then(Value::as_str).ok_or_else(|| {
            self.error(
                ObserveErrorKind::State,
                field,
                "required field must be a string",
            )
        })
    }

    fn boolean(self, tile: &Map<String, Value>, field: &str) -> Result<bool, ObserveError> {
        tile.get(field).and_then(Value::as_bool).ok_or_else(|| {
            self.error(
                ObserveErrorKind::State,
                field,
                "required field must be a bool",
            )
        })
    }

    fn integer(
        self,
        tile: &Map<String, Value>,
        field: &str,
        sentinel: bool,
    ) -> Result<i64, ObserveError> {
        // as_i64 requires the integer JSON representation; 1.0 and 1.5 fail.
        let value = tile.get(field).and_then(Value::as_i64).ok_or_else(|| {
            self.error(
                ObserveErrorKind::IntegerRange,
                field,
                "required field must be a JSON integer fitting int64",
            )
        })?;
        let minimum = if sentinel { -1 } else { 0 };
        if value < minimum {
            return Err(self.error(
                if sentinel {
                    ObserveErrorKind::Sentinel
                } else {
                    ObserveErrorKind::State
                },
                field,
                if sentinel {
                    "date must be -1 or nonnegative"
                } else {
                    "count must be nonnegative"
                },
            ));
        }
        Ok(value)
    }
}

// A single tile's stack scratch, never stored in PreparedObservation or allocated
// as another output tensor. The same parser validates and writes the snapshot.
#[derive(Default)]
struct TileFields {
    kind: i64,
    crop: i64,
    animal: i64,
    ints: [i64; 7],
    floats: [f32; 15],
}

fn parse_tile(
    value: &Value,
    step: i64,
    day: i64,
    episode_steps: i64,
    location: TileLocation,
) -> Result<TileFields, ObserveError> {
    let mut fields = TileFields::default();
    let tile = match value {
        Value::Null => return Ok(fields),
        Value::String(kind) if kind == "LOCKED" => {
            fields.kind = 1;
            return Ok(fields);
        },
        Value::Object(tile) => tile,
        _ => {
            return Err(location.error(
                ObserveErrorKind::State,
                "kind",
                "expected null, LOCKED or an engine tile object",
            ))
        },
    };
    let kind = location.string(tile, "kind")?;
    fields.kind = match kind {
        "WEED" => 2,
        "PLANT" => 3,
        "COOP" => 4,
        "PASTURE" => 5,
        _ => return Err(location.error(ObserveErrorKind::EnumRange, "kind", "unknown tile kind")),
    };
    let mut floats = [0.0_f64; 15];
    if kind == "PLANT" {
        location.keys(tile, &PLANT_KEYS)?;
        fields.crop = match location.string(tile, "crop")? {
            "WHEAT" => 1,
            "CARROT" => 2,
            "TOMATO" => 3,
            "STRAWBERRY" => 4,
            "MELON" => 5,
            _ => return Err(location.error(ObserveErrorKind::EnumRange, "crop", "unknown crop")),
        };
        let yield_units = location.integer(tile, "yield_units", false)?;
        let planted = location.integer(tile, "planted_day", true)?;
        let deadline = location.integer(tile, "max_lifespan_step", true)?;
        let fertilized = location.integer(tile, "fertilized_until_day", true)?;
        let unwatered = location.integer(tile, "consecutive_unwatered", false)?;
        let watered = location.boolean(tile, "watered_today")?;
        fields.ints = [yield_units, planted, deadline, fertilized, 0, unwatered, 0];
        floats[0] = yield_units as f64 / 8.0;
        floats[1] = f64::from(watered);
        floats[2] = unwatered as f64 / 8.0;
        floats[3] = if planted == -1 {
            0.0
        } else {
            (i128::from(day) - i128::from(planted)) as f64 / 30.0
        };
        floats[4] = f64::from(deadline >= 0);
        floats[5] = if deadline == -1 {
            0.0
        } else {
            (i128::from(deadline) - i128::from(step)) as f64 / episode_steps as f64
        };
        floats[6] = f64::from(fertilized >= 0);
        floats[7] = f64::from(fertilized >= day);
        floats[8] = (i128::from(fertilized) - i128::from(day)).max(0) as f64 / 30.0;
    } else if matches!(kind, "COOP" | "PASTURE") && tile.contains_key("animal") {
        location.keys(tile, &ANIMAL_KEYS)?;
        let animal = location.string(tile, "animal")?;
        let (index, structure) = match animal {
            "GOOSE" => (1, "COOP"),
            "COW" => (2, "PASTURE"),
            "SHEEP" => (3, "PASTURE"),
            _ => {
                return Err(location.error(ObserveErrorKind::EnumRange, "animal", "unknown animal"))
            },
        };
        if kind != structure {
            return Err(location.error(
                ObserveErrorKind::State,
                "animal",
                "animal is incompatible with this structure",
            ));
        }
        fields.animal = index;
        let yield_units = location.integer(tile, "yield_units", false)?;
        let placed = location.integer(tile, "placed_day", true)?;
        let unfed = location.integer(tile, "consecutive_unfed", false)?;
        let pending = location.integer(tile, "pending_care_bonus", false)?;
        let fed = location.boolean(tile, "fed_today")?;
        let cared = location.boolean(tile, "cared_today")?;
        let fertilizer = location.boolean(tile, "fertilizer_available")?;
        fields.ints = [yield_units, 0, 0, 0, placed, 0, unfed];
        floats[0] = yield_units as f64 / 8.0;
        floats[9] = if placed == -1 {
            0.0
        } else {
            (i128::from(day) - i128::from(placed)) as f64 / 30.0
        };
        floats[10] = unfed as f64 / 8.0;
        floats[11] = f64::from(fed);
        floats[12] = f64::from(cared);
        floats[13] = f64::from(fertilizer);
        floats[14] = pending as f64 / 8.0;
    } else {
        location.keys(tile, &["kind"])?;
    }
    for (channel, value) in floats.into_iter().enumerate() {
        let narrowed = value as f32;
        if !value.is_finite() || !narrowed.is_finite() {
            return Err(location.error(
                ObserveErrorKind::NonFinite,
                &format!("float[{channel}]"),
                "derived value must be finite in float64 and float32",
            ));
        }
        fields.floats[channel] = narrowed;
    }
    Ok(fields)
}

fn validate_tiles(public: &PublicState, config: &ObservationConfig) -> Result<(), ObserveError> {
    for seat in [Seat::Zero, Seat::One] {
        let tiles = &public.farms[seat.index()].tiles;
        if tiles.len() != 10 || tiles.iter().any(|row| row.len() != 10) {
            return Err(ObserveError::new(
                ObserveErrorKind::State,
                "tiles",
                "farm board must have exactly ten rows of ten cells",
            )
            .at_seat(seat));
        }
        for (y, row) in tiles.iter().enumerate() {
            for (x, tile) in row.iter().enumerate() {
                parse_tile(
                    tile,
                    public.step as i64,
                    public.day as i64,
                    config.episode_steps,
                    TileLocation {
                        seat,
                        token: y * 10 + x,
                    },
                )?;
            }
        }
    }
    Ok(())
}

/// Fully overwrite one seat from its privately constructed, admitted snapshot.
pub fn write_seat(observation: &PreparedObservation, seat: Seat, out: &mut ObsRowMut<'_>) {
    out.clear();
    let public = &observation.snapshot.public;
    let config = &observation.config;
    let own = &public.farms[seat.index()];
    let rival = &public.farms[1 - seat.index()];
    for (role, farm) in [own, rival].into_iter().enumerate() {
        out.banks[role] = farm.money;
        out.player_features[role][0] = (farm.money / 200000.0) as f32;
        out.player_features[role][1] = ((farm.hands.len() + 1) as f64 / 241.0) as f32;
        out.player_features[role][9] = (farm.hires_today as f64 / 240.0) as f32;
        out.player_features[role][10] = config.prepared_hire_cost(farm.hires_today);
        for (y, row) in farm.tiles.iter().enumerate() {
            for (x, tile) in row.iter().enumerate() {
                let cell = y * 10 + x;
                let token = role * 100 + cell;
                // PreparedObservation is privately constructed after both farms
                // pass this identical parser, with immutable snapshot/config.
                // No caller JSON is unwrapped or silently given fallback values.
                let fields = match parse_tile(
                    tile,
                    public.step as i64,
                    public.day as i64,
                    config.episode_steps,
                    TileLocation { seat, token },
                ) {
                    Ok(fields) => fields,
                    Err(error) => unreachable!("prepared tile admission invariant: {error}"),
                };
                out.tile_kind[token] = fields.kind;
                out.tile_crop[token] = fields.crop;
                out.tile_animal[token] = fields.animal;
                out.tile_cell[token] = cell as i64;
                out.tile_role[token] = role as i64;
                out.tiles_int[token] = fields.ints;
                out.tiles_float[token] = fields.floats;
            }
        }
    }
    // Only this seat's private state enters its actor/storage calculations.
    write_actors_and_storage(
        [own, rival],
        &observation.snapshot.privates[seat.index()],
        config,
        out,
    );
    *out.global_features = global_values(public, config).map(|value| value as f32);
    *out.globals_int = [
        public.step as i64,
        public.day as i64,
        public.hour as i64,
        config.episode_steps,
        config.turns_per_day,
        config.orders,
        config.shed_capacity,
        config.farm_hand_cost_mult,
        config.town_shop_unlock_interval,
        config.town_shop_sell_interval,
        config.town_center_sell_interval,
        config.starting_money,
        own.hires_today as i64,
        rival.hires_today as i64,
        (own.hands.len() + 1) as i64,
        (rival.hands.len() + 1) as i64,
    ];
    *out.order_limits = config.orders;
    write_shops_and_market(public, out);
    *out.still_playing = true;
    let frames = potential_frames(own.hands.len() + 1, config.orders)
        .expect("prepared frame-count admission invariant");
    for (frame, present) in out.can_act.iter_mut().enumerate() {
        *present = frame < frames;
    }
}

pub fn write_env(observation: &PreparedObservation, out: &mut ObsEnvMut<'_>) {
    write_seat(observation, Seat::Zero, &mut out.seats[0]);
    write_seat(observation, Seat::One, &mut out.seats[1]);
}

pub fn encode_env(game: &ObservationGame, out: &mut ObsEnvMut<'_>) -> Result<(), ObserveError> {
    let observation = game.prepare()?;
    write_env(&observation, out);
    Ok(())
}

const ITEM_NAMES: [&str; 12] = [
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
const SEED_NAMES: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
const QUADRANTS: [&str; 4] = ["NW", "NE", "SW", "SE"];

fn private_count_field(field: &str, actor: Option<usize>, item: &str) -> String {
    match actor {
        Some(actor) => format!("{field}[{actor},{item}]"),
        None => format!("{field}[{item}]"),
    }
}

fn validate_counts(
    counts: &Counts,
    vocabulary: &[&str],
    field: &str,
    actor: Option<usize>,
    divisor: f64,
    seat: Seat,
) -> Result<(), ObserveError> {
    for (item, count) in counts {
        if !vocabulary.contains(&item.as_str()) {
            return Err(ObserveError::new(
                ObserveErrorKind::EnumRange,
                private_count_field(field, actor, item),
                "unknown item",
            )
            .at_seat(seat));
        }
        if *count < 0 {
            return Err(ObserveError::new(
                ObserveErrorKind::State,
                private_count_field(field, actor, item),
                "count must be nonnegative",
            )
            .at_seat(seat));
        }
        let scaled = *count as f64 / divisor;
        if !scaled.is_finite() || !(scaled as f32).is_finite() {
            return Err(ObserveError::new(
                ObserveErrorKind::NonFinite,
                private_count_field(field, actor, item),
                "scaled count must be finite in float64 and float32",
            )
            .at_seat(seat));
        }
    }
    // IndexMap keys are unique. Known vocabularies bound the ranks by 12;
    // rank/12 and every coordinate, mask and quadrant feature are finite.
    Ok(())
}

fn shed_totals(private: &PrivateState, config: &ObservationConfig) -> (i128, i128) {
    // Admission limits this map to twelve nonnegative i64 values. Both the sum
    // and subtraction fit i128 even when the sum does not fit i64.
    let used: i128 = private.shed.values().map(|count| i128::from(*count)).sum();
    let room = (i128::from(config.shed_capacity) - used).max(0);
    (used, room)
}

fn validate_actors_and_storage(
    snapshot: &StepSnapshot,
    config: &ObservationConfig,
) -> Result<(), ObserveError> {
    for seat in [Seat::Zero, Seat::One] {
        let farm = &snapshot.public.farms[seat.index()];
        let private = &snapshot.privates[seat.index()];
        // validate_public_context already bounded hands.len() to 240.
        if private.inventories.len() != farm.hands.len() + 1 {
            return Err(ObserveError::new(
                ObserveErrorKind::State,
                "actor_inventory",
                format!(
                    "expected one inventory per actor ({}), received {}",
                    farm.hands.len() + 1,
                    private.inventories.len()
                ),
            )
            .at_seat(seat));
        }
        for (actor, position) in std::iter::once(&farm.farmer).chain(&farm.hands).enumerate() {
            if position.len() != 2
                || position
                    .iter()
                    .any(|coordinate| !(0..10).contains(coordinate))
            {
                return Err(ObserveError::new(
                    ObserveErrorKind::State,
                    format!("actor_cell[{actor}]"),
                    "position must be [x,y] with both coordinates in 0..9",
                )
                .at_seat(seat));
            }
        }
        let mut seen = [false; 4];
        for quadrant in &farm.unlocked_quadrants {
            let Some(index) = QUADRANTS.iter().position(|name| *name == quadrant) else {
                return Err(ObserveError::new(
                    ObserveErrorKind::EnumRange,
                    "player_features.quadrants",
                    format!("unknown quadrant {quadrant}"),
                )
                .at_seat(seat));
            };
            if seen[index] {
                return Err(ObserveError::new(
                    ObserveErrorKind::State,
                    "player_features.quadrants",
                    format!("duplicate quadrant {quadrant}"),
                )
                .at_seat(seat));
            }
            seen[index] = true;
        }
        for (actor, inventory) in private.inventories.iter().enumerate() {
            validate_counts(
                inventory,
                &ITEM_NAMES,
                "actor_inventory",
                Some(actor),
                32.0,
                seat,
            )?;
        }
        validate_counts(
            &private.shed,
            &ITEM_NAMES,
            "storage_counts.shed",
            None,
            100.0,
            seat,
        )?;
        validate_counts(
            &private.seeds,
            &SEED_NAMES,
            "storage_counts.seeds",
            None,
            32.0,
            seat,
        )?;
        let (used, room) = shed_totals(private, config);
        checked_f32(
            used as f64 / config.shed_capacity as f64,
            "player_features[0,40]",
            Some(seat),
        )?;
        checked_f32(
            room as f64 / config.shed_capacity as f64,
            "player_features[0,41]",
            Some(seat),
        )?;
    }
    Ok(())
}

fn admitted_index(vocabulary: &[&str], name: &str) -> usize {
    match vocabulary.iter().position(|item| *item == name) {
        Some(index) => index,
        None => unreachable!("prepared item/quadrant admission invariant: {name}"),
    }
}

// Private calculations receive only the requesting seat's state. The rival
// contributes public farms/positions and has no inventory or storage argument.
fn write_actors_and_storage(
    farms: [&Farm; 2],
    own_private: &PrivateState,
    config: &ObservationConfig,
    out: &mut ObsRowMut<'_>,
) {
    for (role, farm) in farms.into_iter().enumerate() {
        out.player_features[role][2] = (farm.unlocked_quadrants.len() as f64 / 4.0) as f32;
        for quadrant in &farm.unlocked_quadrants {
            out.player_features[role][3 + admitted_index(&QUADRANTS, quadrant)] = 1.0;
        }
        let mut empty = 0;
        let mut locked = 0;
        for tile in farm.tiles.iter().flatten() {
            if tile.is_null() {
                empty += 1;
            }
            if tile.as_str() == Some("LOCKED") {
                locked += 1;
            }
        }
        out.player_features[role][7] = (f64::from(empty) / 100.0) as f32;
        out.player_features[role][8] = (f64::from(locked) / 100.0) as f32;
        for (local, position) in std::iter::once(&farm.farmer).chain(&farm.hands).enumerate() {
            let actor = role * 241 + local;
            out.actor_slot[actor] = local as i64;
            out.actor_cell[actor] = 10 * position[1] + position[0];
            out.actor_role[actor] = 2 * role as i64 + i64::from(local > 0);
            out.actor_mask[actor] = true;
            out.actors_float[actor][0] = (position[0] as f64 / 9.0) as f32;
            out.actors_float[actor][1] = (position[1] as f64 / 9.0) as f32;
        }
    }
    for (actor, inventory) in own_private.inventories.iter().enumerate() {
        for (rank, (name, count)) in inventory.iter().enumerate() {
            let item = admitted_index(&ITEM_NAMES, name);
            out.actor_inventory[actor][item] = *count;
            out.actor_inventory_rank[actor][item] = rank as i64 + 1;
            out.actors_float[actor][2 + item] = (*count as f64 / 32.0) as f32;
            out.actors_float[actor][14 + item] = ((rank + 1) as f64 / 12.0) as f32;
        }
    }
    for (rank, (name, count)) in own_private.shed.iter().enumerate() {
        let item = admitted_index(&ITEM_NAMES, name);
        out.storage_counts[item] = *count;
        out.storage_rank[item] = rank as i64 + 1;
        out.player_features[0][11 + item] = (*count as f64 / 100.0) as f32;
        out.player_features[0][28 + item] = ((rank + 1) as f64 / 12.0) as f32;
    }
    for (name, count) in &own_private.seeds {
        let crop = admitted_index(&SEED_NAMES, name);
        out.storage_counts[12 + crop] = *count;
        out.player_features[0][23 + crop] = (*count as f64 / 32.0) as f32;
    }
    let (used, room) = shed_totals(own_private, config);
    out.player_features[0][40] = (used as f64 / config.shed_capacity as f64) as f32;
    out.player_features[0][41] = (room as f64 / config.shed_capacity as f64) as f32;
}

const SHOP_NAMES: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];

fn potential_frames(actors: usize, orders: i64) -> Option<usize> {
    actors
        .checked_add(usize::try_from(orders).ok()?)?
        .checked_add(1)
        .filter(|frames| (1..=252).contains(frames))
}

fn market_integer(value: &Value, item: &str, channel: usize) -> Result<i64, ObserveError> {
    value.as_i64().ok_or_else(|| {
        ObserveError::new(
            ObserveErrorKind::IntegerRange,
            format!("market_int[{item},{channel}]"),
            "expected a JSON integer representation within int64",
        )
    })
}

fn validate_shops_and_market(public: &PublicState) -> Result<(), ObserveError> {
    for (slot, name) in public.town.unlocked_shops.iter().enumerate() {
        if !SHOP_NAMES.contains(&name.as_str()) {
            return Err(ObserveError::new(
                ObserveErrorKind::EnumRange,
                format!("shop_type[{slot}]"),
                format!("unknown shop {name}"),
            ));
        }
    }
    for (channel, values) in [&public.market.inventory, &public.market.prices]
        .into_iter()
        .enumerate()
    {
        for name in values.keys() {
            if !kaggriculture_engine::PRODUCTS.contains(&name.as_str()) {
                return Err(ObserveError::new(
                    ObserveErrorKind::EnumRange,
                    format!("market_int[{name},{channel}]"),
                    "unknown market product",
                ));
            }
        }
        for item in kaggriculture_engine::PRODUCTS {
            let value = values.get(item).ok_or_else(|| {
                ObserveError::new(
                    ObserveErrorKind::State,
                    format!("market_int[{item},{channel}]"),
                    "missing required product",
                )
            })?;
            let integer = market_integer(value, item, channel)?;
            let scaled = integer as f64 / [10000.0, 250.0][channel];
            if !scaled.is_finite() || !(scaled as f32).is_finite() {
                return Err(ObserveError::new(
                    ObserveErrorKind::NonFinite,
                    format!("market_float[{item},{channel}]"),
                    "derived value must be finite",
                ));
            }
        }
    }
    Ok(())
}

fn write_shops_and_market(public: &PublicState, out: &mut ObsRowMut<'_>) {
    for (slot, name) in public.town.unlocked_shops.iter().enumerate() {
        out.shop_type[slot] = admitted_index(&SHOP_NAMES, name) as i64;
        out.shop_slot[slot] = slot as i64;
        out.shop_mask[slot] = true;
    }
    for (product, item) in kaggriculture_engine::PRODUCTS.into_iter().enumerate() {
        out.market_product[product] = product as i64;
        for (channel, values) in [&public.market.inventory, &public.market.prices]
            .into_iter()
            .enumerate()
        {
            let integer = match values.get(item).and_then(Value::as_i64) {
                Some(integer) => integer,
                None => unreachable!("prepared market integer admission invariant: {item}"),
            };
            out.market_int[product][channel] = integer;
            out.market_float[product][channel] =
                (integer as f64 / [10000.0, 250.0][channel]) as f32;
        }
    }
}

fn row_require(condition: bool, field: &str, detail: &str) -> Result<(), ObserveError> {
    if condition {
        Ok(())
    } else {
        Err(ObserveError::new(ObserveErrorKind::State, field, detail))
    }
}

fn row_category(value: i64, upper: i64, field: &str) -> Result<(), ObserveError> {
    if (0..upper).contains(&value) {
        Ok(())
    } else {
        Err(ObserveError::new(
            ObserveErrorKind::EnumRange,
            field,
            format!("category {value} outside 0..{upper}"),
        ))
    }
}

fn row_float(actual: f32, expected: f64, field: &str) -> Result<(), ObserveError> {
    let expected = checked_f32(expected, field, None)?;
    row_require(
        actual.to_bits() == expected.to_bits(),
        field,
        "scaled value or positive-zero padding disagrees with exact context",
    )
}

fn row_finite<'a>(values: impl Iterator<Item = &'a f32>, field: &str) -> Result<(), ObserveError> {
    for (index, value) in values.enumerate() {
        if !value.is_finite() {
            return Err(ObserveError::new(
                ObserveErrorKind::NonFinite,
                format!("{field}[{index}]"),
                "must be finite",
            ));
        }
    }
    Ok(())
}

fn row_ranks(counts: &[i64; 12], ranks: &[i64; 12], field: &str) -> Result<(), ObserveError> {
    let mut seen = [false; 13];
    let mut present = 0;
    for (&count, &rank) in counts.iter().zip(ranks) {
        row_require(count >= 0, field, "count must be nonnegative")?;
        row_category(rank, 13, field)?;
        if rank == 0 {
            row_require(count == 0, field, "absent rank requires zero count")?;
        } else {
            row_require(!seen[rank as usize], field, "duplicate insertion rank")?;
            seen[rank as usize] = true;
            present += 1;
        }
    }
    row_require(
        seen[1..=present].iter().all(|rank| *rank),
        field,
        "insertion ranks must be a contiguous 1-based prefix",
    )
}

fn row_hire_cost(multiplier: i64, hires: i64) -> Result<f32, ObserveError> {
    use num_traits::ToPrimitive;
    if multiplier == 0 {
        return Ok(0.0);
    }
    let multiplier = num_bigint::BigInt::from(multiplier);
    let (mut a, mut b) = (num_bigint::BigInt::from(1), num_bigint::BigInt::from(1));
    // Stop at the first overflow, even for an untrusted i64::MAX count. This
    // diagnostic never iterates a positive-multiplier recurrence to a huge index.
    for index in 0_i64.. {
        let cost = (&multiplier * &a).to_f64().ok_or_else(|| {
            ObserveError::new(
                ObserveErrorKind::NonFinite,
                "player_features.hire_cost",
                "exact cost exceeds float64",
            )
        })?;
        let scaled = checked_f32(cost / 200000.0, "player_features.hire_cost", None)?;
        if index == hires {
            return Ok(scaled);
        }
        (a, b) = (b.clone(), a + b);
    }
    unreachable!("positive Fibonacci costs overflow before i64 index exhaustion")
}

fn check_row_context(out: &ObsRowMut<'_>) -> Result<(), ObserveError> {
    let g = &out.globals_int;
    for (index, value) in g.iter().enumerate() {
        row_require(
            *value >= 0,
            &format!("globals_int[{index}]"),
            "must be nonnegative",
        )?;
    }
    for index in [3, 4, 6, 8, 9, 10] {
        row_require(
            g[index] > 0,
            &format!("globals_int[{index}]"),
            "rule value must be positive",
        )?;
    }
    row_require(
        (1..=10).contains(&g[5]),
        "globals_int[5]",
        "live order limit must be 1..=10",
    )?;
    row_require(
        g[4].checked_mul(g[5]).is_some_and(|product| product <= 240),
        "globals_int",
        "turnsPerDay * orders must be at most 240",
    )?;
    row_require(
        g[1] == g[0] / g[4] && g[2] == g[0] % g[4],
        "globals_int",
        "clock must agree with public rules",
    )?;
    row_require(
        *out.order_limits == g[5],
        "order_limits",
        "must equal configured order limit",
    )?;
    row_require(
        *out.still_playing,
        "still_playing",
        "returned rows must be live",
    )?;
    row_require(
        out.global_features[9] >= 0.0,
        "global_features[9]",
        "weed chance must be nonnegative",
    )?;
    for role in 0..2 {
        row_require(
            (1..=241).contains(&g[14 + role]),
            &format!("globals_int[{}]", 14 + role),
            "actor count must be 1..=241",
        )?;
    }
    let frames = potential_frames(g[14] as usize, *out.order_limits).ok_or_else(|| {
        ObserveError::new(
            ObserveErrorKind::State,
            "action_mask.can_act",
            "invalid potential frame count",
        )
    })?;
    for (frame, present) in out.can_act.iter().enumerate() {
        row_require(
            *present == (frame < frames),
            "action_mask.can_act",
            "mask must agree with own actors + orders + STOP",
        )?;
    }
    let shop_count = out.shop_mask.iter().filter(|present| **present).count();
    let episode = g[3] as f64;
    let day_length = g[4] as f64;
    let remaining = (i128::from((g[3] - 1).max(1)) - i128::from(g[0])).max(0);
    let expected = [
        g[0] as f64 / episode,
        g[2] as f64 / day_length,
        g[1] as f64 / (episode / day_length),
        remaining as f64 / episode,
        day_length / 24.0,
        g[5] as f64 / 10.0,
        g[6] as f64 / 1000.0,
        g[7] as f64 / 100.0,
        episode / 1000.0,
        f64::from(out.global_features[9]),
        g[8] as f64 / day_length,
        g[9] as f64 / day_length,
        g[10] as f64 / day_length,
        g[11] as f64 / 200000.0,
        shop_count as f64 / 8.0,
    ];
    for (index, value) in expected.into_iter().enumerate() {
        row_float(
            out.global_features[index],
            value,
            &format!("global_features[{index}]"),
        )?;
    }
    for role in 0..2 {
        let features = &out.player_features[role];
        row_float(
            features[0],
            out.banks[role] / 200000.0,
            &format!("player_features[{role},0]"),
        )?;
        row_float(
            features[1],
            g[14 + role] as f64 / 241.0,
            &format!("player_features[{role},1]"),
        )?;
        let mut quadrants = 0;
        for (index, value) in features[3..7].iter().enumerate() {
            row_require(
                *value == 0.0 || *value == 1.0,
                &format!("player_features[{role},{}]", index + 3),
                "quadrant flag must be binary",
            )?;
            row_float(
                *value,
                if *value == 1.0 { 1.0 } else { 0.0 },
                "player_features.quadrants",
            )?;
            quadrants += i32::from(*value == 1.0);
        }
        row_float(
            features[2],
            f64::from(quadrants) / 4.0,
            &format!("player_features[{role},2]"),
        )?;
        for (kind, channel) in [(0, 7), (1, 8)] {
            let count = out.tile_kind[role * 100..(role + 1) * 100]
                .iter()
                .filter(|value| **value == kind)
                .count();
            row_float(
                features[channel],
                count as f64 / 100.0,
                &format!("player_features[{role},{channel}]"),
            )?;
        }
        row_float(
            features[9],
            g[12 + role] as f64 / 240.0,
            &format!("player_features[{role},9]"),
        )?;
        let hire = row_hire_cost(g[7], g[12 + role])?;
        row_float(
            features[10],
            f64::from(hire),
            &format!("player_features[{role},10]"),
        )?;
        for (channel, value) in features
            .iter()
            .enumerate()
            .skip(if role == 1 { 11 } else { 42 })
        {
            row_float(*value, 0.0, &format!("player_features[{role},{channel}]"))?;
        }
    }
    Ok(())
}

fn check_row_tiles(out: &ObsRowMut<'_>) -> Result<(), ObserveError> {
    let step = out.globals_int[0];
    let day = out.globals_int[1];
    let episode = out.globals_int[3];
    for token in 0..200 {
        for (field, value, upper) in [
            ("tile_kind", out.tile_kind[token], 6),
            ("tile_crop", out.tile_crop[token], 6),
            ("tile_animal", out.tile_animal[token], 4),
            ("tile_cell", out.tile_cell[token], 100),
            ("tile_role", out.tile_role[token], 2),
        ] {
            row_category(value, upper, &format!("{field}[{token}]"))?;
        }
        row_require(
            out.tile_cell[token] == (token % 100) as i64
                && out.tile_role[token] == (token / 100) as i64,
            "tile_cell/tile_role",
            "tiles must preserve role and row-major cell order",
        )?;
        let plant = out.tile_kind[token] == 3;
        let animal = out.tile_animal[token] != 0;
        row_require(
            if plant {
                out.tile_crop[token] > 0
            } else {
                out.tile_crop[token] == 0
            },
            "tile_crop",
            "crop applicability must match plant kind",
        )?;
        row_require(
            !animal
                || match out.tile_animal[token] {
                    1 => out.tile_kind[token] == 4,
                    2 | 3 => out.tile_kind[token] == 5,
                    _ => false,
                },
            "tile_animal",
            "animal must match compatible structure",
        )?;
        let ints = &out.tiles_int[token];
        let floats = &out.tiles_float[token];
        for (channel, value) in ints.iter().enumerate() {
            let applicable = match channel {
                0 => plant || animal,
                1 | 2 | 3 | 5 => plant,
                4 | 6 => animal,
                _ => unreachable!(),
            };
            if applicable {
                let minimum = if matches!(channel, 1..=4) { -1 } else { 0 };
                if *value < minimum {
                    return Err(ObserveError::new(
                        ObserveErrorKind::Sentinel,
                        format!("tiles_int[{token},{channel}]"),
                        "value below applicable count/sentinel minimum",
                    ));
                }
            } else {
                row_require(
                    *value == 0,
                    &format!("tiles_int[{token},{channel}]"),
                    "inapplicable integer must be zero",
                )?;
            }
        }
        let mut expected = [0.0; 15];
        if plant || animal {
            expected[0] = ints[0] as f64 / 8.0;
        }
        if plant {
            row_require(
                floats[1] == 0.0 || floats[1] == 1.0,
                "tiles_float.watered",
                "flag must be binary",
            )?;
            expected[1] = if floats[1] == 1.0 { 1.0 } else { 0.0 };
            expected[2] = ints[5] as f64 / 8.0;
            expected[3] = if ints[1] == -1 {
                0.0
            } else {
                (i128::from(day) - i128::from(ints[1])) as f64 / 30.0
            };
            expected[4] = f64::from(ints[2] >= 0);
            expected[5] = if ints[2] == -1 {
                0.0
            } else {
                (i128::from(ints[2]) - i128::from(step)) as f64 / episode as f64
            };
            expected[6] = f64::from(ints[3] >= 0);
            expected[7] = f64::from(ints[3] >= day);
            expected[8] = (i128::from(ints[3]) - i128::from(day)).max(0) as f64 / 30.0;
        }
        if animal {
            expected[9] = if ints[4] == -1 {
                0.0
            } else {
                (i128::from(day) - i128::from(ints[4])) as f64 / 30.0
            };
            expected[10] = ints[6] as f64 / 8.0;
            for channel in 11..=13 {
                row_require(
                    floats[channel] == 0.0 || floats[channel] == 1.0,
                    "tiles_float.animal_flag",
                    "flag must be binary",
                )?;
                expected[channel] = if floats[channel] == 1.0 { 1.0 } else { 0.0 };
            }
            row_require(
                floats[14] >= 0.0 && (f64::from(floats[14]) * 8.0).fract() == 0.0,
                "tiles_float.pending_care_bonus",
                "scaled bonus must represent a nonnegative integer divided by eight",
            )?;
            expected[14] = if floats[14] == 0.0 {
                0.0
            } else {
                f64::from(floats[14])
            };
        }
        for (channel, value) in expected.into_iter().enumerate() {
            row_float(
                floats[channel],
                value,
                &format!("tiles_float[{token},{channel}]"),
            )?;
        }
    }
    Ok(())
}

fn check_row_actors_and_storage(out: &ObsRowMut<'_>) -> Result<(), ObserveError> {
    for actor in 0..482 {
        let role = actor / 241;
        let slot = actor % 241;
        row_category(out.actor_slot[actor], 241, "actor_slot")?;
        row_category(out.actor_cell[actor], 100, "actor_cell")?;
        row_category(out.actor_role[actor], 4, "actor_role")?;
        let present = slot < out.globals_int[14 + role] as usize;
        row_require(
            out.actor_mask[actor] == present,
            "actor_mask",
            "mask must be contiguous and agree with exact actor count",
        )?;
        if present {
            row_require(
                out.actor_slot[actor] == slot as i64,
                "actor_slot",
                "present slot must equal local ordinal",
            )?;
            row_require(
                out.actor_role[actor] == 2 * role as i64 + i64::from(slot > 0),
                "actor_role",
                "role must distinguish own/rival farmer/hand",
            )?;
            row_float(
                out.actors_float[actor][0],
                (out.actor_cell[actor] % 10) as f64 / 9.0,
                "actors_float.x",
            )?;
            row_float(
                out.actors_float[actor][1],
                (out.actor_cell[actor] / 10) as f64 / 9.0,
                "actors_float.y",
            )?;
        } else {
            row_require(
                out.actor_slot[actor] == 0
                    && out.actor_cell[actor] == 0
                    && out.actor_role[actor] == 0,
                "actor_padding",
                "absent actor categories must be zero",
            )?;
            for value in &out.actors_float[actor] {
                row_float(*value, 0.0, "actors_float.padding")?;
            }
        }
        if role == 1 {
            for value in &out.actors_float[actor][2..] {
                row_float(*value, 0.0, "actors_float.rival_private")?;
            }
        } else {
            let counts = &out.actor_inventory[actor];
            let ranks = &out.actor_inventory_rank[actor];
            row_ranks(counts, ranks, "actor_inventory_rank")?;
            for item in 0..12 {
                if !present {
                    row_require(
                        counts[item] == 0 && ranks[item] == 0,
                        "actor_inventory.padding",
                        "absent actor counts and ranks must be zero",
                    )?;
                }
                row_float(
                    out.actors_float[actor][2 + item],
                    counts[item] as f64 / 32.0,
                    "actors_float.inventory",
                )?;
                row_float(
                    out.actors_float[actor][14 + item],
                    ranks[item] as f64 / 12.0,
                    "actors_float.rank",
                )?;
            }
        }
    }
    let shed: &[i64; 12] = out.storage_counts[..12]
        .try_into()
        .expect("typed shed count prefix");
    row_ranks(shed, out.storage_rank, "storage_rank")?;
    for (item, count) in out.storage_counts.iter().enumerate() {
        row_require(*count >= 0, "storage_counts", "counts must be nonnegative")?;
        let (channel, divisor) = if item < 12 {
            (11 + item, 100.0)
        } else {
            (23 + item - 12, 32.0)
        };
        row_float(
            out.player_features[0][channel],
            *count as f64 / divisor,
            "player_features.storage",
        )?;
    }
    for (item, rank) in out.storage_rank.iter().enumerate() {
        row_float(
            out.player_features[0][28 + item],
            *rank as f64 / 12.0,
            "player_features.storage_rank",
        )?;
    }
    let used: i128 = out.storage_counts[..12]
        .iter()
        .map(|count| i128::from(*count))
        .sum();
    let room = (i128::from(out.globals_int[6]) - used).max(0);
    row_float(
        out.player_features[0][40],
        used as f64 / out.globals_int[6] as f64,
        "player_features[0,40]",
    )?;
    row_float(
        out.player_features[0][41],
        room as f64 / out.globals_int[6] as f64,
        "player_features[0,41]",
    )?;
    Ok(())
}

fn check_row_shops_and_market(out: &ObsRowMut<'_>) -> Result<(), ObserveError> {
    let count = out.shop_mask.iter().filter(|present| **present).count();
    for slot in 0..8 {
        row_category(out.shop_type[slot], 8, "shop_type")?;
        row_category(out.shop_slot[slot], 8, "shop_slot")?;
        row_require(
            out.shop_mask[slot] == (slot < count),
            "shop_mask",
            "shops must have a contiguous present prefix",
        )?;
        row_require(
            out.shop_slot[slot] == if slot < count { slot as i64 } else { 0 },
            "shop_slot",
            "shop slot must equal present ordinal or zero padding",
        )?;
        if slot >= count {
            row_require(
                out.shop_type[slot] == 0,
                "shop_type",
                "absent shop type must be zero",
            )?;
        }
    }
    for product in 0..9 {
        row_category(out.market_product[product], 9, "market_product")?;
        row_require(
            out.market_product[product] == product as i64,
            "market_product",
            "products must preserve pinned order",
        )?;
        for channel in 0..2 {
            row_float(
                out.market_float[product][channel],
                out.market_int[product][channel] as f64 / [10000.0, 250.0][channel],
                "market_float",
            )?;
        }
    }
    Ok(())
}

/// Scan a complete row for diagnostic/test admission, separate from release
/// preparation and writes. Derived fields are checked against their exact side
/// tensors; masked categories and positive-zero padding are checked as well.
pub fn check_row(out: &ObsRowMut<'_>) -> Result<(), ObserveError> {
    row_finite(out.tiles_float.iter().flatten(), "tiles_float")?;
    row_finite(out.actors_float.iter().flatten(), "actors_float")?;
    row_finite(out.player_features.iter().flatten(), "player_features")?;
    row_finite(out.market_float.iter().flatten(), "market_float")?;
    row_finite(out.global_features.iter(), "global_features")?;
    for bank in out.banks.iter() {
        if !bank.is_finite() {
            return Err(ObserveError::new(
                ObserveErrorKind::NonFinite,
                "banks",
                "must be finite",
            ));
        }
    }
    check_row_context(out)?;
    check_row_tiles(out)?;
    check_row_actors_and_storage(out)?;
    check_row_shops_and_market(out)?;
    Ok(())
}
