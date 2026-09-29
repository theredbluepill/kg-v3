//! Test-only producer for the reviewed observation information-preservation corpus.
use kaggriculture_engine::{
    Config, Counts, Game, InitialState, PublicState, StepSnapshot, TraceHeader, TRACE_FORMAT,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

pub(super) type CorpusResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum OracleSource {
    Official {
        episode: u64,
        step: usize,
    },
    Seeded {
        seed: i64,
        profile: usize,
        step: usize,
        policy_sha256: String,
    },
    Dense {
        case: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OracleRecord {
    pub(super) record_id: String,
    pub(super) source: OracleSource,
    #[serde(deserialize_with = "strict_header")]
    pub(super) header: TraceHeader,
}

fn strict_header<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<TraceHeader, D::Error> {
    let value = Value::deserialize(deserializer)?;
    let object = value
        .as_object()
        .ok_or_else(|| serde::de::Error::custom("header must be object"))?;
    let keys = [
        "format",
        "seed",
        "configuration",
        "shop_schedule",
        "rng_schedule",
        "initial",
        "terminal_banks",
        "transitions",
    ];
    if object.len() != keys.len() || keys.iter().any(|key| !object.contains_key(*key)) {
        return Err(serde::de::Error::custom(
            "complete exact header keys required",
        ));
    }
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}

#[derive(Debug, Serialize)]
struct SeedRun {
    seed: i64,
    profile: usize,
    sampled_steps: Vec<usize>,
    action_sha256: String,
    final_snapshot_sha256: String,
    terminal_step: usize,
    #[serde(skip)]
    actions: Vec<u8>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn native_header(config: Config, seed: i64) -> CorpusResult<TraceHeader> {
    let snapshot = Game::new(config.clone(), seed, 2)?.snapshot();
    Ok(TraceHeader {
        format: TRACE_FORMAT.into(),
        seed: seed.into(),
        configuration: config,
        shop_schedule: vec![],
        rng_schedule: vec![],
        initial: InitialState {
            public: snapshot.public,
            privates: snapshot.privates,
        },
        terminal_banks: vec![],
        transitions: 0,
    })
}

#[test]
fn producer_selection_profiles_and_policy_are_literal() {
    let steps = official_steps();
    assert_eq!(steps.len(), 96);
    assert_eq!(&steps[..3], &[0, 1, 2]);
    assert_eq!(
        (steps[31], steps[32], steps[63], steps[64], steps[95]),
        (31, 344, 375, 687, 718)
    );
    let expected = [
        (24, 10, 1, 100),
        (12, 4, 3, 64),
        (8, 3, 0, 17),
        (6, 1, 7, 256),
        (30, 8, 2, 500),
        (16, 5, 5, 33),
    ];
    for (index, (day, orders, mult, capacity)) in expected.into_iter().enumerate() {
        let value = serde_json::to_value(profile(index)).unwrap();
        assert_eq!(value["episodeSteps"], 96);
        assert_eq!(value["turnsPerDay"], day);
        assert_eq!(value["maxMarketOrdersPerTurn"], orders);
        assert_eq!(value["farmHandCostMult"], mult);
        assert_eq!(value["shedCapacity"], capacity);
    }
    assert_eq!(timing_header(false).unwrap().initial.public.step, 0);
    assert_eq!(
        timing_header(true).unwrap().initial.public.farms[0]
            .hands
            .len(),
        240
    );
}

#[test]
fn producer_policy_uses_all_unit_kinds_and_exact_market_cycle() {
    let mut header = native_header(Config::default(), 42).unwrap();
    header.initial.public.farms[0].hands = vec![vec![0, 0]];
    let initial = policy_actions(&header.initial.public, 3, 24);
    assert_eq!(
        initial[0],
        json!({"farmer":["PASS"],"hands":[["NORTH"]],"market":[["HIRE"],["BUY_LAND"],["HIRE"]]})
    );
    let mut seen = std::collections::BTreeSet::new();
    for step in 0..18 {
        header.initial.public.step = step;
        let actions = policy_actions(&header.initial.public, 10, 24);
        seen.insert(actions[0]["farmer"][0].as_str().unwrap().to_owned());
        assert_eq!(actions[0]["hands"].as_array().unwrap().len(), 1);
        let expected = if step < 8 { 4 } else { 2 };
        assert_eq!(actions[0]["market"].as_array().unwrap().len(), expected);
        if step == 5 {
            assert_eq!(actions[0]["farmer"], json!(["PICKUP", "EGG", 3]));
        }
        if step == 6 {
            assert_eq!(actions[0]["farmer"], json!(["PLACE", "MILK"]));
        }
        if step == 7 {
            assert_eq!(actions[0]["farmer"], json!(["PLANT", "TOMATO"]));
        }
    }
    assert_eq!(seen.len(), 18);
}

// Claude R1 correction (observation-corpus-v2): the v1 market cycle hires on at
// most two of every eight turns and end_of_day clears hands, so no seeded state
// can exceed 16 actors. During hours 0..7 the policy appends [HIRE] entries
// after the unchanged cycle entries until the turn holds min(M, 4) entries.
#[test]
fn producer_policy_v2_appends_hire_burst_within_order_limit() {
    let header = native_header(Config::default(), 42).unwrap();
    let mut public = header.initial.public.clone();
    for (step, orders, expected_len) in [
        (0, 10, 4),
        (7, 10, 4),
        (8, 10, 2),
        (23, 10, 2),
        (24, 10, 4),
        (3, 3, 3),
        (3, 1, 1),
        (3, 2, 2),
    ] {
        public.step = step;
        let v1 = policy_actions(&public, orders.min(2), 24);
        let actions = policy_actions(&public, orders, 24);
        for seat in 0..2 {
            let market = &actions[seat]["market"];
            let entries = market.as_array().unwrap();
            assert_eq!(entries.len(), expected_len, "step={step} M={orders}");
            assert!(entries.len() <= orders);
            let prefix = orders.min(2);
            assert_eq!(
                &entries[..prefix],
                v1[seat]["market"].as_array().unwrap().as_slice()
            );
            for entry in &entries[prefix..] {
                assert_eq!(entry, &json!(["HIRE"]));
            }
        }
    }
    // Burst HIREs stop at 16 hands; cycle entries are never removed.
    public.step = 0;
    for (hands, expected_len) in [(14, 4), (15, 3), (16, 2), (20, 2)] {
        public.farms[0].hands = vec![vec![0, 0]; hands];
        let actions = policy_actions(&public, 10, 24);
        assert_eq!(actions[0]["market"].as_array().unwrap().len(), expected_len);
        assert_eq!(actions[0]["hands"].as_array().unwrap().len(), hands);
        assert_eq!(actions[1]["market"].as_array().unwrap().len(), 4);
    }
}

#[test]
fn producer_dense_pair_changes_only_the_two_key_orders() {
    let first = dense_header(0).unwrap();
    assert_eq!(first.initial.public.farms[0].hands.len(), 240);
    assert_eq!(first.initial.public.farms[1].hands.len(), 240);
    assert_eq!(first.initial.public.farms[1].farmer, vec![3, 1]);
    assert_eq!(first.initial.privates[0].inventories.len(), 241);
    assert_eq!(first.initial.privates[0].inventories[0].len(), 11);
    assert_eq!(
        first.initial.privates[0].inventories[0].first().unwrap().1,
        &0
    );
    assert_eq!(first.initial.public.farms[0].hires_today, 240);
    assert_eq!(
        dense_header(17).unwrap().initial.public.town.unlocked_shops[7],
        "SMOOTHIE_SHOP"
    );
    let a = dense_header(30).unwrap();
    let b = dense_header(31).unwrap();
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    assert_ne!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
    let mut restored = b.clone();
    restored.initial.privates[0].inventories[0] = a.initial.privates[0].inventories[0].clone();
    restored.initial.privates[0].shed = a.initial.privates[0].shed.clone();
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&restored).unwrap()
    );
    assert_eq!(
        a.initial.privates[0].inventories[0].first().unwrap().0,
        b.initial.privates[0].inventories[0].last().unwrap().0
    );
    assert_eq!(
        a.initial.privates[0].shed.first().unwrap().0,
        b.initial.privates[0].shed.last().unwrap().0
    );
    for case in 0..32 {
        verify_roundtrip(&dense_header(case).unwrap()).unwrap();
    }
}

#[test]
fn producer_comparator_detects_values_nested_orders_and_money_bits() {
    assert!(ordered_equal(&json!({"a":1}), &json!({"a":2}), "root").is_err());
    let a: Value = serde_json::from_str(r#"{"outer":{"WHEAT":0,"WOOL":2}}"#).unwrap();
    let b: Value = serde_json::from_str(r#"{"outer":{"WOOL":2,"WHEAT":0}}"#).unwrap();
    assert!(ordered_equal(&a, &b, "root").is_err());
    let mut header = native_header(Config::default(), 42).unwrap();
    header.initial.public.farms[0].money = -0.0;
    header.initial.public.farms[1].money = 1_234_567.123_456_789;
    verify_roundtrip(&header).unwrap();
}

#[test]
fn producer_legacy_domain_rejects_old_oracle_limits_without_narrowing_v3() {
    for case in 0..5 {
        let mut h = native_header(Config::default(), 42).unwrap();
        match case {
            0 => h.configuration.shed_capacity = 16_777_217_i64.into(),
            1 => h.initial.public.farms[0].unlocked_quadrants = vec!["NE".into()],
            2 => {
                h.initial.privates[0]
                    .shed
                    .insert("WHEAT".into(), 9_007_199_254_740_993);
            },
            3 => h.initial.public.farms[0].hires_today = 241,
            4 => {
                h.initial.public.step = 719;
                h.initial.public.day = 29;
                h.initial.public.hour = 23;
            },
            _ => unreachable!(),
        }
        assert!(validate_legacy_domain(&h).is_err(), "case {case}");
    }
}

#[test]
fn producer_seeded_tiny_selection_repeats_exact_bytes_and_action_hashes() {
    let mut first = Vec::new();
    let a = produce_seeded(0, &[0, 6, 94], &mut first).unwrap();
    let mut second = Vec::new();
    let b = produce_seeded(0, &[0, 6, 94], &mut second).unwrap();
    assert_eq!(first, second);
    assert_eq!(a.action_sha256, b.action_sha256);
    assert_eq!(a.final_snapshot_sha256, b.final_snapshot_sha256);
    assert_eq!(a.actions, b.actions);
    assert_eq!(hash_bytes(&a.actions).unwrap(), a.action_sha256);
    assert_eq!(a.terminal_step, 95);
    assert_eq!(a.sampled_steps, [0, 6, 94]);
    assert_eq!(
        first
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .count(),
        3
    );
}

#[test]
fn producer_strict_record_and_quota_guard_reject_bad_inputs() {
    let header = native_header(Config::default(), 42).unwrap();
    let mut row = json!({"record_id":"dense:0","source":{"kind":"dense","case":0},"header":header});
    assert!(serde_json::from_value::<OracleRecord>(row.clone()).is_ok());
    row["extra"] = json!(1);
    assert!(serde_json::from_value::<OracleRecord>(row.clone()).is_err());
    row.as_object_mut().unwrap().shift_remove("extra");
    row["header"]["extra"] = json!(1);
    assert!(serde_json::from_value::<OracleRecord>(row).is_err());
    assert!(quota_error(
        &json!({"non_synthetic":{"actor_gt16_states":0},"dense":{},"shed_order_exception":false})
    )
    .is_err());
}

#[test]
fn producer_gzip_stream_requires_eof_and_success() {
    let path = root().join("engine_rs/fixtures/episode-95324500.jsonl.gz");
    let mut reader = GzipReader::open(&path).unwrap();
    let mut first = String::new();
    reader.read_line(&mut first).unwrap();
    assert!(first.contains("95324500"));
    assert!(reader.finish().is_err());
    let mut reader = GzipReader::open(&path).unwrap();
    io::copy(&mut reader, &mut io::sink()).unwrap();
    reader.finish().unwrap();
}

const EPISODES: [u64; 4] = [95324500, 95901360, 95921764, 95990191];
const PROFILES: [[i64; 9]; 6] = [
    [96, 24, 10, 1, 100, 3, 4, 24, 3000],
    [96, 12, 4, 3, 64, 2, 3, 12, 7500],
    [96, 8, 3, 0, 17, 1, 2, 8, 12345],
    [96, 6, 1, 7, 256, 5, 7, 9, 40000],
    [96, 30, 8, 2, 500, 4, 5, 30, 9000],
    [96, 16, 5, 5, 33, 2, 4, 16, 100000],
];
const WEED: [f64; 6] = [0.005, 0.0, 0.02, 0.01, 0.125, 1.0];
const ITEMS: [&str; 12] = [
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
const UNIT_KINDS: [&str; 18] = [
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
const COVERAGE_TAGS: [&str; 24] = [
    "records",
    "tile_EMPTY",
    "tile_LOCKED",
    "tile_WEED",
    "tile_PLANT",
    "tile_COOP",
    "tile_PASTURE",
    "crop_WHEAT",
    "crop_CARROT",
    "crop_TOMATO",
    "crop_STRAWBERRY",
    "crop_MELON",
    "animal_GOOSE",
    "animal_COW",
    "animal_SHEEP",
    "fert_current",
    "fert_expired",
    "unwatered",
    "unfed",
    "actor_gt16_states",
    "reordered_inventories",
    "reordered_sheds",
    "shops_ge4_states",
    "both_hires_nonzero_states",
];

fn require(condition: bool, message: impl Into<String>) -> CorpusResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn official_steps() -> Vec<usize> {
    (0..32).chain(344..376).chain(687..719).collect()
}
fn profile(index: usize) -> Config {
    let p = PROFILES[index];
    Config {
        episode_steps: p[0].into(),
        turns_per_day: p[1].into(),
        max_market_orders_per_turn: p[2].into(),
        farm_hand_cost_mult: p[3].into(),
        shed_capacity: p[4].into(),
        town_shop_unlock_interval: p[5].into(),
        town_shop_sell_interval: p[6].into(),
        town_center_sell_interval: p[7].into(),
        starting_money: p[8].into(),
        weed_spawn_chance: json!(WEED[index]),
        ..Config::default()
    }
}

fn unit_action(step: usize, seat: usize, actor: usize) -> Value {
    let kind = UNIT_KINDS[(step + seat + actor) % 18];
    let mut command = vec![json!(kind)];
    match kind {
        "PICKUP" | "PLACE" => {
            command.push(json!(ITEMS[(step + actor) % 12]));
            if step % 2 == 1 {
                command.push(json!(1 + step % 3));
            }
        },
        "PLANT" => command.push(json!(ITEMS[(step + actor) % 5])),
        _ => {},
    }
    Value::Array(command)
}

/// Hours per day during which observation-corpus-v2 appends HIRE entries.
const HIRE_BURST_HOURS: usize = 8;
/// Upper bound on market entries per turn under observation-corpus-v2.
const HIRE_BURST_ORDERS: usize = 4;
/// Burst HIREs stop once a farm holds this many hands (17 actors), so the
/// Fibonacci hire cost does not exhaust the bank before later days.
const HIRE_BURST_HANDS: usize = 16;

fn policy_actions(public: &PublicState, orders: usize, turns_per_day: usize) -> Vec<Value> {
    assert!(turns_per_day > 0, "policy requires a positive day length");
    let step = public.step;
    let burst = step % turns_per_day < HIRE_BURST_HOURS;
    public
        .farms
        .iter()
        .enumerate()
        .map(|(seat, farm)| {
            let farmer = unit_action(step, seat, 0);
            let hands: Vec<Value> = (1..=farm.hands.len())
                .map(|actor| unit_action(step, seat, actor))
                .collect();
            let market: Vec<Value> = (0..orders.min(2))
                .map(|entry| match (step + seat + entry) % 8 {
                    0 => json!(["HIRE"]),
                    1 => json!(["BUY_LAND"]),
                    2 => json!(["BUY_SEED", ITEMS[step % 5], 1]),
                    3 => json!(["BUY_PRODUCT", ITEMS[step % 9], 1]),
                    4 => json!(["BUY_ANIMAL", ITEMS[9 + step % 3], 1]),
                    5 => json!(["SELL", ITEMS[step % 12], 1]),
                    6 => json!(["SELL", ITEMS[step % 12], 0]),
                    7 => json!([]),
                    _ => unreachable!(),
                })
                .collect();
            let mut market = market;
            if burst {
                let extra = orders
                    .min(HIRE_BURST_ORDERS)
                    .saturating_sub(market.len())
                    .min(HIRE_BURST_HANDS.saturating_sub(farm.hands.len()));
                market.extend(std::iter::repeat_n(json!(["HIRE"]), extra));
            }
            json!({"farmer":farmer,"hands":hands,"market":market})
        })
        .collect()
}

pub(super) fn dense_header(case: usize) -> CorpusResult<TraceHeader> {
    require(case < 32, "dense case outside recipe")?;
    if case == 31 {
        let mut header = dense_header(30)?;
        let private = &mut header.initial.privates[0];
        let key = private.inventories[0]
            .first()
            .ok_or("empty dense inventory")?
            .0
            .clone();
        let count = private.inventories[0]
            .shift_remove(&key)
            .ok_or("dense inventory key disappeared")?;
        private.inventories[0].insert(key, count);
        let key = private.shed.first().ok_or("empty dense shed")?.0.clone();
        let count = private
            .shed
            .shift_remove(&key)
            .ok_or("dense shed key disappeared")?;
        private.shed.insert(key, count);
        return Ok(header);
    }
    let mut config = profile(2);
    config.max_market_orders_per_turn = if case.is_multiple_of(2) {
        10_i64
    } else {
        3_i64
    }
    .into();
    config.farm_hand_cost_mult = if case.is_multiple_of(2) { 0_i64 } else { 3_i64 }.into();
    let mut header = native_header(config, 11003)?;
    let public = &mut header.initial.public;
    let step = case * 3;
    let day = step / 8;
    public.step = step;
    public.day = day;
    public.hour = step % 8;
    for (seat, (farm, private)) in public
        .farms
        .iter_mut()
        .zip(&mut header.initial.privates)
        .enumerate()
    {
        let position = |actor: usize| {
            vec![
                ((actor + case + 3 * seat) % 10) as i64,
                ((actor / 10 + 2 * case + seat) % 10) as i64,
            ]
        };
        farm.hands = (1..241).map(position).collect();
        farm.farmer = position(0);
        farm.unlocked_quadrants = ["NW", "NE", "SW", "SE"].map(String::from).into();
        farm.hires_today = if case.is_multiple_of(2) {
            240
        } else {
            (case + seat) % 6
        };
        private.inventories = (0..241)
            .map(|actor| {
                let mut counts = Counts::new();
                for item in 0..if actor % 5 == 0 { 11 } else { 12 } {
                    let count = if item == 0 {
                        0
                    } else {
                        (actor + 3 * item + case + seat) % 37
                    };
                    counts.insert(
                        ITEMS[(item + actor + case + seat) % 12].into(),
                        count as i64,
                    );
                }
                counts
            })
            .collect();
        private.shed.clear();
        for item in 0..12 {
            private.shed.insert(
                ITEMS[(item + case + seat) % 12].into(),
                if item == 0 {
                    0
                } else {
                    ((3 * item + case + seat) % 11) as i64
                },
            );
        }
        let mut tiles = vec![
            Value::Null,
            json!("LOCKED"),
            json!({"kind":"WEED"}),
            json!({"kind":"COOP"}),
            json!({"kind":"PASTURE"}),
        ];
        for (index, crop) in ITEMS[..5].iter().enumerate() {
            let crop_index = index + 1;
            tiles.push(json!({"kind":"PLANT","crop":crop,"yield_units":(case+crop_index)%10,
                "watered_today":(case+crop_index).is_multiple_of(2),"consecutive_unwatered":case%4,
                "planted_day":if case.is_multiple_of(4) {-1} else {day.saturating_sub(2) as i64},
                "max_lifespan_step":if case.is_multiple_of(3) {-1} else {step as i64-1},
                "fertilized_until_day":([-1,day.saturating_sub(1) as i64,day as i64,day as i64+2][case%4])}));
        }
        for (index, animal) in ITEMS[9..].iter().enumerate() {
            tiles.push(json!({"kind":if index == 0 {"COOP"} else {"PASTURE"},"animal":animal,
                "yield_units":(case+index+1)%10,"placed_day":if case.is_multiple_of(4) {-1} else {day.saturating_sub(1) as i64},
                "consecutive_unfed":case%3,"fed_today":case.is_multiple_of(2),"cared_today":case.is_multiple_of(3),
                "fertilizer_available":case.is_multiple_of(4),"pending_care_bonus":case%12}));
        }
        for (cell, tile) in tiles.into_iter().enumerate() {
            farm.tiles[cell / 10][cell % 10] = tile;
        }
    }
    public.town.unlocked_shops = [
        "BAKERY",
        "PIZZA_SHOP",
        "BAKERY",
        "YARN_STORE",
        "BRUNCH_SPOT",
        "FARMERS_MARKET",
        "ICE_CREAM_SHOP",
        "PET_CAFE",
    ][..case % 9]
        .iter()
        .map(|name| (*name).into())
        .collect();
    if case == 17 {
        public.town.unlocked_shops[7] = "SMOOTHIE_SHOP".into();
    }
    Ok(header)
}

pub(super) fn ordered_equal(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!("value mismatch at {path}"));
    }
    match (actual, expected) {
        (Value::Object(a), Value::Object(b)) => {
            if a.keys().ne(b.keys()) {
                return Err(format!("key-order mismatch at {path}"));
            }
            for (key, value) in a {
                ordered_equal(value, &b[key], &format!("{path}.{key}"))?;
            }
        },
        (Value::Array(a), Value::Array(b)) => {
            // Equality above establishes equal lengths before zipping.
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                ordered_equal(a, b, &format!("{path}[{index}]"))?;
            }
        },
        _ => {},
    }
    Ok(())
}

fn config_i64(value: &kaggriculture_engine::PyInt) -> CorpusResult<i64> {
    Ok(serde_json::to_string(value)?.parse()?)
}

pub(super) fn validate_legacy_domain(header: &TraceHeader) -> CorpusResult<()> {
    // Keep these additional constraints solely in oracle admission, not v3.
    {
        let game = super::ObservationGame::from_header(header)?;
        game.prepare()?;
    }
    require(header.format == TRACE_FORMAT, "oracle trace format")?;
    require(
        header.shop_schedule.is_empty()
            && header.rng_schedule.is_empty()
            && header.terminal_banks.is_empty()
            && header.transitions == 0,
        "oracle headers must contain no future schedules",
    )?;
    let ep = config_i64(&header.configuration.episode_steps)?;
    let public = &header.initial.public;
    require(
        (public.step as i128) < i128::from((ep - 1).max(1)),
        "oracle samples must be preterminal",
    )?;
    for (field, value) in [
        ("episodeSteps", ep),
        (
            "turnsPerDay",
            config_i64(&header.configuration.turns_per_day)?,
        ),
        (
            "farmHandCostMult",
            config_i64(&header.configuration.farm_hand_cost_mult)?,
        ),
        (
            "shedCapacity",
            config_i64(&header.configuration.shed_capacity)?,
        ),
    ] {
        require(
            (0..=16_777_216).contains(&value),
            format!("legacy suffix {field} exceeds exact f32 domain"),
        )?;
    }
    for farm in &public.farms {
        require(farm.hires_today <= 240, "legacy hires_today outside 0..240")?;
        let names = ["NW", "NE", "SW", "SE"];
        require(
            (1..=4).contains(&farm.unlocked_quadrants.len())
                && farm
                    .unlocked_quadrants
                    .iter()
                    .map(String::as_str)
                    .eq(names[..farm.unlocked_quadrants.len()].iter().copied()),
            "legacy quadrants must be canonical NW/NE/SW/SE prefix",
        )?;
        for tile in farm.tiles.iter().flatten() {
            if let Some(bonus) = tile.get("pending_care_bonus") {
                require(
                    bonus
                        .as_i64()
                        .is_some_and(|bonus| (0..=16_777_216).contains(&bonus)),
                    "legacy pending_care_bonus outside exact f32 domain",
                )?;
            }
        }
    }
    for private in &header.initial.privates {
        for counts in [&private.shed, &private.seeds] {
            let total: u128 = counts
                .values()
                .map(|count| u128::from(count.unsigned_abs()))
                .sum();
            require(
                total <= 1_u128 << 53,
                "legacy private absolute aggregate exceeds 2^53",
            )?;
        }
        for counts in &private.inventories {
            require(
                counts
                    .values()
                    .all(|count| (0..=16_777_216).contains(count)),
                "oracle actor inventory outside exact raw f32 test domain",
            )?;
        }
    }
    Ok(())
}

fn verify_roundtrip(header: &TraceHeader) -> CorpusResult<()> {
    validate_legacy_domain(header)?;
    let bytes = serde_json::to_vec(header)?;
    let parsed: TraceHeader = serde_json::from_slice(&bytes)?;
    let game = Game::from_header(&parsed)?;
    let snapshot = game.snapshot();
    ordered_equal(
        &serde_json::to_value(&snapshot.public)?,
        &serde_json::to_value(&header.initial.public)?,
        "roundtrip.public",
    )?;
    ordered_equal(
        &serde_json::to_value(&snapshot.privates)?,
        &serde_json::to_value(&header.initial.privates)?,
        "roundtrip.privates",
    )?;
    for (actual, expected) in snapshot
        .public
        .farms
        .iter()
        .zip(&header.initial.public.farms)
    {
        require(
            actual.money.to_bits() == expected.money.to_bits(),
            "roundtrip changed money bits",
        )?;
    }
    Ok(())
}

pub(super) struct GzipReader {
    child: Option<Child>,
    inner: BufReader<ChildStdout>,
}
impl GzipReader {
    pub(super) fn open(path: &Path) -> CorpusResult<Self> {
        let mut child = Command::new("gzip")
            .arg("-dc")
            .arg(path)
            .stdout(Stdio::piped())
            .spawn()?;
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err("gzip stdout unavailable".into());
        };
        Ok(Self {
            child: Some(child),
            inner: BufReader::new(stdout),
        })
    }
    pub(super) fn finish(mut self) -> CorpusResult<()> {
        require(
            self.fill_buf()?.is_empty(),
            "gzip stream must reach exact EOF before finish",
        )?;
        let status = self.child.take().ok_or("gzip already finished")?.wait()?;
        require(status.success(), format!("gzip failed: {status}"))
    }
}
impl Read for GzipReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.inner.read(out)
    }
}
impl BufRead for GzipReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }
    fn consume(&mut self, count: usize) {
        self.inner.consume(count);
    }
}
impl Drop for GzipReader {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub(super) fn hash_bytes(bytes: &[u8]) -> CorpusResult<String> {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let result = (|| -> CorpusResult<()> {
        let mut stdin = child.stdin.take().ok_or("shasum stdin unavailable")?;
        stdin.write_all(bytes)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    let output = child.wait_with_output()?;
    result?;
    require(output.status.success(), "shasum failed")?;
    let hash = String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .ok_or("missing SHA-256")?
        .to_owned();
    require(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid shasum output",
    )?;
    Ok(hash)
}

fn verify_fixture_hash(episode: u64) -> CorpusResult<PathBuf> {
    let relative = format!("engine_rs/fixtures/episode-{episode}.jsonl.gz");
    let manifest: Value =
        serde_json::from_reader(File::open(root().join("engine_rs/TRIM_MANIFEST.json"))?)?;
    let retained = manifest["retained"]
        .as_array()
        .ok_or("trim manifest retained list")?;
    let entry = retained
        .iter()
        .find(|entry| entry["path"] == relative)
        .ok_or("fixture missing from trim manifest")?;
    let expected = entry["sha256"].as_str().ok_or("fixture SHA-256 missing")?;
    let path = root().join(&relative);
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(&path)
        .output()?;
    require(output.status.success(), "fixture shasum failed")?;
    require(
        String::from_utf8(output.stdout)?.split_whitespace().next() == Some(expected),
        format!("official fixture hash mismatch: {relative}"),
    )?;
    Ok(path)
}

fn header_from_snapshot(
    config: &Config,
    seed: serde_json::Number,
    snapshot: StepSnapshot,
) -> TraceHeader {
    TraceHeader {
        format: TRACE_FORMAT.into(),
        seed,
        configuration: config.clone(),
        shop_schedule: vec![],
        rng_schedule: vec![],
        initial: InitialState {
            public: snapshot.public,
            privates: snapshot.privates,
        },
        terminal_banks: vec![],
        transitions: 0,
    }
}

fn write_record(record: &OracleRecord, out: &mut impl Write) -> CorpusResult<()> {
    verify_roundtrip(&record.header)?;
    serde_json::to_writer(&mut *out, record)?;
    out.write_all(b"\n")?;
    Ok(())
}

fn produce_seeded(index: usize, steps: &[usize], out: &mut impl Write) -> CorpusResult<SeedRun> {
    require(index < 6, "seed profile outside recipe")?;
    require(
        steps.windows(2).all(|pair| pair[0] < pair[1]) && steps.iter().all(|step| *step <= 94),
        "seed samples must be unique, ordered and preterminal",
    )?;
    let config = profile(index);
    let seed = 11001 + index as i64;
    // Compact action bytes only; sample headers are streamed on the second pass.
    let mut actions = Vec::new();
    let final_snapshot_sha256;
    {
        let mut game = Game::new(config.clone(), seed, 2)?;
        for step in 0..95 {
            let public = game.public_state();
            require(public.step == step, "seeded first-pass clock")?;
            let action = policy_actions(
                &public,
                PROFILES[index][2] as usize,
                PROFILES[index][1] as usize,
            );
            serde_json::to_writer(&mut actions, &action)?;
            actions.push(b'\n');
            game.step(&action)?;
        }
        let final_snapshot = game.snapshot();
        require(
            final_snapshot.done && final_snapshot.public.step == 95,
            "seeded terminal step must equal 95",
        )?;
        final_snapshot_sha256 = hash_bytes(&serde_json::to_vec(&final_snapshot)?)?;
    }
    let action_sha256 = hash_bytes(&actions)?;
    let mut game = Game::new(config.clone(), seed, 2)?;
    let mut count = 0;
    for (step, line) in actions
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        require(
            game.public_state().step == step,
            "seed replay clock mismatch",
        )?;
        if steps.contains(&step) {
            let record = OracleRecord {
                record_id: format!("seeded:{seed}:{step}"),
                source: OracleSource::Seeded {
                    seed,
                    profile: index,
                    step,
                    policy_sha256: action_sha256.clone(),
                },
                header: header_from_snapshot(&config, seed.into(), game.snapshot()),
            };
            write_record(&record, out)?;
            count += 1;
        }
        let action: Vec<Value> = serde_json::from_slice(line)?;
        game.step(&action)?;
    }
    require(count == steps.len(), "seeded sample count mismatch")?;
    require(
        hash_bytes(&serde_json::to_vec(&game.snapshot())?)? == final_snapshot_sha256,
        "seed replay final snapshot hash mismatch",
    )?;
    Ok(SeedRun {
        seed,
        profile: index,
        sampled_steps: steps.to_vec(),
        action_sha256,
        final_snapshot_sha256,
        terminal_step: 95,
        actions,
    })
}

fn read_json_line(reader: &mut impl BufRead, context: &str) -> CorpusResult<Value> {
    let mut line = String::new();
    require(
        reader.read_line(&mut line)? > 0,
        format!("unexpected EOF: {context}"),
    )?;
    require(
        line.ends_with('\n'),
        format!("unterminated JSONL record: {context}"),
    )?;
    Ok(serde_json::from_str(&line)?)
}

fn official_header(reader: &mut impl BufRead, episode: u64) -> CorpusResult<(TraceHeader, Value)> {
    let raw = read_json_line(reader, "official header")?;
    require(
        raw["type"] == "header" && raw["source"]["episode_id"] == episode,
        "official header identity mismatch",
    )?;
    require(
        raw["source"]["module_version"] == "1.32.7"
            && raw["source"]["engine_sha256"]
                == "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
        "official engine identity mismatch",
    )?;
    let header: TraceHeader = serde_json::from_value(raw.clone())?;
    require(
        header.format == TRACE_FORMAT && header.transitions == 719,
        "official header format/transition mismatch",
    )?;
    Ok((header, raw))
}

fn compare_snapshot(
    snapshot: &StepSnapshot,
    public: &Value,
    privates: &Value,
    context: &str,
) -> CorpusResult<()> {
    ordered_equal(
        &serde_json::to_value(&snapshot.public)?,
        public,
        &format!("{context}.public"),
    )?;
    ordered_equal(
        &serde_json::to_value(&snapshot.privates)?,
        privates,
        &format!("{context}.privates"),
    )?;
    let expected_farms = public["farms"]
        .as_array()
        .ok_or("official public farms must be array")?;
    require(
        snapshot.public.farms.len() == expected_farms.len(),
        "official farm count mismatch",
    )?;
    for (actual, expected) in snapshot.public.farms.iter().zip(expected_farms) {
        let money = expected["money"].as_f64().ok_or("official money missing")?;
        require(
            actual.money.to_bits() == money.to_bits(),
            format!("{context}: money bits differ"),
        )?;
    }
    Ok(())
}

fn produce_official(episode: u64, out: &mut impl Write) -> CorpusResult<()> {
    let mut reader = GzipReader::open(&verify_fixture_hash(episode)?)?;
    let (header, raw) = official_header(&mut reader, episode)?;
    let mut game = Game::new(
        header.configuration.clone(),
        header.seed.as_i64().ok_or("official seed must fit i64")?,
        2,
    )?;
    let initial = game.snapshot();
    compare_snapshot(
        &initial,
        &raw["initial"]["public"],
        &raw["initial"]["privates"],
        &format!("episode {episode} initial"),
    )?;
    require(
        !initial.done && initial.public.step == 0,
        "official initial clock/status mismatch",
    )?;
    write_record(
        &OracleRecord {
            record_id: format!("official:{episode}:0"),
            source: OracleSource::Official { episode, step: 0 },
            header: header_from_snapshot(&header.configuration, header.seed.clone(), initial),
        },
        out,
    )?;
    let selected = official_steps();
    let mut emitted = 1;
    for from_step in 0..719 {
        let row = read_json_line(
            &mut reader,
            &format!("episode {episode} transition {from_step}"),
        )?;
        require(
            row["type"] == "transition" && row["from_step"].as_u64() == Some(from_step as u64),
            "official transition sequence mismatch",
        )?;
        game.step(
            row["actions"]
                .as_array()
                .ok_or("official actions must be array")?,
        )?;
        let snapshot = game.snapshot();
        let context = format!("episode {episode} step {}", from_step + 1);
        compare_snapshot(&snapshot, &row["expected"], &row["privates"], &context)?;
        let statuses: Vec<String> = serde_json::from_value(row["statuses"].clone())?;
        let rewards: Vec<f64> = serde_json::from_value(row["rewards"].clone())?;
        require(
            snapshot.statuses == statuses && snapshot.rewards == rewards,
            format!("{context}: status/reward mismatch"),
        )?;
        require(
            snapshot.public.step == from_step + 1 && snapshot.done == (from_step == 718),
            format!("{context}: clock/done mismatch"),
        )?;
        if selected.contains(&(from_step + 1)) {
            let step = from_step + 1;
            write_record(
                &OracleRecord {
                    record_id: format!("official:{episode}:{step}"),
                    source: OracleSource::Official { episode, step },
                    header: header_from_snapshot(
                        &header.configuration,
                        header.seed.clone(),
                        snapshot,
                    ),
                },
                out,
            )?;
            emitted += 1;
        }
    }
    require(emitted == 96, "official sample count must equal 96")?;
    require(
        game.terminal_banks() == Some(header.terminal_banks.as_slice()),
        "official terminal banks mismatch",
    )?;
    reader.finish()
}

/// Deterministic sparse/dense sources shared by the allocation and timing checks.
pub(super) fn timing_header(dense: bool) -> CorpusResult<TraceHeader> {
    if dense {
        return dense_header(0);
    }
    let mut reader = GzipReader::open(&verify_fixture_hash(EPISODES[0])?)?;
    let (header, raw) = official_header(&mut reader, EPISODES[0])?;
    let snapshot = Game::new(
        header.configuration.clone(),
        header.seed.as_i64().ok_or("official seed must fit i64")?,
        2,
    )?
    .snapshot();
    compare_snapshot(
        &snapshot,
        &raw["initial"]["public"],
        &raw["initial"]["privates"],
        "timing initial",
    )?;
    // Even this single-header utility consumes and checks the gzip child fully.
    io::copy(&mut reader, &mut io::sink())?;
    reader.finish()?;
    Ok(header_from_snapshot(
        &header.configuration,
        header.seed,
        snapshot,
    ))
}

type Coverage = BTreeMap<String, u64>;

fn empty_coverage() -> Coverage {
    COVERAGE_TAGS
        .into_iter()
        .map(|tag| (tag.into(), 0))
        .collect()
}
fn increment(coverage: &mut Coverage, tag: &str) -> CorpusResult<()> {
    *coverage
        .get_mut(tag)
        .ok_or_else(|| format!("unknown coverage tag {tag}"))? += 1;
    Ok(())
}
fn reordered(counts: &Counts) -> CorpusResult<bool> {
    let mut previous = None;
    for key in counts.keys() {
        let rank = ITEMS
            .iter()
            .position(|item| item == key)
            .ok_or("unknown item in coverage")?;
        if previous.is_some_and(|previous| rank < previous) {
            return Ok(true);
        }
        previous = Some(rank);
    }
    Ok(false)
}

fn add_coverage(coverage: &mut Coverage, header: &TraceHeader) -> CorpusResult<()> {
    let public = &header.initial.public;
    increment(coverage, "records")?;
    if public.farms.iter().any(|farm| farm.hands.len() + 1 > 16) {
        increment(coverage, "actor_gt16_states")?;
    }
    if public.town.unlocked_shops.len() >= 4 {
        increment(coverage, "shops_ge4_states")?;
    }
    if public.farms.iter().all(|farm| farm.hires_today > 0) {
        increment(coverage, "both_hires_nonzero_states")?;
    }
    for private in &header.initial.privates {
        if reordered(&private.shed)? {
            increment(coverage, "reordered_sheds")?;
        }
        for counts in &private.inventories {
            if reordered(counts)? {
                increment(coverage, "reordered_inventories")?;
            }
        }
    }
    for farm in &public.farms {
        for tile in farm.tiles.iter().flatten() {
            let kind = if tile.is_null() {
                "EMPTY"
            } else if tile == "LOCKED" {
                "LOCKED"
            } else {
                tile["kind"].as_str().ok_or("coverage tile kind missing")?
            };
            increment(coverage, &format!("tile_{kind}"))?;
            if kind == "PLANT" {
                increment(
                    coverage,
                    &format!(
                        "crop_{}",
                        tile["crop"].as_str().ok_or("coverage crop missing")?
                    ),
                )?;
                let fert = tile["fertilized_until_day"]
                    .as_i64()
                    .ok_or("coverage fertilizer deadline missing")?;
                if i128::from(fert) >= public.day as i128 {
                    increment(coverage, "fert_current")?;
                }
                if fert >= 0 && i128::from(fert) < public.day as i128 {
                    increment(coverage, "fert_expired")?;
                }
                if tile["consecutive_unwatered"]
                    .as_i64()
                    .ok_or("coverage unwatered missing")?
                    > 0
                {
                    increment(coverage, "unwatered")?;
                }
            }
            if let Some(animal) = tile.get("animal") {
                increment(
                    coverage,
                    &format!(
                        "animal_{}",
                        animal.as_str().ok_or("coverage animal missing")?
                    ),
                )?;
                if tile["consecutive_unfed"]
                    .as_i64()
                    .ok_or("coverage unfed missing")?
                    > 0
                {
                    increment(coverage, "unfed")?;
                }
            }
        }
    }
    Ok(())
}

fn quota_error(coverage: &Value) -> CorpusResult<()> {
    let non = coverage["non_synthetic"]
        .as_object()
        .ok_or("non-synthetic coverage missing")?;
    let mut failures = Vec::new();
    for tag in COVERAGE_TAGS {
        let minimum = match tag {
            "records" => 480,
            "tile_EMPTY"
            | "tile_LOCKED"
            | "tile_WEED"
            | "tile_PLANT"
            | "tile_COOP"
            | "tile_PASTURE"
            | "reordered_inventories"
            | "shops_ge4_states"
            | "both_hires_nonzero_states" => 8,
            "reordered_sheds" => u64::from(coverage["shed_order_exception"] != true),
            _ => 4,
        };
        let count = non
            .get(tag)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("coverage count missing: {tag}"))?;
        if count < minimum {
            failures.push(format!("{tag}={count} < {minimum}"));
        }
    }
    require(
        failures.is_empty(),
        format!(
            "R1 non-synthetic coverage quotas failed: {}",
            failures.join(", ")
        ),
    )
}

fn produce_all(directory: &Path) -> CorpusResult<()> {
    for name in [
        "manifest.json",
        "states.jsonl.gz",
        "reference.f32le.gz",
        "states.jsonl",
        "generation.json",
    ] {
        require(
            !directory.join(name).exists(),
            format!("refusing existing corpus file {name}"),
        )?;
    }
    let states = directory.join("states.jsonl");
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&states)?,
    );
    for episode in EPISODES {
        produce_official(episode, &mut output)?;
    }
    let steps: Vec<usize> = (0..16).map(|index| index * 94 / 15).collect();
    let mut seed_runs = Vec::new();
    for profile in 0..6 {
        let run = produce_seeded(profile, &steps, &mut output)?;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(format!("actions-{}.jsonl", run.seed)))?
            .write_all(&run.actions)?;
        seed_runs.push(run);
    }
    for case in 0..32 {
        write_record(
            &OracleRecord {
                record_id: format!("dense:{case}"),
                source: OracleSource::Dense { case },
                header: dense_header(case)?,
            },
            &mut output,
        )?;
    }
    output.flush()?;
    drop(output);
    let mut non_synthetic = empty_coverage();
    let mut dense = empty_coverage();
    for line in BufReader::new(File::open(&states)?).lines() {
        let record: OracleRecord = serde_json::from_str(&line?)?;
        add_coverage(
            if matches!(record.source, OracleSource::Dense { .. }) {
                &mut dense
            } else {
                &mut non_synthetic
            },
            &record.header,
        )?;
    }
    let records = non_synthetic["records"] + dense["records"];
    let shed_order_exception =
        non_synthetic["reordered_sheds"] == 0 && dense["reordered_sheds"] > 0;
    let coverage = json!({"non_synthetic":non_synthetic,"dense":dense,"shed_order_exception":shed_order_exception});
    let mut report = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("generation.json"))?,
    );
    serde_json::to_writer_pretty(
        &mut report,
        &json!({"records":records,"coverage":coverage,"seed_runs":seed_runs}),
    )?;
    report.write_all(b"\n")?;
    report.flush()?;
    // Retain actual coverage and seed hashes even when the fixed R1 recipe fails.
    require(
        records == 512,
        format!("corpus contains {records} records, expected 512"),
    )?;
    quota_error(&coverage)
}

#[test]
#[ignore = "explicit bounded oracle generation; R1 coverage failure blocks final publication"]
fn generate_observation_oracle_inputs() {
    let directory = PathBuf::from(
        std::env::var_os("KG_OBS_ORACLE_OUT").expect("KG_OBS_ORACLE_OUT is required"),
    );
    std::fs::create_dir_all(&directory).unwrap();
    produce_all(&directory).unwrap();
}

const LEGACY_FEATURES: usize = 8176;

fn legacy_group(offset: usize) -> &'static str {
    match offset {
        0..=15 => "public/private summary",
        16..=615 => "packed public tiles",
        616..=679 => "first sixteen actor coordinates",
        680..=871 => "first sixteen inventories",
        872..=888 => "seeds and shed",
        889..=906 => "market blocks",
        907..=959 => "dropped positive-zero range",
        960..=1023 => "ordered shop slots",
        1024 => "availability constant",
        1025..=1026 => "full actor counts",
        1027..=3826 => "tile maintenance",
        3827..=5272 => "all actor coordinates",
        5273..=8164 => "all own inventories",
        8165 => "investment constant",
        8166..=8175 => "exact rule suffix",
        _ => unreachable!("legacy offset outside vector"),
    }
}

fn compare_legacy_bits(
    actual: &[f32; LEGACY_FEATURES],
    recorded: &[f32; LEGACY_FEATURES],
    record: &str,
    seat: usize,
) -> CorpusResult<()> {
    for offset in 0..LEGACY_FEATURES {
        if actual[offset].to_bits() != recorded[offset].to_bits() {
            return Err(format!(
                "record={record} seat={seat} offset={offset} field={}: actual={:?} recorded={:?}",
                legacy_group(offset),
                actual[offset],
                recorded[offset],
            )
            .into());
        }
    }
    Ok(())
}

struct LegacyWrites {
    values: [f32; LEGACY_FEATURES],
    covered: [bool; LEGACY_FEATURES],
}

impl LegacyWrites {
    fn new() -> Self {
        Self {
            values: [0.0; LEGACY_FEATURES],
            covered: [false; LEGACY_FEATURES],
        }
    }
    fn put(&mut self, offset: usize, value: f64) {
        assert!(
            !self.covered[offset],
            "legacy offset {offset} written twice"
        );
        self.covered[offset] = true;
        self.values[offset] = value as f32;
    }
    fn finish(self) -> [f32; LEGACY_FEATURES] {
        for (offset, covered) in self.covered.into_iter().enumerate() {
            assert!(covered, "legacy offset {offset} never reconstructed");
        }
        self.values
    }
}

/// Tensor-only reconstruction of the reviewed audit table. Legacy packing and
/// all offsets are isolated in this test module. Corpus-domain checks happen
/// before encoding; they are intentionally not v3 production admission limits.
pub(super) fn reconstruct_legacy(row: &super::ObsRowMut<'_>) -> [f32; LEGACY_FEATURES] {
    let g = &row.globals_int;
    let step = g[0] as f64;
    let day = g[1] as f64;
    let episode = g[3] as f64;
    let turns = g[4] as f64;
    let total = |values: &[i64]| values.iter().map(|v| i128::from(*v)).sum::<i128>() as f64;
    let mut out = LegacyWrites::new();
    let quadrant_count = |role: usize| {
        row.player_features[role][3..7]
            .iter()
            .map(|v| f64::from(*v))
            .sum::<f64>()
    };
    let tile_count = |role: usize, kind| {
        row.tile_kind[role * 100..(role + 1) * 100]
            .iter()
            .filter(|v| **v == kind)
            .count() as f64
    };
    for (offset, value) in [
        step / episode,
        (step % turns) / turns,
        day / (episode / turns).max(1.0),
        row.banks[0] / 200000.0,
        row.banks[1] / 200000.0,
        g[14].min(16) as f64 / 16.0,
        g[15].min(16) as f64 / 16.0,
        quadrant_count(0) / 4.0,
        quadrant_count(1) / 4.0,
        tile_count(0, 0) / 100.0,
        tile_count(1, 0) / 100.0,
        tile_count(0, 1) / 100.0,
        tile_count(1, 1) / 100.0,
        total(&row.storage_counts[..12]) / 100.0,
        total(&row.storage_counts[12..]) / 100.0,
        row.shop_mask.iter().filter(|v| **v).count() as f64 / 8.0,
    ]
    .into_iter()
    .enumerate()
    {
        out.put(offset, value);
    }
    for tile in 0..200 {
        let kind = row.tile_kind[tile];
        let animal = row.tile_animal[tile];
        let ints = row.tiles_int[tile];
        let floats = row.tiles_float[tile];
        let code = match kind {
            0 => 0.05,
            1 => 0.0,
            2 => 0.1,
            3 => 0.2 + 0.025 * (row.tile_crop[tile] - 1) as f64,
            _ if animal != 0 => 0.6 + 0.05 * (animal - 1) as f64,
            4 => 0.45,
            5 => 0.5,
            _ => unreachable!("admitted tile kind"),
        };
        let age = if kind == 3 {
            ((day - ints[1] as f64) / 30.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let state = if kind == 3 {
            (ints[0] as f64 / 8.0).min(1.0)
                + 0.25 * f64::from(floats[1])
                + 0.125 * f64::from(ints[3] >= 0)
        } else if animal != 0 {
            (ints[0] as f64 / 8.0).min(1.0)
                + 0.25 * f64::from(floats[11])
                + 0.125 * f64::from(floats[12])
                + 0.0625 * f64::from(floats[13])
        } else {
            0.0
        };
        for (channel, value) in [code, age, state].into_iter().enumerate() {
            out.put(16 + tile * 3 + channel, value);
        }
        let maintenance = if kind == 3 {
            [
                1.0,
                0.0,
                ints[0] as f64,
                f64::from(floats[1]),
                ints[5] as f64,
                ints[1] as f64,
                ints[2] as f64,
                ints[3] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        } else if animal != 0 {
            [
                0.0,
                1.0,
                ints[0] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                ints[4] as f64,
                ints[6] as f64,
                f64::from(floats[11]),
                f64::from(floats[12]),
                f64::from(floats[13]),
                8.0 * f64::from(floats[14]),
            ]
        } else {
            [0.0; 14]
        };
        for (channel, value) in maintenance.into_iter().enumerate() {
            out.put(1027 + tile * 14 + channel, value);
        }
    }
    for role in 0..2 {
        let count = row.actor_mask[role * 241..(role + 1) * 241]
            .iter()
            .filter(|v| **v)
            .count();
        assert_eq!(count as i64, g[14 + role], "legacy actor count role {role}");
        out.put(1025 + role, count as f64);
        for actor in 0..16 {
            let index = role * 241 + actor;
            let cell = if row.actor_mask[index] {
                row.actor_cell[index]
            } else {
                0
            };
            out.put(616 + 32 * role + 2 * actor, (cell % 10) as f64 / 10.0);
            out.put(617 + 32 * role + 2 * actor, (cell / 10) as f64 / 10.0);
        }
    }
    for actor in 0..482 {
        let present = row.actor_mask[actor];
        let cell = if present { row.actor_cell[actor] } else { 0 };
        out.put(3827 + actor * 3, f64::from(present));
        out.put(3828 + actor * 3, (cell % 10) as f64);
        out.put(3829 + actor * 3, (cell / 10) as f64);
    }
    for actor in 0..241 {
        for item in 0..12 {
            let count = row.actor_inventory[actor][item] as f64;
            if actor < 16 {
                out.put(680 + actor * 12 + item, count / 32.0);
            }
            out.put(5273 + actor * 12 + item, count);
        }
    }
    for crop in 0..5 {
        out.put(872 + crop, row.storage_counts[12 + crop] as f64 / 32.0);
    }
    for item in 0..12 {
        out.put(877 + item, row.storage_counts[item] as f64 / 100.0);
    }
    for product in 0..9 {
        out.put(889 + product, row.market_int[product][0] as f64 / 10000.0);
        out.put(898 + product, row.market_int[product][1] as f64 / 250.0);
    }
    for offset in 907..960 {
        out.put(offset, 0.0);
    }
    for slot in 0..8 {
        for category in 0..8 {
            let present = (0..8).any(|shop| {
                row.shop_mask[shop]
                    && row.shop_slot[shop] == slot
                    && row.shop_type[shop] == category
            });
            out.put(960 + (slot * 8 + category) as usize, f64::from(present));
        }
    }
    out.put(1024, 1.0);
    out.put(8165, 1.0);
    let quadrant_mask = (0..4)
        .map(|quadrant| (row.player_features[0][3 + quadrant] as i64) * (1 << quadrant))
        .sum::<i64>();
    for (offset, value) in [
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
        g[12],
        g[7],
        quadrant_mask,
        g[6],
        g[5],
    ]
    .into_iter()
    .enumerate()
    {
        out.put(8166 + offset, value as f64);
    }
    out.finish()
}

// This is a literal tensor fixture, independent of native encoding and corpus
// recipes. Its fixed expected values below are hand calculations, not oracle
// recordings. A passing fixture never stands in for full corpus qualification.
fn hand_reconstruction_row(row: &mut super::ObsRowMut<'_>) {
    row.clear();
    *row.globals_int = [
        239, 9, 23, 720, 24, 3, 100, 7, 2, 3, 24, 3000, 4, 5, 241, 241,
    ];
    *row.banks = [1234.125, -64.5];
    row.player_features[0][3..7].copy_from_slice(&[1.0, 1.0, 0.0, 0.0]);
    row.player_features[1][3..7].copy_from_slice(&[1.0; 4]);
    row.tile_kind[..5].copy_from_slice(&[3, 1, 2, 4, 5]);
    row.tile_crop[0] = 3;
    row.tiles_int[0] = [3, 1, 200, 8, 0, 2, 0];
    row.tiles_float[0][1] = 1.0;
    row.tiles_float[0][6] = 1.0;
    row.tiles_float[0][7] = 0.0;
    row.tile_kind[199] = 5;
    row.tile_animal[199] = 3;
    row.tiles_int[199] = [2, 0, 0, 0, 3, 0, 1];
    row.tiles_float[199][12] = 1.0;
    row.tiles_float[199][13] = 1.0;
    row.tiles_float[199][14] = 0.5;
    row.actor_mask.fill(true);
    for (actor, cell) in [(0, 27), (15, 98), (240, 39), (256, 46), (481, 87)] {
        row.actor_cell[actor] = cell;
    }
    row.actor_inventory[0][0] = 7;
    row.actor_inventory[15][11] = 9;
    row.actor_inventory[240][11] = 11;
    row.storage_counts[0] = 20;
    row.storage_counts[11] = 5;
    row.storage_counts[12] = 3;
    row.storage_counts[16] = 4;
    row.market_int[0] = [-12345, 77];
    row.market_int[8] = [55, 7777];
    *row.shop_type = [5, 0, 5, 1, 2, 3, 6, 7];
    *row.shop_slot = [0, 1, 2, 3, 4, 5, 6, 7];
    row.shop_mask.fill(true);
}

fn hand_reconstruction_expected() -> [f32; LEGACY_FEATURES] {
    let mut expected = [0.0; LEGACY_FEATURES];
    expected[..16].copy_from_slice(
        &[
            239.0 / 720.0,
            23.0 / 24.0,
            9.0 / 30.0,
            1234.125 / 200000.0,
            -64.5 / 200000.0,
            1.0,
            1.0,
            0.5,
            1.0,
            0.95,
            0.99,
            0.01,
            0.0,
            0.25,
            0.07,
            1.0,
        ]
        .map(|value: f64| value as f32),
    );
    for tile in 0..200 {
        expected[16 + 3 * tile] = 0.05;
    }
    expected[16..19].copy_from_slice(&[0.25, (8.0_f64 / 30.0) as f32, 0.75]);
    expected[19] = 0.0;
    expected[22] = 0.1;
    expected[25] = 0.45;
    expected[28] = 0.5;
    expected[613..616].copy_from_slice(&[0.7, 0.0, 0.4375]);
    for (offset, value) in [
        (616, 0.7),
        (617, 0.2),
        (646, 0.8),
        (647, 0.9),
        (678, 0.6),
        (679, 0.4),
    ] {
        expected[offset] = value;
    }
    expected[680] = 7.0 / 32.0;
    expected[871] = 9.0 / 32.0;
    expected[872] = 3.0 / 32.0;
    expected[876] = 4.0 / 32.0;
    expected[877] = 0.2;
    expected[888] = 0.05;
    expected[889] = (-12345.0_f64 / 10000.0) as f32;
    expected[897] = (55.0_f64 / 10000.0) as f32;
    expected[898] = (77.0_f64 / 250.0) as f32;
    expected[906] = (7777.0_f64 / 250.0) as f32;
    for offset in [965, 968, 981, 985, 994, 1003, 1014, 1023] {
        expected[offset] = 1.0;
    }
    expected[1024..1027].copy_from_slice(&[1.0, 241.0, 241.0]);
    expected[1027..1041].copy_from_slice(&[
        1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 200.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    expected[3813..3827].copy_from_slice(&[
        0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 1.0, 4.0,
    ]);
    for actor in 0..482 {
        expected[3827 + 3 * actor] = 1.0;
    }
    for (offset, x, y) in [
        (3827, 7.0, 2.0),
        (3872, 8.0, 9.0),
        (4547, 9.0, 3.0),
        (4595, 6.0, 4.0),
        (5270, 7.0, 8.0),
    ] {
        expected[offset + 1] = x;
        expected[offset + 2] = y;
    }
    expected[5273] = 7.0;
    expected[5464] = 9.0;
    expected[8164] = 11.0;
    expected[8165..].copy_from_slice(&[
        1.0, 239.0, 9.0, 23.0, 720.0, 24.0, 4.0, 7.0, 3.0, 100.0, 3.0,
    ]);
    expected
}

#[test]
fn reconstruction_hand_anchors_cover_every_range_and_boundary() {
    let mut storage = super::ObsStaging::new(1).unwrap();
    let mut buffers = storage.buffers_mut();
    let mut env = buffers.envs_mut().next().unwrap();
    let row = &mut env.seats[0];
    hand_reconstruction_row(row);
    // Temporary mutation receipt injection point: clean source has no mutation.
    let actual = reconstruct_legacy(row);
    let expected = hand_reconstruction_expected();
    compare_legacy_bits(&actual, &expected, "independent-hand-vector", 0).unwrap();
    for (offset, value) in [
        (615, 0.4375),
        (616, 0.7),
        (679, 0.4),
        (680, 7.0 / 32.0),
        (871, 9.0 / 32.0),
        (872, 3.0 / 32.0),
        (906, (7777.0_f64 / 250.0) as f32),
        (907, 0.0),
        (959, 0.0),
        (960, 0.0),
        (1024, 1.0),
        (1027, 1.0),
        (3827, 1.0),
        (5273, 7.0),
        (8165, 1.0),
        (8175, 3.0),
    ] {
        assert_eq!(
            actual[offset].to_bits(),
            value.to_bits(),
            "hand boundary {offset}"
        );
    }
}

#[test]
fn reconstruction_old_day_rule_sentinel_age_and_clamping_are_explicit() {
    let mut storage = super::ObsStaging::new(1).unwrap();
    let mut buffers = storage.buffers_mut();
    let mut env = buffers.envs_mut().next().unwrap();
    let row = &mut env.seats[0];
    hand_reconstruction_row(row);
    row.globals_int[3] = 2;
    row.global_features[2] = 108.0; // new rule: day/(Ep/D), which legacy must ignore
    row.tiles_int[0][1] = -1;
    row.tiles_int[0][0] = 80;
    let actual = reconstruct_legacy(row);
    assert_eq!(actual[2], 9.0); // legacy max(1, Ep/D), even in synthetic postterminal context
    assert_eq!(actual[17], (10.0_f64 / 30.0) as f32);
    assert_eq!(actual[18], 1.375);
    for (planted, age) in [(10, 0.0), (0, 0.3), (-1, (10.0_f64 / 30.0) as f32)] {
        row.tiles_int[0][1] = planted;
        assert_eq!(reconstruct_legacy(row)[17], age);
    }
}

#[test]
fn reconstruction_comparator_names_record_seat_group_offset_and_signed_zero() {
    let expected = hand_reconstruction_expected();
    for offset in [
        0, 16, 616, 680, 872, 889, 907, 960, 1024, 1025, 1027, 3827, 5273, 8165, 8175,
    ] {
        let mut actual = expected;
        actual[offset] += 1.0;
        let error = compare_legacy_bits(&actual, &expected, "control", 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("record=control seat=1 offset={offset}")));
        assert!(error.contains(legacy_group(offset)));
    }
    let mut actual = expected;
    actual[907] = -0.0;
    assert!(compare_legacy_bits(&actual, &expected, "signed-zero", 0).is_err());
}

fn check_recorded_redundancies(
    recorded: &[f32; LEGACY_FEATURES],
    record: &str,
    seat: usize,
) -> CorpusResult<()> {
    let equal = |offset: usize, expected: f32| {
        require(recorded[offset].to_bits() == expected.to_bits(), format!(
        "record={record} seat={seat} offset={offset} field={}: recorded constant/duplicate mismatch",
        legacy_group(offset),
    ))
    };
    for offset in 907..960 {
        equal(offset, 0.0)?;
    }
    equal(1024, 1.0)?;
    equal(8165, 1.0)?;
    let step = f64::from(recorded[8166]);
    let day = f64::from(recorded[8167]);
    let episode = f64::from(recorded[8169]);
    let turns = f64::from(recorded[8170]);
    equal(0, (step / episode) as f32)?;
    equal(1, ((step % turns) / turns) as f32)?;
    equal(2, (day / (episode / turns).max(1.0)) as f32)?;
    for role in 0..2 {
        let mut count = 0;
        for actor in 0..241 {
            let raw = 3827 + (role * 241 + actor) * 3;
            let present = recorded[raw];
            require(
                present.to_bits() == 0 || present == 1.0,
                format!(
                    "record={record} seat={seat} offset={raw}: invalid recorded actor presence"
                ),
            )?;
            count += usize::from(present == 1.0);
            if present == 0.0 {
                equal(raw + 1, 0.0)?;
                equal(raw + 2, 0.0)?;
            }
            if actor < 16 {
                equal(
                    616 + role * 32 + actor * 2,
                    (f64::from(recorded[raw + 1]) / 10.0) as f32,
                )?;
                equal(
                    617 + role * 32 + actor * 2,
                    (f64::from(recorded[raw + 2]) / 10.0) as f32,
                )?;
            }
        }
        equal(1025 + role, count as f32)?;
        equal(5 + role, (count.min(16) as f64 / 16.0) as f32)?;
    }
    for actor in 0..16 {
        for item in 0..12 {
            equal(
                680 + actor * 12 + item,
                (f64::from(recorded[5273 + actor * 12 + item]) / 32.0) as f32,
            )?;
        }
    }
    Ok(())
}

fn decode_feature_byteplanes(
    source: &mut (impl Read + io::Seek),
    out: &mut impl Write,
    bytes: u64,
) -> CorpusResult<()> {
    require(
        bytes.is_multiple_of(4),
        "byte-plane stream must be f32 aligned",
    )?;
    require(
        source.seek(io::SeekFrom::End(0))? == bytes,
        "byte-plane stream length differs from expected bytes",
    )?;
    let values = bytes / 4;
    let mut plane_bytes = [0_u8; 8192];
    let mut interleaved = [0_u8; 8192 * 4];
    let mut start = 0;
    while start < values {
        let length = usize::try_from((values - start).min(8192))?;
        for plane in 0..4 {
            source.seek(io::SeekFrom::Start(plane as u64 * values + start))?;
            source.read_exact(&mut plane_bytes[..length])?;
            for index in 0..length {
                interleaved[index * 4 + plane] = plane_bytes[index];
            }
        }
        out.write_all(&interleaved[..length * 4])?;
        start += length as u64;
    }
    Ok(())
}

#[test]
fn reconstruction_recorded_constants_and_duplicates_are_checked_independently() {
    let golden = hand_reconstruction_expected();
    check_recorded_redundancies(&golden, "hand-recorded-control", 1).unwrap();
    for offset in [907, 959, 1024, 8165, 616, 679, 680, 871, 5, 6, 1025, 1026] {
        let mut corrupted = golden;
        corrupted[offset] += 1.0;
        assert!(
            check_recorded_redundancies(&corrupted, "corrupt-control", 1).is_err(),
            "missed offset {offset}"
        );
    }
    let mut negative_zero = golden;
    negative_zero[907] = -0.0;
    assert!(check_recorded_redundancies(&negative_zero, "negative-zero", 0).is_err());
}

#[test]
fn reconstruction_byteplane_decode_preserves_bits_and_rejects_bad_lengths() {
    let patterns = [0_u32, 0x8000_0000, 0x3dcc_cccd, 0xc0e8_0000, 1];
    let raw: Vec<u8> = (0..8195)
        .map(|index| patterns[index % patterns.len()])
        .flat_map(u32::to_le_bytes)
        .collect();
    let mut shuffled = Vec::new();
    for plane in 0..4 {
        for value in raw.chunks_exact(4) {
            shuffled.push(value[plane]);
        }
    }
    let mut decoded = Vec::new();
    let bytes = raw.len() as u64;
    decode_feature_byteplanes(&mut io::Cursor::new(&shuffled), &mut decoded, bytes).unwrap();
    assert_eq!(decoded, raw);
    assert!(decode_feature_byteplanes(
        &mut io::Cursor::new(&shuffled[..shuffled.len() - 1]),
        &mut Vec::new(),
        bytes
    )
    .is_err());
    assert!(
        decode_feature_byteplanes(&mut io::Cursor::new(&shuffled), &mut Vec::new(), bytes - 1)
            .is_err()
    );
}

#[test]
#[should_panic(expected = "written twice")]
fn reconstruction_coverage_rejects_duplicate_offset() {
    let mut coverage = LegacyWrites::new();
    coverage.put(615, 0.0);
    coverage.put(615, 0.0);
}

#[test]
#[should_panic(expected = "never reconstructed")]
fn reconstruction_coverage_rejects_unwritten_zero_hole() {
    let mut coverage = LegacyWrites::new();
    for offset in 0..LEGACY_FEATURES {
        if offset != 959 {
            coverage.put(offset, 0.0);
        }
    }
    coverage.finish();
}

struct FeatureScratch {
    directory: PathBuf,
}

impl FeatureScratch {
    fn new() -> CorpusResult<Self> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "kg-observation-features-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        Ok(Self { directory })
    }
}

impl Drop for FeatureScratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

enum ReferenceFeatures {
    Plain(GzipReader),
    Decoded {
        reader: BufReader<File>,
        _scratch: FeatureScratch,
    },
}

impl ReferenceFeatures {
    fn open(path: &Path, format: &str, bytes: u64) -> CorpusResult<Self> {
        match format {
            "kaggriculture-observation-oracle-f32le-gzip-v1" => {
                Ok(Self::Plain(GzipReader::open(path)?))
            },
            "kaggriculture-observation-oracle-f32le-byteplanes-gzip-v1" => {
                let scratch = FeatureScratch::new()?;
                let shuffled = scratch.directory.join("planes.bin");
                let decoded = scratch.directory.join("reference.f32le");
                let mut gzip = GzipReader::open(path)?;
                let mut stored = OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create_new(true)
                    .open(&shuffled)?;
                let copied = io::copy(&mut (&mut gzip).take(bytes + 1), &mut stored)?;
                require(
                    copied == bytes,
                    "shuffled reference feature byte count mismatch",
                )?;
                gzip.finish()?;
                let mut raw = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&decoded)?;
                decode_feature_byteplanes(&mut stored, &mut raw, bytes)?;
                raw.flush()?;
                Ok(Self::Decoded {
                    reader: BufReader::new(File::open(decoded)?),
                    _scratch: scratch,
                })
            },
            _ => Err(format!("unsupported reference feature format {format}").into()),
        }
    }

    fn finish(self) -> CorpusResult<()> {
        match self {
            Self::Plain(gzip) => gzip.finish(),
            Self::Decoded {
                mut reader,
                _scratch,
            } => require(
                reader.fill_buf()?.is_empty(),
                "decoded features have trailing bytes",
            ),
        }
    }
}

impl Read for ReferenceFeatures {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Plain(gzip) => gzip.read(bytes),
            Self::Decoded { reader, .. } => reader.read(bytes),
        }
    }
}

#[test]
fn reconstruction_feature_streams_support_both_formats_and_exact_eof() {
    let scratch = FeatureScratch::new().unwrap();
    let raw: Vec<u8> = [0_u32, 0x8000_0000, 0x3dcc_cccd, 0xc0e8_0000, 1]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let mut planes = Vec::new();
    for plane in 0..4 {
        for value in raw.chunks_exact(4) {
            planes.push(value[plane]);
        }
    }
    for (index, (format, bytes)) in [
        ("kaggriculture-observation-oracle-f32le-gzip-v1", &raw),
        (
            "kaggriculture-observation-oracle-f32le-byteplanes-gzip-v1",
            &planes,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let path = scratch.directory.join(format!("fixture-{index}.gz"));
        let mut compressor = Command::new("gzip")
            .args(["-n", "-c"])
            .stdin(Stdio::piped())
            .stdout(File::create(&path).unwrap())
            .spawn()
            .unwrap();
        compressor.stdin.take().unwrap().write_all(bytes).unwrap();
        assert!(compressor.wait().unwrap().success());
        let mut reader = ReferenceFeatures::open(&path, format, 20).unwrap();
        let mut result = [0; 20];
        reader.read_exact(&mut result).unwrap();
        assert_eq!(result.as_slice(), raw);
        reader.finish().unwrap();
        let mut unread = ReferenceFeatures::open(&path, format, 20).unwrap();
        unread.read_exact(&mut [0_u8; 4]).unwrap();
        assert!(unread.finish().is_err());
    }
    assert!(
        ReferenceFeatures::open(&scratch.directory.join("absent"), "unknown-format", 20).is_err()
    );
}

fn compare_corpus_streams(directory: &Path) -> CorpusResult<()> {
    let manifest_path = directory.join("manifest.json");
    require(
        manifest_path.is_file(),
        format!(
            "observation oracle fixture missing: {} not found; regenerate it with \
             scripts/kaggriculture_observation_oracle/regenerate.py --output {}",
            manifest_path.display(),
            directory.display()
        ),
    )?;
    // Admission checks hashes, exact record/source identity, all coverage quotas,
    // record order and per-seat byte blocks before any tensor comparison. The
    // existing environment is reused: nested editable Cargo builds are forbidden.
    let validation = Command::new("uv")
        .current_dir(root())
        .args([
            "run",
            "--offline",
            "--no-sync",
            "python",
            "scripts/kaggriculture_observation_oracle/regenerate.py",
            "--check",
            "--output",
        ])
        .arg(directory)
        .status()?;
    require(
        validation.success(),
        "source-bound observation oracle custody validation failed",
    )?;
    let manifest: Value = serde_json::from_reader(File::open(manifest_path)?)?;
    require(
        manifest["reference_shape"] == json!([512, 2, LEGACY_FEATURES]),
        "reference shape differs from [512,2,8176]",
    )?;
    let format = manifest["format"]
        .as_str()
        .ok_or("missing reference format")?;
    let mut states = GzipReader::open(&directory.join("states.jsonl.gz"))?;
    let mut features =
        ReferenceFeatures::open(&directory.join("reference.f32le.gz"), format, 33_488_896)?;
    let mut output = super::ObsStaging::new(1)?;
    for record_index in 0..512 {
        let record: OracleRecord =
            serde_json::from_value(read_json_line(&mut states, "oracle state")?)?;
        require(
            manifest["records"][record_index]["record_id"] == record.record_id,
            format!("record order differs from admitted manifest at {record_index}"),
        )?;
        validate_legacy_domain(&record.header)?;
        let game = super::ObservationGame::from_header(&record.header)?;
        let mut buffers = output.buffers_mut();
        let mut env = buffers
            .envs_mut()
            .next()
            .ok_or("missing output environment")?;
        super::encode_env(&game, &mut env)?;
        for (seat, row) in env.seats.iter().enumerate() {
            super::check_row(row)?;
            let mut bytes = [0_u8; LEGACY_FEATURES * 4];
            features.read_exact(&mut bytes).map_err(|error| {
                format!(
                    "record={} seat={seat}: incomplete reference row: {error}",
                    record.record_id
                )
            })?;
            let mut recorded = [0_f32; LEGACY_FEATURES];
            for (value, bytes) in recorded.iter_mut().zip(bytes.chunks_exact(4)) {
                *value =
                    f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            check_recorded_redundancies(&recorded, &record.record_id, seat)?;
            compare_legacy_bits(&reconstruct_legacy(row), &recorded, &record.record_id, seat)?;
        }
    }
    states.finish()?;
    features.finish()
}

#[test]
fn missing_observation_oracle_names_files_and_regeneration_only() {
    let scratch = FeatureScratch::new().unwrap();
    let message = compare_corpus_streams(&scratch.directory)
        .unwrap_err()
        .to_string();
    assert!(message.contains("manifest.json"), "{message}");
    assert!(message.contains("regenerate.py"), "{message}");
    // Absent files are not coverage evidence; no quota diagnosis is asserted.
    assert!(!message.contains("quota"), "{message}");
    assert!(!message.contains("0 < 4"), "{message}");
}

#[test]
fn compare_observation_oracle() {
    let directory = root().join("tests/fixtures/kaggriculture/observation-v3");
    compare_corpus_streams(&directory).unwrap();
}
