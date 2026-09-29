//! Literal native port of EcoBot v7 Unified Evaluator.
//!
//! Frozen Python source: `agents/ecobot/main.py`, SHA-256
//! `0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067`.
//! The controller owns evaluator drift/cull state and its destructive daily route
//! queues.  It never calls Python and never substitutes a prerecorded action tape.

// Several deliberately literal translations preserve Python's boundary expressions,
// indexed board walk, nested dictionary guards, and purchase-order branch shape.  Keep
// these localized to this parity port rather than rewriting already-proved semantics.
#![allow(
    clippy::collapsible_if,
    clippy::if_same_then_else,
    clippy::int_plus_one,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_unwrap
)]

use crate::{Farm, Game, Inventory, PrivateState};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

type Pos = (i64, i64);
type Counts = IndexMap<String, i64>;

const TOTAL_DAYS: usize = 30;
const TURNS_PER_DAY: usize = 24;
const PURCHASE_HOUR: usize = 21;
const MAX_MARKET_ORDERS: usize = 10;
const MAX_HANDS: usize = 13;
const BOARD_SIZE: usize = 10;
const I0: i64 = 10_000;
const LAST_DAY: usize = 29;
const LIQUIDATION_DAY: usize = 28;
const MELON_LAST_PLANT_DAY: usize = 18;
const STRAWBERRY_LAST_PLANT_DAY: usize = 14;
const WHEAT_LAST_PLANT_DAY: usize = 26;
const PASTURE_RESERVATION_LAST_DAY: usize = 16;
const MAX_TOTAL_HERD: i64 = 18;
const HAND_COST_PER_ANIMAL_DAY: f64 = 2.0;
const SEED_PURCHASE_CASH_RESERVE: f64 = 200.0;
const FEED_WHEAT_RESERVE: i64 = 4;
const FEED_WHEAT_RESERVE_MIN_RATIO: i64 = 2;
const FEED_WHEAT_SELL_RATIO: i64 = 4;
const TOWN_CENTER_FLAT: f64 = 1.0;
const TOWN_SHOP_UNLOCK_INTERVAL: usize = 3;
const MAX_SHOP_UNLOCKS: usize = 8;
const RECOURSE_PROBABILITY_THRESHOLD: f64 = 0.5;

const PASTURE_CLUSTER: [Pos; 18] = [
    (4, 4),
    (5, 4),
    (4, 5),
    (4, 3),
    (3, 4),
    (5, 3),
    (6, 4),
    (3, 5),
    (4, 6),
    (3, 3),
    (4, 2),
    (2, 4),
    (6, 3),
    (5, 2),
    (7, 4),
    (2, 5),
    (3, 6),
    (4, 7),
];
const SHED_TILES: [Pos; 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
const SHED_CENTER: Pos = (4, 4);
const BLOCKED_CLUSTER: [Pos; 12] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (9, 0),
    (8, 0),
    (9, 1),
    (0, 9),
    (1, 9),
    (0, 8),
    (9, 9),
    (8, 9),
    (9, 8),
];
const SELLABLE_ITEMS: [&str; 9] = [
    "FERTILIZER",
    "MILK",
    "WOOL",
    "EGG",
    "MELON",
    "STRAWBERRY",
    "CARROT",
    "TOMATO",
    "WHEAT",
];
const ANIMAL_SPECIES: [&str; 3] = ["COW", "SHEEP", "GOOSE"];
const PASTURE_SPECIES: [&str; 2] = ["COW", "SHEEP"];
const PREMIUM_DROP_CROPS: [&str; 2] = ["MELON", "STRAWBERRY"];
const FIBONACCI_HIRE_COSTS: [i64; 20] = [
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
];

const URG_FEED_CRITICAL: i64 = 100;
const URG_PLACE: i64 = 95;
const URG_HARVEST_PREMIUM: i64 = 85;
const URG_HARVEST: i64 = 80;
const URG_FERTILIZE: i64 = 77;
const URG_WATER: i64 = 75;
const URG_PLANT_PRIORITY: i64 = 72;
const URG_PLANT: i64 = 71;
const URG_FEED: i64 = 70;
const URG_COLLECT_FERTILIZER: i64 = 68;
const URG_CARE: i64 = 65;
const URG_DIG_EXPIRED: i64 = 30;
const URG_DIG_WEED: i64 = 20;

#[derive(Clone, Copy)]
struct CropSpec {
    seed_cost: i64,
    first_yield_day: i64,
    max_yield_day: i64,
    max_units_base: i64,
    ongoing: bool,
    yield_days: &'static [i64],
}

fn crop(name: &str) -> Option<CropSpec> {
    Some(match name {
        "WHEAT" => CropSpec {
            seed_cost: 10,
            first_yield_day: 2,
            max_yield_day: 4,
            max_units_base: 4,
            ongoing: false,
            yield_days: &[],
        },
        "CARROT" => CropSpec {
            seed_cost: 20,
            first_yield_day: 2,
            max_yield_day: 3,
            max_units_base: 3,
            ongoing: false,
            yield_days: &[],
        },
        "TOMATO" => CropSpec {
            seed_cost: 50,
            first_yield_day: 8,
            max_yield_day: 11,
            max_units_base: 0,
            ongoing: true,
            yield_days: &[8, 9, 10, 11],
        },
        "STRAWBERRY" => CropSpec {
            seed_cost: 100,
            first_yield_day: 10,
            max_yield_day: 16,
            max_units_base: 0,
            ongoing: true,
            yield_days: &[10, 12, 14, 16],
        },
        "MELON" => CropSpec {
            seed_cost: 80,
            first_yield_day: 10,
            max_yield_day: 10,
            max_units_base: 6,
            ongoing: false,
            yield_days: &[],
        },
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct AnimalSpec {
    cost: i64,
    first_yield_day: i64,
    interval: i64,
    product: &'static str,
}

fn animal(name: &str) -> Option<AnimalSpec> {
    Some(match name {
        "GOOSE" => AnimalSpec {
            cost: 300,
            first_yield_day: 4,
            interval: 1,
            product: "EGG",
        },
        "COW" => AnimalSpec {
            cost: 400,
            first_yield_day: 8,
            interval: 2,
            product: "MILK",
        },
        "SHEEP" => AnimalSpec {
            cost: 500,
            first_yield_day: 6,
            interval: 3,
            product: "WOOL",
        },
        _ => return None,
    })
}

fn shop_demands(shop: &str) -> &'static [(&'static str, f64)] {
    match shop {
        "BAKERY" => &[("EGG", 6.0), ("WHEAT", 6.0)],
        "PIZZA_SHOP" => &[("MILK", 6.0), ("TOMATO", 6.0), ("WHEAT", 6.0)],
        "BRUNCH_SPOT" => &[("EGG", 6.0), ("WHEAT", 6.0), ("STRAWBERRY", 6.0)],
        "YARN_STORE" => &[("WOOL", 12.0)],
        "ICE_CREAM_SHOP" => &[("STRAWBERRY", 6.0), ("MILK", 6.0), ("WHEAT", 6.0)],
        "PET_CAFE" => &[("CARROT", 12.0)],
        "SMOOTHIE_SHOP" => &[("STRAWBERRY", 6.0), ("MILK", 6.0)],
        "FARMERS_MARKET" => &[
            ("WHEAT", 6.0),
            ("CARROT", 6.0),
            ("TOMATO", 6.0),
            ("STRAWBERRY", 6.0),
        ],
        _ => &[],
    }
}

fn count(map: &Counts, key: &str) -> i64 {
    *map.get(key).unwrap_or(&0)
}
fn inv_count(map: &Inventory, key: &str) -> i64 {
    *map.get(key).unwrap_or(&0)
}
fn total_hire_cost(n: usize, already: usize) -> i64 {
    FIBONACCI_HIRE_COSTS[already..already + n].iter().sum()
}
fn shape(kind: &str, x: f64, t: f64) -> f64 {
    match kind {
        "linear" => x,
        "sq" => x * x,
        "sqrt" => x.max(0.0).sqrt(),
        "log" => x.max(0.0).ln_1p(),
        "hinge" => {
            let u = x / t.max(1.0);
            u + 8.0 * (u - 1.0).max(0.0).powi(2)
        }
        _ => panic!("unknown shape {kind}"),
    }
}
fn market_params(item: &str) -> (f64, f64, &'static str, f64, &'static str, f64) {
    match item {
        "WHEAT" => (25.0, 400.0, "sqrt", 0.80, "log", 0.20),
        "CARROT" => (35.0, 450.0, "hinge", 1.00, "sqrt", 0.70),
        "TOMATO" => (60.0, 200.0, "hinge", 0.40, "sqrt", 0.60),
        "STRAWBERRY" => (120.0, 100.0, "sqrt", 0.70, "linear", 1.60),
        "MELON" => (250.0, 300.0, "log", 0.20, "sq", 3.60),
        "EGG" => (50.0, 332.0, "hinge", 0.40, "log", 0.20),
        "MILK" => (160.0, 122.0, "sqrt", 0.60, "linear", 1.60),
        "WOOL" => (200.0, 105.0, "log", 0.20, "sq", 3.20),
        "FERTILIZER" => (100.0, 200.0, "linear", 0.40, "linear", 0.40),
        _ => panic!("unknown market item {item}"),
    }
}
fn market_price(item: &str, inventory: i64) -> i64 {
    let (base, t, bf, bt, af, at) = market_params(item);
    if inventory == I0 {
        return base.round_ties_even() as i64;
    }
    let (kind, target, sign, x) = if inventory < I0 {
        (bf, bt, 1.0, (I0 - inventory) as f64)
    } else {
        (af, at, -1.0, (inventory - I0) as f64)
    };
    let denom = shape(kind, t, t);
    let amp = if denom > 0.0 {
        target * base / denom
    } else {
        0.0
    };
    (base + sign * amp * shape(kind, x, t))
        .round_ties_even()
        .max(1.0) as i64
}
fn manhattan(a: Pos, b: Pos) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
fn quad_of(p: Pos) -> &'static str {
    match (p.1 < 5, p.0 >= 5) {
        (true, true) => "NE",
        (true, false) => "NW",
        (false, true) => "SE",
        (false, false) => "SW",
    }
}
fn bfs_step(start: Pos, goal: Pos) -> Option<&'static str> {
    if start == goal {
        return None;
    }
    let (dx, dy) = (goal.0 - start.0, goal.1 - start.1);
    if dx.abs() >= dy.abs() {
        Some(if dx > 0 { "EAST" } else { "WEST" })
    } else {
        Some(if dy > 0 { "SOUTH" } else { "NORTH" })
    }
}

#[derive(Clone, Default)]
struct AnimalCensus {
    field_cows: i64,
    field_sheep: i64,
    field_geese: i64,
    shed_cows: i64,
    shed_sheep: i64,
    shed_geese: i64,
    carried_cows: i64,
    carried_sheep: i64,
    carried_geese: i64,
}
impl AnimalCensus {
    fn total_cows(&self) -> i64 {
        self.field_cows + self.shed_cows + self.carried_cows
    }
    fn total_sheep(&self) -> i64 {
        self.field_sheep + self.shed_sheep + self.carried_sheep
    }
    fn total_geese(&self) -> i64 {
        self.field_geese + self.shed_geese + self.carried_geese
    }
    fn total(&self) -> i64 {
        self.total_cows() + self.total_sheep()
    }
    fn total_all(&self) -> i64 {
        self.total() + self.total_geese()
    }
}

#[derive(Clone)]
struct AnimalTile {
    pos: Pos,
    animal: String,
    fed_today: bool,
    consecutive_unfed: i64,
    cared_today: bool,
    fertilizer_available: bool,
    yield_units: i64,
}
#[derive(Clone)]
struct PlantTile {
    pos: Pos,
    crop: String,
    age: i64,
    watered_today: bool,
    water_needed: bool,
    yield_units: i64,
    fertilize_due: bool,
    fert_until_day: i64,
    expired: bool,
    ongoing: bool,
    first_yield_day: i64,
    max_yield_day: i64,
}
#[derive(Clone, Default)]
struct FarmState {
    animals: Vec<AnimalTile>,
    empty_pastures: Vec<Pos>,
    empty_coops: Vec<Pos>,
    plants: Vec<PlantTile>,
    weeds: Vec<Pos>,
    empty_tiles: Vec<Pos>,
    unlocked_count: usize,
}

fn obj_i64(object: &serde_json::Map<String, Value>, key: &str, default: i64) -> i64 {
    object
        .get(key)
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|x| x as i64)))
        .unwrap_or(default)
}
fn obj_bool(object: &serde_json::Map<String, Value>, key: &str) -> bool {
    object.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn parse_farm_state(tiles: &[Vec<Value>], day: usize) -> Result<FarmState, String> {
    let mut state = FarmState::default();
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let tile = &tiles[y][x];
            let pos = (x as i64, y as i64);
            if tile.as_str() == Some("LOCKED") {
                continue;
            }
            state.unlocked_count += 1;
            if tile.is_null() {
                state.empty_tiles.push(pos);
                continue;
            }
            let o = tile
                .as_object()
                .ok_or_else(|| format!("unexpected tile at {pos:?}: {tile}"))?;
            match o.get("kind").and_then(Value::as_str) {
                Some("WEED") => state.weeds.push(pos),
                Some(kind @ ("PASTURE" | "COOP")) => {
                    if let Some(an) = o.get("animal").and_then(Value::as_str) {
                        state.animals.push(AnimalTile {
                            pos,
                            animal: an.into(),
                            fed_today: obj_bool(o, "fed_today"),
                            consecutive_unfed: obj_i64(o, "consecutive_unfed", 0),
                            cared_today: obj_bool(o, "cared_today"),
                            fertilizer_available: obj_bool(o, "fertilizer_available"),
                            yield_units: obj_i64(o, "yield_units", 0),
                        });
                    } else if kind == "COOP" {
                        state.empty_coops.push(pos)
                    } else {
                        state.empty_pastures.push(pos)
                    }
                }
                Some("PLANT") => {
                    let name = o
                        .get("crop")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("missing crop at {pos:?}"))?;
                    let spec = crop(name).ok_or_else(|| format!("unknown crop {name}"))?;
                    let planted = obj_i64(o, "planted_day", day as i64);
                    let age = day as i64 - planted;
                    let watered = obj_bool(o, "watered_today");
                    let fert_until = obj_i64(o, "fertilized_until_day", -1);
                    let unwatered = o
                        .get("consecutive_unwatered")
                        .and_then(Value::as_i64)
                        .ok_or_else(|| format!("missing consecutive_unwatered at {pos:?}"))?;
                    let expired = if spec.ongoing {
                        age >= spec.yield_days[spec.yield_days.len() - 1] + 1
                    } else {
                        age >= spec.max_yield_day + 1
                    };
                    let mut fert_due = false;
                    let mut bonus = false;
                    if spec.ongoing {
                        if let Some(next) = spec.yield_days.iter().copied().find(|v| *v >= age + 1)
                        {
                            let reachable = day + 1 < LAST_DAY;
                            let tomorrow = next == age + 1;
                            fert_due = reachable && tomorrow && fert_until < day as i64 + 1;
                            bonus =
                                reachable && tomorrow && (fert_until >= day as i64 + 1 || fert_due);
                        }
                    } else {
                        bonus =
                            age >= if name == "MELON" { 6 } else { 2 } && age <= spec.max_yield_day;
                    }
                    let water_needed = !watered && day < LAST_DAY && (unwatered >= 1 || bonus);
                    state.plants.push(PlantTile {
                        pos,
                        crop: name.into(),
                        age,
                        watered_today: watered,
                        water_needed,
                        yield_units: obj_i64(o, "yield_units", 0),
                        fertilize_due: fert_due,
                        fert_until_day: fert_until,
                        expired,
                        ongoing: spec.ongoing,
                        first_yield_day: spec.first_yield_day,
                        max_yield_day: spec.max_yield_day,
                    });
                }
                other => return Err(format!("unknown tile kind {other:?} at {pos:?}")),
            }
        }
    }
    Ok(state)
}

fn count_animal_census(
    state: &FarmState,
    shed: &Counts,
    inventories: &[Inventory],
) -> AnimalCensus {
    let field = |s: &str| state.animals.iter().filter(|a| a.animal == s).count() as i64;
    let carried = |s: &str| inventories.iter().map(|v| inv_count(v, s)).sum();
    AnimalCensus {
        field_cows: field("COW"),
        field_sheep: field("SHEEP"),
        field_geese: field("GOOSE"),
        shed_cows: count(shed, "COW"),
        shed_sheep: count(shed, "SHEEP"),
        shed_geese: count(shed, "GOOSE"),
        carried_cows: carried("COW"),
        carried_sheep: carried("SHEEP"),
        carried_geese: carried("GOOSE"),
    }
}
fn kept_feedable_count(c: &AnimalCensus, caps: &Counts) -> i64 {
    [
        (&c.field_cows, &c.shed_cows, "COW"),
        (&c.field_sheep, &c.shed_sheep, "SHEEP"),
        (&c.field_geese, &c.shed_geese, "GOOSE"),
    ]
    .iter()
    .map(|(f, s, n)| (**f + **s).min(count_or(caps, n, 999)))
    .sum()
}
fn count_or(m: &Counts, k: &str, d: i64) -> i64 {
    *m.get(k).unwrap_or(&d)
}
fn wheat_feed_thresholds(c: &AnimalCensus, caps: &Counts, day: usize) -> (i64, i64) {
    if day >= LIQUIDATION_DAY {
        return (0, 0);
    }
    let n = kept_feedable_count(c, caps);
    (
        FEED_WHEAT_RESERVE.max(n * FEED_WHEAT_RESERVE_MIN_RATIO),
        (FEED_WHEAT_RESERVE * FEED_WHEAT_SELL_RATIO).max(n * FEED_WHEAT_SELL_RATIO),
    )
}
fn planted_count(f: &FarmState, name: &str) -> i64 {
    f.plants.iter().filter(|p| p.crop == name).count() as i64
}
fn committed_count(f: &FarmState, seeds: &Counts, name: &str) -> i64 {
    planted_count(f, name) + count(seeds, name)
}
fn count_fertilize_due(f: &FarmState) -> i64 {
    f.plants.iter().filter(|p| p.fertilize_due).count() as i64
}
fn count_fertilize_due_tomorrow(f: &FarmState, day: usize) -> i64 {
    if day + 2 >= LAST_DAY {
        return 0;
    }
    f.plants
        .iter()
        .filter(|p| {
            p.ongoing
                && !p.expired
                && crop(&p.crop)
                    .unwrap()
                    .yield_days
                    .iter()
                    .copied()
                    .find(|v| *v >= p.age + 2)
                    == Some(p.age + 2)
                && p.fert_until_day < day as i64 + 2
        })
        .count() as i64
}
fn count_maturing_tomorrow(f: &FarmState, name: &str) -> i64 {
    f.plants
        .iter()
        .filter(|p| p.crop == name && p.age + 1 >= p.max_yield_day)
        .count() as i64
}
fn coop_capacity(f: &FarmState) -> i64 {
    f.empty_coops.len() as i64 + f.animals.iter().filter(|a| a.animal == "GOOSE").count() as i64
}
fn live_reserved_structures(
    f: &FarmState,
    quads: &[String],
    grazer: i64,
    goose: i64,
    day: usize,
) -> HashSet<Pos> {
    let gn = if day <= PASTURE_RESERVATION_LAST_DAY {
        grazer.max(if quads.len() == 1 {
            4
        } else {
            MAX_TOTAL_HERD.min(quads.len() as i64 * 6)
        })
    } else {
        0
    };
    let en = if day < LIQUIDATION_DAY && quads.len() >= 2 {
        (goose - coop_capacity(f)).max(0)
    } else {
        0
    };
    let mut out = HashSet::new();
    for p in PASTURE_CLUSTER {
        if out.len() as i64 >= gn + en {
            break;
        }
        if quads.iter().any(|q| q == quad_of(p)) {
            out.insert(p);
        }
    }
    out
}

#[derive(Clone)]
struct Task {
    urgency: i64,
    pos: Pos,
    actions: Vec<Vec<Value>>,
    need: Vec<&'static str>,
    produces: Option<(String, i64)>,
}
impl Task {
    fn new(urgency: i64, pos: Pos, actions: Vec<Vec<Value>>) -> Self {
        Self {
            urgency,
            pos,
            actions,
            need: vec![],
            produces: None,
        }
    }
}
fn a1(op: &str) -> Vec<Value> {
    vec![Value::String(op.into())]
}
fn a2(op: &str, arg: &str) -> Vec<Value> {
    vec![Value::String(op.into()), Value::String(arg.into())]
}
fn a3(op: &str, arg: &str, n: i64) -> Vec<Value> {
    vec![
        Value::String(op.into()),
        Value::String(arg.into()),
        Value::from(n),
    ]
}
fn plant_task(urgency: i64, pos: Pos, name: &str) -> Task {
    Task::new(urgency, pos, vec![a2("PLANT", name), a1("WATER")])
}
fn harvest_actions(watered: bool, day: usize) -> Vec<Vec<Value>> {
    if !watered && day < LAST_DAY {
        vec![a1("WATER"), a1("HARVEST")]
    } else {
        vec![a1("HARVEST")]
    }
}
fn ready_to_harvest(p: &PlantTile, day: usize) -> bool {
    p.yield_units > 0 && (p.ongoing || p.age >= p.max_yield_day || day >= LIQUIDATION_DAY)
}
fn ready_premium(f: &FarmState, day: usize) -> Vec<PlantTile> {
    f.plants
        .iter()
        .filter(|p| {
            !p.expired && PREMIUM_DROP_CROPS.contains(&p.crop.as_str()) && ready_to_harvest(p, day)
        })
        .cloned()
        .collect()
}
fn select_target_crops(day: usize, seeds: &Counts) -> Vec<&'static str> {
    if day > WHEAT_LAST_PLANT_DAY {
        return vec![];
    }
    if day >= 21 {
        return if count(seeds, "WHEAT") > 0 {
            vec!["WHEAT"]
        } else {
            vec![]
        };
    }
    let premium = if day <= 12 {
        ["MELON", "STRAWBERRY"]
    } else {
        ["STRAWBERRY", "MELON"]
    };
    premium
        .into_iter()
        .chain(["WHEAT"])
        .filter(|name| {
            count(seeds, name) > 0
                && day
                    <= match *name {
                        "MELON" => MELON_LAST_PLANT_DAY,
                        "STRAWBERRY" => STRAWBERRY_LAST_PLANT_DAY,
                        _ => WHEAT_LAST_PLANT_DAY,
                    }
        })
        .collect()
}
fn compute_needed_pastures(f: &FarmState, c: &AnimalCensus) -> Vec<Pos> {
    let planted: HashSet<_> = f.plants.iter().map(|p| p.pos).collect();
    let occupied = c.field_cows + c.field_sheep;
    let mut out = vec![];
    for p in PASTURE_CLUSTER {
        if out.len() as i64 + occupied + f.empty_pastures.len() as i64 >= c.total() {
            break;
        }
        if f.empty_tiles.contains(&p) || f.weeds.contains(&p) || planted.contains(&p) {
            out.push(p)
        }
    }
    out
}
fn compute_needed_coops(f: &FarmState, c: &AnimalCensus, exclude: &HashSet<Pos>) -> Vec<Pos> {
    let planted: HashSet<_> = f.plants.iter().map(|p| p.pos).collect();
    let mut out = vec![];
    for p in PASTURE_CLUSTER {
        if exclude.contains(&p) {
            continue;
        }
        if out.len() as i64 + c.field_geese + f.empty_coops.len() as i64 >= c.total_geese() {
            break;
        }
        if f.empty_tiles.contains(&p) || f.weeds.contains(&p) || planted.contains(&p) {
            out.push(p)
        }
    }
    out
}
fn structure_actions(pos: Pos, f: &FarmState, build: &str) -> Option<Vec<&'static str>> {
    if f.weeds.contains(&pos) {
        return Some(vec![
            "DIG",
            if build == "BUILD_PASTURE" {
                "BUILD_PASTURE"
            } else {
                "BUILD_COOP"
            },
        ]);
    }
    if let Some(p) = f.plants.iter().find(|p| p.pos == pos) {
        if p.yield_units > 0 || (!p.ongoing && p.age >= p.first_yield_day - 1) {
            return None;
        }
        return Some(vec![
            "DIG",
            if build == "BUILD_PASTURE" {
                "BUILD_PASTURE"
            } else {
                "BUILD_COOP"
            },
        ]);
    }
    if f.empty_tiles.contains(&pos) {
        Some(vec![if build == "BUILD_PASTURE" {
            "BUILD_PASTURE"
        } else {
            "BUILD_COOP"
        }])
    } else {
        None
    }
}
fn kept_animal_positions(f: &FarmState, caps: &Counts) -> HashSet<Pos> {
    let mut kept = HashSet::new();
    for species in ANIMAL_SPECIES {
        let cap = count_or(caps, species, 999);
        if cap <= 0 {
            continue;
        }
        let mut rows: Vec<_> = f.animals.iter().filter(|a| a.animal == species).collect();
        rows.sort_by_key(|a| (a.fed_today, manhattan(a.pos, SHED_CENTER)));
        for a in rows.into_iter().take(cap as usize) {
            kept.insert(a.pos);
        }
    }
    kept
}
#[derive(Clone, Default)]
struct DispatchHints {
    reserved_grazer_slots: i64,
    reserved_geese_slots: i64,
    crop_limits: Counts,
}

fn action_repr(actions: &[Vec<Value>]) -> String {
    let mut s = String::from("(");
    for (i, row) in actions.iter().enumerate() {
        if i > 0 {
            s.push_str(", ")
        }
        s.push('[');
        for (j, v) in row.iter().enumerate() {
            if j > 0 {
                s.push_str(", ")
            }
            if let Some(x) = v.as_str() {
                s.push('\'');
                s.push_str(x);
                s.push('\'')
            } else {
                s.push_str(&v.to_string())
            }
        }
        s.push(']')
    }
    if actions.len() == 1 {
        s.push(',')
    }
    s.push(')');
    s
}
fn build_task_catalog(
    f: &FarmState,
    shed: &Counts,
    seeds: &Counts,
    needed_pastures: &[Pos],
    needed_coops: &[Pos],
    units: &[(Pos, Inventory)],
    day: usize,
    hour: usize,
    hints: &DispatchHints,
    quads: &[String],
    caps: &Counts,
) -> Vec<Task> {
    let mut tasks = Vec::new();
    let kept = kept_animal_positions(f, caps);
    let carried = |sp: &str| units.iter().map(|(_, v)| inv_count(v, sp)).sum::<i64>();
    let wheat_available = count(shed, "WHEAT") + carried("WHEAT");
    let fert_available = count(shed, "FERTILIZER") + carried("FERTILIZER");
    let reserved = live_reserved_structures(
        f,
        quads,
        hints.reserved_grazer_slots,
        hints.reserved_geese_slots,
        day,
    );
    let needed: HashSet<_> = needed_pastures
        .iter()
        .chain(needed_coops)
        .copied()
        .collect();
    for a in &f.animals {
        let product = animal(&a.animal).unwrap().product;
        if !kept.contains(&a.pos) {
            if a.yield_units > 0 {
                let mut t = Task::new(URG_HARVEST, a.pos, vec![a1("HARVEST")]);
                t.produces = Some((product.into(), a.yield_units));
                tasks.push(t)
            }
            if a.fertilizer_available {
                let mut t = Task::new(
                    URG_COLLECT_FERTILIZER,
                    a.pos,
                    vec![a1("COLLECT_FERTILIZER")],
                );
                t.produces = Some(("FERTILIZER".into(), 1));
                tasks.push(t)
            }
            continue;
        }
        if !a.fed_today && wheat_available > 0 {
            let mut t = Task::new(
                if a.consecutive_unfed >= 1 {
                    URG_FEED_CRITICAL
                } else {
                    URG_FEED
                },
                a.pos,
                vec![a1("FEED")],
            );
            t.need = vec!["WHEAT"];
            tasks.push(t)
        }
        if !a.cared_today {
            tasks.push(Task::new(URG_CARE, a.pos, vec![a1("CARE")]))
        }
        if a.fertilizer_available {
            let mut t = Task::new(
                URG_COLLECT_FERTILIZER,
                a.pos,
                vec![a1("COLLECT_FERTILIZER")],
            );
            t.produces = Some(("FERTILIZER".into(), 1));
            tasks.push(t)
        }
        if a.yield_units > 0 {
            let mut t = Task::new(URG_HARVEST, a.pos, vec![a1("HARVEST")]);
            t.produces = Some((product.into(), a.yield_units));
            tasks.push(t)
        }
    }
    let field = |sp: &str| f.animals.iter().filter(|a| a.animal == sp).count() as i64;
    let waiting = |sp: &str| {
        if field(sp) < count_or(caps, sp, 999) {
            count(shed, sp) + carried(sp)
        } else {
            0
        }
    };
    let pw = waiting("COW") + waiting("SHEEP");
    let gw = waiting("GOOSE");
    for (spots, n, species) in [
        (&f.empty_pastures, pw, PASTURE_SPECIES.as_slice()),
        (&f.empty_coops, gw, ["GOOSE"].as_slice()),
    ] {
        if n > 0 {
            let mut spots = spots.clone();
            spots.sort_by_key(|p| (manhattan(*p, SHED_CENTER), p.1, p.0));
            for p in spots.into_iter().take(n as usize) {
                let mut t = Task::new(URG_PLACE, p, vec![a1("PLACE")]);
                t.need = species.to_vec();
                tasks.push(t)
            }
        }
    }
    let target_crops = select_target_crops(day, seeds);
    let mut freed_assignment = HashMap::<Pos, &'static str>::new();
    if hour < TURNS_PER_DAY - 1 {
        let usable: Vec<_> = f
            .empty_tiles
            .iter()
            .copied()
            .filter(|p| !BLOCKED_CLUSTER.contains(p))
            .collect();
        let future: HashSet<_> = reserved.difference(&needed).copied().collect();
        let mut open: Vec<_> = usable
            .iter()
            .copied()
            .filter(|p| !future.contains(p) && !needed.contains(p))
            .collect();
        let mut res: Vec<_> = usable
            .iter()
            .copied()
            .filter(|p| future.contains(p))
            .collect();
        let mut freed = HashSet::new();
        for p in &f.plants {
            if p.ongoing
                || p.expired
                || PREMIUM_DROP_CROPS.contains(&p.crop.as_str())
                || !ready_to_harvest(p, day)
                || needed.contains(&p.pos)
            {
                continue;
            }
            freed.insert(p.pos);
            if future.contains(&p.pos) {
                res.push(p.pos)
            } else {
                open.push(p.pos)
            }
        }
        open.sort_by_key(|p| (manhattan(*p, SHED_CENTER), p.1, p.0));
        res.sort_by_key(|p| (manhattan(*p, SHED_CENTER), p.1, p.0));
        let mut queue = Vec::<(&'static str, i64, i64)>::new();
        for name in ["MELON", "STRAWBERRY"] {
            if target_crops.contains(&name) {
                queue.push((name, URG_PLANT_PRIORITY, count(seeds, name)))
            }
        }
        for name in ["CARROT", "TOMATO"] {
            queue.push((
                name,
                URG_PLANT,
                count(seeds, name).min(count(&hints.crop_limits, name)),
            ))
        }
        for name in &target_crops {
            if !PREMIUM_DROP_CROPS.contains(name) {
                queue.push((*name, URG_PLANT, count(seeds, name)))
            }
        }
        for (name, urg, limit) in queue {
            if open.is_empty() && res.is_empty() {
                break;
            }
            let mut remaining = limit;
            let pools = if name == "WHEAT" { 2 } else { 1 };
            for which in 0..pools {
                let pool = if which == 0 { &mut open } else { &mut res };
                if remaining <= 0 || pool.is_empty() {
                    continue;
                }
                let n = (pool.len() as i64).min(remaining) as usize;
                for p in pool.drain(..n) {
                    if freed.contains(&p) {
                        freed_assignment.insert(p, name);
                    } else {
                        tasks.push(plant_task(urg, p, name))
                    }
                }
                remaining -= n as i64
            }
        }
    }
    for p in &f.plants {
        if p.expired {
            tasks.push(Task::new(URG_DIG_EXPIRED, p.pos, vec![a1("DIG")]))
        } else if ready_to_harvest(p, day) {
            // Premium crops are consumed exclusively by `plan_premium`.  Preserve
            // Python's outer `elif`: a ready premium bush must not fall through to
            // the ordinary WATER/FERTILIZE branches below.
            if !PREMIUM_DROP_CROPS.contains(&p.crop.as_str()) {
                let mut actions = harvest_actions(p.watered_today, day);
                if let Some(name) = freed_assignment.get(&p.pos) {
                    actions.push(a2("PLANT", name));
                    actions.push(a1("WATER"))
                }
                let mut t = Task::new(URG_HARVEST, p.pos, actions);
                if p.ongoing && p.fertilize_due && fert_available > 0 {
                    t.actions.insert(0, a1("FERTILIZE"));
                    t.need = vec!["FERTILIZER"]
                }
                t.produces = Some((p.crop.clone(), p.yield_units));
                tasks.push(t)
            }
        } else if p.water_needed && p.fertilize_due && fert_available > 0 {
            let mut t = Task::new(URG_FERTILIZE, p.pos, vec![a1("WATER"), a1("FERTILIZE")]);
            t.need = vec!["FERTILIZER"];
            tasks.push(t)
        } else if p.water_needed {
            tasks.push(Task::new(URG_WATER, p.pos, vec![a1("WATER")]))
        } else if p.fertilize_due && fert_available > 0 {
            let mut t = Task::new(URG_FERTILIZE, p.pos, vec![a1("FERTILIZE")]);
            t.need = vec!["FERTILIZER"];
            tasks.push(t)
        }
    }
    if day <= WHEAT_LAST_PLANT_DAY {
        for p in &f.weeds {
            if !BLOCKED_CLUSTER.contains(p) {
                tasks.push(Task::new(URG_DIG_WEED, *p, vec![a1("DIG")]))
            }
        }
    }
    let pasture_left = (pw - f.empty_pastures.len() as i64).max(0);
    let mut n = 0;
    for p in needed_pastures {
        if n >= pasture_left {
            break;
        }
        if let Some(acts) = structure_actions(*p, f, "BUILD_PASTURE") {
            let mut rows: Vec<_> = acts.into_iter().map(a1).collect();
            rows.push(a1("PLACE"));
            let mut t = Task::new(URG_PLACE, *p, rows);
            t.need = PASTURE_SPECIES.to_vec();
            tasks.push(t);
            n += 1
        }
    }
    let coop_left = (gw - f.empty_coops.len() as i64).max(0);
    let mut n = 0;
    for p in needed_coops {
        if n >= coop_left {
            break;
        }
        if let Some(acts) = structure_actions(*p, f, "BUILD_COOP") {
            let mut rows: Vec<_> = acts.into_iter().map(a1).collect();
            rows.push(a1("PLACE"));
            let mut t = Task::new(URG_PLACE, *p, rows);
            t.need = vec!["GOOSE"];
            tasks.push(t);
            n += 1
        }
    }
    tasks.sort_by(|a, b| {
        b.urgency
            .cmp(&a.urgency)
            .then_with(|| manhattan(a.pos, SHED_CENTER).cmp(&manhattan(b.pos, SHED_CENTER)))
            .then_with(|| a.pos.1.cmp(&b.pos.1))
            .then_with(|| a.pos.0.cmp(&b.pos.0))
            .then_with(|| action_repr(&a.actions).cmp(&action_repr(&b.actions)))
    });
    tasks
}

#[derive(Clone)]
struct Stop {
    pos: Pos,
    actions: Vec<Vec<Value>>,
    #[allow(dead_code)]
    produces: Option<(String, i64)>,
}
#[derive(Clone)]
struct Route {
    start: Pos,
    stops: Vec<Stop>,
    cost: i64,
    carried: Counts,
    pickup_of: IndexMap<String, usize>,
    insertable_upto: Option<usize>,
    insertable_from: usize,
}
fn init_routes(units: &[(Pos, Inventory)]) -> Vec<Route> {
    units
        .iter()
        .map(|(p, inv)| Route {
            start: *p,
            stops: vec![],
            cost: 0,
            carried: inv
                .iter()
                .filter(|(_, n)| **n > 0)
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            pickup_of: IndexMap::new(),
            insertable_upto: None,
            insertable_from: 0,
        })
        .collect()
}
fn route_cost(r: &Route) -> i64 {
    let mut p = r.start;
    let mut n = 0;
    for s in &r.stops {
        n += manhattan(p, s.pos) + s.actions.len() as i64;
        p = s.pos
    }
    n
}
fn best_shed(prev: Pos, next: Option<Pos>) -> (Pos, i64) {
    let base = next.map(|p| manhattan(prev, p)).unwrap_or(0);
    let mut best = SHED_TILES[0];
    let mut bd = i64::MAX;
    for p in SHED_TILES {
        let d = manhattan(prev, p) + next.map(|q| manhattan(p, q)).unwrap_or(0) - base;
        if d < bd {
            bd = d;
            best = p
        }
    }
    (best, bd)
}
fn drop_cost(p: Pos) -> i64 {
    best_shed(p, None).1 + 1
}
fn rebuild_pickups(r: &Route) -> IndexMap<String, usize> {
    let mut out = IndexMap::new();
    for (i, s) in r.stops.iter().enumerate() {
        for a in &s.actions {
            if a.first().and_then(Value::as_str) == Some("PICKUP") {
                if let Some(item) = a.get(1).and_then(Value::as_str) {
                    if !out.contains_key(item) {
                        out.insert(item.into(), i);
                    }
                }
            }
        }
    }
    out
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Plain,
    Carried,
    Bump,
    Extend,
    New,
}
#[derive(Clone)]
struct Candidate {
    cost: i64,
    insert_at: usize,
    item: Option<&'static str>,
    mode: Mode,
    pickup_idx: isize,
    pickup_pos: Option<Pos>,
}

fn eval_route(route: &Route, task: &Task, shed: &Counts) -> Option<Candidate> {
    let n = route.stops.len();
    let limit = route.insertable_upto.unwrap_or(n);
    let floor = route.insertable_from;
    let mut pref_cost = vec![i64::MAX; n + 1];
    let mut pref_k = vec![-1isize; n + 1];
    let mut best = i64::MAX;
    let mut best_k = -1;
    for k in 0..n {
        if k >= floor {
            let prev = if k == 0 {
                route.start
            } else {
                route.stops[k - 1].pos
            };
            let d = best_shed(prev, Some(route.stops[k].pos)).1 + 1;
            if d < best {
                best = d;
                best_k = k as isize
            }
        }
        pref_cost[k + 1] = best;
        pref_k[k + 1] = best_k
    }
    let earliest = route.pickup_of.values().min().copied();
    let items: Vec<Option<&'static str>> = if task.need.is_empty() {
        vec![None]
    } else {
        task.need.iter().copied().map(Some).collect()
    };
    let mut result = None;
    for i in floor..=limit {
        let prev = if i == 0 {
            route.start
        } else {
            route.stops[i - 1].pos
        };
        let next = if i < n {
            Some(route.stops[i].pos)
        } else {
            None
        };
        let back = next
            .map(|q| manhattan(task.pos, q) - manhattan(prev, q))
            .unwrap_or(0);
        let d_task = manhattan(prev, task.pos) + back + task.actions.len() as i64;
        for item in &items {
            let cand = if let Some(item) = item {
                if count(&route.carried, item) > 0 {
                    Candidate {
                        cost: d_task,
                        insert_at: i,
                        item: Some(item),
                        mode: Mode::Carried,
                        pickup_idx: -1,
                        pickup_pos: None,
                    }
                } else if count(shed, item) <= 0 {
                    continue;
                } else if route.pickup_of.get(*item).is_some_and(|j| *j < i) {
                    Candidate {
                        cost: d_task,
                        insert_at: i,
                        item: Some(item),
                        mode: Mode::Bump,
                        pickup_idx: -1,
                        pickup_pos: None,
                    }
                } else if earliest.is_some_and(|j| j < i) {
                    Candidate {
                        cost: d_task + 1,
                        insert_at: i,
                        item: Some(item),
                        mode: Mode::Extend,
                        pickup_idx: -1,
                        pickup_pos: None,
                    }
                } else {
                    let mut d_new = pref_cost[i];
                    let mut k_new = pref_k[i];
                    let (p_det, d) = best_shed(prev, Some(task.pos));
                    if d + 1 <= d_new {
                        d_new = d + 1;
                        k_new = i as isize
                    }
                    if d_new == i64::MAX {
                        continue;
                    }
                    Candidate {
                        cost: d_task + d_new,
                        insert_at: i,
                        item: Some(item),
                        mode: Mode::New,
                        pickup_idx: k_new,
                        pickup_pos: if k_new == i as isize {
                            Some(p_det)
                        } else {
                            None
                        },
                    }
                }
            } else {
                Candidate {
                    cost: d_task,
                    insert_at: i,
                    item: None,
                    mode: Mode::Plain,
                    pickup_idx: -1,
                    pickup_pos: None,
                }
            };
            if result
                .as_ref()
                .is_none_or(|old: &Candidate| cand.cost < old.cost)
            {
                result = Some(cand)
            }
        }
    }
    result
}

fn commit(route: &mut Route, task: &Task, cand: &Candidate, shed: &mut Counts) {
    let before = route.stops.len();
    let mut at = cand.insert_at;
    let item = cand.item;
    match cand.mode {
        Mode::New => {
            let name = item.unwrap();
            let k = cand.pickup_idx as usize;
            let pos = cand.pickup_pos.unwrap_or_else(|| {
                let prev = if k == 0 {
                    route.start
                } else {
                    route.stops[k - 1].pos
                };
                best_shed(prev, Some(route.stops[k].pos)).0
            });
            route.stops.insert(
                k,
                Stop {
                    pos,
                    actions: vec![a3("PICKUP", name, 1)],
                    produces: None,
                },
            );
            at += 1;
            *shed.entry(name.into()).or_default() -= 1
        }
        Mode::Bump => {
            let name = item.unwrap();
            let j = route.pickup_of[name];
            for line in &mut route.stops[j].actions {
                if line.first().and_then(Value::as_str) == Some("PICKUP")
                    && line.get(1).and_then(Value::as_str) == Some(name)
                {
                    let n = line[2].as_i64().unwrap();
                    line[2] = Value::from(n + 1);
                    break;
                }
            }
            *shed.entry(name.into()).or_default() -= 1
        }
        Mode::Extend => {
            let name = item.unwrap();
            let j = *route.pickup_of.values().min().unwrap();
            route.stops[j].actions.push(a3("PICKUP", name, 1));
            *shed.entry(name.into()).or_default() -= 1
        }
        Mode::Carried => {
            let name = item.unwrap();
            *route.carried.entry(name.into()).or_default() -= 1
        }
        Mode::Plain => {}
    }
    let actions = task
        .actions
        .iter()
        .map(|a| {
            if a.first().and_then(Value::as_str) == Some("PLACE") && a.len() == 1 && item.is_some()
            {
                a2("PLACE", item.unwrap())
            } else {
                a.clone()
            }
        })
        .collect();
    route.stops.insert(
        at,
        Stop {
            pos: task.pos,
            actions,
            produces: task.produces.clone(),
        },
    );
    if let Some(u) = &mut route.insertable_upto {
        *u += route.stops.len() - before
    }
    route.cost = route_cost(route);
    route.pickup_of = rebuild_pickups(route)
}
fn final_drop_reserve(route: &Route, task: &Task, c: &Candidate, enforce: bool) -> i64 {
    if !enforce {
        return 0;
    }
    let tail = if c.insert_at >= route.stops.len() {
        task.pos
    } else {
        route.stops.last().unwrap().pos
    };
    drop_cost(tail)
}
fn plan_routes(
    routes: &mut [Route],
    catalog: &[Task],
    shed: &mut Counts,
    budgets: &[i64],
    reserve: bool,
) {
    let mut prev = i64::MAX;
    for task in catalog {
        if task.urgency < prev {
            for r in routes.iter_mut() {
                r.insertable_from = r.stops.len()
            }
        }
        prev = task.urgency;
        let mut best: Option<(i64, usize, Candidate)> = None;
        for (i, r) in routes.iter().enumerate() {
            if let Some(c) = eval_route(r, task, shed) {
                let extra = final_drop_reserve(r, task, &c, reserve);
                if r.cost + c.cost + extra > budgets[i] {
                    continue;
                }
                if best.as_ref().is_none_or(|b| c.cost < b.0) {
                    best = Some((c.cost, i, c))
                }
            }
        }
        if let Some((_, i, c)) = best {
            commit(&mut routes[i], task, &c, shed)
        }
    }
}

fn append_drops(routes: &mut [Route], budgets: &[i64]) {
    for (i, r) in routes.iter_mut().enumerate() {
        if r.stops.is_empty()
            || r.stops
                .last()
                .unwrap()
                .actions
                .last()
                .unwrap()
                .first()
                .and_then(Value::as_str)
                == Some("DROP")
        {
            continue;
        }
        let produced = r.stops.iter().flat_map(|s| &s.actions).any(|a| {
            matches!(
                a.first().and_then(Value::as_str),
                Some("HARVEST" | "COLLECT_FERTILIZER")
            )
        });
        let carried = SELLABLE_ITEMS.iter().any(|n| count(&r.carried, n) > 0);
        if !produced && !carried {
            continue;
        }
        let last = r.stops.last().unwrap().pos;
        let delta = drop_cost(last);
        if r.cost + delta > budgets[i] {
            continue;
        }
        if SHED_TILES.contains(&last) {
            r.stops.last_mut().unwrap().actions.push(a1("DROP"))
        } else {
            r.stops.push(Stop {
                pos: best_shed(last, None).0,
                actions: vec![a1("DROP")],
                produces: None,
            })
        }
        r.cost = route_cost(r)
    }
}
fn pair_premium(ready: &[PlantTile]) -> Vec<Vec<PlantTile>> {
    let mut pool = ready.to_vec();
    let mut blocks = vec![];
    while let Some(a) = pool.pop() {
        if pool.is_empty() {
            blocks.push(vec![a]);
            break;
        }
        let mut j = 0;
        let mut best = i64::MAX;
        for (i, b) in pool.iter().enumerate() {
            let d = manhattan(a.pos, b.pos);
            if d < best {
                best = d;
                j = i
            }
        }
        blocks.push(vec![a, pool.remove(j)])
    }
    blocks
}
fn block_cost(tail: Pos, block: &[PlantTile], day: usize) -> (i64, Vec<PlantTile>) {
    let mut orders = vec![block.to_vec()];
    if block.len() == 2 {
        let mut rev = block.to_vec();
        rev.reverse();
        orders.push(rev)
    }
    let mut bc = i64::MAX;
    let mut bo = block.to_vec();
    for order in orders {
        let mut pos = tail;
        let mut n = 0;
        for p in &order {
            n += manhattan(pos, p.pos) + harvest_actions(p.watered_today, day).len() as i64;
            pos = p.pos
        }
        n += drop_cost(pos);
        if n < bc {
            bc = n;
            bo = order
        }
    }
    (bc, bo)
}
fn plan_premium(routes: &mut [Route], ready: &[PlantTile], day: usize, budgets: &[i64]) {
    for block in pair_premium(ready) {
        let mut best: Option<(i64, usize, Vec<PlantTile>)> = None;
        for (i, r) in routes.iter().enumerate() {
            let tail = r.stops.last().map(|s| s.pos).unwrap_or(r.start);
            let (cost, order) = block_cost(tail, &block, day);
            if r.cost + cost > budgets[i] {
                continue;
            }
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, i, order))
            }
        }
        let Some((_, i, order)) = best else { continue };
        let r = &mut routes[i];
        if r.insertable_upto.is_none() {
            r.insertable_upto = Some(r.stops.len())
        }
        for p in &order {
            r.stops.push(Stop {
                pos: p.pos,
                actions: harvest_actions(p.watered_today, day),
                produces: Some((p.crop.clone(), p.yield_units)),
            })
        }
        r.stops.push(Stop {
            pos: best_shed(order.last().unwrap().pos, None).0,
            actions: vec![a1("DROP")],
            produces: None,
        });
        r.cost = route_cost(r)
    }
}
fn solve_day(
    catalog: &[Task],
    premium: &[PlantTile],
    units: &[(Pos, Inventory)],
    shed: &Counts,
    budgets: &[i64],
    day: usize,
) -> Vec<Route> {
    let mut routes = init_routes(units);
    let mut stock = shed.clone();
    let critical: Vec<_> = catalog
        .iter()
        .filter(|t| t.urgency > URG_HARVEST_PREMIUM)
        .cloned()
        .collect();
    let rest: Vec<_> = catalog
        .iter()
        .filter(|t| t.urgency <= URG_HARVEST_PREMIUM)
        .cloned()
        .collect();
    plan_routes(&mut routes, &critical, &mut stock, budgets, false);
    plan_premium(&mut routes, premium, day, budgets);
    plan_routes(&mut routes, &rest, &mut stock, budgets, day == LAST_DAY);
    append_drops(&mut routes, budgets);
    routes
}
fn required_hand_count(
    catalog: &[Task],
    premium: &[PlantTile],
    base: &[(Pos, Inventory)],
    shed: &Counts,
    max_hands: usize,
    day: usize,
) -> usize {
    if catalog.is_empty() && premium.is_empty() {
        return 0;
    }
    let current = base.len();
    let max_team = max_hands + 1;
    if current >= max_team {
        return 0;
    }
    let buffer = if day == LAST_DAY { 2 } else { 0 };
    let budget = (TURNS_PER_DAY as i64 - 1 - buffer).max(1);
    let target = catalog.len() + premium.len();
    let mut best_extra = 0;
    for team in current..=max_team {
        let mut units = base.to_vec();
        for i in current..team {
            units.push((SHED_TILES[i % SHED_TILES.len()], IndexMap::new()))
        }
        let routes = solve_day(
            catalog,
            premium,
            &units,
            shed,
            &vec![budget; units.len()],
            day,
        );
        let inserted = routes
            .iter()
            .flat_map(|r| &r.stops)
            .filter(|s| {
                s.actions
                    .first()
                    .and_then(|a| a.first())
                    .and_then(Value::as_str)
                    .is_some_and(|op| op != "PICKUP" && op != "DROP")
            })
            .count();
        if inserted == target {
            return team - current;
        }
        let active = routes
            .iter()
            .filter(|r| {
                r.stops.iter().any(|s| {
                    s.actions
                        .first()
                        .and_then(|a| a.first())
                        .and_then(Value::as_str)
                        .is_some_and(|op| op != "PICKUP" && op != "DROP")
                })
            })
            .count();
        if active == team {
            best_extra = team - current
        }
    }
    best_extra
}
#[derive(Clone)]
struct DayPlan {
    day: usize,
    routes: IndexMap<usize, Vec<Vec<Value>>>,
}
fn materialize(route: &Route) -> Vec<Vec<Value>> {
    let mut queue = vec![];
    let mut pos = route.start;
    for stop in &route.stops {
        while pos != stop.pos {
            let step = bfs_step(pos, stop.pos).unwrap();
            queue.push(a1(step));
            match step {
                "EAST" => pos.0 += 1,
                "WEST" => pos.0 -= 1,
                "SOUTH" => pos.1 += 1,
                "NORTH" => pos.1 -= 1,
                _ => {}
            }
        }
        queue.extend(stop.actions.clone())
    }
    queue
}
#[allow(clippy::too_many_arguments)]
fn build_day_plan(
    units: &[(Pos, Inventory)],
    f: &FarmState,
    shed: &Counts,
    seeds: &Counts,
    day: usize,
    hour: usize,
    needed_pastures: &[Pos],
    needed_coops: &[Pos],
    hints: &DispatchHints,
    quads: &[String],
    caps: &Counts,
    pending_hires: usize,
) -> DayPlan {
    let effective = hour.max(1);
    let buffer = if day == LAST_DAY { 2 } else { 0 };
    let budget = (TURNS_PER_DAY as i64 - effective as i64 - buffer).max(0);
    let mut full = units.to_vec();
    let mut budgets = vec![budget; units.len()];
    budgets.extend(vec![budget - 1; pending_hires]);
    for i in 0..pending_hires {
        full.push((
            SHED_TILES[(units.len() + i) % SHED_TILES.len()],
            IndexMap::new(),
        ))
    }
    let routes = if budget > 0 && !full.is_empty() {
        let catalog = build_task_catalog(
            f,
            shed,
            seeds,
            needed_pastures,
            needed_coops,
            &full,
            day,
            effective,
            hints,
            quads,
            caps,
        );
        solve_day(&catalog, &ready_premium(f, day), &full, shed, &budgets, day)
    } else {
        init_routes(&full)
    };
    let materialized: IndexMap<_, _> = routes
        .iter()
        .enumerate()
        .map(|(i, r)| (i, materialize(r)))
        .collect();
    for (i, q) in &materialized {
        assert!(q.len() as i64 <= budgets[*i], "EcoBot route exceeds budget")
    }
    DayPlan {
        day,
        routes: materialized,
    }
}
fn next_action(unit: usize, day: usize, plan: &mut DayPlan) -> Result<Vec<Value>, String> {
    if plan.day != day {
        return Err(format!("stale EcoBot plan {} vs {day}", plan.day));
    }
    let queue = plan
        .routes
        .get_mut(&unit)
        .ok_or_else(|| format!("EcoBot unit {unit} missing route"))?;
    Ok(if queue.is_empty() {
        a1("PASS")
    } else {
        queue.remove(0)
    })
}

#[derive(Clone)]
struct DriftState {
    prev_inv: Counts,
    prev_day: i64,
    observed: IndexMap<String, f64>,
}
impl Default for DriftState {
    fn default() -> Self {
        Self {
            prev_inv: IndexMap::new(),
            prev_day: -1,
            observed: IndexMap::new(),
        }
    }
}
#[derive(Clone)]
struct CullState {
    last_day: i64,
    negative_days: Counts,
    downsized: HashSet<String>,
    retained_caps: Counts,
}
impl Default for CullState {
    fn default() -> Self {
        Self {
            last_day: -1,
            negative_days: IndexMap::new(),
            downsized: HashSet::new(),
            retained_caps: IndexMap::new(),
        }
    }
}
#[derive(Clone, Default)]
struct EvalState {
    drift: DriftState,
    cull: CullState,
}
#[derive(Clone)]
struct CandidateScore {
    kind: &'static str,
    #[allow(dead_code)]
    item: &'static str,
    #[allow(dead_code)]
    cost: f64,
    daily_profit: f64,
    npv: f64,
}
#[derive(Clone)]
struct TurnDecision {
    market_orders: Vec<Vec<Value>>,
    hints: DispatchHints,
    retained_caps: Counts,
    needed_pastures: Vec<Pos>,
    needed_coops: Vec<Pos>,
}

fn compute_daily_demand(shops: &[String]) -> IndexMap<String, f64> {
    let mut d: IndexMap<_, _> = SELLABLE_ITEMS
        .iter()
        .filter(|x| **x != "FERTILIZER")
        .map(|x| ((*x).into(), TOWN_CENTER_FLAT))
        .collect();
    for shop in shops {
        for (item, q) in shop_demands(shop) {
            *d.entry((*item).into()).or_default() += q
        }
    }
    d
}
fn remaining_shop_draws(n: usize, horizon: usize) -> usize {
    ((n + 1)..=MAX_SHOP_UNLOCKS)
        .filter(|i| i * TOWN_SHOP_UNLOCK_INTERVAL <= horizon)
        .count()
}
fn shop_type_probability(shops: &[String], horizon: usize, members: &[&str]) -> f64 {
    let m = remaining_shop_draws(shops.len(), horizon);
    if m == 0 {
        return 0.0;
    }
    let hit = members.len() as f64 / 8.0;
    1.0 - (1.0 - hit).powi(m as i32)
}
fn update_drift(day: usize, market: &Counts, state: &mut DriftState) -> IndexMap<String, f64> {
    if day as i64 != state.prev_day {
        if state.prev_day >= 0 {
            for item in SELLABLE_ITEMS {
                if let (Some(a), Some(b)) = (state.prev_inv.get(item), market.get(item)) {
                    state.observed.insert(item.into(), (a - b) as f64);
                }
            }
        }
        state.prev_inv = market.clone();
        state.prev_day = day as i64
    }
    state.observed.clone()
}
fn effective_drift(
    demand: &IndexMap<String, f64>,
    observed: &IndexMap<String, f64>,
) -> IndexMap<String, f64> {
    SELLABLE_ITEMS
        .iter()
        .map(|i| {
            (
                (*i).into(),
                demand
                    .get(*i)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.6 * observed.get(*i).copied().unwrap_or(0.0)),
            )
        })
        .collect()
}
fn get_price(item: &str, market: &Counts) -> i64 {
    market_price(item, count_or(market, item, I0))
}
fn herd_feed_cost(herd: i64, drift: &IndexMap<String, f64>, market: &Counts) -> f64 {
    let d = drift.get("WHEAT").copied().unwrap_or(0.0) + herd as f64;
    market_price(
        "WHEAT",
        (count_or(market, "WHEAT", I0) - (d * 8.0) as i64).max(1),
    ) as f64
}
fn expected_price(
    item: &str,
    lag: i64,
    market: &Counts,
    drift: &IndexMap<String, f64>,
    supply: f64,
) -> i64 {
    let net = drift.get(item).copied().unwrap_or(0.0) - supply;
    market_price(
        item,
        (count_or(market, item, I0) - (net * lag as f64) as i64).max(1),
    )
}
fn demand_multiplier(item: &str, demand: &IndexMap<String, f64>, market: &Counts) -> f64 {
    let d = demand.get(item).copied().unwrap_or(0.0);
    let scarcity = I0 - count_or(market, item, I0);
    if d >= 12.0 {
        1.6
    } else if d >= 6.0 {
        1.3
    } else if scarcity as f64 > 0.6 * market_params(item).1 {
        1.4
    } else {
        1.0
    }
}

fn evaluate_animal(
    species: &'static str,
    day: usize,
    total_herd: i64,
    current: i64,
    drift: &IndexMap<String, f64>,
    market: &Counts,
    demand: &IndexMap<String, f64>,
) -> CandidateScore {
    let s = animal(species).unwrap();
    let cost = s.cost as f64;
    let days = TOTAL_DAYS as i64 - day as i64 - s.first_yield_day;
    if days <= 0 {
        return CandidateScore {
            kind: species,
            item: s.product,
            cost,
            daily_profit: 0.0,
            npv: -1.0,
        };
    }
    let harvests = days / s.interval;
    let lifetime = harvests * s.interval;
    let p = expected_price(s.product, s.first_yield_day, market, drift, current as f64);
    let revenue = lifetime as f64 * p as f64 * demand_multiplier(s.product, demand, market);
    let fert = (TOTAL_DAYS - day) as f64 * get_price("FERTILIZER", market) as f64;
    let feed = (TOTAL_DAYS - day) as f64 * herd_feed_cost(total_herd + 1, drift, market);
    let labor = (TOTAL_DAYS - day) as f64 * HAND_COST_PER_ANIMAL_DAY;
    let npv = revenue + fert - cost - feed - labor;
    CandidateScore {
        kind: species,
        item: s.product,
        cost,
        daily_profit: npv / (TOTAL_DAYS - day).max(1) as f64,
        npv,
    }
}
fn evaluate_fixed_crop(
    name: &'static str,
    day: usize,
    drift: &IndexMap<String, f64>,
    market: &Counts,
    demand: &IndexMap<String, f64>,
    current: f64,
) -> CandidateScore {
    let s = crop(name).unwrap();
    let cost = s.seed_cost as f64;
    let last = if name == "STRAWBERRY" {
        STRAWBERRY_LAST_PLANT_DAY
    } else {
        TOTAL_DAYS - s.yield_days[0] as usize - 2
    };
    if day > last {
        return CandidateScore {
            kind: name,
            item: name,
            cost,
            daily_profit: 0.0,
            npv: -1.0,
        };
    }
    let valid = s
        .yield_days
        .iter()
        .filter(|k| day + **k as usize <= LAST_DAY)
        .count();
    if valid == 0 {
        return CandidateScore {
            kind: name,
            item: name,
            cost,
            daily_profit: 0.0,
            npv: -1.0,
        };
    }
    let units = valid as f64 * 1.5;
    let p = expected_price(name, s.yield_days[0], market, drift, current);
    let npv = units * p as f64 * demand_multiplier(name, demand, market) - cost;
    let lifespan = (TOTAL_DAYS - day)
        .min(s.yield_days[s.yield_days.len() - 1] as usize + 1)
        .max(1);
    CandidateScore {
        kind: name,
        item: name,
        cost,
        daily_profit: npv / lifespan as f64,
        npv,
    }
}
fn evaluate_replant_crop(
    name: &'static str,
    day: usize,
    drift: &IndexMap<String, f64>,
    market: &Counts,
    demand: &IndexMap<String, f64>,
    current: f64,
) -> CandidateScore {
    let s = crop(name).unwrap();
    let cost = s.seed_cost as f64;
    if name == "MELON" && day > MELON_LAST_PLANT_DAY {
        return CandidateScore {
            kind: name,
            item: name,
            cost,
            daily_profit: 0.0,
            npv: -1.0,
        };
    }
    let last = if name == "WHEAT" {
        WHEAT_LAST_PLANT_DAY
    } else {
        TOTAL_DAYS - s.max_yield_day as usize - 1
    };
    if day > last {
        return CandidateScore {
            kind: name,
            item: name,
            cost,
            daily_profit: 0.0,
            npv: -1.0,
        };
    }
    let p = expected_price(name, s.max_yield_day, market, drift, current);
    let revenue = s.max_units_base as f64 * p as f64 * demand_multiplier(name, demand, market);
    let daily = (revenue - cost) / s.max_yield_day.max(1) as f64;
    CandidateScore {
        kind: name,
        item: name,
        cost,
        daily_profit: daily,
        npv: daily * (TOTAL_DAYS - day) as f64,
    }
}
fn daily_ap(kind: &str) -> f64 {
    match kind {
        "COW" => 3.3,
        "SHEEP" => 3.25,
        "GOOSE" => 4.0,
        "STRAWBERRY" | "TOMATO" | "WHEAT" => 1.5,
        "CARROT" => 1.6,
        _ => 1.5,
    }
}
fn estimate_marginal_ap(kind: &str, cows: i64, sheep: i64, geese: i64, crops: &Counts) -> f64 {
    let mut ap = cows as f64 * 3.3
        + sheep as f64 * 3.25
        + geese as f64 * 4.0
        + crops
            .iter()
            .map(|(k, n)| *n as f64 * daily_ap(k))
            .sum::<f64>();
    ap += 15.max(cows + sheep + geese) as f64 * 1.5;
    let before = ((ap / 22.0).ceil() as i64 - 1).max(0) as usize;
    let after = (((ap + daily_ap(kind)) / 22.0).ceil() as i64 - 1).max(0) as usize;
    if after > before {
        (total_hire_cost(after, 0) - total_hire_cost(before, 0)) as f64
    } else {
        0.0
    }
}

fn update_cull(
    day: usize,
    drift: &IndexMap<String, f64>,
    market: &Counts,
    c: &AnimalCensus,
    state: &mut CullState,
) -> Counts {
    if day as i64 == state.last_day {
        return state.retained_caps.clone();
    }
    state.last_day = day as i64;
    if day >= LIQUIDATION_DAY {
        for s in ANIMAL_SPECIES {
            state.retained_caps.insert(s.into(), 0);
        }
        return state.retained_caps.clone();
    }
    for species in ANIMAL_SPECIES {
        let spec = animal(species).unwrap();
        let n = match species {
            "COW" => c.total_cows(),
            "SHEEP" => c.total_sheep(),
            _ => c.total_geese(),
        };
        let projected = (count_or(market, spec.product, I0)
            - (drift.get(spec.product).copied().unwrap_or(0.0) * spec.first_yield_day as f64)
                as i64)
            .max(1);
        let revenue = market_price(spec.product, projected) as f64 / spec.interval as f64
            + get_price("FERTILIZER", market) as f64;
        let profit = revenue - get_price("WHEAT", market) as f64 - HAND_COST_PER_ANIMAL_DAY;
        if state.downsized.contains(species) {
            if !state.retained_caps.contains_key(species) {
                state
                    .retained_caps
                    .insert(species.into(), (n as f64 * 0.5).ceil().max(1.0) as i64);
            }
            continue;
        }
        if n <= 0 {
            state.negative_days.insert(species.into(), 0);
            state.retained_caps.insert(species.into(), 999);
            continue;
        }
        if profit < 0.0 {
            let neg = count(&state.negative_days, species) + 1;
            state.negative_days.insert(species.into(), neg);
            if neg >= 2 {
                state
                    .retained_caps
                    .insert(species.into(), (n as f64 * 0.5).ceil().max(1.0) as i64);
                state.downsized.insert(species.into());
                state.negative_days.insert(species.into(), 0);
            } else {
                state.retained_caps.insert(species.into(), 999);
            }
        } else {
            state.negative_days.insert(species.into(), 0);
            state.retained_caps.insert(species.into(), 999);
        }
    }
    state.retained_caps.clone()
}
fn throttled_qty(item: &str, qty: i64, market: &Counts, day: usize) -> i64 {
    if day >= LIQUIDATION_DAY || !["MELON", "WOOL", "STRAWBERRY", "WHEAT"].contains(&item) {
        return qty;
    }
    let now = get_price(item, market);
    if now <= 1 {
        return qty;
    }
    let inventory = count_or(market, item, I0);
    let (mut lo, mut hi, mut best) = (1, qty, 1);
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let after = market_price(item, inventory + mid);
        if (now - after) as f64 / now as f64 <= 0.15 {
            best = mid;
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    best
}
fn plan_sells(
    day: usize,
    shed: &Counts,
    f: &FarmState,
    market: &Counts,
    census: &AnimalCensus,
    caps: &Counts,
) -> (Vec<Vec<Value>>, f64) {
    let mut orders = vec![];
    let mut revenue = 0.0;
    let reserve = count_fertilize_due(f) + count_fertilize_due_tomorrow(f, day);
    let fert = (count(shed, "FERTILIZER") - reserve).max(0);
    if fert > 0 {
        orders.push(a3("SELL", "FERTILIZER", fert));
        revenue += (fert * get_price("FERTILIZER", market)) as f64
    }
    for item in [
        "MELON",
        "STRAWBERRY",
        "MILK",
        "WOOL",
        "EGG",
        "CARROT",
        "TOMATO",
    ] {
        let q = count(shed, item);
        if q > 0 {
            let n = throttled_qty(item, q, market, day);
            if n > 0 {
                orders.push(a3("SELL", item, n));
                revenue += (n * get_price(item, market)) as f64
            }
        }
    }
    let (_, ceiling) = wheat_feed_thresholds(census, caps, day);
    let wheat = count(shed, "WHEAT");
    if wheat > ceiling && day > 0 {
        let n = throttled_qty("WHEAT", wheat - ceiling, market, day);
        if n > 0 {
            orders.push(a3("SELL", "WHEAT", n));
            revenue += (n * get_price("WHEAT", market)) as f64
        }
    }
    (orders, revenue)
}
fn settle_orders(
    orders: &[Vec<Value>],
    money: f64,
    reserve: f64,
    market: &Counts,
    hires_today: usize,
    quads: &[String],
) -> Vec<Vec<Value>> {
    let mut settled = vec![];
    let mut cash = money;
    let mut hires = hires_today;
    for order in orders {
        let kind = order[0].as_str().unwrap();
        if kind == "SELL" {
            cash += order[2].as_i64().unwrap() as f64
                * get_price(order[1].as_str().unwrap(), market) as f64;
            settled.push(order.clone());
            continue;
        }
        if kind == "HIRE" {
            let cost = total_hire_cost(1, hires) as f64;
            if cash - cost < 0.0 {
                continue;
            }
            cash -= cost;
            hires += 1;
            settled.push(order.clone());
            continue;
        }
        if kind == "BUY_LAND" {
            let cost = match quads.len() {
                1 => 1000.0,
                2 => 2000.0,
                3 => 4000.0,
                _ => panic!("unexpected quads"),
            };
            if cash - cost < reserve {
                continue;
            }
            cash -= cost;
            settled.push(order.clone());
            continue;
        }
        let item = order[1].as_str().unwrap();
        let wanted = order[2].as_i64().unwrap();
        let unit = match kind {
            "BUY_PRODUCT" => get_price(item, market),
            "BUY_SEED" => crop(item).unwrap().seed_cost,
            "BUY_ANIMAL" => animal(item).unwrap().cost,
            _ => panic!("unknown order"),
        };
        let affordable = wanted.min(((cash - reserve).max(0.0) / unit as f64).floor() as i64);
        if affordable <= 0 {
            continue;
        }
        cash -= affordable as f64 * unit as f64;
        settled.push(a3(kind, item, affordable))
    }
    debug_assert!(cash >= -1e-6);
    settled
}
fn plan_seed_buy(orders: &mut Vec<Vec<Value>>, name: &str, want: i64) -> i64 {
    if want <= 0 {
        return 0;
    }
    orders.push(a3("BUY_SEED", name, want));
    want
}
fn desired_hires(
    day: usize,
    catalog: &[Task],
    premium: &[PlantTile],
    units: &[(Pos, Inventory)],
    shed: &Counts,
    n_quads: usize,
) -> usize {
    required_hand_count(
        catalog,
        premium,
        units,
        shed,
        MAX_HANDS.min(if n_quads < 3 { 11 } else { MAX_HANDS }),
        day,
    )
}

fn world_units(farm: &Farm, private: &PrivateState) -> Result<Vec<(Pos, Inventory)>, String> {
    let farmer = (
        farm.farmer.first().copied().ok_or("missing farmer x")?,
        farm.farmer.get(1).copied().ok_or("missing farmer y")?,
    );
    let mut out = vec![(
        farmer,
        private.inventories.first().cloned().unwrap_or_default(),
    )];
    for (i, p) in farm.hands.iter().enumerate() {
        out.push((
            (
                p.first().copied().ok_or("missing hand x")?,
                p.get(1).copied().ok_or("missing hand y")?,
            ),
            private.inventories.get(i + 1).cloned().unwrap_or_default(),
        ))
    }
    Ok(out)
}

fn evaluate_turn(game: &Game, seat: usize, state: &mut EvalState) -> Result<TurnDecision, String> {
    let step = game.step_index();
    let day = step / TURNS_PER_DAY;
    let hour = step % TURNS_PER_DAY;
    let me = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("EcoBot seat {seat} missing farm"))?;
    let private = game
        .privates()
        .get(seat)
        .ok_or_else(|| format!("EcoBot seat {seat} missing private"))?;
    let market: Counts = game
        .market()
        .inventory
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_i64()
                    .or_else(|| v.as_f64().map(|x| x as i64))
                    .unwrap_or(I0),
            )
        })
        .collect();
    let shops = &game.town().unlocked_shops;
    let units = world_units(me, private)?;
    let farm_state = parse_farm_state(&me.tiles, day)?;
    let census = count_animal_census(&farm_state, &private.shed, &private.inventories);
    let needed_pastures = compute_needed_pastures(&farm_state, &census);
    let needed_coops = compute_needed_coops(
        &farm_state,
        &census,
        &needed_pastures.iter().copied().collect(),
    );
    let demand = compute_daily_demand(shops);
    let observed = update_drift(day, &market, &mut state.drift);
    let drift = effective_drift(&demand, &observed);
    let retained = update_cull(day, &drift, &market, &census, &mut state.cull);
    let downsized = state.cull.downsized.clone();
    let mut orders = vec![];
    let mut budget = me.money;
    let cash_reserve = if (4..LIQUIDATION_DAY).contains(&day) {
        SEED_PURCHASE_CASH_RESERVE
    } else {
        0.0
    };
    if day == 0 && hour == 0 && count(&private.shed, "SHEEP") == 0 && farm_state.animals.is_empty()
    {
        orders.extend((0..4).map(|_| a1("HIRE")));
        orders.push(a3("BUY_ANIMAL", "SHEEP", 1));
        orders.push(a3("BUY_ANIMAL", "COW", 3));
        orders.push(a3("BUY_SEED", "MELON", 10));
        orders.push(a3("BUY_SEED", "WHEAT", 8));
        orders.push(a3("BUY_PRODUCT", "WHEAT", 4));
        let hints = DispatchHints {
            reserved_grazer_slots: 4,
            reserved_geese_slots: 0,
            crop_limits: IndexMap::from([("MELON".into(), 10), ("WHEAT".into(), 8)]),
        };
        let mut settled = settle_orders(
            &orders,
            me.money,
            cash_reserve,
            &market,
            me.hires_today,
            &me.unlocked_quadrants,
        );
        settled.truncate(MAX_MARKET_ORDERS);
        return Ok(TurnDecision {
            market_orders: settled,
            hints,
            retained_caps: retained,
            needed_pastures,
            needed_coops,
        });
    }
    let (sells, revenue) = plan_sells(day, &private.shed, &farm_state, &market, &census, &retained);
    orders.extend(sells);
    budget += revenue;
    let n_kept = kept_feedable_count(&census, &retained);
    if n_kept > 0 && day < LIQUIDATION_DAY {
        let (floor, _) = wheat_feed_thresholds(&census, &retained, day);
        let avail = count(&private.shed, "WHEAT")
            + private
                .inventories
                .iter()
                .map(|v| inv_count(v, "WHEAT"))
                .sum::<i64>();
        if avail < floor {
            let want = floor - avail;
            orders.push(a3("BUY_PRODUCT", "WHEAT", want));
            budget -= (get_price("WHEAT", &market) * want) as f64
        }
    }
    let (mut cows, mut sheep, mut geese) = (
        census.total_cows(),
        census.total_sheep(),
        census.total_geese(),
    );
    let mut crops = Counts::new();
    let quad_count = me.unlocked_quadrants.len();
    let blocked = BLOCKED_CLUSTER
        .iter()
        .filter(|p| me.unlocked_quadrants.iter().any(|q| q == quad_of(**p)))
        .count() as i64;
    let pasture_cap = if day > PASTURE_RESERVATION_LAST_DAY {
        cows + sheep
    } else {
        PASTURE_CLUSTER
            .iter()
            .filter(|p| me.unlocked_quadrants.iter().any(|q| q == quad_of(**p)))
            .count() as i64
    };
    let coop_cap = quad_count as i64 * 2;
    let tile_capacity = quad_count as i64 * 25 - blocked;
    let egg = *demand.get("EGG").unwrap_or(&0.0);
    let wool = *demand.get("WOOL").unwrap_or(&0.0);
    let milk = *demand.get("MILK").unwrap_or(&0.0);
    let milk_p = shop_type_probability(
        shops,
        PASTURE_RESERVATION_LAST_DAY,
        &["PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"],
    );
    let wool_p = shop_type_probability(shops, PASTURE_RESERVATION_LAST_DAY, &["YARN_STORE"]);
    let egg_p = shop_type_probability(
        shops,
        PASTURE_RESERVATION_LAST_DAY,
        &["BAKERY", "BRUNCH_SPOT"],
    );
    for _ in 0..60 {
        let total = cows + sheep + geese;
        let grazers = cows + sheep;
        let mut candidates = Vec::new();
        let mut add = |mut c: CandidateScore| {
            let ap = estimate_marginal_ap(c.kind, cows, sheep, geese, &crops);
            if ap > 0.0 {
                let days = (TOTAL_DAYS - day).max(1) as f64;
                c.daily_profit -= ap;
                c.npv -= ap * days
            }
            if c.npv > 0.0 && c.daily_profit > 0.0 {
                candidates.push(c)
            }
        };
        if grazers < pasture_cap && total < MAX_TOTAL_HERD {
            if !downsized.contains("COW") {
                let cap = count_or(&retained, "COW", 999);
                let max = if milk < 6.0 {
                    if milk_p >= RECOURSE_PROBABILITY_THRESHOLD {
                        4
                    } else {
                        2
                    }
                } else if milk < 12.0 {
                    6
                } else {
                    14.min(2 + (milk / 6.0).floor() as i64 * 4)
                };
                if cows < cap && cows < max {
                    add(evaluate_animal(
                        "COW", day, total, cows, &drift, &market, &demand,
                    ))
                }
            }
            if !downsized.contains("SHEEP") {
                let cap = count_or(&retained, "SHEEP", 999);
                let yarn = shops.iter().filter(|s| s.as_str() == "YARN_STORE").count() as i64;
                let base = if yarn == 0 {
                    if wool_p >= RECOURSE_PROBABILITY_THRESHOLD {
                        4
                    } else {
                        2
                    }
                } else if yarn == 1 {
                    4
                } else {
                    8.min(yarn * 4)
                };
                let cow_negative =
                    evaluate_animal("COW", day, total, cows, &drift, &market, &demand).npv <= 0.0;
                let max = if cow_negative && (yarn > 0 || wool >= 6.0) {
                    MAX_TOTAL_HERD
                } else {
                    base
                };
                if sheep < cap && sheep < max {
                    add(evaluate_animal(
                        "SHEEP", day, total, sheep, &drift, &market, &demand,
                    ))
                }
            }
        }
        let egg_shops = shops
            .iter()
            .filter(|s| matches!(s.as_str(), "BAKERY" | "BRUNCH_SPOT"))
            .count() as i64;
        if (egg_shops > 0 || egg_p >= RECOURSE_PROBABILITY_THRESHOLD)
            && geese < coop_cap
            && total < MAX_TOTAL_HERD
            && !downsized.contains("GOOSE")
        {
            let cap = count_or(&retained, "GOOSE", 999);
            let floor = if egg_shops == 0 && egg_p >= RECOURSE_PROBABILITY_THRESHOLD {
                2
            } else {
                0
            };
            let max = (egg as i64).max(floor);
            if geese < cap && geese < max {
                add(evaluate_animal(
                    "GOOSE", day, total, geese, &drift, &market, &demand,
                ))
            }
        }
        let available = (tile_capacity - pasture_cap - coop_cap - n_kept).max(0);
        let commercial: i64 = crops.values().sum();
        if commercial < available {
            if day <= STRAWBERRY_LAST_PLANT_DAY {
                let n = count(&crops, "STRAWBERRY");
                add(evaluate_fixed_crop(
                    "STRAWBERRY",
                    day,
                    &drift,
                    &market,
                    &demand,
                    n as f64 * 0.46,
                ))
            }
            if day <= MELON_LAST_PLANT_DAY {
                let n = count(&crops, "MELON");
                add(evaluate_replant_crop(
                    "MELON", day, &drift, &market, &demand, n as f64,
                ))
            }
            if demand.get("TOMATO").copied().unwrap_or(0.0) >= 6.0 {
                let n = count(&crops, "TOMATO");
                if n < 8 {
                    add(evaluate_fixed_crop(
                        "TOMATO",
                        day,
                        &drift,
                        &market,
                        &demand,
                        n as f64 * 0.5,
                    ))
                }
            }
            if demand.get("CARROT").copied().unwrap_or(0.0) >= 6.0 {
                let n = count(&crops, "CARROT");
                if n < 8 {
                    add(evaluate_replant_crop(
                        "CARROT", day, &drift, &market, &demand, n as f64,
                    ))
                }
            }
            if day <= WHEAT_LAST_PLANT_DAY {
                let n = count(&crops, "WHEAT");
                add(evaluate_replant_crop(
                    "WHEAT", day, &drift, &market, &demand, n as f64,
                ))
            }
        }
        let Some(mut best) = candidates.first().cloned() else {
            break;
        };
        for c in candidates.into_iter().skip(1) {
            if c.daily_profit > best.daily_profit {
                best = c
            }
        }
        if best.daily_profit <= 0.0 {
            break;
        }
        match best.kind {
            "COW" => cows += 1,
            "SHEEP" => sheep += 1,
            "GOOSE" => geese += 1,
            name => *crops.entry(name.into()).or_default() += 1,
        }
    }
    let total_needed = cows + sheep + geese + crops.values().sum::<i64>() + n_kept;
    let mut target_quads = quad_count;
    if quad_count < 3 {
        let next = if quad_count == 1 {
            1000.0
        } else if quad_count == 2 {
            2000.0
        } else {
            4000.0
        };
        let land_budget = budget >= next + cash_reserve;
        let empty = farm_state
            .empty_tiles
            .iter()
            .filter(|p| !BLOCKED_CLUSTER.contains(p))
            .count() as i64;
        let util = 1.0 - empty as f64 / (tile_capacity.max(1) as f64);
        let constrained = (total_needed >= tile_capacity && util >= 0.8)
            || empty <= 2
            || (cows + sheep)
                > PASTURE_CLUSTER
                    .iter()
                    .filter(|p| me.unlocked_quadrants.iter().any(|q| q == quad_of(**p)))
                    .count() as i64
            || geese > quad_count as i64 * 2;
        if land_budget && constrained {
            target_quads = quad_count + 1
        }
    }
    let hints = DispatchHints {
        reserved_grazer_slots: cows + sheep,
        reserved_geese_slots: geese,
        crop_limits: crops.clone(),
    };
    if hour == 0 || hour == 1 {
        let catalog = build_task_catalog(
            &farm_state,
            &private.shed,
            &private.seeds,
            &needed_pastures,
            &needed_coops,
            &units,
            day,
            hour,
            &hints,
            &me.unlocked_quadrants,
            &retained,
        );
        let n = desired_hires(
            day,
            &catalog,
            &ready_premium(&farm_state, day),
            &units,
            &private.shed,
            quad_count,
        );
        let mut hire = vec![a1("HIRE"); n];
        hire.extend(orders);
        orders = hire
    }
    if hour == PURCHASE_HOUR {
        let reserved = live_reserved_structures(
            &farm_state,
            &me.unlocked_quadrants,
            hints.reserved_grazer_slots,
            hints.reserved_geese_slots,
            day,
        );
        let needed: HashSet<_> = needed_pastures
            .iter()
            .chain(&needed_coops)
            .copied()
            .collect();
        let mut bought = 0;
        let mut grazer_bought = 0;
        let current_grazers = census.total_cows() + census.total_sheep();
        let current_total = census.total_all();
        let mut animal_orders = vec![];
        for (species, target) in [("COW", cows), ("SHEEP", sheep), ("GOOSE", geese)] {
            if downsized.contains(species) || (species == "GOOSE" && quad_count < 2) {
                continue;
            }
            let cap = count_or(&retained, species, 999);
            let have = match species {
                "COW" => census.total_cows(),
                "SHEEP" => census.total_sheep(),
                _ => census.total_geese(),
            };
            let mut effective = target.min(cap);
            if species == "GOOSE" {
                effective = effective.min(coop_capacity(&farm_state) + reserved.len() as i64)
            }
            let deficit = effective - have;
            if deficit <= 0 {
                continue;
            }
            let herd_slots = (MAX_TOTAL_HERD - (current_total + bought)).max(0);
            let max_buy = if species == "COW" || species == "SHEEP" {
                herd_slots.min((MAX_TOTAL_HERD - (current_grazers + grazer_bought)).max(0))
            } else {
                herd_slots
            };
            let want = deficit.min(max_buy);
            if want > 0 {
                animal_orders.push(a3("BUY_ANIMAL", species, want));
                bought += want;
                if species != "GOOSE" {
                    grazer_bought += want
                }
            }
        }
        let mut land_orders = vec![];
        let want_land = target_quads > quad_count;
        if !me.unlocked_quadrants.iter().any(|q| q == "NE") && want_land {
            land_orders.push(a1("BUY_LAND"))
        } else if !me.unlocked_quadrants.iter().any(|q| q == "SW")
            && me.unlocked_quadrants.iter().any(|q| q == "NE")
            && want_land
        {
            land_orders.push(a1("BUY_LAND"))
        } else if !me.unlocked_quadrants.iter().any(|q| q == "SE")
            && me.unlocked_quadrants.iter().any(|q| q == "SW")
            && target_quads >= 4
        {
            land_orders.push(a1("BUY_LAND"))
        }
        let wheat_only: HashSet<_> = reserved.difference(&needed).copied().collect();
        let mut empty = farm_state
            .empty_tiles
            .iter()
            .filter(|p| {
                !reserved.contains(p) && !needed.contains(p) && !BLOCKED_CLUSTER.contains(p)
            })
            .count() as i64;
        let mut wheat_reserved = farm_state
            .empty_tiles
            .iter()
            .filter(|p| wheat_only.contains(p))
            .count() as i64;
        let mut wheat_this = 0;
        if (4..=WHEAT_LAST_PLANT_DAY).contains(&day) {
            let curr = committed_count(&farm_state, &private.seeds, "WHEAT");
            let capacity = empty + wheat_reserved;
            if curr < n_kept && capacity > 0 {
                let b = plan_seed_buy(&mut orders, "WHEAT", (n_kept - curr).min(capacity));
                let from_open = b.min(empty);
                empty -= from_open;
                wheat_reserved -= b - from_open;
                wheat_this += b
            }
        }
        let straw_target = count(&crops, "STRAWBERRY");
        let total_straw = committed_count(&farm_state, &private.seeds, "STRAWBERRY");
        if day <= STRAWBERRY_LAST_PLANT_DAY && total_straw < straw_target && empty > 0 {
            let b = plan_seed_buy(
                &mut orders,
                "STRAWBERRY",
                (straw_target - total_straw).min(empty),
            );
            empty = (empty - b).max(0)
        }
        if !crops.is_empty() {
            let straw_done = committed_count(&farm_state, &private.seeds, "STRAWBERRY");
            for (name, target) in &crops {
                if matches!(name.as_str(), "STRAWBERRY" | "WHEAT")
                    || crop(name).is_none()
                    || *target <= 0
                {
                    continue;
                }
                if straw_done < straw_target {
                    let straw_npv = evaluate_fixed_crop(
                        "STRAWBERRY",
                        day,
                        &drift,
                        &market,
                        &demand,
                        straw_done as f64 * 0.46,
                    )
                    .npv;
                    let crop_npv = if crop(name).unwrap().ongoing {
                        evaluate_fixed_crop(
                            match name.as_str() {
                                "TOMATO" => "TOMATO",
                                _ => unreachable!(),
                            },
                            day,
                            &drift,
                            &market,
                            &demand,
                            0.0,
                        )
                        .npv
                    } else {
                        evaluate_replant_crop(
                            match name.as_str() {
                                "MELON" => "MELON",
                                "CARROT" => "CARROT",
                                _ => unreachable!(),
                            },
                            day,
                            &drift,
                            &market,
                            &demand,
                            0.0,
                        )
                        .npv
                    };
                    if straw_npv > crop_npv {
                        continue;
                    }
                }
                let spec = crop(name).unwrap();
                let lag = if spec.ongoing {
                    spec.first_yield_day
                } else {
                    spec.max_yield_day
                };
                if day > TOTAL_DAYS - lag as usize - 1 {
                    continue;
                }
                let curr = committed_count(&farm_state, &private.seeds, name);
                let freeing = if name == "CARROT" {
                    count_maturing_tomorrow(&farm_state, name)
                } else {
                    0
                };
                let want = *target + freeing - curr;
                if want > 0 {
                    let b = plan_seed_buy(&mut orders, name, want);
                    empty = (empty - b).max(0)
                }
            }
        }
        if (1..=WHEAT_LAST_PLANT_DAY).contains(&day) && (day < WHEAT_LAST_PLANT_DAY || hour < 12) {
            let target = n_kept + count(&crops, "WHEAT");
            let committed = committed_count(&farm_state, &private.seeds, "WHEAT") + wheat_this;
            let freeing = count_maturing_tomorrow(&farm_state, "WHEAT");
            let have = count(&private.seeds, "WHEAT") + wheat_this;
            let room = target - committed;
            let capacity = empty + wheat_reserved;
            let new_plant = capacity.min(room.max(0));
            let replant = freeing.min(room + freeing).max(0);
            plan_seed_buy(&mut orders, "WHEAT", (new_plant + replant - have).max(0));
        }
        orders.extend(land_orders);
        orders.extend(animal_orders);
    }
    let mut settled = settle_orders(
        &orders,
        me.money,
        cash_reserve,
        &market,
        me.hires_today,
        &me.unlocked_quadrants,
    );
    settled.truncate(MAX_MARKET_ORDERS);
    Ok(TurnDecision {
        market_orders: settled,
        hints,
        retained_caps: retained,
        needed_pastures,
        needed_coops,
    })
}

#[derive(Clone, Default)]
pub struct EcoBotController {
    eval_state: EvalState,
    day_plan: Option<DayPlan>,
    last_step: i64,
    last_debug: Value,
}

impl std::fmt::Debug for EcoBotController {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EcoBotController")
            .field("last_step", &self.last_step)
            .field("last_debug", &self.last_debug)
            .finish_non_exhaustive()
    }
}

impl EcoBotController {
    fn reset(&mut self) {
        self.eval_state = EvalState::default();
        self.day_plan = None;
        self.last_step = -1;
        self.last_debug = Value::Null;
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        let step = game.step_index();
        if step == 0 || (step as i64) < self.last_step {
            self.reset();
        }
        let day = step / TURNS_PER_DAY;
        let hour = step % TURNS_PER_DAY;
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("EcoBot seat {seat} missing farm"))?;
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("EcoBot seat {seat} missing private state"))?;
        let units = world_units(farm, private)?;
        let decision = evaluate_turn(game, seat, &mut self.eval_state)?;
        let actions = if hour == 0 {
            vec![a1("PASS"); units.len()]
        } else {
            if self
                .day_plan
                .as_ref()
                .is_none_or(|p| p.day != day || p.routes.len() < units.len())
            {
                let farm_state = parse_farm_state(&farm.tiles, day)?;
                let pending_existing = farm.hires_today.saturating_sub(farm.hands.len());
                let pending_ordered = decision
                    .market_orders
                    .iter()
                    .filter(|o| o.first().and_then(Value::as_str) == Some("HIRE"))
                    .count();
                self.day_plan = Some(build_day_plan(
                    &units,
                    &farm_state,
                    &private.shed,
                    &private.seeds,
                    day,
                    hour,
                    &decision.needed_pastures,
                    &decision.needed_coops,
                    &decision.hints,
                    &farm.unlocked_quadrants,
                    &decision.retained_caps,
                    pending_existing + pending_ordered,
                ));
            }
            let plan = self.day_plan.as_mut().unwrap();
            let mut rows = Vec::with_capacity(units.len());
            for unit in 0..units.len() {
                rows.push(next_action(unit, day, plan)?);
            }
            rows
        };
        let farmer = actions.first().cloned().unwrap_or_else(|| a1("PASS"));
        let hands = actions.into_iter().skip(1).collect::<Vec<_>>();
        self.last_step = step as i64;
        self.last_debug = json!({
            "step": step,
            "day": day,
            "hour": hour,
            "retained_caps": decision.retained_caps,
            "reserved_grazer_slots": decision.hints.reserved_grazer_slots,
            "reserved_geese_slots": decision.hints.reserved_geese_slots,
            "crop_limits": decision.hints.crop_limits,
            "plan_day": self.day_plan.as_ref().map(|p| p.day),
        });
        Ok(json!({"farmer": farmer, "hands": hands, "market": decision.market_orders}))
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}
