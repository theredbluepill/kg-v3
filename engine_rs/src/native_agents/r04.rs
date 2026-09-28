//! Complete native port of the frozen `agents/r04/main.py` controller.
//!
//! Keep this deliberately literal.  The parity suite, rather than stylistic
//! refactoring, is the authority for every ordering and arithmetic choice.

// The frozen Python-shaped port intentionally preserves indexing, branching, and
// intermediate bindings that clippy would rewrite. Exact parity outranks style here.
#![allow(dead_code, unused_mut)]
#![allow(
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::needless_range_loop
)]

use crate::{Farm, Game, Inventory};
use indexmap::IndexMap;
use serde_json::{Value, json};

const BOARD: usize = 10;
const TPD: usize = 24;
const LAST_DAY: usize = 29;
const LAST_STEP: usize = 718;
const SHED_CAP: i64 = 100;
const MARKET_I0: i64 = 10_000;
const SHED_TILES: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
const PRODUCTS: [&str; 9] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];
const PREMIUM: [&str; 4] = ["MILK", "WOOL", "STRAWBERRY", "MELON"];

// Frozen R04 values after its module-level environment reset/overrides.
const ANIMAL_ZONE: i64 = 12;
const ANIMAL_FAR_LIMIT: f64 = 4.0;
const SHEEP_CAP: i64 = 10;
const SHEEP_PER_YARN: i64 = 7;
const SHEEP_LAST_BUY: usize = 21;
const SHEEP_FLOOR: i64 = 2;
const MELON2_MAX: i64 = 400;
const MELON2_FULL: i64 = 400;
const MELON2_DAY: usize = 8;
const MELON2_MIN_MONEY: f64 = 1_000.0;
const CARROT_CAP: i64 = 12;
const CARROT_MIN_CAFES: i64 = 1;
const CARROT_LAST_PLANT: usize = 25;
const STRAW_CAP: i64 = 30;
const STRAW_LAST_PLANT: usize = 13;
const OPEN_MELONS: usize = 11;
const OPEN_SHEEP: i64 = 2;
const OPEN_COWS: i64 = 2;
const OPEN_WHEAT_SEEDS: i64 = 6;
const OPEN_WHEAT_FEED: i64 = 4;
const OPEN_HIRES: usize = 5;
const STRAW_RAMP: [i64; 4] = [6, 18, 30, 36];
const STRAW_FLOOR: i64 = 24;
const GOOSE_PER_EGGSHOP: i64 = 1;
const GOOSE_CAP: i64 = 3;
const GOOSE_FROM_DAY: usize = 3;
const GOOSE_LAST_BUY: usize = 18;
const COW_TARGET: i64 = 8;
const COW_NO_MILK_SHOP: i64 = 6;
const COW_TARGET_RICH: i64 = 9;

const URGENT_HOUR: usize = 15;
const FERT_PRIO: i64 = 84;
const CARE_PRIO: i64 = 70;
const ANIMAL_HARVEST_PRIO: i64 = 68;
const COLLECT_PRIO: i64 = 62;
const FEED_URGENT_HOUR: usize = 11;
const DYING_HOUR: usize = 13;
const WHEAT_LAST_PLANT: usize = 26;
const WALK_COST: f64 = 25.0;
const STICK_BONUS: f64 = 60.0;
const ON_TILE_BONUS: f64 = 3500.0;
const TIER_GAP: f64 = 3000.0;
const PRIO_W: f64 = 8.0;
const ANCHOR_W: f64 = 4.0;
const HIRE_WALK: f64 = 1.8;
const PREM_DROP_SCALE: f64 = 2.0;

const FERT_HOLD_MAX: i64 = 30;
const FERT_MIN_PRICE: i64 = 12;
const WHEAT_MIN_PRICE: i64 = 22;
const FLOOR_DECAY_DAY: usize = 25;
const OVERFLOW_PROJ: i64 = 88;
const OVERFLOW_TOTAL: i64 = 80;
const LAND_NE_DAY: usize = 6;
const LAND_SW_DAY: usize = 8;
const LAND_LAST_DAY: usize = 16;

#[derive(Clone, Copy)]
struct CropData {
    seed: i64,
    first: i64,
    maxday: i64,
    max_yield: i64,
}

fn crop(name: &str) -> Option<CropData> {
    Some(match name {
        "WHEAT" => CropData {
            seed: 10,
            first: 2,
            maxday: 4,
            max_yield: 6,
        },
        "CARROT" => CropData {
            seed: 20,
            first: 2,
            maxday: 3,
            max_yield: 4,
        },
        "TOMATO" => CropData {
            seed: 50,
            first: 8,
            maxday: 8,
            max_yield: 4,
        },
        "STRAWBERRY" => CropData {
            seed: 100,
            first: 10,
            maxday: 10,
            max_yield: 4,
        },
        "MELON" => CropData {
            seed: 80,
            first: 10,
            maxday: 12,
            max_yield: 6,
        },
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct AnimalData {
    cost: i64,
    first: i64,
    interval: i64,
}

fn animal(name: &str) -> Option<AnimalData> {
    Some(match name {
        "GOOSE" => AnimalData {
            cost: 300,
            first: 4,
            interval: 1,
        },
        "COW" => AnimalData {
            cost: 400,
            first: 8,
            interval: 2,
        },
        "SHEEP" => AnimalData {
            cost: 500,
            first: 6,
            interval: 3,
        },
        _ => return None,
    })
}

fn market_base(item: &str) -> i64 {
    match item {
        "WHEAT" => 25,
        "CARROT" => 35,
        "TOMATO" => 60,
        "STRAWBERRY" => 120,
        "MELON" => 250,
        "EGG" => 50,
        "MILK" => 160,
        "WOOL" => 200,
        "FERTILIZER" => 100,
        _ => 0,
    }
}

fn market_spec(item: &str) -> (f64, f64, &'static str, f64, &'static str, f64) {
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
        _ => (0.0, 1.0, "linear", 0.0, "linear", 0.0),
    }
}

fn shape(kind: &str, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match kind {
        "linear" => x,
        "sq" => x * x,
        "sqrt" => x.sqrt(),
        "log" => x.ln_1p(),
        "hinge" => {
            let u = x / t;
            u + 8.0 * (u - 1.0).max(0.0).powi(2)
        }
        _ => x,
    }
}

fn market_price(item: &str, inventory: i64) -> i64 {
    let (base, t, below, bt, above, at) = market_spec(item);
    let price = if inventory < MARKET_I0 {
        base + bt * base / shape(below, t, t) * shape(below, (MARKET_I0 - inventory) as f64, t)
    } else {
        base - at * base / shape(above, t, t) * shape(above, (inventory - MARKET_I0) as f64, t)
    };
    (price.round_ties_even() as i64).max(1)
}

fn units_above_price(item: &str, mut inventory: i64, floor: i64, max_units: i64) -> i64 {
    let mut n = 0;
    while n < max_units {
        let price = market_price(item, inventory);
        if price < floor {
            break;
        }
        n += 1;
        if price > 1 {
            inventory += 1;
        }
    }
    n
}

fn fib_cost(n: usize) -> i64 {
    let (mut a, mut b) = (1_i64, 1_i64);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

fn hire_total(n: usize) -> i64 {
    (0..n).map(fib_cost).sum()
}
fn dist(a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn nearest_shed(p: (i64, i64)) -> ((i64, i64), i64) {
    let mut best = SHED_TILES[0];
    let mut bd = 99;
    for shed in SHED_TILES {
        let d = dist(p, shed);
        if d < bd {
            best = shed;
            bd = d;
        }
    }
    (best, bd)
}

fn center_dist(p: (i64, i64)) -> f64 {
    (p.0 as f64 - 4.5).abs() + (p.1 as f64 - 4.5).abs()
}
fn euclid2(p: (i64, i64)) -> f64 {
    (p.0 as f64 - 4.5).powi(2) + (p.1 as f64 - 4.5).powi(2)
}
fn quadrant(x: i64, y: i64) -> &'static str {
    match (y < 5, x < 5) {
        (true, true) => "NW",
        (true, false) => "NE",
        (false, true) => "SW",
        (false, false) => "SE",
    }
}

fn step_toward(p: (i64, i64), q: (i64, i64)) -> &'static str {
    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
    if dx.abs() >= dy.abs() {
        if dx > 0 { "EAST" } else { "WEST" }
    } else if dy > 0 {
        "SOUTH"
    } else {
        "NORTH"
    }
}

fn count(map: &IndexMap<String, i64>, key: &str) -> i64 {
    *map.get(key).unwrap_or(&0)
}
fn number(map: &IndexMap<String, Value>, key: &str, default: i64) -> i64 {
    map.get(key).and_then(Value::as_i64).unwrap_or(default)
}
fn price_value(map: &IndexMap<String, Value>, key: &str, default: i64) -> i64 {
    map.get(key)
        .and_then(Value::as_i64)
        .or_else(|| map.get(key).and_then(Value::as_f64).map(|v| v as i64))
        .unwrap_or(default)
}
fn object(tile: &Value) -> Option<&serde_json::Map<String, Value>> {
    tile.as_object()
}
fn kind(tile: &Value) -> Option<&str> {
    object(tile)?.get("kind")?.as_str()
}
fn is_plant(tile: &Value) -> bool {
    kind(tile) == Some("PLANT")
}
fn is_weed(tile: &Value) -> bool {
    kind(tile) == Some("WEED")
}
fn is_struct(tile: &Value) -> bool {
    matches!(kind(tile), Some("COOP" | "PASTURE"))
}
fn has_animal(tile: &Value) -> bool {
    object(tile)
        .and_then(|m| m.get("animal"))
        .is_some_and(|v| !v.is_null())
}
fn is_empty_struct(tile: &Value) -> bool {
    is_struct(tile) && !has_animal(tile)
}
fn field_i64(tile: &Value, key: &str, default: i64) -> i64 {
    object(tile)
        .and_then(|m| m.get(key))
        .and_then(Value::as_i64)
        .unwrap_or(default)
}
fn field_bool(tile: &Value, key: &str) -> bool {
    object(tile)
        .and_then(|m| m.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}
fn field_str<'a>(tile: &'a Value, key: &str) -> Option<&'a str> {
    object(tile)?.get(key)?.as_str()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Animal,
    Melon,
    Straw,
    Carrot,
    Wheat,
}

#[derive(Clone, Debug, Default)]
struct R04State {
    last_step: i64,
    roles: IndexMap<(i64, i64), Role>,
    targets: IndexMap<usize, (i64, i64)>,
    anchors: IndexMap<usize, (f64, f64)>,
    melon_tiles: Vec<(i64, i64)>,
    melon_plant_day: usize,
    melon_wave2: bool,
    melon_wave2_n: usize,
    plan_day: i64,
    hires_wanted: usize,
    straw_target: i64,
    carrot_target: i64,
    cow_target: i64,
    sheep_target: i64,
    goose_target: i64,
}

impl R04State {
    fn fresh() -> Self {
        Self {
            last_step: -1,
            melon_wave2_n: 12,
            plan_day: -1,
            cow_target: OPEN_COWS,
            sheep_target: OPEN_SHEEP,
            ..Self::default()
        }
    }
}

#[derive(Clone)]
struct World {
    step: usize,
    day: usize,
    hour: usize,
    farm: Farm,
    money: f64,
    tiles: Vec<Vec<Value>>,
    units: Vec<(i64, i64)>,
    unlocked: Vec<String>,
    hires_today: usize,
    shed: IndexMap<String, i64>,
    seeds: IndexMap<String, i64>,
    invs: Vec<Inventory>,
    market_inventory: IndexMap<String, Value>,
    prices: IndexMap<String, Value>,
    shops: Vec<String>,
    steps_left: i64,
}

impl World {
    fn from_game(game: &Game, seat: usize) -> Result<Self, String> {
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("R04 seat {seat} missing farm"))?
            .clone();
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("R04 seat {seat} missing private state"))?
            .clone();
        let step = game.step_index();
        let mut units = vec![point(&farm.farmer)?];
        units.extend(
            farm.hands
                .iter()
                .map(|p| point(p))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut invs = private.inventories.clone();
        while invs.len() < units.len() {
            invs.push(IndexMap::new());
        }
        Ok(Self {
            step,
            day: step / TPD,
            hour: step % TPD,
            money: farm.money,
            tiles: farm.tiles.clone(),
            units,
            unlocked: farm.unlocked_quadrants.clone(),
            hires_today: farm.hires_today,
            shed: private.shed,
            seeds: private.seeds,
            invs,
            market_inventory: game.market().inventory.clone(),
            prices: game.market().prices.clone(),
            shops: game.town().unlocked_shops.clone(),
            steps_left: LAST_STEP as i64 - step as i64,
            farm,
        })
    }

    fn tile(&self, p: (i64, i64)) -> &Value {
        &self.tiles[p.1 as usize][p.0 as usize]
    }
    fn unlocked_tiles(&self) -> Vec<(i64, i64)> {
        let mut out = Vec::new();
        for y in 0..BOARD {
            for x in 0..BOARD {
                if self.tiles[y][x] != Value::String("LOCKED".into()) {
                    out.push((x as i64, y as i64));
                }
            }
        }
        out
    }
    fn shed_total(&self) -> i64 {
        self.shed.values().filter(|v| **v != 0).sum()
    }
    fn shop_count(&self, product: &str) -> i64 {
        self.shops
            .iter()
            .map(|shop| {
                let products: &[&str] = match shop.as_str() {
                    "BAKERY" => &["EGG", "WHEAT"],
                    "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
                    "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
                    "YARN_STORE" => &["WOOL"],
                    "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
                    "PET_CAFE" => &["CARROT"],
                    "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
                    "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
                    _ => &[],
                };
                if products.contains(&product) {
                    if products.len() == 1 { 2 } else { 1 }
                } else {
                    0
                }
            })
            .sum()
    }
}

fn point(value: &[i64]) -> Result<(i64, i64), String> {
    if value.len() != 2 {
        return Err(format!("R04 position must have two coordinates: {value:?}"));
    }
    Ok((value[0], value[1]))
}

fn count_animals(world: &World, wanted: Option<&str>) -> i64 {
    world
        .tiles
        .iter()
        .flatten()
        .filter(|tile| {
            has_animal(tile) && wanted.is_none_or(|w| field_str(tile, "animal") == Some(w))
        })
        .count() as i64
}

fn count_crop(world: &World, wanted: &str) -> i64 {
    world
        .tiles
        .iter()
        .flatten()
        .filter(|tile| is_plant(tile) && field_str(tile, "crop") == Some(wanted))
        .count() as i64
}

fn owned_animals(world: &World, wanted: &str) -> i64 {
    count_animals(world, Some(wanted))
        + count(&world.shed, wanted)
        + world.invs.iter().map(|inv| count(inv, wanted)).sum::<i64>()
}

#[derive(Clone, Debug)]
struct Job {
    tile: (i64, i64),
    actions: Vec<Value>,
    prio: i64,
    need: Option<&'static str>,
    kind: &'static str,
}

impl Job {
    fn verbs(tile: (i64, i64), actions: &[&str], prio: i64, kind: &'static str) -> Self {
        Self {
            tile,
            actions: actions.iter().map(|v| Value::String((*v).into())).collect(),
            prio,
            need: None,
            kind,
        }
    }
    fn compound(
        tile: (i64, i64),
        op: &str,
        arg: &str,
        prio: i64,
        need: Option<&'static str>,
        kind: &'static str,
    ) -> Self {
        Self {
            tile,
            actions: vec![Value::String(op.into()), Value::String(arg.into())],
            prio,
            need,
            kind,
        }
    }
}

#[derive(Clone, Debug)]
pub struct R04Controller {
    state: R04State,
    last_debug: Value,
}

impl Default for R04Controller {
    fn default() -> Self {
        Self {
            state: R04State::fresh(),
            last_debug: Value::Null,
        }
    }
}

fn ordered_tiles(world: &World) -> Vec<(i64, i64)> {
    let mut tiles = world.unlocked_tiles();
    tiles.sort_by(|a, b| {
        center_dist(*a)
            .partial_cmp(&center_dist(*b))
            .unwrap()
            .then_with(|| euclid2(*a).partial_cmp(&euclid2(*b)).unwrap())
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.0.cmp(&b.0))
    });
    tiles
}

fn assign_roles(state: &mut R04State, world: &World) {
    let tiles = world.unlocked_tiles();
    let mut n_animal = (state.cow_target + state.sheep_target + state.goose_target).max(4);
    if world.day >= 6 {
        n_animal = n_animal.max(ANIMAL_ZONE);
    } else if world.day >= 2 {
        n_animal = n_animal.max(8);
    }
    let ordered = ordered_tiles(world);
    let mut roles = IndexMap::new();

    if world.day == 0 && state.melon_tiles.is_empty() {
        let mut nw: Vec<_> = ordered
            .iter()
            .copied()
            .filter(|p| quadrant(p.0, p.1) == "NW")
            .collect();
        nw.sort_by(|a, b| {
            center_dist(*b)
                .partial_cmp(&center_dist(*a))
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.0.cmp(&b.0))
        });
        state.melon_tiles = nw.into_iter().take(OPEN_MELONS).collect();
    }
    if state.melon_wave2 && state.melon_tiles.is_empty() {
        let mut far: Vec<_> = tiles
            .iter()
            .copied()
            .filter(|p| world.tile(*p).is_null() || is_weed(world.tile(*p)))
            .collect();
        far.sort_by(|a, b| {
            center_dist(*b)
                .partial_cmp(&center_dist(*a))
                .unwrap()
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.0.cmp(&b.0))
        });
        state.melon_tiles = far.into_iter().take(state.melon_wave2_n).collect();
    }
    let mut animal_left = n_animal;
    for p in &ordered {
        if is_struct(world.tile(*p)) {
            roles.insert(*p, Role::Animal);
            animal_left -= 1;
        }
    }
    for p in &ordered {
        if roles.contains_key(p) || !state.melon_tiles.contains(p) {
            continue;
        }
        let tile = world.tile(*p);
        if state.melon_plant_day == world.day
            || (is_plant(tile) && field_str(tile, "crop") == Some("MELON"))
        {
            roles.insert(*p, Role::Melon);
        }
    }
    let mut n_zone = roles.values().filter(|role| **role == Role::Animal).count() as i64;
    for p in &ordered {
        if roles.contains_key(p) || animal_left <= 0 {
            continue;
        }
        let tile = world.tile(*p);
        if is_plant(tile) && field_str(tile, "crop") == Some("STRAWBERRY") {
            continue;
        }
        if n_zone >= ANIMAL_ZONE && center_dist(*p) > ANIMAL_FAR_LIMIT {
            continue;
        }
        roles.insert(*p, Role::Animal);
        animal_left -= 1;
        n_zone += 1;
    }
    let mut straw_left = state.straw_target;
    for p in &ordered {
        if roles.contains_key(p) {
            continue;
        }
        let tile = world.tile(*p);
        if is_plant(tile) && field_str(tile, "crop") == Some("STRAWBERRY") {
            roles.insert(*p, Role::Straw);
            straw_left -= 1;
        }
    }
    let mut rest: Vec<_> = ordered
        .iter()
        .copied()
        .filter(|p| !roles.contains_key(p))
        .collect();
    rest.sort_by(|a, b| {
        let af = world.tile(*a).is_null() || is_weed(world.tile(*a));
        let bf = world.tile(*b).is_null() || is_weed(world.tile(*b));
        (!af)
            .cmp(&(!bf))
            .then_with(|| center_dist(*a).partial_cmp(&center_dist(*b)).unwrap())
    });
    for p in rest {
        if straw_left <= 0 {
            break;
        }
        roles.insert(p, Role::Straw);
        straw_left -= 1;
    }

    let mut carrot_left = state.carrot_target;
    for p in &ordered {
        if roles.contains_key(p) {
            continue;
        }
        let tile = world.tile(*p);
        if is_plant(tile) && field_str(tile, "crop") == Some("CARROT") {
            roles.insert(*p, Role::Carrot);
            carrot_left -= 1;
        }
    }
    let mut rest: Vec<_> = ordered
        .iter()
        .copied()
        .filter(|p| !roles.contains_key(p))
        .collect();
    rest.sort_by(|a, b| {
        let af = world.tile(*a).is_null() || is_weed(world.tile(*a));
        let bf = world.tile(*b).is_null() || is_weed(world.tile(*b));
        (!af)
            .cmp(&(!bf))
            .then_with(|| center_dist(*a).partial_cmp(&center_dist(*b)).unwrap())
    });
    for p in rest {
        if carrot_left <= 0 {
            break;
        }
        roles.insert(p, Role::Carrot);
        carrot_left -= 1;
    }
    for p in ordered {
        roles.entry(p).or_insert(Role::Wheat);
    }
    state.roles = roles;
}

fn update_targets(state: &mut R04State, world: &World) {
    let day = world.day;
    let yarn = world.shop_count("WOOL") / 2;
    let milk_shops = world.shop_count("MILK");
    let straw_shops = world.shop_count("STRAWBERRY");
    state.cow_target = match day {
        0..=3 => OPEN_COWS,
        4 => 2,
        5 => 3,
        6 => 5,
        7 => 7,
        _ => {
            if milk_shops >= 1 {
                COW_TARGET
            } else {
                COW_NO_MILK_SHOP
            }
        }
    };
    if day >= 12 && milk_shops >= 4 {
        state.cow_target = COW_TARGET_RICH;
    }
    state.sheep_target = if day < 6 {
        OPEN_SHEEP
    } else {
        (OPEN_SHEEP + SHEEP_PER_YARN * yarn).clamp(SHEEP_FLOOR, SHEEP_CAP)
    };
    if day > 15 {
        state.cow_target = state.cow_target.min(owned_animals(world, "COW"));
    }
    if day > SHEEP_LAST_BUY {
        state.sheep_target = state.sheep_target.min(owned_animals(world, "SHEEP"));
    }
    if (GOOSE_FROM_DAY..=GOOSE_LAST_BUY).contains(&day) {
        state.goose_target = (GOOSE_PER_EGGSHOP * world.shop_count("EGG")).min(GOOSE_CAP);
    } else if day > GOOSE_LAST_BUY {
        state.goose_target = state.goose_target.min(owned_animals(world, "GOOSE"));
    }

    if (MELON2_DAY..=MELON2_DAY + 1).contains(&day) && !state.melon_wave2 {
        let melon_inventory = price_value(&world.market_inventory, "MELON", MARKET_I0);
        if melon_inventory <= MARKET_I0 + MELON2_MAX && world.money >= MELON2_MIN_MONEY {
            state.melon_wave2 = true;
            state.melon_wave2_n = if melon_inventory <= MARKET_I0 + MELON2_FULL {
                12
            } else {
                6
            };
            state.melon_tiles.clear();
            state.melon_plant_day = day;
        }
    }
    let mut target = match day {
        0..=2 => 0,
        3..=5 => STRAW_RAMP[0],
        6..=7 => STRAW_RAMP[1],
        8..=9 => STRAW_RAMP[2],
        _ => STRAW_RAMP[3],
    };
    if day >= 9 {
        target = (target + 3 * straw_shops - if straw_shops == 0 { 4 } else { 0 })
            .clamp(STRAW_FLOOR, STRAW_CAP);
    }
    let now = count_crop(world, "STRAWBERRY");
    if day > STRAW_LAST_PLANT {
        target = now;
    }
    state.straw_target = target.max(now);
    let pet_cafes = world
        .shops
        .iter()
        .filter(|shop| shop.as_str() == "PET_CAFE")
        .count() as i64;
    state.carrot_target = if day <= CARROT_LAST_PLANT && pet_cafes >= CARROT_MIN_CAFES {
        (4 * pet_cafes).min(CARROT_CAP)
    } else {
        0
    };
}

fn desired_hands(world: &World) -> usize {
    if world.day == 0 {
        return 5;
    }
    if world.day == 1 {
        return 1;
    }
    if world.day >= LAST_DAY {
        return 8;
    }
    let mut work = count_animals(world, None) as f64 * 4.0;
    work += world
        .tiles
        .iter()
        .flatten()
        .filter(|tile| is_plant(tile))
        .count() as f64
        * 1.25;
    let empties = world
        .unlocked_tiles()
        .iter()
        .filter(|p| world.tile(**p).is_null())
        .count()
        .min(20);
    work += empties as f64;
    if world.day == 8 {
        work += 40.0;
    }
    let mut n = (work * HIRE_WALK / 22.0).ceil() as i64 - 1;
    let (lo, hi) = match world.day {
        2..=5 => (3, 5),
        6..=7 => (7, 9),
        8..=9 => (10, 12),
        10..=27 => (12, 12),
        _ => (9, 11),
    };
    n = n.clamp(lo, hi);
    n as usize
}

fn compute_anchors(world: &World, n_hands: usize) -> IndexMap<usize, (f64, f64)> {
    let mut anchors = IndexMap::new();
    if n_hands == 0 {
        return anchors;
    }
    let mut points = Vec::new();
    for y in 0..BOARD {
        for x in 0..BOARD {
            let p = (x as i64, y as i64);
            if world.tile(p) == &Value::String("LOCKED".into()) {
                continue;
            }
            points.push((
                (y as f64 - 4.5).atan2(x as f64 - 4.5),
                x as f64,
                y as f64,
                if world.tile(p).is_null() { 0.35 } else { 1.0 },
            ));
        }
    }
    points.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then_with(|| a.1.partial_cmp(&b.1).unwrap())
            .then_with(|| a.2.partial_cmp(&b.2).unwrap())
            .then_with(|| a.3.partial_cmp(&b.3).unwrap())
    });
    if points.is_empty() {
        return anchors;
    }
    let per = points.iter().map(|p| p.3).sum::<f64>() / n_hands as f64;
    let mut acc = 0.0;
    let mut group: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut gi = 1;
    for p in points {
        acc += p.3;
        group.push(p);
        if acc >= per * gi as f64 && gi <= n_hands {
            let weight = group.iter().map(|p| p.3).sum::<f64>().max(1e-9);
            anchors.insert(
                gi,
                (
                    group.iter().map(|p| p.1 * p.3).sum::<f64>() / weight,
                    group.iter().map(|p| p.2 * p.3).sum::<f64>() / weight,
                ),
            );
            gi += 1;
            group.clear();
        }
    }
    if !group.is_empty() && gi <= n_hands {
        let weight = group.iter().map(|p| p.3).sum::<f64>().max(1e-9);
        anchors.insert(
            gi,
            (
                group.iter().map(|p| p.1 * p.3).sum::<f64>() / weight,
                group.iter().map(|p| p.2 * p.3).sum::<f64>() / weight,
            ),
        );
    }
    anchors
}

fn plan_day(state: &mut R04State, world: &World) {
    state.plan_day = world.day as i64;
    update_targets(state, world);
    assign_roles(state, world);
    state.hires_wanted = desired_hands(world);
    state.anchors = compute_anchors(world, state.hires_wanted);
}

fn water_wanted(crop: &str, age: i64, tile: &Value) -> bool {
    if age == 0 {
        return true;
    }
    if field_i64(tile, "consecutive_unwatered", 0) >= 1 {
        return true;
    }
    match crop {
        "WHEAT" => matches!(age, 2..=4),
        "CARROT" => matches!(age, 2..=3),
        "MELON" => {
            if age >= 6 {
                true
            } else {
                age % 2 == 0
            }
        }
        "STRAWBERRY" => {
            if age >= 9 {
                age % 2 == 1 || age <= 9
            } else {
                age % 2 == 0
            }
        }
        "TOMATO" => true,
        _ => age % 2 == 0,
    }
}

fn strawberry_harvest_wanted(age: i64, yield_units: i64, day: usize) -> bool {
    if yield_units <= 0 || age < 10 {
        return false;
    }
    if age >= 16 || day >= LAST_DAY || yield_units >= 1 {
        return true;
    }
    matches!(age, 11 | 13 | 15) && yield_units >= 3
}

fn structures_needed(state: &R04State, world: &World) -> i64 {
    if world.day > SHEEP_LAST_BUY {
        return 0;
    }
    let cows = owned_animals(world, "COW");
    let sheep = owned_animals(world, "SHEEP");
    let geese = owned_animals(world, "GOOSE");
    let placed = count_animals(world, None);
    let unplaced = cows + sheep + geese - placed;
    let empties = world
        .tiles
        .iter()
        .flatten()
        .filter(|tile| is_empty_struct(tile))
        .count() as i64;
    let wanted = (state.cow_target - cows).max(0)
        + (state.sheep_target - sheep).max(0)
        + (state.goose_target - geese).max(0);
    (unplaced + wanted - empties).max(0)
}

fn build_jobs(state: &R04State, world: &World) -> Vec<Job> {
    let mut jobs = Vec::new();
    let endgame = world.day >= LAST_DAY;
    let mut build_left = structures_needed(state, world);
    let mut coops_left = 0_i64;
    if state.goose_target > 0 {
        let empty_coops = world
            .tiles
            .iter()
            .flatten()
            .filter(|tile| is_empty_struct(tile) && kind(tile) == Some("COOP"))
            .count() as i64;
        coops_left = (state.goose_target.max(owned_animals(world, "GOOSE"))
            - count_animals(world, Some("GOOSE"))
            - empty_coops)
            .max(0);
    }
    let plant_wheat_prio = if world.day >= 14 { 66 } else { 58 };
    let mut seeds = world.seeds.clone();
    let mut animals_available: IndexMap<&str, i64> = ["GOOSE", "COW", "SHEEP"]
        .into_iter()
        .map(|a| {
            (
                a,
                count(&world.shed, a) + world.invs.iter().map(|inv| count(inv, a)).sum::<i64>(),
            )
        })
        .collect();
    let mut wheat_available = count(&world.shed, "WHEAT")
        + world
            .invs
            .iter()
            .map(|inv| count(inv, "WHEAT"))
            .sum::<i64>();
    let mut fert_available = count(&world.shed, "FERTILIZER")
        + world
            .invs
            .iter()
            .map(|inv| count(inv, "FERTILIZER"))
            .sum::<i64>();

    for (p, role) in &state.roles {
        let tile = world.tile(*p);
        if tile == &Value::String("LOCKED".into()) {
            continue;
        }
        if is_plant(tile) {
            let crop_name = field_str(tile, "crop").unwrap_or("");
            let age = world.day as i64 - field_i64(tile, "planted_day", world.day as i64);
            let yield_units = field_i64(tile, "yield_units", 0);
            let watered = field_bool(tile, "watered_today");
            if crop_name == "MELON" {
                if age >= 10 {
                    let mut acts = Vec::new();
                    if !watered && yield_units < 6 && age <= 12 && !endgame {
                        acts.push(Value::String("WATER".into()));
                    }
                    acts.push(Value::String("HARVEST".into()));
                    jobs.push(Job {
                        tile: *p,
                        actions: acts,
                        prio: 100,
                        need: None,
                        kind: "harvest",
                    });
                } else if !watered && water_wanted(crop_name, age, tile) {
                    let mut prio = if age < 6 { 76 } else { 79 };
                    if field_i64(tile, "consecutive_unwatered", 0) >= 1 {
                        prio = prio.max(if world.hour < DYING_HOUR { 82 } else { 91 });
                    }
                    if world.hour >= URGENT_HOUR {
                        prio = prio.max(92);
                    }
                    jobs.push(Job::verbs(*p, &["WATER"], prio, "tend"));
                }
                continue;
            }
            if crop_name == "WHEAT" || crop_name == "CARROT" {
                let data = crop(crop_name).unwrap();
                if age >= data.maxday || (endgame && yield_units > 0 && age >= data.first) {
                    let mut acts = Vec::new();
                    let win_start = (data.maxday + 1) / 2;
                    if !watered
                        && (win_start..=data.maxday).contains(&age)
                        && yield_units < data.max_yield
                        && !(endgame && world.hour >= 21)
                    {
                        acts.push(Value::String("WATER".into()));
                    }
                    acts.push(Value::String("HARVEST".into()));
                    let mut prio = if age >= data.maxday { 84 } else { 70 };
                    if age > data.maxday {
                        prio = 96;
                    }
                    jobs.push(Job {
                        tile: *p,
                        actions: acts,
                        prio,
                        need: None,
                        kind: "tend",
                    });
                } else if !watered && water_wanted(crop_name, age, tile) {
                    let mut prio = 74;
                    if field_i64(tile, "consecutive_unwatered", 0) >= 1 {
                        prio = if world.hour < DYING_HOUR { 82 } else { 91 };
                    }
                    if world.hour >= URGENT_HOUR {
                        prio = prio.max(92);
                    }
                    jobs.push(Job::verbs(*p, &["WATER"], prio, "tend"));
                }
                continue;
            }
            if crop_name == "STRAWBERRY" {
                if age >= 16 {
                    if yield_units > 0 {
                        jobs.push(Job::verbs(
                            *p,
                            if world.day <= 27 {
                                &["HARVEST", "DIG"]
                            } else {
                                &["HARVEST"]
                            },
                            88,
                            "tend",
                        ));
                    } else if world.day <= 27 {
                        jobs.push(Job::verbs(*p, &["DIG"], 64, "dig"));
                    }
                    continue;
                }
                let mut actions = Vec::new();
                let mut prio = 0;
                let mut need = None;
                if strawberry_harvest_wanted(age, yield_units, world.day) {
                    actions.push(Value::String("HARVEST".into()));
                    prio = if endgame {
                        95
                    } else if age < 16 {
                        76
                    } else {
                        88
                    };
                }
                let fert_until = field_i64(tile, "fertilized_until_day", -1);
                if !endgame && fert_until < world.day as i64 && fert_available > 0 {
                    let fp = if matches!(age, 9 | 13) {
                        Some(FERT_PRIO)
                    } else if matches!(age, 11 | 15) {
                        Some(FERT_PRIO - 6)
                    } else {
                        None
                    };
                    if let Some(fp) = fp {
                        actions.push(Value::String("FERTILIZE".into()));
                        need = Some("FERTILIZER");
                        fert_available -= 1;
                        prio = prio.max(fp);
                    }
                }
                if !watered && water_wanted(crop_name, age, tile) && !endgame {
                    actions.push(Value::String("WATER".into()));
                    let mut p2 = if matches!(age, 9 | 11 | 13 | 15) {
                        78
                    } else {
                        74
                    };
                    if field_i64(tile, "consecutive_unwatered", 0) >= 1 {
                        p2 = if world.hour < DYING_HOUR { 82 } else { 91 };
                    }
                    if world.hour >= URGENT_HOUR {
                        p2 = p2.max(92);
                    }
                    prio = prio.max(p2);
                }
                if !actions.is_empty() {
                    jobs.push(Job {
                        tile: *p,
                        actions,
                        prio,
                        need,
                        kind: "tend",
                    });
                }
                continue;
            }
            if let Some(data) = crop(crop_name) {
                if yield_units > 0 && age >= data.first {
                    jobs.push(Job::verbs(*p, &["HARVEST"], 70, "tend"));
                } else if !watered && !endgame {
                    jobs.push(Job::verbs(*p, &["WATER"], 74, "tend"));
                }
            }
            continue;
        }

        if has_animal(tile) {
            let animal_name = field_str(tile, "animal").unwrap_or("");
            let data = animal(animal_name).unwrap();
            let placed = field_i64(tile, "placed_day", 0);
            let yield_units = field_i64(tile, "yield_units", 0);
            let fed = field_bool(tile, "fed_today");
            let cared = field_bool(tile, "cared_today");
            let pending = field_i64(tile, "pending_care_bonus", 0);
            let fert_av = field_bool(tile, "fertilizer_available");
            let unfed = field_i64(tile, "consecutive_unfed", 0);
            let mut prod_tonight = false;
            let mut future_prod = false;
            for e in world.day as i64..29 {
                let ds = e + 1 - placed - data.first;
                if ds >= 0 && ds % data.interval == 0 {
                    prod_tonight = e == world.day as i64;
                    future_prod = true;
                    break;
                }
            }
            if endgame {
                if yield_units > 0 {
                    jobs.push(Job::verbs(*p, &["HARVEST"], 95, "animal"));
                }
                continue;
            }
            let mut actions = Vec::new();
            let mut prio = 0;
            let mut need_feed = !fed && (world.day < 28 || prod_tonight) && world.day >= 1;
            if world.day == 28 && !fed && !prod_tonight {
                need_feed = false;
            }
            if need_feed && wheat_available <= 0 {
                need_feed = false;
            }
            if need_feed {
                actions.push(Value::String("FEED".into()));
                prio = if unfed >= 1 { 86 } else { 80 };
                if world.hour >= FEED_URGENT_HOUR {
                    prio = prio.max(90);
                }
                if world.hour >= URGENT_HOUR {
                    prio = prio.max(93);
                }
            }
            if !cared && future_prod && pending < 5 && (fed || need_feed) && world.day < 28 {
                actions.push(Value::String("CARE".into()));
                prio = prio.max(CARE_PRIO);
            }
            if yield_units > 0 {
                actions.push(Value::String("HARVEST".into()));
                prio = prio.max(ANIMAL_HARVEST_PRIO);
            }
            if fert_av && world.day < LAST_DAY {
                actions.push(Value::String("COLLECT_FERTILIZER".into()));
                prio = prio.max(COLLECT_PRIO);
            }
            if !actions.is_empty() {
                jobs.push(Job {
                    tile: *p,
                    actions,
                    prio,
                    need: if need_feed { Some("WHEAT") } else { None },
                    kind: "animal",
                });
            }
            continue;
        }

        if is_empty_struct(tile) {
            let candidates: &[&str] = if kind(tile) == Some("PASTURE") {
                &["COW", "SHEEP"]
            } else {
                &["GOOSE"]
            };
            let mut placed = false;
            for candidate in candidates {
                if *animals_available.get(candidate).unwrap_or(&0) > 0 {
                    *animals_available.get_mut(candidate).unwrap() -= 1;
                    jobs.push(Job::compound(
                        *p,
                        "PLACE",
                        candidate,
                        72,
                        Some(candidate),
                        "place",
                    ));
                    placed = true;
                    break;
                }
            }
            if !placed && *role != Role::Animal && !endgame && world.day < 27 {
                jobs.push(Job::verbs(*p, &["DIG"], 20, "dig"));
            }
            continue;
        }
        if is_weed(tile) {
            if !endgame && world.day <= 27 {
                let prio = if *role == Role::Straw && (8..=15).contains(&world.day) {
                    60
                } else if world.day >= 14 {
                    62
                } else {
                    52
                };
                jobs.push(Job::verbs(*p, &["DIG"], prio, "dig"));
            }
            continue;
        }
        if tile.is_null() && !endgame {
            match role {
                Role::Animal => {
                    if build_left > 0 {
                        build_left -= 1;
                        if coops_left > 0 {
                            coops_left -= 1;
                            jobs.push(Job::verbs(*p, &["BUILD_COOP"], 66, "build"));
                        } else {
                            jobs.push(Job::verbs(*p, &["BUILD_PASTURE"], 66, "build"));
                        }
                    } else if (world.day <= 1 || world.day >= 18)
                        && world.day <= WHEAT_LAST_PLANT
                        && count(&seeds, "WHEAT") > 0
                    {
                        *seeds.get_mut("WHEAT").unwrap() -= 1;
                        jobs.push(Job::compound(
                            *p,
                            "PLANT",
                            "WHEAT",
                            plant_wheat_prio,
                            None,
                            "plant",
                        ));
                    }
                }
                Role::Melon => {
                    if world.day == state.melon_plant_day && count(&seeds, "MELON") > 0 {
                        *seeds.get_mut("MELON").unwrap() -= 1;
                        jobs.push(Job::compound(*p, "PLANT", "MELON", 70, None, "plant"));
                    } else if world.day > state.melon_plant_day
                        && count(&seeds, "WHEAT") > 0
                        && world.day <= WHEAT_LAST_PLANT
                    {
                        *seeds.get_mut("WHEAT").unwrap() -= 1;
                        jobs.push(Job::compound(
                            *p,
                            "PLANT",
                            "WHEAT",
                            plant_wheat_prio,
                            None,
                            "plant",
                        ));
                    }
                }
                Role::Straw => {
                    if count(&seeds, "STRAWBERRY") > 0 && world.day <= STRAW_LAST_PLANT {
                        *seeds.get_mut("STRAWBERRY").unwrap() -= 1;
                        jobs.push(Job::compound(*p, "PLANT", "STRAWBERRY", 68, None, "plant"));
                    } else if count(&seeds, "WHEAT") > 0 && world.day <= WHEAT_LAST_PLANT {
                        *seeds.get_mut("WHEAT").unwrap() -= 1;
                        jobs.push(Job::compound(
                            *p,
                            "PLANT",
                            "WHEAT",
                            plant_wheat_prio - 2,
                            None,
                            "plant",
                        ));
                    }
                }
                Role::Carrot => {
                    if count(&seeds, "CARROT") > 0 && world.day <= CARROT_LAST_PLANT {
                        *seeds.get_mut("CARROT").unwrap() -= 1;
                        jobs.push(Job::compound(
                            *p,
                            "PLANT",
                            "CARROT",
                            plant_wheat_prio,
                            None,
                            "plant",
                        ));
                    } else if count(&seeds, "WHEAT") > 0 && world.day <= WHEAT_LAST_PLANT {
                        *seeds.get_mut("WHEAT").unwrap() -= 1;
                        jobs.push(Job::compound(
                            *p,
                            "PLANT",
                            "WHEAT",
                            plant_wheat_prio - 2,
                            None,
                            "plant",
                        ));
                    }
                }
                Role::Wheat => {
                    if count(&seeds, "WHEAT") > 0 && world.day <= WHEAT_LAST_PLANT {
                        *seeds.get_mut("WHEAT").unwrap() -= 1;
                        jobs.push(Job::compound(
                            *p,
                            "PLANT",
                            "WHEAT",
                            plant_wheat_prio,
                            None,
                            "plant",
                        ));
                    }
                }
            }
        }
    }
    jobs
}

fn job_tier(prio: i64) -> i64 {
    if prio >= 90 {
        3
    } else if prio >= 60 {
        2
    } else {
        1
    }
}

#[derive(Clone, Debug)]
struct Pair {
    score: f64,
    unit: usize,
    job: usize,
    extra: i64,
}

fn global_assignment_pairs(pairs: &[Pair], n_units: usize, n_jobs: usize) -> Vec<Pair> {
    if n_units == 0 || pairs.is_empty() {
        return Vec::new();
    }
    let mut best: IndexMap<(usize, usize), Pair> = IndexMap::new();
    for pair in pairs {
        let key = (pair.unit, pair.job);
        if best.get(&key).is_none_or(|prior| pair.score > prior.score) {
            best.insert(key, pair.clone());
        }
    }
    let n = n_units;
    let m = n_jobs + n_units;
    let missing = 1e15;
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut matched_row = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    let cost = |row1: usize, col1: usize| -> f64 {
        let col = col1 - 1;
        if col >= n_jobs {
            0.0
        } else {
            best.get(&(row1 - 1, col))
                .map(|p| -p.score)
                .unwrap_or(missing)
        }
    };
    for row1 in 1..=n {
        matched_row[0] = row1;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        let mut col0 = 0usize;
        loop {
            used[col0] = true;
            let row0 = matched_row[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0;
            for col in 1..=m {
                if used[col] {
                    continue;
                }
                let cur = cost(row0, col) - u[row0] - v[col];
                if cur < minv[col] {
                    minv[col] = cur;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }
            for col in 0..=m {
                if used[col] {
                    u[matched_row[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;
            if matched_row[col0] == 0 {
                break;
            }
        }
        loop {
            let col1 = way[col0];
            matched_row[col0] = matched_row[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }
    let mut selected = Vec::new();
    for col1 in 1..=n_jobs {
        let row1 = matched_row[col1];
        if row1 > 0 {
            if let Some(pair) = best.get(&(row1 - 1, col1 - 1)) {
                selected.push(pair.clone());
            }
        }
    }
    selected.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    selected
}

fn action1(verb: &str) -> Value {
    json!([verb])
}

fn assign(state: &mut R04State, world: &World, jobs: &[Job]) -> Vec<Value> {
    let n_units = world.units.len();
    let mut actions = vec![action1("PASS"); n_units];
    let endgame = world.day >= LAST_DAY;
    let turns_left_today = TPD as i64 - 1 - world.hour as i64;
    let shed = world.shed.clone();
    let shed_room = SHED_CAP - world.shed_total();
    let invs = world.invs.clone();
    let mut unit_forced: IndexMap<usize, (i64, i64)> = IndexMap::new();
    for (index, position) in world.units.iter().enumerate() {
        let inv = &invs[index];
        let mut count_items = 0;
        let mut total_value = 0.0;
        for (item, quantity) in inv {
            if PRODUCTS.contains(&item.as_str()) && *quantity > 0 {
                total_value +=
                    *quantity as f64 * price_value(&world.prices, item, market_base(item)) as f64;
                count_items += *quantity;
            }
        }
        let (shed_tile, shed_distance) = nearest_shed(*position);
        if count_items == 0 {
            continue;
        }
        let mut must_drop = endgame && shed_distance >= world.steps_left - 1;
        if count(inv, "MELON") >= 6 {
            must_drop = true;
        }
        let premium_value: i64 = PREMIUM
            .iter()
            .map(|item| count(inv, item) * price_value(&world.prices, item, market_base(item)))
            .sum();
        if premium_value as f64 >= 300.0 * PREM_DROP_SCALE && shed_distance <= 2 {
            must_drop = true;
        }
        if premium_value as f64 >= 900.0 * PREM_DROP_SCALE && shed_distance <= 4 {
            must_drop = true;
        }
        if premium_value as f64 >= 2000.0 * PREM_DROP_SCALE && shed_distance <= 7 {
            must_drop = true;
        }
        if world.hour >= 22 && shed_distance <= 1 && shed_room > count_items {
            must_drop = true;
        }
        if must_drop {
            unit_forced.insert(index, shed_tile);
        }
        let _ = total_value;
    }

    let mut carried: IndexMap<&str, i64> = IndexMap::new();
    for inv in &invs {
        for (item, quantity) in inv {
            if *quantity > 0 {
                *carried.entry(item).or_default() += *quantity;
            }
        }
    }
    let mut pairs = Vec::new();
    for (unit, position) in world.units.iter().enumerate() {
        if unit_forced.contains_key(&unit) {
            continue;
        }
        let inv = &invs[unit];
        let stick = state.targets.get(&unit).copied();
        for (job_index, job) in jobs.iter().enumerate() {
            let d = dist(*position, job.tile);
            let mut extra = 0;
            let mut carrier_bonus = 0;
            if let Some(need) = job.need {
                if count(inv, need) > 0 {
                    carrier_bonus = 45;
                } else {
                    if count(&shed, need) <= 0 {
                        continue;
                    }
                    let (shed_tile, shed_distance) = nearest_shed(*position);
                    extra = shed_distance + dist(shed_tile, job.tile) - d;
                }
            }
            let total_distance = d + extra;
            if endgame {
                let (_, back) = nearest_shed(job.tile);
                if total_distance + job.actions.len() as i64 + back + 1 > world.steps_left {
                    continue;
                }
            } else if total_distance + job.actions.len() as i64 > turns_left_today + 1
                && job.prio < 100
            {
                continue;
            }
            let mut score = (job_tier(job.prio) as f64 * TIER_GAP) + job.prio as f64 * PRIO_W
                - total_distance as f64 * WALK_COST
                + carrier_bonus as f64;
            if stick == Some(job.tile) {
                score += STICK_BONUS;
            }
            if let Some(anchor) = state.anchors.get(&unit) {
                if job.kind != "animal" {
                    score -= ANCHOR_W
                        * ((job.tile.0 as f64 - anchor.0).abs()
                            + (job.tile.1 as f64 - anchor.1).abs());
                }
            }
            if d == 0 && extra == 0 {
                score += ON_TILE_BONUS;
            }
            score += 6.0 * job.actions.len() as f64;
            pairs.push(Pair {
                score,
                unit,
                job: job_index,
                extra,
            });
        }
    }
    pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let selected = global_assignment_pairs(&pairs, n_units, jobs.len());
    let selected_keys: Vec<_> = selected.iter().map(|p| (p.unit, p.job)).collect();
    let mut ordered_pairs = selected;
    ordered_pairs.extend(
        pairs
            .into_iter()
            .filter(|p| !selected_keys.contains(&(p.unit, p.job))),
    );

    let mut unit_done: Vec<usize> = unit_forced.keys().copied().collect();
    let mut job_taken = Vec::<usize>::new();
    let mut new_targets = IndexMap::new();
    let mut fetch_reserved: IndexMap<&str, i64> = IndexMap::new();
    for pair in ordered_pairs {
        if unit_done.contains(&pair.unit) || job_taken.contains(&pair.job) {
            continue;
        }
        let job = &jobs[pair.job];
        let position = world.units[pair.unit];
        let inv = &invs[pair.unit];
        job_taken.push(pair.job);
        unit_done.push(pair.unit);
        new_targets.insert(pair.unit, job.tile);
        if let Some(need) = job.need {
            if count(inv, need) <= 0 {
                let (shed_tile, shed_distance) = nearest_shed(position);
                if shed_distance == 0 {
                    let available = count(&shed, need) - *fetch_reserved.get(need).unwrap_or(&0);
                    if available <= 0 {
                        unit_done.retain(|v| *v != pair.unit);
                        job_taken.retain(|v| *v != pair.job);
                        continue;
                    }
                    let mut quantity = 1;
                    if matches!(need, "WHEAT" | "FERTILIZER") {
                        let remaining = jobs
                            .iter()
                            .enumerate()
                            .filter(|(j, other)| other.need == Some(need) && !job_taken.contains(j))
                            .count() as i64
                            + 1;
                        let uncovered = (remaining - *carried.get(need).unwrap_or(&0)).max(1);
                        quantity = available.min(uncovered).min(5).max(1);
                        *carried.entry(need).or_default() += quantity;
                    }
                    *fetch_reserved.entry(need).or_default() += quantity;
                    actions[pair.unit] = json!(["PICKUP", need, quantity]);
                } else {
                    actions[pair.unit] = action1(step_toward(position, shed_tile));
                }
                continue;
            }
        }
        if position == job.tile {
            let op = job.actions[0].as_str().unwrap();
            if matches!(op, "PLACE" | "PLANT") {
                actions[pair.unit] = json!([op, job.actions[1].as_str().unwrap()]);
            } else {
                actions[pair.unit] = action1(op);
            }
        } else {
            actions[pair.unit] = action1(step_toward(position, job.tile));
        }
    }
    for (unit, shed_tile) in unit_forced {
        let position = world.units[unit];
        actions[unit] = if position == shed_tile || SHED_TILES.contains(&position) {
            action1("DROP")
        } else {
            action1(step_toward(position, shed_tile))
        };
        new_targets.insert(unit, shed_tile);
    }
    for (unit, position) in world.units.iter().enumerate() {
        if unit_done.contains(&unit) {
            continue;
        }
        let carried_count: i64 = invs[unit]
            .iter()
            .filter(|(item, _)| PRODUCTS.contains(&item.as_str()))
            .map(|(_, quantity)| *quantity)
            .sum();
        let (shed_tile, shed_distance) = nearest_shed(*position);
        if carried_count > 0 {
            actions[unit] = if shed_distance == 0 {
                action1("DROP")
            } else {
                action1(step_toward(*position, shed_tile))
            };
            new_targets.insert(unit, shed_tile);
        } else if shed_distance > 1 {
            actions[unit] = action1(step_toward(*position, shed_tile));
        }
    }
    state.targets = new_targets;
    actions
}

fn daily_drain(world: &World, item: &str) -> i64 {
    (if item == "FERTILIZER" { 0 } else { 1 }) + world.shop_count(item) * (TPD as i64 / 4)
}

fn premium_floor(item: &str, world: &World) -> i64 {
    if world.step >= LAST_STEP - 1 || item == "MELON" {
        return 1;
    }
    let drain = daily_drain(world, item);
    let mut factor = match item {
        "WOOL" => {
            if drain >= 7 {
                0.55
            } else {
                0.20
            }
        }
        "MILK" => {
            if drain >= 7 {
                0.55
            } else {
                0.30
            }
        }
        _ => {
            if drain >= 7 {
                0.50
            } else {
                0.30
            }
        }
    };
    if world.day >= FLOOR_DECAY_DAY {
        let frac =
            (LAST_STEP as f64 - world.step as f64) / (LAST_STEP - FLOOR_DECAY_DAY * TPD) as f64;
        factor *= frac.clamp(0.0, 1.0);
    }
    ((market_base(item) as f64 * factor) as i64).max(1)
}

fn order(op: &str, item: &str, quantity: i64) -> Value {
    json!([op, item, quantity])
}

fn market_orders(state: &R04State, world: &World, _actions: &[Value]) -> Vec<Value> {
    let endgame = world.day >= LAST_DAY;
    let mut orders = Vec::new();
    let n_animals = count_animals(world, None);
    let n_cows = owned_animals(world, "COW");
    let n_sheep = owned_animals(world, "SHEEP");
    let mut cash_reserve = hire_total(state.hires_wanted.min(12)) + n_animals * 45 + 60;
    if world.day == 0 {
        cash_reserve = 0;
    }

    let mut fert_need = 0;
    for (p, _) in &state.roles {
        let tile = world.tile(*p);
        if is_plant(tile) {
            let age = world.day as i64 - field_i64(tile, "planted_day", world.day as i64);
            if field_str(tile, "crop") == Some("STRAWBERRY")
                && matches!(age, 8 | 9 | 11 | 12 | 13 | 15)
                && field_i64(tile, "fertilized_until_day", -1) < world.day as i64
            {
                fert_need += 1;
            }
        }
    }
    fert_need = fert_need.min(FERT_HOLD_MAX);
    if world.day >= 27 || (world.money < 40.0 && world.day < 8) {
        fert_need = 0;
    }
    let fertilizer = count(&world.shed, "FERTILIZER");
    let sell_f = fertilizer - fert_need;
    if sell_f > 0
        && (price_value(&world.prices, "FERTILIZER", 100) >= FERT_MIN_PRICE
            || endgame
            || world.shed_total() > OVERFLOW_TOTAL)
    {
        orders.push(order("SELL", "FERTILIZER", sell_f));
    }

    let wheat = count(&world.shed, "WHEAT");
    let mut reserve = if world.day < 28 {
        n_animals * (28_i64 - world.day as i64 + 1).max(0).min(3)
    } else {
        0
    };
    if world.day >= 27 {
        reserve = n_animals * (28_i64 - world.day as i64 + 1).max(0);
    }
    if endgame {
        reserve = 0;
    }
    let sell_wheat = wheat - reserve;
    if sell_wheat > 0
        && (price_value(&world.prices, "WHEAT", 25) >= WHEAT_MIN_PRICE
            || endgame
            || world.shed_total() > 75)
    {
        orders.push(order("SELL", "WHEAT", sell_wheat));
    }
    let melon = count(&world.shed, "MELON");
    if melon > 0 {
        orders.push(order("SELL", "MELON", melon));
    }

    let total = world.shed_total();
    let carried_total: i64 = world
        .invs
        .iter()
        .map(|inv| {
            inv.iter()
                .filter(|(item, _)| PRODUCTS.contains(&item.as_str()))
                .map(|(_, q)| *q)
                .sum::<i64>()
        })
        .sum();
    let projected = total + carried_total;
    let mut overflow = if world.hour >= 16 {
        (projected - OVERFLOW_PROJ).max(0)
    } else {
        (total - OVERFLOW_TOTAL).max(0)
    };
    for value in &orders {
        overflow -= value.as_array().unwrap()[2].as_i64().unwrap();
    }
    overflow = overflow.max(0);
    let mut premium = vec!["MILK", "WOOL", "STRAWBERRY", "EGG", "CARROT", "TOMATO"];
    premium.sort_by_key(|item| price_value(&world.prices, item, market_base(item)));
    for item in premium {
        let n = count(&world.shed, item);
        if n <= 0 {
            continue;
        }
        let inventory = price_value(&world.market_inventory, item, MARKET_I0);
        let floor = if matches!(item, "EGG" | "CARROT" | "TOMATO") {
            if endgame {
                1
            } else {
                (market_base(item) as f64 * 0.7) as i64
            }
        } else {
            premium_floor(item, world)
        };
        let mut quantity = units_above_price(item, inventory, floor, n);
        if overflow > 0 && quantity < n {
            let extra = (n - quantity).min(overflow);
            quantity += extra;
            overflow -= extra;
        }
        if endgame {
            quantity = n;
        }
        if quantity > 0 {
            orders.push(order("SELL", item, quantity));
        }
    }
    if overflow > 0 {
        let wheat_left = count(&world.shed, "WHEAT")
            - orders
                .iter()
                .filter(|v| {
                    v.as_array()
                        .is_some_and(|a| a.get(1).and_then(Value::as_str) == Some("WHEAT"))
                })
                .map(|v| v.as_array().unwrap()[2].as_i64().unwrap())
                .sum::<i64>();
        if wheat_left > 0 {
            let q = wheat_left.min(overflow);
            orders.push(order("SELL", "WHEAT", q));
            overflow -= q;
        }
        let fert_left = count(&world.shed, "FERTILIZER")
            - orders
                .iter()
                .filter(|v| {
                    v.as_array()
                        .is_some_and(|a| a.get(1).and_then(Value::as_str) == Some("FERTILIZER"))
                })
                .map(|v| v.as_array().unwrap()[2].as_i64().unwrap())
                .sum::<i64>();
        if overflow > 0 && fert_left > 0 {
            orders.push(order("SELL", "FERTILIZER", fert_left.min(overflow)));
        }
    }
    let mut proceeds = 0.0;
    for value in &orders {
        let row = value.as_array().unwrap();
        if row[0].as_str() != Some("SELL") {
            continue;
        }
        let item = row[1].as_str().unwrap();
        let mut inventory = price_value(&world.market_inventory, item, MARKET_I0);
        let mut total = 0;
        for _ in 0..row[2].as_i64().unwrap() {
            let price = market_price(item, inventory);
            total += price;
            if price > 1 {
                inventory += 1;
            }
        }
        proceeds += total as f64;
    }
    let mut cash = world.money + 0.9 * proceeds;

    if !endgame || world.hour <= 2 {
        let wanted = if endgame {
            state.hires_wanted.min(8)
        } else {
            state.hires_wanted
        };
        let mut needed = wanted.saturating_sub(world.hires_today);
        if world.hour > 3 {
            needed = 0;
        }
        let mut cost = 0;
        let mut hires = 0;
        while hires < needed && orders.len() < 10 {
            let c = fib_cost(world.hires_today + hires);
            if cash - ((cost + c) as f64) < 0.0 {
                break;
            }
            cost += c;
            hires += 1;
        }
        for _ in 0..hires {
            orders.push(json!(["HIRE"]));
        }
        cash -= cost as f64;
    }
    let unlocked = world.unlocked.len();
    if unlocked < 3 && !endgame && world.day <= LAND_LAST_DAY && orders.len() < 10 {
        let land_price = [1000, 2000, 4000][unlocked - 1];
        let cow_deficit = (state.cow_target - n_cows).max(0);
        let mut ok = false;
        if unlocked == 1
            && world.day >= LAND_NE_DAY
            && cash - land_price as f64 >= cash_reserve as f64
        {
            ok = true;
        }
        if unlocked == 2
            && world.day >= LAND_SW_DAY
            && cash - land_price as f64 >= (cash_reserve + 400 * cow_deficit + 300) as f64
        {
            ok = true;
        }
        if ok {
            orders.push(json!(["BUY_LAND"]));
            cash -= land_price as f64;
        }
    }
    if !endgame && orders.len() < 10 {
        let mut wanted: IndexMap<&str, i64> = IndexMap::new();
        for (p, role) in &state.roles {
            let tile = world.tile(*p);
            if !(tile.is_null() || is_weed(tile)) {
                continue;
            }
            match role {
                Role::Melon if world.day == state.melon_plant_day => {
                    *wanted.entry("MELON").or_default() += 1
                }
                Role::Straw if world.day >= 3 && world.day <= STRAW_LAST_PLANT => {
                    *wanted.entry("STRAWBERRY").or_default() += 1
                }
                Role::Carrot if world.day <= CARROT_LAST_PLANT => {
                    *wanted.entry("CARROT").or_default() += 1
                }
                Role::Wheat | Role::Straw | Role::Melon | Role::Carrot
                    if world.day >= 1 && world.day <= WHEAT_LAST_PLANT =>
                {
                    *wanted.entry("WHEAT").or_default() += 1
                }
                Role::Animal
                    if (world.day <= 1 || world.day >= 18)
                        && world.day <= WHEAT_LAST_PLANT
                        && structures_needed(state, world) == 0 =>
                {
                    *wanted.entry("WHEAT").or_default() += 1
                }
                _ => {}
            }
        }
        let max_plant_today = (world.units.len() as i64 * 2).max(2);
        for crop_name in ["MELON", "STRAWBERRY", "CARROT", "WHEAT"] {
            let mut n = *wanted.get(crop_name).unwrap_or(&0) - count(&world.seeds, crop_name);
            if crop_name == "STRAWBERRY" {
                n = n.min(40);
            } else if crop_name == "WHEAT" {
                n = n.min(max_plant_today);
            }
            if n <= 0 {
                continue;
            }
            let cost = crop(crop_name).unwrap().seed;
            let reserve = if matches!(crop_name, "WHEAT" | "CARROT") {
                cash_reserve.min(120)
            } else {
                cash_reserve
            };
            let quantity = (((cash - reserve as f64) / cost as f64).floor() as i64)
                .max(0)
                .min(n);
            if quantity > 0 {
                orders.push(order("BUY_SEED", crop_name, quantity));
                cash -= (quantity * cost) as f64;
            }
        }
    }
    if !endgame && world.day <= SHEEP_LAST_BUY && world.hour <= 6 && orders.len() < 10 {
        let mut capacity = 0;
        for (p, role) in &state.roles {
            let tile = world.tile(*p);
            if is_empty_struct(tile) && matches!(kind(tile), Some("PASTURE" | "COOP")) {
                capacity += 1;
            }
            if *role == Role::Animal && tile.is_null() {
                capacity += 1;
            }
        }
        let n_geese = owned_animals(world, "GOOSE");
        capacity -= (n_cows - count_animals(world, Some("COW")))
            + (n_sheep - count_animals(world, Some("SHEEP")))
            + (n_geese - count_animals(world, Some("GOOSE")));
        for (animal_name, wanted) in [
            ("COW", (state.cow_target - n_cows).max(0)),
            ("SHEEP", (state.sheep_target - n_sheep).max(0)),
            (
                "GOOSE",
                if world.day <= GOOSE_LAST_BUY {
                    (state.goose_target - n_geese).max(0)
                } else {
                    0
                },
            ),
        ] {
            let mut quantity = 0;
            let cost = animal(animal_name).unwrap().cost;
            while quantity < wanted && capacity > 0 {
                if cash - (cost as f64) < cash_reserve as f64 {
                    break;
                }
                cash -= cost as f64;
                capacity -= 1;
                quantity += 1;
            }
            if quantity > 0 {
                orders.push(order("BUY_ANIMAL", animal_name, quantity));
            }
        }
    }
    if !endgame && orders.len() < 10 {
        let wheat_now = count(&world.shed, "WHEAT");
        let need_today = state
            .roles
            .keys()
            .filter(|p| has_animal(world.tile(**p)) && !field_bool(world.tile(**p), "fed_today"))
            .count() as i64;
        let carried = world
            .invs
            .iter()
            .map(|inv| count(inv, "WHEAT"))
            .sum::<i64>();
        let mut short = if world.hour <= 20 {
            need_today - carried - wheat_now
        } else {
            n_animals - wheat_now - carried
        };
        if world.day >= 28 {
            short = short.min(0);
        }
        if short > 0 {
            let price = price_value(&world.prices, "WHEAT", 25) + 3;
            let quantity = short.min(((cash - 5.0) / price as f64).floor().max(0.0) as i64);
            if quantity > 0 {
                orders.push(order("BUY_PRODUCT", "WHEAT", quantity));
            }
        }
    }
    // R04 fixes PREMIUM_FIRST=1 and PREMIUM_ORDER=value.
    let front = ["STRAWBERRY", "MILK", "WOOL"];
    let mut premium_sells: Vec<Value> = orders
        .iter()
        .filter(|value| {
            value.as_array().is_some_and(|row| {
                row.first().and_then(Value::as_str) == Some("SELL")
                    && row
                        .get(1)
                        .and_then(Value::as_str)
                        .is_some_and(|item| front.contains(&item))
            })
        })
        .cloned()
        .collect();
    premium_sells.sort_by(|a, b| {
        let aa = a.as_array().unwrap();
        let bb = b.as_array().unwrap();
        let av = market_base(aa[1].as_str().unwrap()) * aa[2].as_i64().unwrap();
        let bv = market_base(bb[1].as_str().unwrap()) * bb[2].as_i64().unwrap();
        bv.cmp(&av)
    });
    let mut reordered = premium_sells.clone();
    reordered.extend(
        orders
            .into_iter()
            .filter(|value| !premium_sells.contains(value)),
    );
    orders = reordered;
    if world.step == 0 {
        orders = (0..OPEN_HIRES).map(|_| json!(["HIRE"])).collect();
        orders.extend([
            order("BUY_ANIMAL", "SHEEP", OPEN_SHEEP),
            order("BUY_ANIMAL", "COW", OPEN_COWS),
            order("BUY_SEED", "MELON", OPEN_MELONS as i64),
            order("BUY_SEED", "WHEAT", OPEN_WHEAT_SEEDS),
            order("BUY_PRODUCT", "WHEAT", OPEN_WHEAT_FEED),
        ]);
    }
    orders.truncate(10);
    orders
}

impl R04Controller {
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        let world = World::from_game(game, seat)?;
        if world.step == 0 || (world.step as i64) < self.state.last_step {
            self.state = R04State::fresh();
        }
        self.state.last_step = world.step as i64;
        if self.state.plan_day != world.day as i64 {
            plan_day(&mut self.state, &world);
        } else if self.state.roles.len() != world.unlocked_tiles().len() {
            assign_roles(&mut self.state, &world);
        }
        let jobs = build_jobs(&self.state, &world);
        let actions = assign(&mut self.state, &world, &jobs);
        let orders = market_orders(&self.state, &world, &actions);
        self.last_debug = json!({
            "step": world.step,
            "roles": self.state.roles.iter().map(|(p, role)| json!([p.0, p.1, format!("{role:?}")])).collect::<Vec<_>>(),
            "anchors": self.state.anchors.iter().map(|(u, p)| json!([u, p.0, p.1])).collect::<Vec<_>>(),
            "targets": self.state.targets.iter().map(|(u, p)| json!([u, p.0, p.1])).collect::<Vec<_>>(),
            "jobs": jobs.iter().map(|job| json!({"tile":[job.tile.0,job.tile.1],"actions":job.actions,"prio":job.prio,"need":job.need,"kind":job.kind})).collect::<Vec<_>>(),
        });
        let farmer = actions.first().cloned().unwrap_or_else(|| action1("PASS"));
        let hands = actions.iter().skip(1).cloned().collect::<Vec<_>>();
        Ok(json!({"farmer": farmer, "hands": hands, "market": orders}))
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}
