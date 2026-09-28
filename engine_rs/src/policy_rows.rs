//! ET-F policy rows in Rust: job set, feasible edges, model features and the action
//! executor, computed straight from the native `Game` state.
//!
//! This mirrors `scripts/et_f_actions.py` and the feature layout of
//! `scripts/et_f_policy.py` exactly (same job order, same edge order, same float
//! formulas evaluated in f64 then cast to f32), so the Python versions remain the parity
//! reference and the model sees identical inputs.  Rows are packed into one byte buffer
//! per request; the jobs and edges of every row are cached on the batch so the sampled
//! choices can be turned into engine actions without any Python-side state.

use crate::{Farm, Game, PrivateState};
use serde_json::{Map, Value};

pub const BOARD: i64 = 10;
pub const SHED_TILES: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
pub const MAX_ORDERS: usize = 10;
pub const CROP_ORDER: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
pub const ANIMAL_ORDER: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub const ANIMAL_STRUCTURE: [&str; 3] = ["COOP", "PASTURE", "PASTURE"];
pub const ANIMAL_PRODUCT: [&str; 3] = ["EGG", "MILK", "WOOL"];
pub const PRODUCTS: [&str; 9] = [
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
pub const ITEM_ORDER: [&str; 12] = [
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
pub const SHOP_ORDER: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];
pub const LAND_ORDER: [&str; 3] = ["NE", "SW", "SE"];
pub const JOB_KINDS: [&str; 7] = [
    "animal", "build", "dig", "harvest", "place", "plant", "tend",
];
pub const ACTION_VERBS: [&str; 18] = [
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "PLACE",
    "FEED",
    "CARE",
    "COLLECT_FERTILIZER",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
];
pub const TILE_KINDS: [&str; 6] = ["EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE"];
pub const TILE_CROPS: [&str; 5] = CROP_ORDER;
pub const TILE_ANIMALS: [&str; 3] = ANIMAL_ORDER;
pub const GLOBAL_DIM: usize = 96;
pub const TILE_DIM: usize = 29;
pub const PUBLIC_TILES: usize = 200;
pub const UNIT_DIM: usize = 21;
pub const JOB_DIM: usize = 43;
pub const RELATION_DIM: usize = 6;
pub const MARKET_LOGITS: usize = 77;
pub const MARKET_FACTORS: usize = 21;
pub const SEMANTIC_ACTIONS: usize = 18;
pub const SELL_FRACTIONS: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
pub const HIRE_OPTIONS: [i64; 4] = [0, 1, 2, 3];
pub const SEED_OPTIONS: [i64; 4] = [0, 2, 5, 10];
pub const ANIMAL_OPTIONS: [i64; 3] = [0, 1, 2];
pub const WHEAT_OPTIONS: [i64; 3] = [0, 5, 10];
pub const FERTILIZER_OPTIONS: [i64; 3] = [0, 2, 5];
pub const IDLE: i32 = -1;

#[derive(Clone, Debug)]
pub struct Job {
    pub tile: (i64, i64),
    pub verb: &'static str,
    pub argument: Option<&'static str>,
    pub need: Option<&'static str>,
    pub kind: &'static str,
    pub unit: Option<usize>,
}

impl Job {
    fn new(tile: (i64, i64), verb: &'static str, kind: &'static str) -> Self {
        Self {
            tile,
            verb,
            argument: None,
            need: None,
            kind,
            unit: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct World {
    pub player: usize,
    pub step: i64,
    pub day: i64,
    pub hour: i64,
    pub units: Vec<(i64, i64)>,
    pub inventories: Vec<Vec<i64>>, // indexed by ITEM_ORDER
    pub shed: Vec<i64>,             // ITEM_ORDER
    pub seeds: Vec<i64>,            // CROP_ORDER
    pub unlocked: Vec<String>,
    pub prices: Vec<f64>, // PRODUCTS
}

/// Everything cached per (game, seat) between `rows` and `actions`.
#[derive(Clone, Debug)]
pub struct CachedRow {
    pub world: World,
    pub jobs: Vec<Job>,
    pub edges: Vec<(usize, usize, i64)>,
}

#[derive(Clone, Debug, Default)]
pub struct PackedRow {
    pub counts: [u32; 4],
    pub global: Vec<f32>,
    pub tiles: Vec<f32>,
    pub units: Vec<f32>,
    pub jobs: Vec<f32>,
    pub edge_unit: Vec<i32>,
    pub edge_job: Vec<i32>,
    pub relation: Vec<f32>,
    pub market_mask: Vec<u8>,
}

/// Versioned compact transport for the same ET-F candidate/action surface.
///
/// Jobs remain the claim identities cached in `CachedRow`; only their redundant
/// 43-float feature rows are replaced on the wire by `(local tile, semantic action)`.
/// Edge positions remain the action ABI consumed by `re_batch_policy_actions`.
#[derive(Clone, Debug, Default)]
pub struct CompactPackedRow {
    pub counts: [u32; 5], // tiles, units, jobs, edges, player
    pub global: Vec<f32>,
    pub tiles: Vec<f32>,
    pub tile_public_index: Vec<u8>,
    pub units: Vec<f32>,
    pub job_code: Vec<u16>,
    pub edge_unit: Vec<u8>,
    pub edge_job: Vec<u16>,
    pub edge_extra_walk: Vec<u8>,
    pub market_mask: Vec<u8>,
}

fn map_i64(map: &Map<String, Value>, key: &str) -> i64 {
    match map.get(key) {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        Some(Value::Bool(b)) => *b as i64,
        _ => 0,
    }
}

fn map_bool(map: &Map<String, Value>, key: &str) -> bool {
    match map.get(key) {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        _ => false,
    }
}

fn map_str<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    map.get(key).and_then(Value::as_str)
}

fn number_f64(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn number_i64(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        _ => 0,
    }
}

/// Legacy feasibility decoding retains Python's falsy fallback semantics.
/// Public observation dates use `tile_date_i64` instead, preserving valid day zero.
fn falsy_i64(map: &Map<String, Value>, key: &str, default: i64) -> i64 {
    let value = number_i64(map.get(key));
    if map.get(key).is_none() || value == 0 {
        default
    } else {
        value
    }
}

/// Public observation dates must distinguish a valid day zero from a missing date.
fn tile_date_i64(map: &Map<String, Value>, key: &str, default: i64) -> i64 {
    match map.get(key) {
        None | Some(Value::Null) => default,
        value => number_i64(value),
    }
}

fn dist(a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn nearest_shed(position: (i64, i64)) -> ((i64, i64), i64) {
    let mut best = SHED_TILES[0];
    let mut best_distance = 99;
    for shed in SHED_TILES {
        let d = dist(position, shed);
        if d < best_distance {
            best = shed;
            best_distance = d;
        }
    }
    (best, best_distance)
}

fn step_toward(position: (i64, i64), target: (i64, i64)) -> Option<&'static str> {
    let dx = target.0 - position.0;
    let dy = target.1 - position.1;
    if dx == 0 && dy == 0 {
        return None;
    }
    if dx.abs() >= dy.abs() {
        Some(if dx > 0 { "EAST" } else { "WEST" })
    } else {
        Some(if dy > 0 { "SOUTH" } else { "NORTH" })
    }
}

fn item_index(item: &str) -> Option<usize> {
    ITEM_ORDER.iter().position(|candidate| *candidate == item)
}

fn crop_index(crop: &str) -> Option<usize> {
    CROP_ORDER.iter().position(|candidate| *candidate == crop)
}

enum Tile<'a> {
    Locked,
    Empty,
    Dict(&'a Map<String, Value>),
}

fn tile_at(farm: &Farm, x: usize, y: usize) -> Tile<'_> {
    match farm.tiles.get(y).and_then(|row| row.get(x)) {
        Some(Value::String(s)) if s == "LOCKED" => Tile::Locked,
        Some(Value::Object(map)) => Tile::Dict(map),
        _ => Tile::Empty,
    }
}

pub fn world_of(game: &Game, seat: usize) -> Result<World, String> {
    let farm = game.farms().get(seat).ok_or("seat outside farms")?;
    let private = game.privates().get(seat).ok_or("seat outside privates")?;
    let (day, hour) = game.config.turns_per_day.div_rem_usize(game.step);
    let step = game.step as i64;
    let mut units = vec![(farm.farmer[0], farm.farmer[1])];
    for hand in &farm.hands {
        units.push((hand[0], hand[1]));
    }
    let mut inventories = Vec::with_capacity(units.len());
    for index in 0..units.len() {
        let mut row = vec![0i64; ITEM_ORDER.len()];
        if let Some(inventory) = private.inventories.get(index) {
            for (item, count) in inventory {
                if let Some(position) = item_index(item) {
                    row[position] = *count;
                }
            }
        }
        inventories.push(row);
    }
    let mut shed = vec![0i64; ITEM_ORDER.len()];
    for (item, count) in &private.shed {
        if let Some(position) = item_index(item) {
            shed[position] = *count;
        }
    }
    let mut seeds = vec![0i64; CROP_ORDER.len()];
    for (item, count) in &private.seeds {
        if let Some(position) = crop_index(item) {
            seeds[position] = *count;
        }
    }
    let prices = PRODUCTS
        .iter()
        .map(|item| number_f64(game.market().prices.get(*item)))
        .collect();
    Ok(World {
        player: seat,
        step,
        day: day as i64,
        hour: hour as i64,
        units,
        inventories,
        shed,
        seeds,
        unlocked: farm.unlocked_quadrants.clone(),
        prices,
    })
}

fn available(world: &World, item: usize) -> i64 {
    world.shed[item] + world.inventories.iter().map(|row| row[item]).sum::<i64>()
}

pub fn full_job_set(game: &Game, world: &World) -> Vec<Job> {
    let farm = &game.farms()[world.player];
    let day = world.day;
    let fertilizer = available(world, 8);
    let wheat = available(world, 0);
    let animals_on_hand: Vec<i64> = (0..3).map(|k| available(world, 9 + k)).collect();
    let mut jobs = Vec::new();
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            let position = (x as i64, y as i64);
            match tile_at(farm, x, y) {
                Tile::Locked => continue,
                Tile::Empty => {
                    for (index, crop) in CROP_ORDER.iter().enumerate() {
                        if world.seeds[index] > 0 {
                            let mut job = Job::new(position, "PLANT", "plant");
                            job.argument = Some(crop);
                            jobs.push(job);
                        }
                    }
                    jobs.push(Job::new(position, "BUILD_COOP", "build"));
                    jobs.push(Job::new(position, "BUILD_PASTURE", "build"));
                }
                Tile::Dict(tile) => {
                    let kind = map_str(tile, "kind").unwrap_or("");
                    if kind == "WEED" {
                        jobs.push(Job::new(position, "DIG", "dig"));
                    } else if kind == "PLANT" {
                        if !map_bool(tile, "watered_today") {
                            jobs.push(Job::new(position, "WATER", "tend"));
                        }
                        if map_i64(tile, "yield_units") > 0 {
                            jobs.push(Job::new(position, "HARVEST", "harvest"));
                        }
                        let fertilized_until = falsy_i64(tile, "fertilized_until_day", -1);
                        if fertilizer > 0 && fertilized_until < day {
                            let mut job = Job::new(position, "FERTILIZE", "tend");
                            job.need = Some("FERTILIZER");
                            jobs.push(job);
                        }
                        jobs.push(Job::new(position, "DIG", "dig"));
                    } else if kind == "COOP" || kind == "PASTURE" {
                        let animal = map_str(tile, "animal").filter(|a| !a.is_empty());
                        if animal.is_some() {
                            if !map_bool(tile, "fed_today") && wheat > 0 {
                                let mut job = Job::new(position, "FEED", "animal");
                                job.need = Some("WHEAT");
                                jobs.push(job);
                            }
                            if !map_bool(tile, "cared_today") {
                                jobs.push(Job::new(position, "CARE", "animal"));
                            }
                            if map_i64(tile, "yield_units") > 0 {
                                jobs.push(Job::new(position, "HARVEST", "harvest"));
                            }
                            if map_bool(tile, "fertilizer_available") {
                                jobs.push(Job::new(position, "COLLECT_FERTILIZER", "animal"));
                            }
                        } else {
                            for (k, name) in ANIMAL_ORDER.iter().enumerate() {
                                if ANIMAL_STRUCTURE[k] == kind && animals_on_hand[k] > 0 {
                                    let mut job = Job::new(position, "PLACE", "place");
                                    job.argument = Some(name);
                                    job.need = Some(name);
                                    jobs.push(job);
                                }
                            }
                            jobs.push(Job::new(position, "DIG", "dig"));
                        }
                    }
                }
            }
        }
    }
    for (unit_index, inventory) in world.inventories.iter().enumerate() {
        if inventory.iter().any(|count| *count > 0) {
            let (shed, _) = nearest_shed(world.units[unit_index]);
            let mut job = Job::new(shed, "DROP", "drop");
            job.unit = Some(unit_index);
            jobs.push(job);
        }
    }
    jobs
}

pub fn feasible_edges(world: &World, jobs: &[Job]) -> Vec<(usize, usize, i64)> {
    let mut edges = Vec::new();
    for (unit_index, position) in world.units.iter().enumerate() {
        for (job_index, job) in jobs.iter().enumerate() {
            if let Some(owner) = job.unit {
                if owner == unit_index {
                    edges.push((unit_index, job_index, 0));
                }
                continue;
            }
            let mut extra = 0;
            if let Some(need) = job.need {
                let item = item_index(need).expect("need item");
                if world.inventories[unit_index][item] <= 0 {
                    if world.shed[item] <= 0 {
                        continue;
                    }
                    let (shed, to_shed) = nearest_shed(*position);
                    extra = to_shed + dist(shed, job.tile) - dist(*position, job.tile);
                }
            }
            edges.push((unit_index, job_index, extra));
        }
    }
    edges
}

fn one_hot(out: &mut Vec<f32>, value: &str, vocabulary: &[&str]) {
    for item in vocabulary {
        out.push(if *item == value { 1.0 } else { 0.0 });
    }
}

struct FarmSummary {
    money: f64,
    farmer: (i64, i64),
    hands: usize,
    hires_today: usize,
    unlocked: usize,
    crops: [i64; 5],
    animals: [i64; 3],
    structures: [i64; 2], // COOP, PASTURE
    ready: [i64; 9],
}

fn farm_summary(farm: &Farm) -> FarmSummary {
    let mut summary = FarmSummary {
        money: farm.money,
        farmer: (farm.farmer[0], farm.farmer[1]),
        hands: farm.hands.len(),
        hires_today: farm.hires_today,
        unlocked: farm.unlocked_quadrants.len(),
        crops: [0; 5],
        animals: [0; 3],
        structures: [0; 2],
        ready: [0; 9],
    };
    for row in &farm.tiles {
        for tile in row {
            if let Value::Object(map) = tile {
                let kind = map_str(map, "kind").unwrap_or("");
                if kind == "PLANT" {
                    if let Some(crop) = map_str(map, "crop").and_then(crop_index) {
                        summary.crops[crop] += 1;
                        summary.ready[crop] += map_i64(map, "yield_units");
                    }
                } else if kind == "COOP" || kind == "PASTURE" {
                    summary.structures[if kind == "COOP" { 0 } else { 1 }] += 1;
                    if let Some(animal) = map_str(map, "animal")
                        && let Some(k) = ANIMAL_ORDER.iter().position(|a| *a == animal)
                    {
                        summary.animals[k] += 1;
                        let product = PRODUCTS
                            .iter()
                            .position(|p| *p == ANIMAL_PRODUCT[k])
                            .unwrap();
                        summary.ready[product] += map_i64(map, "yield_units");
                    }
                }
            }
        }
    }
    summary
}

fn global_features(game: &Game, world: &World) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(GLOBAL_DIM);
    out.push((world.step as f64 / 719.0) as f32);
    out.push((world.day as f64 / 29.0) as f32);
    out.push((world.hour as f64 / 23.0) as f32);
    let market = game.market();
    for item in PRODUCTS {
        // Python: int(inventory) truncates fractional custom `I0` overrides
        out.push(((number_f64(market.inventory.get(item)).trunc() - 10_000.0) / 1_000.0) as f32);
    }
    for item in PRODUCTS {
        out.push((number_f64(market.prices.get(item)) / 300.0) as f32);
    }
    let shops = &game.town().unlocked_shops;
    for shop in SHOP_ORDER {
        let count = shops.iter().filter(|s| s.as_str() == shop).count();
        out.push((count as f64 / 8.0) as f32);
    }
    for farm in game.farms() {
        let summary = farm_summary(farm);
        out.push((summary.money / 200_000.0) as f32);
        out.push((summary.farmer.0 as f64 / 9.0) as f32);
        out.push((summary.farmer.1 as f64 / 9.0) as f32);
        out.push((summary.hands as f64 / 20.0) as f32);
        out.push((summary.hires_today as f64 / 20.0) as f32);
        out.push((summary.unlocked as f64 / 4.0) as f32);
        for count in summary.crops {
            out.push((count as f64 / 100.0) as f32);
        }
        for count in summary.animals {
            out.push((count as f64 / 50.0) as f32);
        }
        for count in summary.structures {
            out.push((count as f64 / 50.0) as f32);
        }
        for count in summary.ready {
            out.push((count as f64 / 100.0) as f32);
        }
    }
    for count in &world.shed {
        out.push((*count as f64 / 100.0) as f32);
    }
    for count in &world.seeds {
        out.push((*count as f64 / 100.0) as f32);
    }
    debug_assert_eq!(out.len(), GLOBAL_DIM);
    out
}

fn tile_features(game: &Game, world: &World) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(PUBLIC_TILES * TILE_DIM);
    let day = world.day;
    for (farm_index, farm) in game.farms().iter().enumerate() {
        let owner = if farm_index == world.player { 0 } else { 1 };
        for y in 0..10usize {
            for x in 0..10usize {
                let (kind, map): (&str, Option<&Map<String, Value>>) = match tile_at(farm, x, y) {
                    Tile::Locked => ("LOCKED", None),
                    Tile::Empty => ("EMPTY", None),
                    Tile::Dict(map) => {
                        let kind = map_str(map, "kind").unwrap_or("EMPTY");
                        (
                            if TILE_KINDS.contains(&kind) {
                                kind
                            } else {
                                "EMPTY"
                            },
                            Some(map),
                        )
                    }
                };
                let empty = Map::new();
                let payload = map.unwrap_or(&empty);
                let crop = map_str(payload, "crop").unwrap_or("");
                let animal = map_str(payload, "animal").unwrap_or("");
                let planted = tile_date_i64(payload, "planted_day", day);
                let placed = tile_date_i64(payload, "placed_day", day);
                let fertilized_until = tile_date_i64(payload, "fertilized_until_day", -1);
                out.push(if owner == 0 { 1.0 } else { 0.0 });
                out.push(if owner == 1 { 1.0 } else { 0.0 });
                out.push((x as f64 / 9.0) as f32);
                out.push((y as f64 / 9.0) as f32);
                one_hot(&mut out, kind, &TILE_KINDS);
                one_hot(&mut out, crop, &TILE_CROPS);
                one_hot(&mut out, animal, &TILE_ANIMALS);
                out.push(if kind == "PLANT" {
                    (((day - planted) as f64) / 30.0).max(0.0) as f32
                } else {
                    0.0
                });
                out.push(if !animal.is_empty() {
                    (((day - placed) as f64) / 30.0).max(0.0) as f32
                } else {
                    0.0
                });
                out.push(map_bool(payload, "watered_today") as i32 as f32);
                out.push((map_i64(payload, "consecutive_unwatered") as f64 / 2.0) as f32);
                out.push((map_i64(payload, "yield_units") as f64 / 10.0) as f32);
                out.push((((fertilized_until - day + 1) as f64) / 3.0).max(0.0) as f32);
                out.push(map_bool(payload, "fed_today") as i32 as f32);
                out.push((map_i64(payload, "consecutive_unfed") as f64 / 2.0) as f32);
                out.push(map_bool(payload, "cared_today") as i32 as f32);
                out.push(map_bool(payload, "fertilizer_available") as i32 as f32);
                out.push((map_i64(payload, "pending_care_bonus") as f64 / 6.0) as f32);
            }
        }
    }
    debug_assert_eq!(out.len(), PUBLIC_TILES * TILE_DIM);
    out
}

fn compact_tile_features(game: &Game, world: &World, jobs: &[Job]) -> (Vec<f32>, Vec<u8>) {
    // Reuse the exact dense feature arithmetic, then retain action-relevant rows in
    // stable public order: all unlocked tiles plus exact active-seat job targets.
    // DROP may target a shed coordinate before its quadrant is unlocked, so target
    // closure—not tile kind alone—is the sparse action ABI.
    let dense = tile_features(game, world);
    let mut local_job_targets = [false; 100];
    for job in jobs {
        if (0..BOARD).contains(&job.tile.0) && (0..BOARD).contains(&job.tile.1) {
            local_job_targets[(job.tile.1 * BOARD + job.tile.0) as usize] = true;
        }
    }
    let mut out = Vec::with_capacity(dense.len());
    let mut indices = Vec::with_capacity(PUBLIC_TILES);
    for (farm_index, farm) in game.farms().iter().enumerate() {
        for y in 0..10usize {
            for x in 0..10usize {
                if matches!(tile_at(farm, x, y), Tile::Locked)
                    && !(farm_index == world.player && local_job_targets[y * 10 + x])
                {
                    continue;
                }
                let public_index = farm_index * 100 + y * 10 + x;
                let start = public_index * TILE_DIM;
                out.extend_from_slice(&dense[start..start + TILE_DIM]);
                indices.push(u8::try_from(public_index).expect("public tile index fits u8"));
            }
        }
    }
    debug_assert_eq!(out.len(), indices.len() * TILE_DIM);
    (out, indices)
}

fn unit_features(world: &World) -> Vec<f32> {
    let mut out = Vec::with_capacity(world.units.len() * UNIT_DIM);
    for (index, position) in world.units.iter().enumerate() {
        out.push((position.0 as f64 / 9.0) as f32);
        out.push((position.1 as f64 / 9.0) as f32);
        out.push(if index == 0 { 1.0 } else { 0.0 });
        for count in &world.inventories[index] {
            out.push((*count as f64 / 100.0) as f32);
        }
        out.extend_from_slice(&[0.0; 6]);
    }
    out
}

fn job_features(jobs: &[Job]) -> Vec<f32> {
    let mut out = Vec::with_capacity(jobs.len() * JOB_DIM);
    for job in jobs {
        out.push((job.tile.0 as f64 / 9.0) as f32);
        out.push((job.tile.1 as f64 / 9.0) as f32);
        out.push(0.0);
        out.push(0.0);
        out.push(if job.need.is_some() {
            (1.0f64 / 5.0) as f32
        } else {
            0.0
        });
        one_hot(
            &mut out,
            if job.kind == "drop" { "" } else { job.kind },
            &JOB_KINDS,
        );
        one_hot(&mut out, job.verb, &ACTION_VERBS);
        let need = job.need.unwrap_or("NONE");
        out.push(if need == "NONE" { 1.0 } else { 0.0 });
        one_hot(&mut out, need, &ITEM_ORDER);
    }
    out
}

/// Ordered semantic action used by the compact wire.  It expands the argument-bearing
/// PLANT/PLACE verbs, so `(tile, action)` determines today's complete 43-wide job row.
fn semantic_action(job: &Job) -> Result<u16, String> {
    let action = match job.verb {
        "PLANT" => crop_index(job.argument.ok_or("PLANT job lacks its crop")?)
            .ok_or("PLANT job carries an unknown crop")? as u16,
        "PLACE" => {
            5 + ANIMAL_ORDER
                .iter()
                .position(|name| Some(*name) == job.argument)
                .ok_or("PLACE job carries an unknown animal")? as u16
        }
        "WATER" => 8,
        "HARVEST" => 9,
        "FERTILIZE" => 10,
        "DIG" => 11,
        "FEED" => 12,
        "CARE" => 13,
        "COLLECT_FERTILIZER" => 14,
        "BUILD_COOP" => 15,
        "BUILD_PASTURE" => 16,
        "DROP" => 17,
        verb => return Err(format!("compact row has unsupported job verb {verb}")),
    };
    debug_assert!(action < SEMANTIC_ACTIONS as u16);
    Ok(action)
}

fn compact_job_codes(jobs: &[Job]) -> Result<Vec<u16>, String> {
    jobs.iter()
        .map(|job| {
            let local_tile = u16::try_from(job.tile.1 * BOARD + job.tile.0)
                .map_err(|_| "job tile outside the board".to_string())?;
            Ok(local_tile * SEMANTIC_ACTIONS as u16 + semantic_action(job)?)
        })
        .collect()
}

fn relation_features(world: &World, jobs: &[Job], edges: &[(usize, usize, i64)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(edges.len() * RELATION_DIM);
    for (unit, job, extra) in edges {
        let position = world.units[*unit];
        let tile = jobs[*job].tile;
        out.push(0.0);
        out.push((*extra as f64 / 20.0) as f32);
        out.push((dist(position, tile) as f64 / 18.0) as f32);
        out.push(0.0);
        out.push(((tile.0 - position.0) as f64 / 9.0) as f32);
        out.push(((tile.1 - position.1) as f64 / 9.0) as f32);
    }
    out
}

fn market_mask(world: &World) -> Vec<u8> {
    let mut out = Vec::with_capacity(MARKET_LOGITS);
    for product in 0..PRODUCTS.len() {
        let stock = world.shed[product] > 0;
        out.push(1);
        for _ in 1..SELL_FRACTIONS.len() {
            out.push(stock as u8);
        }
    }
    out.extend(std::iter::repeat_n(1u8, HIRE_OPTIONS.len()));
    for _ in CROP_ORDER {
        out.extend(std::iter::repeat_n(1u8, SEED_OPTIONS.len()));
    }
    for _ in ANIMAL_ORDER {
        out.extend(std::iter::repeat_n(1u8, ANIMAL_OPTIONS.len()));
    }
    let remaining_land = LAND_ORDER
        .iter()
        .any(|q| !world.unlocked.iter().any(|u| u == q));
    out.push(1);
    out.push(remaining_land as u8);
    out.extend(std::iter::repeat_n(1u8, WHEAT_OPTIONS.len()));
    out.extend(std::iter::repeat_n(1u8, FERTILIZER_OPTIONS.len()));
    debug_assert_eq!(out.len(), MARKET_LOGITS);
    out
}

pub fn policy_row(game: &Game, seat: usize) -> Result<(PackedRow, CachedRow), String> {
    let world = world_of(game, seat)?;
    let jobs = full_job_set(game, &world);
    let edges = feasible_edges(&world, &jobs);
    let packed = PackedRow {
        counts: [
            world.units.len() as u32,
            jobs.len() as u32,
            edges.len() as u32,
            seat as u32,
        ],
        global: global_features(game, &world),
        tiles: tile_features(game, &world),
        units: unit_features(&world),
        jobs: job_features(&jobs),
        edge_unit: edges.iter().map(|(u, _, _)| *u as i32).collect(),
        edge_job: edges.iter().map(|(_, j, _)| *j as i32).collect(),
        relation: relation_features(&world, &jobs, &edges),
        market_mask: market_mask(&world),
    };
    Ok((packed, CachedRow { world, jobs, edges }))
}

/// The compact-v1 wire carries only irreducible row descriptors.  Its cached row is
/// byte-for-byte the same action authority as [`policy_row`]: sampled edge positions
/// are still resolved through the original ordered `jobs` and `edges` vectors.
pub fn compact_policy_row_v1(
    game: &Game,
    seat: usize,
) -> Result<(CompactPackedRow, CachedRow), String> {
    let world = world_of(game, seat)?;
    let jobs = full_job_set(game, &world);
    let edges = feasible_edges(&world, &jobs);
    let (tiles, tile_public_index) = compact_tile_features(game, &world, &jobs);
    if world.units.len() > u8::MAX as usize {
        return Err("compact-v1 row has more than 255 units".to_string());
    }
    if jobs.len() > u16::MAX as usize {
        return Err("compact-v1 row has more than 65535 jobs".to_string());
    }
    let mut edge_unit = Vec::with_capacity(edges.len());
    let mut edge_job = Vec::with_capacity(edges.len());
    let mut edge_extra_walk = Vec::with_capacity(edges.len());
    for (unit, job, extra) in &edges {
        edge_unit.push(
            u8::try_from(*unit).map_err(|_| "compact-v1 edge unit does not fit u8".to_string())?,
        );
        edge_job.push(
            u16::try_from(*job).map_err(|_| "compact-v1 edge job does not fit u16".to_string())?,
        );
        edge_extra_walk.push(
            u8::try_from(*extra)
                .map_err(|_| "compact-v1 edge extra walk does not fit u8".to_string())?,
        );
    }
    let packed = CompactPackedRow {
        counts: [
            tile_public_index.len() as u32,
            world.units.len() as u32,
            jobs.len() as u32,
            edges.len() as u32,
            seat as u32,
        ],
        global: global_features(game, &world),
        tiles,
        tile_public_index,
        units: unit_features(&world),
        job_code: compact_job_codes(&jobs)?,
        edge_unit,
        edge_job,
        edge_extra_walk,
        market_mask: market_mask(&world),
    };
    Ok((packed, CachedRow { world, jobs, edges }))
}

/// Serialize rows back to back: counts u32[4], then the fields in FIELD order.
pub fn pack_rows(rows: &[PackedRow]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for row in rows {
        for count in row.counts {
            out.extend_from_slice(&count.to_le_bytes());
        }
        for value in row
            .global
            .iter()
            .chain(&row.tiles)
            .chain(&row.units)
            .chain(&row.jobs)
        {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in row.edge_unit.iter().chain(&row.edge_job) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in &row.relation {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&row.market_mask);
    }
    out
}

/// Serialize compact-v1 rows back to back.  The version is part of the exported ABI
/// symbol; the payload starts with the row count and then, per row, counts u32[5]
/// followed by the fields in `CompactPackedRow` declaration order.
pub fn pack_compact_rows_v1(rows: &[CompactPackedRow]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for row in rows {
        for count in row.counts {
            out.extend_from_slice(&count.to_le_bytes());
        }
        for value in row.global.iter().chain(&row.tiles) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&row.tile_public_index);
        for value in &row.units {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in &row.job_code {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&row.edge_unit);
        for value in &row.edge_job {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&row.edge_extra_walk);
        out.extend_from_slice(&row.market_mask);
    }
    out
}

fn verb_action(verb: &str) -> Value {
    Value::Array(vec![Value::String(verb.to_string())])
}

/// Turn the sampled choices into this step's engine action for one seat.
pub fn policy_action(
    cached: &CachedRow,
    assignment: &[i32],
    market: &[i8],
) -> Result<Value, String> {
    let world = &cached.world;
    if assignment.len() != world.units.len() {
        return Err("assignment must cover every unit".to_string());
    }
    if market.len() != MARKET_FACTORS {
        return Err("market choices must cover every factor".to_string());
    }
    let mut verbs: Vec<Value> = vec![verb_action("PASS"); world.units.len()];
    let mut seeds = world.seeds.clone();
    let mut shed = world.shed.clone();
    for (unit_index, position) in world.units.iter().enumerate() {
        let choice = assignment[unit_index];
        if choice == IDLE {
            continue;
        }
        let edge = cached
            .edges
            .get(choice as usize)
            .ok_or("edge choice outside the row")?;
        if edge.0 != unit_index {
            return Err("edge chosen for the wrong unit".to_string());
        }
        let job = &cached.jobs[edge.1];
        if job.verb == "DROP" {
            verbs[unit_index] = if SHED_TILES.contains(position) {
                verb_action("DROP")
            } else {
                verb_action(step_toward(*position, job.tile).unwrap_or("PASS"))
            };
            continue;
        }
        if let Some(need) = job.need {
            let item = item_index(need).unwrap();
            if world.inventories[unit_index][item] <= 0 {
                let (shed_tile, to_shed) = nearest_shed(*position);
                if to_shed == 0 {
                    let available = shed[item];
                    if available <= 0 {
                        continue;
                    }
                    let cap = match need {
                        "WHEAT" => 5,
                        "FERTILIZER" => 3,
                        _ => 1,
                    };
                    let quantity = available.min(cap);
                    shed[item] = available - quantity;
                    verbs[unit_index] = Value::Array(vec![
                        Value::String("PICKUP".into()),
                        Value::String(need.into()),
                        Value::from(quantity),
                    ]);
                } else {
                    verbs[unit_index] =
                        verb_action(step_toward(*position, shed_tile).unwrap_or("PASS"));
                }
                continue;
            }
        }
        if *position != job.tile {
            verbs[unit_index] = verb_action(step_toward(*position, job.tile).unwrap_or("PASS"));
            continue;
        }
        match job.verb {
            "PLANT" => {
                let crop = job.argument.unwrap();
                let index = crop_index(crop).unwrap();
                if seeds[index] <= 0 {
                    continue;
                }
                seeds[index] -= 1;
                verbs[unit_index] = Value::Array(vec![
                    Value::String("PLANT".into()),
                    Value::String(crop.into()),
                ]);
            }
            "PLACE" => {
                verbs[unit_index] = Value::Array(vec![
                    Value::String("PLACE".into()),
                    Value::String(job.argument.unwrap().into()),
                ]);
            }
            other => verbs[unit_index] = verb_action(other),
        }
    }
    let orders = compile_market_orders(world, market)?;
    let mut action = Map::new();
    action.insert("farmer".into(), verbs[0].clone());
    action.insert("hands".into(), Value::Array(verbs[1..].to_vec()));
    action.insert("market".into(), Value::Array(orders));
    Ok(Value::Object(action))
}

fn option_at<T: Copy>(table: &[T], choice: i8, label: &str) -> Result<T, String> {
    table
        .get(usize::try_from(choice).map_err(|_| format!("negative {label} choice"))?)
        .copied()
        .ok_or_else(|| format!("{label} choice {choice} outside {} options", table.len()))
}

fn compile_market_orders(world: &World, choices: &[i8]) -> Result<Vec<Value>, String> {
    let mut sells: Vec<(f64, Value)> = Vec::new();
    let mut hires = 0i64;
    let mut land = 0i64;
    let mut animals = Vec::new();
    let mut seeds = Vec::new();
    let mut products = Vec::new();
    let mut cursor = 0usize;
    for (product, name) in PRODUCTS.iter().enumerate() {
        let fraction = option_at(&SELL_FRACTIONS, choices[cursor], "sell")?;
        cursor += 1;
        let stock = world.shed[product];
        let quantity = (stock as f64 * fraction).ceil() as i64;
        if quantity > 0 {
            let price = world.prices[product];
            sells.push((
                quantity as f64 * price,
                Value::Array(vec![
                    Value::String("SELL".into()),
                    Value::String((*name).into()),
                    Value::from(quantity),
                ]),
            ));
        }
    }
    hires = hires.max(option_at(&HIRE_OPTIONS, choices[cursor], "hire")?);
    cursor += 1;
    for crop in CROP_ORDER {
        let count = option_at(&SEED_OPTIONS, choices[cursor], "seed")?;
        cursor += 1;
        if count > 0 {
            seeds.push(Value::Array(vec![
                Value::String("BUY_SEED".into()),
                Value::String(crop.into()),
                Value::from(count),
            ]));
        }
    }
    for animal in ANIMAL_ORDER {
        let count = option_at(&ANIMAL_OPTIONS, choices[cursor], "animal")?;
        cursor += 1;
        if count > 0 {
            animals.push(Value::Array(vec![
                Value::String("BUY_ANIMAL".into()),
                Value::String(animal.into()),
                Value::from(count),
            ]));
        }
    }
    land = land.max(option_at(&[0i64, 1], choices[cursor], "land")?);
    cursor += 1;
    let wheat = option_at(&WHEAT_OPTIONS, choices[cursor], "wheat")?;
    cursor += 1;
    if wheat > 0 {
        products.push(Value::Array(vec![
            Value::String("BUY_PRODUCT".into()),
            Value::String("WHEAT".into()),
            Value::from(wheat),
        ]));
    }
    let fertilizer = option_at(&FERTILIZER_OPTIONS, choices[cursor], "fertilizer")?;
    if fertilizer > 0 {
        products.push(Value::Array(vec![
            Value::String("BUY_PRODUCT".into()),
            Value::String("FERTILIZER".into()),
            Value::from(fertilizer),
        ]));
    }
    // stable sort by descending quoted value, exactly like Python's list.sort
    sells.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut orders: Vec<Value> = sells.into_iter().map(|(_, row)| row).collect();
    for _ in 0..hires {
        orders.push(verb_action("HIRE"));
    }
    if land > 0 {
        orders.push(verb_action("BUY_LAND"));
    }
    orders.extend(animals);
    orders.extend(seeds);
    orders.extend(products);
    orders.truncate(MAX_ORDERS);
    Ok(orders)
}

/// Accessors on Game for this module (fields are private to lib.rs).
impl Game {
    pub fn farms(&self) -> &[Farm] {
        &self.farms
    }
    pub fn privates(&self) -> &[PrivateState] {
        &self.privates
    }
    pub fn market(&self) -> &crate::Market {
        &self.market
    }
    pub fn town(&self) -> &crate::Town {
        &self.town
    }
    pub fn step_index(&self) -> usize {
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    #[test]
    fn public_tile_dates_preserve_day_zero_and_nonzero_dates() {
        let payload = serde_json::json!({
            "planted_day": 0, "placed_day": 0, "fertilized_until_day": 0,
            "later": 7, "sentinel": -1
        });
        let map = payload.as_object().unwrap();
        assert_eq!(tile_date_i64(map, "planted_day", 12), 0);
        assert_eq!(tile_date_i64(map, "placed_day", 12), 0);
        assert_eq!(tile_date_i64(map, "fertilized_until_day", -1), 0);
        assert_eq!(tile_date_i64(map, "later", 12), 7);
        assert_eq!(tile_date_i64(map, "sentinel", 12), -1);
    }

    #[test]
    fn public_tile_dates_default_only_when_missing_or_null() {
        let payload = serde_json::json!({"null_date": null});
        let map = payload.as_object().unwrap();
        for key in ["missing_date", "null_date"] {
            assert_eq!(tile_date_i64(map, key, 12), 12);
            assert_eq!(tile_date_i64(map, key, -1), -1);
        }
    }

    #[test]
    fn public_tile_dates_reach_native_rows_without_advancing_a_game() {
        // Explicit state fixture, not native reset, rollout, or an episode test.
        fn fixture(day: usize, birth: Value) -> Game {
            let mut tiles = vec![vec![Value::Null; 10]; 10];
            tiles[0][0] = serde_json::json!({
                "kind": "PLANT", "crop": "MELON", "planted_day": birth,
                "fertilized_until_day": 0, "yield_units": 1
            });
            tiles[0][1] = serde_json::json!({
                "kind": "COOP", "animal": "GOOSE", "placed_day": birth
            });
            let farm = serde_json::json!({
                "money": 3000, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "unlocked_quadrants": ["NW"], "tiles": tiles
            });
            let private = serde_json::json!({"shed": {}, "seeds": {}, "inventories": [{}]});
            let header: crate::TraceHeader = serde_json::from_value(serde_json::json!({
                "format": crate::TRACE_FORMAT, "seed": 0, "configuration": {},
                "shop_schedule": [], "rng_schedule": [], "terminal_banks": [], "transitions": 0,
                "initial": {"public": {
                    "step": day * 24, "day": day, "hour": 0, "farms": [farm, farm],
                    "market": {"inventory": {}, "prices": {}}, "town": {"unlocked_shops": []}
                }, "privates": [private, private]}
            }))
            .unwrap();
            Game::from_header(&header).unwrap()
        }
        let old = fixture(12, Value::from(0));
        let fresh = fixture(12, Value::from(12));
        let old_rows = tile_features(&old, &world_of(&old, 0).unwrap());
        let fresh_rows = tile_features(&fresh, &world_of(&fresh, 0).unwrap());
        assert_eq!(old_rows.len(), PUBLIC_TILES * TILE_DIM);
        for offset in [0, 100 * TILE_DIM] {
            assert_eq!(old_rows[offset + 18], 12.0 / 30.0);
            assert_eq!(old_rows[offset + TILE_DIM + 19], 12.0 / 30.0);
            assert_eq!(fresh_rows[offset + 18], 0.0);
            assert_eq!(fresh_rows[offset + TILE_DIM + 19], 0.0);
        }
        let changed: Vec<_> = old_rows
            .iter()
            .zip(&fresh_rows)
            .enumerate()
            .filter_map(|(i, (left, right))| (left != right).then_some(i))
            .collect();
        assert_eq!(
            changed,
            vec![18, TILE_DIM + 19, 100 * TILE_DIM + 18, 101 * TILE_DIM + 19]
        );
        let day_zero = fixture(0, Value::from(0));
        let rows = tile_features(&day_zero, &world_of(&day_zero, 0).unwrap());
        assert_eq!(rows[23], 1.0 / 3.0);
        assert_eq!(old.step, 288);
        assert_eq!(fresh.step, 288);
    }

    #[test]
    fn compact_v1_keeps_ordered_action_authority_and_omits_dense_rows() {
        let game = Game::new(Config::default(), 7, 2).unwrap();
        let (dense, dense_cache) = policy_row(&game, 0).unwrap();
        let (compact, compact_cache) = compact_policy_row_v1(&game, 0).unwrap();
        assert_eq!(compact.counts[1] as usize, dense.counts[0] as usize);
        assert_eq!(compact.counts[2] as usize, dense.counts[1] as usize);
        assert_eq!(compact.counts[3] as usize, dense.counts[2] as usize);
        assert_eq!(compact.edge_unit.len(), dense.edge_unit.len());
        assert_eq!(compact.edge_job.len(), dense.edge_job.len());
        assert_eq!(compact.job_code.len(), dense_cache.jobs.len());
        assert!(
            compact
                .job_code
                .iter()
                .all(|code| *code < 100 * SEMANTIC_ACTIONS as u16)
        );
        assert!(
            compact
                .tile_public_index
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert_eq!(
            compact.tiles.len(),
            compact.tile_public_index.len() * TILE_DIM
        );
        assert_eq!(dense_cache.edges, compact_cache.edges);
        let assignment = vec![IDLE; dense_cache.world.units.len()];
        let market = vec![0i8; MARKET_FACTORS];
        assert_eq!(
            policy_action(&dense_cache, &assignment, &market).unwrap(),
            policy_action(&compact_cache, &assignment, &market).unwrap()
        );

        let packed = pack_compact_rows_v1(std::slice::from_ref(&compact));
        let expected = 4
            + 5 * 4
            + compact.global.len() * 4
            + compact.tiles.len() * 4
            + compact.tile_public_index.len()
            + compact.units.len() * 4
            + compact.job_code.len() * 2
            + compact.edge_unit.len()
            + compact.edge_job.len() * 2
            + compact.edge_extra_walk.len()
            + compact.market_mask.len();
        assert_eq!(packed.len(), expected);
    }
}
