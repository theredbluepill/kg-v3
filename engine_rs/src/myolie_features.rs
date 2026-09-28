//! Exact Myolie v3 legacy-prefix plus observation-v5 features from native state.
//!
//! Reference: `myolie/features.py`, `myolie/features_v3.py` in the frozen
//! unit-workspace and `ops/myolie-dagger-2026-09-22/observation_v5.py`.
//! This module encodes the requesting seat's public/own-private observation only.
//! It never creates a snapshot, serializes JSON, advances the game or reads RNG.
//! Arithmetic follows the Python scalar formulas in f64, then casts once to f32.

use crate::{ANIMAL_NAMES, CROP_NAMES, Counts, Farm, Game, PRODUCTS, SORTED_SHOPS};
use serde_json::{Map, Value};

pub const FEATURE_COUNT: usize = 8165;
/// Opt-in observation suffix v1; the legacy feature/array ABI remains 8165.
pub const INVEST_FEATURE_COUNT: usize = FEATURE_COUNT + 11;
const MAX_EXACT_INTEGER: f64 = (1_u64 << 24) as f64;

fn invest_integer(value: f64, name: &str, minimum: f64, maximum: f64) -> Result<f32, String> {
    if !value.is_finite() || value.fract() != 0.0 || value < minimum || value > maximum {
        return Err(format!("investment {name} is not an exact supported FP32 integer"));
    }
    Ok(value as f32)
}

/// Versioned suffix from exactly the public state and own public hire/land fields.
/// Configuration integer semantics are pinned to the official JSON schema.
pub fn encode_invest(game: &Game, seat: usize, out: &mut [f32]) -> Result<(), String> {
    if out.len() != INVEST_FEATURE_COUNT {
        return Err(format!("investment feature buffer requires {INVEST_FEATURE_COUNT} values"));
    }
    // Leave the complete old arithmetic and write order unchanged.
    encode(game, seat, &mut out[..FEATURE_COUNT])?;
    let config = &game.config;
    let episode = invest_integer(config.episode_steps.to_f64("episodeSteps")?, "episodeSteps", 1.0, MAX_EXACT_INTEGER)?;
    let turns = invest_integer(config.turns_per_day.to_f64("turnsPerDay")?, "turnsPerDay", 1.0, MAX_EXACT_INTEGER)?;
    let orders = invest_integer(config.max_market_orders_per_turn.to_f64("maxMarketOrdersPerTurn")?, "maxMarketOrdersPerTurn", 1.0, 10.0)?;
    if turns * orders > 240.0 {
        return Err("investment actor envelope exceeds 241".into());
    }
    let step = invest_integer(game.step as f64, "step", 0.0, f64::from(episode) - 1.0)?;
    let (day, hour) = config.turns_per_day.div_rem_usize(game.step);
    let day = invest_integer(day as f64, "day", 0.0, MAX_EXACT_INTEGER)?;
    let hour = invest_integer(hour as f64, "hour", 0.0, MAX_EXACT_INTEGER)?;
    let farm = &game.farms[seat];
    let order = ["NW", "NE", "SW", "SE"];
    let count = farm.unlocked_quadrants.len();
    if !(1..=4).contains(&count) || farm.unlocked_quadrants.iter().map(String::as_str).ne(order[..count].iter().copied()) {
        return Err("investment land must follow NW, NE, SW, SE without duplicates".into());
    }
    let hires = invest_integer(farm.hires_today as f64, "own.hires_today", 0.0, 240.0)?;
    let mult = invest_integer(config.farm_hand_cost_mult.to_f64("farmHandCostMult")?, "farmHandCostMult", 0.0, MAX_EXACT_INTEGER)?;
    let shed = invest_integer(config.shed_capacity.to_f64("shedCapacity")?, "shedCapacity", 1.0, MAX_EXACT_INTEGER)?;
    out[FEATURE_COUNT..].copy_from_slice(&[1.0, step, day, hour, episode, turns, hires, mult,
        ((1_u32 << count) - 1) as f32, shed, orders]);
    Ok(())
}

const MAX_ACTORS: usize = 241;
const AVAILABLE: usize = 1024;
const TILE_OFFSET: usize = 1027;
const ACTOR_OFFSET: usize = 3827;
const INVENTORY_OFFSET: usize = 5273;
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
const PLANT_FIELDS: [(&str, usize); 6] = [
    ("yield_units", 2),
    ("watered_today", 3),
    ("consecutive_unwatered", 4),
    ("planted_day", 5),
    ("max_lifespan_step", 6),
    ("fertilized_until_day", 7),
];
const ANIMAL_FIELDS: [(&str, usize); 7] = [
    ("yield_units", 2),
    ("placed_day", 8),
    ("consecutive_unfed", 9),
    ("fed_today", 10),
    ("cared_today", 11),
    ("fertilizer_available", 12),
    ("pending_care_bonus", 13),
];

fn number(value: &Value, field: &str) -> Result<f64, String> {
    // Ordinary engine numbers need no arbitrary-width integer/string adapter.
    // Keep that shared Python conversion as a fallback for explicit fixtures.
    if let Some(value) = value.as_i64() {
        return Ok(value as f64);
    }
    if let Some(value) = value.as_u64() {
        return Ok(value as f64);
    }
    if let Some(value) = value.as_f64() {
        return Ok(value);
    }
    if let Some(value) = value.as_bool() {
        return Ok(f64::from(value));
    }
    crate::python_float(value, field)
}

fn optional_number(tile: &Map<String, Value>, field: &str, default: f64) -> Result<f64, String> {
    tile.get(field)
        .map_or(Ok(default), |value| number(value, field))
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64() != Some(0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn nonnegative_int(value: Option<&Value>) -> Result<bool, String> {
    let Some(value) = value else { return Ok(false) };
    if let Some(value) = value.as_i64() {
        return Ok(value >= 0);
    }
    crate::python_int(value, "fertilized_until_day")
        .map(|integer| integer >= 0.into())
        .map_err(|error| error.to_string())
}

// Python min(1.0, value), including its ordering behavior for NaN. Required
// maintenance fields are separately checked for finiteness below.
fn min_one(value: f64) -> f64 {
    if value < 1.0 { value } else { 1.0 }
}

fn legacy_tile(tile: &Value, day: f64) -> Result<[f64; 3], String> {
    if tile.is_null() {
        return Ok([0.05, 0.0, 0.0]);
    }
    let Some(tile) = tile.as_object() else {
        return Ok([0.0; 3]);
    };
    let kind = tile.get("kind").and_then(Value::as_str).unwrap_or("");
    let animal = tile.get("animal").and_then(Value::as_str).unwrap_or("");
    let code = match kind {
        "WEED" => 0.1,
        "PLANT" => {
            let crop = tile.get("crop").and_then(Value::as_str).unwrap_or("");
            CROP_NAMES
                .iter()
                .position(|name| *name == crop)
                .map_or(0.2, |index| 0.2 + 0.025 * index as f64)
        }
        _ => ANIMAL_NAMES.iter().position(|name| *name == animal).map_or(
            match kind {
                "COOP" => 0.45,
                "PASTURE" => 0.5,
                _ => 0.0,
            },
            |index| 0.6 + 0.05 * index as f64,
        ),
    };
    let age = match tile.get("planted_day") {
        None | Some(Value::Null) => f64::from(truthy(tile.get("watered_today"))),
        Some(value) => number(value, "planted_day").map_or(0.0, |planted| {
            let age = min_one((day - planted) / 30.0);
            if age > 0.0 { age } else { 0.0 }
        }),
    };
    let state = if kind == "PLANT" {
        min_one(optional_number(tile, "yield_units", 0.0)? / 8.0)
            + 0.25 * f64::from(truthy(tile.get("watered_today")))
            + 0.125 * f64::from(nonnegative_int(tile.get("fertilized_until_day"))?)
    } else if tile.contains_key("animal") {
        min_one(optional_number(tile, "yield_units", 0.0)? / 8.0)
            + 0.25 * f64::from(truthy(tile.get("fed_today")))
            + 0.125 * f64::from(truthy(tile.get("cared_today")))
            + 0.0625 * f64::from(truthy(tile.get("fertilizer_available")))
    } else {
        0.0
    };
    Ok([code, age, state])
}

fn maintenance(tile: &Value, out: &mut [f32]) -> Result<(), String> {
    let Some(tile) = tile.as_object() else {
        return Ok(());
    };
    let fields: &[(&str, usize)] = if tile.get("kind").and_then(Value::as_str) == Some("PLANT") {
        out[0] = 1.0;
        &PLANT_FIELDS
    } else if tile.contains_key("animal") {
        out[1] = 1.0;
        &ANIMAL_FIELDS
    } else {
        return Ok(());
    };
    for &(name, index) in fields {
        let value = tile
            .get(name)
            .ok_or_else(|| format!("missing v5 observation field: {name}"))?;
        let value = number(value, name)?;
        if !value.is_finite() {
            return Err(format!("nonfinite v5 observation field: {name}"));
        }
        out[index] = value as f32;
    }
    Ok(())
}

fn positions(farm: &Farm) -> impl Iterator<Item = &Vec<i64>> {
    std::iter::once(&farm.farmer).chain(&farm.hands)
}

fn count(counts: &Counts, item: &str) -> f64 {
    *counts.get(item).unwrap_or(&0) as f64
}

fn total(counts: &Counts) -> Result<f64, String> {
    // Python 3.11 sums floats sequentially; 3.12+ compensates rounding. Bound
    // the absolute aggregate so every integer and every partial sum is exactly
    // representable in f64 on both versions, including canceling signed inputs.
    // Explicit exotic states outside this scope fail instead of drifting.
    let mut remaining = 1_u64 << 53;
    let mut sum = 0.0;
    for &value in counts.values() {
        remaining = remaining.checked_sub(value.unsigned_abs()).ok_or_else(|| {
            "Myolie private count absolute aggregate exceeds exact f64 bound 2^53".to_string()
        })?;
        sum += value as f64;
    }
    Ok(sum)
}

fn push(out: &mut [f32], cursor: &mut usize, value: f64) {
    out[*cursor] = value as f32;
    *cursor += 1;
}

/// Encode one legal own-seat observation into the caller's exact-width buffer.
///
/// The buffer is scratch output: on error it may be partially written and must
/// not be consumed. Callers publishing batches must fail the entire operation.
/// Own shed/seed absolute aggregates above 2^53 are deliberately unsupported:
/// Python versions differ in floating-point summation outside this exact range.
pub fn encode(game: &Game, seat: usize, out: &mut [f32]) -> Result<(), String> {
    if out.len() != FEATURE_COUNT {
        return Err(format!(
            "Myolie feature buffer requires {FEATURE_COUNT} values, got {}",
            out.len()
        ));
    }
    if seat > 1 || game.farms.len() != 2 || game.privates.len() != 2 {
        return Err("v5 requires the pinned two-player observation".into());
    }
    let config = &game.config;
    if config.board_size != 10 {
        return Err("v5 observation schema requires the pinned 10x10 board".into());
    }
    let turns = config.turns_per_day.capped_usize().max(1);
    let orders = config.max_market_orders_per_turn.capped_usize().max(1);
    if orders > 10 {
        return Err("v5 retains the decoder's maximum of 10 market orders per turn".into());
    }
    if turns
        .checked_mul(orders)
        .is_none_or(|count| count > MAX_ACTORS - 1)
    {
        return Err("configuration actor bound exceeds v5's 241 actors per farm".into());
    }
    let farms = [&game.farms[seat], &game.farms[1 - seat]];
    let private = &game.privates[seat];
    let actor_counts = [1 + farms[0].hands.len(), 1 + farms[1].hands.len()];
    if actor_counts.iter().any(|&count| count > MAX_ACTORS) {
        return Err("observed actor count exceeds v5's 241 actors per farm".into());
    }
    if private.inventories.len() != actor_counts[0] {
        return Err("v5 requires one observable inventory per own actor".into());
    }
    for farm in farms {
        if farm.tiles.len() != 10 || farm.tiles.iter().any(|row| row.len() != 10) {
            return Err("v5 requires complete 10x10 public farm tiles".into());
        }
        if positions(farm).any(|position| position.len() != 2) {
            return Err("v5 requires a public coordinate pair for every actor".into());
        }
    }
    if game.town.unlocked_shops.len() > 8 {
        return Err("Myolie v3 features require at most eight unlocked shops".into());
    }
    out.fill(0.0);
    let step = game.step as f64;
    let turns = turns as f64;
    let horizon = config.episode_steps.to_f64("episodeSteps")?.max(1.0);
    let day = config.turns_per_day.div_rem_usize(game.step).0 as f64;
    let empty = |farm: &Farm| {
        farm.tiles
            .iter()
            .flatten()
            .filter(|tile| tile.is_null())
            .count() as f64
    };
    let locked = |farm: &Farm| {
        farm.tiles
            .iter()
            .flatten()
            .filter(|tile| tile.as_str() == Some("LOCKED"))
            .count() as f64
    };
    let mut at = 0;
    for value in [
        step / horizon,
        (step % turns) / turns,
        day / (horizon / turns).max(1.0),
        farms[0].money / 200_000.0,
        farms[1].money / 200_000.0,
        actor_counts[0].min(16) as f64 / 16.0,
        actor_counts[1].min(16) as f64 / 16.0,
        farms[0].unlocked_quadrants.len() as f64 / 4.0,
        farms[1].unlocked_quadrants.len() as f64 / 4.0,
        empty(farms[0]) / 100.0,
        empty(farms[1]) / 100.0,
        locked(farms[0]) / 100.0,
        locked(farms[1]) / 100.0,
        total(&private.shed)? / 100.0,
        total(&private.seeds)? / 100.0,
        game.town.unlocked_shops.len() as f64 / 8.0,
    ] {
        push(out, &mut at, value);
    }
    for farm in farms {
        for tile in farm.tiles.iter().flatten() {
            for value in legacy_tile(tile, day)? {
                push(out, &mut at, value);
            }
        }
    }
    debug_assert_eq!(at, 616);
    for farm in farms {
        for position in positions(farm).take(16) {
            push(out, &mut at, position[0] as f64 / 10.0);
            push(out, &mut at, position[1] as f64 / 10.0);
        }
        at += (16 - (1 + farm.hands.len()).min(16)) * 2;
    }
    debug_assert_eq!(at, 680);
    for index in 0..16 {
        for item in ITEMS {
            let value = private
                .inventories
                .get(index)
                .map_or(0.0, |inventory| count(inventory, item));
            push(out, &mut at, value / 32.0);
        }
    }
    for item in CROP_NAMES {
        push(out, &mut at, count(&private.seeds, item) / 32.0);
    }
    for item in ITEMS {
        push(out, &mut at, count(&private.shed, item) / 100.0);
    }
    for (map, denominator) in [
        (&game.market.inventory, 10_000.0),
        (&game.market.prices, 250.0),
    ] {
        for item in PRODUCTS {
            let value = map.get(item).map_or(Ok(0.0), |value| number(value, item))?;
            push(out, &mut at, value / denominator);
        }
    }
    debug_assert_eq!(at, 907);
    for (position, shop) in game.town.unlocked_shops.iter().enumerate() {
        let index = SORTED_SHOPS
            .iter()
            .position(|name| *name == shop)
            .ok_or_else(|| format!("unknown shop type: {shop:?}"))?;
        out[960 + position * 8 + index] = 1.0;
    }
    out[AVAILABLE] = 1.0;
    out[AVAILABLE + 1] = actor_counts[0] as f32;
    out[AVAILABLE + 2] = actor_counts[1] as f32;
    at = TILE_OFFSET;
    for farm in farms {
        for tile in farm.tiles.iter().flatten() {
            maintenance(tile, &mut out[at..at + 14])?;
            at += 14;
        }
    }
    debug_assert_eq!(at, ACTOR_OFFSET);
    for farm in farms {
        for (index, position) in positions(farm).enumerate() {
            let offset = at + index * 3;
            out[offset] = 1.0;
            out[offset + 1] = (position[0] as f64) as f32;
            out[offset + 2] = (position[1] as f64) as f32;
        }
        at += MAX_ACTORS * 3;
    }
    debug_assert_eq!(at, INVENTORY_OFFSET);
    for inventory in &private.inventories {
        for item in ITEMS {
            push(out, &mut at, count(inventory, item));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use serde_json::json;

    fn fresh() -> Game {
        Game::new(Config::default(), 123, 2).unwrap()
    }
    fn row(game: &Game, seat: usize) -> Vec<f32> {
        let mut row = vec![f32::NAN; FEATURE_COUNT];
        encode(game, seat, &mut row).unwrap();
        row
    }
    fn plant() -> Value {
        json!({"kind":"PLANT","crop":"TOMATO","yield_units":3,
               "watered_today":true,"consecutive_unwatered":2,"planted_day":1,
               "max_lifespan_step":200,"fertilized_until_day":-1})
    }
    fn animal() -> Value {
        json!({"kind":"PASTURE","animal":"COW","yield_units":2,
               "placed_day":3,"consecutive_unfed":1,"fed_today":false,
               "cared_today":true,"fertilizer_available":true,"pending_care_bonus":4})
    }

    #[test]
    fn default_layout_and_padding_match_scalar_formulas() {
        let game = fresh();
        let output = row(&game, 0);
        assert_eq!(output.len(), 8165);
        assert_eq!(
            &output[..16],
            &[
                0., 0., 0., 0.015, 0.015, 0.0625, 0.0625, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.,
                0., 0.
            ]
        );
        assert_eq!(output[16], 0.05);
        assert_eq!(output[16 + 5 * 3], 0.0);
        assert!(output[907..1024].iter().all(|value| *value == 0.0));
        assert_eq!(&output[1024..1027], &[1., 1., 1.]);
        assert_eq!(&output[ACTOR_OFFSET..ACTOR_OFFSET + 3], &[1., 4., 4.]);
        assert!(output[INVENTORY_OFFSET..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn maintenance_and_ordered_shop_identity_are_separate() {
        let mut game = fresh();
        game.step = 239;
        game.farms[0].tiles[0][0] = plant();
        game.farms[1].tiles[0][0] = animal();
        game.town.unlocked_shops = vec!["PIZZA_SHOP".into(), "BAKERY".into(), "PIZZA_SHOP".into()];
        let output = row(&game, 0);
        assert_eq!(&output[16..19], &[0.25, (8.0_f64 / 30.0) as f32, 0.625]);
        assert_eq!(&output[316..319], &[0.65, 0., 0.4375]);
        assert_eq!(
            &output[TILE_OFFSET..TILE_OFFSET + 14],
            &[1., 0., 3., 1., 2., 1., 200., -1., 0., 0., 0., 0., 0., 0.]
        );
        assert_eq!(
            &output[TILE_OFFSET + 1400..TILE_OFFSET + 1414],
            &[0., 1., 2., 0., 0., 0., 0., 0., 3., 1., 0., 1., 1., 4.]
        );
        assert_eq!(output[960 + 5], 1.0);
        assert_eq!(output[968], 1.0);
        assert_eq!(output[976 + 5], 1.0);
        assert_eq!(output[960..1024].iter().sum::<f32>(), 3.0);
    }

    #[test]
    fn actor_extension_preserves_clipped_legacy_and_own_private_isolation() {
        let mut game = fresh();
        game.farms[0].hands = (0..240).map(|i| vec![i % 10, i / 10]).collect();
        game.privates[0].inventories = (0..241)
            .map(|i| Counts::from([("WHEAT".into(), i)]))
            .collect();
        let output = row(&game, 0);
        assert_eq!(output[5], 1.0);
        assert_eq!(output[1025], 241.0);
        assert_eq!(output[680 + 15 * 12], 15. / 32.);
        assert_eq!(output[INVENTORY_OFFSET + 240 * 12], 240.0);
        assert_eq!(
            &output[ACTOR_OFFSET + 240 * 3..ACTOR_OFFSET + 241 * 3],
            &[1., 9., 23.]
        );
        let other = row(&game, 1);
        assert_eq!(other[1025], 1.0);
        assert_eq!(other[1026], 241.0);
        assert!(other[INVENTORY_OFFSET..].iter().all(|value| *value == 0.0));
        game.privates[1].shed.insert("WHEAT".into(), 999);
        game.privates[1].inventories[0].insert("WHEAT".into(), 222);
        assert_eq!(row(&game, 0), output);
    }

    #[test]
    fn unsupported_shapes_and_incomplete_maintenance_fail() {
        let mut game = fresh();
        let mut output = vec![0.; FEATURE_COUNT];
        assert!(encode(&game, 2, &mut output).is_err());
        assert!(encode(&game, 0, &mut output[..FEATURE_COUNT - 1]).is_err());
        game.farms[0].tiles[0][0] = plant();
        game.farms[0].tiles[0][0]
            .as_object_mut()
            .unwrap()
            .remove("consecutive_unwatered");
        assert!(
            encode(&game, 0, &mut output)
                .unwrap_err()
                .contains("consecutive_unwatered")
        );
        game.farms[0].tiles[0][0] = plant();
        game.farms[0].tiles[0][0]["yield_units"] = Value::String("nan".into());
        assert!(
            encode(&game, 0, &mut output)
                .unwrap_err()
                .contains("nonfinite")
        );
        let mut game = fresh();
        game.config.board_size = 11;
        assert!(encode(&game, 0, &mut output).is_err());
        game.config.board_size = 10;
        game.config.turns_per_day = 25_i64.into();
        assert!(encode(&game, 0, &mut output).is_err());
        game.config.turns_per_day = 24_i64.into();
        game.config.max_market_orders_per_turn = 11_i64.into();
        assert!(encode(&game, 0, &mut output).is_err());
        let mut game = fresh();
        game.privates[0].inventories.clear();
        assert!(encode(&game, 0, &mut output).is_err());
        let mut game = fresh();
        game.farms[0].farmer.push(9);
        assert!(encode(&game, 0, &mut output).is_err());
        let mut game = fresh();
        game.town.unlocked_shops.push("UNKNOWN".into());
        assert!(encode(&game, 0, &mut output).is_err());
    }

    #[test]
    fn private_aggregate_bound_is_absolute_and_inclusive() {
        let mut game = fresh();
        let mut output = vec![0.; FEATURE_COUNT];
        game.privates[0].shed.insert("WHEAT".into(), 1_i64 << 53);
        encode(&game, 0, &mut output).unwrap();
        assert_eq!(output[13], ((1_u64 << 53) as f64 / 100.0) as f32);
        game.privates[0].shed.insert("CARROT".into(), 1);
        assert!(encode(&game, 0, &mut output).unwrap_err().contains("2^53"));
        game.privates[0].shed.insert("WHEAT".into(), -(1_i64 << 53));
        assert!(encode(&game, 0, &mut output).unwrap_err().contains("2^53"));
        game.privates[0].shed.insert("WHEAT".into(), i64::MIN);
        assert!(encode(&game, 0, &mut output).unwrap_err().contains("2^53"));
        // An unsupported opponent private state remains unobserved by this seat.
        encode(&game, 1, &mut output).unwrap();
    }
}
