//! JAT-05 option library: the lifecycle rules behind `CROP_CYCLE`, `ANIMAL_CYCLE`
//! (and their hired forms), plus the seat's shed-sale rule.
//!
//! Every rule is a read-only function of the live engine state that returns this
//! step's official verb for one unit, the orders the lifecycle needs, whether the
//! lifecycle is finished and why, and (for the animal rule, the one rule that leaves
//! its tile) the tile the unit should stand on.  The Python compiler owns movement,
//! the reserved market slots and the termination bookkeeping; the observation-based
//! mirror in `scripts/ja_action.py` must agree with these functions verb for verb.

use crate::{animal, crop, map_bool, map_i64, map_str, shed_access_tiles, Farm, Game, PrivateState, PRODUCTS};
use serde_json::{json, Value};

const TURNS_PER_DAY: usize = 24;
/// Wheat taken from the shed per trip, and bought per order, by an animal line.
pub const WHEAT_PER_TRIP: i64 = 5;

pub const REASON_HARVESTED: &str = "harvested";
pub const REASON_STUCK_TILE: &str = "stuck_tile";
pub const REASON_STUCK_BROKE: &str = "stuck_broke";

fn pass() -> Value {
    json!(["PASS"])
}

fn seat_state<'a>(game: &'a Game, seat: usize, rule: &str) -> Result<(&'a Farm, &'a PrivateState), String> {
    let farm = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("{rule} seat {seat} is missing its farm"))?;
    let private = game
        .privates()
        .get(seat)
        .ok_or_else(|| format!("{rule} seat {seat} is missing its private state"))?;
    Ok((farm, private))
}

fn current_day(game: &Game) -> Result<i64, String> {
    i64::try_from(game.step_index() / TURNS_PER_DAY)
        .map_err(|_| "library day index exceeds i64".to_string())
}

fn tile_at<'a>(farm: &'a Farm, x: i64, y: i64, rule: &str) -> Result<&'a Value, String> {
    usize::try_from(y)
        .ok()
        .and_then(|y| farm.tiles.get(y))
        .and_then(|row| usize::try_from(x).ok().and_then(|x| row.get(x)))
        .ok_or_else(|| format!("{rule} tile ({x}, {y}) is off the board"))
}

fn finished(reason: &str) -> Value {
    json!({"verb": pass(), "orders": [], "done": true, "reason": reason})
}

/// `CROP_CYCLE` executor at an assigned tile for one crop.  Weeds are dug, an empty
/// tile is planted (one seed bought when none is held and one is affordable), the
/// crop is watered once a day and harvested when ripe: one-time crops on their
/// `max_yield_day` (the kit starter's rule, the last day before the plant dies),
/// ongoing crops whenever they hold yield past their first yield day.  A one-time
/// crop's lifecycle ends at its harvest; an ongoing crop stands until the tile holds
/// something else or a needed seed is unaffordable.  Nothing here sells: the seat's
/// `sweep_step` sells whatever reaches the shed.
pub fn crop_cycle_step(game: &Game, seat: usize, x: i64, y: i64, name: &str) -> Result<Value, String> {
    let data = crop(name).ok_or_else(|| format!("crop cycle has no crop {name:?}"))?;
    let (farm, private) = seat_state(game, seat, "crop cycle")?;
    let day = current_day(game)?;
    let tile = tile_at(farm, x, y, "crop cycle")?;
    let seeds = private.seeds.get(name).copied().unwrap_or(0);
    let affordable = farm.money >= data.seed as f64;
    let mut orders = Vec::new();
    let mut verb = pass();
    let mut done = false;
    let mut reason = Value::Null;
    if tile.is_null() {
        if seeds > 0 {
            verb = json!(["PLANT", name]);
        } else if affordable {
            orders.push(json!(["BUY_SEED", name, 1]));
        } else {
            done = true;
            reason = Value::from(REASON_STUCK_BROKE);
        }
    } else if let Some(object) = tile.as_object() {
        let kind = map_str(object, "kind");
        if kind == Some("WEED") {
            verb = json!(["DIG"]);
        } else if kind == Some("PLANT") && map_str(object, "crop") == Some(name) {
            let age = day - map_i64(object, "planted_day");
            let watered = map_bool(object, "watered_today");
            let ripe = if data.ongoing {
                map_i64(object, "yield_units") > 0 && age >= data.first_yield_day
            } else {
                age >= data.max_yield_day
            };
            if ripe && (data.ongoing || watered) {
                // a one-time crop is watered once more on its last day (one more
                // unit inside the yield window) and harvested on the next step; the
                // kit starter harvests unwatered and leaves that unit on the tile
                verb = json!(["HARVEST"]);
                if !data.ongoing {
                    done = true;
                    reason = Value::from(REASON_HARVESTED);
                }
            } else if !watered {
                verb = json!(["WATER"]);
            }
        } else {
            done = true;
            reason = Value::from(REASON_STUCK_TILE);
        }
    } else {
        // "LOCKED"
        done = true;
        reason = Value::from(REASON_STUCK_TILE);
    }
    Ok(json!({
        "verb": verb,
        "orders": orders,
        "done": done,
        "reason": reason,
        "crop": name,
        "seeds": seeds,
        "seed_affordable": affordable,
    }))
}

fn nearest_shed(at: (i64, i64), board_size: usize) -> (i64, i64) {
    let mut best = None;
    for tile in shed_access_tiles(board_size) {
        let distance = (tile.0 - at.0).abs() + (tile.1 - at.1).abs();
        match best {
            Some((_, best_distance)) if best_distance <= distance => {}
            _ => best = Some((tile, distance)),
        }
    }
    best.map(|(tile, _)| tile).unwrap_or((0, 0))
}

fn count(map: &crate::Counts, item: &str) -> i64 {
    map.get(item).copied().unwrap_or(0)
}

/// `ANIMAL_CYCLE` executor for one animal at an assigned tile, for the unit `unit`
/// standing at `at`.  Returns `move_to`, the tile the unit should stand on this step
/// (the target, or the nearest shed tile for a pickup), beside the verb to use once
/// there.  Weeds are dug; an empty tile gets the animal's structure; an empty structure
/// gets the animal (bought as an order, picked up at the shed, placed); a placed
/// animal is fed once a day with wheat from the hand (else fetched from the shed, else
/// bought as an order), harvested when it holds product, cared once a day, and its
/// fertilizer collected.  The line stands until the tile holds something else or the
/// animal is unaffordable and none is held.
pub fn animal_cycle_step(
    game: &Game,
    seat: usize,
    x: i64,
    y: i64,
    name: &str,
    unit: usize,
    at: (i64, i64),
) -> Result<Value, String> {
    let data = animal(name).ok_or_else(|| format!("animal cycle has no animal {name:?}"))?;
    let (farm, private) = seat_state(game, seat, "animal cycle")?;
    let tile = tile_at(farm, x, y, "animal cycle")?;
    let board_size = farm.tiles.len();
    let empty = crate::Counts::new();
    let inventory = private.inventories.get(unit).unwrap_or(&empty);
    let carried_animal = count(inventory, name);
    let carried_wheat = count(inventory, "WHEAT");
    let shed_animal = count(&private.shed, name);
    let shed_wheat = count(&private.shed, "WHEAT");
    let affordable = farm.money >= data.cost as f64;
    let target = (x, y);
    let shed = nearest_shed(at, board_size);
    let mut orders = Vec::new();

    // the animal itself: needed unless one is on the tile, in the hand or in the shed
    let placed = tile
        .as_object()
        .is_some_and(|object| map_str(object, "animal") == Some(name));
    let structure_ok = tile
        .as_object()
        .is_some_and(|object| map_str(object, "kind") == Some(data.structure));
    let foreign = match tile {
        Value::Null => false,
        Value::Object(object) => {
            let kind = map_str(object, "kind");
            if kind == Some("WEED") {
                false
            } else if kind == Some(data.structure) {
                object.contains_key("animal") && !placed
            } else {
                true
            }
        }
        _ => true, // "LOCKED"
    };
    if foreign {
        return Ok(finished(REASON_STUCK_TILE));
    }
    let held = placed || carried_animal > 0 || shed_animal > 0;
    if !held {
        if affordable {
            orders.push(json!(["BUY_ANIMAL", name, 1]));
        } else {
            return Ok(finished(REASON_STUCK_BROKE));
        }
    }

    let (verb, move_to): (Value, (i64, i64)) = if tile.is_null() {
        (json!([if data.structure == "COOP" { "BUILD_COOP" } else { "BUILD_PASTURE" }]), target)
    } else if tile.as_object().is_some_and(|object| map_str(object, "kind") == Some("WEED")) {
        (json!(["DIG"]), target)
    } else if placed {
        let object = tile.as_object().expect("placed animal tile is an object");
        let fed = map_bool(object, "fed_today");
        let wants_wheat = !fed && carried_wheat == 0;
        if wants_wheat && shed_wheat == 0 {
            orders.push(json!(["BUY_PRODUCT", "WHEAT", WHEAT_PER_TRIP]));
        }
        if !fed && carried_wheat > 0 {
            (json!(["FEED"]), target)
        } else if wants_wheat && shed_wheat > 0 {
            // Acquire available feed before optional animal work.
            (json!(["PICKUP", "WHEAT", shed_wheat.min(WHEAT_PER_TRIP)]), shed)
        } else if map_i64(object, "yield_units") > 0 {
            (json!(["HARVEST"]), target)
        } else if !map_bool(object, "cared_today") {
            (json!(["CARE"]), target)
        } else if map_bool(object, "fertilizer_available") {
            (json!(["COLLECT_FERTILIZER"]), target)
        } else {
            (pass(), target)
        }
    } else if structure_ok {
        if carried_animal > 0 {
            (json!(["PLACE", name]), target)
        } else if shed_animal > 0 {
            (json!(["PICKUP", name, 1]), shed)
        } else {
            // the order is in flight: wait at the shed, where the animal will arrive
            (pass(), shed)
        }
    } else {
        return Ok(finished(REASON_STUCK_TILE));
    };
    Ok(json!({
        "verb": verb,
        "orders": orders,
        "done": false,
        "reason": Value::Null,
        "move_to": [move_to.0, move_to.1],
        "animal": name,
    }))
}

/// The seat's shed-sale rule: one `SELL` per positive-stock product in the shed,
/// except products in `hold` (the policy's SELL / HOLD gate from the last event
/// boundary). Plant and animal presence never overrides the policy's sale intent.
pub fn sweep_step(game: &Game, seat: usize, hold: &[String]) -> Result<Value, String> {
    let (_, private) = seat_state(game, seat, "sweep")?;
    let orders: Vec<Value> = PRODUCTS
        .iter()
        .filter_map(|item| {
            let held = count(&private.shed, item);
            if held <= 0 || hold.iter().any(|name| name == item) {
                None
            } else {
                Some(json!(["SELL", item, held]))
            }
        })
        .collect();
    Ok(json!({"verb": pass(), "orders": orders, "done": false, "reason": Value::Null}))
}

/// Fertilizer taken from the shed per trip by a fertilized crop line.
pub const FERTILIZER_PER_TRIP: i64 = 3;

/// `FERTILIZED_CROP_CYCLE` executor (JAT-06): the crop rule plus fertilizer.  While the
/// plant is in its bonus window and its last bag has lapsed, the unit fertilizes from
/// a bag in hand, else fetches one at the shed (`move_to` says where to stand), else
/// orders one (`BUY_PRODUCT FERTILIZER 1`) and keeps tending.  One-time crops take the
/// bag from the first day of their yield window, ongoing crops from the day before
/// their first production; HARVEST and WATER come before the bag, and the shed trip
/// only when nothing else is due at the tile.
pub fn fertilized_crop_cycle_step(
    game: &Game,
    seat: usize,
    x: i64,
    y: i64,
    name: &str,
    unit: usize,
    at: (i64, i64),
) -> Result<Value, String> {
    let data = crop(name).ok_or_else(|| format!("fertilized crop cycle has no crop {name:?}"))?;
    let mut base = crop_cycle_step(game, seat, x, y, name)?;
    let target = (x, y);
    if base["done"].as_bool() == Some(true) {
        base["move_to"] = json!([target.0, target.1]);
        return Ok(base);
    }
    let (farm, private) = seat_state(game, seat, "fertilized crop cycle")?;
    let day = current_day(game)?;
    let tile = tile_at(farm, x, y, "fertilized crop cycle")?;
    let empty = crate::Counts::new();
    let inventory = private.inventories.get(unit).unwrap_or(&empty);
    let carried = count(inventory, "FERTILIZER");
    let in_shed = count(&private.shed, "FERTILIZER");
    let mut orders: Vec<Value> = base["orders"].as_array().cloned().unwrap_or_default();
    let mut verb = base["verb"].clone();
    let mut move_to = target;
    let mut wants_bag = false;
    if let Some(object) = tile.as_object()
        && map_str(object, "kind") == Some("PLANT")
        && map_str(object, "crop") == Some(name)
    {
        let age = day - map_i64(object, "planted_day");
        let lapsed = map_i64(object, "fertilized_until_day") < day;
        let in_window = if data.ongoing {
            age >= data.first_yield_day - 1
        } else {
            (data.max_yield_day + 1) / 2 <= age && age <= data.max_yield_day
        };
        wants_bag = lapsed && in_window;
        if wants_bag && carried == 0 && in_shed == 0 {
            orders.push(json!(["BUY_PRODUCT", "FERTILIZER", 1]));
        }
        if wants_bag && verb == pass() {
            if carried > 0 {
                verb = json!(["FERTILIZE"]);
            } else if in_shed > 0 {
                verb = json!(["PICKUP", "FERTILIZER", in_shed.min(FERTILIZER_PER_TRIP)]);
                move_to = nearest_shed(at, farm.tiles.len());
            }
        }
    }
    base["verb"] = verb;
    base["orders"] = Value::Array(orders);
    base["move_to"] = json!([move_to.0, move_to.1]);
    base["wants_fertilizer"] = Value::from(wants_bag);
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    fn game_with(config: Config) -> Game {
        Game::new(config, 7, 2).expect("default two-player game")
    }

    #[test]
    fn animal_resupply_shared_saved_state_cases() {
        // Explicit own-state fixtures only: no random reset or engine stepping.
        let fixture: Value = serde_json::from_str(include_str!(
            "../../fixtures/jat-animal-resupply.json"
        )).unwrap();
        for name in ["SHEEP", "COW", "GOOSE"] {
            for case in fixture["cases"].as_array().unwrap() {
                let mut farm = fixture["farm"].clone();
                let mut private = fixture["private"].clone();
                let tile = farm["tiles"][0][1].as_object_mut().unwrap();
                tile.insert("animal".into(), json!(name));
                tile.insert("kind".into(), json!(if name == "GOOSE" { "COOP" } else { "PASTURE" }));
                if let Some(patch) = case["tile"].as_object() {
                    tile.extend(patch.clone());
                }
                private["inventories"][0] = case.get("inventory").cloned().unwrap_or(json!({}));
                if let Some(patch) = case["shed"].as_object() {
                    private["shed"].as_object_mut().unwrap().extend(patch.clone());
                }
                let at = case.get("at").cloned().unwrap_or(json!([1, 0]));
                farm["farmer"] = at.clone();
                let header: crate::TraceHeader = serde_json::from_value(json!({
                    "format": crate::TRACE_FORMAT, "seed": 0, "configuration": Config::default(),
                    "shop_schedule": [], "rng_schedule": [], "terminal_banks": [], "transitions": 0,
                    "initial": {"public": {"step": 55, "day": 2, "hour": 7,
                        "farms": [farm.clone(), farm], "market": {"inventory": {}, "prices": {}},
                        "town": {"unlocked_shops": []}}, "privates": [private.clone(), private]}
                })).unwrap();
                let game = Game::from_header(&header).unwrap();
                let before = serde_json::to_value(game.snapshot()).unwrap();
                let result = animal_cycle_step(&game, 0, 1, 0, name, 0,
                    (at[0].as_i64().unwrap(), at[1].as_i64().unwrap())).unwrap();
                assert_eq!(result, json!({
                    "verb": case["verb"], "move_to": case.get("move_to").cloned().unwrap_or(json!([1, 0])),
                    "orders": case.get("orders").cloned().unwrap_or(json!([])),
                    "done": false, "reason": null, "animal": name
                }), "{} {name}", case["id"]);
                assert_eq!(serde_json::to_value(game.snapshot()).unwrap(), before);
            }
        }
    }

    fn quiet_config() -> Config {
        Config {
            weed_spawn_chance: Value::from(0.0),
            ..Config::default()
        }
    }

    fn pass_action() -> Value {
        json!({"farmer": ["PASS"], "hands": [], "market": []})
    }

    fn step_toward(from: (i64, i64), to: (i64, i64)) -> &'static str {
        if from.0 < to.0 {
            "EAST"
        } else if from.0 > to.0 {
            "WEST"
        } else if from.1 < to.1 {
            "SOUTH"
        } else {
            "NORTH"
        }
    }

    fn farmer_at(game: &Game, seat: usize) -> (i64, i64) {
        let farmer = &game.farms()[seat].farmer;
        (farmer[0], farmer[1])
    }

    /// Drive seat 0's farmer with one rule for `steps` steps, the way the compiler
    /// does: walk toward `move_to` (or the target) one step at a time, act on arrival,
    /// put every order and the sweep's orders in the market.  Returns the verbs used.
    fn drive(game: &mut Game, steps: usize, target: (i64, i64), hold: &[String], mut rule: impl FnMut(&Game, (i64, i64)) -> Value) -> Vec<Value> {
        let mut verbs = Vec::new();
        for _ in 0..steps {
            let at = farmer_at(game, 0);
            let result = rule(game, at);
            let sweep = sweep_step(game, 0, hold).unwrap();
            let mut market: Vec<Value> = result["orders"].as_array().cloned().unwrap_or_default();
            market.extend(sweep["orders"].as_array().cloned().unwrap_or_default());
            let goal = result
                .get("move_to")
                .and_then(Value::as_array)
                .map(|xy| (xy[0].as_i64().unwrap(), xy[1].as_i64().unwrap()))
                .unwrap_or(target);
            let verb = if at != goal {
                json!([step_toward(at, goal)])
            } else {
                result["verb"].clone()
            };
            verbs.push(verb.clone());
            game.step(&[json!({"farmer": verb, "hands": [], "market": market}), pass_action()])
                .unwrap();
            if result["done"].as_bool() == Some(true) {
                break;
            }
        }
        verbs
    }

    #[test]
    fn a_wheat_cycle_buys_plants_waters_and_harvests_on_day_four_then_ends() {
        let mut game = game_with(quiet_config());
        let target = farmer_at(&game, 0);
        let mut results = Vec::new();
        let verbs = drive(&mut game, 200, target, &[], |game, _| {
            let result = crop_cycle_step(game, 0, target.0, target.1, "WHEAT").unwrap();
            results.push(result.clone());
            result
        });
        // day 0: buy a seed, plant, water; days 1-4: water; day 4 (step 97): harvest
        assert_eq!(verbs[0], json!(["PASS"]));
        assert_eq!(results[0]["orders"], json!([["BUY_SEED", "WHEAT", 1]]));
        assert_eq!(verbs[1], json!(["PLANT", "WHEAT"]));
        assert_eq!(verbs[2], json!(["WATER"]));
        assert_eq!(verbs[24], json!(["WATER"]));
        assert_eq!(verbs[96], json!(["WATER"]));
        assert_eq!(verbs[97], json!(["HARVEST"]));
        assert_eq!(verbs.len(), 98, "the one-time crop's lifecycle ends at its harvest");
        let last = results.last().unwrap();
        assert_eq!(last["done"], json!(true));
        assert_eq!(last["reason"], json!(REASON_HARVESTED));
        // watered on days 2, 3 and 4 inside the yield window [2, 4]: 1 + 3 units
        assert_eq!(game.privates()[0].inventories[0]["WHEAT"], 4);
    }

    #[test]
    fn a_tomato_line_harvests_every_production_and_stands() {
        let mut game = game_with(quiet_config());
        let target = farmer_at(&game, 0);
        let verbs = drive(&mut game, 24 * 13, target, &[], |game, _| {
            crop_cycle_step(game, 0, target.0, target.1, "TOMATO").unwrap()
        });
        let harvests = verbs.iter().filter(|verb| **verb == json!(["HARVEST"])).count();
        // first yield at day 8 (planted day 0), then daily: days 8, 9, 10, 11
        assert_eq!(harvests, 4);
        assert_eq!(verbs.len(), 24 * 13, "an ongoing crop's line does not end on its own");
        // every harvest reached the shed at the end of its day and was sold next
        // morning (four tomatoes near the base price of 60); the plant decayed to a
        // weed after its last production and the standing line dug and replanted it
        // (a second seed), so two seeds were bought
        assert!(game.farms()[0].money > 3_000.0 - 100.0 + 4.0 * 55.0);
        assert!(verbs.contains(&json!(["DIG"])));
        assert_eq!(game.privates()[0].shed.get("TOMATO").copied().unwrap_or(0), 0);
    }

    #[test]
    fn a_goose_line_builds_buys_places_feeds_cares_and_collects_eggs() {
        let mut game = game_with(quiet_config());
        let start = farmer_at(&game, 0);
        let target = (start.0 - 2, start.1 - 2); // (2, 2), two steps from the shed tile
        let mut results = Vec::new();
        let verbs = drive(&mut game, 24 * 7, target, &["WHEAT".to_string()], |game, at| {
            let result = animal_cycle_step(game, 0, target.0, target.1, "GOOSE", 0, at).unwrap();
            results.push(result.clone());
            result
        });
        assert_eq!(results[0]["orders"], json!([["BUY_ANIMAL", "GOOSE", 1]]));
        assert!(verbs.contains(&json!(["BUILD_COOP"])));
        assert!(verbs.contains(&json!(["PICKUP", "GOOSE", 1])));
        assert!(verbs.contains(&json!(["PLACE", "GOOSE"])));
        assert!(verbs.contains(&json!(["PICKUP", "WHEAT", 5])));
        assert!(verbs.iter().filter(|verb| **verb == json!(["FEED"])).count() >= 5);
        assert!(verbs.iter().filter(|verb| **verb == json!(["CARE"])).count() >= 5);
        assert!(verbs.contains(&json!(["HARVEST"])), "eggs from day 4 on");
        assert!(verbs.contains(&json!(["COLLECT_FERTILIZER"])));
        let tile = &game.farms()[0].tiles[target.1 as usize][target.0 as usize];
        assert_eq!(tile["animal"], json!("GOOSE"));
        assert!(results.iter().all(|result| result["done"] == json!(false)));
        // This lifecycle fixture explicitly holds wheat for feeding.
        assert!(results.iter().any(|result| result["orders"]
            .as_array()
            .unwrap()
            .iter()
            .any(|order| order[0] == json!("BUY_PRODUCT"))));
        assert!(sweep_step(&game, 0, &["WHEAT".to_string()]).unwrap()["orders"]
            .as_array()
            .unwrap()
            .iter()
            .all(|order| order[1] != json!("WHEAT")));
    }

    #[test]
    fn a_crop_line_on_a_foreign_plant_or_a_locked_tile_is_stuck() {
        let game = game_with(quiet_config());
        let locked = crop_cycle_step(&game, 0, 9, 9, "MELON").unwrap();
        assert_eq!(locked["done"], json!(true));
        assert_eq!(locked["reason"], json!(REASON_STUCK_TILE));
        let mut planted = game_with(quiet_config());
        let start = farmer_at(&planted, 0);
        planted
            .step(&[
                json!({"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "CARROT", 1]]}),
                pass_action(),
            ])
            .unwrap();
        planted
            .step(&[json!({"farmer": ["PLANT", "CARROT"], "hands": [], "market": []}), pass_action()])
            .unwrap();
        let wheat = crop_cycle_step(&planted, 0, start.0, start.1, "WHEAT").unwrap();
        assert_eq!(wheat["reason"], json!(REASON_STUCK_TILE));
        let carrot = crop_cycle_step(&planted, 0, start.0, start.1, "CARROT").unwrap();
        assert_eq!(carrot["verb"], json!(["WATER"]));
        assert_eq!(carrot["done"], json!(false));
    }

    #[test]
    fn a_broke_seat_cannot_start_a_seed_or_an_animal() {
        let game = game_with(Config {
            starting_money: 5_i64.into(),
            ..quiet_config()
        });
        let start = farmer_at(&game, 0);
        let seed = crop_cycle_step(&game, 0, start.0, start.1, "WHEAT").unwrap();
        assert_eq!(seed["reason"], json!(REASON_STUCK_BROKE));
        let animal = animal_cycle_step(&game, 0, start.0, start.1, "SHEEP", 0, start).unwrap();
        assert_eq!(animal["reason"], json!(REASON_STUCK_BROKE));
    }

    #[test]
    fn a_fertilized_tomato_line_buys_a_bag_fetches_it_and_fertilizes_before_the_first_production() {
        let mut game = game_with(quiet_config());
        let target = farmer_at(&game, 0);
        let mut results = Vec::new();
        let verbs = drive(&mut game, 24 * 9, target, &["FERTILIZER".to_string()], |game, at| {
            let result = fertilized_crop_cycle_step(game, 0, target.0, target.1, "TOMATO", 0, at).unwrap();
            results.push(result.clone());
            result
        });
        // planted on day 0; the bag is wanted from day 7 (first production at day 8):
        // ordered, picked up at the shed, applied, and the plant is fertilized
        let ordered = results.iter().any(|r| r["orders"].as_array().unwrap().iter().any(|o| o[0] == json!("BUY_PRODUCT") && o[1] == json!("FERTILIZER")));
        assert!(ordered, "a bag was ordered");
        assert!(verbs.iter().any(|v| v[0] == json!("PICKUP") && v[1] == json!("FERTILIZER")), "fetched at the shed");
        assert!(verbs.contains(&json!(["FERTILIZE"])), "applied at the tile");
        let tile = &game.farms()[0].tiles[target.1 as usize][target.0 as usize];
        assert!(tile["fertilized_until_day"].as_i64().unwrap() >= 7, "{tile}");
        assert!(!verbs[..24 * 7].iter().any(|v| v[0] == json!("PICKUP")), "no trip before the window");
        assert!(results.iter().all(|r| r["done"] == json!(false)));
        // This lifecycle fixture explicitly holds fertilizer.
        assert!(sweep_step(&game, 0, &["FERTILIZER".to_string()]).unwrap()["orders"].as_array().unwrap().iter().all(|o| o[1] != json!("FERTILIZER")));
    }

    #[test]
    fn the_sweep_keeps_the_products_the_policy_holds() {
        let mut game = game_with(quiet_config());
        game.step(&[
            json!({"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 2]]}),
            pass_action(),
        ])
        .unwrap();
        let held = ["WHEAT".to_string()];
        assert_eq!(sweep_step(&game, 0, &held).unwrap()["orders"], json!([]));
        assert_eq!(sweep_step(&game, 0, &[]).unwrap()["orders"], json!([["SELL", "WHEAT", 2]]));
    }

    #[test]
    fn the_sweep_sells_every_shed_product_even_with_animals() {
        let mut game = game_with(quiet_config());
        game.step(&[
            json!({"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 2], ["BUY_ANIMAL", "COW", 1]]}),
            pass_action(),
        ])
        .unwrap();
        let sweep = sweep_step(&game, 0, &[]).unwrap();
        assert_eq!(sweep["orders"], json!([["SELL", "WHEAT", 2]]), "a cow does not override SELL");
        let mut plain = game_with(quiet_config());
        plain
            .step(&[
                json!({"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 2], ["BUY_PRODUCT", "FERTILIZER", 1]]}),
                pass_action(),
            ])
            .unwrap();
        let sweep = sweep_step(&plain, 0, &[]).unwrap();
        assert_eq!(sweep["orders"], json!([["SELL", "WHEAT", 2], ["SELL", "FERTILIZER", 1]]));
    }
}

#[cfg(test)]
#[path = "sell_intent_tests.rs"]
mod sell_intent_tests;
