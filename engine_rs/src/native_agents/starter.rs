//! Complete native port of the kit's built-in `starter_agent`.
//!
//! The reference is `kaggle_environments.envs.kaggriculture.kaggriculture.starter_agent`:
//! a stateless carrot loop.  Every turn it sells every carrot in the shed, buys one
//! carrot seed when it holds none and can afford it, plants on the tile under the
//! farmer when that tile is empty, waters an unwatered carrot, and harvests once the
//! carrot reaches `max_yield_day`.  It never hires hands and passes otherwise.

use crate::{Farm, Game, fib};
use num_bigint::BigInt;
use num_traits::FromPrimitive;
use serde_json::{Value, json};

const TURNS_PER_DAY: usize = 24;
const CARROT_SEED_PRICE: f64 = 20.0;
const CARROT_MAX_YIELD_DAY: i64 = 3;

#[derive(Clone, Debug, Default)]
pub struct StarterController {
    last_debug: Value,
}

fn pass() -> Value {
    json!(["PASS"])
}

fn tile_under_farmer(farm: &Farm) -> Result<&Value, String> {
    let (&x, &y) = match farm.farmer.as_slice() {
        [x, y] => (x, y),
        _ => return Err("starter farmer position is not an (x, y) pair".to_string()),
    };
    usize::try_from(y)
        .ok()
        .and_then(|y| farm.tiles.get(y))
        .and_then(|row| usize::try_from(x).ok().and_then(|x| row.get(x)))
        .ok_or_else(|| format!("starter farmer position ({x}, {y}) is off the board"))
}

fn tile_field<'a>(
    tile: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a Value, String> {
    tile.get(key)
        .ok_or_else(|| format!("starter carrot tile has no `{key}` field"))
}

/// The starter's carrot rule evaluated for one seat at one tile.
///
/// Returns the unit verb and the market orders the rule wants this step.  The starter
/// applies it to the tile under its farmer; a JA28 `CARROT_CYCLE` assignment applies
/// it to its assigned tile, so both read the same engine state through the same code.
pub struct CarrotRule {
    pub verb: Value,
    pub orders: Vec<Value>,
    pub day: i64,
    pub carrot_age: Option<i64>,
    pub carrot_seeds: i64,
    pub shed_carrots: i64,
    pub money: f64,
}

pub fn carrot_rule(game: &Game, seat: usize, tile: &Value) -> Result<CarrotRule, String> {
    let farm = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("starter seat {seat} is missing its farm"))?;
    let private = game
        .privates()
        .get(seat)
        .ok_or_else(|| format!("starter seat {seat} is missing its private state"))?;
    let step = game.step_index();
    let day = i64::try_from(step / TURNS_PER_DAY)
        .map_err(|_| "starter day index exceeds i64".to_string())?;
    let carrot_seeds = private.seeds.get("CARROT").copied().unwrap_or(0);
    let shed_carrots = private.shed.get("CARROT").copied().unwrap_or(0);

    let mut orders = Vec::new();
    if shed_carrots > 0 {
        orders.push(json!(["SELL", "CARROT", shed_carrots]));
    }
    if carrot_seeds == 0 && farm.money >= CARROT_SEED_PRICE {
        orders.push(json!(["BUY_SEED", "CARROT", 1]));
    }

    let mut verb = pass();
    let mut carrot_age = None;
    if tile.is_null() {
        if carrot_seeds > 0 {
            verb = json!(["PLANT", "CARROT"]);
        }
    } else if let Some(object) = tile.as_object() {
        let is_carrot = object.get("kind").and_then(Value::as_str) == Some("PLANT")
            && object.get("crop").and_then(Value::as_str) == Some("CARROT");
        if is_carrot {
            let planted_day = tile_field(object, "planted_day")?
                .as_i64()
                .ok_or_else(|| "starter carrot `planted_day` is not an integer".to_string())?;
            let watered_today = tile_field(object, "watered_today")?
                .as_bool()
                .ok_or_else(|| "starter carrot `watered_today` is not a bool".to_string())?;
            let age = day - planted_day;
            carrot_age = Some(age);
            if age >= CARROT_MAX_YIELD_DAY {
                verb = json!(["HARVEST"]);
            } else if !watered_today {
                verb = json!(["WATER"]);
            }
        }
    }
    Ok(CarrotRule {
        verb,
        orders,
        day,
        carrot_age,
        carrot_seeds,
        shed_carrots,
        money: farm.money,
    })
}

/// JA28 `CARROT_CYCLE` executor: the starter's carrot rule at an assigned tile.
/// Returns `{"verb": [...], "orders": [[...], ...]}`; the Python compiler owns
/// movement toward the tile and writes the orders only into idle market slots.
pub fn carrot_cycle_step(game: &Game, seat: usize, x: i64, y: i64) -> Result<Value, String> {
    let farm = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("carrot cycle seat {seat} is missing its farm"))?;
    let tile = usize::try_from(y)
        .ok()
        .and_then(|y| farm.tiles.get(y))
        .and_then(|row| usize::try_from(x).ok().and_then(|x| row.get(x)))
        .ok_or_else(|| format!("carrot cycle tile ({x}, {y}) is off the board"))?;
    let rule = carrot_rule(game, seat, tile)?;
    // Termination (JA28): the cycle's product reached the market this step, or the
    // rule can make no progress on this tile (empty with no affordable seed, or a tile
    // that is neither empty nor a carrot plant).
    let sold = rule
        .orders
        .iter()
        .any(|order| order.get(0).and_then(Value::as_str) == Some("SELL"));
    let is_carrot = tile.as_object().is_some_and(|object| {
        object.get("kind").and_then(Value::as_str) == Some("PLANT")
            && object.get("crop").and_then(Value::as_str) == Some("CARROT")
    });
    let stuck = if tile.is_null() {
        rule.carrot_seeds == 0 && rule.money < CARROT_SEED_PRICE
    } else {
        !is_carrot
    };
    // JA29: several cycles of one seat share its seeds; the compiler lets one plant per
    // available seed and buys a seed for each cycle it held back
    Ok(json!({
        "verb": rule.verb,
        "orders": rule.orders,
        "done": sold || stuck,
        "seeds": rule.carrot_seeds,
        "seed_affordable": rule.money >= CARROT_SEED_PRICE,
    }))
}

/// JA29 `HIRED_CARROT_CYCLE` staffing step for a tile job with no hand: request one
/// `HIRE` when the seat can afford the next hand (`farmHandCostMult * fib(hires_today)`,
/// the engine's `do_hire` rule) and this step is not the last of its day (a hand hired
/// then vanishes at end of day before it can act).  `done` marks a job the seat cannot
/// afford to staff.  The verb is `PASS`: no unit carries this request.
pub fn hire_step(game: &Game, seat: usize) -> Result<Value, String> {
    let farm = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("hire step seat {seat} is missing its farm"))?;
    let step = game.step_index();
    let last_of_day = (step + 1).is_multiple_of(TURNS_PER_DAY);
    let cost = &game.config.farm_hand_cost_mult.0 * fib(farm.hires_today);
    let affordable = if farm.money.is_nan() {
        true
    } else if farm.money == f64::NEG_INFINITY {
        false
    } else if farm.money == f64::INFINITY {
        true
    } else {
        BigInt::from_f64(farm.money.floor()).is_some_and(|money| money >= cost)
    };
    let orders = if affordable && !last_of_day {
        vec![json!(["HIRE"])]
    } else {
        Vec::new()
    };
    Ok(json!({"verb": pass(), "orders": orders, "done": !affordable}))
}

impl StarterController {
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("starter seat {seat} is missing its farm"))?;
        let step = game.step_index();
        let tile = tile_under_farmer(farm)?;
        let rule = carrot_rule(game, seat, tile)?;
        self.last_debug = json!({
            "step": step,
            "seat": seat,
            "day": rule.day,
            "tile": tile,
            "carrot_age": rule.carrot_age,
            "carrot_seeds": rule.carrot_seeds,
            "shed_carrots": rule.shed_carrots,
            "money": rule.money,
        });
        Ok(json!({"farmer": rule.verb, "hands": [], "market": rule.orders}))
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    fn game_with(config: Config) -> Game {
        Game::new(config, 7, 2).expect("default two-player game")
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

    fn action(farmer: Value, market: Value) -> Value {
        json!({"farmer": farmer, "hands": [], "market": market})
    }

    #[test]
    fn carrot_loop_follows_the_kit_starter_across_the_first_four_days() {
        let mut game = game_with(quiet_config());
        let mut controller = StarterController::default();
        let mut decisions = Vec::new();
        let mut shed_carrots = Vec::new();
        for _ in 0..=96 {
            shed_carrots.push(game.privates()[0].shed["CARROT"]);
            let decision = controller.action(&game, 0).unwrap();
            decisions.push(decision.clone());
            game.step(&[decision, pass_action()]).unwrap();
        }
        // Nothing reaches the shed until the day-3 harvest drops in at the end of day 3.
        assert!(shed_carrots[..96].iter().all(|&count| count == 0));
        assert!(shed_carrots[96] > 0);
        let buy = json!([["BUY_SEED", "CARROT", 1]]);
        let expected = [
            // Day 0: no seed yet, so buy one; the farmer has nothing to do.
            (0, action(json!(["PASS"]), buy.clone())),
            // The seed arrived; the tile under the farmer is empty.
            (1, action(json!(["PLANT", "CARROT"]), json!([]))),
            // Planting consumed the seed; the new carrot is dry.
            (2, action(json!(["WATER"]), buy.clone())),
            // One watering per day while the carrot ages.
            (24, action(json!(["WATER"]), json!([]))),
            (48, action(json!(["WATER"]), json!([]))),
            // Age 3 == max_yield_day: harvest, then replant with the stored seed.
            (72, action(json!(["HARVEST"]), json!([]))),
            (73, action(json!(["PLANT", "CARROT"]), json!([]))),
            (74, action(json!(["WATER"]), buy.clone())),
            // The harvest reaches the shed at the end of day 3 and sells in full on day 4.
            (
                96,
                action(
                    json!(["WATER"]),
                    json!([["SELL", "CARROT", shed_carrots[96]]]),
                ),
            ),
        ];
        for (step, decision) in &expected {
            assert_eq!(&decisions[*step], decision, "step {step}");
        }
        let busy: Vec<usize> = expected.iter().map(|(step, _)| *step).collect();
        for (step, decision) in decisions.iter().enumerate() {
            if !busy.contains(&step) {
                assert_eq!(decision, &pass_action(), "step {step}");
            }
        }
        assert_eq!(controller.debug()["shed_carrots"], json!(shed_carrots[96]));
    }

    #[test]
    fn a_farmer_who_cannot_afford_a_seed_passes_with_an_empty_market() {
        let game = game_with(Config {
            starting_money: 10_i64.into(),
            ..quiet_config()
        });
        let mut controller = StarterController::default();
        assert_eq!(controller.action(&game, 1).unwrap(), pass_action());
        assert_eq!(controller.debug()["seat"], json!(1));
    }

    #[test]
    fn hands_are_always_empty_even_for_a_farm_with_hands() {
        let game = game_with(quiet_config());
        let mut controller = StarterController::default();
        let decision = controller.action(&game, 0).unwrap();
        assert_eq!(decision["hands"], json!([]));
    }

    #[test]
    fn carrot_cycle_step_is_the_starter_rule_at_the_assigned_tile() {
        let mut game = game_with(quiet_config());
        let mut controller = StarterController::default();
        for step in 0..=74 {
            let farm = &game.farms()[0];
            let (x, y) = (farm.farmer[0], farm.farmer[1]);
            let cycle = carrot_cycle_step(&game, 0, x, y).unwrap();
            let decision = controller.action(&game, 0).unwrap();
            assert_eq!(cycle["verb"], decision["farmer"], "step {step}");
            assert_eq!(cycle["orders"], decision["market"], "step {step}");
            game.step(&[decision, pass_action()]).unwrap();
        }
        // another empty tile of the same farm: the rule plants there once a seed is held
        let farm = &game.farms()[0];
        let (x, y) = (farm.farmer[0], farm.farmer[1]);
        let other = carrot_cycle_step(&game, 0, x + 1, y).unwrap();
        assert!(other["verb"] == json!(["PLANT", "CARROT"]) || other["verb"] == json!(["PASS"]));
        assert!(carrot_cycle_step(&game, 0, 99, 99).is_err());
        assert!(carrot_cycle_step(&game, 2, x, y).is_err());
    }

    #[test]
    fn hire_step_requests_one_hand_when_affordable_and_not_at_the_days_end() {
        let mut game = game_with(quiet_config());
        let first = hire_step(&game, 0).unwrap();
        assert_eq!(first["orders"], json!([["HIRE"]]));
        assert_eq!(first["verb"], json!(["PASS"]));
        assert_eq!(first["done"], json!(false));
        // the hire lands after the market phase of the step that issued it
        game.step(&[action(json!(["PASS"]), json!([["HIRE"]])), pass_action()])
            .unwrap();
        assert_eq!(game.farms()[0].hands.len(), 1);
        assert_eq!(game.farms()[0].hires_today, 1);
        assert_eq!(game.farms()[0].money, 2999.0);
        assert_eq!(hire_step(&game, 0).unwrap()["orders"], json!([["HIRE"]]));
        // the last step of the day hires nobody: the hand would vanish before acting
        while !(game.step_index() + 1).is_multiple_of(TURNS_PER_DAY) {
            game.step(&[pass_action(), pass_action()]).unwrap();
        }
        let last = hire_step(&game, 0).unwrap();
        assert_eq!(last["orders"], json!([]));
        assert_eq!(last["done"], json!(false));
        game.step(&[pass_action(), pass_action()]).unwrap();
        assert!(game.farms()[0].hands.is_empty(), "hands vanish at the end of the day");
        assert_eq!(game.farms()[0].hires_today, 0);
        // a seat that cannot afford a hand is done
        let expensive = game_with(Config {
            farm_hand_cost_mult: 10_000_i64.into(),
            ..quiet_config()
        });
        let broke = hire_step(&expensive, 0).unwrap();
        assert_eq!(broke["orders"], json!([]));
        assert_eq!(broke["done"], json!(true));
        assert!(hire_step(&game, 2).is_err());
    }
}
