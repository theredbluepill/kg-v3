//! Native translation of Yhay81's Shop Router 0909 published Python agent.
//! Source: https://www.kaggle.com/code/yhay81/shop-router-0909
//! The author credits aurax7's public Reactive Router for sell timing and shed
//! projection; the ordered shop routes and worker-local day queues are Yhay81's.
//! Original Apache license and attribution are retained in agents/shoprouter/.
//!
//! The embedded 13 tapes are immutable policy data, not learned labels. Repairs
//! intentionally retain the source's lightweight (not engine-exact) shed estimate,
//! requested-sale accounting, fixed turn boundaries, and same-day worker queues.

use crate::{Farm, Game, PrivateState};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::sync::OnceLock;

const LAST_STEP: usize = 718;
const MAX_ORDERS: usize = 10;
const SHED_CAPACITY: i64 = 100;
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
type Command = Vec<Value>;
type Stock = IndexMap<String, i64>;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Action {
    farmer: Command,
    hands: Vec<Command>,
    market: Vec<Command>,
}

impl Action {
    fn workers(&self) -> impl Iterator<Item = &[Value]> {
        std::iter::once(self.farmer.as_slice()).chain(self.hands.iter().map(Vec::as_slice))
    }
}

fn tapes() -> &'static Vec<Vec<Action>> {
    static TAPES: OnceLock<Vec<Vec<Action>>> = OnceLock::new();
    TAPES.get_or_init(|| {
        let decoded: Vec<Vec<Action>> =
            serde_json::from_str(include_str!("../../fixtures/shop-router-0909-actions.json"))
                .expect("embedded Shop Router tapes must be valid JSON");
        assert_eq!(decoded.len(), 13, "Shop Router requires 13 tapes");
        assert!(decoded.iter().all(|tape| tape.len() == LAST_STEP + 1));
        decoded
    })
}

fn command(operation: &str) -> Command {
    vec![Value::from(operation)]
}

fn operation(work: &[Value]) -> Option<&str> {
    work.first().and_then(Value::as_str)
}

fn integer(value: Option<&Value>, default: i64) -> i64 {
    value
        .and_then(|value| value.as_i64().or_else(|| value.as_f64().map(|n| n as i64)))
        .unwrap_or(default)
}

fn count(stock: &Stock, item: &str) -> i64 {
    stock.get(item).copied().unwrap_or(0)
}

fn plan_for_shops(shops: &[String]) -> usize {
    match (
        shops.first().map(String::as_str),
        shops.get(1).map(String::as_str),
    ) {
        (Some("BAKERY"), Some("YARN_STORE")) => 3,
        (Some("BRUNCH_SPOT"), Some("YARN_STORE")) => 4,
        (Some("FARMERS_MARKET" | "PET_CAFE"), Some("YARN_STORE")) => 5,
        (Some("ICE_CREAM_SHOP"), Some("YARN_STORE")) => 6,
        (Some("PIZZA_SHOP"), Some("YARN_STORE")) => 7,
        (Some("SMOOTHIE_SHOP"), Some("YARN_STORE")) => 8,
        (Some("YARN_STORE"), Some("BAKERY" | "BRUNCH_SPOT" | "ICE_CREAM_SHOP")) => 9,
        (Some("YARN_STORE"), Some("FARMERS_MARKET")) => 1,
        (Some("YARN_STORE"), Some("PET_CAFE")) => 10,
        (Some("YARN_STORE"), Some("PIZZA_SHOP")) => 6,
        (Some("YARN_STORE"), Some("SMOOTHIE_SHOP")) => 11,
        (Some("YARN_STORE"), Some("YARN_STORE")) => 12,
        _ => 0,
    }
}

#[derive(Clone, Debug, Default)]
struct DayState {
    plan: usize,
    last_step: Option<usize>,
    day: Option<usize>,
    queues: IndexMap<usize, VecDeque<Command>>,
    sale_due_step: Option<usize>,
    advanced_sales: Stock,
}

impl DayState {
    fn begin(&mut self, step: usize, shops: &[String]) {
        if self.last_step.is_some_and(|previous| step <= previous) {
            *self = Self::default();
        }
        self.last_step = Some(step);
        if step == 144 {
            self.plan = plan_for_shops(shops);
        }
        if step == 648 {
            self.plan = 2;
        }
    }
}

fn positions(farm: &Farm) -> impl Iterator<Item = &[i64]> {
    std::iter::once(farm.farmer.as_slice()).chain(farm.hands.iter().map(Vec::as_slice))
}

fn beside_shed(position: &[i64], board: usize) -> bool {
    let center = (board / 2) as i64;
    position
        .first()
        .is_some_and(|x| *x == center - 1 || *x == center)
        && position
            .get(1)
            .is_some_and(|y| *y == center - 1 || *y == center)
}

fn repair_weeds(action: &mut Action, farm: &Farm, state: &mut DayState, step: usize) {
    let day = step / 24;
    if state.day != Some(day) {
        state.day = Some(day);
        state.queues.clear();
    }
    if action.farmer.is_empty() {
        action.farmer = command("PASS");
    }
    let workers = std::iter::once(&mut action.farmer).chain(action.hands.iter_mut());
    for (worker, (work, position)) in workers.zip(positions(farm)).enumerate() {
        let queue = state.queues.entry(worker).or_default();
        queue.push_back(work.clone());
        let tile = position
            .get(1)
            .and_then(|y| usize::try_from(*y).ok().and_then(|y| farm.tiles.get(y)))
            .and_then(|row| {
                position
                    .first()
                    .and_then(|x| usize::try_from(*x).ok())
                    .and_then(|x| row.get(x))
            });
        let blocked = matches!(
            queue.front().and_then(|front| operation(front)),
            Some("PLANT" | "BUILD_COOP" | "BUILD_PASTURE")
        ) && tile
            .and_then(|tile| tile.get("kind"))
            .and_then(Value::as_str)
            == Some("WEED");
        *work = if blocked {
            command("DIG")
        } else {
            queue.pop_front().expect("just appended work")
        };
    }
}

fn projected_shed(action: &Action, farm: &Farm, private: &PrivateState) -> Stock {
    let mut stock: Stock = PRODUCTS
        .iter()
        .map(|item| (item.to_string(), count(&private.shed, item).max(0)))
        .collect();
    for (item, quantity) in &private.shed {
        stock.insert(item.clone(), (*quantity).max(0));
    }
    let mut total: i64 = stock.values().sum();
    for (worker, (work, position)) in action.workers().zip(positions(farm)).enumerate() {
        if !beside_shed(position, farm.tiles.len()) {
            continue;
        }
        let empty = Stock::new();
        let inventory = private.inventories.get(worker).unwrap_or(&empty);
        let item = work.get(1).and_then(Value::as_str);
        match operation(work) {
            Some("PICKUP") if item.is_some_and(|item| stock.contains_key(item)) => {
                let item = item.expect("guard checked item");
                let taken = count(&stock, item).min(integer(work.get(2), 1).max(0));
                *stock.get_mut(item).expect("guard checked stock") -= taken;
                total -= taken;
            }
            Some("DROP") => {
                for (item, held) in inventory {
                    let added = (*held).max(0).min((SHED_CAPACITY - total).max(0));
                    if added > 0 {
                        *stock.entry(item.clone()).or_insert(0) += added;
                        total += added;
                    }
                }
            }
            Some("PLACE")
                if item.is_some_and(|item| !matches!(item, "GOOSE" | "COW" | "SHEEP")) =>
            {
                let item = item.expect("guard checked item");
                let added = integer(work.get(2), 1)
                    .max(0)
                    .min(count(inventory, item).max(0))
                    .min((SHED_CAPACITY - total).max(0));
                if added > 0 {
                    *stock.entry(item.to_string()).or_insert(0) += added;
                    total += added;
                }
            }
            _ => {}
        }
    }
    stock
}

fn subtract_advanced_sales(action: &mut Action, state: &mut DayState, step: usize) {
    if state.sale_due_step == Some(step) {
        let mut remaining = state.advanced_sales.clone();
        for order in &mut action.market {
            if operation(order) == Some("SELL") && order.len() >= 3 {
                let Some(item) = order[1].as_str() else {
                    continue;
                };
                let quantity = integer(order.get(2), 0);
                let removed = quantity.max(0).min(count(&remaining, item));
                if removed > 0 {
                    *remaining.get_mut(item).expect("positive remainder exists") -= removed;
                    order[2] = Value::from(quantity - removed);
                }
            }
        }
    }
    state.advanced_sales.clear();
    state.sale_due_step = None;
}

fn advance_sales(
    action: &mut Action,
    farm: &Farm,
    private: &PrivateState,
    prices: &IndexMap<String, Value>,
    state: &mut DayState,
    tape: &[Action],
    step: usize,
) {
    let next_step = step + 1;
    if next_step > LAST_STEP || next_step.is_multiple_of(72) || step.is_multiple_of(4) {
        return;
    }
    let mut planned = Stock::new();
    for order in &tape[next_step].market {
        if operation(order) == Some("SELL")
            && order.len() >= 3
            && let Some(item) = order[1].as_str().filter(|item| PRODUCTS.contains(item))
        {
            *planned.entry(item.to_string()).or_insert(0) += integer(order.get(2), 0).max(0);
        }
    }
    // Snapshot the original set, matching the source comprehension before append.
    let already_selling: Vec<String> = action
        .market
        .iter()
        .filter(|order| operation(order) == Some("SELL"))
        .filter_map(|order| order.get(1).and_then(Value::as_str).map(str::to_string))
        .collect();
    let stock = projected_shed(action, farm, private);
    for item in PRODUCTS {
        if matches!(item, "WHEAT" | "FERTILIZER")
            || already_selling.iter().any(|existing| existing == item)
        {
            continue;
        }
        let quantity = count(&stock, item).min(count(&planned, item));
        if quantity <= 0 || integer(prices.get(item), 0) < 2 {
            continue;
        }
        if action.market.len() >= MAX_ORDERS {
            break;
        }
        action.market.push(vec![
            Value::from("SELL"),
            Value::from(item),
            Value::from(quantity),
        ]);
        state.advanced_sales.insert(item.to_string(), quantity);
    }
    if !state.advanced_sales.is_empty() {
        state.sale_due_step = Some(next_step);
    }
}

fn liquidate(farm: &Farm, private: &PrivateState, prices: &IndexMap<String, Value>) -> Action {
    let mut workers = positions(farm).enumerate().map(|(worker, position)| {
        command(
            if beside_shed(position, farm.tiles.len())
                && private
                    .inventories
                    .get(worker)
                    .is_some_and(|inventory| !inventory.is_empty())
            {
                "DROP"
            } else {
                "PASS"
            },
        )
    });
    let mut action = Action {
        farmer: workers.next().expect("farm has farmer"),
        hands: workers.collect(),
        market: Vec::new(),
    };
    let stock = projected_shed(&action, farm, private);
    action.market = PRODUCTS
        .iter()
        .filter(|item| count(&stock, item) > 0)
        .map(|item| {
            vec![
                Value::from("SELL"),
                Value::from(*item),
                Value::from(count(&stock, item)),
            ]
        })
        .collect();
    // Stable sort preserves PRODUCTS order for tied nominal values, like Python.
    action.market.sort_by_key(|order| {
        Reverse(
            integer(prices.get(order[1].as_str().expect("product string")), 0)
                * integer(order.get(2), 0),
        )
    });
    action
}

#[derive(Clone, Debug, Default)]
pub struct ShopRouterController {
    players: [DayState; 2],
    last_debug: Value,
}

impl ShopRouterController {
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        let state = self
            .players
            .get_mut(seat)
            .ok_or_else(|| format!("Shop Router seat {seat} is outside the two-player game"))?;
        let step = game.step_index();
        state.begin(step, &game.town().unlocked_shops);
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("Shop Router seat {seat} has no farm"))?;
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("Shop Router seat {seat} has no private state"))?;
        let tape = &tapes()[state.plan];
        let mut action = tape
            .get(step)
            .cloned()
            .ok_or_else(|| format!("Shop Router tape has no step {step}"))?;
        repair_weeds(&mut action, farm, state, step);
        subtract_advanced_sales(&mut action, state, step);
        advance_sales(
            &mut action,
            farm,
            private,
            &game.market().prices,
            state,
            tape,
            step,
        );
        action.market.truncate(MAX_ORDERS);
        if step == LAST_STEP {
            action = liquidate(farm, private, &game.market().prices);
        }
        self.last_debug = json!({"step": step, "seat": seat, "plan": state.plan,
            "day": state.day, "queue_lengths": state.queues.iter().map(|(worker, queue)| (worker.to_string(), queue.len())).collect::<IndexMap<_,_>>(),
            "sale_due_step": state.sale_due_step, "advanced_sales": state.advanced_sales});
        serde_json::to_value(action).map_err(|error| error.to_string())
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(value: Value) -> Action {
        serde_json::from_value(value).unwrap()
    }
    fn pass() -> Action {
        action(json!({"farmer":["PASS"],"hands":[],"market":[]}))
    }
    fn farm() -> Farm {
        Farm {
            money: 0.0,
            tiles: vec![vec![Value::Null; 10]; 10],
            farmer: vec![4, 4],
            hands: vec![vec![5, 5]],
            unlocked_quadrants: vec![],
            hires_today: 0,
        }
    }
    fn private() -> PrivateState {
        PrivateState {
            shed: Stock::new(),
            seeds: Stock::new(),
            inventories: vec![Stock::new(), Stock::new()],
        }
    }

    #[test]
    fn all_thirteen_tapes_are_complete() {
        assert_eq!(tapes().len(), 13);
        assert!(tapes().iter().all(|tape| tape.len() == 719));
    }

    #[test]
    fn ordered_shop_pairs_select_every_branch() {
        for (first, second, expected) in [
            ("BAKERY", "YARN_STORE", 3),
            ("BRUNCH_SPOT", "YARN_STORE", 4),
            ("FARMERS_MARKET", "YARN_STORE", 5),
            ("ICE_CREAM_SHOP", "YARN_STORE", 6),
            ("PET_CAFE", "YARN_STORE", 5),
            ("PIZZA_SHOP", "YARN_STORE", 7),
            ("SMOOTHIE_SHOP", "YARN_STORE", 8),
            ("YARN_STORE", "BAKERY", 9),
            ("YARN_STORE", "BRUNCH_SPOT", 9),
            ("YARN_STORE", "FARMERS_MARKET", 1),
            ("YARN_STORE", "ICE_CREAM_SHOP", 9),
            ("YARN_STORE", "PET_CAFE", 10),
            ("YARN_STORE", "PIZZA_SHOP", 6),
            ("YARN_STORE", "SMOOTHIE_SHOP", 11),
            ("YARN_STORE", "YARN_STORE", 12),
            ("BAKERY", "PET_CAFE", 0),
        ] {
            assert_eq!(
                plan_for_shops(&[first.into(), second.into(), "IGNORED".into()]),
                expected
            );
        }
        assert_eq!(plan_for_shops(&[]), 0);
        assert_eq!(plan_for_shops(&["YARN_STORE".into()]), 0);
    }

    #[test]
    fn repeated_or_rewound_step_resets_all_memory_and_final_plan_is_fixed() {
        let shops = vec!["BAKERY".into(), "YARN_STORE".into()];
        let mut state = DayState::default();
        state.begin(144, &shops);
        assert_eq!(state.plan, 3);
        state.begin(145, &[]);
        assert_eq!(state.plan, 3);
        state.advanced_sales.insert("MILK".into(), 4);
        state.sale_due_step = Some(146);
        state.queues.insert(0, VecDeque::from([command("WATER")]));
        state.begin(145, &[]);
        assert_eq!(state.plan, 0);
        assert!(state.queues.is_empty() && state.advanced_sales.is_empty());
        assert_eq!(state.sale_due_step, None);
        state.begin(648, &shops);
        assert_eq!(state.plan, 2);
        state.begin(647, &shops);
        assert_eq!(state.plan, 0);
    }

    #[test]
    fn weeds_shift_only_one_worker_and_queues_expire_at_dawn() {
        let mut farm = farm();
        farm.tiles[4][4] = json!({"kind":"WEED"});
        let mut state = DayState::default();
        let mut first = action(json!({"farmer":["PLANT","WHEAT"],"hands":[["WATER"]],"market":[]}));
        repair_weeds(&mut first, &farm, &mut state, 21);
        assert_eq!(first.farmer, command("DIG"));
        assert_eq!(first.hands[0], command("WATER"));
        farm.tiles[4][4] = Value::Null;
        let mut next = pass();
        next.farmer = command("WATER");
        repair_weeds(&mut next, &farm, &mut state, 22);
        assert_eq!(
            next.farmer,
            vec![Value::from("PLANT"), Value::from("WHEAT")]
        );
        let mut dawn = pass();
        dawn.farmer = command("EAST");
        repair_weeds(&mut dawn, &farm, &mut state, 24);
        assert_eq!(dawn.farmer, command("EAST"));
        assert!(state.queues[&0].is_empty());
    }

    #[test]
    fn projection_preserves_worker_inventory_order_and_pickup_frees_capacity() {
        let farm = farm();
        let mut private = private();
        private.shed.insert("WHEAT".into(), 100);
        private.shed.insert("NEGATIVE".into(), -5);
        private.inventories[1] = Stock::from([("MILK".into(), 7), ("WOOL".into(), 7)]);
        let work = action(json!({"farmer":["PICKUP","WHEAT",10],"hands":[["DROP"]],"market":[]}));
        let stock = projected_shed(&work, &farm, &private);
        assert_eq!(stock["WHEAT"], 90);
        assert_eq!(stock["MILK"], 7);
        assert_eq!(stock["WOOL"], 3);
        assert_eq!(stock["NEGATIVE"], 0);
        private.inventories[0] = Stock::from([("GOOSE".into(), 2), ("MILK".into(), 3)]);
        private.shed.clear();
        let animal = action(json!({"farmer":["PLACE","GOOSE",2],"hands":[],"market":[]}));
        assert_eq!(count(&projected_shed(&animal, &farm, &private), "GOOSE"), 0);
        let product = action(json!({"farmer":["PLACE","MILK",9],"hands":[],"market":[]}));
        assert_eq!(projected_shed(&product, &farm, &private)["MILK"], 3);
    }

    #[test]
    fn advanced_subtraction_keeps_zero_slots_and_expires_on_skipped_step() {
        let mut state = DayState {
            sale_due_step: Some(2),
            advanced_sales: Stock::from([("MILK".into(), 7)]),
            ..DayState::default()
        };
        let mut work = action(
            json!({"farmer":["PASS"],"hands":[],"market":[["SELL","MILK",4],["BUY_PRODUCT","WHEAT",1],["SELL","MILK",5]]}),
        );
        subtract_advanced_sales(&mut work, &mut state, 2);
        assert_eq!(work.market[0][2], json!(0));
        assert_eq!(work.market[2][2], json!(2));
        assert_eq!(work.market.len(), 3);
        assert!(state.advanced_sales.is_empty());
        state.sale_due_step = Some(3);
        state.advanced_sales.insert("MILK".into(), 4);
        subtract_advanced_sales(&mut work, &mut state, 4);
        assert_eq!(work.market[2][2], json!(2));
        assert_eq!(state.sale_due_step, None);
    }

    #[test]
    fn advances_exclude_reserves_current_sales_price_floor_and_boundary_turns() {
        let farm = farm();
        let mut private = private();
        private.shed = Stock::from(PRODUCTS.map(|item| (item.into(), 10)));
        let prices = PRODUCTS
            .map(|item| (item.into(), json!(if item == "MELON" { 1 } else { 20 })))
            .into_iter()
            .collect();
        let future = action(
            json!({"farmer":["PASS"],"hands":[],"market":[["SELL","WHEAT",10],["SELL","FERTILIZER",10],["SELL","MELON",10],["SELL","CARROT",10],["SELL","MILK",3],["SELL","MILK",4]]}),
        );
        let tape = vec![future; 719];
        let mut work = pass();
        work.market
            .push(vec![json!("SELL"), json!("CARROT"), json!(0)]);
        let mut state = DayState::default();
        advance_sales(&mut work, &farm, &private, &prices, &mut state, &tape, 1);
        assert_eq!(work.market.len(), 2);
        assert_eq!(work.market[1], vec![json!("SELL"), json!("MILK"), json!(7)]);
        assert_eq!(state.sale_due_step, Some(2));
        for step in [0, 4, 71, 143, 718] {
            let mut work = pass();
            let mut state = DayState::default();
            advance_sales(&mut work, &farm, &private, &prices, &mut state, &tape, step);
            assert!(work.market.is_empty() && state.advanced_sales.is_empty());
        }
        let mut full = pass();
        full.market = vec![command("PASS"); 10];
        advance_sales(
            &mut full,
            &farm,
            &private,
            &prices,
            &mut DayState::default(),
            &tape,
            1,
        );
        assert_eq!(full.market.len(), 10);
    }

    #[test]
    fn final_liquidation_uses_reachable_stock_and_stable_nominal_value_order() {
        let mut farm = farm();
        farm.hands[0] = vec![0, 0];
        let mut private = private();
        private.shed = Stock::from([
            ("WHEAT".into(), 4),
            ("CARROT".into(), 2),
            ("GOOSE".into(), 3),
        ]);
        private.inventories[0] = Stock::from([("MILK".into(), 1)]);
        private.inventories[1] = Stock::from([("WOOL".into(), 9)]);
        let prices = IndexMap::from([
            ("WHEAT".into(), json!(5)),
            ("CARROT".into(), json!(10)),
            ("MILK".into(), json!(1)),
        ]);
        let result = liquidate(&farm, &private, &prices);
        assert_eq!(result.farmer, command("DROP"));
        assert_eq!(result.hands[0], command("PASS"));
        assert_eq!(
            result.market,
            vec![
                vec![json!("SELL"), json!("WHEAT"), json!(4)],
                vec![json!("SELL"), json!("CARROT"), json!(2)],
                vec![json!("SELL"), json!("MILK"), json!(1)]
            ]
        );
    }
}
