//! Native port of Tetsutani's "Shape the Shop Work the Pasture" controller.
//!
//! The public submission selects among four prefix-compatible 720-step route
//! tapes, then applies four observation-dependent repairs: weed digging,
//! projected-shed accounting, day-close capacity protection, and dead-stock
//! liquidation.  The frozen Python output remains the differential authority;
//! this module performs no Python calls.

use crate::{Farm, Game, PrivateState};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Value, json};
use std::cmp::Reverse;
use std::sync::OnceLock;

const SOURCE_SHA256: &str = "2b97e2c653018ac4aeffb8463ec91c8f26b097b4cef81289acc25ccdbc68f916";
const PROGRAM_STEPS: usize = 720;
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

const MAIN: &str = "7015cc00acfa4922";
const YARN: &str = "dc76e4003029ac51";
const YARN_CARROT: &str = "ab9669b9abfbea4e";
const MILK_GLUT: &str = "a84d06f1d12add7c";

#[derive(Debug, Deserialize)]
struct RouteFixture {
    source_sha256: String,
    routes: IndexMap<String, Vec<Value>>,
}

fn fixture() -> &'static RouteFixture {
    static FIXTURE: OnceLock<RouteFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let decoded: RouteFixture = serde_json::from_str(include_str!(
            "../../fixtures/tetsutani-shape-shop-pasture-routes.json"
        ))
        .expect("embedded Tetsutani route fixture must be valid JSON");
        assert_eq!(
            decoded.source_sha256, SOURCE_SHA256,
            "Tetsutani source digest changed"
        );
        for route in [MAIN, YARN, YARN_CARROT, MILK_GLUT] {
            let rows = decoded
                .routes
                .get(route)
                .unwrap_or_else(|| panic!("Tetsutani route {route} is missing"));
            assert_eq!(
                rows.len(),
                PROGRAM_STEPS,
                "Tetsutani route {route} must contain 720 actions"
            );
        }
        decoded
    })
}

#[derive(Clone, Debug, PartialEq)]
struct Action {
    farmer: Vec<Value>,
    hands: Vec<Vec<Value>>,
    market: Vec<Vec<Value>>,
}

impl Action {
    fn pass() -> Self {
        Self {
            farmer: vec![Value::String("PASS".into())],
            hands: Vec::new(),
            market: Vec::new(),
        }
    }

    fn from_value(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "Tetsutani route action is not an object".to_string())?;
        let farmer = object
            .get("farmer")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![Value::String("PASS".into())]);
        let hands = object
            .get("hands")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|row| {
                row.as_array()
                    .cloned()
                    .ok_or_else(|| "Tetsutani hand command is not an array".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let market = object
            .get("market")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|row| {
                row.as_array()
                    .cloned()
                    .ok_or_else(|| "Tetsutani market order is not an array".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            farmer,
            hands,
            market,
        })
    }

    fn into_value(self) -> Value {
        json!({"farmer": self.farmer, "hands": self.hands, "market": self.market})
    }
}

fn route_action(route: &str, step: usize) -> Result<Action, String> {
    if step >= PROGRAM_STEPS {
        return Ok(Action::pass());
    }
    let row = fixture()
        .routes
        .get(route)
        .and_then(|rows| rows.get(step))
        .ok_or_else(|| format!("Tetsutani route {route} has no step {step}"))?;
    Action::from_value(row)
}

fn text(value: Option<&Value>) -> Option<&str> {
    value.and_then(Value::as_str)
}

fn integer(value: Option<&Value>, default: i64) -> i64 {
    value
        .and_then(|value| value.as_i64().or_else(|| value.as_f64().map(|x| x as i64)))
        .unwrap_or(default)
}

fn count(map: &IndexMap<String, i64>, item: &str) -> i64 {
    map.get(item).copied().unwrap_or(0)
}

fn market_integer(map: &IndexMap<String, Value>, item: &str) -> i64 {
    integer(map.get(item), 0)
}

fn animal_structure(item: &str) -> Option<&'static str> {
    match item {
        "GOOSE" => Some("COOP"),
        "COW" | "SHEEP" => Some("PASTURE"),
        _ => None,
    }
}

fn shed_adjacent(x: i64, y: i64, board: usize) -> bool {
    let half = i64::try_from(board / 2).unwrap_or(i64::MAX);
    matches!(
        (x, y),
        (a, b) if (a == half - 1 || a == half) && (b == half - 1 || b == half)
    )
}

fn position(row: &[i64]) -> Option<(i64, i64)> {
    Some((*row.first()?, *row.get(1)?))
}

fn tile_at(farm: &Farm, x: i64, y: i64) -> Option<&Value> {
    let x = usize::try_from(x).ok()?;
    let y = usize::try_from(y).ok()?;
    farm.tiles.get(y)?.get(x)
}

fn object_kind(tile: &Value) -> Option<&str> {
    tile.as_object()?.get("kind")?.as_str()
}

fn animal_present(tile: &Value) -> bool {
    tile.as_object()
        .and_then(|object| object.get("animal"))
        .is_some_and(|animal| !animal.is_null())
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
        Some(Value::Null) | None => false,
    }
}

fn certainly_noop(
    command: &[Value],
    tile: &Value,
    inventory: &IndexMap<String, i64>,
    seeds: &IndexMap<String, i64>,
    x: i64,
    y: i64,
    board: usize,
) -> bool {
    let Some(operation) = command.first().and_then(Value::as_str) else {
        return true;
    };
    if let Some((dx, dy)) = match operation {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    } {
        let edge = i64::try_from(board).unwrap_or(i64::MAX);
        return !(0 <= x + dx && x + dx < edge && 0 <= y + dy && y + dy < edge);
    }
    match operation {
        "PASS" => return true,
        "DROP" => return !shed_adjacent(x, y, board) || inventory.is_empty(),
        "PICKUP" => return !shed_adjacent(x, y, board),
        "PLACE" => {
            let item = text(command.get(1));
            if let Some(structure) = item.and_then(animal_structure)
                && object_kind(tile) == Some(structure)
                && tile
                    .as_object()
                    .and_then(|object| object.get("animal"))
                    .is_some_and(Value::is_null)
            {
                return item.is_none_or(|item| count(inventory, item) <= 0);
            }
            if shed_adjacent(x, y, board) {
                return item.is_none_or(|item| count(inventory, item) <= 0);
            }
            return true;
        }
        _ => {}
    }
    if tile.as_str() == Some("LOCKED") {
        return true;
    }
    let object = tile.as_object();
    let kind = object_kind(tile);
    let animal = animal_present(tile);
    match operation {
        "PLANT" => {
            !tile.is_null() || text(command.get(1)).is_none_or(|item| count(seeds, item) <= 0)
        }
        "WATER" => {
            kind != Some("PLANT") || truthy(object.and_then(|object| object.get("watered_today")))
        }
        "HARVEST" => {
            object.is_none() || integer(object.and_then(|object| object.get("yield_units")), 0) <= 0
        }
        "FERTILIZE" => kind != Some("PLANT") || count(inventory, "FERTILIZER") <= 0,
        "DIG" => tile.is_null() || animal,
        "BUILD_COOP" | "BUILD_PASTURE" => !tile.is_null(),
        "FEED" => {
            !animal
                || truthy(object.and_then(|object| object.get("fed_today")))
                || count(inventory, "WHEAT") <= 0
        }
        "COLLECT_FERTILIZER" => {
            !animal || !truthy(object.and_then(|object| object.get("fertilizer_available")))
        }
        "CARE" => !animal || truthy(object.and_then(|object| object.get("cared_today"))),
        _ => true,
    }
}

fn future_sells(route: &str, item: &str, step: usize) -> i64 {
    fixture()
        .routes
        .get(route)
        .into_iter()
        .flatten()
        .skip(step)
        .flat_map(|row| {
            row.get("market")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(Value::as_array)
        .filter(|order| text(order.first()) == Some("SELL") && text(order.get(1)) == Some(item))
        .map(|order| integer(order.get(2), 0))
        .sum()
}

fn switch_ok(current: &str, target: &str, turn: usize) -> bool {
    if current == target {
        return false;
    }
    let routes = &fixture().routes;
    let Some(a) = routes.get(current) else {
        return false;
    };
    let Some(b) = routes.get(target) else {
        return false;
    };
    a.iter()
        .zip(b)
        .take(turn)
        .all(|(left, right)| left == right)
}

fn projected_shed(farm: &Farm, private: &PrivateState, action: &Action) -> IndexMap<String, i64> {
    let mut projected = private.shed.clone();
    let mut room = SHED_CAPACITY - projected.values().sum::<i64>();
    let positions =
        std::iter::once(farm.farmer.as_slice()).chain(farm.hands.iter().map(Vec::as_slice));
    let commands =
        std::iter::once(action.farmer.as_slice()).chain(action.hands.iter().map(Vec::as_slice));
    for (index, (position_row, command)) in positions.zip(commands).enumerate() {
        if room <= 0 {
            break;
        }
        let Some((x, y)) = position(position_row) else {
            continue;
        };
        let Some(inventory) = private.inventories.get(index) else {
            continue;
        };
        if inventory.is_empty() || !shed_adjacent(x, y, farm.tiles.len()) {
            continue;
        }
        match text(command.first()) {
            Some("DROP") => {
                for (item, quantity) in inventory {
                    let taken = (*quantity).min(room);
                    if taken > 0 {
                        *projected.entry(item.clone()).or_insert(0) += taken;
                        room -= taken;
                    }
                }
            }
            Some("PLACE") => {
                let Some(item) = text(command.get(1)) else {
                    continue;
                };
                if animal_structure(item).is_some() {
                    continue;
                }
                let requested = integer(command.get(2), 1);
                let taken = requested.min(count(inventory, item)).min(room);
                if taken > 0 {
                    *projected.entry(item.to_string()).or_insert(0) += taken;
                    room -= taken;
                }
            }
            _ => {}
        }
    }
    projected
}

fn weed_repair(farm: &Farm, private: &PrivateState, action: &mut Action) {
    let board = farm.tiles.len();
    let positions =
        std::iter::once(farm.farmer.as_slice()).chain(farm.hands.iter().map(Vec::as_slice));
    let commands = std::iter::once(&mut action.farmer).chain(action.hands.iter_mut());
    for (index, (position_row, command)) in positions.zip(commands).enumerate() {
        let Some((x, y)) = position(position_row) else {
            continue;
        };
        let Some(tile) = tile_at(farm, x, y) else {
            continue;
        };
        let empty = IndexMap::new();
        let inventory = private.inventories.get(index).unwrap_or(&empty);
        if object_kind(tile) == Some("WEED")
            && certainly_noop(command, tile, inventory, &private.seeds, x, y, board)
        {
            *command = vec![Value::String("DIG".into())];
        }
    }
}

fn day_close_capacity_guard(
    game: &Game,
    farm: &Farm,
    private: &PrivateState,
    route: &str,
    step: usize,
    action: &mut Action,
) {
    if step % 24 != 23 {
        return;
    }
    let carried: i64 = private
        .inventories
        .iter()
        .flat_map(IndexMap::values)
        .map(|quantity| (*quantity).max(0))
        .sum();
    let positions =
        std::iter::once(farm.farmer.as_slice()).chain(farm.hands.iter().map(Vec::as_slice));
    let commands =
        std::iter::once(action.farmer.as_slice()).chain(action.hands.iter().map(Vec::as_slice));
    let mut produced = 0;
    let mut consumed = 0;
    for (position_row, command) in positions.zip(commands) {
        let Some((x, y)) = position(position_row) else {
            continue;
        };
        let Some(tile) = tile_at(farm, x, y) else {
            continue;
        };
        match text(command.first()) {
            Some("HARVEST") if tile.is_object() => {
                produced += integer(tile.get("yield_units"), 0).max(0);
            }
            Some("COLLECT_FERTILIZER")
                if tile.is_object() && truthy(tile.get("fertilizer_available")) =>
            {
                produced += 1;
            }
            Some("FEED" | "FERTILIZE") => consumed += 1,
            Some("PLACE")
                if text(command.get(1)).is_some_and(|item| animal_structure(item).is_some()) =>
            {
                consumed += 1;
            }
            _ => {}
        }
    }

    let mut planned_sells: IndexMap<String, i64> = IndexMap::new();
    let mut planned_buys = 0;
    for order in &action.market {
        match text(order.first()) {
            Some("SELL") => {
                if let Some(item) = text(order.get(1)) {
                    *planned_sells.entry(item.to_string()).or_insert(0) +=
                        integer(order.get(2), 0).max(0);
                }
            }
            Some("BUY_PRODUCT" | "BUY_ANIMAL") => {
                planned_buys += integer(order.get(2), 0).max(0);
            }
            _ => {}
        }
    }
    let shed_total: i64 = private
        .shed
        .values()
        .map(|quantity| (*quantity).max(0))
        .sum();
    let actual_existing_sells: i64 = planned_sells
        .iter()
        .map(|(item, planned)| count(&private.shed, item).max(0).min(*planned))
        .sum();
    let mut needed = shed_total + carried + produced - consumed + planned_buys
        - actual_existing_sells
        - (SHED_CAPACITY - 1);
    if needed <= 0 {
        return;
    }

    let mut priority: Vec<(usize, &str)> = PRODUCTS.iter().copied().enumerate().collect();
    priority.sort_by_key(|(index, item)| {
        (
            future_sells(route, item, step + 1) > 0,
            Reverse(market_integer(&game.market().prices, item)),
            *item,
            *index,
        )
    });
    for (_, item) in priority {
        let already = planned_sells.get(item).copied().unwrap_or(0);
        let available = (count(&private.shed, item) - already).max(0);
        let quantity = needed.min(available);
        if quantity <= 0 {
            continue;
        }
        if let Some(order) = action
            .market
            .iter_mut()
            .find(|order| text(order.first()) == Some("SELL") && text(order.get(1)) == Some(item))
        {
            let value = integer(order.get(2), 0).max(0) + quantity;
            order[2] = Value::from(value);
        } else if action.market.len() < MAX_ORDERS {
            action.market.push(vec![
                Value::String("SELL".into()),
                Value::String(item.into()),
                Value::from(quantity),
            ]);
        } else {
            continue;
        }
        planned_sells.insert(item.to_string(), already + quantity);
        needed -= quantity;
        if needed <= 0 {
            break;
        }
    }
}

fn clamp_sells(projected: &IndexMap<String, i64>, action: &mut Action) {
    let mut available = projected.clone();
    let mut kept = Vec::with_capacity(action.market.len());
    for mut order in action.market.drain(..) {
        if text(order.first()) != Some("SELL") {
            kept.push(order);
            continue;
        }
        let Some(item) = text(order.get(1)).map(str::to_string) else {
            continue;
        };
        let have = count(&available, &item);
        if have <= 0 {
            continue;
        }
        let quantity = integer(order.get(2), 0).min(have);
        if quantity <= 0 {
            continue;
        }
        available.insert(item, have - quantity);
        order[2] = Value::from(quantity);
        kept.push(order);
    }
    action.market = kept;
}

fn append_dead_stock(
    game: &Game,
    projected: &IndexMap<String, i64>,
    route: &str,
    step: usize,
    action: &mut Action,
) {
    let mut planned: IndexMap<String, i64> = IndexMap::new();
    for order in &action.market {
        if text(order.first()) == Some("SELL")
            && let Some(item) = text(order.get(1))
        {
            *planned.entry(item.to_string()).or_insert(0) += integer(order.get(2), 0);
        }
    }
    let day = step / 24;
    let mut extra: Vec<(usize, i64, Vec<Value>)> = Vec::new();
    for (index, item) in PRODUCTS.iter().copied().enumerate() {
        let have = count(projected, item) - planned.get(item).copied().unwrap_or(0);
        if have <= 0 {
            continue;
        }
        let surplus = if day >= 29 {
            have
        } else {
            have - future_sells(route, item, step + 1)
        };
        let price = market_integer(&game.market().prices, item);
        if surplus > 0 && price > 1 {
            extra.push((
                index,
                price * surplus,
                vec![
                    Value::String("SELL".into()),
                    Value::String(item.into()),
                    Value::from(surplus),
                ],
            ));
        }
    }
    extra.sort_by_key(|(index, value, _)| (Reverse(*value), *index));
    action
        .market
        .extend(extra.into_iter().map(|(_, _, order)| order));
    action.market.truncate(MAX_ORDERS);
}

#[derive(Clone, Debug)]
pub struct TetsutaniController {
    current: [String; 2],
    last_debug: Value,
}

impl Default for TetsutaniController {
    fn default() -> Self {
        Self {
            current: std::array::from_fn(|_| MAIN.to_string()),
            last_debug: Value::Null,
        }
    }
}

impl TetsutaniController {
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat >= 2 {
            return Err(format!(
                "Tetsutani seat {seat} is outside the two-player game"
            ));
        }
        let step = game.step_index();
        if step == 0 {
            self.current[seat] = MAIN.to_string();
        }
        match step {
            226 if game
                .town()
                .unlocked_shops
                .iter()
                .filter(|shop| shop.as_str() == "YARN_STORE")
                .count()
                >= 1
                && switch_ok(&self.current[seat], YARN, step) =>
            {
                self.current[seat] = YARN.to_string();
            }
            360 if market_integer(&game.market().prices, "CARROT") >= 42
                && switch_ok(&self.current[seat], YARN_CARROT, step) =>
            {
                self.current[seat] = YARN_CARROT.to_string();
            }
            433 if market_integer(&game.market().inventory, "MILK") >= 10_067
                && switch_ok(&self.current[seat], MILK_GLUT, step) =>
            {
                self.current[seat] = MILK_GLUT.to_string();
            }
            _ => {}
        }

        let route = self.current[seat].clone();
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("Tetsutani seat {seat} is missing its farm"))?;
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("Tetsutani seat {seat} is missing its private state"))?;
        let mut action = route_action(&route, step)?;
        weed_repair(farm, private, &mut action);
        let projected = projected_shed(farm, private, &action);
        day_close_capacity_guard(game, farm, private, &route, step, &mut action);
        clamp_sells(&projected, &mut action);
        append_dead_stock(game, &projected, &route, step, &mut action);

        self.last_debug = json!({"step": step, "seat": seat, "route": route});
        Ok(action.into_value())
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_contains_four_complete_routes() {
        let fixture = fixture();
        assert_eq!(fixture.source_sha256, SOURCE_SHA256);
        assert_eq!(fixture.routes.len(), 4);
        for route in [MAIN, YARN, YARN_CARROT, MILK_GLUT] {
            assert_eq!(fixture.routes[route].len(), PROGRAM_STEPS);
        }
    }

    #[test]
    fn route_prefixes_enforce_the_three_public_switches() {
        assert!(switch_ok(MAIN, YARN, 226));
        assert!(switch_ok(YARN, YARN_CARROT, 360));
        assert!(switch_ok(MAIN, MILK_GLUT, 433));
        assert!(!switch_ok(YARN, MILK_GLUT, 433));
    }
}
