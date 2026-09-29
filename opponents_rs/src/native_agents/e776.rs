//! Complete native port of the frozen E776 controller stack.
//!
//! E776 is a guarded, stateful programme over a frozen 719-step Kenjo trace:
//! E749/E750 execution guards, E773 visible-demand animal substitution, E774
//! terminal liquidation, and E775/E776's latent-pasture activation/delivery.
//! Keep this implementation literal; JA17's full-season action parity is the
//! authority for ordering, state reset, and arithmetic semantics.

use crate::{Farm, Game, PrivateState};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const MAX_STEPS: usize = 720;
const MAX_MARKET_ORDERS: usize = 10;
const SHED_CAPACITY: i64 = 100;
const WEED_REPLAY_STEPS: usize = 8;
const I0: i64 = 10_000;
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
const ANIMAL_PRODUCTS: [&str; 2] = ["MILK", "WOOL"];
const ACTIVATION_STEP: usize = 313;
const FINAL_EXECUTABLE_STEP: usize = 718;
const TARGET_TILE: (usize, usize) = (5, 3);

fn trace() -> &'static Vec<Value> {
    static TRACE: OnceLock<Vec<Value>> = OnceLock::new();
    TRACE.get_or_init(|| {
        serde_json::from_str(include_str!("../../fixtures/e776-kenjo-trace.json"))
            .expect("embedded E776 trace must be valid JSON")
    })
}

#[derive(Clone, Debug, PartialEq)]
struct Action {
    farmer: Vec<Value>,
    hands: Vec<Vec<Value>>,
    market: Vec<Vec<Value>>,
}

impl Action {
    fn pass() -> Vec<Value> {
        vec![Value::String("PASS".into())]
    }

    fn from_value(value: &Value) -> Self {
        let object = value.as_object();
        let farmer = object
            .and_then(|v| v.get("farmer"))
            .and_then(Value::as_array)
            .filter(|v| !v.is_empty())
            .cloned()
            .unwrap_or_else(Self::pass);
        let hands = object
            .and_then(|v| v.get("hands"))
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.as_array()
                            .filter(|v| !v.is_empty())
                            .cloned()
                            .unwrap_or_else(Self::pass)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let market = object
            .and_then(|v| v.get("market"))
            .and_then(Value::as_array)
            .map(|rows| rows.iter().filter_map(Value::as_array).cloned().collect())
            .unwrap_or_default();
        Self {
            farmer,
            hands,
            market,
        }
    }

    fn into_value(self) -> Value {
        json!({"farmer": self.farmer, "hands": self.hands, "market": self.market})
    }

    fn commands(&self) -> Vec<Vec<Value>> {
        std::iter::once(self.farmer.clone())
            .chain(self.hands.clone())
            .collect()
    }

    fn set_commands(&mut self, commands: Vec<Vec<Value>>) {
        self.farmer = commands.first().cloned().unwrap_or_else(Self::pass);
        self.hands = commands.into_iter().skip(1).collect();
    }
}

fn text(value: &Value) -> Option<&str> {
    value.as_str()
}
fn command_op(command: &[Value]) -> Option<&str> {
    command.first().and_then(text)
}
fn int_value(value: Option<&Value>, default: i64) -> i64 {
    value
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|x| x as i64)))
        .unwrap_or(default)
}
fn int_or(value: Option<&Value>, default: i64) -> i64 {
    let parsed = int_value(value, default);
    if parsed == 0 { default } else { parsed }
}
fn float_or_zero(value: Option<&Value>) -> f64 {
    value.and_then(Value::as_f64).unwrap_or(0.0)
}
fn product(name: &str) -> bool {
    PRODUCTS.contains(&name)
}

fn source_trace_action(step: usize) -> Action {
    let rows = trace();
    Action::from_value(&rows[step.min(rows.len().saturating_sub(1))])
}

fn align_hands(mut action: Action, farm: &Farm) -> Action {
    action.hands.resize_with(farm.hands.len(), Action::pass);
    action.hands.truncate(farm.hands.len());
    action
}

fn tile_at<'a>(farm: &'a Farm, position: &[i64]) -> Option<&'a Value> {
    let (x, y) = (*position.first()?, *position.get(1)?);
    if x < 0 || y < 0 {
        return None;
    }
    farm.tiles.get(y as usize)?.get(x as usize)
}

fn tile_kind(tile: Option<&Value>) -> Option<&str> {
    tile?.as_object()?.get("kind")?.as_str()
}

fn is_empty_pasture(tile: Option<&Value>) -> bool {
    let Some(object) = tile.and_then(Value::as_object) else {
        return false;
    };
    object.get("kind").and_then(Value::as_str) == Some("PASTURE") && !object.contains_key("animal")
}

fn shape(kind: &str, value: f64, scale: f64) -> f64 {
    let value = value.max(0.0);
    match kind {
        "linear" => value,
        "sq" => value * value,
        "sqrt" => value.sqrt(),
        "log" => value.ln_1p(),
        "hinge" => {
            let ratio = value / scale;
            ratio + 8.0 * (ratio - 1.0).max(0.0).powi(2)
        }
        _ => value,
    }
}

fn price_params(item: &str) -> (f64, f64, &'static str, f64, &'static str, f64) {
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
        _ => (1.0, 1.0, "linear", 0.0, "linear", 0.0),
    }
}

fn price(item: &str, inventory: i64) -> i64 {
    let (base, scale, below_fn, below_target, above_fn, above_target) = price_params(item);
    let value = if inventory < I0 {
        let amplitude = below_target * base / shape(below_fn, scale, scale);
        base + amplitude * shape(below_fn, (I0 - inventory) as f64, scale)
    } else {
        let amplitude = above_target * base / shape(above_fn, scale, scale);
        base - amplitude * shape(above_fn, (inventory - I0) as f64, scale)
    };
    (value.round_ties_even() as i64).max(1)
}

fn contested_value(item: &str, inventory: i64, quantity: i64) -> i64 {
    let quantity = quantity.clamp(1, 60);
    let first: i64 = (0..quantity)
        .map(|offset| price(item, inventory + offset))
        .sum();
    let second: i64 = (0..quantity)
        .map(|offset| price(item, inventory + quantity + offset))
        .sum();
    first - second
}

fn fib(index: usize) -> i64 {
    let (mut a, mut b) = (1_i64, 1_i64);
    for _ in 0..index {
        (a, b) = (b, a + b);
    }
    a
}

fn seed_cost(item: &str) -> i64 {
    match item {
        "WHEAT" => 10,
        "CARROT" => 20,
        "TOMATO" => 50,
        "STRAWBERRY" => 100,
        "MELON" => 80,
        _ => 1_000_000_000,
    }
}
fn animal_cost(item: &str) -> i64 {
    match item {
        "GOOSE" => 300,
        "COW" => 400,
        "SHEEP" => 500,
        _ => 1_000_000_000,
    }
}

#[derive(Clone, Debug)]
struct WeedTransaction {
    start: usize,
    intended: Vec<Value>,
}

#[derive(Clone, Debug, Default)]
struct LatentState {
    active: bool,
    animal: Option<String>,
    cancelled: bool,
    cancel_reason: Option<String>,
    intervention_steps: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct E776Controller {
    last_step: i64,
    weed_transactions: IndexMap<usize, WeedTransaction>,
    assignments: IndexMap<String, String>,
    cow_to_sheep: i64,
    sheep_to_cow: i64,
    parent_latent: LatentState,
    exact_latent: LatentState,
    last_debug: Value,
}

impl Default for E776Controller {
    fn default() -> Self {
        Self {
            last_step: -1,
            weed_transactions: IndexMap::new(),
            assignments: IndexMap::new(),
            cow_to_sheep: 0,
            sheep_to_cow: 0,
            parent_latent: LatentState::default(),
            exact_latent: LatentState::default(),
            last_debug: Value::Null,
        }
    }
}

impl E776Controller {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn farm_private<'a>(
        &self,
        game: &'a Game,
        seat: usize,
    ) -> Result<(&'a Farm, &'a PrivateState), String> {
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("E776 seat {seat} missing farm"))?;
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("E776 seat {seat} missing private state"))?;
        Ok((farm, private))
    }

    fn pressure(&self, game: &Game) -> (f64, f64, f64) {
        let mut counts: IndexMap<&str, i64> = IndexMap::new();
        for shop in &game.town().unlocked_shops {
            *counts.entry(shop.as_str()).or_default() += 1;
        }
        let milk_price = float_or_zero(game.market().prices.get("MILK"));
        let wool_price = float_or_zero(game.market().prices.get("WOOL"));
        let wool = 2.0 * *counts.get("YARN_STORE").unwrap_or(&0) as f64 + wool_price / 200.0;
        let dairy = (*counts.get("PIZZA_SHOP").unwrap_or(&0)
            + *counts.get("ICE_CREAM_SHOP").unwrap_or(&0)
            + *counts.get("SMOOTHIE_SHOP").unwrap_or(&0)) as f64
            + milk_price / 160.0;
        (wool, dairy, wool - dairy)
    }

    fn assign_bundle(&mut self, game: &Game, step: usize) {
        let Some((name, source, quantity)) = purchase_bundle(step) else {
            return;
        };
        if self.assignments.contains_key(name) {
            return;
        }
        let (_, _, gap) = self.pressure(game);
        let mut target = source;
        if source == "COW" && gap >= 1.0 && self.cow_to_sheep + quantity <= 3 {
            target = "SHEEP";
            self.cow_to_sheep += quantity;
        } else if source == "SHEEP" && gap <= -1.0 && self.sheep_to_cow + quantity <= 1 {
            target = "COW";
            self.sheep_to_cow += quantity;
        }
        self.assignments.insert(name.into(), target.into());
    }

    fn replace_bundle(&self, mut action: Action, step: usize) -> Action {
        let Some((name, source)) = step_bundle(step) else {
            return action;
        };
        let target = self
            .assignments
            .get(name)
            .map(String::as_str)
            .unwrap_or(source);
        if target == source {
            return action;
        }
        for command in std::iter::once(&mut action.farmer).chain(action.hands.iter_mut()) {
            if matches!(command_op(command), Some("PICKUP" | "PLACE"))
                && command.get(1).and_then(text) == Some(source)
            {
                command[1] = Value::String(target.into());
            }
        }
        for order in &mut action.market {
            if command_op(order) == Some("BUY_ANIMAL")
                && order.get(1).and_then(text) == Some(source)
            {
                order[1] = Value::String(target.into());
            }
        }
        action
    }

    fn source_action(&self, game: &Game, seat: usize, step: usize) -> Result<Action, String> {
        let action = self.replace_bundle(source_trace_action(step), step);
        self.reallocate_animal_sales(game, seat, action)
    }

    #[allow(clippy::type_complexity)] // Literal return shape from the frozen controller.
    fn project_shed(
        &self,
        game: &Game,
        seat: usize,
        action: &Action,
    ) -> Result<(IndexMap<String, i64>, IndexMap<String, i64>), String> {
        let (farm, private) = self.farm_private(game, seat)?;
        let mut shed = private.shed.clone();
        for quantity in shed.values_mut() {
            *quantity = (*quantity).max(0);
        }
        let positions =
            std::iter::once(farm.farmer.as_slice()).chain(farm.hands.iter().map(Vec::as_slice));
        let commands =
            std::iter::once(action.farmer.as_slice()).chain(action.hands.iter().map(Vec::as_slice));
        let mut reserve: IndexMap<String, i64> = IndexMap::new();
        let board_size = farm.tiles.len().max(1) as i64;
        let half = board_size / 2;
        let cross = [
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        ];
        for (index, (position, command)) in positions.zip(commands).enumerate() {
            let point = (
                *position.first().unwrap_or(&-1),
                *position.get(1).unwrap_or(&-1),
            );
            if !cross.contains(&point) {
                continue;
            }
            match command_op(command) {
                Some("PICKUP") if command.len() >= 2 => {
                    let item = command[1].as_str().unwrap_or("").to_string();
                    let quantity = int_value(command.get(2), 1).max(1);
                    let taken = quantity.min(*shed.get(&item).unwrap_or(&0));
                    *shed.entry(item.clone()).or_default() -= taken;
                    *reserve.entry(item).or_default() += taken;
                }
                Some("DROP") => {
                    if let Some(inventory) = private.inventories.get(index) {
                        for (item, quantity) in inventory {
                            let room = (SHED_CAPACITY - shed.values().sum::<i64>()).max(0);
                            *shed.entry(item.clone()).or_default() += (*quantity).min(room);
                        }
                    }
                }
                Some("PLACE") if command.len() >= 2 => {
                    let item = command[1].as_str().unwrap_or("").to_string();
                    let quantity = int_value(command.get(2), 1).max(1);
                    let room = (SHED_CAPACITY - shed.values().sum::<i64>()).max(0);
                    let carried = private
                        .inventories
                        .get(index)
                        .and_then(|v| v.get(&item))
                        .copied()
                        .unwrap_or(0);
                    *shed.entry(item).or_default() += quantity.min(carried).min(room);
                }
                _ => {}
            }
        }
        Ok((shed, reserve))
    }

    fn reallocate_animal_sales(
        &self,
        game: &Game,
        seat: usize,
        mut action: Action,
    ) -> Result<Action, String> {
        if self.cow_to_sheep + self.sheep_to_cow == 0 {
            return Ok(action);
        }
        let positions: Vec<usize> = action
            .market
            .iter()
            .enumerate()
            .filter_map(|(index, order)| {
                (order.len() >= 3
                    && command_op(order) == Some("SELL")
                    && order
                        .get(1)
                        .and_then(text)
                        .is_some_and(|v| ANIMAL_PRODUCTS.contains(&v)))
                .then_some(index)
            })
            .collect();
        if positions.is_empty() {
            return Ok(action);
        }
        let (mut available, _) = self.project_shed(game, seat, &action)?;
        let mut inventory: IndexMap<&str, i64> = ANIMAL_PRODUCTS
            .into_iter()
            .map(|item| (item, int_or(game.market().inventory.get(item), I0)))
            .collect();
        let (wool, dairy, _) = self.pressure(game);
        for position in positions {
            let original = action.market[position][1]
                .as_str()
                .unwrap_or("")
                .to_string();
            let requested = int_value(action.market[position].get(2), 0).max(0);
            let mut best: Option<(i64, i64, f64, bool, &str)> = None;
            for item in ANIMAL_PRODUCTS {
                let quantity = requested.min((*available.get(item).unwrap_or(&0)).max(0));
                let revenue: i64 = (0..quantity)
                    .map(|offset| price(item, inventory[item] + offset))
                    .sum();
                let candidate = (
                    revenue,
                    quantity,
                    if item == "MILK" { dairy } else { wool },
                    item == original,
                    item,
                );
                if best.as_ref().is_none_or(|current| {
                    animal_candidate_cmp(&candidate, current) == Ordering::Greater
                }) {
                    best = Some(candidate);
                }
            }
            let (_, quantity, _, _, selected) = best.expect("two animal products");
            action.market[position][1] = Value::String(selected.into());
            *available.entry(selected.into()).or_default() -= quantity;
            *inventory.entry(selected).or_default() += quantity;
        }
        Ok(action)
    }

    fn weed_repair(
        &mut self,
        game: &Game,
        seat: usize,
        mut action: Action,
        step: usize,
    ) -> Result<Action, String> {
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("E776 seat {seat} missing farm"))?;
        action = align_hands(action, farm);
        let positions = std::iter::once(farm.farmer.clone())
            .chain(farm.hands.clone())
            .collect::<Vec<_>>();
        let mut commands = action.commands();
        let keys: Vec<usize> = self.weed_transactions.keys().copied().collect();
        for actor in keys {
            let Some(transaction) = self.weed_transactions.get(&actor).cloned() else {
                continue;
            };
            if actor >= commands.len() {
                self.weed_transactions.shift_remove(&actor);
                continue;
            }
            let age = step.saturating_sub(transaction.start);
            if age == 1 {
                commands[actor] = transaction.intended;
            } else if (2..=1 + WEED_REPLAY_STEPS).contains(&age) {
                let prior = self.source_action(game, seat, step.saturating_sub(1))?;
                commands[actor] = if actor == 0 {
                    prior.farmer
                } else {
                    prior
                        .hands
                        .get(actor - 1)
                        .cloned()
                        .unwrap_or_else(Action::pass)
                };
            } else {
                self.weed_transactions.shift_remove(&actor);
            }
        }
        for (index, (position, intended)) in positions.iter().zip(commands.iter_mut()).enumerate() {
            if self.weed_transactions.contains_key(&index) {
                continue;
            }
            if !matches!(
                command_op(intended),
                Some("BUILD_PASTURE" | "BUILD_COOP" | "PLANT")
            ) {
                continue;
            }
            if tile_kind(tile_at(farm, position)) != Some("WEED") {
                continue;
            }
            self.weed_transactions.insert(
                index,
                WeedTransaction {
                    start: step,
                    intended: intended.clone(),
                },
            );
            *intended = vec![Value::String("DIG".into())];
        }
        action.set_commands(commands);
        Ok(align_hands(action, farm))
    }

    fn cap_sales(&self, game: &Game, seat: usize, mut action: Action) -> Result<Action, String> {
        let (mut available, _) = self.project_shed(game, seat, &action)?;
        let mut capped = Vec::new();
        for mut order in action.market {
            if order.len() >= 3
                && command_op(&order) == Some("SELL")
                && order.get(1).and_then(text).is_some_and(product)
            {
                let item = order[1].as_str().unwrap().to_string();
                let requested = int_value(order.get(2), 0).max(0);
                let quantity = requested.min((*available.get(&item).unwrap_or(&0)).max(0));
                *available.entry(item).or_default() -= quantity;
                if quantity > 0 {
                    order[2] = Value::from(quantity);
                    capped.push(order);
                }
            } else {
                capped.push(order);
            }
        }
        capped.truncate(MAX_MARKET_ORDERS);
        action.market = capped;
        Ok(action)
    }

    fn assign_sell_slots(&self, game: &Game, mut action: Action) -> Action {
        let positions: Vec<usize> = action
            .market
            .iter()
            .enumerate()
            .filter_map(|(index, order)| {
                (order.len() >= 3
                    && command_op(order) == Some("SELL")
                    && order.get(1).and_then(text).is_some_and(product))
                .then_some(index)
            })
            .collect();
        if positions.len() < 2 {
            return action;
        }
        let mut ranked: Vec<Vec<Value>> = positions
            .iter()
            .map(|index| action.market[*index].clone())
            .collect();
        ranked.sort_by(|left, right| {
            let score = |order: &[Value]| {
                let item = order[1].as_str().unwrap_or("");
                let quantity = int_or(order.get(2), 1).max(1);
                let current = int_or(game.market().inventory.get(item), I0);
                (
                    contested_value(item, current, quantity),
                    price(item, current),
                    item.to_string(),
                )
            };
            let a = score(left);
            let b = score(right);
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        for (position, order) in positions.into_iter().zip(ranked) {
            action.market[position] = order;
        }
        action
    }

    fn fund_market(&self, game: &Game, seat: usize, mut action: Action) -> Result<Action, String> {
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("E776 seat {seat} missing farm"))?;
        let mut cash = farm.money;
        let mut hires = farm.hires_today;
        let mut unlocked = farm.unlocked_quadrants.len();
        let (mut shed, _) = self.project_shed(game, seat, &action)?;
        let mut shed_units: i64 = shed.values().sum();
        let mut inventory: IndexMap<String, i64> = PRODUCTS
            .into_iter()
            .map(|item| (item.into(), int_or(game.market().inventory.get(item), I0)))
            .collect();
        let mut output = Vec::new();
        for mut order in action.market.into_iter().take(MAX_MARKET_ORDERS) {
            if order.is_empty() {
                continue;
            }
            let op = command_op(&order).unwrap_or("").to_string();
            let item = order.get(1).and_then(text).unwrap_or("").to_string();
            if op == "SELL" && product(&item) && order.len() >= 3 {
                let requested = int_value(order.get(2), 0).max(0);
                let quantity = requested.min((*shed.get(&item).unwrap_or(&0)).max(0));
                let mut proceeds = 0;
                for _ in 0..quantity {
                    proceeds += price(&item, inventory[&item]);
                    *inventory.get_mut(&item).unwrap() += 1;
                }
                *shed.entry(item).or_default() -= quantity;
                shed_units -= quantity;
                cash += proceeds as f64;
                if quantity > 0 {
                    order[2] = Value::from(quantity);
                    output.push(order);
                }
                continue;
            }
            if op == "HIRE" {
                let cost = fib(hires);
                if cash >= cost as f64 {
                    cash -= cost as f64;
                    hires += 1;
                    output.push(order);
                }
                continue;
            }
            if op == "BUY_LAND" {
                let index = unlocked.saturating_sub(1);
                let cost = [1_000_i64, 2_000, 4_000]
                    .get(index)
                    .copied()
                    .unwrap_or(1_000_000_000);
                if index < 3 && cash >= cost as f64 {
                    cash -= cost as f64;
                    unlocked += 1;
                    output.push(order);
                }
                continue;
            }
            if !matches!(op.as_str(), "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL") || order.len() < 3
            {
                output.push(order);
                continue;
            }
            let requested = int_value(order.get(2), 0).max(0);
            let mut quantity = 0;
            for _ in 0..requested {
                let (cost, needs_shed) = match op.as_str() {
                    "BUY_SEED" => (seed_cost(&item), false),
                    "BUY_ANIMAL" => (animal_cost(&item), true),
                    _ => (
                        price(&item, inventory.get(&item).copied().unwrap_or(I0) - 1),
                        true,
                    ),
                };
                if cash < cost as f64 || (needs_shed && shed_units >= SHED_CAPACITY) {
                    break;
                }
                cash -= cost as f64;
                quantity += 1;
                if needs_shed {
                    shed_units += 1;
                }
                if op == "BUY_PRODUCT" {
                    *inventory.entry(item.clone()).or_insert(I0) -= 1;
                }
            }
            if quantity > 0 {
                order[2] = Value::from(quantity);
                output.push(order);
            }
        }
        output.truncate(MAX_MARKET_ORDERS);
        action.market = output;
        Ok(action)
    }

    fn terminal_frontier(
        &self,
        game: &Game,
        seat: usize,
        mut action: Action,
        step: usize,
    ) -> Result<Action, String> {
        if step != FINAL_EXECUTABLE_STEP {
            return Ok(action);
        }
        let (mut available, _) = self.project_shed(game, seat, &action)?;
        for order in &action.market {
            if order.len() >= 3 && command_op(order) == Some("SELL") {
                let item = order[1].as_str().unwrap_or("").to_string();
                let quantity = int_value(order.get(2), 0).max(0);
                let current = available.get(&item).copied().unwrap_or(0);
                *available.entry(item).or_default() = (current - quantity).max(0);
            }
        }
        let mut ranked: Vec<(i64, &str, i64)> = Vec::new();
        for item in ANIMAL_PRODUCTS {
            let quantity = (*available.get(item).unwrap_or(&0)).max(0);
            if quantity == 0 {
                continue;
            }
            let inventory = int_or(game.market().inventory.get(item), I0);
            let revenue = (0..quantity)
                .map(|offset| price(item, inventory + offset))
                .sum();
            ranked.push((revenue, item, quantity));
        }
        ranked.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        let room = MAX_MARKET_ORDERS.saturating_sub(action.market.len());
        action
            .market
            .extend(ranked.into_iter().take(room).map(|(_, item, quantity)| {
                vec![
                    Value::String("SELL".into()),
                    Value::String(item.into()),
                    Value::from(quantity),
                ]
            }));
        Ok(action)
    }

    fn activation_candidate(
        &self,
        game: &Game,
        seat: usize,
        action: &Action,
    ) -> Result<Option<(usize, String)>, String> {
        let (farm, private) = self.farm_private(game, seat)?;
        if !is_empty_pasture(
            farm.tiles
                .get(TARGET_TILE.1)
                .and_then(|row| row.get(TARGET_TILE.0)),
        ) {
            return Ok(None);
        }
        let positions = std::iter::once(farm.farmer.as_slice())
            .chain(farm.hands.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let mut cross_counts: BTreeMap<(i64, i64), usize> = BTreeMap::new();
        for position in &positions {
            let point = (
                *position.first().unwrap_or(&-1),
                *position.get(1).unwrap_or(&-1),
            );
            if [(4, 4), (5, 4), (4, 5), (5, 5)].contains(&point) {
                *cross_counts.entry(point).or_default() += 1;
            }
        }
        if positions.len() != 8
            || farm.hires_today != 7
            || [(4, 4), (5, 4), (4, 5), (5, 5)]
                .into_iter()
                .any(|point| cross_counts.get(&point) != Some(&2))
        {
            return Ok(None);
        }
        if action.market.len() >= MAX_MARKET_ORDERS {
            return Ok(None);
        }
        let purchases: Vec<(usize, String)> = action
            .market
            .iter()
            .enumerate()
            .filter_map(|(index, order)| {
                let animal = order.get(1).and_then(text)?;
                (order.len() >= 3
                    && command_op(order) == Some("BUY_ANIMAL")
                    && matches!(animal, "COW" | "SHEEP")
                    && int_value(order.get(2), 0) == 1)
                    .then(|| (index, animal.to_string()))
            })
            .collect();
        if purchases.len() != 1 {
            return Ok(None);
        }
        let hire = vec![Value::String("HIRE".into())];
        if action.market.iter().filter(|order| **order == hire).count() != 3 {
            return Ok(None);
        }
        let (purchase_index, animal) = purchases[0].clone();
        if farm.money < (animal_cost(&animal) + 89 + 1_000) as f64 {
            return Ok(None);
        }
        if private.shed.values().map(|v| (*v).max(0)).sum::<i64>() > 90 {
            return Ok(None);
        }
        Ok(Some((purchase_index, animal)))
    }

    fn parent_activation(
        &mut self,
        game: &Game,
        seat: usize,
        mut action: Action,
        step: usize,
    ) -> Result<Action, String> {
        if step == ACTIVATION_STEP {
            if let Some((index, animal)) = self.activation_candidate(game, seat, &action)? {
                action.market[index][2] = Value::from(2);
                action.market.push(vec![Value::String("HIRE".into())]);
                self.parent_latent.active = true;
                self.parent_latent.animal = Some(animal);
                self.parent_latent.intervention_steps.push(step);
            }
            return Ok(action);
        }
        if !self.parent_latent.active || !(314..=317).contains(&step) {
            return Ok(action);
        }
        delivery_command(game, seat, action, step, &mut self.parent_latent, false)
    }

    fn exact_delivery(
        &mut self,
        game: &Game,
        seat: usize,
        action: Action,
        step: usize,
    ) -> Result<Action, String> {
        if step == ACTIVATION_STEP {
            if self.parent_latent.active {
                self.exact_latent.active = true;
                self.exact_latent.animal = self.parent_latent.animal.clone();
                self.exact_latent.intervention_steps.push(step);
            }
            return Ok(action);
        }
        if !self.exact_latent.active || !(314..=316).contains(&step) {
            return Ok(action);
        }
        delivery_command(game, seat, action, step, &mut self.exact_latent, true)
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        let step = game.step_index().min(MAX_STEPS - 1);
        if step == 0 || step as i64 <= self.last_step {
            self.reset();
        }
        self.assign_bundle(game, step);
        let farm = game
            .farms()
            .get(seat)
            .ok_or_else(|| format!("E776 seat {seat} missing farm"))?;
        let mut action = self.source_action(game, seat, step)?;
        action = self.weed_repair(game, seat, action, step)?;
        action = self.cap_sales(game, seat, action)?;
        action = self.assign_sell_slots(game, action);
        action = self.fund_market(game, seat, action)?;
        action = align_hands(action, farm);
        action = self.terminal_frontier(game, seat, action, step)?;
        action = self.parent_activation(game, seat, action, step)?;
        action = self.exact_delivery(game, seat, action, step)?;
        self.last_step = step as i64;
        let (wool, dairy, gap) = self.pressure(game);
        self.last_debug = json!({
            "step": step,
            "assignments": self.assignments,
            "cow_to_sheep": self.cow_to_sheep,
            "sheep_to_cow": self.sheep_to_cow,
            "pressure": {"wool": wool, "dairy": dairy, "gap": gap},
            "active_weed_transactions": self.weed_transactions.len(),
            "parent_latent": {"active": self.parent_latent.active, "animal": self.parent_latent.animal, "cancelled": self.parent_latent.cancelled, "reason": self.parent_latent.cancel_reason, "steps": self.parent_latent.intervention_steps},
            "exact_latent": {"active": self.exact_latent.active, "animal": self.exact_latent.animal, "cancelled": self.exact_latent.cancelled, "reason": self.exact_latent.cancel_reason, "steps": self.exact_latent.intervention_steps},
        });
        Ok(action.into_value())
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}

fn purchase_bundle(step: usize) -> Option<(&'static str, &'static str, i64)> {
    match step {
        88 => Some(("cow88", "COW", 1)),
        150 => Some(("cow150", "COW", 2)),
        169 => Some(("cow169", "COW", 1)),
        176 => Some(("cow176", "COW", 1)),
        313 => Some(("sheep313", "SHEEP", 1)),
        _ => None,
    }
}

fn step_bundle(step: usize) -> Option<(&'static str, &'static str)> {
    match step {
        88 | 92 | 95 => Some(("cow88", "COW")),
        150 | 152 | 153 | 156 => Some(("cow150", "COW")),
        169 | 175 | 177 => Some(("cow169", "COW")),
        176 | 180 | 183 => Some(("cow176", "COW")),
        313 | 329 | 333 => Some(("sheep313", "SHEEP")),
        _ => None,
    }
}

fn animal_candidate_cmp(
    left: &(i64, i64, f64, bool, &str),
    right: &(i64, i64, f64, bool, &str),
) -> Ordering {
    left.0
        .cmp(&right.0)
        .then_with(|| left.1.cmp(&right.1))
        .then_with(|| left.2.partial_cmp(&right.2).unwrap_or(Ordering::Equal))
        .then_with(|| left.3.cmp(&right.3))
        .then_with(|| left.4.cmp(right.4))
}

fn cancel(state: &mut LatentState, reason: String) {
    state.active = false;
    state.cancelled = true;
    state.cancel_reason = Some(reason);
}

fn delivery_command(
    game: &Game,
    seat: usize,
    mut action: Action,
    step: usize,
    state: &mut LatentState,
    exact: bool,
) -> Result<Action, String> {
    let farm = game
        .farms()
        .get(seat)
        .ok_or_else(|| format!("E776 seat {seat} missing farm"))?;
    let private = game
        .privates()
        .get(seat)
        .ok_or_else(|| format!("E776 seat {seat} missing private state"))?;
    if farm.hands.len() != 11 || action.hands.len() != 11 || private.inventories.len() < 12 {
        cancel(state, "missing appended hand".into());
        return Ok(action);
    }
    if action.hands.last() != Some(&Action::pass()) {
        cancel(state, "parent claimed appended hand".into());
        return Ok(action);
    }
    let animal = state.animal.clone().unwrap_or_default();
    let position = &farm.hands[10];
    let inventory = &private.inventories[11];
    let (expected_position, expected_inventory, command): (
        (i64, i64),
        IndexMap<String, i64>,
        Vec<Value>,
    ) = if exact {
        match step {
            314 => (
                (5, 4),
                IndexMap::new(),
                json!(["PICKUP", animal, 1]).as_array().unwrap().clone(),
            ),
            315 => (
                (5, 4),
                IndexMap::from([(animal.clone(), 1)]),
                json!(["NORTH"]).as_array().unwrap().clone(),
            ),
            316 => (
                (5, 3),
                IndexMap::from([(animal.clone(), 1)]),
                json!(["PLACE", animal]).as_array().unwrap().clone(),
            ),
            _ => return Ok(action),
        }
    } else {
        match step {
            314 => (
                (5, 5),
                IndexMap::new(),
                json!(["PICKUP", animal, 1]).as_array().unwrap().clone(),
            ),
            315 => (
                (5, 5),
                IndexMap::from([(animal.clone(), 1)]),
                json!(["NORTH"]).as_array().unwrap().clone(),
            ),
            316 => (
                (5, 4),
                IndexMap::from([(animal.clone(), 1)]),
                json!(["NORTH"]).as_array().unwrap().clone(),
            ),
            317 => (
                (5, 3),
                IndexMap::from([(animal.clone(), 1)]),
                json!(["PLACE", animal]).as_array().unwrap().clone(),
            ),
            _ => return Ok(action),
        }
    };
    let current_position = (
        *position.first().unwrap_or(&-1),
        *position.get(1).unwrap_or(&-1),
    );
    if current_position != expected_position || inventory != &expected_inventory {
        cancel(
            state,
            format!(
                "visible {}delivery mismatch at step {step}",
                if exact { "corrected-" } else { "" }
            ),
        );
        return Ok(action);
    }
    let place_step = if exact { 316 } else { 317 };
    if step == place_step
        && !is_empty_pasture(
            farm.tiles
                .get(TARGET_TILE.1)
                .and_then(|row| row.get(TARGET_TILE.0)),
        )
    {
        cancel(state, "target pasture no longer empty".into());
        return Ok(action);
    }
    action.hands[10] = command;
    state.intervention_steps.push(step);
    Ok(action)
}
