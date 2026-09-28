//! Complete native port of `evgendvorkin/kaggriculture`.
//!
//! The public agent is a stateful guard layer over a frozen 720-step c27 action
//! tape.  This module deliberately keeps the tape, clone detector, weed-repair
//! transactions, premium-sale front running, late liquidation, and final-three
//! observation-driven actions together.  It never calls Python in the rollout
//! hot path.

use crate::{Farm, Game, PrivateState};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const MAX_STEPS: usize = 720;
const MAX_MARKET_ORDERS: usize = 10;
const WEED_REPLAY_STEPS: usize = 8;
const FRONT_RUN_HORIZON: usize = 1;
const I0: i64 = 10_000;
const SELLABLE: [&str; 9] = [
    "STRAWBERRY",
    "MELON",
    "MILK",
    "WOOL",
    "EGG",
    "TOMATO",
    "CARROT",
    "WHEAT",
    "FERTILIZER",
];
const FRONT_RUN_ITEMS: [&str; 4] = ["MELON", "STRAWBERRY", "MILK", "WOOL"];

fn trace() -> &'static Vec<Value> {
    static TRACE: OnceLock<Vec<Value>> = OnceLock::new();
    TRACE.get_or_init(|| {
        serde_json::from_str(include_str!("../../fixtures/evgen-c27-trace.json"))
            .expect("embedded Evgen c27 trace must be valid JSON")
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

fn command_op(command: &[Value]) -> Option<&str> {
    command.first().and_then(Value::as_str)
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

fn tile_object(tile: Option<&Value>) -> Option<&serde_json::Map<String, Value>> {
    tile?.as_object()
}

fn tile_int(tile: Option<&Value>, key: &str) -> i64 {
    int_value(tile_object(tile).and_then(|object| object.get(key)), 0)
}

fn base_price(item: &str) -> f64 {
    match item {
        "MELON" => 250.0,
        "STRAWBERRY" => 120.0,
        "MILK" => 160.0,
        "WOOL" => 200.0,
        _ => 0.0,
    }
}

fn glut_weight(item: &str) -> f64 {
    match item {
        "MELON" => 3.5,
        "STRAWBERRY" | "MILK" => 2.0,
        "WOOL" => 3.2,
        _ => 0.0,
    }
}

fn market_shape(kind: &str, value: f64) -> f64 {
    match kind {
        "linear" => value,
        "sq" => value * value,
        "sqrt" => value.sqrt(),
        "log10" => (1.0 + value).log10(),
        _ => value.ln_1p(),
    }
}

fn market_params(item: &str) -> (f64, f64, &'static str, f64, &'static str, f64) {
    match item {
        "WHEAT" => (25.0, 400.0, "sqrt", 0.80, "log", 0.20),
        "CARROT" => (35.0, 450.0, "log", 0.20, "sqrt", 0.70),
        "TOMATO" => (60.0, 200.0, "linear", 0.40, "sqrt", 0.60),
        "STRAWBERRY" => (120.0, 100.0, "sqrt", 0.70, "linear", 1.60),
        "MELON" => (250.0, 300.0, "log", 0.20, "sq", 3.60),
        "EGG" => (50.0, 332.0, "linear", 0.40, "log", 0.20),
        "MILK" => (160.0, 122.0, "sqrt", 0.60, "linear", 1.60),
        "WOOL" => (200.0, 105.0, "log", 0.20, "sq", 3.20),
        "FERTILIZER" => (100.0, 200.0, "linear", 0.40, "linear", 0.40),
        _ => (1.0, 1.0, "linear", 0.0, "linear", 0.0),
    }
}

/// Literal copy of this anchor's private `_mprice`, including Python's
/// round-to-even integer conversion and its hard-coded market parameters.
fn market_price(item: &str, inventory: i64) -> i64 {
    let (base, throughput, below_fn, below_target, above_fn, above_target) = market_params(item);
    let value = if inventory < I0 {
        let amplitude = below_target * base / market_shape(below_fn, throughput);
        base + amplitude * market_shape(below_fn, (I0 - inventory) as f64)
    } else {
        let amplitude = above_target * base / market_shape(above_fn, throughput);
        base - amplitude * market_shape(above_fn, (inventory - I0) as f64)
    };
    (value.round_ties_even() as i64).max(1)
}

#[derive(Clone, Debug)]
struct WeedTransaction {
    start: usize,
    intended: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct EvgenController {
    last_step: i64,
    clone_confidence: i64,
    weed_last_step: i64,
    weed_transactions: IndexMap<usize, WeedTransaction>,
    last_debug: Value,
}

impl Default for EvgenController {
    fn default() -> Self {
        Self {
            last_step: -1,
            clone_confidence: 0,
            weed_last_step: -1,
            weed_transactions: IndexMap::new(),
            last_debug: Value::Null,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PublicSignature {
    hands: usize,
    quadrants: Vec<String>,
    positions: Vec<Vec<i64>>,
    counts: BTreeMap<&'static str, i64>,
}

fn public_signature(farm: &Farm) -> PublicSignature {
    const OBJECTS: [&str; 11] = [
        "COW",
        "SHEEP",
        "GOOSE",
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "PASTURE",
        "COOP",
        "WEED",
    ];
    let mut counts: BTreeMap<&'static str, i64> =
        OBJECTS.into_iter().map(|name| (name, 0)).collect();
    for tile in farm.tiles.iter().flatten() {
        let Some(object) = tile.as_object() else {
            continue;
        };
        for key in ["animal", "crop", "kind"] {
            let Some(value) = object.get(key).and_then(Value::as_str) else {
                continue;
            };
            if let Some(count) = counts.get_mut(value) {
                *count += 1;
                break;
            }
        }
    }
    let mut quadrants = farm.unlocked_quadrants.clone();
    quadrants.sort();
    let mut positions: Vec<Vec<i64>> = std::iter::once(farm.farmer.clone())
        .chain(farm.hands.clone())
        .collect();
    positions.sort();
    PublicSignature {
        hands: farm.hands.len(),
        quadrants,
        positions,
        counts,
    }
}

fn signature_distance(left: &PublicSignature, right: &PublicSignature) -> i64 {
    let mut distance = left.hands.abs_diff(right.hands) as i64;
    distance += 3 * left.quadrants.len().abs_diff(right.quadrants.len()) as i64;
    distance += left
        .counts
        .iter()
        .map(|(name, count)| (count - right.counts.get(name).copied().unwrap_or(0)).abs())
        .sum::<i64>();
    if left.positions != right.positions {
        distance += 2;
    }
    distance
}

impl EvgenController {
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
            .ok_or_else(|| format!("Evgen seat {seat} missing farm"))?;
        let private = game
            .privates()
            .get(seat)
            .ok_or_else(|| format!("Evgen seat {seat} missing private state"))?;
        Ok((farm, private))
    }

    fn update_clone_profile(&mut self, game: &Game, seat: usize, step: usize) {
        if step != 4 && step != 24 && !(step >= 48 && step.is_multiple_of(24)) {
            return;
        }
        let farms = game.farms();
        if farms.len() < 2 || seat >= farms.len() {
            return;
        }
        let distance = signature_distance(
            &public_signature(&farms[seat]),
            &public_signature(&farms[1 - seat]),
        );
        if distance <= 1 {
            self.clone_confidence = (self.clone_confidence + 1).min(8);
        } else if distance <= 4 {
            self.clone_confidence = (self.clone_confidence - 1).max(0);
        } else {
            self.clone_confidence = (self.clone_confidence - 3).max(0);
        }
    }

    fn front_run(
        &self,
        game: &Game,
        seat: usize,
        action: &mut Action,
        step: usize,
    ) -> Result<(), String> {
        if self.clone_confidence < 2
            || FRONT_RUN_HORIZON == 0
            || action.market.len() >= MAX_MARKET_ORDERS
        {
            return Ok(());
        }
        let (_, private) = self.farm_private(game, seat)?;
        let mut already: BTreeMap<&str, i64> = BTreeMap::new();
        for order in &action.market {
            if order.len() >= 3
                && command_op(order) == Some("SELL")
                && let Some(item) = order.get(1).and_then(Value::as_str)
            {
                *already.entry(item).or_default() += int_value(order.get(2), 0).max(0);
            }
        }

        let mut planned: BTreeMap<String, (usize, i64)> = BTreeMap::new();
        let end = trace().len().min(step + FRONT_RUN_HORIZON + 1);
        for future_step in step + 1..end {
            let future = source_trace_action(future_step);
            for order in &future.market {
                let Some(item) = order.get(1).and_then(Value::as_str) else {
                    continue;
                };
                if order.len() < 3
                    || command_op(order) != Some("SELL")
                    || !FRONT_RUN_ITEMS.contains(&item)
                {
                    continue;
                }
                let quantity = int_value(order.get(2), 0).max(0);
                planned
                    .entry(item.to_string())
                    .and_modify(|row| row.1 += quantity)
                    .or_insert((future_step - step, quantity));
            }
        }

        let mut choices: Vec<(f64, String, i64)> = Vec::new();
        for (item, (distance, planned_quantity)) in planned {
            let available = private.shed.get(&item).copied().unwrap_or(0).max(0)
                - already.get(item.as_str()).copied().unwrap_or(0);
            let quantity = available.max(0).min(planned_quantity);
            if quantity <= 0 {
                continue;
            }
            let price = game
                .market()
                .prices
                .get(&item)
                .and_then(Value::as_f64)
                .unwrap_or_else(|| base_price(&item));
            let priority = price * quantity as f64 * glut_weight(&item)
                + (FRONT_RUN_HORIZON + 1 - distance) as f64 * base_price(&item);
            choices.push((priority, item, quantity));
        }
        if let Some((_, item, quantity)) = choices.into_iter().max_by(front_run_cmp) {
            action
                .market
                .push(json!(["SELL", item, quantity]).as_array().unwrap().clone());
        }
        Ok(())
    }

    fn terminal_liquidation(
        &self,
        game: &Game,
        seat: usize,
        action: &mut Action,
        step: usize,
    ) -> Result<(), String> {
        if step < 680 {
            return Ok(());
        }
        let (_, private) = self.farm_private(game, seat)?;
        let already: BTreeSet<String> = action
            .market
            .iter()
            .filter(|order| order.len() >= 2 && command_op(order) == Some("SELL"))
            .filter_map(|order| order.get(1).and_then(Value::as_str).map(str::to_string))
            .collect();
        for item in SELLABLE {
            let quantity = private.shed.get(item).copied().unwrap_or(0);
            if quantity > 0 && !already.contains(item) && action.market.len() < MAX_MARKET_ORDERS {
                action
                    .market
                    .push(json!(["SELL", item, quantity]).as_array().unwrap().clone());
            }
        }
        Ok(())
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
            .ok_or_else(|| format!("Evgen seat {seat} missing farm"))?;
        action = align_hands(action, farm);
        if step == 0 || (step as i64) < self.weed_last_step {
            self.weed_transactions.clear();
        }
        self.weed_last_step = step as i64;
        let positions = std::iter::once(farm.farmer.clone())
            .chain(farm.hands.clone())
            .collect::<Vec<_>>();
        let mut commands = action.commands();

        let active_actors: Vec<usize> = self.weed_transactions.keys().copied().collect();
        for actor in active_actors {
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
                let prior = source_trace_action(step.saturating_sub(1));
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
            if self.weed_transactions.contains_key(&index)
                || !matches!(command_op(intended), Some("BUILD_PASTURE" | "PLANT"))
            {
                continue;
            }
            let is_weed = tile_object(tile_at(farm, position))
                .and_then(|tile| tile.get("kind"))
                .and_then(Value::as_str)
                == Some("WEED");
            if !is_weed {
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

    fn sort_market(&self, game: &Game, seat: usize, mut action: Action) -> Action {
        // `_RESERVE` is frozen empty in the public submission, so its planner
        // contributes no replacement rows.  The live behavior is precisely a
        // stable, descending-impact promotion of premium SELLs ahead of `keep`.
        let private = game.privates().get(seat);
        let mut promoted = Vec::new();
        let mut rest = Vec::new();
        for order in action.market {
            let is_promoted = command_op(&order) == Some("SELL")
                && order
                    .get(1)
                    .and_then(Value::as_str)
                    .is_some_and(|item| FRONT_RUN_ITEMS.contains(&item));
            if is_promoted {
                promoted.push(order);
            } else {
                rest.push(order);
            }
        }
        promoted.sort_by(|left, right| {
            sell_priority(right, game, private).total_cmp(&sell_priority(left, game, private))
        });
        promoted.extend(rest);
        promoted.truncate(MAX_MARKET_ORDERS);
        action.market = promoted;
        action
    }

    fn terminal_action(&self, game: &Game, seat: usize) -> Result<Action, String> {
        let (farm, private) = self.farm_private(game, seat)?;
        let positions = std::iter::once(farm.farmer.clone())
            .chain(farm.hands.clone())
            .collect::<Vec<_>>();
        let mut inventories = private.inventories.clone();
        inventories.resize_with(positions.len(), IndexMap::new);
        let sheds = shed_access(farm.tiles.len());
        let mut available: BTreeSet<(i64, i64)> = farm
            .tiles
            .iter()
            .enumerate()
            .flat_map(|(y, row)| {
                row.iter().enumerate().filter_map(move |(x, tile)| {
                    (tile_int(Some(tile), "yield_units") > 0).then_some((x as i64, y as i64))
                })
            })
            .collect();

        let mut actions = Vec::with_capacity(positions.len());
        let mut pending: BTreeMap<String, i64> = BTreeMap::new();
        for (position, inventory) in positions.iter().zip(inventories.iter()) {
            let point = (
                *position.first().unwrap_or(&0),
                *position.get(1).unwrap_or(&0),
            );
            let load: i64 = inventory.values().map(|quantity| (*quantity).max(0)).sum();
            let tile = tile_at(farm, position);
            let command = if load > 0 && sheds.contains(&point) {
                for (item, quantity) in inventory {
                    if SELLABLE.contains(&item.as_str()) {
                        *pending.entry(item.clone()).or_default() += (*quantity).max(0);
                    }
                }
                vec![Value::String("DROP".into())]
            } else if tile_int(tile, "yield_units") > 0 {
                available.remove(&point);
                vec![Value::String("HARVEST".into())]
            } else if load > 0 {
                let target = python_min_shed(point, farm.tiles.len(), &sheds)
                    .expect("a non-empty board always has shed access tiles");
                move_toward(point, target, &farm.tiles)
            } else if !available.is_empty() {
                let target = *available
                    .iter()
                    .min_by_key(|(x, y)| ((x - point.0).abs() + (y - point.1).abs(), *y, *x))
                    .expect("checked non-empty");
                available.remove(&target);
                move_toward(point, target, &farm.tiles)
            } else if tile_object(tile)
                .and_then(|object| object.get("fertilizer_available"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                vec![Value::String("COLLECT_FERTILIZER".into())]
            } else {
                Action::pass()
            };
            actions.push(command);
        }

        let mut shed = private.shed.clone();
        for (item, quantity) in pending {
            *shed.entry(item).or_default() += quantity;
        }
        let mut sells: Vec<(i64, &str, i64)> = SELLABLE
            .into_iter()
            .filter_map(|item| {
                let quantity = shed.get(item).copied().unwrap_or(0);
                (quantity > 0).then(|| {
                    let price = int_or(game.market().prices.get(item), 1);
                    (quantity * price, item, quantity)
                })
            })
            .collect();
        sells.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.cmp(left.1))
                .then_with(|| right.2.cmp(&left.2))
        });
        let mut market: Vec<Vec<Value>> = sells
            .into_iter()
            .take(MAX_MARKET_ORDERS)
            .map(|(_, item, quantity)| json!(["SELL", item, quantity]).as_array().unwrap().clone())
            .collect();
        if game.public_state().hour <= 1 {
            let hires = (8_usize.saturating_sub(farm.hires_today))
                .min(MAX_MARKET_ORDERS.saturating_sub(market.len()));
            market.extend((0..hires).map(|_| vec![Value::String("HIRE".into())]));
        }
        Ok(Action {
            farmer: actions.first().cloned().unwrap_or_else(Action::pass),
            hands: actions.into_iter().skip(1).collect(),
            market,
        })
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        let step = game.step_index().min(MAX_STEPS - 1);
        if step == 0 || (step as i64) <= self.last_step {
            self.reset();
        }
        self.update_clone_profile(game, seat, step);
        let action = if step >= 717 {
            self.terminal_action(game, seat)?
        } else {
            let mut action = source_trace_action(step);
            self.front_run(game, seat, &mut action, step)?;
            self.terminal_liquidation(game, seat, &mut action, step)?;
            let action = self.weed_repair(game, seat, action, step)?;
            self.sort_market(game, seat, action)
        };
        self.last_step = step as i64;
        self.last_debug = json!({
            "step": step,
            "clone_confidence": self.clone_confidence,
            "active_weed_transactions": self.weed_transactions.len(),
        });
        Ok(action.into_value())
    }

    pub fn debug(&self) -> &Value {
        &self.last_debug
    }
}

fn front_run_cmp(left: &(f64, String, i64), right: &(f64, String, i64)) -> Ordering {
    left.0
        .total_cmp(&right.0)
        .then_with(|| left.1.cmp(&right.1))
        .then_with(|| left.2.cmp(&right.2))
}

fn sell_priority(order: &[Value], game: &Game, private: Option<&PrivateState>) -> f64 {
    if order.len() < 3 || command_op(order) != Some("SELL") {
        return -1.0;
    }
    let Some(item) = order.get(1).and_then(Value::as_str) else {
        return -1.0;
    };
    if !market_params_known(item) {
        return -1.0;
    }
    let mut quantity = int_value(order.get(2), 0);
    if quantity <= 0 {
        return -1.0;
    }
    let inventory = int_or(game.market().inventory.get(item), I0);
    let held = private
        .and_then(|state| state.shed.get(item))
        .copied()
        .unwrap_or(0);
    if held > 0 {
        quantity = quantity.min(held);
    }
    quantity as f64
        * (market_price(item, inventory) - market_price(item, inventory + quantity)) as f64
}

fn market_params_known(item: &str) -> bool {
    SELLABLE.contains(&item)
}

fn shed_access(size: usize) -> BTreeSet<(i64, i64)> {
    let half = size as i64 / 2;
    [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
    .into_iter()
    .collect()
}

/// Python's `set` iteration is observable because `_terminal_action` calls
/// `min(sheds, key=distance)` without a coordinate tie-break.  For integer
/// tuples under the notebook's CPython 3.12 runtime, this reproduces that order
/// for the four center cells instead of silently imposing Rust's BTree order.
fn python_shed_iteration(size: usize) -> [(i64, i64); 4] {
    let half = size as i64 / 2;
    match half {
        1 => [
            (half, half - 1),
            (half - 1, half),
            (half, half),
            (half - 1, half - 1),
        ],
        2 => [
            (half - 1, half - 1),
            (half - 1, half),
            (half, half - 1),
            (half, half),
        ],
        3 => [
            (half - 1, half),
            (half, half - 1),
            (half, half),
            (half - 1, half - 1),
        ],
        4 => [
            (half, half),
            (half - 1, half - 1),
            (half - 1, half),
            (half, half - 1),
        ],
        5 => [
            (half - 1, half - 1),
            (half, half - 1),
            (half, half),
            (half - 1, half),
        ],
        6 => [
            (half, half),
            (half - 1, half - 1),
            (half - 1, half),
            (half, half - 1),
        ],
        7 => [
            (half - 1, half - 1),
            (half - 1, half),
            (half, half - 1),
            (half, half),
        ],
        _ => [
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        ],
    }
}

fn python_min_shed(
    point: (i64, i64),
    size: usize,
    sheds: &BTreeSet<(i64, i64)>,
) -> Option<(i64, i64)> {
    python_shed_iteration(size)
        .into_iter()
        .filter(|candidate| sheds.contains(candidate))
        .min_by_key(|(x, y)| (x - point.0).abs() + (y - point.1).abs())
}

fn move_toward(position: (i64, i64), target: (i64, i64), tiles: &[Vec<Value>]) -> Vec<Value> {
    let (x, y) = position;
    let (tx, ty) = target;
    let mut choices = Vec::with_capacity(2);
    if tx < x {
        choices.push(("WEST", (x - 1, y)));
    }
    if tx > x {
        choices.push(("EAST", (x + 1, y)));
    }
    if ty < y {
        choices.push(("NORTH", (x, y - 1)));
    }
    if ty > y {
        choices.push(("SOUTH", (x, y + 1)));
    }
    let size = tiles.len() as i64;
    for (operation, (next_x, next_y)) in choices {
        if next_x >= 0
            && next_y >= 0
            && next_x < size
            && next_y < size
            && tiles
                .get(next_y as usize)
                .and_then(|row| row.get(next_x as usize))
                != Some(&Value::String("LOCKED".into()))
        {
            return vec![Value::String(operation.into())];
        }
    }
    Action::pass()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_trace_has_all_720_steps() {
        assert_eq!(trace().len(), MAX_STEPS);
        assert_eq!(source_trace_action(0).market.len(), 9);
        assert_eq!(source_trace_action(719).farmer, Action::pass());
    }

    #[test]
    fn python_center_set_order_matches_default_board() {
        assert_eq!(python_shed_iteration(10), [(4, 4), (5, 4), (5, 5), (4, 5)]);
    }

    #[test]
    fn price_examples_match_python_anchor() {
        assert_eq!(market_price("MELON", 10_000), 250);
        assert_eq!(market_price("WOOL", 10_105), 1);
        assert_eq!(market_price("STRAWBERRY", 9_900), 204);
    }
}
