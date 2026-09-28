//! Ordered JSON helpers for the native V39 translation.
//!
//! Python mapping order is retained by serde_json's preserve_order feature,
//! including worker inventories whose deposit order can affect capacity.

use serde_json::{Value, json};
use std::sync::OnceLock;

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
pub const ANIMALS: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub const MOVES: [&str; 4] = ["NORTH", "SOUTH", "EAST", "WEST"];

pub fn int(value: &Value) -> i64 {
    value
        .as_i64()
        .or_else(|| value.as_f64().map(|x| x as i64))
        .or_else(|| value.as_str().and_then(|x| x.parse().ok()))
        .unwrap_or_else(|| i64::from(value.as_bool().unwrap_or(false)))
}

pub fn num(value: &Value) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|x| x.parse().ok()))
        .unwrap_or_else(|| f64::from(value.as_bool().unwrap_or(false)))
}

pub fn truth(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(x) => *x,
        Value::Number(_) => num(value) != 0.0,
        Value::String(x) => !x.is_empty(),
        Value::Array(x) => !x.is_empty(),
        Value::Object(x) => !x.is_empty(),
    }
}

pub fn array(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or(&[])
}

pub fn text(value: &Value) -> &str {
    value.as_str().unwrap_or("")
}

pub fn position(value: &Value) -> (i64, i64) {
    (int(&value[0]), int(&value[1]))
}

pub fn step_of(observation: &Value) -> i64 {
    if observation["step"].is_null() {
        int(&observation["day"]) * 24 + int(&observation["hour"])
    } else {
        int(&observation["step"])
    }
}

pub fn shed_adjacent(pos: &Value, board: i64) -> bool {
    if array(pos).len() < 2 {
        return false;
    }
    let (x, y) = position(pos);
    let half = board / 2;
    (x == half - 1 || x == half) && (y == half - 1 || y == half)
}

pub fn tile_at<'a>(tiles: &'a Value, pos: &Value) -> &'a Value {
    static LOCKED: OnceLock<Value> = OnceLock::new();
    let (x, y) = position(pos);
    if x < 0 || y < 0 {
        return LOCKED.get_or_init(|| json!("LOCKED"));
    }
    tiles
        .get(y as usize)
        .and_then(|row| row.get(x as usize))
        .unwrap_or_else(|| LOCKED.get_or_init(|| json!("LOCKED")))
}

pub fn pass_action() -> Value {
    json!({"farmer": ["PASS"], "hands": [], "market": []})
}

pub fn commands(action: &Value) -> Vec<Value> {
    let mut result = vec![if truth(&action["farmer"]) {
        action["farmer"].clone()
    } else {
        json!(["PASS"])
    }];
    result.extend(array(&action["hands"]).iter().cloned());
    result
}

pub fn set_commands(action: &mut Value, commands: Vec<Value>) {
    action["farmer"] = commands.first().cloned().unwrap_or_else(|| json!(["PASS"]));
    action["hands"] = json!(commands.get(1..).unwrap_or(&[]));
}

pub fn orders(action: &Value) -> Vec<Value> {
    array(&action["market"]).to_vec()
}

pub fn set_orders(action: &mut Value, orders: Vec<Value>) {
    action["market"] = json!(orders);
}

pub fn stock_total(stock: &Value) -> i64 {
    stock
        .as_object()
        .map(|m| m.values().map(int).sum())
        .unwrap_or(0)
}

pub fn count(stock: &Value, item: &str) -> i64 {
    int(&stock[item])
}

pub fn increment(stock: &mut Value, item: &str, delta: i64) {
    if !stock.is_object() {
        *stock = json!({});
    }
    stock[item] = json!(int(&stock[item]) + delta);
}

pub fn quantity(command: &Value, default: i64) -> i64 {
    command.get(2).map(int).unwrap_or(default)
}

pub fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n {
        (a, b) = (b, a.saturating_add(b));
    }
    a
}

pub fn routes() -> &'static Value {
    static ROUTES: OnceLock<Value> = OnceLock::new();
    ROUTES.get_or_init(|| {
        serde_json::from_str(include_str!("../../../fixtures/v39-routes.json"))
            .expect("frozen V39 route JSON")
    })
}

pub fn route_action(route: i64, step: i64) -> Value {
    if step >= 0
        && let Some(value) = routes()[route.to_string()]
            .get(step as usize)
            .filter(|x| x.is_object())
    {
        return value.clone();
    }
    pass_action()
}

#[derive(Clone, Debug)]
pub struct View {
    pub farm: Value,
    pub rival: Value,
    pub shed: Value,
    pub seeds: Value,
    pub invs: Vec<Value>,
    pub prices: Value,
    pub money: f64,
    pub tiles: Value,
    pub board: i64,
    pub positions: Vec<Value>,
    pub hires_today: i64,
    pub quadrants: usize,
}

impl View {
    pub fn new(observation: &Value) -> Self {
        let seat = int(&observation["player"]) as usize;
        let farm = observation["farms"][seat].clone();
        let rival = observation["farms"][1usize.saturating_sub(seat)].clone();
        let private = &observation["private"];
        let mut shed = json!({});
        if let Some(entries) = private["shed"].as_object() {
            for (item, qty) in entries {
                shed[item] = json!(int(qty).max(0));
            }
        }
        let mut prices = json!({});
        if let Some(entries) = observation["market"]["prices"].as_object() {
            for (item, price) in entries {
                prices[item] = json!(int(price));
            }
        }
        let mut positions = vec![farm["farmer"].clone()];
        positions.extend(array(&farm["hands"]).iter().cloned());
        let tiles = farm["tiles"].clone();
        let board = if array(&tiles).is_empty() {
            10
        } else {
            array(&tiles).len() as i64
        };
        Self {
            money: num(&farm["money"]),
            hires_today: int(&farm["hires_today"]),
            quadrants: array(&farm["unlocked_quadrants"]).len(),
            farm,
            rival,
            shed,
            seeds: private["seeds"].clone(),
            invs: array(&private["inventories"])
                .iter()
                .map(|v| if truth(v) { v.clone() } else { json!({}) })
                .collect(),
            prices,
            tiles,
            board,
            positions,
        }
    }

    pub fn inv(&self, index: usize) -> &Value {
        static EMPTY: OnceLock<Value> = OnceLock::new();
        self.invs
            .get(index)
            .unwrap_or_else(|| EMPTY.get_or_init(|| json!({})))
    }

    pub fn in_hands(&self, item: &str) -> i64 {
        self.invs.iter().map(|inv| int(&inv[item]).max(0)).sum()
    }
}
