//! Helpers shared by the 2945 Farm layers.
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};

pub type Pos = (i64, i64);
pub type Fallible<T> = Result<T, &'static str>;

pub const MAX_ORDERS: usize = 10;

pub fn inc(v: &mut Value, k: &str, n: i64) {
    v[k] = json!(int(&v[k]) + n);
}

pub fn seat(obs: &Value) -> usize {
    int(&obs["player"]) as usize
}

pub fn own_farm(obs: &Value) -> &Value {
    &obs["farms"][seat(obs)]
}

pub fn tile(farm: &Value, p: Pos) -> &Value {
    if p.0 < 0 || p.1 < 0 {
        return &Value::Null;
    }
    &farm["tiles"][p.1 as usize][p.0 as usize]
}

/// Farmer then hands, as the source's `[farm["farmer"]] + list(farm["hands"])`.
pub fn unit_positions(farm: &Value) -> Vec<Pos> {
    let mut out = vec![position(&farm["farmer"])];
    out.extend(array(&farm["hands"]).iter().map(position));
    out
}

pub fn key(p: Pos) -> String {
    format!("{},{}", p.0, p.1)
}

pub fn unkey(k: &str) -> Pos {
    let mut parts = k.split(',').map(|c| c.parse().unwrap_or(0));
    (parts.next().unwrap_or(0), parts.next().unwrap_or(0))
}

pub fn move_delta(op: &str) -> Option<Pos> {
    match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    }
}

/// `_IMPL.chassis.players.get(player)` with a route the chassis knows.
pub fn native_route(core: &Core, obs: &Value) -> Option<i64> {
    let native = core.players.get(seat(obs).to_string())?;
    if native["route"].is_null() || routes()[int(&native["route"]).to_string()].is_null() {
        return None;
    }
    Some(int(&native["route"]))
}

pub fn tape_len(route: i64) -> i64 {
    array(&routes()[route.to_string()]).len() as i64
}

/// `tape[t]` when it is an action, otherwise an empty action.
pub fn tape(route: i64, t: i64) -> Value {
    if t >= 0
        && let Some(a) = routes()[route.to_string()]
            .get(t as usize)
            .filter(|a| a.is_object())
    {
        return a.clone();
    }
    json!({})
}

/// `_ca_tape` / `_ch_tape` / `_sr_tape` / `_cs_tape`: the terminal route from step 648.
pub fn late_tape(core: &Core, obs: &Value, t: i64) -> Value {
    let native = &core.players[seat(obs).to_string()];
    if !truth(native) || t > 719 {
        return json!({});
    }
    let route = if t >= 648 { 2 } else { int(&native["route"]) };
    if (core.hybrid_opening || core.cha22) && route == 0 && (0..96).contains(&t) {
        core.configured_route_action(route, t)
    } else {
        tape(route, t)
    }
}

/// `[act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])`.
pub fn units(action: &Value) -> Vec<Value> {
    commands(action)
}

pub fn is_pair(v: &Value, a: &str, b: &str) -> bool {
    array(v).len() >= 2 && v[0] == a && v[1] == b
}

/// A SELL/BUY order of `item` that carries a quantity.
pub fn is_order(v: &Value, op: &str, item: &str) -> bool {
    array(v).len() >= 3 && v[0] == op && v[1] == item
}

pub fn hires(action: &Value) -> usize {
    orders(action).iter().filter(|o| o[0] == "HIRE").count()
}

/// `_ca_spawn` / `_cs_spawn`: the least occupied shed-access tile, first on ties.
pub fn spawn(positions: &[Pos], board: i64) -> Pos {
    let half = board / 2;
    let access = [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ];
    *access
        .iter()
        .min_by_key(|a| positions.iter().filter(|p| p == a).count())
        .unwrap()
}

/// Python `round(x, 3)`: correctly rounded decimal conversion of the exact double,
/// ties to even. Scaling by 1000 first rounds 32.9995 to 33.0 where Python gives 32.999.
pub fn py_round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
}

/// Python float `//`.
pub fn floor_div(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div == 0.0 {
        return 0.0;
    }
    let floor = div.floor();
    if div - floor > 0.5 {
        floor + 1.0
    } else {
        floor
    }
}

/// `Chassis.future_sells`: planned SELL quantity of `item` on `route` from `step`.
pub fn future_sells(route: i64, item: &str, step: i64) -> i64 {
    let n = tape_len(route);
    if !(0..=n).contains(&step) || !PRODUCTS.contains(&item) {
        return 0;
    }
    (step..n)
        .flat_map(|t| orders(&tape(route, t)))
        .filter(|o| is_order(o, "SELL", item))
        .map(|o| int(&o[2]).max(0))
        .sum()
}

/// Rebuild `action` with new unit commands (`result["farmer"], result["hands"] = …`).
pub fn with_units(action: &Value, units: Vec<Value>) -> Value {
    let mut result = action.clone();
    set_commands(&mut result, units);
    result
}

/// Add `qty` to the first SELL of `item`; false when there is none.
pub fn bump_sell(market: &mut [Value], item: &str, qty: i64) -> bool {
    for o in market.iter_mut() {
        if is_order(o, "SELL", item) {
            o[2] = json!(int(&o[2]) + qty);
            return true;
        }
    }
    false
}

pub fn selling(market: &[Value], item: &str) -> i64 {
    market
        .iter()
        .filter(|o| is_order(o, "SELL", item))
        .map(|o| int(&o[2]))
        .sum()
}
