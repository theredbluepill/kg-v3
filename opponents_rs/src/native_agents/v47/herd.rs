//! V47 v44y shop-aware herd wrapper: substitute animal purchases by unlocked shops,
//! carry the substitution through pickup/place/build, and sell credited production.
//! Source: agents/v47/main.py (`_Y_*`, `_y_*`, `_y_agent_shopherd`), after Seyit Kaan Gunes.
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};
use std::collections::HashSet;

fn product(animal: &str) -> Option<&'static str> {
    match animal {
        "COW" => Some("MILK"),
        "SHEEP" => Some("WOOL"),
        "GOOSE" => Some("EGG"),
        _ => None,
    }
}
fn structure(animal: &str) -> Option<&'static str> {
    match animal {
        "COW" | "SHEEP" => Some("PASTURE"),
        "GOOSE" => Some("COOP"),
        _ => None,
    }
}
fn cost(animal: &str) -> Option<i64> {
    match animal {
        "COW" => Some(400),
        "SHEEP" => Some(500),
        "GOOSE" => Some(300),
        _ => None,
    }
}
fn seed(crop: &str) -> Option<i64> {
    match crop {
        "WHEAT" => Some(10),
        "CARROT" => Some(20),
        "TOMATO" => Some(50),
        "STRAWBERRY" => Some(100),
        "MELON" => Some(80),
        _ => None,
    }
}
const DAYS: (i64, i64) = (8, 11);
const MAXQ: i64 = 2;
const MARGIN: f64 = 100.0;

#[derive(Clone, Debug)]
pub struct Herd {
    pub states: Value,
    pub report: Value,
}
impl Default for Herd {
    fn default() -> Self {
        Self {
            states: json!({}),
            report: json!({"swaps":0,"pick":0,"place":0,"harvest":0,"boost":0,"errors":0,"coop":0,"declined":0}),
        }
    }
}

pub fn new_state() -> Value {
    json!({"last":-1,"pending":[],"credit":{},"sites":{},"sale":{},"coop_swap":0})
}

/// `_y_target` under the frozen `_Y_CFG` (yarnsheep and yarngeese only).
fn target(kind: &str, shops: &[Value]) -> Option<&'static str> {
    let yarn = shops.iter().any(|s| s == "YARN_STORE");
    if kind == "COW" && yarn {
        return Some("SHEEP");
    }
    if kind == "GOOSE" && yarn {
        return Some("SHEEP");
    }
    None
}

/// `_y_cash`: projected cash after this market list.
pub fn cash(obs: &Value, action: &Value, market: &[Value]) -> Result<f64, &'static str> {
    let seat = int(&obs["player"]) as usize;
    let farm = obs["farms"].get(seat).ok_or("herd: farm")?;
    let prices = obs["market"].get("prices").ok_or("herd: prices")?;
    let scratch = json!({
        "farmer": if truth(&action["farmer"]) { action["farmer"].clone() } else { json!(["PASS"]) },
        "hands": if truth(&action["hands"]) { action["hands"].clone() } else { json!([]) },
        "market": [],
    });
    let shed = Core::projected_shed(&scratch, &View::new(obs));
    let mut money = num(farm.get("money").ok_or("herd: money")?);
    let mut total = 0.0_f64;
    let mut hires = int(&farm["hires_today"]);
    let quads = array(&farm["unlocked_quadrants"]).len() as i64;
    for o in market {
        if !truth(o) {
            continue;
        }
        let n = array(o).len();
        let item = text(&o[1]);
        match text(&o[0]) {
            "SELL" if n >= 3 => {
                let q = int(&o[2]).max(0).min(int(&shed[item]));
                money += 0.8 * q as f64 * num(&prices[item]);
            }
            "BUY_ANIMAL" if n >= 3 => total += (int(&o[2]) * cost(item).unwrap_or(500)) as f64,
            "BUY_PRODUCT" if n >= 3 => total += int(&o[2]) as f64 * (num(&prices[item]) + 10.0),
            "BUY_SEED" if n >= 3 => total += (int(&o[2]) * seed(item).unwrap_or(100)) as f64,
            "BUY_LAND" => total += [1000.0, 2000.0, 4000.0][(quads - 1).clamp(0, 2) as usize],
            "HIRE" => {
                total += fib(hires) as f64;
                hires += 1;
            }
            _ => {}
        }
    }
    Ok(money - total)
}

fn site_key(x: i64, y: i64) -> String {
    format!("{x},{y}")
}

/// `_y_controller`. Mutates `state` in place exactly as the source does, so a
/// failure leaves the partial state the source would leave.
pub fn controller(
    obs: &Value,
    action: &Value,
    state: &mut Value,
    report: &mut Value,
) -> Result<Value, &'static str> {
    let step = int(obs.get("step").ok_or("herd: step")?);
    let day = step / 24;
    let seat = int(&obs["player"]) as usize;
    let farm = obs["farms"].get(seat).ok_or("herd: farm")?;
    let private = obs.get("private").ok_or("herd: private")?;
    let shed = private.get("shed").ok_or("herd: shed")?;
    let inventories = array(&private["inventories"]);
    let shops = array(&obs["town"]["unlocked_shops"]);
    let tiles = &farm["tiles"];
    let center = array(tiles).len() as i64 / 2;
    // 1. confirm last step's swapped purchases (physical shed gain)
    let mut gained = json!({});
    for p in array(&state["pending"]).to_vec() {
        let to = text(&p["to"]).to_owned();
        if gained.get(&to).is_none() {
            gained[&to] = json!((int(&shed[&to]) - int(&p["before"])).max(0));
        }
        let got = int(&p["qty"]).min(int(&gained[&to]));
        increment(&mut gained, &to, -got);
        if got > 0 {
            let from = text(&p["from"]).to_owned();
            let key = format!("{from}->{to}");
            increment(&mut state["credit"], &key, got);
            if structure(&from) != structure(&to) {
                increment(state, "coop_swap", got);
            }
        }
    }
    state["pending"] = json!([]);
    let mut result = action.clone();
    let mut market = orders(&result);
    // 2. purchase-point substitution by unlocked shops (cash-checked)
    if (DAYS.0..=DAYS.1).contains(&day) {
        for index in 0..market.len() {
            let o = market[index].clone();
            if !(array(&o).len() >= 3
                && o[0] == "BUY_ANIMAL"
                && cost(text(&o[1])).is_some()
                && (1..=MAXQ).contains(&int(&o[2])))
            {
                continue;
            }
            let kind = text(&o[1]).to_owned();
            let Some(to) = target(&kind, shops) else {
                continue;
            };
            if to == kind {
                continue;
            }
            let mut trial = market.clone();
            for t in trial.iter_mut() {
                if t.is_array() && *t == o {
                    t[1] = json!(to);
                }
            }
            if cash(obs, &result, &trial)? < MARGIN {
                increment(report, "declined", 1);
                continue;
            }
            state["pending"]
                .as_array_mut()
                .unwrap()
                .push(json!({"from":kind,"to":to,"qty":int(&o[2]),"before":int(&shed[to])}));
            increment(report, "swaps", int(&o[2]));
            market[index][1] = json!(to);
        }
    }
    // 3. worker command rewrites (PICKUP / PLACE / BUILD_COOP) and harvest credit
    let mut workers = commands(&result);
    let mut positions = vec![farm.get("farmer").ok_or("herd: farmer")?.clone()];
    positions.extend(
        array(farm.get("hands").ok_or("herd: hands")?)
            .iter()
            .cloned(),
    );
    let mut avail = json!({});
    for kind in ANIMALS {
        avail[kind] = json!(int(&shed[kind]));
    }
    let mut seen: HashSet<(i64, i64)> = HashSet::new();
    let mut occupied: HashSet<(i64, i64)> = HashSet::new();
    let empty = json!({});
    for actor in 0..workers.len().min(positions.len()) {
        let work = &mut workers[actor];
        if !truth(work) || !work.is_array() {
            continue;
        }
        let inv = inventories.get(actor).unwrap_or(&empty);
        if !inv.is_object() {
            return Err("herd: inventory");
        }
        if array(&positions[actor]).len() != 2 {
            return Err("herd: position");
        }
        let (x, y) = position(&positions[actor]);
        if x < 0 || y < 0 {
            return Err("herd: negative position");
        }
        let tile = tiles
            .get(y as usize)
            .and_then(|row| row.get(x as usize))
            .ok_or("herd: tile")?;
        let site = (x, y);
        let len = array(work).len();
        let op = text(&work[0]).to_owned();
        let item = if len > 1 {
            text(&work[1]).to_owned()
        } else {
            String::new()
        };
        if op == "PICKUP" && len >= 2 && cost(&item).is_some() {
            let qty = if len > 2 { int(&work[2]).max(1) } else { 1 };
            if int(&avail[&item]) >= qty {
                increment(&mut avail, &item, -qty);
                continue;
            }
            if !((x == center - 1 || x == center) && (y == center - 1 || y == center)) {
                continue;
            }
            if ANIMALS.iter().any(|a| truth(&inv[*a])) {
                continue;
            }
            let credits: Vec<(String, i64)> = state["credit"]
                .as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), int(v))).collect())
                .unwrap_or_default();
            for (key, c) in credits {
                let Some((frm, to)) = key.split_once("->") else {
                    continue;
                };
                if frm == item && c >= qty && int(&avail[to]) >= qty {
                    work[1] = json!(to);
                    state["credit"][&key] = json!(c - qty);
                    increment(&mut avail, to, -qty);
                    increment(report, "pick", qty);
                    break;
                }
            }
        } else if op == "PLACE" && len >= 2 && cost(&item).is_some() {
            if int(&inv[&item]) > 0 {
                continue;
            }
            for to in ["COW", "SHEEP", "GOOSE"] {
                if to != item
                    && int(&inv[to]) > 0
                    && tile.is_object()
                    && tile["kind"] == structure(to).unwrap()
                    && !truth(&tile["animal"])
                    && !occupied.contains(&site)
                {
                    work[1] = json!(to);
                    state["sites"][site_key(x, y)] = json!(to);
                    occupied.insert(site);
                    increment(report, "place", 1);
                    break;
                }
            }
        } else if op == "BUILD_COOP" && int(&state["coop_swap"]) > 0 {
            work[0] = json!("BUILD_PASTURE");
            increment(report, "coop", 1);
        } else if op == "HARVEST"
            && state["sites"].get(site_key(x, y)).is_some()
            && !seen.contains(&site)
        {
            if tile.is_object() && tile["animal"] == state["sites"][site_key(x, y)] {
                let units = int(&tile["yield_units"]).max(0);
                if units != 0 {
                    let prod = product(text(&tile["animal"])).ok_or("herd: product")?;
                    increment(&mut state["sale"], prod, units);
                    increment(report, "harvest", units);
                }
            }
            seen.insert(site);
        }
    }
    set_commands(&mut result, workers);
    // 4. sell the extra production at the tape's own existing sale slots
    let sales: Vec<(String, i64)> = state["sale"]
        .as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), int(v))).collect())
        .unwrap_or_default();
    for (prod, credit) in sales {
        let credit = credit.min(int(&shed[&prod]));
        state["sale"][&prod] = json!(credit);
        if credit <= 0 {
            continue;
        }
        let planned: i64 = market
            .iter()
            .filter(|o| array(o).len() >= 3 && o[0] == "SELL" && o[1] == prod.as_str())
            .map(|o| int(&o[2]).max(0))
            .sum();
        if planned <= 0 {
            continue;
        }
        let extra = credit.min(int(&shed[&prod]) - planned);
        if extra <= 0 {
            continue;
        }
        for o in market.iter_mut() {
            if array(o).len() >= 3 && o[0] == "SELL" && o[1] == prod.as_str() && int(&o[2]) > 0 {
                o[2] = json!(int(&o[2]) + extra);
                state["sale"][&prod] = json!(credit - extra);
                increment(report, "boost", extra);
                break;
            }
        }
    }
    set_orders(&mut result, market);
    Ok(result)
}

impl Herd {
    pub fn layer(&mut self, obs: &Value, action: Value) -> Value {
        let outcome: Result<Value, &'static str> = (|| {
            let seat = int(obs.get("player").ok_or("herd: player")?);
            let step = int(obs.get("step").ok_or("herd: step")?);
            let key = seat.to_string();
            if self.states[&key].is_null() || step <= int(&self.states[&key]["last"]) {
                self.states[&key] = new_state();
            }
            self.states[&key]["last"] = json!(step);
            controller(obs, &action, &mut self.states[&key], &mut self.report)
        })();
        match outcome {
            Ok(result) => result,
            Err(_) => {
                increment(&mut self.report, "errors", 1);
                action
            }
        }
    }
}
