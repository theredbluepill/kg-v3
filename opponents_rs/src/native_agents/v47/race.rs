//! V47 EXP283/288/293 race layer: clone-gated sale pre-emption horizon.
//!
//! Source: agents/v47/main.py (`_RACE_*`, `_race_*`, race `agent`). The v44y
//! override sets the clone horizon to 9. The horizon reaches the frozen V43
//! `_r36_reserve` through `Production::race_horizons`.
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use crate::native_agents::v43::production::similarity;
use serde_json::{Value, json};

pub const HORIZON_CLONE: i64 = 9;
pub const HORIZON_ESCALATED: i64 = 24;
pub const HORIZON_MIRROR: i64 = 24;
pub const ITEMS: [&str; 7] = [
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
];
const REPORT_KEYS: [&str; 5] = [
    "race_clone_turns",
    "race_horizon_turns",
    "race_lost_races",
    "race_escalations",
    "race_errors",
];

fn shop_items(shop: &str) -> &'static [&'static str] {
    match shop {
        "BAKERY" => &["EGG", "WHEAT"],
        "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE" => &["WOOL"],
        "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE" => &["CARROT"],
        "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
        "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        _ => &[],
    }
}

/// `configuration is None or all(configuration.get(k, v) == v ...)`.
pub fn standard(config: &Value, starting_money: bool) -> bool {
    [
        ("boardSize", 10),
        ("turnsPerDay", 24),
        ("shedCapacity", 100),
        ("maxMarketOrdersPerTurn", 10),
        ("startingMoney", 3000),
    ]
    .iter()
    .take(if starting_money { 5 } else { 4 })
    .all(|(k, v)| {
        config
            .get(*k)
            .map(|x| x.is_number() && num(x) == *v as f64)
            .unwrap_or(true)
    })
}

#[derive(Clone, Debug)]
pub struct Race {
    pub states: Value,
    pub report: Value,
}
impl Default for Race {
    fn default() -> Self {
        let mut report = json!({});
        for k in REPORT_KEYS {
            report[k] = json!(0);
        }
        Self {
            states: json!({}),
            report,
        }
    }
}

fn new_state() -> Value {
    json!({"step":-1,"hist":[],"horizon":0,"level":HORIZON_CLONE,"prev":null,"prev_action":null})
}

fn positions_equal(farms: &Value, player: usize) -> bool {
    let own = &farms[player];
    let rival = &farms[1 - player];
    !array(&own["hands"]).is_empty()
        && own["hands"] == rival["hands"]
        && own["farmer"] == rival["farmer"]
}

fn clone_detected(obs: &Value, state: &mut Value) -> bool {
    let farms = &obs["farms"];
    let player = int(&obs["player"]) as usize;
    if !array(&farms[player]["hands"]).is_empty() {
        let equal = positions_equal(farms, player);
        if !state["hist"].is_array() {
            state["hist"] = json!([]);
        }
        let hist = state["hist"].as_array_mut().unwrap();
        hist.push(json!(equal));
        if hist.len() > 6 {
            hist.remove(0);
        }
    }
    let hist = array(&state["hist"]);
    hist.len() >= 4 && hist.iter().filter(|v| truth(v)).count() >= 4 && similarity(obs) >= 0.95
}

fn town(step: i64, shops: &[Value]) -> Value {
    let mut out = json!({});
    if step.rem_euclid(4) == 0 {
        for shop in shops {
            let items = shop_items(text(shop));
            for item in items {
                increment(&mut out, item, if items.len() == 1 { 2 } else { 1 });
            }
        }
    }
    if step.rem_euclid(24) == 0 {
        for item in ITEMS {
            increment(&mut out, item, 1);
        }
    }
    out
}

/// EXP293 `_race_lost`: the rival sold a held race product last turn while the
/// common tape sells it only 5..24 turns ahead.
pub fn lost(obs: &Value, state: &Value, core: &Core) -> Result<bool, &'static str> {
    let prev = &state["prev"];
    let prev_action = &state["prev_action"];
    if prev.is_null() || prev_action.is_null() {
        return Ok(false);
    }
    let step = int(obs.get("step").ok_or("race: step")?);
    let player = int(&obs["player"]);
    if step != int(&prev["step"]) + 1 || step.rem_euclid(24) == 0 {
        return Ok(false);
    }
    let inv = &obs["market"]["inventory"];
    let pinv = &prev["inventory"];
    let prices = &prev["prices"];
    let consumed = town(step - 1, array(&prev["shops"]));
    let sold: Vec<String> = orders(prev_action)
        .iter()
        .filter(|o| array(o).len() >= 3 && o[0] == "SELL" && ITEMS.contains(&text(&o[1])))
        .map(|o| text(&o[1]).to_owned())
        .collect();
    let native = &core.players[player.to_string()];
    if native.is_null() {
        return Err("race: missing native state");
    }
    let route = int(&native["route"]);
    if native["route"].is_null() || routes()[route.to_string()].is_null() {
        return Err("race: unknown route");
    }
    for item in ITEMS {
        let before = int(&prev["shed"][item]);
        if before <= 0 || sold.iter().any(|s| s == item) || int(&prices[item]) <= 1 {
            continue;
        }
        if inv.get(item).is_none() || pinv.get(item).is_none() {
            return Err("race: missing market inventory");
        }
        let rival = int(&inv[item]) - int(&pinv[item]) + int(&consumed[item]);
        if rival <= 0 {
            continue;
        }
        let planned = |t: i64| {
            orders(&route_action(if t >= 648 { 2 } else { route }, t))
                .iter()
                .any(|o| array(o).len() >= 3 && o[0] == "SELL" && o[1] == item)
        };
        if (step - 1..719.min(step + 5)).any(planned) {
            continue;
        }
        if (step + 5..719.min(step + 24)).any(planned) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn snapshot(obs: &Value) -> Result<Value, &'static str> {
    let market = &obs["market"];
    if !market["inventory"].is_object() || !market["prices"].is_object() || !obs["town"].is_object()
    {
        return Err("race: snapshot fields");
    }
    let step = int(obs.get("step").ok_or("race: step")?);
    let shops = obs["town"]
        .get("unlocked_shops")
        .cloned()
        .unwrap_or_else(|| json!([]));
    Ok(json!({
        "step": step,
        "inventory": market["inventory"],
        "prices": market["prices"],
        "shops": shops,
        "shed": View::new(obs).shed,
    }))
}

impl Race {
    /// Pre-parent phase of the race `agent`. Returns the seat key and whether
    /// the seat state exists for the post-parent phase; publishes the horizon.
    pub fn before(
        &mut self,
        obs: &Value,
        config: &Value,
        core: &Core,
        race_horizons: &mut Value,
    ) -> (String, bool) {
        let key = int(&obs["player"]).to_string();
        let mut has_state = false;
        let outcome: Result<(), &'static str> = (|| {
            let step = int(obs.get("step").ok_or("race: step")?);
            if self.states[&key].is_null() || step <= int(&self.states[&key]["step"]) {
                self.states[&key] = new_state();
            }
            has_state = true;
            if step == 0 {
                for k in REPORT_KEYS {
                    self.report[k] = json!(0);
                }
            }
            self.states[&key]["step"] = json!(step);
            self.states[&key]["horizon"] = json!(0);
            if step == 1 {
                let player = int(&obs["player"]) as usize;
                let farms = &obs["farms"];
                match (
                    farms.get(1 - player).and_then(|f| f.get("money")),
                    farms.get(player).and_then(|f| f.get("money")),
                ) {
                    (Some(rival), Some(own)) => {
                        let mirror = (num(rival) - num(own)).abs() < 0.5;
                        if mirror && HORIZON_MIRROR > int(&self.states[&key]["level"]) {
                            self.states[&key]["level"] = json!(HORIZON_MIRROR);
                        }
                        self.report["race_mirror"] = json!(i64::from(mirror));
                    }
                    _ => increment(&mut self.report, "race_errors", 1),
                }
            }
            if standard(config, false)
                && (216..696).contains(&step)
                && clone_detected(obs, &mut self.states[&key])
            {
                increment(&mut self.report, "race_clone_turns", 1);
                if int(&self.states[&key]["level"]) < HORIZON_ESCALATED
                    && lost(obs, &self.states[&key], core)?
                {
                    increment(&mut self.report, "race_lost_races", 1);
                    self.states[&key]["level"] = json!(HORIZON_ESCALATED);
                    increment(&mut self.report, "race_escalations", 1);
                }
                let level = self.states[&key]["level"].clone();
                self.states[&key]["horizon"] = level;
                increment(&mut self.report, "race_horizon_turns", 1);
            }
            Ok(())
        })();
        if outcome.is_err() {
            increment(&mut self.report, "race_errors", 1);
        }
        race_horizons[&key] = json!(int(&self.states[&key]["horizon"]));
        (key, has_state)
    }

    pub fn snapshot_phase(&mut self, obs: &Value, has_state: bool) -> Option<Value> {
        if !has_state {
            return None;
        }
        let Some(step) = obs.get("step").map(int) else {
            increment(&mut self.report, "race_errors", 1);
            return None;
        };
        if !(215..696).contains(&step) {
            return None;
        }
        match snapshot(obs) {
            Ok(s) => Some(s),
            Err(_) => {
                increment(&mut self.report, "race_errors", 1);
                None
            }
        }
    }

    pub fn after(&mut self, key: &str, has_state: bool, snapshot: Option<Value>, action: &Value) {
        if !has_state {
            return;
        }
        match snapshot {
            Some(s) => {
                self.states[key]["prev"] = s;
                self.states[key]["prev_action"] = action.clone();
            }
            None => {
                self.states[key]["prev"] = Value::Null;
                self.states[key]["prev_action"] = Value::Null;
            }
        }
    }

    /// v44y `_v44y_clone_gate`.
    pub fn clone_gate(&self, obs: &Value) -> bool {
        let key = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        let st = &self.states[&key];
        if int(&st["horizon"]) > 0 {
            return true;
        }
        let hist = array(&st["hist"]);
        if step >= 696 && hist.len() >= 4 && hist.iter().filter(|v| truth(v)).count() >= 4 {
            return similarity(obs) >= 0.95;
        }
        false
    }
}
