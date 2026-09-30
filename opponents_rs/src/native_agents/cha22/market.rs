//! cha22 FX/EV, DAWNPX, MIDPX, BUYDIP, MODELPX and SHIELD-MILK.
//! Source: agents/cha22/main.py, lines 6555–7170, frozen sha256 127ed3e6….
//! Tuple flow keys `(step, item)` are exported as `"step,item"`.

use crate::native_agents::farm2945::race::town_draw;
use crate::native_agents::farm2945::util::{MAX_ORDERS, is_order, native_route, tape, tape_len};
use crate::native_agents::v43::{common::*, core::Core, production::market_price};
use serde_json::{Value, json};

const FX_ITEMS: [&str; 7] = [
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
];
const MPX_ITEMS: [&str; 3] = ["MILK", "STRAWBERRY", "WOOL"];
const SM_PRODUCTS: [&str; 9] = [
    "MELON",
    "STRAWBERRY",
    "MILK",
    "WOOL",
    "EGG",
    "TOMATO",
    "CARROT",
    "WHEAT",
    "FERTILIZER",
];

#[derive(Clone, Debug)]
pub struct Market {
    pub fx_states: Value,
    pub ev_states: Value,
    pub bd_states: Value,
    pub mpx_hist: Value,
    pub fx_report: Value,
    pub dp_report: Value,
    pub mp_report: Value,
    pub bd_report: Value,
    pub mpx_report: Value,
    pub sm_report: Value,
}

impl Default for Market {
    fn default() -> Self {
        Self {
            fx_states: json!({}),
            ev_states: json!({}),
            bd_states: json!({}),
            mpx_hist: json!({}),
            fx_report: json!({"fx_fires":0,"fx_units":0,"fx_rival_seen":0,"fx_errors":0}),
            dp_report: json!({"dp_fires":0,"dp_units":0,"dp_errors":0}),
            mp_report: json!({"mp_fires":0,"mp_units":0,"mp_errors":0}),
            bd_report: json!({"bd_split":0,"bd_pending_units":0,"bd_released":0,"bd_errors":0}),
            mpx_report: json!({"mpx_fires":0,"mpx_units":0,"mpx_errors":0}),
            sm_report: json!({"sm_shields":0,"sm_units":0,"sm_errors":0}),
        }
    }
}

fn player_key(obs: &Value) -> String {
    int(&obs["player"]).to_string()
}

fn fresh_fx() -> Value {
    json!({"step":-1,"flow":{},"quotes":{},"prev":null})
}

fn flow_key(key: &str) -> (i64, &str) {
    let (step, item) = key.split_once(',').unwrap_or(("0", ""));
    (step.parse().unwrap_or(0), item)
}

fn planned_sales(core: &Core, route: i64, item: &str, step: i64, horizon: i64) -> i64 {
    ((step + 1)..tape_len(route).min(step + horizon + 1))
        .flat_map(|t| {
            // Keep route-zero overlay support explicit; the current windows all
            // begin after the inherited opening overlay has ended.
            if core.hybrid_opening && route == 0 && (0..96).contains(&t) {
                orders(&core.configured_route_action(route, t))
            } else {
                orders(&tape(route, t))
            }
        })
        .filter(|o| is_order(o, "SELL", item))
        .map(|o| int(&o[2]).max(0))
        .sum()
}

fn quote_holds(hist: &Value, item: &str, step: i64, price: i64) -> bool {
    let vals: Vec<i64> = hist
        .as_object()
        .into_iter()
        .flat_map(|m| m.iter())
        .filter_map(|(t, quotes)| {
            let t: i64 = t.parse().ok()?;
            let q = int(&quotes[item]);
            (step - 12 <= t && t < step && q > 0).then_some(q)
        })
        .collect();
    vals.is_empty() || price as f64 >= vals.iter().sum::<i64>() as f64 / vals.len() as f64
}

fn has_item(market: &[Value], item: &str, include_buy: bool) -> bool {
    market.iter().any(|o| {
        array(o).len() > 1
            && o[1] == item
            && (o[0] == "SELL" || (include_buy && o[0] == "BUY_PRODUCT"))
    })
}

impl Market {
    /// Outer wrappers reset before their parent; FX additionally updates its
    /// previous-turn flow before the inner policy and mirror wrapper execute.
    pub fn before(&mut self, obs: &Value) {
        if int(&obs["step"]) == 0 {
            self.dp_report = json!({"dp_fires":0,"dp_units":0,"dp_errors":0});
            self.mp_report = json!({"mp_fires":0,"mp_units":0,"mp_errors":0});
            self.bd_report =
                json!({"bd_split":0,"bd_pending_units":0,"bd_released":0,"bd_errors":0});
            self.bd_states = json!({});
            self.mpx_report = json!({"mpx_fires":0,"mpx_units":0,"mpx_errors":0});
            self.mpx_hist = json!({});
            self.sm_report = json!({"sm_shields":0,"sm_units":0,"sm_errors":0});
        }
        self.fx_before(obs);
    }

    pub fn fx_before(&mut self, obs: &Value) {
        let key = player_key(obs);
        let step = int(&obs["step"]);
        if self.fx_states[&key].is_null() || step <= int(&self.fx_states[&key]["step"]) {
            self.fx_states[&key] = fresh_fx();
            if step == 0 {
                // Python's dict.update deliberately leaves any existing EV
                // counters present, rather than resetting the whole report.
                for k in ["fx_fires", "fx_units", "fx_rival_seen", "fx_errors"] {
                    self.fx_report[k] = json!(0);
                }
            }
        }
        self.fx_states[&key]["step"] = json!(step);
        self.fx_update(obs);
    }

    pub fn fx_update(&mut self, obs: &Value) {
        let key = player_key(obs);
        let prev = self.fx_states[&key]["prev"].clone();
        let step = int(&obs["step"]);
        if !truth(&prev) || int(&prev["step"]) != step - 1 {
            return;
        }
        let Some(inventory) = obs["market"]["inventory"].as_object() else {
            increment(&mut self.fx_report, "fx_errors", 1);
            return;
        };
        for item in FX_ITEMS {
            if int(&prev["prices"][item]) <= 3 {
                continue;
            }
            let (Some(now), Some(prior)) = (inventory.get(item), prev["inventory"].get(item))
            else {
                increment(&mut self.fx_report, "fx_errors", 1);
                return;
            };
            let t = int(&prev["step"]);
            let sold = int(now) - int(prior) + town_draw(&prev["shops"], t, item)
                - int(&prev["own"][item]);
            if sold > 0 {
                self.fx_states[&key]["flow"][format!("{t},{item}")] = json!(sold);
                increment(&mut self.fx_report, "fx_rival_seen", 1);
            }
        }
        if let Some(flow) = self.fx_states[&key]["flow"].as_object_mut()
            && flow.len() > 512
        {
            let mut keys: Vec<String> = flow.keys().cloned().collect();
            keys.sort_by(|a, b| flow_key(a).cmp(&flow_key(b)));
            for k in keys.into_iter().take(256) {
                flow.remove(&k);
            }
        }
    }

    /// `_fx_apply` including quote-history recording, without wrapper reset or
    /// previous-turn recovery. Call `fx_before` first for the complete wrapper.
    pub fn flow_apply(
        &mut self,
        obs: &Value,
        action: Value,
        core: &Core,
        mirror_like: bool,
    ) -> Value {
        let step = int(&obs["step"]);
        let key = player_key(obs);
        if self.fx_states[&key].is_null() {
            self.fx_states[&key] = fresh_fx();
        }
        let mut quotes = json!({});
        for item in FX_ITEMS {
            quotes[item] = json!(int(&obs["market"]["prices"][item]));
        }
        self.fx_states[&key]["quotes"][step.to_string()] = quotes;
        if let Some(hist) = self.fx_states[&key]["quotes"].as_object_mut()
            && hist.len() > 96
        {
            let mut keys: Vec<String> = hist.keys().cloned().collect();
            keys.sort_by_key(|s| s.parse::<i64>().unwrap_or(0));
            for k in keys.into_iter().take(48) {
                hist.remove(&k);
            }
        }
        if !(96..700).contains(&step) || mirror_like {
            return action;
        }
        let mut market = orders(&action);
        if market.len() >= MAX_ORDERS {
            return action;
        }
        let stock = Core::projected_shed(&action, &View::new(obs));
        let Some(route) = native_route(core, obs) else {
            return action;
        };
        let mut added = false;
        for item in FX_ITEMS {
            if has_item(&market, item, true) || market.len() >= MAX_ORDERS {
                continue;
            }
            let flow: i64 = self.fx_states[&key]["flow"]
                .as_object()
                .into_iter()
                .flat_map(|m| m.iter())
                .filter(|(k, _)| {
                    let (t, i) = flow_key(k);
                    i == item && step - 8 <= t && t < step
                })
                .map(|(_, q)| int(q))
                .sum();
            if flow < 999 {
                continue;
            }
            let price = int(&obs["market"]["prices"][item]);
            if price <= 3 || !quote_holds(&self.fx_states[&key]["quotes"], item, step, price) {
                continue;
            }
            let qty = int(&stock[item]).min(planned_sales(core, route, item, step, 12));
            if qty <= 0 {
                continue;
            }
            market.insert(0, json!(["SELL", item, qty]));
            increment(&mut self.fx_report, "fx_fires", 1);
            increment(&mut self.fx_report, "fx_units", qty);
            added = true;
        }
        if !added {
            return action;
        }
        let mut result = action;
        market.truncate(MAX_ORDERS);
        set_orders(&mut result, market);
        result
    }

    fn window_sale(&mut self, obs: &Value, action: Value, core: &Core, prefix: &str) -> Value {
        let step = int(&obs["step"]);
        let hours: &[i64] = match prefix {
            "ev" => &[15, 16, 17, 18, 19, 20],
            "dp" => &[0, 1, 2],
            "mp" => &[10, 11, 12, 13],
            _ => return action,
        };
        if !(96..700).contains(&step) || !hours.contains(&(step % 24)) {
            return action;
        }
        let mut market = orders(&action);
        if market.len() >= MAX_ORDERS {
            return action;
        }
        let stock = Core::projected_shed(&action, &View::new(obs));
        let Some(route) = native_route(core, obs) else {
            return action;
        };
        let key = player_key(obs);
        let mut added = false;
        for item in FX_ITEMS {
            if has_item(&market, item, true) || market.len() >= MAX_ORDERS {
                continue;
            }
            let price = int(&obs["market"]["prices"][item]);
            if price <= 3 || !quote_holds(&self.fx_states[&key]["quotes"], item, step, price) {
                continue;
            }
            let planned = planned_sales(core, route, item, step, 8);
            if planned <= 0 {
                continue;
            }
            let qty = int(&stock[item]).min(1.max((3 * planned + 3) / 4));
            if qty <= 0 {
                continue;
            }
            market.insert(0, json!(["SELL", item, qty]));
            let report = match prefix {
                "ev" => &mut self.fx_report,
                "dp" => &mut self.dp_report,
                _ => &mut self.mp_report,
            };
            increment(report, &format!("{prefix}_fires"), 1);
            increment(report, &format!("{prefix}_units"), qty);
            added = true;
        }
        if !added {
            return action;
        }
        let mut result = action;
        market.truncate(MAX_ORDERS);
        set_orders(&mut result, market);
        result
    }

    pub fn evening(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        self.window_sale(obs, action, core, "ev")
    }

    pub fn dawn(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        self.window_sale(obs, action, core, "dp")
    }

    pub fn midday(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        self.window_sale(obs, action, core, "mp")
    }

    pub fn fx_ev(&mut self, obs: &Value, action: Value, core: &Core, mirror_like: bool) -> Value {
        let action = self.flow_apply(obs, action, core, mirror_like);
        let out = self.evening(obs, action, core);
        let mut own = json!({});
        for o in orders(&out) {
            if array(&o).len() >= 3 && o[0] == "SELL" {
                increment(&mut own, text(&o[1]), int(&o[2]).max(0));
            }
        }
        self.fx_states[player_key(obs)]["prev"] = json!({
            "step":int(&obs["step"]),
            "inventory":obs["market"]["inventory"],
            "prices":obs["market"]["prices"],
            "own":own,
            "shops":obs["town"]["unlocked_shops"],
        });
        out
    }

    pub fn bankdrip(&mut self, obs: &Value, action: Value) -> Value {
        let step = int(&obs["step"]);
        if !(24..690).contains(&step) || !action.is_object() {
            return action;
        }
        let key = player_key(obs);
        if self.bd_states[&key].is_null() {
            self.bd_states[&key] = json!({"hist":[],"pending":0,"since":null});
        }
        let state = &mut self.bd_states[&key];
        let price = int(&obs["market"]["prices"]["WHEAT"]);
        if price > 0 {
            let hist = state["hist"].as_array_mut().expect("BUYDIP hist");
            hist.push(json!([step, price]));
            if hist.len() > 48 {
                hist.drain(..24);
            }
        }
        let mut market = orders(&action);
        let vals: Vec<i64> = array(&state["hist"])
            .iter()
            .filter(|v| step - 12 <= int(&v[0]) && int(&v[0]) < step)
            .map(|v| int(&v[1]))
            .collect();
        let avg = if vals.is_empty() {
            price as f64
        } else {
            vals.iter().sum::<i64>() as f64 / vals.len() as f64
        };
        if int(&state["pending"]) > 0 {
            let due = !state["since"].is_null() && step - int(&state["since"]) >= 6;
            if price > 0 && (price as f64 <= avg || due) {
                let has = market.iter().any(|o| is_order(o, "BUY_PRODUCT", "WHEAT"));
                if market.len() < 10 && !has {
                    let qty = int(&state["pending"]).min(8);
                    market.push(json!(["BUY_PRODUCT", "WHEAT", qty]));
                    increment(state, "pending", -qty);
                    increment(&mut self.bd_report, "bd_released", qty);
                }
                if int(&state["pending"]) <= 0 {
                    state["since"] = Value::Null;
                }
            }
        }
        if int(&state["pending"]) < 64 {
            for o in &mut market {
                if is_order(o, "BUY_PRODUCT", "WHEAT") && int(&o[2]) >= 8 {
                    let q = int(&o[2]);
                    if price > 0 && price as f64 > avg {
                        let hold = (q - 1).min(64 - int(&state["pending"])).min(q / 2);
                        if hold > 0 {
                            o[2] = json!(q - hold);
                            increment(state, "pending", hold);
                            if state["since"].is_null() {
                                state["since"] = json!(step);
                            }
                            increment(&mut self.bd_report, "bd_split", 1);
                            increment(&mut self.bd_report, "bd_pending_units", hold);
                        }
                    }
                }
            }
        }
        let mut result = action;
        set_orders(&mut result, market);
        result
    }

    pub fn premium(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        if !(144..696).contains(&step) || !(12..23).contains(&(step % 24)) {
            return action;
        }
        let key = player_key(obs);
        if self.mpx_hist[&key].is_null() {
            self.mpx_hist[&key] = json!({});
        }
        let hist = &mut self.mpx_hist[&key];
        let prev = hist["prev"].clone();
        let mut inv_now = json!({});
        for item in MPX_ITEMS {
            inv_now[item] = json!(int(&obs["market"]["inventory"][item]));
        }
        let draw_units = |s| i64::from(s % 4 == 0) + i64::from(s % 24 == 0);
        if truth(&prev) && int(&prev["step"]) == step - 1 {
            for item in MPX_ITEMS {
                let d = int(&inv_now[item]) - int(&prev["inv"][item]) + draw_units(step - 1)
                    - int(&prev["own"][item]);
                if hist[item].is_null() {
                    hist[item] = json!([]);
                }
                let item_hist = hist[item].as_array_mut().expect("MODELPX history");
                item_hist.push(json!(d.max(0)));
                if item_hist.len() > 12 {
                    item_hist.drain(..6);
                }
            }
        }
        hist["prev"] = json!({"step":step,"inv":inv_now,"own":{}});
        let mut market = orders(&action);
        for o in &market {
            if array(o).len() >= 3 && o[0] == "SELL" && MPX_ITEMS.contains(&text(&o[1])) {
                increment(&mut hist["prev"]["own"], text(&o[1]), int(&o[2]));
            }
        }
        if native_route(core, obs).is_none() {
            return action;
        }
        let stock = Core::projected_shed(&action, &View::new(obs));
        let mut added = false;
        for item in MPX_ITEMS {
            if has_item(&market, item, false) || market.len() >= 10 {
                continue;
            }
            let avail = int(&stock[item]);
            if avail <= 0 || int(&obs["market"]["prices"][item]) <= 1 {
                continue;
            }
            let rival = array(&hist[item]);
            let recent = &rival[rival.len().saturating_sub(4)..];
            let rival_avg = if recent.is_empty() {
                0.0
            } else {
                recent.iter().map(int).sum::<i64>() as f64 / recent.len() as f64
            };
            let inv = int(&obs["market"]["inventory"][item]);
            let p_cur = market_price(item, inv as f64, None) as f64;
            let inv_next = inv as f64 + rival_avg + 6.0 - draw_units(step) as f64;
            let p_next = market_price(item, (inv_next as i64).max(0) as f64, None) as f64;
            if p_next < p_cur - 0.5 {
                let take = avail.min(3);
                market.insert(0, json!(["SELL", item, take]));
                increment(&mut self.mpx_report, "mpx_fires", 1);
                increment(&mut self.mpx_report, "mpx_units", take);
                increment(&mut hist["prev"]["own"], item, take);
                added = true;
            }
        }
        if !added {
            return action;
        }
        let mut result = action;
        market.truncate(10);
        set_orders(&mut result, market);
        result
    }

    pub fn shield(&mut self, obs: &Value, action: Value) -> Value {
        if !action.is_object()
            || !array(&obs["town"]["unlocked_shops"])
                .iter()
                .any(|s| ["PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"].contains(&text(s)))
        {
            return action;
        }
        let mut market = orders(&action);
        if market.is_empty() {
            return action;
        }
        let farm = &obs["farms"][int(&obs["player"]) as usize];
        let shed = &obs["private"]["shed"];
        let prices = &obs["market"]["prices"];
        let quads = array(&farm["unlocked_quadrants"]).len() as i64;
        let mut cash = num(&farm["money"]);
        let mut sold = json!({});
        for o in &market {
            let op = text(&o[0]);
            let item = text(&o[1]);
            if op == "SELL" && array(o).len() >= 3 {
                let take = int(&o[2]).max(0).min(int(&shed[item]) - int(&sold[item]));
                if take > 0 {
                    cash += take as f64 * num(&prices[item]);
                    increment(&mut sold, item, take);
                }
            } else if op == "BUY_ANIMAL" && array(o).len() >= 3 {
                let cost = match item {
                    "SHEEP" => 500,
                    "GOOSE" => 300,
                    _ => 400,
                };
                cash -= (cost * int(&o[2]).max(0)) as f64;
            } else if op == "BUY_SEED" && array(o).len() >= 3 {
                let cost = match item {
                    "WHEAT" => 10,
                    "CARROT" => 20,
                    "TOMATO" => 50,
                    "MELON" => 80,
                    _ => 100,
                };
                cash -= (cost * int(&o[2]).max(0)) as f64;
            } else if op == "BUY_PRODUCT" && array(o).len() >= 3 {
                cash -= num(&prices[item]) * int(&o[2]).max(0) as f64;
            } else if op == "BUY_LAND" {
                cash -= match quads - 1 {
                    0 => 1000.0,
                    1 => 2000.0,
                    _ => 4000.0,
                };
            }
        }
        let mut need = -cash;
        if need <= 0.0 {
            return action;
        }
        let mut candidates: Vec<(f64, &str, i64)> = SM_PRODUCTS
            .into_iter()
            .filter_map(|item| {
                let price = num(&prices[item]);
                let avail = int(&shed[item]) - int(&sold[item]);
                (price > 1.0 && avail > 0).then_some((price, item, avail))
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.cmp(a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        let mut added = Vec::new();
        for (price, item, avail) in candidates {
            if need <= 0.0 || market.len() + added.len() >= 10 {
                break;
            }
            let q = avail.min((need / price) as i64 + 1);
            if q <= 0 {
                continue;
            }
            added.push(json!(["SELL", item, q]));
            need -= q as f64 * price;
            increment(&mut self.sm_report, "sm_units", q);
        }
        if added.is_empty() {
            return action;
        }
        increment(&mut self.sm_report, "sm_shields", 1);
        let idx = market
            .iter()
            .position(|o| text(&o[0]).starts_with("BUY"))
            .unwrap_or(market.len());
        market.splice(idx..idx, added);
        market.truncate(10);
        let mut result = action;
        set_orders(&mut result, market);
        result
    }

    pub fn reports(&self) -> Value {
        json!({"fx":self.fx_report,"dp":self.dp_report,"mp":self.mp_report,
            "bd":self.bd_report,"mpx":self.mpx_report,"sm":self.sm_report})
    }

    pub fn states(&self) -> Value {
        json!({"fx":self.fx_states,"ev":self.ev_states,"bd":self.bd_states,"mpx_hist":self.mpx_hist})
    }

    pub fn inject(&mut self, states: &Value) {
        for (key, target) in [
            ("fx", &mut self.fx_states),
            ("ev", &mut self.ev_states),
            ("bd", &mut self.bd_states),
            ("mpx_hist", &mut self.mpx_hist),
        ] {
            if let Some(v) = states.get(key) {
                *target = v.clone();
            }
        }
        if let Some(reports) = states.get("reports") {
            for (key, target) in [
                ("fx", &mut self.fx_report),
                ("dp", &mut self.dp_report),
                ("mp", &mut self.mp_report),
                ("bd", &mut self.bd_report),
                ("mpx", &mut self.mpx_report),
                ("sm", &mut self.sm_report),
            ] {
                if let Some(v) = reports.get(key) {
                    *target = v.clone();
                }
            }
        }
    }
}
