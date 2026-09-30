//! V47 EXP293 sale advance and market front-loading.
//! Source: agents/v47/main.py (`_ADV_*`, `_adv_apply`, `_adv_frontload`, advance `agent`).
//! Mechanism after sdy623 / jaxa623 "Beyond 48-0" (Apache-2.0), own implementation
//! by the upstream project.
use super::race::{Race, standard};
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};

pub const LOOK: i64 = 3;
pub const FROM: i64 = 144;
pub const TO: i64 = 718;
pub const ITEMS: [&str; 7] = [
    "STRAWBERRY",
    "WOOL",
    "EGG",
    "MILK",
    "MELON",
    "CARROT",
    "TOMATO",
];

/// tetsutani v65 "demand-preserving" lookahead (`_ADV_LOOK=14`).
pub const DEMAND_LOOK: i64 = 14;

#[derive(Clone, Debug)]
pub struct Advance {
    pub report: Value,
    /// tetsutani v65: 14-turn lookahead unless two bakeries are open, and
    /// WOOL sales five or more turns out are left alone while a yarn store
    /// is open. False for the frozen V47/V48 controllers.
    pub demand_preserving: bool,
    /// cha22 subtracts already-booked native lead-sale debt.
    pub subtract_debts: bool,
}
impl Default for Advance {
    fn default() -> Self {
        Self {
            report: json!({"adv_turns":0,"adv_units":0,"adv_errors":0}),
            demand_preserving: false,
            subtract_debts: false,
        }
    }
}

fn future(route: i64, t: i64) -> Result<Vec<Value>, &'static str> {
    let r = if t >= 648 { 2 } else { route };
    if routes()[r.to_string()].is_null() {
        return Err("adv: unknown route");
    }
    Ok(orders(&route_action(r, t)))
}

impl Advance {
    /// `_adv_apply`: sell held cash products the tape sells within three turns.
    /// Equal-price items keep their first tape appearance order; the source
    /// iterates a Python set there, which is unordered on ties.
    pub fn apply(
        &mut self,
        obs: &Value,
        action: Value,
        core: &mut Core,
    ) -> Result<Value, &'static str> {
        let step = int(obs.get("step").ok_or("adv: step")?);
        let key = int(&obs["player"]).to_string();
        if step.rem_euclid(24) == 23 || !(FROM..TO).contains(&step) {
            return Ok(action);
        }
        let native = core
            .players
            .get_mut(&key)
            .ok_or("adv: missing native state")?;
        if !native["sell_state"].is_object() {
            return Err("adv: missing sell_state");
        }
        if native["sell_state"]["r36_debts"].is_null() {
            native["sell_state"]["r36_debts"] = json!({});
        }
        let route = int(&native["route"]);
        let mut plan: Vec<(i64, String, i64)> = vec![];
        let mut first: Option<Value> = None;
        let shops = array(&obs["town"]["unlocked_shops"]);
        let yarn = shops.iter().any(|s| s == "YARN_STORE");
        let look = if self.demand_preserving && shops.iter().filter(|s| *s == "BAKERY").count() < 2
        {
            DEMAND_LOOK
        } else {
            LOOK
        };
        for off in 1..=look {
            let t = step + off;
            if t > 718 {
                break;
            }
            for o in future(route, t)? {
                if !truth(&o) || array(&o).len() < 3 {
                    continue;
                }
                if first.is_none() {
                    first = Some(o.clone());
                }
                if self.demand_preserving && off >= 5 && o[0] == "SELL" && o[1] == "WOOL" && yarn {
                    continue;
                }
                if o[0] == "SELL" && ITEMS.contains(&text(&o[1])) {
                    let q = int(&o[2]).max(0)
                        - if self.subtract_debts {
                            int(&native["sell_state"]["r36_debts"][t.to_string()][text(&o[1])])
                        } else {
                            0
                        };
                    if q > 0 {
                        plan.push((t, text(&o[1]).to_owned(), q));
                    }
                }
            }
        }
        let protected = first
            .filter(|f| f[0] == "SELL")
            .map(|f| text(&f[1]).to_owned());
        plan.retain(|(_, item, _)| Some(item) != protected.as_ref());
        if plan.is_empty() {
            return Ok(action);
        }
        let mut market = orders(&action);
        if market
            .iter()
            .any(|o| array(o).len() > 1 && o[0] == "BUY_PRODUCT")
        {
            return Ok(action);
        }
        let stock = Core::projected_shed(&action, &View::new(obs));
        let mut selling = json!({});
        for o in &market {
            if array(o).len() >= 3 && o[0] == "SELL" {
                increment(&mut selling, text(&o[1]), int(&o[2]).max(0));
            }
        }
        let picked: Vec<String> = commands(&action)
            .iter()
            .filter(|c| array(c).len() > 1 && c[0] == "PICKUP")
            .map(|c| text(&c[1]).to_owned())
            .collect();
        let prices = obs["market"].get("prices").ok_or("adv: prices")?;
        let mut items: Vec<String> = vec![];
        for (_, item, _) in &plan {
            if !items.contains(item) {
                items.push(item.clone());
            }
        }
        items.sort_by_key(|it| -int(&prices[it]));
        let mut added = 0;
        let mut extra: Vec<Value> = vec![];
        for item in items {
            if picked.contains(&item) || int(&prices[&item]) < 2 {
                continue;
            }
            let mut avail = int(&stock[&item]) - int(&selling[&item]);
            if avail < 1 {
                continue;
            }
            let hit = market
                .iter()
                .position(|o| array(o).len() >= 3 && o[0] == "SELL" && o[1] == item.as_str());
            if hit.is_none() && market.len() + extra.len() >= 10 {
                continue;
            }
            let mut n = 0;
            for (_, it, q) in &plan {
                if *it != item || avail <= 0 {
                    continue;
                }
                let take = (*q).min(avail);
                n += take;
                avail -= take;
            }
            if n < 1 {
                continue;
            }
            match hit {
                Some(index) => {
                    let current = int(&market[index][2]);
                    market[index][2] = json!(current + n);
                }
                None => extra.push(json!(["SELL", item, n])),
            }
            added += n;
        }
        if added == 0 {
            return Ok(action);
        }
        increment(&mut self.report, "adv_turns", 1);
        increment(&mut self.report, "adv_units", added);
        let mut result = action;
        extra.extend(market);
        set_orders(&mut result, extra);
        Ok(result)
    }

    /// `_adv_frontload`: sales first, then product purchases with the sales of
    /// items the list also buys, then everything else, all in original order.
    pub fn frontload(&mut self, obs: &Value, action: Value) -> Value {
        let market: Vec<Value> = orders(&action).into_iter().filter(truth).collect();
        if market.len() < 2 || int(&obs["step"]) < FROM {
            return action;
        }
        let buys: Vec<String> = market
            .iter()
            .filter(|o| array(o).len() > 1 && o[0] == "BUY_PRODUCT")
            .map(|o| text(&o[1]).to_owned())
            .collect();
        let is_front = |o: &Value| {
            array(o).len() >= 3 && o[0] == "SELL" && !buys.contains(&text(&o[1]).to_owned())
        };
        let is_mid = |o: &Value| {
            array(o).len() >= 3
                && (o[0] == "BUY_PRODUCT"
                    || (o[0] == "SELL" && buys.contains(&text(&o[1]).to_owned())))
        };
        let front: Vec<Value> = market.iter().filter(|o| is_front(o)).cloned().collect();
        let mid: Vec<Value> = market.iter().filter(|o| is_mid(o)).cloned().collect();
        let rest: Vec<Value> = market
            .iter()
            .filter(|o| !front.contains(o) && !mid.contains(o))
            .cloned()
            .collect();
        let mut new = front;
        new.extend(mid);
        new.extend(rest);
        if new == market {
            return action;
        }
        increment(&mut self.report, "front_turns", 1);
        let mut result = action;
        set_orders(&mut result, new);
        result
    }

    pub fn layer(
        &mut self,
        obs: &Value,
        config: &Value,
        action: Value,
        core: &mut Core,
        race: &mut Race,
    ) -> Value {
        let mut result = action;
        let outcome: Result<(), &'static str> = (|| {
            let step = int(obs.get("step").ok_or("adv: step")?);
            if step == 0 {
                for k in ["adv_turns", "adv_units", "adv_errors"] {
                    self.report[k] = json!(0);
                }
            }
            let standard = standard(config, false);
            if standard {
                result = self.apply(obs, result.clone(), core)?;
                result = self.frontload(obs, result.clone());
            }
            let key = int(&obs["player"]).to_string();
            if let Some(st) = race.states.get_mut(&key)
                && !st["prev_action"].is_null()
                && int(&st["step"]) == step
            {
                st["prev_action"] = result.clone();
            }
            Ok(())
        })();
        if outcome.is_err() {
            increment(&mut self.report, "adv_errors", 1);
        }
        result
    }
}
