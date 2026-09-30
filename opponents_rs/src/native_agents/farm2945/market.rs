//! 2945 Farm market-side layers: CTRTABLE, OVERFLOW and SHEDROOM.
//! Source: agents/farm2945/main.py (`_ct_apply`, `_ov_apply`, SHEDROOM `agent`).
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::contracts::{fields, same_stock};
use crate::native_agents::v43::core::Core;
use crate::native_agents::v43::late::{budget, delivery, market_stock};
use serde_json::{Value, json};

const SR_MARGIN: i64 = 4;

#[derive(Clone, Debug)]
pub struct Market {
    pub ctrtable: Value,
    /// The Smaller Market Shock overlay clears `CT_TABLE` before each parent
    /// call, so its counter layer must remain inert for the whole controller.
    pub disable_ctrtable: bool,
    /// cha22 empties only the counter table; its wrapper state still advances.
    pub empty_ctrtable: bool,
    pub report: Value,
}
impl Default for Market {
    fn default() -> Self {
        Self {
            ctrtable: json!({}),
            disable_ctrtable: false,
            empty_ctrtable: false,
            report: json!({"ct_fired":0,"ct_errors":0,"ov_turns":0,"ov_units":0,"ov_errors":0,
                "sr_turns":0,"sr_units":0,"sr_errors":0}),
        }
    }
}

/// `CT_TABLE`: (step, slot, order) wheat counters per turn-2 rival key.
fn counter_plan(money: f64, wheat: i64) -> Value {
    if money == 979.0 && wheat == 9989 {
        json!([
            [7, 0, ["BUY_PRODUCT", "WHEAT", 5]],
            [7, 1, ["SELL", "WHEAT", 5]],
            [8, 0, ["BUY_PRODUCT", "WHEAT", 5]],
            [8, 1, ["SELL", "WHEAT", 5]]
        ])
    } else if money == 33.0 && wheat == 9990 {
        json!([
            [3, 0, ["BUY_PRODUCT", "WHEAT", 20]],
            [3, 1, ["SELL", "WHEAT", 20]]
        ])
    } else {
        Value::Null
    }
}

impl Market {
    /// CTRTABLE: trade alongside a recognised rival tape's early wheat orders.
    pub fn ctrtable(&mut self, obs: &Value, action: Value) -> Value {
        if self.disable_ctrtable {
            return action;
        }
        let step = int(&obs["step"]);
        let key = seat(obs).to_string();
        if self.ctrtable[&key].is_null() || step <= int(&self.ctrtable[&key]["step"]) {
            self.ctrtable[&key] = json!({"step":-1});
        }
        let st = &mut self.ctrtable[&key];
        st["step"] = json!(step);
        if step == 2 {
            let rival = &obs["farms"][1 - seat(obs)];
            let money = py_round3(num(&rival["money"]));
            st["plan"] = if self.empty_ctrtable {
                Value::Null
            } else {
                counter_plan(money, int(&obs["market"]["inventory"]["WHEAT"]))
            };
            if truth(&st["plan"]) {
                inc(&mut self.report, "ct_fired", 1);
            }
        }
        let mut items: Vec<(i64, Value)> = array(&st["plan"])
            .iter()
            .filter(|entry| int(&entry[0]) == step)
            .map(|entry| (int(&entry[1]), entry[2].clone()))
            .collect();
        if items.is_empty() {
            return action;
        }
        items.sort_by_key(|(slot, _)| *slot);
        let mut market = orders(&action);
        for (slot, order) in items {
            while market.len() < slot as usize {
                market.push(json!(["SELL", "WHEAT", 0]));
            }
            market.insert(slot as usize, order);
        }
        market.truncate(MAX_ORDERS);
        let mut result = action;
        set_orders(&mut result, market);
        result
    }

    /// OVERFLOW (public V43 R148): sell the shed stock tonight's destroyed cargo replaces.
    pub fn overflow(&mut self, obs: &Value, action: Value) -> Value {
        match self.overflow_apply(obs, &action) {
            Ok(Some(result)) => result,
            Ok(None) => action,
            Err(_) => {
                inc(&mut self.report, "ov_errors", 1);
                action
            }
        }
    }

    fn overflow_apply(&mut self, obs: &Value, action: &Value) -> Fallible<Option<Value>> {
        if int(&obs["step"]) % 24 != 23 {
            return Ok(None);
        }
        let os = orders(action);
        if os.len() >= MAX_ORDERS || !budget(obs, &os) {
            return Ok(None);
        }
        let (_, private) = fields(obs, action)?;
        let (stock, _, _) = market_stock(&private["shed"], &os);
        let (original, loss) = delivery(stock.clone(), &private, true);
        if !truth(&loss) {
            return Ok(None);
        }
        let mut remaining = (100 - stock_total(&stock)).max(0);
        let mut tail = vec![];
        for bag in array(&private["inventories"]) {
            for (item, n) in bag.as_object().ok_or("overflow bag")? {
                let n = int(n).max(0);
                let take = n.min(remaining);
                remaining -= take;
                tail.extend((0..n - take).map(|_| item.clone()));
            }
        }
        let mut released = json!({});
        let mut best = None;
        for item in tail {
            increment(&mut released, &item, 1);
            if obs["market"]["prices"].get(&item).is_none()
                || int(&released[&item]) > int(&stock[&item])
            {
                break;
            }
            let lots = released.as_object().ok_or("overflow lots")?;
            if os.len() + lots.len() > MAX_ORDERS {
                break;
            }
            let mut proposed = os.clone();
            proposed.extend(lots.iter().map(|(p, n)| json!(["SELL", p, n])));
            let (after, _, _) = market_stock(&private["shed"], &proposed);
            let (final_stock, _) = delivery(after, &private, true);
            if same_stock(&original, &final_stock) {
                best = Some((proposed, stock_total(&released)));
            }
        }
        let Some((proposed, units)) = best else {
            return Ok(None);
        };
        inc(&mut self.report, "ov_turns", 1);
        inc(&mut self.report, "ov_units", units);
        let mut result = action.clone();
        set_orders(&mut result, proposed);
        Ok(Some(result))
    }

    /// SHEDROOM: sell shed goods at hours 22-23 before the night drop overflows.
    pub fn shedroom(&mut self, obs: &Value, action: Value, config: &Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        if step == 0 {
            for k in ["sr_turns", "sr_units", "sr_errors"] {
                self.report[k] = json!(0);
            }
        }
        if !(matches!(step % 24, 22 | 23) || (core.metav4 && step % 24 == 21)) || step >= 717 {
            return action;
        }
        let farm = own_farm(obs);
        let inventories = array(&obs["private"]["inventories"]);
        let view = View::new(obs);
        let proj = Core::projected_shed(&action, &view);
        let mut market = orders(&action);
        let mut left = proj.clone();
        let mut night_shed: i64 = proj
            .as_object()
            .into_iter()
            .flatten()
            .map(|(_, v)| int(v).max(0))
            .sum();
        for o in &market {
            if array(o).len() < 3 {
                continue;
            }
            if o[0] == "SELL" {
                let item = text(&o[1]);
                let got = int(&o[2]).max(0).min(int(&left[item]).max(0));
                increment(&mut left, item, -got);
                night_shed -= got;
            } else if o[0] == "BUY_PRODUCT" || o[0] == "BUY_ANIMAL" {
                night_shed += int(&o[2]).max(0);
            }
        }
        let commands = units(&action);
        let mut carried = 0;
        for (i, pos) in view.positions.iter().enumerate() {
            let mut held: i64 = inventories
                .get(i)
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
                .map(|(_, v)| int(v).max(0))
                .sum();
            let pass = json!(["PASS"]);
            let cmd = commands.get(i).filter(|c| truth(c)).unwrap_or(&pass);
            let t = if pos.is_array() {
                tile(farm, position(pos))
            } else {
                &Value::Null
            };
            let adjacent = shed_adjacent(pos, view.board);
            match text(&cmd[0]) {
                "DROP" if adjacent => held = 0,
                "HARVEST" if t.is_object() => held += int(&t["yield_units"]).max(0),
                "COLLECT_FERTILIZER" if t.is_object() && truth(&t["fertilizer_available"]) => {
                    held += 1
                }
                "FEED" | "FERTILIZE" if held > 0 => held -= 1,
                "PICKUP" if array(cmd).len() >= 2 && adjacent => {
                    held += if array(cmd).len() >= 3 {
                        int(&cmd[2]).max(1)
                    } else {
                        1
                    }
                }
                _ => {}
            }
            carried += held;
        }
        let cap = if config.is_object() {
            config.get("shedCapacity").map(int).unwrap_or(100)
        } else {
            100
        };
        let mut overflow = night_shed + carried - cap + if core.metav4 { 8 } else { SR_MARGIN };
        if overflow <= 0 {
            return action;
        }
        let (mut feed, mut fertilize) = (0, 0);
        for t in step + 1..719.min(step + 25) {
            for c in units(&late_tape(core, obs, t)) {
                match text(&c[0]) {
                    "FEED" => feed += 1,
                    "FERTILIZE" => fertilize += 1,
                    _ => {}
                }
            }
        }
        let prices = &obs["market"]["prices"];
        let mut cands = vec![];
        for item in PRODUCTS {
            let need = match item {
                "WHEAT" => feed,
                "FERTILIZER" => fertilize,
                _ => 0,
            };
            let spare = int(&left[item]) - need;
            if spare > 0 && int(&prices[item]) >= 2 {
                cands.push((int(&prices[item]), item, spare));
            }
        }
        cands.sort();
        let mut sold_now = 0;
        for (_, item, spare) in cands {
            if overflow <= 0 {
                break;
            }
            let q = spare.min(overflow);
            if !bump_sell(&mut market, item, q) {
                if market.len() >= MAX_ORDERS {
                    continue;
                }
                market.push(json!(["SELL", item, q]));
            }
            overflow -= q;
            sold_now += q;
        }
        if sold_now == 0 {
            return action;
        }
        inc(&mut self.report, "sr_turns", 1);
        inc(&mut self.report, "sr_units", sold_now);
        let mut result = action;
        set_orders(&mut result, market);
        result
    }
}
