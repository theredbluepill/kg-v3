//! 2945 Farm CAPHARV layer: harvest animals whose next production would overflow.
//! Source: agents/farm2945/main.py (`_ch_*` and the CAPHARV `agent`).
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};
use std::collections::HashMap;

const SHED: i64 = 90;
const REPORT_KEYS: [&str; 6] = [
    "ch_collect_swaps",
    "ch_care_swaps",
    "ch_saved",
    "ch_sold",
    "ch_shed_block",
    "ch_errors",
];

/// `_CH_ANIMALS`: product, yield cap, first production age and interval.
fn animal(name: &str) -> Option<(&'static str, i64, i64, i64)> {
    match name {
        "GOOSE" => Some(("EGG", 4, 4, 1)),
        "COW" => Some(("MILK", 6, 8, 2)),
        "SHEEP" => Some(("WOOL", 6, 6, 3)),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct CapHarv {
    pub states: Value,
    pub report: Value,
}
impl Default for CapHarv {
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

/// `_ch_visits_today`: non-move commands per tile from the next step to midnight.
fn visits_today(obs: &Value, action: &Value, core: &Core) -> HashMap<Pos, Vec<String>> {
    let step = int(&obs["step"]);
    let farm = own_farm(obs);
    let board = array(&farm["tiles"]).len() as i64;
    let mut positions = unit_positions(farm);
    let mut out: HashMap<Pos, Vec<String>> = HashMap::new();
    for t in step..(step / 24 + 1) * 24 {
        let act = if t == step {
            action.clone()
        } else {
            late_tape(core, obs, t)
        };
        let commands = units(&act);
        for (i, p) in positions.iter_mut().enumerate() {
            let op = commands
                .get(i)
                .filter(|c| truth(c))
                .map(|c| text(&c[0]))
                .unwrap_or("PASS");
            if let Some((dx, dy)) = move_delta(op) {
                let (nx, ny) = (p.0 + dx, p.1 + dy);
                if (0..board).contains(&nx) && (0..board).contains(&ny) {
                    *p = (nx, ny);
                }
            } else if t > step {
                out.entry(*p).or_default().push(op.to_owned());
            }
        }
        for _ in 0..hires(&act) {
            positions.push(spawn(&positions, board));
        }
    }
    out
}

impl CapHarv {
    pub fn layer(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        let who = seat(obs).to_string();
        if step == 0 || self.states[&who].is_null() || step <= int(&self.states[&who]["step"]) {
            self.states[&who] = json!({"step":-1,"credit":{}});
            if step == 0 {
                for k in REPORT_KEYS {
                    self.report[k] = json!(0);
                }
            }
        }
        let st = &mut self.states[&who];
        st["step"] = json!(step);
        if step > 717 {
            return action;
        }
        let day = step / 24;
        let farm = own_farm(obs);
        let private = &obs["private"];
        let prices = &obs["market"]["prices"];
        let positions = unit_positions(farm);
        let mut commands = units(&action);
        let mut market = orders(&action);
        let mut changed = false;
        let mut visits = None;
        let total = |stock: &Value| -> i64 {
            stock
                .as_object()
                .into_iter()
                .flatten()
                .map(|(_, v)| int(v))
                .sum()
        };
        for i in 0..positions.len().min(commands.len()) {
            let op = text(&commands[i][0]).to_owned();
            if op != "CARE" && op != "COLLECT_FERTILIZER" {
                continue;
            }
            let pos = positions[i];
            let t = tile(farm, pos);
            let Some((product, cap, first, interval)) = animal(text(&t["animal"])) else {
                continue;
            };
            let since = day + 1 - t.get("placed_day").map(int).unwrap_or(99) - first;
            if since < 0 || since % interval != 0 {
                continue;
            }
            let y = int(&t["yield_units"]);
            let later = visits
                .get_or_insert_with(|| visits_today(obs, &action, core))
                .get(&pos)
                .cloned()
                .unwrap_or_default();
            if later.iter().any(|op| op == "HARVEST") {
                continue;
            }
            let fed = truth(&t["fed_today"]) || later.iter().any(|op| op == "FEED");
            let produced = 1 + if fed {
                int(&t["pending_care_bonus"])
            } else {
                0
            };
            let overflow = y + produced - cap;
            if overflow <= 0 || y <= 0 {
                continue;
            }
            let collecting = op == "COLLECT_FERTILIZER";
            if collecting {
                if overflow * int(&prices[product]) <= int(&prices["FERTILIZER"]) {
                    continue;
                }
            } else if later.iter().any(|op| op == "COLLECT_FERTILIZER") || overflow <= 1 {
                continue;
            }
            let carried: i64 = array(&private["inventories"]).iter().map(total).sum();
            if total(&private["shed"]) + carried + y >= if core.cha22 { 100 } else { SHED } {
                inc(&mut self.report, "ch_shed_block", 1);
                continue;
            }
            commands[i] = json!(["HARVEST"]);
            let (swaps, saved) = if collecting {
                ("ch_collect_swaps", overflow)
            } else {
                ("ch_care_swaps", overflow - 1)
            };
            inc(&mut self.report, swaps, 1);
            inc(&mut self.report, "ch_saved", saved);
            increment(&mut st["credit"], product, saved);
            changed = true;
        }
        let credits: Vec<(String, i64)> = st["credit"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), int(v)))
            .collect();
        if credits.iter().any(|(_, v)| *v > 0) {
            let mut view_action = with_units(&json!({}), commands.clone());
            set_orders(&mut view_action, market.clone());
            let stock = Core::projected_shed(&view_action, &View::new(obs));
            for (product, credit) in credits {
                if credit <= 0 || market.len() >= MAX_ORDERS || int(&prices[&product]) < 2 {
                    continue;
                }
                let q = credit.min(int(&stock[&product]) - selling(&market, &product));
                if q > 0 {
                    if !bump_sell(&mut market, &product, q) {
                        market.insert(0, json!(["SELL", product, q]));
                    }
                    st["credit"][&product] = json!(credit - q);
                    inc(&mut self.report, "ch_sold", q);
                    changed = true;
                }
            }
        }
        if !changed {
            return action;
        }
        market.truncate(MAX_ORDERS);
        let mut result = with_units(&action, commands);
        set_orders(&mut result, market);
        result
    }
}
