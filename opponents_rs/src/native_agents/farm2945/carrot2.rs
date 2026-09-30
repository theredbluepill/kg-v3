//! 2945 Farm CARROT2 layer: carrot instead of wheat when the carrot book pays,
//! judged per tile from the workers' remaining visits.
//! Source: agents/farm2945/main.py (`_ca_*` and the CARROT2 `agent`).
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};

const FROM: i64 = 6;
const TO: i64 = 28;
const BUFFER: i64 = 8;
const FEED_DAYS: i64 = 2;
const CASH: i64 = 800;
/// `3 * carrot - 20 > 4 * wheat - 10 + _CA_MARGIN` with `_CA_MARGIN = -5`, `_CA_DROP = 0`.
const MARGIN: i64 = -5;

const REPORT_KEYS: [&str; 9] = [
    "ca_swaps",
    "ca_rescues",
    "ca_harvested",
    "ca_sold",
    "ca_seed_bought",
    "ca_wheat_seed_saved",
    "ca_carrot_seed_saved",
    "ca_feed_block",
    "ca_errors",
];

#[derive(Clone, Debug)]
pub struct Carrot2 {
    pub states: Value,
    pub report: Value,
}
impl Default for Carrot2 {
    fn default() -> Self {
        let mut report = json!({"ca_min_wheat": 999});
        for k in REPORT_KEYS {
            report[k] = json!(0);
        }
        Self {
            states: json!({}),
            report,
        }
    }
}

/// `_ca_visits`: non-move commands issued on `pos` from this step until `t_end`.
pub fn visits(
    obs: &Value,
    action: &Value,
    core: &Core,
    pos: Pos,
    t_end: i64,
    start: Option<i64>,
) -> Vec<(i64, String)> {
    let step = int(&obs["step"]);
    let farm = own_farm(obs);
    let board = array(&farm["tiles"]).len() as i64;
    let half = board / 2;
    let mut positions = unit_positions(farm);
    let mut out = vec![];
    for t in step..=t_end.min(719) {
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
            } else if *p == pos && start.is_none_or(|s| t >= s) {
                out.push((t, op.to_owned()));
            }
        }
        for _ in 0..hires(&act) {
            positions.push(spawn(&positions, board));
        }
        if t % 24 == 23 {
            positions = vec![(half - 1, half - 1)];
        }
    }
    out
}

/// `_ca_decays`: decay events at steps s in [max(a, mls), b) with (s - mls) even.
fn decays(mls: i64, a: i64, b: i64) -> i64 {
    let a = a.max(mls);
    if b <= a {
        return 0;
    }
    let first = if (a - mls) % 2 == 0 { a } else { a + 1 };
    if first >= b {
        0
    } else {
        (b - 1 - first) / 2 + 1
    }
}

pub struct Growth {
    pub y0: i64,
    pub fert_until: i64,
    pub watered_day: i64,
    pub now_step: i64,
}

/// `_ca_yield_path`: (harvest units, best rescue units) for a crop following `visits`.
pub fn yield_path(
    crop: &str,
    planted: i64,
    visits: &[(i64, String)],
    growth: Growth,
) -> (i64, i64) {
    let (myd, cap) = if crop == "WHEAT" { (4, 6) } else { (3, 4) };
    let lo = (myd + 1) / 2;
    let mls = (planted + myd + 1) * 24;
    let mut y = growth.y0;
    let mut watered_day = growth.watered_day;
    let mut best_rescue = 0;
    for (t, op) in visits {
        let day = t / 24;
        let age = day - planted;
        let now = y - decays(mls, growth.now_step, *t);
        if now <= 0 && *t > mls {
            return (0, best_rescue);
        }
        match op.as_str() {
            "HARVEST" => return (if age >= 2 { now.max(0) } else { 0 }, best_rescue),
            "PLANT" | "DIG" | "BUILD_COOP" | "BUILD_PASTURE" => return (0, best_rescue),
            _ => {}
        }
        if age >= 2 && now > best_rescue && *t > growth.now_step {
            best_rescue = now;
        }
        if op == "WATER" && (lo..=myd).contains(&age) && day != watered_day {
            watered_day = day;
            y = cap.min(y + if growth.fert_until >= day { 2 } else { 1 });
        }
    }
    (0, best_rescue)
}

fn wheat_total(obs: &Value) -> i64 {
    int(&obs["private"]["shed"]["WHEAT"])
        + array(&obs["private"]["inventories"])
            .iter()
            .map(|i| int(&i["WHEAT"]))
            .sum::<i64>()
}

fn feed_need(obs: &Value, core: &Core, step: i64, days: i64) -> i64 {
    (step..=719.min(step + 24 * days))
        .flat_map(|t| units(&late_tape(core, obs, t)))
        .filter(|c| c[0] == "FEED")
        .count() as i64
}

impl Carrot2 {
    pub fn layer(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        let who = seat(obs).to_string();
        if step == 0 || self.states[&who].is_null() || step <= int(&self.states[&who]["step"]) {
            self.states[&who] =
                json!({"step":-1,"tiles":{},"spare_wheat":0,"spare_carrot":0,"credit":0});
            if step == 0 {
                for k in REPORT_KEYS {
                    self.report[k] = json!(0);
                }
                self.report["ca_min_wheat"] = json!(999);
            }
        }
        self.states[&who]["step"] = json!(step);
        if step > 717 {
            return action;
        }
        let st = &mut self.states[&who];
        let day = step / 24;
        let from = if core.cha22 { 10 } else { FROM };
        let margin = if core.cha22 { -20 } else { MARGIN };
        let farm = own_farm(obs);
        let private = &obs["private"];
        let prices = &obs["market"]["prices"];
        let (p_c, p_w) = (int(&prices["CARROT"]), int(&prices["WHEAT"]));
        let positions = unit_positions(farm);
        let mut commands = units(&action);
        let mut market = orders(&action);
        let mut changed = false;
        if (from..=TO + 4).contains(&day) {
            let low = int(&self.report["ca_min_wheat"]).min(wheat_total(obs));
            self.report["ca_min_wheat"] = json!(low);
        }
        // 1. bookkeeping and rescue of swapped carrots
        let tracked: Vec<(String, i64)> = st["tiles"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| (k.clone(), int(v)))
            .collect();
        for (name, planted) in tracked {
            let pos = unkey(&name);
            let t = tile(farm, pos);
            let forget = |st: &mut Value| {
                if let Some(m) = st["tiles"].as_object_mut() {
                    m.shift_remove(&name);
                }
            };
            let same = t.is_object()
                && t["crop"] == "CARROT"
                && t.get("planted_day").map(int).unwrap_or(-9) == planted;
            if !same {
                forget(st);
                continue;
            }
            let Some(i) = positions
                .iter()
                .enumerate()
                .position(|(i, p)| *p == pos && i < commands.len())
            else {
                continue;
            };
            let op = text(&commands[i][0]).to_owned();
            let yu = int(&t["yield_units"]);
            if op == "HARVEST" {
                if day - planted >= 2 && yu > 0 {
                    inc(st, "credit", yu);
                    inc(&mut self.report, "ca_harvested", yu);
                    forget(st);
                }
                continue;
            }
            if move_delta(&op).is_some() || day - planted < 2 || yu <= 0 {
                continue;
            }
            let seen = visits(obs, &action, core, pos, (planted + 5) * 24, None);
            let (harvest, later) = yield_path(
                "CARROT",
                planted,
                &seen,
                Growth {
                    y0: yu,
                    fert_until: t.get("fertilized_until_day").map(int).unwrap_or(-1),
                    watered_day: if truth(&t["watered_today"]) { day } else { -1 },
                    now_step: step,
                },
            );
            if yu > harvest.max(later) {
                commands[i] = json!(["HARVEST"]);
                inc(st, "credit", yu);
                inc(&mut self.report, "ca_harvested", yu);
                inc(&mut self.report, "ca_rescues", 1);
                forget(st);
                changed = true;
            }
        }
        // 2. swaps
        let planting_carrot = |commands: &[Value]| {
            commands
                .iter()
                .filter(|c| is_pair(c, "PLANT", "CARROT"))
                .count() as i64
        };
        let pays_now = 3 * p_c - 20 > 4 * p_w - 10 + margin;
        if (from..=TO).contains(&day) && pays_now {
            let mut seeds = int(&st["spare_carrot"])
                .min(int(&private["seeds"]["CARROT"]) - planting_carrot(&commands));
            let mut wheat_ok = None;
            for i in 0..commands.len() {
                if !is_pair(&commands[i], "PLANT", "WHEAT") || i >= positions.len() || seeds <= 0 {
                    continue;
                }
                let pos = positions[i];
                if !tile(farm, pos).is_null() {
                    continue;
                }
                let ok = *wheat_ok.get_or_insert_with(|| {
                    wheat_total(obs)
                        >= feed_need(obs, core, step, if core.metav4 { 1 } else { FEED_DAYS })
                });
                if !ok {
                    inc(&mut self.report, "ca_feed_block", 1);
                    break;
                }
                let seen = visits(obs, &action, core, pos, (day + 6) * 24, Some(step + 1));
                let fresh = || Growth {
                    y0: 1,
                    fert_until: -1,
                    watered_day: -1,
                    now_step: 0,
                };
                let (wheat_units, _) = yield_path("WHEAT", day, &seen, fresh());
                let (harvest, rescue) = yield_path("CARROT", day, &seen, fresh());
                let carrot_units = harvest.max(rescue);
                if carrot_units * p_c - 20 > wheat_units * p_w - 10 + margin {
                    commands[i] = json!(["PLANT", "CARROT"]);
                    seeds -= 1;
                    inc(st, "spare_carrot", -1);
                    st["tiles"][key(pos)] = json!(day);
                    inc(st, "spare_wheat", 1);
                    inc(&mut self.report, "ca_swaps", 1);
                    changed = true;
                }
            }
        }
        // 3. seeds
        let mut kept = vec![];
        for mut o in market {
            if array(&o).len() >= 3 && o[0] == "BUY_SEED" && (o[1] == "WHEAT" || o[1] == "CARROT") {
                let wheat = o[1] == "WHEAT";
                let spare = if wheat { "spare_wheat" } else { "spare_carrot" };
                let cut = int(&o[2]).min(int(&st[spare]));
                if cut > 0 {
                    inc(st, spare, -cut);
                    let saved = if wheat {
                        "ca_wheat_seed_saved"
                    } else {
                        "ca_carrot_seed_saved"
                    };
                    inc(&mut self.report, saved, cut);
                    changed = true;
                    if int(&o[2]) - cut <= 0 {
                        continue;
                    }
                    o = json!([o[0], o[1], int(&o[2]) - cut]);
                }
            }
            kept.push(o);
        }
        market = kept;
        if (from..TO).contains(&day) && pays_now && market.len() < MAX_ORDERS {
            let have = int(&private["seeds"]["CARROT"]) - planting_carrot(&commands);
            let buying: i64 = market
                .iter()
                .filter(|o| is_order(o, "BUY_SEED", "CARROT"))
                .map(|o| int(&o[2]))
                .sum();
            let q = BUFFER - have - buying;
            if q > 0 && int(&farm["money"]) >= CASH + 20 * q {
                market.push(json!(["BUY_SEED", "CARROT", q]));
                inc(st, "spare_carrot", q);
                inc(&mut self.report, "ca_seed_bought", q);
                changed = true;
            }
        }
        // 4. sell credited carrots
        if int(&st["credit"]) > 0 && p_c >= 2 && market.len() < MAX_ORDERS {
            let mut view_action = with_units(&json!({}), commands.clone());
            set_orders(&mut view_action, market.clone());
            let stock = int(&Core::projected_shed(&view_action, &View::new(obs))["CARROT"]);
            let q = int(&st["credit"]).min(stock - selling(&market, "CARROT"));
            if q > 0 {
                market.insert(0, json!(["SELL", "CARROT", q]));
                inc(st, "credit", -q);
                inc(&mut self.report, "ca_sold", q);
                changed = true;
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
