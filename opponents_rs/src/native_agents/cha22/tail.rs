//! Native cha22 final suffix, in source order: CXD, EXP410, EXP402, MG, IG.
//! Source and upstream notices: agents/cha22/main.py, SHA-256
//! 127ed3e62988c0474d386db6527ae8ca9de9bb1fe7004128557ddef67126c652.
//! EXP410/EXP402 and carrot helpers are AST-identical to V56. The per-item
//! factor-margin cache is copied from the V47 port of the identical upstream
//! `_v44y_factor_margin`; shared `lockstep` retains the per-unit price semantics.
use crate::native_agents::farm2945::{
    carrot2::{self, Growth},
    util::*,
};
use crate::native_agents::metav4::Metav4Controller;
use crate::native_agents::v43::{
    common::{PRODUCTS, View, array, int, num, text, truth},
    core::Core,
    terminal,
};
use crate::native_agents::v47::lockstep::lockstep;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};

const CASH: [&str; 7] = [
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
];
const FIXED: [&str; 4] = ["HIRE", "BUY_SEED", "BUY_ANIMAL", "BUY_LAND"];
const CXD_BUDGET: i64 = 800;

#[derive(Clone, Debug)]
pub struct Tail {
    pub cxd_report: Value,
    pub fert_report: Value,
    pub seed_report: Value,
    pub mg_report: Value,
    pub ig_report: Value,
    pub seed_cache: Value,
    pub cxd_models: Value,
    pub cxd_parent_orders: Value,
}

impl Default for Tail {
    fn default() -> Self {
        Self {
            cxd_report: json!({"cxd_turns":0,"cxd_gain":0.0,"cxd_evals":0,"cxd_budget_hits":0,"cxd_errors":0}),
            fert_report: json!({"skips":0,"covered":0,"capped":0,"errors":0}),
            seed_report: json!({"cut_units":0,"saved_cost":0,"changed_turns":0,"errors":0}),
            mg_report: json!({"mg_turns":0,"mg_merged":0,"mg_errors":0}),
            ig_report: json!({"queue_changed_turns":0,"zeroed_orders":0,"pulled_orders":0,"pulled_slots":0,"errors":0}),
            seed_cache: json!({}),
            cxd_models: json!([]),
            cxd_parent_orders: json!([]),
        }
    }
}

fn zero(report: &mut Value) {
    for value in report.as_object_mut().unwrap().values_mut() {
        *value = json!(0);
    }
}

/// Fallible Python integer coercion for quantities in the two new queue layers.
fn quantity(v: &Value) -> Result<i64, &'static str> {
    if v.is_number() || v.is_boolean() {
        return Ok(int(v));
    }
    v.as_str()
        .and_then(|s| s.trim().parse().ok())
        .ok_or("invalid quantity")
}

/// Next k-permutation of 0..n, in itertools.permutations input order.
fn next_positions(positions: &mut [usize], n: usize) -> bool {
    for i in (0..positions.len()).rev() {
        if let Some(next) = (positions[i] + 1..n).find(|p| !positions[..i].contains(p)) {
            positions[i] = next;
            for j in i + 1..positions.len() {
                positions[j] = (0..n).find(|p| !positions[..j].contains(p)).unwrap();
            }
            return true;
        }
    }
    false
}

type Schedule = Vec<(usize, String, i64)>;

fn valid(o: &Value) -> bool {
    truth(o)
        && array(o).len() >= 3
        && (o[0] == "SELL" || o[0] == "BUY_PRODUCT")
        && PRODUCTS.contains(&text(&o[1]))
}

/// `_v44y_factor_margin`: independent per-item schedules with exact caching.
struct Margin<'a> {
    opp_len: usize,
    opp_items: Vec<String>,
    opp_schedules: HashMap<String, Vec<Value>>,
    inv0: BTreeMap<String, i64>,
    stock: BTreeMap<String, i64>,
    params: Option<&'a Value>,
    cache: HashMap<(String, Schedule), f64>,
}

impl<'a> Margin<'a> {
    fn new(
        opp: &[Value],
        inv0: BTreeMap<String, i64>,
        stock: BTreeMap<String, i64>,
        params: Option<&'a Value>,
    ) -> Self {
        let mut opp_items = vec![];
        let mut opp_schedules: HashMap<String, Vec<Value>> = HashMap::new();
        for (i, order) in opp.iter().enumerate() {
            if valid(order) {
                let item = text(&order[1]).to_owned();
                if !opp_items.contains(&item) {
                    opp_items.push(item.clone());
                }
                let padded = opp_schedules
                    .entry(item)
                    .or_insert_with(|| vec![json!([]); opp.len()]);
                padded[i] = order.clone();
            }
        }
        Self {
            opp_len: opp.len(),
            opp_items,
            opp_schedules,
            inv0,
            stock,
            params,
            cache: HashMap::new(),
        }
    }

    fn of(&mut self, cand: &[Value]) -> Result<f64, &'static str> {
        let mut order: Vec<String> = self.opp_items.clone();
        let mut schedules: HashMap<String, Schedule> =
            order.iter().map(|item| (item.clone(), vec![])).collect();
        for (i, o) in cand.iter().enumerate() {
            if valid(o) {
                let item = text(&o[1]).to_owned();
                if !schedules.contains_key(&item) {
                    order.push(item.clone());
                }
                schedules.entry(item).or_default().push((
                    i,
                    text(&o[0]).to_owned(),
                    quantity(&o[2])?,
                ));
            }
        }
        let mut total = 0.0;
        for item in order {
            let schedule = schedules.remove(&item).unwrap_or_default();
            let key = (item.clone(), schedule.clone());
            let value = match self.cache.get(&key) {
                Some(v) => *v,
                None => {
                    let mut mine = vec![json!([]); cand.len()];
                    for (i, op, n) in &schedule {
                        mine[*i] = json!([op, item, n]);
                    }
                    let theirs = self
                        .opp_schedules
                        .get(&item)
                        .cloned()
                        .unwrap_or_else(|| vec![json!([]); self.opp_len]);
                    let start = *self.inv0.get(&item).ok_or("lockstep: inv0")?;
                    let held = *self.stock.get(&item).unwrap_or(&0);
                    let mut inv = BTreeMap::from([(item.clone(), start)]);
                    let mut stock = [
                        BTreeMap::from([(item.clone(), held)]),
                        BTreeMap::from([(item.clone(), held)]),
                    ];
                    let (a, b) = lockstep(&mine, &theirs, &mut inv, &mut stock, self.params)?;
                    let v = a - b;
                    self.cache.insert(key, v);
                    v
                }
            };
            total += value;
        }
        Ok(total)
    }
}

impl Tail {
    /// MG and IG reset before calling their parent; other resets happen in the
    /// corresponding after-parent method. CXD models/parent orders are not reset.
    pub fn before(&mut self, obs: &Value) {
        if int(&obs["step"]) == 0 {
            zero(&mut self.mg_report);
            zero(&mut self.ig_report);
        }
    }

    pub fn reorder(&mut self, obs: &Value, action: Value) -> Value {
        if int(&obs["step"]) == 0 {
            zero(&mut self.cxd_report);
            self.cxd_report["cxd_gain"] = json!(0.0);
        }
        if int(&obs["step"]) < 0 {
            return action;
        }
        match self.reorder_apply(obs, action.clone()) {
            Ok(a) => a,
            Err(_) => {
                inc(&mut self.cxd_report, "cxd_errors", 1);
                action
            }
        }
    }

    /// `_cxd_reorder`; caller receives errors, while `reorder` is its agent wrapper.
    pub fn reorder_apply(&mut self, obs: &Value, mut action: Value) -> Result<Value, &'static str> {
        let orders = array(&action["market"]).to_vec();
        if orders.len() < 2 {
            return Ok(action);
        }
        let bought: Vec<Value> = orders
            .iter()
            .filter(|o| array(o).len() > 1 && o[0] == "BUY_PRODUCT")
            .map(|o| o[1].clone())
            .collect();
        let mut slots = vec![];
        let mut sells = vec![];
        let mut fixed = vec![];
        for (i, o) in orders.iter().enumerate() {
            if !truth(o) {
                continue;
            }
            if FIXED.contains(&text(&o[0])) {
                slots.push(i);
                fixed.push(o.clone());
            } else if o[0] == "SELL" && array(o).len() > 1 && !bought.contains(&o[1]) {
                slots.push(i);
                sells.push(o.clone());
            }
        }
        if sells.is_empty() || slots.len() < 2 {
            return Ok(action);
        }
        let stock: BTreeMap<String, i64> = Core::projected_shed(&action, &View::new(obs))
            .as_object()
            .ok_or("cxd: projected shed")?
            .iter()
            .map(|(k, v)| (k.clone(), int(v).max(0)))
            .collect();
        let inv0: BTreeMap<String, i64> = obs["market"]["inventory"]
            .as_object()
            .ok_or("cxd: market inventory")?
            .iter()
            .map(|(k, v)| (k.clone(), int(v)))
            .collect();
        self.cxd_parent_orders = json!(
            orders
                .iter()
                .filter(|o| truth(o))
                .cloned()
                .collect::<Vec<_>>()
        );
        let mut models: Vec<Vec<Value>> = array(&self.cxd_models)
            .iter()
            .filter(|m| truth(m))
            .map(|m| array(m).to_vec())
            .collect();
        if models.is_empty() {
            models.push(orders.clone());
        }
        let params = obs["market"].get("params");
        let mut margins: Vec<_> = models
            .iter()
            .map(|model| Margin::new(model, inv0.clone(), stock.clone(), params))
            .collect();
        let mut margin = |candidate: &[Value]| -> Result<f64, &'static str> {
            let mut result = f64::INFINITY;
            for m in &mut margins {
                result = result.min(m.of(candidate)?);
            }
            Ok(result)
        };
        let base = margin(&orders)?;
        let mut best = base;
        let mut best_orders = None;
        let mut evals = 0;
        let mut positions: Vec<usize> = (0..sells.len()).collect();
        loop {
            let mut candidate = orders.clone();
            let rest: Vec<usize> = (0..slots.len())
                .filter(|i| !positions.contains(i))
                .collect();
            for (&i, o) in positions.iter().zip(&sells) {
                candidate[slots[i]] = o.clone();
            }
            for (&i, o) in rest.iter().zip(&fixed) {
                candidate[slots[i]] = o.clone();
            }
            if candidate != orders {
                evals += 1;
                if evals > CXD_BUDGET {
                    inc(&mut self.cxd_report, "cxd_budget_hits", 1);
                    break;
                }
                let value = margin(&candidate)?;
                if value > best + 0.5 {
                    best = value;
                    best_orders = Some(candidate);
                }
            }
            if !next_positions(&mut positions, slots.len()) {
                break;
            }
        }
        inc(&mut self.cxd_report, "cxd_evals", evals);
        if let Some(chosen) = best_orders {
            inc(&mut self.cxd_report, "cxd_turns", 1);
            self.cxd_report["cxd_gain"] = json!(num(&self.cxd_report["cxd_gain"]) + (best - base));
            action["market"] = json!(chosen);
        }
        Ok(action)
    }

    pub fn remaining(&mut self, route: i64, step: i64, base: &Metav4Controller) -> i64 {
        let k = format!("{route},{step}");
        if let Some(v) = self.seed_cache.get(&k) {
            return int(v);
        }
        let core = &base.base.base.core;
        let n = (step + 1..719)
            .flat_map(|t| units(&core.configured_route_action(if t >= 648 { 2 } else { route }, t)))
            .filter(|c| is_pair(c, "PLANT", "WHEAT") || is_pair(c, "PLANT", "CARROT"))
            .count() as i64;
        self.seed_cache[&k] = json!(n);
        n
    }
    pub fn seeds(&mut self, obs: &Value, mut action: Value, base: &mut Metav4Controller) -> Value {
        let step = int(&obs["step"]);
        let who = seat(obs).to_string();
        if step == 0 {
            self.seed_cache = json!({});
            for v in self.seed_report.as_object_mut().unwrap().values_mut() {
                *v = json!(0);
            }
        }
        if step < 624
            || !array(&action["market"])
                .iter()
                .any(|o| is_order(o, "BUY_SEED", "WHEAT") || is_order(o, "BUY_SEED", "CARROT"))
        {
            return action;
        }
        let native = base.base.base.core.players[&who].clone();
        if !native.is_object() {
            inc(&mut self.seed_report, "errors", 1);
            return action;
        }
        let mut remaining = self.remaining(int(&native["route"]), step, base);
        for q in native["pending"]
            .as_object()
            .into_iter()
            .flat_map(|o| o.values())
        {
            remaining += array(q)
                .iter()
                .filter(|p| is_pair(&p[1], "PLANT", "WHEAT") || is_pair(&p[1], "PLANT", "CARROT"))
                .count() as i64;
        }
        let commands = units(&action);
        let mut available = [0i64; 2];
        for (i, p) in ["WHEAT", "CARROT"].iter().enumerate() {
            available[i] = (int(&obs["private"]["seeds"][p])
                - commands.iter().filter(|c| is_pair(c, "PLANT", p)).count() as i64)
                .max(0);
        }
        let mut out = array(&action["market"]).to_vec();
        let mut changed = false;
        for o in &mut out {
            for (i, p) in ["WHEAT", "CARROT"].iter().enumerate() {
                if is_order(o, "BUY_SEED", p) {
                    let qty = int(&o[2]).max(0);
                    let keep = qty.min((remaining - available[i]).max(0));
                    available[i] += keep;
                    if keep < qty {
                        let cut = qty - keep;
                        changed = true;
                        inc(&mut self.seed_report, "cut_units", cut);
                        inc(
                            &mut self.seed_report,
                            "saved_cost",
                            cut * if i == 0 { 10 } else { 20 },
                        );
                        if i == 1 {
                            let st = &mut base.base.carrot2.states[&who];
                            if !st.is_null() {
                                st["spare_carrot"] = json!((int(&st["spare_carrot"]) - cut).max(0));
                            }
                        }
                        *o = if keep > 0 {
                            json!(["BUY_SEED", p, keep])
                        } else {
                            json!([])
                        };
                    }
                    break;
                }
            }
        }
        if changed {
            inc(&mut self.seed_report, "changed_turns", 1);
            action["market"] = json!(out);
        }
        action
    }
    pub fn fertilizer(
        &mut self,
        obs: &Value,
        mut action: Value,
        base: &mut Metav4Controller,
    ) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        let who = seat(obs).to_string();
        if step == 0 {
            for v in self.fert_report.as_object_mut().unwrap().values_mut() {
                *v = json!(0);
            }
        }
        let mut commands = units(&action);
        if !commands.iter().any(|c| *c == json!(["FERTILIZE"])) {
            return action;
        }
        let result = (|| -> Result<bool, &'static str> {
            let (mut farm, mut private) = terminal::clone_state(own_farm(obs), &obs["private"]);
            let positions = unit_positions(&farm);
            let core = &base.base.base.core;
            let native = &core.players[&who];
            if !native.is_object() {
                return Err("missing native");
            }
            let expected = (day * 24..((day + 1) * 24).min(719))
                .map(|t| {
                    array(&core.configured_route_action(int(&native["route"]), t)["hands"]).len()
                })
                .max()
                .ok_or("empty native day")?;
            let reactive = &base.base.base.late.input[&who]["workers"];
            let size = array(&farm["tiles"]).len() as i64;
            let mut changed = false;
            for i in 0..commands.len().min(positions.len()) {
                let pos = positions[i];
                let tile = tile(&farm, pos);
                if commands[i] == json!(["FERTILIZE"])
                    && (tile["crop"] == "WHEAT" || tile["crop"] == "CARROT")
                    && int(&private["inventories"][i]["FERTILIZER"]) > 0
                {
                    let until = tile.get("fertilized_until_day").map(int).unwrap_or(-1);
                    let covered = until >= day + 2;
                    let mut skip = covered;
                    if !skip && i <= expected && reactive.get(i.to_string()).is_none() {
                        let planted = int(&tile["planted_day"]);
                        let visits = carrot2::visits(
                            obs,
                            &action,
                            core,
                            pos,
                            718.min((planted + 6) * 24),
                            Some(step + 1),
                        );
                        let growth = |fert_until| Growth {
                            y0: int(&tile["yield_units"]),
                            fert_until,
                            watered_day: if truth(&tile["watered_today"]) {
                                day
                            } else {
                                -1
                            },
                            now_step: step,
                        };
                        let crop = tile["crop"].as_str().unwrap();
                        let old = carrot2::yield_path(crop, planted, &visits, growth(until)).0;
                        let new =
                            carrot2::yield_path(crop, planted, &visits, growth(until.max(day + 2)))
                                .0;
                        skip = old > 0 && old == new;
                    }
                    if skip {
                        commands[i] = json!(["PASS"]);
                        changed = true;
                        inc(&mut self.fert_report, "skips", 1);
                        inc(
                            &mut self.fert_report,
                            if covered { "covered" } else { "capped" },
                            1,
                        );
                    }
                }
                terminal::apply_unit_action(
                    &mut farm,
                    &mut private,
                    i,
                    &commands[i],
                    size,
                    day,
                    24,
                    100,
                )?;
            }
            Ok(changed)
        })();
        match result {
            Ok(true) => {
                action["farmer"] = commands[0].clone();
                action["hands"] = json!(commands[1..].to_vec());
            }
            Err(_) => inc(&mut self.fert_report, "errors", 1),
            _ => {}
        }
        action
    }
    pub fn merge(&mut self, obs: &Value, action: Value) -> Value {
        if int(&obs["step"]) == 0 {
            zero(&mut self.mg_report);
        }
        match self.merge_apply(action.clone()) {
            Ok(a) => a,
            Err(_) => {
                inc(&mut self.mg_report, "mg_errors", 1);
                action
            }
        }
    }

    pub fn merge_apply(&mut self, mut action: Value) -> Result<Value, &'static str> {
        if !action.is_object() {
            return Ok(action);
        }
        let market = array(&action["market"]);
        if market.len() < 2 {
            return Ok(action);
        }
        let mut seen: Vec<((Value, Value), usize)> = vec![];
        let mut out: Vec<Value> = vec![];
        let mut changed = false;
        for order in market {
            if array(order).len() >= 3
                && ["SELL", "BUY_PRODUCT", "BUY_SEED"].contains(&text(&order[0]))
            {
                let key = (order[0].clone(), order[1].clone());
                if let Some((_, index)) = seen.iter().find(|(k, _)| *k == key) {
                    out[*index][2] = json!(quantity(&out[*index][2])? + quantity(&order[2])?);
                    changed = true;
                    continue;
                }
                seen.push((key, out.len()));
            }
            out.push(order.clone());
        }
        if changed {
            inc(&mut self.mg_report, "mg_turns", 1);
            inc(
                &mut self.mg_report,
                "mg_merged",
                (market.len() - out.len()) as i64,
            );
            action["market"] = json!(out);
        }
        Ok(action)
    }

    pub fn queue(&mut self, obs: &Value, action: Value) -> Value {
        if int(&obs["step"]) == 0 {
            zero(&mut self.ig_report);
        }
        match self.queue_apply(obs, action.clone()) {
            Ok(a) => a,
            Err(_) => {
                inc(&mut self.ig_report, "errors", 1);
                action
            }
        }
    }

    pub fn queue_apply(&mut self, obs: &Value, mut action: Value) -> Result<Value, &'static str> {
        if !action.is_object() {
            return Ok(action);
        }
        let market = array(&action["market"]);
        if market.len() < 2 {
            return Ok(action);
        }
        let projected = Core::projected_shed(&action, &View::new(obs));
        let mut remaining: HashMap<&str, i64> = CASH
            .iter()
            .map(|item| (*item, int(&projected[*item]).max(0)))
            .collect();
        let mut revised = vec![];
        let mut zeroed = 0;
        for order in market {
            if array(order).len() >= 3 && order[0] == "SELL" && CASH.contains(&text(&order[1])) {
                let held = remaining.get_mut(text(&order[1])).unwrap();
                let executed = quantity(&order[2])?.max(0).min(*held);
                *held -= executed;
                if executed <= 0 {
                    revised.push(json!([]));
                    zeroed += 1;
                } else {
                    revised.push(order.clone());
                }
            } else {
                revised.push(order.clone());
            }
        }
        let mut holes = VecDeque::new();
        let mut pulled = 0;
        let mut distance = 0;
        for index in 0..revised.len() {
            let order = &revised[index];
            if !truth(order) {
                holes.push_back(index);
                continue;
            }
            let movable = array(order).len() >= 3
                && order[0] == "SELL"
                && CASH.contains(&text(&order[1]))
                && quantity(&order[2])? > 0;
            if !movable || holes.is_empty() {
                continue;
            }
            let target = holes.pop_front().unwrap();
            revised[target] = order.clone();
            revised[index] = json!([]);
            holes.push_back(index);
            pulled += 1;
            distance += (index - target) as i64;
        }
        if revised != market {
            inc(&mut self.ig_report, "queue_changed_turns", 1);
            inc(&mut self.ig_report, "zeroed_orders", zeroed);
            inc(&mut self.ig_report, "pulled_orders", pulled);
            inc(&mut self.ig_report, "pulled_slots", distance);
            action["market"] = json!(revised);
        }
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::next_positions;

    #[test]
    fn partial_permutations_keep_itertools_order() {
        let mut positions = vec![0, 1];
        let mut actual = vec![positions.clone()];
        while next_positions(&mut positions, 3) {
            actual.push(positions.clone());
        }
        assert_eq!(
            actual,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 2],
                vec![2, 0],
                vec![2, 1]
            ]
        );
    }

    #[test]
    fn full_permutations_keep_itertools_order() {
        let mut positions = vec![0, 1, 2];
        let mut actual = vec![positions.clone()];
        while next_positions(&mut positions, 3) {
            actual.push(positions.clone());
        }
        assert_eq!(
            actual,
            vec![
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0]
            ]
        );
    }
}
