//! Exact active V43 opening, service, funding and overflow contracts.
//! Source: agents/v43/main.py (_R124, _R127, _R128, _R148).
//! The source's _R148_SEEDS flag is false; those disabled repairs are not applied.
use super::common::*;
use super::core::Core;
use super::late::{budget, cfg_ok, delivery, market_stock, supply};
use super::production::market_price;
use super::terminal::{apply_unit_action, clone_state};
use serde_json::{Value, json};
use std::collections::BTreeSet;

type Result<T> = std::result::Result<T, &'static str>;
const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
const REPORT124: &[&str] = &[
    "opening_seed_budget_units",
    "opening_seed_budget_cost",
    "opening_atomic_dropped",
    "opening_atomic_rescued_requests",
    "opening_atomic_rescued_confirmed",
    "opening_atomic_plant_errors",
    "opening_day1_cash",
    "opening_day1_hires_requested",
    "opening_day1_hires_confirmed",
    "opening_day1_hire_shortfalls",
    "opening_contract_errors",
];
const REPORT127: &[&str] = &[
    "last_hour_plants_dropped",
    "priority_grain_orders",
    "priority_grain_units",
    "priority_grain_confirmed",
    "priority_grain_shortfalls",
    "priority_contract_errors",
];
const REPORT128: &[&str] = &[
    "service_requested_units",
    "service_confirmed_units",
    "service_idle_preloads",
    "service_swaps_started",
    "service_swaps_completed",
    "service_capacity_declines",
    "service_arrival_errors",
    "service_swap_errors",
    "sale_credit_orders",
    "sale_credit_lower_bound",
    "sale_credit_grain_units",
    "sale_credit_confirmed_units",
    "sale_credit_errors",
    "service_errors",
];
const REPORT148: &[&str] = &[
    "overflow_turns",
    "overflow_units_reclaimed",
    "overflow_quote_exposure",
    "overflow_contract_checks",
    "overflow_contract_errors",
    "atomic_turns",
    "atomic_kept_requests",
    "atomic_removed_requests",
    "seed_prefund_turns",
    "seed_prefund_units",
    "targeted_errors",
];

fn farm(obs: &Value) -> &Value {
    &obs["farms"][int(&obs["player"]) as usize]
}
fn is(c: &Value, op: &str, item: &str) -> bool {
    c[0] == op && c[1] == item
}
fn amount(c: &Value) -> i64 {
    quantity(c, 1).max(0)
}
fn pop(v: &mut Value, key: &str) -> Value {
    v.as_object_mut()
        .and_then(|m| m.shift_remove(key))
        .unwrap_or(Value::Null)
}
fn positions(f: &Value) -> Vec<Value> {
    let mut result = vec![f["farmer"].clone()];
    result.extend(array(&f["hands"]).iter().cloned());
    result
}
fn actor_position(f: &Value, actor: usize) -> Value {
    if actor == 0 {
        f["farmer"].clone()
    } else {
        f["hands"][actor - 1].clone()
    }
}
fn hire_positions(f: &Value, os: &[Value], night: bool) -> Vec<(i64, i64)> {
    if night {
        return vec![(4, 4)];
    }
    let mut ps: Vec<_> = positions(f).iter().map(position).collect();
    for order in os {
        if order[0] == "HIRE" {
            let p = *ACCESS
                .iter()
                .min_by_key(|p| ps.iter().filter(|q| q == p).count())
                .unwrap();
            ps.push(p);
        }
    }
    ps
}
fn future(obs: &Value, core: &Core, offset: i64) -> Value {
    let step = int(&obs["step"]) + offset;
    let route = if step >= 648 {
        2
    } else {
        int(&core.players[int(&obs["player"]).to_string()]["route"])
    };
    Core::route_action(route, step)
}
fn shifted_cash(obs: &Value, delta: i64) -> Value {
    let mut result = obs.clone();
    let seat = int(&obs["player"]) as usize;
    result["farms"][seat]["money"] = json!(num(&obs["farms"][seat]["money"]) + delta as f64);
    result
}
pub fn same_stock(a: &Value, b: &Value) -> bool {
    a.as_object()
        .into_iter()
        .flat_map(|m| m.keys())
        .chain(b.as_object().into_iter().flat_map(|m| m.keys()))
        .all(|p| int(&a[p]) == int(&b[p]))
}

/// Scratch own unit effects, including whole-block excess-PLANT cancellation.
pub fn fields(obs: &Value, action: &Value) -> Result<(Value, Value)> {
    let (mut f, mut private) = clone_state(farm(obs), &obs["private"]);
    let cs = commands(action);
    let mut demand = json!({});
    for c in &cs {
        if array(c).len() > 1 && c[0] == "PLANT" {
            increment(&mut demand, text(&c[1]), 1);
        }
    }
    let blocked: BTreeSet<_> = demand
        .as_object()
        .unwrap()
        .iter()
        .filter(|(p, n)| int(n) > int(&private["seeds"][*p]))
        .map(|(p, _)| p.clone())
        .collect();
    for (actor, c) in cs
        .iter()
        .take(array(&private["inventories"]).len())
        .enumerate()
    {
        if array(c).len() > 1 && c[0] == "PLANT" && blocked.contains(text(&c[1])) {
            continue;
        }
        apply_unit_action(
            &mut f,
            &mut private,
            actor,
            c,
            10,
            int(&obs["step"]) / 24,
            24,
            100,
        )?;
    }
    Ok((f, private))
}

pub fn prefix_bound(obs: &Value, quantity: i64) -> i64 {
    let inventory = int(&obs["market"]["inventory"]["WHEAT"]);
    (1..=quantity)
        .map(|j| market_price("WHEAT", (inventory - (2 * j - 1)) as f64, None))
        .sum()
}
pub fn sale_credit(obs: &Value, action: &Value) -> Result<i64> {
    let os = orders(action);
    let Some(first) = os.first() else {
        return Ok(0);
    };
    if array(first).len() < 3 || first[0] != "SELL" || first[1] == "WHEAT" {
        return Ok(0);
    }
    let item = text(&first[1]);
    if obs["market"]["inventory"].get(item).is_none() {
        return Ok(0);
    }
    let (_, private) = fields(obs, action)?;
    let q = amount(first).min(int(&private["shed"][item]).max(0));
    let inventory = int(&obs["market"]["inventory"][item]);
    Ok((0..q)
        .map(|j| market_price(item, (inventory + 2 * j) as f64, None))
        .sum())
}

#[derive(Clone, Debug)]
pub struct Contracts {
    pub states124: Value,
    pub states127: Value,
    pub states128: Value,
    pub pending148: Value,
    pub report: Value,
    pub stages: Value,
}
impl Default for Contracts {
    fn default() -> Self {
        let mut result = Self {
            states124: json!({}),
            states127: json!({}),
            states128: json!({}),
            pending148: json!({}),
            report: json!({}),
            stages: json!({}),
        };
        // R148 initializes only at step zero. A fresh mid-season callback that
        // reaches its first successful proposal instead raises at the source's
        // missing report counter, and the wrapper retains its parent action.
        for names in [REPORT124, REPORT127, REPORT128] {
            result.reset_report(names);
        }
        result
    }
}
impl Contracts {
    fn reset_report(&mut self, names: &[&str]) {
        for name in names {
            self.report[*name] = json!(0);
        }
    }
    fn add(&mut self, key: &str, n: i64) {
        increment(&mut self.report, key, n);
    }
    fn add_required(&mut self, key: &str, n: i64) -> Result<()> {
        if self.report.get(key).is_none() {
            return Err("uninitialized contract counter");
        }
        self.add(key, n);
        Ok(())
    }

    pub fn seed_budget(&mut self, obs: &Value, mut action: Value, reserve: i64) -> Result<Value> {
        let mut os = orders(&action);
        if !os.iter().any(|o| o[0] == "BUY_SEED") {
            return Ok(action);
        }
        let bounded = shifted_cash(obs, -reserve);
        if budget(&bounded, &os) {
            return Ok(action);
        }
        for i in (0..os.len()).rev() {
            if array(&os[i]).len() < 3 || os[i][0] != "BUY_SEED" {
                continue;
            }
            let before = amount(&os[i]);
            os[i][2] = json!(before);
            while int(&os[i][2]) > 0 && !budget(&bounded, &os) {
                os[i][2] = json!(int(&os[i][2]) - 1);
            }
            let removed = before - int(&os[i][2]);
            if removed > 0 {
                self.add("opening_seed_budget_units", removed);
                let price = match text(&os[i][1]) {
                    "WHEAT" => 10,
                    "CARROT" => 20,
                    "TOMATO" => 50,
                    "STRAWBERRY" => 100,
                    "MELON" => 80,
                    _ => return Err("unknown seed"),
                };
                self.add("opening_seed_budget_cost", removed * price);
            }
            if budget(&bounded, &os) {
                break;
            }
        }
        set_orders(&mut action, os);
        Ok(action)
    }

    pub fn atomic(&mut self, obs: &Value, mut action: Value, state: &mut Value) -> Result<Value> {
        let mut cs = commands(&action);
        let mut demand = json!({});
        for c in &cs {
            if array(c).len() > 1 && c[0] == "PLANT" {
                increment(&mut demand, text(&c[1]), 1);
            }
        }
        if !truth(&demand) {
            return Ok(action);
        }
        let blocked: BTreeSet<_> = demand
            .as_object()
            .unwrap()
            .iter()
            .filter(|(p, n)| int(n) > int(&obs["private"]["seeds"][*p]))
            .map(|(p, _)| p.clone())
            .collect();
        let (mut f, mut private) = clone_state(farm(obs), &obs["private"]);
        let mut kept = Vec::new();
        let mut changed = false;
        for (actor, c) in cs.iter_mut().enumerate() {
            if array(c).len() > 1 && c[0] == "PLANT" {
                let pos = if actor >= array(&private["inventories"]).len() {
                    Value::Null
                } else {
                    actor_position(&f, actor)
                };
                let valid = !pos.is_null()
                    && tile_at(&f["tiles"], &pos).is_null()
                    && int(&private["seeds"][text(&c[1])]) > 0;
                if !valid {
                    *c = json!(["PASS"]);
                    changed = true;
                    self.add("opening_atomic_dropped", 1);
                } else if blocked.contains(text(&c[1])) {
                    kept.push(json!({"xy":pos,"crop":c[1],"birth":0}));
                    self.add("opening_atomic_rescued_requests", 1);
                }
            }
            if actor < array(&private["inventories"]).len() {
                apply_unit_action(&mut f, &mut private, actor, c, 10, 0, 24, 100)?;
            }
        }
        if !kept.is_empty() {
            state["pending_plants"] = json!(kept);
        }
        if changed {
            set_commands(&mut action, cs);
        }
        Ok(action)
    }

    pub fn last_hour(&mut self, obs: &Value, action: Value) -> Result<Value> {
        let step = int(&obs["step"]);
        if step % 24 != 23 {
            return Ok(action);
        }
        let mut cs = commands(&action);
        if !cs.iter().any(|c| c[0] == "PLANT") {
            return Ok(action);
        }
        let mut result = action.clone();
        let mut changed = false;
        for _ in 0..=cs.len() {
            let (f, _) = fields(obs, &result)?;
            let ps = positions(farm(obs));
            let mut dropped = Vec::new();
            for (actor, c) in cs.iter().enumerate() {
                if c[0] != "PLANT" {
                    continue;
                }
                let t = ps
                    .get(actor)
                    .map(|pos| tile_at(&f["tiles"], pos))
                    .unwrap_or(&Value::Null);
                if !(t.is_object()
                    && t["kind"] == "PLANT"
                    && t["crop"] == c[1]
                    && int(&t["planted_day"]) == step / 24
                    && truth(&t["watered_today"]))
                {
                    dropped.push(actor);
                }
            }
            if dropped.is_empty() {
                break;
            }
            for actor in &dropped {
                cs[*actor] = json!(["PASS"]);
            }
            set_commands(&mut result, cs.clone());
            changed = true;
            self.add("last_hour_plants_dropped", dropped.len() as i64);
        }
        Ok(if changed { result } else { action })
    }

    pub fn priority(
        &mut self,
        obs: &Value,
        mut action: Value,
        core: &Core,
        state: &mut Value,
    ) -> Result<Value> {
        let step = int(&obs["step"]);
        if !(144..695).contains(&step) {
            return Ok(action);
        }
        let os = orders(&action);
        if os.len() > 9
            || os
                .iter()
                .any(|o| is(o, "BUY_PRODUCT", "WHEAT") || is(o, "SELL", "WHEAT"))
        {
            return Ok(action);
        }
        let cs = commands(&future(obs, core, 1));
        if !cs.iter().any(|c| is(c, "PICKUP", "WHEAT")) || !budget(obs, &os) {
            return Ok(action);
        }
        let (f, private) = fields(obs, &action)?;
        let night = step % 24 == 23;
        let ps = hire_positions(&f, &os, night);
        let need: i64 = ps
            .iter()
            .zip(cs.iter())
            .filter(|(p, c)| ACCESS.contains(p) && is(c, "PICKUP", "WHEAT"))
            .map(|(_, c)| amount(c))
            .sum();
        let (stock, buys, _) = market_stock(&private["shed"], &os);
        let (before, loss) = delivery(stock, &private, night);
        let shortage = (need - int(&before["WHEAT"])).max(0);
        if shortage == 0 || shortage > 100 - stock_total(&private["shed"]) {
            return Ok(action);
        }
        if !budget(&shifted_cash(obs, -prefix_bound(obs, shortage)), &os) {
            return Ok(action);
        }
        let mut proposed = vec![json!(["BUY_PRODUCT", "WHEAT", shortage])];
        proposed.extend(os);
        let (stock, after_buys, _) = market_stock(&private["shed"], &proposed);
        let (after, after_loss) = delivery(stock, &private, night);
        if int(&after["WHEAT"]) < need
            || buys
                .iter()
                .any(|(i, q)| after_buys.get(&(i + 1)).copied().unwrap_or(0) < *q)
            || after_loss
                .as_object()
                .unwrap()
                .iter()
                .any(|(p, q)| int(q) > int(&loss[p]))
        {
            return Ok(action);
        }
        state["pending_grain"] = json!([step + 1, int(&after["WHEAT"]), shortage]);
        self.add("priority_grain_orders", 1);
        self.add("priority_grain_units", shortage);
        set_orders(&mut action, proposed);
        Ok(action)
    }

    fn next_need(
        &self,
        obs: &Value,
        f: &Value,
        os: &[Value],
        advanced: &Value,
        core: &Core,
    ) -> i64 {
        let ps = hire_positions(f, os, int(&obs["step"]) % 24 == 23);
        ps.iter()
            .zip(commands(&future(obs, core, 1)))
            .enumerate()
            .filter(|(actor, (p, c))| {
                advanced.get(actor.to_string()).is_none()
                    && ACCESS.contains(p)
                    && is(c, "PICKUP", "WHEAT")
            })
            .map(|(_, (_, c))| amount(&c))
            .sum()
    }

    pub fn field_safe(
        &self,
        obs: &Value,
        before: &Value,
        after: &Value,
        advanced: &Value,
        core: &Core,
    ) -> Result<Option<Value>> {
        let os = orders(before);
        if !budget(obs, &os) {
            return Ok(None);
        }
        let (_, oldprivate) = fields(obs, before)?;
        let (f, private) = fields(obs, after)?;
        let oldcs = commands(before);
        let cs = commands(after);
        for (i, c) in oldcs
            .iter()
            .take(array(&private["inventories"]).len())
            .enumerate()
        {
            if c[0] == "PICKUP" && Some(c) == cs.get(i) {
                let item = text(&c[1]);
                if int(&private["inventories"][i][item]) < int(&oldprivate["inventories"][i][item])
                {
                    return Ok(None);
                }
            }
        }
        let (oldstock, oldbuys, oldsales) = market_stock(&oldprivate["shed"], &os);
        let (stock, buys, sales) = market_stock(&private["shed"], &os);
        let night = int(&obs["step"]) % 24 == 23;
        let (oldfinal, oldloss) = delivery(oldstock, &oldprivate, night);
        let (final_stock, loss) = delivery(stock, &private, night);
        if oldbuys
            .iter()
            .any(|(i, q)| buys.get(i).copied().unwrap_or(0) < *q)
            || oldsales
                .iter()
                .any(|(i, q)| sales.get(i).copied().unwrap_or(0) < *q)
            || loss
                .as_object()
                .unwrap()
                .iter()
                .any(|(p, q)| int(q) > int(&oldloss[p]))
        {
            return Ok(None);
        }
        let mut need = self.next_need(obs, &f, &os, advanced, core);
        let pending = &self.states127[int(&obs["player"]).to_string()]["pending_grain"];
        if truth(pending) && int(&pending[0]) == int(&obs["step"]) + 1 {
            need = need.max(int(&pending[1]));
        }
        if int(&final_stock["WHEAT"]) < need.min(int(&oldfinal["WHEAT"])) {
            return Ok(None);
        }
        Ok(Some(private))
    }

    pub fn food_need(
        &self,
        obs: &Value,
        f: &Value,
        private: &Value,
        actor: usize,
        core: &Core,
    ) -> Result<i64> {
        let (mut f, mut private) = (f.clone(), private.clone());
        let step = int(&obs["step"]);
        let mut missing = 0;
        for offset in 1..5.min(24 - step % 24) {
            let cs = commands(&future(obs, core, offset));
            let c = cs.get(actor).cloned().unwrap_or_else(|| json!(["PASS"]));
            if is(&c, "PICKUP", "WHEAT") {
                break;
            }
            if c == json!(["FEED"]) {
                let pos = actor_position(&f, actor);
                let t = tile_at(&f["tiles"], &pos);
                if t.is_object()
                    && truth(&t["animal"])
                    && !truth(&t["fed_today"])
                    && int(&private["inventories"][actor]["WHEAT"]) <= 0
                {
                    private["inventories"][actor]["WHEAT"] = json!(1);
                    missing += 1;
                }
            }
            apply_unit_action(&mut f, &mut private, actor, &c, 10, step / 24, 24, 100)?;
        }
        Ok(missing)
    }

    pub fn service(
        &mut self,
        obs: &Value,
        action: Value,
        core: &Core,
        state: &mut Value,
    ) -> Result<Value> {
        let step = int(&obs["step"]);
        let f = farm(obs);
        let private = &obs["private"];
        let ps = positions(f);
        for arrival in array(&pop(state, "arrivals")) {
            let actor = int(&arrival["actor"]) as usize;
            if int(&arrival["step"]) != step
                || actor >= array(&private["inventories"]).len()
                || int(&private["inventories"][actor]["WHEAT"]) < int(&arrival["expected"])
            {
                self.add("service_arrival_errors", 1);
            } else {
                self.add("service_confirmed_units", int(&arrival["quantity"]));
            }
        }
        let old_pending = pop(state, "swaps");
        let mut result = action;
        let mut cs = commands(&result);
        if let Some(entries) = old_pending.as_object() {
            for (actor, pending) in entries {
                let actor: usize = actor.parse().map_err(|_| "swap actor")?;
                if step != int(&pending["step"])
                    || actor >= ps.len()
                    || ps[actor] != pending["xy"]
                    || cs.get(actor) != Some(&pending["pickup"])
                {
                    self.add("service_swap_errors", 1);
                    continue;
                }
                let t = tile_at(&f["tiles"], &ps[actor]);
                if !t.is_object()
                    || t["animal"] != pending["animal"]
                    || t["placed_day"] != pending["birth"]
                {
                    self.add("service_swap_errors", 1);
                    continue;
                }
                if int(&private["inventories"][actor]["WHEAT"]) < int(&pending["quantity"]) + 1 {
                    self.add("service_swap_errors", 1);
                    continue;
                }
                cs[actor] = json!([if truth(&t["fed_today"]) {
                    "PASS"
                } else {
                    "FEED"
                }]);
                self.add("service_swaps_completed", 1);
            }
        }
        if truth(&old_pending) {
            let mut proposed = result.clone();
            set_commands(&mut proposed, cs);
            if self
                .field_safe(obs, &result, &proposed, &json!({}), core)?
                .is_some()
            {
                result = proposed;
            } else {
                self.add("service_swap_errors", 1);
            }
        }
        if !(144..647).contains(&step) {
            return Ok(result);
        }
        let cs = commands(&result);
        let future_cs = commands(&future(obs, core, 1));
        let mut proposed = result.clone();
        let mut proposed_cs = commands(&proposed);
        let mut swaps = json!({});
        let mut arrivals = Vec::new();
        let mut prefetches = 0;
        let (projected_farm, projected_private) = fields(obs, &result)?;
        for (actor, c) in cs
            .iter()
            .take(array(&private["inventories"]).len())
            .enumerate()
        {
            let Some(pos) = ps.get(actor) else {
                return Err("service actor position");
            };
            if !ACCESS.contains(&position(pos)) {
                continue;
            }
            let held = int(&private["inventories"][actor]["WHEAT"]).max(0);
            let t = tile_at(&f["tiles"], pos);
            let nxt = future_cs
                .get(actor)
                .cloned()
                .unwrap_or_else(|| json!(["PASS"]));
            if c == &json!(["FEED"])
                && held == 0
                && step % 24 <= 21
                && t.is_object()
                && truth(&t["animal"])
                && !truth(&t["fed_today"])
                && is(&nxt, "PICKUP", "WHEAT")
            {
                let q = amount(&nxt);
                if q == 0 {
                    continue;
                }
                proposed_cs[actor] = json!(["PICKUP", "WHEAT", q + 1]);
                swaps[actor.to_string()] = json!({"step":step+1,"xy":pos,"animal":t["animal"],"birth":t["placed_day"],"quantity":q,"pickup":nxt});
                arrivals.push(json!({"step":step+1,"actor":actor,"expected":q+1,"quantity":q+1}));
            } else if c == &json!(["PASS"]) {
                let q = self.food_need(obs, &projected_farm, &projected_private, actor, core)?;
                if q > 0 {
                    proposed_cs[actor] = json!(["PICKUP", "WHEAT", q]);
                    prefetches += 1;
                    arrivals
                        .push(json!({"step":step+1,"actor":actor,"expected":held+q,"quantity":q}));
                }
            }
        }
        if arrivals.is_empty() {
            return Ok(result);
        }
        set_commands(&mut proposed, proposed_cs);
        let after = self.field_safe(obs, &result, &proposed, &swaps, core)?;
        if after.as_ref().is_none_or(|p| {
            arrivals.iter().any(|a| {
                int(&p["inventories"][int(&a["actor"]) as usize]["WHEAT"]) < int(&a["expected"])
            })
        }) {
            self.add("service_capacity_declines", 1);
            return Ok(result);
        }
        self.add(
            "service_swaps_started",
            swaps.as_object().unwrap().len() as i64,
        );
        self.add("service_idle_preloads", prefetches);
        self.add(
            "service_requested_units",
            arrivals.iter().map(|a| int(&a["quantity"])).sum(),
        );
        state["swaps"] = swaps;
        state["arrivals"] = json!(arrivals);
        Ok(proposed)
    }

    pub fn credit_supply(
        &mut self,
        obs: &Value,
        action: Value,
        core: &Core,
        state: &mut Value,
    ) -> Result<Value> {
        let step = int(&obs["step"]);
        if !(144..695).contains(&step) || budget(obs, &orders(&action)) {
            return Ok(action);
        }
        let credit = sale_credit(obs, &action)?;
        if credit == 0 {
            return Ok(action);
        }
        let bounded = shifted_cash(obs, credit);
        if !budget(&bounded, &orders(&action)) {
            return Ok(action);
        }
        let result = supply(&bounded, action.clone(), core);
        if result == action {
            return Ok(action);
        }
        if result["market"][0] != action["market"][0] {
            return Err("changed funding sale");
        }
        let (_, private) = fields(obs, &result)?;
        let (stock, _, _) = market_stock(&private["shed"], &orders(&result));
        let (final_stock, _) = delivery(stock, &private, step % 24 == 23);
        let qty = |a: &Value| -> i64 {
            orders(a)
                .iter()
                .filter(|o| array(o).len() > 2 && is(o, "BUY_PRODUCT", "WHEAT"))
                .map(amount)
                .sum()
        };
        let extra = (qty(&result) - qty(&action)).max(0);
        state["credit_pending"] = json!([step + 1, int(&final_stock["WHEAT"]), extra]);
        self.add("sale_credit_orders", 1);
        self.add("sale_credit_lower_bound", credit);
        self.add("sale_credit_grain_units", extra);
        Ok(result)
    }

    pub fn overflow(&mut self, obs: &Value, mut action: Value) -> Result<Value> {
        let step = int(&obs["step"]);
        if step % 24 != 23 {
            return Ok(action);
        }
        let os = orders(&action);
        if os.len() >= 10 || !budget(obs, &os) {
            return Ok(action);
        }
        let (_, private) = fields(obs, &action)?;
        let (stock, _, _) = market_stock(&private["shed"], &os);
        let (original, loss) = delivery(stock.clone(), &private, true);
        if !truth(&loss) {
            return Ok(action);
        }
        let mut remaining = (100 - stock_total(&stock)).max(0);
        let mut tail = Vec::new();
        for bag in array(&private["inventories"]) {
            for (item, n) in bag.as_object().ok_or("overflow bag")? {
                let n = int(n).max(0);
                let take = n.min(remaining);
                remaining -= take;
                for _ in 0..n - take {
                    tail.push(item.clone());
                }
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
            if os.len() + released.as_object().unwrap().len() > 10 {
                break;
            }
            let mut proposed = os.clone();
            proposed.extend(
                released
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(p, n)| json!(["SELL", p, n])),
            );
            let (after, _, _) = market_stock(&private["shed"], &proposed);
            let (final_stock, _) = delivery(after, &private, true);
            if same_stock(&original, &final_stock) {
                best = Some((proposed, released.clone(), final_stock));
            }
        }
        let Some((proposed, released, final_stock)) = best else {
            return Ok(action);
        };
        self.add_required("overflow_turns", 1)?;
        self.add_required("overflow_units_reclaimed", stock_total(&released))?;
        let exposure = released
            .as_object()
            .unwrap()
            .iter()
            .map(|(p, n)| int(n) * int(&obs["market"]["prices"][p]))
            .sum();
        self.add_required("overflow_quote_exposure", exposure)?;
        self.pending148[int(&obs["player"]).to_string()] = json!([step + 1, final_stock]);
        set_orders(&mut action, proposed);
        Ok(action)
    }

    fn layer124(
        &mut self,
        obs: &Value,
        action: &mut Value,
        core: &Core,
        standard: bool,
        state: &mut Value,
    ) -> Result<()> {
        let step = int(&obs["step"]);
        let f = farm(obs);
        for p in array(&pop(state, "pending_plants")) {
            let t = tile_at(&f["tiles"], &p["xy"]);
            let key = if t.is_object() && t["crop"] == p["crop"] && t["planted_day"] == p["birth"] {
                "opening_atomic_rescued_confirmed"
            } else {
                "opening_atomic_plant_errors"
            };
            self.add(key, 1);
        }
        if standard && (0..24).contains(&step) {
            let route = int(&core.players[int(&obs["player"]).to_string()]["route"]);
            let hires = orders(&Core::route_action(route, 24))
                .iter()
                .filter(|o| o[0] == "HIRE")
                .count() as i64;
            *action = self.seed_budget(obs, action.clone(), (0..hires).map(fib).sum())?;
            *action = self.atomic(obs, action.clone(), state)?;
        }
        if step == 24 {
            self.report["opening_day1_cash"] = f["money"].clone();
            state["hires"] = json!(orders(action).iter().filter(|o| o[0] == "HIRE").count());
            self.report["opening_day1_hires_requested"] = state["hires"].clone();
        }
        if step == 25 {
            let actual = array(&f["hands"]).len() as i64;
            self.report["opening_day1_hires_confirmed"] = json!(actual);
            self.report["opening_day1_hire_shortfalls"] =
                json!((int(&state["hires"]) - actual).max(0));
        }
        Ok(())
    }

    /// Called after the inherited R97 supply layer; each wrapper retains its
    /// own reset/pending semantics and falls back to its received action.
    pub fn after(&mut self, obs: &Value, mut action: Value, core: &Core, config: &Value) -> Value {
        let player = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        let standard = cfg_ok(config, true);
        let mut state = self.states124[&player].clone();
        if !state.is_object() || step <= int(&state["step"]) {
            state = json!({"step":-1,"pending_plants":[]});
            self.reset_report(REPORT124);
        }
        state["step"] = json!(step);
        if self
            .layer124(obs, &mut action, core, standard, &mut state)
            .is_err()
        {
            self.add("opening_contract_errors", 1);
        }
        self.states124[&player] = state;
        self.stages["r124"] = action.clone();

        let mut state = self.states127[&player].clone();
        if !state.is_object() || step <= int(&state["step"]) {
            state = json!({"step":-1});
            self.reset_report(REPORT127);
        }
        let pending = pop(&mut state, "pending_grain");
        if truth(&pending) && step == int(&pending[0]) {
            if int(&obs["private"]["shed"]["WHEAT"]) >= int(&pending[1]) {
                self.add("priority_grain_confirmed", int(&pending[2]));
            } else {
                self.add("priority_grain_shortfalls", 1);
            }
        }
        state["step"] = json!(step);
        if standard {
            let result = self
                .last_hour(obs, action.clone())
                .and_then(|a| self.priority(obs, a, core, &mut state));
            match result {
                Ok(result) => action = result,
                Err(_) => self.add("priority_contract_errors", 1),
            }
        }
        self.states127[&player] = state;
        self.stages["r127"] = action.clone();

        let mut state = self.states128[&player].clone();
        if !state.is_object() || step <= int(&state["step"]) {
            state = json!({"step":-1});
            self.reset_report(REPORT128);
        }
        state["step"] = json!(step);
        let pending = pop(&mut state, "credit_pending");
        if truth(&pending) {
            if int(&pending[0]) != step || int(&obs["private"]["shed"]["WHEAT"]) < int(&pending[1])
            {
                self.add("sale_credit_errors", 1);
            } else {
                self.add("sale_credit_confirmed_units", int(&pending[2]));
            }
        }
        if standard {
            let result = self
                .service(obs, action.clone(), core, &mut state)
                .and_then(|a| self.credit_supply(obs, a, core, &mut state));
            match result {
                Ok(result) => action = result,
                Err(_) => self.add("service_errors", 1),
            }
        }
        self.states128[&player] = state;
        self.stages["r128"] = action.clone();

        if step == 0 {
            pop(&mut self.pending148, &player);
            self.reset_report(REPORT148);
        }
        let pending = pop(&mut self.pending148, &player);
        if truth(&pending) {
            if step != int(&pending[0]) || !same_stock(&obs["private"]["shed"], &pending[1]) {
                self.add("overflow_contract_errors", 1);
            } else {
                self.add("overflow_contract_checks", 1);
            }
        }
        if standard {
            match self.overflow(obs, action.clone()) {
                Ok(result) => action = result,
                Err(_) => self.add("targeted_errors", 1),
            }
        }
        self.stages["r148"] = action.clone();
        action
    }
}
