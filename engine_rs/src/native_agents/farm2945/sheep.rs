//! 2945 Farm SL / VE / VT overrides of the inherited six-sheep project.
//! Source: agents/farm2945/main.py (`_sl_*`, `_VE_*`, `_VT_*` and the
//! redefinitions of `_v233_request`, `_v233_worker`, `_v233_eligible`).
//! Called from the shared V43 production module when its `farm2945` flag is set.
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::production::{Production, distance, fib, home, walk};
use serde_json::{Value, json};

type Pos = (i64, i64);

const SL_TILES: [Pos; 6] = [(5, 5), (6, 5), (7, 5), (7, 6), (6, 6), (5, 6)];
const VT_TILES: [Pos; 6] = [(5, 5), (6, 5), (7, 5), (5, 6), (6, 6), (7, 6)];
const ACCESS: [Pos; 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];

fn inc(v: &mut Value, k: &str, n: i64) {
    v[k] = json!(int(&v[k]) + n);
}

fn farm(obs: &Value) -> &Value {
    &obs["farms"][int(&obs["player"]) as usize]
}

fn tile(obs: &Value, p: Pos) -> &Value {
    &farm(obs)["tiles"][p.1 as usize][p.0 as usize]
}

fn is_sheep(t: &Value) -> bool {
    t.is_object() && t["animal"] == "SHEEP"
}

fn points(v: &Value) -> Vec<Pos> {
    array(v).iter().map(position).collect()
}

fn travel(start: Pos, path: &[Pos]) -> i64 {
    distance(start, path[0]) + path.windows(2).map(|w| distance(w[0], w[1])).sum::<i64>()
}

/// `_sl_path`: the cheapest visiting order, ties broken by the path itself.
pub fn sl_path(start: Pos, targets: &[Pos]) -> Vec<Pos> {
    fn visit(
        start: Pos,
        rest: &mut Vec<Pos>,
        path: &mut Vec<Pos>,
        best: &mut Option<(i64, Vec<Pos>)>,
    ) {
        if rest.is_empty() {
            let key = (travel(start, path), path.clone());
            if best.as_ref().is_none_or(|b| key < *b) {
                *best = Some(key);
            }
            return;
        }
        for i in 0..rest.len() {
            let p = rest.remove(i);
            path.push(p);
            visit(start, rest, path, best);
            path.pop();
            rest.insert(i, p);
        }
    }
    if targets.is_empty() {
        return vec![];
    }
    let mut best = None;
    visit(start, &mut targets.to_vec(), &mut vec![], &mut best);
    best.map(|b| b.1).unwrap_or_default()
}

/// `_r62_input_start`: first acting step and spawn tile of the next extra hire.
fn input_start(obs: &Value, action: &Value, index: usize) -> (i64, Pos) {
    let f = farm(obs);
    let mut positions = vec![position(&f["farmer"])];
    positions.extend(array(&f["hands"]).iter().map(position));
    let n = positions.len();
    for (p, c) in positions.iter_mut().zip(commands(action).iter().take(n)) {
        let delta = match text(&c[0]) {
            "NORTH" => (0, -1),
            "SOUTH" => (0, 1),
            "EAST" => (1, 0),
            "WEST" => (-1, 0),
            _ => continue,
        };
        *p = ((p.0 + delta.0).clamp(0, 9), (p.1 + delta.1).clamp(0, 9));
    }
    let hires = orders(action).iter().filter(|o| o[0] == "HIRE").count();
    let mut chosen = ACCESS[0];
    for _ in 0..hires + index + 1 {
        chosen = *ACCESS
            .iter()
            .min_by_key(|p| positions.iter().filter(|q| q == p).count())
            .unwrap();
        positions.push(chosen);
    }
    (int(&obs["step"]) + 2, chosen)
}

/// VE `_v233_eligible` on day 11; `base` is the inherited eligibility test.
pub fn eligible_day11(
    obs: &Value,
    native: &Value,
    report: &mut Value,
    base: fn(&Value, &Value) -> bool,
) -> bool {
    let step = int(&obs["step"]);
    let route = int(&native["route"]);
    let private = &obs["private"];
    let held = int(&private["shed"]["SHEEP"])
        + array(&private["inventories"])
            .iter()
            .map(|i| int(&i["SHEEP"]))
            .sum::<i64>();
    let mut pickups = 0;
    for t in step..12 * 24 {
        for c in commands(&route_action(route, t)) {
            if c[0] == "PICKUP" && c[1] == "SHEEP" {
                pickups += if array(&c).len() > 2 { int(&c[2]) } else { 1 };
            }
        }
    }
    if held > pickups {
        return false;
    }
    let mut clean = obs.clone();
    clean["private"]["shed"]["SHEEP"] = json!(0);
    if let Some(inventories) = clean["private"]["inventories"].as_array_mut() {
        for inventory in inventories {
            if let Some(m) = inventory.as_object_mut() {
                m.shift_remove("SHEEP");
            }
        }
    }
    if !base(&clean, native) {
        return false;
    }
    inc(report, "ve_day11_checks", 1);
    let prices = &obs["market"]["prices"];
    let length = array(&routes()[route.to_string()]).len() as i64;
    let mut spend = 0;
    let mut hires = std::collections::HashMap::new();
    for t in step + 1..length.min(13 * 24) {
        for o in orders(&route_action(route, t)) {
            if !truth(&o) {
                continue;
            }
            let item = text(&o[1]);
            match text(&o[0]) {
                "BUY_LAND" => return false,
                "BUY_ANIMAL" if item == "SHEEP" => return false,
                "BUY_SEED" => {
                    spend += int(&o[2])
                        * match item {
                            "WHEAT" => 10,
                            "CARROT" => 20,
                            "TOMATO" => 50,
                            "MELON" => 80,
                            _ => 100,
                        }
                }
                "BUY_PRODUCT" => {
                    let quote = prices.get(item).map(int).unwrap_or(50);
                    spend += int(&o[2]) * (quote + 10);
                }
                "BUY_ANIMAL" => {
                    spend += int(&o[2])
                        * match item {
                            "COW" => 400,
                            "GOOSE" => 300,
                            _ => 500,
                        }
                }
                "HIRE" => {
                    let n = hires.entry(t / 24).or_insert(0);
                    spend += fib(*n);
                    *n += 1;
                }
                _ => {}
            }
        }
    }
    if num(&farm(obs)["money"]) < (7000 + 3000 + spend) as f64 {
        inc(report, "ve_day11_budget_declines", 1);
        return false;
    }
    inc(report, "ve_day11_ok", 1);
    true
}

fn path_json(path: &[Pos]) -> Value {
    json!(path.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>())
}

/// SL `_v233_request`: one hand instead of two when it can feed and care for
/// all six sheep alone. `result` is the granted base request for `action`.
pub fn compact_request(
    obs: &Value,
    action: &Value,
    result: Value,
    state: &mut Value,
    report: &mut Value,
) -> Value {
    if !truth(&state["pending"]) || truth(&state["pending"]["initial"]) {
        return result;
    }
    let tiles: Vec<&Value> = SL_TILES.iter().map(|p| tile(obs, *p)).collect();
    if !tiles
        .iter()
        .all(|t| is_sheep(t) && int(&t["yield_units"]) == 0)
    {
        return result;
    }
    let step = int(&obs["step"]);
    let (ready, spawn) = input_start(obs, action, 0);
    let path = sl_path(spawn, &SL_TILES);
    let cost = travel(spawn, &path);
    let mandatory = tiles.iter().filter(|t| !truth(&t["fed_today"])).count() as i64
        + tiles.iter().filter(|t| !truth(&t["cared_today"])).count() as i64;
    let available = ((step / 24 + 1) * 24).min(719) - ready;
    if cost + mandatory > available {
        return result;
    }
    let native_hires = orders(action).iter().filter(|o| o[0] == "HIRE").count() as i64;
    let saved = fib(int(&farm(obs)["hires_today"]) + native_hires + 1);
    let last = *path.last().unwrap();
    let delivery = if step / 24 == 29 {
        distance(last, home(last)) + 1
    } else {
        0
    };
    let possible = (available - cost - mandatory - delivery).clamp(0, 6);
    if saved <= (6 - possible) * int(&obs["market"]["prices"]["FERTILIZER"]) {
        return result;
    }
    let mut out = result;
    let mut market = orders(&out);
    market.pop();
    set_orders(&mut out, market);
    state["pending"]["count"] = json!(1);
    state["pending"]["targets"] = path_json(&path);
    inc(report, "sheep_hire_requests", -1);
    inc(report, "sl_compact_days", 1);
    out
}

fn wool_tiles(obs: &Value) -> Vec<Pos> {
    VT_TILES
        .into_iter()
        .filter(|p| is_sheep(tile(obs, *p)) && int(&tile(obs, *p)["yield_units"]) > 0)
        .collect()
}

/// VT `_v233_request`: day 29 hires nothing without wool, else one harvest hand.
pub fn final_days_request(
    obs: &Value,
    action: Value,
    result: Value,
    state: &mut Value,
    report: &mut Value,
) -> Value {
    if int(&obs["step"]) / 24 != 29 || !truth(&state["committed"]) {
        return result;
    }
    if !truth(&state["pending"]) || truth(&state["pending"]["initial"]) {
        return result;
    }
    let wool = wool_tiles(obs);
    let hires = orders(&result)
        .iter()
        .skip(orders(&action).len())
        .filter(|o| **o == json!(["HIRE"]))
        .count() as i64;
    inc(report, "sheep_feed_buy_requests", -6);
    if wool.is_empty() {
        if let Some(m) = state.as_object_mut() {
            m.shift_remove("pending");
        }
        inc(report, "sheep_hire_requests", -hires);
        inc(report, "vt_no_hire_days", 1);
        return action;
    }
    let mut out = action.clone();
    let mut market = orders(&out);
    market.push(json!(["HIRE"]));
    set_orders(&mut out, market);
    inc(report, "sheep_hire_requests", -(hires - 1));
    state["pending"]["count"] = json!(1);
    state["pending"]["targets"] = path_json(&sl_path(input_start(obs, &action, 0).1, &wool));
    inc(report, "vt_single_harvest_days", 1);
    out
}

fn place_cargo(pos: Pos, dest: Pos, inv: &Value, item: &str) -> Value {
    walk(pos, dest).unwrap_or_else(|| json!(["PLACE", item, inv[item]]))
}

/// VT `_v233_worker` on days 28 and 29.
fn final_days_worker(obs: &Value, actor: usize, targets: &Value, report: &mut Value) -> Value {
    let step = int(&obs["step"]);
    let day = step / 24;
    let view = View::new(obs);
    let pos = position(&view.positions[actor]);
    let inv = view.inv(actor);
    let dest = home(pos);
    let cargo: Vec<&str> = ["WOOL", "FERTILIZER"]
        .into_iter()
        .filter(|i| truth(&inv[*i]))
        .collect();
    let last = if day == 29 { 717 } else { day * 24 + 23 };
    if !cargo.is_empty() && step >= last - distance(pos, dest) {
        return place_cargo(pos, dest, inv, cargo[0]);
    }
    let shed = &obs["private"]["shed"];
    let sheep: Vec<Pos> = points(targets)
        .into_iter()
        .filter(|p| is_sheep(tile(obs, *p)))
        .collect();
    if day == 28 {
        let hungry = sheep
            .iter()
            .filter(|p| !truth(&tile(obs, **p)["fed_today"]))
            .count() as i64;
        if hungry != 0 && !truth(&inv["WHEAT"]) && truth(&shed["WHEAT"]) {
            return walk(pos, dest)
                .unwrap_or_else(|| json!(["PICKUP", "WHEAT", hungry.min(int(&shed["WHEAT"]))]));
        }
    }
    let mut tasks = vec![];
    for p in sheep {
        let t = tile(obs, p);
        let command = if day == 28 && !truth(&t["fed_today"]) && truth(&inv["WHEAT"]) {
            Some("FEED")
        } else if truth(&t["yield_units"]) {
            Some("HARVEST")
        } else if day == 28 && truth(&t["fertilizer_available"]) {
            Some("COLLECT_FERTILIZER")
        } else {
            None
        };
        if day == 28 && !truth(&t["cared_today"]) && command.is_none() {
            inc(report, "vt_skipped_care", 1);
        }
        if let Some(command) = command {
            tasks.push((distance(pos, p), p, command));
        }
    }
    if let Some((_, target, command)) = tasks.into_iter().min() {
        return walk(pos, target).unwrap_or_else(|| json!([command]));
    }
    if !cargo.is_empty() {
        return place_cargo(pos, dest, inv, cargo[0]);
    }
    json!(["PASS"])
}

/// SL `_v233_worker` for the single six-sheep hand; `None` defers to the base worker.
fn compact_worker(obs: &Value, actor: usize, targets: &Value, report: &mut Value) -> Option<Value> {
    let step = int(&obs["step"]);
    let day = step / 24;
    let view = View::new(obs);
    let pos = position(&view.positions[actor]);
    let inv = view.inv(actor);
    let shed = &obs["private"]["shed"];
    let targets = points(targets);
    let mut needed = vec![];
    let mut hungry = 0;
    for p in &targets {
        let t = tile(obs, *p);
        if !is_sheep(t) {
            return None;
        }
        if !truth(&t["fed_today"]) {
            hungry += 1;
        }
        if !truth(&t["fed_today"]) || !truth(&t["cared_today"]) {
            needed.push(*p);
        }
    }
    let dest = home(pos);
    if hungry > int(&inv["WHEAT"]) {
        return Some(
            walk(pos, dest)
                .unwrap_or_else(|| json!(["PICKUP", "WHEAT", hungry.min(int(&shed["WHEAT"]))])),
        );
    }
    if !needed.is_empty() {
        let path = sl_path(pos, &needed);
        let current = tile(obs, pos);
        if targets.contains(&pos)
            && current.is_object()
            && truth(&current["fed_today"])
            && truth(&current["cared_today"])
            && truth(&current["fertilizer_available"])
        {
            let work = path
                .iter()
                .map(|p| {
                    let t = tile(obs, *p);
                    i64::from(!truth(&t["fed_today"])) + i64::from(!truth(&t["cared_today"]))
                })
                .sum::<i64>();
            let last = *path.last().unwrap();
            let delivery = if day == 29 {
                distance(last, home(last)) + 1
            } else {
                0
            };
            let remaining = ((day + 1) * 24).min(719) - step;
            if 1 + travel(pos, &path) + work + delivery <= remaining {
                inc(report, "sl_collect", 1);
                return Some(json!(["COLLECT_FERTILIZER"]));
            }
        }
        let target = path[0];
        let t = tile(obs, target);
        return Some(walk(pos, target).unwrap_or_else(|| {
            json!([if !truth(&t["fed_today"]) {
                "FEED"
            } else {
                "CARE"
            }])
        }));
    }
    // Essential work is complete: collect what can still reach the shed.
    let remaining = if day == 29 {
        719 - step
    } else {
        24 - step % 24
    };
    let mut tasks = vec![];
    for p in &targets {
        if !truth(&tile(obs, *p)["fertilizer_available"]) {
            continue;
        }
        let dist = distance(pos, *p);
        let ret = if day == 29 {
            distance(*p, home(*p)) + 1
        } else {
            0
        };
        if dist + 1 + ret <= remaining {
            tasks.push((dist, *p));
        }
    }
    if let Some((_, target)) = tasks.into_iter().min() {
        let command = walk(pos, target);
        if command.is_none() {
            inc(report, "sl_collect", 1);
        }
        return Some(command.unwrap_or_else(|| json!(["COLLECT_FERTILIZER"])));
    }
    if truth(&inv["FERTILIZER"]) {
        return Some(
            walk(pos, dest).unwrap_or_else(|| json!(["PLACE", "FERTILIZER", inv["FERTILIZER"]])),
        );
    }
    Some(json!(["PASS"]))
}

/// `_v233_worker` as redefined by SL then VT.
pub fn worker(obs: &Value, actor: usize, targets: &Value, report: &mut Value) -> Value {
    worker_configured(obs, actor, targets, report, false)
}

pub fn worker_configured(
    obs: &Value,
    actor: usize,
    targets: &Value,
    report: &mut Value,
    cha22: bool,
) -> Value {
    let day = int(&obs["step"]) / 24;
    if day == 28 || day == 29 {
        return final_days_worker(obs, actor, targets, report);
    }
    if array(targets).len() == 6
        && let Some(command) = compact_worker(obs, actor, targets, report)
    {
        return command;
    }
    Production::sheep_worker_configured(obs, actor, targets, cha22)
}
