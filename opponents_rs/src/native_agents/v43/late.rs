//! Native translation of V43 EXP182–231 input, feed, and funding layers.
//! Source and Apache-2.0 attribution: agents/v43/{main.py,ATTRIBUTION.md}.
use super::common::*;
use super::core::Core;
use super::production::{Production, market_price};
use super::terminal::{apply_unit_action, clone_state};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
fn arr(v: &Value) -> Vec<Value> {
    v.as_array().cloned().unwrap_or_default()
}
fn op(c: &Value) -> &str {
    c[0].as_str().unwrap_or("")
}
fn is(c: &Value, verb: &str, item: &str) -> bool {
    op(c) == verb && c[1] == item
}
fn amount(c: &Value) -> i64 {
    c.get(2).map(int).unwrap_or(1).max(0)
}
fn sum(v: &Value) -> i64 {
    v.as_object()
        .map(|m| m.values().map(int).sum())
        .unwrap_or(0)
}
fn put(v: &mut Value, k: &str, n: i64) {
    if !v.is_object() {
        *v = json!({});
    }
    v[k] = json!(n);
}
fn farm(obs: &Value) -> &Value {
    &obs["farms"][int(&obs["player"]) as usize]
}
fn dist(a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
fn delta(s: &str) -> Option<(i64, i64)> {
    match s {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    }
}
fn walk(a: (i64, i64), b: (i64, i64)) -> Option<Value> {
    if a.0 != b.0 {
        Some(json!([if a.0 < b.0 { "EAST" } else { "WEST" }]))
    } else if a.1 != b.1 {
        Some(json!([if a.1 < b.1 { "SOUTH" } else { "NORTH" }]))
    } else {
        None
    }
}
fn tile_at(obs: &Value, p: (i64, i64)) -> &Value {
    &farm(obs)["tiles"][p.1 as usize][p.0 as usize]
}
fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1_i64, 1_i64);
    for _ in 0..n {
        (a, b) = (b, a.saturating_add(b));
    }
    a
}
fn route(core: &Core, obs: &Value) -> i64 {
    int(&core.players[int(&obs["player"]).to_string()]["route"])
}
fn tape_action(route: i64, t: i64) -> Value {
    route_action(if t >= 648 { 2 } else { route }, t)
}
fn native_day(core: &Core, obs: &Value) -> Vec<Value> {
    let day = int(&obs["step"]) / 24;
    (day * 24..((day + 1) * 24).min(719))
        .map(|t| route_action(route(core, obs), t))
        .collect()
}
fn expected(tape: &[Value]) -> usize {
    tape.iter()
        .map(|a| arr(&a["hands"]).len())
        .max()
        .unwrap_or(0)
}
pub fn cfg_ok(cfg: &Value, hire: bool) -> bool {
    [
        ("boardSize", 10),
        ("turnsPerDay", 24),
        ("shedCapacity", 100),
        ("maxMarketOrdersPerTurn", 10),
        ("farmHandCostMult", 1),
    ]
    .iter()
    .take(if hire { 5 } else { 4 })
    .all(|(k, v)| {
        cfg.get(*k)
            .map(|x| (x.is_number() || x.is_boolean()) && num(x) == *v as f64)
            .unwrap_or(true)
    })
}
fn positions_after(obs: &Value, action: &Value) -> Vec<(i64, i64)> {
    let f = farm(obs);
    let mut ps = vec![position(&f["farmer"])];
    ps.extend(arr(&f["hands"]).iter().map(position));
    for (p, c) in ps.iter_mut().zip(commands(action)) {
        if let Some((dx, dy)) = delta(op(&c)) {
            *p = ((p.0 + dx).clamp(0, 9), (p.1 + dy).clamp(0, 9));
        }
    }
    ps
}
fn spawn(ps: &mut Vec<(i64, i64)>) -> (i64, i64) {
    let p = *ACCESS
        .iter()
        .min_by_key(|p| ps.iter().filter(|q| q == p).count())
        .unwrap();
    ps.push(p);
    p
}
fn apply_units(
    obs: &Value,
    action: &Value,
    block_plants: bool,
) -> Result<(Value, Value), &'static str> {
    let (mut f, mut p) = clone_state(farm(obs), &obs["private"]);
    let cs = commands(action);
    let mut demand = BTreeMap::new();
    for c in &cs {
        if op(c) == "PLANT" {
            *demand
                .entry(c[1].as_str().unwrap_or("").to_string())
                .or_insert(0_i64) += 1;
        }
    }
    let pass = json!(["PASS"]);
    for (actor, c) in cs.iter().take(arr(&p["inventories"]).len()).enumerate() {
        let blocked = block_plants
            && op(c) == "PLANT"
            && demand
                .get(c[1].as_str().unwrap_or(""))
                .copied()
                .unwrap_or(0)
                > int(&p["seeds"][c[1].as_str().unwrap_or("")]);
        apply_unit_action(
            &mut f,
            &mut p,
            actor,
            if blocked { &pass } else { c },
            10,
            int(&obs["step"]) / 24,
            24,
            100,
        )?;
    }
    Ok((f, p))
}

pub fn parent_fert_qty(obs: &Value, action: &Value, planned: &[Value], offset: i64) -> i64 {
    let mut stock = Core::projected_shed(action, &View::new(obs));
    for o in orders(action) {
        if arr(&o).len() < 3 {
            continue;
        }
        let item = o[1].as_str().unwrap_or("");
        let q = amount(&o);
        if op(&o) == "SELL" {
            let n = (int(&stock[item]) - q).max(0);
            put(&mut stock, item, n);
        } else if matches!(op(&o), "BUY_PRODUCT" | "BUY_ANIMAL") {
            let n = int(&stock[item]) + q.min((100 - sum(&stock)).max(0));
            put(&mut stock, item, n);
        }
    }
    let next = planned
        .get((offset + 1) as usize)
        .cloned()
        .unwrap_or(json!({}));
    let need: i64 = commands(&next)
        .iter()
        .filter(|c| is(c, "PICKUP", "FERTILIZER"))
        .map(amount)
        .sum();
    let quantity = 10.max(10 + need - int(&stock["FERTILIZER"]).max(0));
    if quantity > (100 - sum(&stock)).max(0) {
        10
    } else {
        quantity
    }
}

pub fn tomato_fertilizer_worthwhile(obs: &Value, action: &Value) -> bool {
    if num(&obs["market"]["prices"]["FERTILIZER"]) <= 30.0 {
        return true;
    }
    let day = int(&obs["step"]) / 24;
    let mut bonus = 0;
    for y in [5, 6] {
        for x in 5..10 {
            let t = &farm(obs)["tiles"][y][x];
            if !t.is_object() || t["crop"] != "TOMATO" {
                continue;
            }
            let birth = int(&t["planted_day"]);
            let until = t.get("fertilized_until_day").map(int).unwrap_or(-1);
            bonus += (day..day + 3)
                .filter(|d| until < *d && (8..=11).contains(&(d + 1 - birth)))
                .count() as i64;
        }
    }
    if bonus == 0 {
        return false;
    }
    let inv = &obs["market"]["inventory"];
    let price =
        (market_price("TOMATO", num(&inv["TOMATO"]) + (bonus + 10) as f64, None) - 2).max(1);
    let fertilizer = (market_price("FERTILIZER", num(&inv["FERTILIZER"]) - 10.0, None) + 2).max(1);
    let hires = orders(action).iter().filter(|o| op(o) == "HIRE").count() as i64;
    let labor = fib(int(&farm(obs)["hires_today"]) + hires + 3);
    bonus * price >= 2 * (10 * fertilizer + labor) + 100
}

pub fn labor_assignment(obs: &Value, action: &Value, fertilizer: bool) -> Option<Value> {
    let step = int(&obs["step"]);
    let day = step / 24;
    if ![26, 27, 28].contains(&day) || step % 24 > 2 || (day == 27 && !fertilizer) {
        return None;
    }
    let count = if fertilizer { 3 } else { 2 };
    let mut ps = positions_after(obs, action);
    let hires = orders(action).iter().filter(|o| op(o) == "HIRE").count();
    let mut spawns = vec![];
    for i in 0..hires + count {
        let p = spawn(&mut ps);
        if i >= hires {
            spawns.push(p);
        }
    }
    let groups: Vec<Vec<(i64, i64)>> = if fertilizer {
        vec![
            vec![(5, 5), (6, 5), (7, 5), (8, 5)],
            vec![(9, 5), (9, 6), (8, 6)],
            vec![(5, 6), (6, 6), (7, 6)],
        ]
    } else {
        vec![
            (5..10).map(|x| (x, 5)).collect(),
            (5..10).map(|x| (x, 6)).collect(),
        ]
    };
    let perms = if fertilizer {
        vec![
            vec![0, 1, 2],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
        ]
    } else {
        vec![vec![0, 1], vec![1, 0]]
    };
    let remaining = 23 - step % 24;
    let mut choices = vec![];
    for perm in perms {
        let paths: Vec<_> = perm.iter().map(|i| groups[*i].clone()).collect();
        let costs: Vec<i64> = spawns
            .iter()
            .zip(&paths)
            .map(|(start, path)| {
                dist(*start, path[0])
                    + path.windows(2).map(|p| dist(p[0], p[1])).sum::<i64>()
                    + ACCESS
                        .iter()
                        .map(|p| dist(*path.last().unwrap(), *p))
                        .min()
                        .unwrap()
                    + (if fertilizer { 3 } else { 2 }) * path.len() as i64
                    + 1
                    + i64::from(fertilizer)
            })
            .collect();
        let maximum = *costs.iter().max().unwrap();
        if maximum <= remaining {
            choices.push((maximum, costs.iter().sum::<i64>(), paths));
        }
    }
    choices.sort();
    choices.first().map(|(_,_,paths)|json!({"paths":paths,"spawns":spawns,"remaining":remaining,"workers":count,"fertilizer":fertilizer}))
}

#[derive(Clone, Debug)]
struct Target {
    xy: (i64, i64),
    crop: String,
    birth: i64,
    yield_units: i64,
    until: i64,
    watered: bool,
    water: Vec<i64>,
    harvest: Option<i64>,
    first: i64,
    last: i64,
    cap: i64,
}
fn forecast(obs: &Value, r: i64, expected: usize) -> Vec<Target> {
    let step = int(&obs["step"]);
    let day = step / 24;
    let f = farm(obs);
    let mut ps = vec![position(&f["farmer"])];
    ps.extend(arr(&f["hands"]).iter().take(expected).map(position));
    let mut targets = vec![];
    for (y, row) in arr(&f["tiles"]).iter().enumerate() {
        for (x, t) in arr(row).iter().enumerate() {
            let (crop, first, last, cap) = match t["crop"].as_str() {
                Some("WHEAT") => ("WHEAT", 2, 4, 6),
                Some("CARROT") => ("CARROT", 2, 3, 4),
                _ => continue,
            };
            let birth = int(&t["planted_day"]);
            if day - birth >= 1 && day - birth < last {
                targets.push(Target {
                    xy: (x as i64, y as i64),
                    crop: crop.into(),
                    birth,
                    yield_units: int(&t["yield_units"]),
                    until: t.get("fertilized_until_day").map(int).unwrap_or(-1),
                    watered: truth(&t["watered_today"]),
                    water: vec![],
                    harvest: None,
                    first,
                    last,
                    cap,
                });
            }
        }
    }
    let mut seen = BTreeSet::new();
    for t in step..712.min((day + 4) * 24) {
        let a = tape_action(r, t);
        for (actor, c) in commands(&a).iter().take(ps.len()).enumerate() {
            if let Some(target) = targets
                .iter_mut()
                .find(|q| q.xy == ps[actor] && q.harvest.is_none())
            {
                if op(c) == "WATER"
                    && seen.insert((t / 24, target.xy))
                    && !(t / 24 == day && target.watered)
                    && (target.first..=target.last).contains(&(t / 24 - target.birth))
                {
                    target.water.push(t);
                }
                if op(c) == "HARVEST" {
                    target.harvest = Some(t);
                }
            }
            if let Some((dx, dy)) = delta(op(c)) {
                ps[actor] = (
                    (ps[actor].0 + dx).clamp(0, 9),
                    (ps[actor].1 + dy).clamp(0, 9),
                );
            }
        }
        for o in orders(&a) {
            if op(&o) == "HIRE" {
                spawn(&mut ps);
            }
        }
        if (t + 1) % 24 == 0 {
            ps = vec![(4, 4)];
        }
    }
    targets
}
fn gain(t: &Target, arrival: i64, day: i64) -> i64 {
    let Some(h) = t.harvest else {
        return 0;
    };
    if h <= arrival {
        return 0;
    }
    let extra = t
        .water
        .iter()
        .filter(|x| {
            arrival < **x && **x <= h && (day..=day + 2).contains(&(**x / 24)) && **x / 24 > t.until
        })
        .count() as i64;
    let baseline = t.yield_units
        + t.water
            .iter()
            .map(|x| if *x / 24 <= t.until { 2 } else { 1 })
            .sum::<i64>();
    extra.min(t.cap - baseline).max(0)
}
type Path = Vec<(i64, i64, String, i64)>;
#[derive(Clone)]
struct Beam {
    score: f64,
    gross: i64,
    now: i64,
    pos: (i64, i64),
    path: Path,
    used: BTreeSet<(i64, i64)>,
    units: [i64; 2],
}
fn beam_cmp(a: &Beam, b: &Beam) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then(b.gross.cmp(&a.gross))
        .then(a.now.cmp(&b.now))
        .then(a.path.cmp(&b.path))
}
fn input_path(obs: &Value, targets: &[Target], action: &Value, index: usize) -> (Path, [i64; 2]) {
    let step = int(&obs["step"]);
    let day = step / 24;
    let mut ps = positions_after(obs, action);
    let hires = orders(action).iter().filter(|o| op(o) == "HIRE").count();
    let mut start = (4, 4);
    for _ in 0..hires + index + 1 {
        start = spawn(&mut ps);
    }
    let price = [
        (int(&obs["market"]["prices"]["WHEAT"]) - 2).max(1),
        (int(&obs["market"]["prices"]["CARROT"]) - 2).max(1),
    ];
    let fertilizer = (market_price(
        "FERTILIZER",
        num(&obs["market"]["inventory"]["FERTILIZER"]) - 16.0,
        None,
    ) + 2)
        .max(1);
    let mut beam = vec![Beam {
        score: 0.0,
        gross: 0,
        now: step + 2,
        pos: start,
        path: vec![],
        used: BTreeSet::new(),
        units: [0, 0],
    }];
    let mut best: Option<Beam> = None;
    for depth in 0..8 {
        let mut expanded = vec![];
        for b in &beam {
            for target in targets {
                if b.used.contains(&target.xy) {
                    continue;
                }
                let arrival = b.now + dist(b.pos, target.xy);
                if arrival >= day * 24 + 23 {
                    continue;
                }
                let g = gain(target, arrival, day);
                if g == 0 {
                    continue;
                }
                let item = usize::from(target.crop != "WHEAT");
                let gross = b.gross + g * price[item];
                let mut next = b.clone();
                next.path
                    .push((target.xy.0, target.xy.1, target.crop.clone(), target.birth));
                next.score = gross as f64 - 1.5 * fertilizer as f64 * next.path.len() as f64;
                next.gross = gross;
                next.now = arrival + 1;
                next.pos = target.xy;
                next.used.insert(target.xy);
                next.units[item] += g;
                expanded.push(next);
            }
        }
        if expanded.is_empty() {
            break;
        }
        expanded.sort_by(beam_cmp);
        expanded.truncate(8);
        beam = expanded;
        if depth >= 2 && (best.as_ref().is_none_or(|b| beam_cmp(&beam[0], b).is_lt())) {
            best = Some(beam[0].clone());
        }
    }
    best.map(|b| (b.path, b.units)).unwrap_or_default()
}
fn joint_plans(
    obs: &Value,
    action: &Value,
    targets: &[Target],
    stock: &Value,
    purchases: i64,
    topup: i64,
) -> (Vec<Value>, i64) {
    type PlanChoice = ((i64, i64, i64, i64, i64), Vec<Value>, i64);
    let mut best: Option<PlanChoice> = None;
    for (mode, first) in [None, Some("WHEAT"), Some("CARROT")].iter().enumerate() {
        let mut remaining = targets.to_vec();
        let mut plans = vec![];
        let (mut total_q, mut total_cost, mut total_value) = (0_i64, 0_i64, 0_i64);
        let mut all = [0, 0];
        for i in 0..2 {
            let subset: Vec<_> = remaining
                .iter()
                .filter(|t| i != 0 || first.is_none() || first == &Some(t.crop.as_str()))
                .cloned()
                .collect();
            let (path, units) = input_path(obs, &subset, action, i);
            let q = path.len() as i64;
            if q < 3
                || orders(action).len() + 2 + i > 10
                || sum(stock) + purchases + total_q + q + topup > 95
            {
                break;
            }
            let quote = market_price(
                "FERTILIZER",
                num(&obs["market"]["inventory"]["FERTILIZER"]) - (total_q + q + topup) as f64,
                None,
            );
            let cost = (q + if i == 0 { topup } else { 0 }) * (quote + 2)
                + fib(int(&farm(obs)["hires_today"]) + i as i64);
            let value: i64 = ["WHEAT", "CARROT"]
                .iter()
                .enumerate()
                .map(|(j, item)| {
                    units[j]
                        * (market_price(
                            item,
                            num(&obs["market"]["inventory"][*item]) + (all[j] + units[j]) as f64,
                            None,
                        ) - 2)
                            .max(1)
                })
                .sum();
            if (value as f64) < 1.5 * cost as f64 + 50.0
                || num(&farm(obs)["money"]) < (total_cost + cost + 3000) as f64
            {
                break;
            }
            plans.push(json!({"path":path,"quantity":q,"loaded":false}));
            total_q += q;
            total_cost += cost;
            total_value += value;
            for j in 0..2 {
                all[j] += units[j];
            }
            remaining.retain(|t| !path.iter().any(|(x, y, _, _)| (*x, *y) == t.xy));
        }
        let key = (
            total_value - total_cost,
            total_value,
            -total_cost,
            -(plans.len() as i64),
            -(mode as i64),
        );
        if best.as_ref().is_none_or(|b| key > b.0) {
            best = Some((key, plans, total_q));
        }
    }
    let (_, plans, q) = best.unwrap();
    (plans, q)
}

#[derive(Clone, Debug, Default)]
pub struct Late {
    pub input: Value,
    feed_cache: BTreeMap<(i64, i64), BTreeSet<(i64, i64)>>,
}
impl Late {
    pub fn after(
        &mut self,
        obs: &Value,
        mut action: Value,
        core: &Core,
        production: &Production,
        cfg: &Value,
    ) -> Value {
        let seat = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        if !self.input.is_object() {
            self.input = json!({});
        }
        let mut state = self.input[&seat].clone();
        if !state.is_object() || step <= int(&state["step"]) {
            state = json!({"step":-1});
        }
        state["step"] = json!(step);
        if !cfg_ok(cfg, false) {
            self.input[&seat] = state;
            return action;
        }
        action = self.input_control(obs, action, core, production, &mut state);
        self.input[&seat] = state;
        action = close_warehouse(obs, action, core);
        action = self.feed(obs, action, core);
        action = self.fertilizer(obs, action, core, production);
        if step % 24 == 23 {
            action = close_warehouse(obs, action, core);
        }
        action = replenish(obs, action, core);
        if cfg_ok(cfg, true) {
            action = supply(obs, action, core);
        }
        action
    }
    fn input_control(
        &self,
        obs: &Value,
        mut action: Value,
        core: &Core,
        production: &Production,
        state: &mut Value,
    ) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        let hour = step % 24;
        let seat = int(&obs["player"]).to_string();
        let f = farm(obs);
        let native = &core.players[&seat];
        if state.get("day").map(int) != Some(day) {
            state["day"] = json!(day);
            state["workers"] = json!({});
            state["pending"] = Value::Null;
            state["placed"] = json!([]);
        }
        state["placed"] = json!([]);
        if let Some(pending) = state["pending"]
            .as_object()
            .filter(|pending| !pending.is_empty())
            .cloned()
        {
            state.as_object_mut().unwrap().shift_remove("pending");
            for (actor, plan) in pending {
                if arr(&f["hands"]).len() >= actor.parse::<usize>().unwrap_or(usize::MAX) {
                    state["workers"][&actor] = plan;
                }
            }
        }
        if let Some(workers) = state["workers"]
            .as_object()
            .cloned()
            .filter(|w| !w.is_empty())
        {
            let mut placed = vec![];
            for (actor, mut plan) in workers {
                let idx = actor.parse::<usize>().unwrap();
                let inv = &obs["private"]["inventories"][idx];
                let pos = position(&f["hands"][idx - 1]);
                let mut cmd = json!(["PASS"]);
                if !truth(&plan["loaded"]) {
                    let stock = Core::projected_shed(&action, &View::new(obs));
                    let q = int(&plan["quantity"]).min(int(&stock["FERTILIZER"]).max(0));
                    if q > 0 && ACCESS.contains(&pos) {
                        cmd = json!(["PICKUP", "FERTILIZER", q]);
                        plan["loaded"] = json!(true);
                    }
                } else if int(&inv["FERTILIZER"]) != 0 {
                    let mut path = arr(&plan["path"]);
                    while !path.is_empty() {
                        let entry = &path[0];
                        let xy = (int(&entry[0]), int(&entry[1]));
                        let tile = tile_at(obs, xy);
                        if !tile.is_object()
                            || tile["crop"] != entry[2]
                            || tile["planted_day"] != entry[3]
                            || tile.get("fertilized_until_day").map(int).unwrap_or(-1) >= day + 2
                        {
                            path.remove(0);
                            continue;
                        }
                        cmd = walk(pos, xy).unwrap_or(json!(["FERTILIZE"]));
                        if op(&cmd) == "FERTILIZE" {
                            placed.push(json!(xy));
                            path.remove(0);
                        }
                        break;
                    }
                    plan["path"] = json!(path);
                }
                if let Some(h) = action["hands"]
                    .as_array_mut()
                    .and_then(|h| h.get_mut(idx - 1))
                {
                    *h = cmd;
                } else {
                    return json!({"farmer":["PASS"],"hands":[],"market":[]});
                }
                state["workers"][&actor] = plan;
            }
            state["placed"] = json!(placed);
            return action;
        }
        if ![1, 2, 3].contains(&hour) || !(12..=28).contains(&day) {
            return action;
        }
        let planned = native_day(core, obs);
        let expected = expected(&planned);
        if planned
            .iter()
            .skip(hour as usize)
            .any(|a| orders(a).iter().any(|o| op(o) == "HIRE"))
            || truth(&native["pending"])
        {
            return action;
        }
        let parents = [&production.v219[&seat], &production.v233[&seat]];
        if [12, 18].contains(&day)
            || parents
                .iter()
                .any(|p| truth(&p["committed"]) && p.get("requested_day").map(int) != Some(day))
        {
            return action;
        }
        if parents.iter().any(|p| truth(&p["pending"]))
            || orders(&action).iter().any(|o| op(o) == "HIRE")
        {
            return action;
        }
        let mut owned: BTreeSet<usize> = (1..=expected).collect();
        for p in parents {
            if let Some(ws) = p["workers"].as_object() {
                for a in ws.keys() {
                    let actor = a.parse().unwrap();
                    if !owned.insert(actor) {
                        return action;
                    }
                }
            }
        }
        if owned != (1..=arr(&f["hands"]).len()).collect() {
            return action;
        }
        let targets = forecast(obs, route(core, obs), expected);
        let stock = Core::projected_shed(&action, &View::new(obs));
        let purchases = orders(&action)
            .iter()
            .filter(|o| matches!(op(o), "BUY_PRODUCT" | "BUY_ANIMAL") && arr(o).len() > 2)
            .map(amount)
            .sum();
        let mut available = int(&stock["FERTILIZER"]).max(0);
        for o in orders(&action) {
            if is(&o, "SELL", "FERTILIZER") {
                available = (available - amount(&o)).max(0);
            } else if is(&o, "BUY_PRODUCT", "FERTILIZER") {
                available += amount(&o);
            }
        }
        let need: i64 = commands(&planned[(hour + 1) as usize])
            .iter()
            .filter(|c| is(c, "PICKUP", "FERTILIZER"))
            .map(amount)
            .sum();
        let topup = (need - available).max(0);
        let (plans, q) = joint_plans(obs, &action, &targets, &stock, purchases, topup);
        if plans.is_empty() {
            return action;
        }
        let mut pending = json!({});
        for (i, p) in plans.iter().enumerate() {
            pending[(arr(&f["hands"]).len() + 1 + i).to_string()] = p.clone();
        }
        state["pending"] = pending;
        let mut os = orders(&action);
        os.push(json!(["BUY_PRODUCT", "FERTILIZER", q + topup]));
        os.extend(plans.iter().map(|_| json!(["HIRE"])));
        set_orders(&mut action, os);
        action
    }

    fn next_feed(&mut self, obs: &Value, target: (i64, i64), core: &Core) -> bool {
        let day = int(&obs["step"]) / 24;
        if day == 28 {
            return true;
        }
        let tomorrow = day + 1;
        let r = if tomorrow >= 27 { 2 } else { route(core, obs) };
        let key = (r, tomorrow);
        let feeds = self.feed_cache.entry(key).or_insert_with(|| {
            let mut ps = vec![(4, 4)];
            let mut wheat = vec![0_i64];
            let mut feeds = BTreeSet::new();
            for hour in 0..24 {
                let a = route_action(r, tomorrow * 24 + hour);
                for (actor, c) in commands(&a).iter().take(ps.len()).enumerate() {
                    let pos = ps[actor];
                    if let Some((dx, dy)) = delta(op(c)) {
                        ps[actor] = ((pos.0 + dx).clamp(0, 9), (pos.1 + dy).clamp(0, 9));
                    } else if is(c, "PICKUP", "WHEAT") && ACCESS.contains(&pos) {
                        wheat[actor] += amount(c);
                    } else if op(c) == "FEED" && wheat[actor] > 0 {
                        wheat[actor] -= 1;
                        if hour <= 21 {
                            feeds.insert(pos);
                        }
                    } else if op(c) == "DROP" && ACCESS.contains(&pos) {
                        wheat[actor] = 0;
                    } else if is(c, "PLACE", "WHEAT") && ACCESS.contains(&pos) {
                        wheat[actor] = (wheat[actor] - amount(c)).max(0);
                    }
                }
                for o in orders(&a) {
                    if op(&o) == "HIRE" {
                        spawn(&mut ps);
                        wheat.push(0);
                    }
                }
            }
            feeds
        });
        feeds.contains(&target)
    }
    fn feed(&mut self, obs: &Value, mut action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        if !(10..=28).contains(&day) || step % 24 > 21 {
            return action;
        }
        let expected = expected(&native_day(core, obs));
        let f = farm(obs);
        let mut ps = vec![position(&f["farmer"])];
        ps.extend(arr(&f["hands"]).iter().map(position));
        let mut cs = commands(&action);
        for (actor, c) in cs.iter_mut().take(expected + 1).enumerate() {
            if *c != json!(["FEED"]) || actor >= ps.len() {
                continue;
            }
            let tile = tile_at(obs, ps[actor]);
            let item = match tile["animal"].as_str() {
                Some("GOOSE") => "EGG",
                Some("COW") => "MILK",
                Some("SHEEP") => "WOOL",
                _ => continue,
            };
            if truth(&tile["fed_today"])
                || int(&tile["consecutive_unfed"]) != 0
                || int(&obs["private"]["inventories"][actor]["WHEAT"]) <= 0
            {
                continue;
            }
            let bonus = feed_bonus(tile, day);
            if bonus as f64 * (num(&obs["market"]["prices"][item]) + 5.0) * 1.25
                >= num(&obs["market"]["prices"]["WHEAT"])
            {
                continue;
            }
            if self.next_feed(obs, ps[actor], core) {
                *c = json!(["PASS"]);
            }
        }
        set_commands(&mut action, cs);
        action
    }
    fn reserve(&self, obs: &Value, core: &Core, production: &Production) -> i64 {
        let step = int(&obs["step"]);
        let seat = int(&obs["player"]).to_string();
        let r = route(core, obs);
        let mut reserve = 0_i64;
        for t in ((step + 1).min(719)..719).rev() {
            let a = tape_action(r, t);
            let pickup: i64 = commands(&a)
                .iter()
                .filter(|c| is(c, "PICKUP", "FERTILIZER"))
                .map(amount)
                .sum();
            let purchase: i64 = orders(&a)
                .iter()
                .filter(|o| is(o, "BUY_PRODUCT", "FERTILIZER"))
                .map(amount)
                .sum();
            reserve = pickup + (reserve - purchase).max(0);
        }
        let mut dedicated = 0;
        for parent in [&production.v219[&seat], &production.v233[&seat]] {
            if let Some(workers) = parent["workers"].as_object() {
                for (actor, role) in workers {
                    if !role.is_object()
                        || !truth(&role["needs_fertilizer"])
                        || truth(&role["loaded"])
                    {
                        continue;
                    }
                    let desired = role
                        .get("fertilizer_quantity")
                        .map(int)
                        .unwrap_or(if role["kind"] == "fertilizer" { 10 } else { 5 });
                    let carried = int(
                        &obs["private"]["inventories"][actor.parse::<usize>().unwrap()]["FERTILIZER"]
                    );
                    dedicated += (desired - carried).max(0);
                }
            }
            if truth(&parent["pending"]["fertilizer"]) {
                dedicated += 10;
            }
        }
        let input = &self.input[&seat];
        let mut plans = input["workers"].as_object().cloned().unwrap_or_default();
        if let Some(pending) = input["pending"].as_object() {
            plans.extend(pending.clone());
        }
        for plan in plans.values() {
            if !truth(&plan["loaded"]) {
                dedicated += int(&plan["quantity"]).max(0);
            }
        }
        14.max(reserve + dedicated)
    }
    fn fertilizer(
        &self,
        obs: &Value,
        mut action: Value,
        core: &Core,
        production: &Production,
    ) -> Value {
        let day = int(&obs["step"]) / 24;
        if !(6..=28).contains(&day) {
            return action;
        }
        let mut os = orders(&action);
        if os.len() >= 10 || os.iter().any(|o| !arr(o).is_empty() && op(o) != "SELL") {
            return action;
        }
        let stock = Core::projected_shed(&action, &View::new(obs));
        let sold: i64 = os
            .iter()
            .filter(|o| is(o, "SELL", "FERTILIZER"))
            .map(amount)
            .sum();
        let extra = int(&stock["FERTILIZER"]).max(0) - sold - self.reserve(obs, core, production);
        if extra > 0 {
            os.push(json!(["SELL", "FERTILIZER", extra]));
            set_orders(&mut action, os);
        }
        action
    }
}

fn feed_bonus(tile: &Value, day: i64) -> i64 {
    let (first, interval) = match tile["animal"].as_str() {
        Some("GOOSE") => (4, 1),
        Some("COW") => (8, 2),
        _ => (6, 3),
    };
    let first = first + int(&tile["placed_day"]);
    let tomorrow = day + 1;
    let produces = tomorrow >= first && (tomorrow - first) % interval == 0;
    let pending = if produces {
        int(&tile["pending_care_bonus"]).max(0)
    } else {
        0
    };
    let mut next = first;
    if next <= tomorrow {
        next += ((tomorrow - next) / interval + 1) * interval;
    }
    pending + i64::from(next <= 29)
}

fn close_warehouse(obs: &Value, mut action: Value, core: &Core) -> Value {
    let step = int(&obs["step"]);
    let day = step / 24;
    if step % 24 != 23
        || !(12..=28).contains(&day)
        || orders(&action)
            .iter()
            .any(|o| !arr(o).is_empty() && op(o) != "SELL")
    {
        return action;
    }
    let Ok((_, private)) = apply_units(obs, &action, true) else {
        return action;
    };
    let mut post = private["shed"].clone();
    let mut os = orders(&action);
    for o in &os {
        if op(o) == "SELL" && arr(o).len() >= 3 {
            let item = o[1].as_str().unwrap_or("");
            let n = (int(&post[item]) - amount(o)).max(0);
            put(&mut post, item, n);
        }
    }
    let mut needed = sum(&post)
        + arr(&private["inventories"])
            .iter()
            .map(|inv| {
                inv.as_object()
                    .map(|m| m.values().map(|v| int(v).max(0)).sum::<i64>())
                    .unwrap_or(0)
            })
            .sum::<i64>()
        - 100;
    if needed <= 0 {
        return action;
    }
    let mut products: Vec<_> = PRODUCTS
        .iter()
        .filter(|p| !matches!(**p, "WHEAT" | "FERTILIZER"))
        .copied()
        .collect();
    products.sort_by_key(|p| -int(&obs["market"]["prices"][*p]));
    for item in products {
        let qty = needed.min(int(&post[item]));
        if qty <= 0 {
            continue;
        }
        if let Some(o) = os.iter_mut().find(|o| is(o, "SELL", item)) {
            o[2] = json!(amount(o) + qty);
        } else if os.len() < 10 {
            os.push(json!(["SELL", item, qty]));
        } else {
            continue;
        }
        needed -= qty;
        let n = int(&post[item]) - qty;
        put(&mut post, item, n);
        if needed <= 0 {
            break;
        }
    }
    if needed > 0 {
        let mut reserve = 0;
        for t in step + 1..719 {
            let a = tape_action(route(core, obs), t);
            reserve += commands(&a)
                .iter()
                .filter(|c| is(c, "PICKUP", "WHEAT"))
                .map(amount)
                .sum::<i64>();
            if orders(&a).iter().any(|o| is(o, "BUY_PRODUCT", "WHEAT")) {
                break;
            }
        }
        let incoming: i64 = arr(&private["inventories"])
            .iter()
            .map(|inv| int(&inv["WHEAT"]).max(0))
            .sum();
        let others = sum(&post) - int(&post["WHEAT"])
            + arr(&private["inventories"])
                .iter()
                .map(|inv| {
                    inv.as_object()
                        .map(|m| {
                            m.iter()
                                .filter(|(k, _)| k.as_str() != "WHEAT")
                                .map(|(_, v)| int(v).max(0))
                                .sum::<i64>()
                        })
                        .unwrap_or(0)
                })
                .sum::<i64>();
        let qty = if 100 - others >= reserve {
            needed
                .min(int(&post["WHEAT"]))
                .min((int(&post["WHEAT"]) + incoming - reserve).max(0))
        } else {
            0
        };
        if qty > 0 {
            if let Some(o) = os.iter_mut().find(|o| is(o, "SELL", "WHEAT")) {
                o[2] = json!(amount(o) + qty);
            } else if os.len() < 10 {
                os.push(json!(["SELL", "WHEAT", qty]));
            }
        }
    }
    set_orders(&mut action, os);
    action
}

fn wheat_reserve(obs: &Value, core: &Core) -> i64 {
    let step = int(&obs["step"]);
    let mut demand = 6;
    for t in step + 1..719.min(step + 49) {
        let a = tape_action(route(core, obs), t);
        demand += commands(&a)
            .iter()
            .filter(|c| is(c, "PICKUP", "WHEAT"))
            .map(amount)
            .sum::<i64>();
        demand += orders(&a)
            .iter()
            .filter(|o| is(o, "SELL", "WHEAT") && arr(o).len() > 2)
            .map(amount)
            .sum::<i64>();
    }
    if arr(&obs["town"]["unlocked_shops"])
        .iter()
        .filter(|s| **s == "YARN_STORE")
        .count()
        >= 2
    {
        demand += 6
            * (step + 1..step + 49)
                .filter(|t| t / 24 >= 12)
                .map(|t| t / 24)
                .collect::<BTreeSet<_>>()
                .len() as i64;
    }
    demand
}
fn replenish(obs: &Value, mut action: Value, core: &Core) -> Value {
    let step = int(&obs["step"]);
    if !(10..=11).contains(&(step / 24)) {
        return action;
    }
    let mut os = orders(&action);
    if !os
        .iter()
        .any(|o| is(o, "BUY_PRODUCT", "WHEAT") && arr(o).len() > 2 && int(&o[2]) > 0)
        || os.iter().any(|o| is(o, "SELL", "WHEAT"))
    {
        return action;
    }
    let Ok((_, private)) = apply_units(obs, &action, false) else {
        return action;
    };
    let mut held = int(&private["shed"]["WHEAT"]).max(0);
    let reserve = wheat_reserve(obs, core);
    for o in &mut os {
        if !is(o, "BUY_PRODUCT", "WHEAT") || arr(o).len() < 3 {
            continue;
        }
        let q = amount(o);
        let retained = q.min((reserve - held).max(0));
        held += retained;
        if retained < q {
            o[2] = json!(retained);
        }
    }
    set_orders(&mut action, os);
    action
}

pub fn market_stock(
    shed: &Value,
    os: &[Value],
) -> (Value, BTreeMap<usize, i64>, BTreeMap<usize, i64>) {
    let mut stock = shed.clone();
    let (mut buys, mut sales) = (BTreeMap::new(), BTreeMap::new());
    for (index, o) in os.iter().enumerate() {
        if arr(o).len() < 3 {
            continue;
        }
        let item = o[1].as_str().unwrap_or("");
        let n = amount(o);
        if op(o) == "SELL" {
            let q = n.min(int(&stock[item]).max(0));
            let held = int(&stock[item]) - q;
            put(&mut stock, item, held);
            sales.insert(index, q);
        } else if matches!(op(o), "BUY_PRODUCT" | "BUY_ANIMAL") {
            let q = n.min((100 - sum(&stock)).max(0));
            let held = int(&stock[item]) + q;
            put(&mut stock, item, held);
            buys.insert(index, q);
        }
    }
    (stock, buys, sales)
}
pub(crate) fn delivery(mut stock: Value, private: &Value, night: bool) -> (Value, Value) {
    let mut lost = json!({});
    if night {
        for inv in arr(&private["inventories"]) {
            if let Some(items) = inv.as_object() {
                for (item, q) in items {
                    let q = int(q).max(0);
                    let take = q.min((100 - sum(&stock)).max(0));
                    let n = int(&stock[item]) + take;
                    put(&mut stock, item, n);
                    if q > take {
                        let n = int(&lost[item]) + q - take;
                        put(&mut lost, item, n);
                    }
                }
            }
        }
    }
    (stock, lost)
}
pub(crate) fn budget(obs: &Value, os: &[Value]) -> bool {
    let mut cost = 0_i64;
    let mut hires = int(&farm(obs)["hires_today"]);
    let prices: [i64; 2] = ["WHEAT", "FERTILIZER"]
        .map(|p| market_price(p, num(&obs["market"]["inventory"][p]) - 2000.0, None));
    for o in os {
        match op(o) {
            "HIRE" => {
                cost = cost.saturating_add(fib(hires));
                hires += 1;
            }
            "BUY_LAND" => cost += 4000,
            "BUY_PRODUCT" | "BUY_ANIMAL" | "BUY_SEED" if arr(o).len() > 2 => {
                let item = o[1].as_str().unwrap_or("");
                let q = amount(o);
                let price = match op(o) {
                    "BUY_PRODUCT" => {
                        if item == "WHEAT" {
                            prices[0]
                        } else if item == "FERTILIZER" {
                            prices[1]
                        } else {
                            return false;
                        }
                    }
                    "BUY_ANIMAL" => match item {
                        "GOOSE" => 300,
                        "COW" => 400,
                        "SHEEP" => 500,
                        _ => return false,
                    },
                    _ => match item {
                        "WHEAT" => 10,
                        "CARROT" => 20,
                        "TOMATO" => 50,
                        "STRAWBERRY" => 100,
                        "MELON" => 80,
                        _ => return false,
                    },
                };
                cost = cost.saturating_add(q.saturating_mul(price));
            }
            _ => (),
        }
    }
    cost as f64 <= num(&farm(obs)["money"])
}
fn demand(cs: &[Value]) -> i64 {
    cs.iter()
        .filter(|c| is(c, "PICKUP", "WHEAT"))
        .map(amount)
        .sum()
}
pub(super) fn supply(obs: &Value, mut action: Value, core: &Core) -> Value {
    let step = int(&obs["step"]);
    if !(144..695).contains(&step) {
        return action;
    }
    let future = tape_action(route(core, obs), step + 1);
    let following = tape_action(route(core, obs), step + 2);
    let cs = commands(&future);
    let next_orders = orders(&future);
    let mut prefund = 0;
    if next_orders.len() == 10
        && !next_orders
            .iter()
            .any(|o| is(o, "BUY_PRODUCT", "WHEAT") || is(o, "SELL", "WHEAT"))
    {
        let later = commands(&following);
        if demand(&later) > 0 {
            prefund = demand(&cs) + demand(&later);
        }
    }
    if prefund == 0 && !cs.iter().any(|c| is(c, "PICKUP", "WHEAT")) {
        return action;
    }
    let original_orders = orders(&action);
    if original_orders.len() > 10 || !budget(obs, &original_orders) {
        return action;
    }
    let Ok((f, private)) = apply_units(obs, &action, false) else {
        return action;
    };
    let night = step % 24 == 23;
    let mut ps = vec![position(&f["farmer"])];
    ps.extend(arr(&f["hands"]).iter().map(position));
    if night {
        ps = vec![(4, 4)];
    } else {
        for o in &original_orders {
            if op(o) == "HIRE" {
                spawn(&mut ps);
            }
        }
    }
    let need = prefund.max(
        ps.iter()
            .zip(&cs)
            .filter(|(p, c)| ACCESS.contains(p) && is(c, "PICKUP", "WHEAT"))
            .map(|(_, c)| amount(c))
            .sum(),
    );
    if need == 0 {
        return action;
    }
    let (original_stock, original_buys, _) = market_stock(&private["shed"], &original_orders);
    let (original_final, original_loss) = delivery(original_stock, &private, night);
    if int(&original_final["WHEAT"]) >= need {
        return action;
    }
    let project = |candidate: &[Value]| {
        let (stock, buys, sales) = market_stock(&private["shed"], candidate);
        let (final_stock, loss) = delivery(stock, &private, night);
        let safe = original_buys
            .iter()
            .all(|(i, q)| buys.get(i).copied().unwrap_or(0) >= *q)
            && loss
                .as_object()
                .unwrap()
                .iter()
                .all(|(item, q)| int(q) <= int(&original_loss[item]));
        (final_stock, sales, safe)
    };
    let mut proposed = original_orders.clone();
    for index in (0..proposed.len()).rev() {
        if !is(&proposed[index], "SELL", "WHEAT") {
            continue;
        }
        let (final_stock, sales, _) = project(&proposed);
        let shortage = (need - int(&final_stock["WHEAT"])).max(0);
        if shortage == 0 {
            break;
        }
        let sold = sales.get(&index).copied().unwrap_or(0);
        if sold == 0 {
            continue;
        }
        let old = proposed[index][2].clone();
        proposed[index][2] = json!((sold - shortage).max(0));
        let (after, _, safe) = project(&proposed);
        if !safe || int(&after["WHEAT"]) <= int(&final_stock["WHEAT"]) {
            proposed[index][2] = old;
        }
    }
    let (final_stock, _, _) = project(&proposed);
    let shortage = (need - int(&final_stock["WHEAT"])).max(0);
    if shortage > 0 {
        let last_sale = proposed
            .iter()
            .rposition(|o| is(o, "SELL", "WHEAT"))
            .map(|i| i as i64)
            .unwrap_or(-1);
        let index = (last_sale + 1..proposed.len() as i64)
            .rev()
            .find(|i| is(&proposed[*i as usize], "BUY_PRODUCT", "WHEAT"));
        if let Some(i) = index {
            let o = &mut proposed[i as usize];
            o[2] = json!(amount(o) + shortage);
        } else if proposed.len() < 10 {
            proposed.push(json!(["BUY_PRODUCT", "WHEAT", shortage]));
        } else {
            return action;
        }
    }
    let (final_stock, _, safe) = project(&proposed);
    if !safe || int(&final_stock["WHEAT"]) < need || !budget(obs, &proposed) {
        return action;
    }
    set_orders(&mut action, proposed);
    action
}
