// SPDX-License-Identifier: Apache-2.0
// Native translation of Ahmed Berat Ozer V39 / Dmitrii Gluzdov E182 terminal
// rescue, and the embedded Kaggle 1.32.7 deterministic unit/decay model.
// This is deliberately the source's own-state model, not the full game engine.
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde_json::{Value, json};

use super::common::{PRODUCTS, int, num, position, truth};
use super::core::{Core, shop_terminal};

const START: i64 = 712;
const FINAL: i64 = 718;
type Result<T> = std::result::Result<T, &'static str>;

fn array(v: &Value) -> Vec<Value> {
    v.as_array().cloned().unwrap_or_default()
}
fn quantity(v: &Value, item: &str) -> i64 {
    int(&v[item])
}
fn total(v: &Value) -> i64 {
    v.as_object()
        .map(|m| m.values().map(int).sum())
        .unwrap_or(0)
}
fn increment(v: &mut Value, item: &str, n: i64) {
    let q = quantity(v, item) + n;
    v[item] = json!(q);
}
fn remove(v: &mut Value, key: &str) {
    if let Some(m) = v.as_object_mut() {
        m.shift_remove(key);
    }
}
fn take(v: &mut Value, item: &str, n: i64) -> bool {
    if quantity(v, item) < n {
        return false;
    }
    increment(v, item, -n);
    if quantity(v, item) == 0 {
        remove(v, item);
    }
    true
}
fn with_default(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).map(int).unwrap_or(default)
}
fn commands(action: &Value, n: usize) -> Vec<Value> {
    let mut commands = vec![action.get("farmer").cloned().unwrap_or(json!(["PASS"]))];
    commands.extend(array(&action["hands"]));
    commands.truncate(n);
    commands.resize(n, json!(["PASS"]));
    commands
}
fn physical(obs: &Value) -> Result<(Value, Value)> {
    let seat = int(&obs["player"]) as usize;
    let mut farm = obs["farms"].get(seat).cloned().ok_or("missing own farm")?;
    if !farm.is_object() || !obs["private"].is_object() {
        return Err("missing physical state");
    }
    remove(&mut farm, "money");
    Ok((farm, obs["private"].clone()))
}
fn actor_position(farm: &Value, actor: usize) -> Value {
    if actor == 0 {
        farm["farmer"].clone()
    } else {
        farm["hands"][actor - 1].clone()
    }
}
fn shed_adjacent(pos: (i64, i64), size: i64) -> bool {
    let h = size / 2;
    [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)].contains(&pos)
}
fn crop_data(crop: &str) -> Option<(i64, i64, i64, bool)> {
    match crop {
        "WHEAT" => Some((2, 4, 6, false)),
        "CARROT" => Some((2, 3, 4, false)),
        "TOMATO" => Some((8, 8, 4, true)),
        "STRAWBERRY" => Some((10, 10, 4, true)),
        "MELON" => Some((10, 12, 6, false)),
        _ => None,
    }
}
fn animal_data(animal: &str) -> Option<(&'static str, &'static str)> {
    match animal {
        "GOOSE" => Some(("COOP", "EGG")),
        "COW" => Some(("PASTURE", "MILK")),
        "SHEEP" => Some(("PASTURE", "WOOL")),
        _ => None,
    }
}

/// The embedded model has no market RNG, night refresh or interpreter callback.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_unit_action(
    farm: &mut Value,
    private: &mut Value,
    actor: usize,
    action: &Value,
    size: i64,
    day: i64,
    turns: i64,
    cap: i64,
) -> Result<()> {
    let Some(a) = action.as_array() else {
        return Ok(());
    };
    let Some(op) = a.first().and_then(Value::as_str) else {
        return Ok(());
    };
    let pos = actor_position(farm, actor);
    if pos.is_null() {
        return Ok(());
    }
    let (x, y) = position(&pos);
    while private["inventories"]
        .as_array()
        .ok_or("inventory list")?
        .len()
        <= actor
    {
        private["inventories"]
            .as_array_mut()
            .unwrap()
            .push(json!({}));
    }
    let movement = match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    };
    if let Some((dx, dy)) = movement {
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && ny >= 0 && nx < size && ny < size {
            if actor == 0 {
                farm["farmer"] = json!([nx, ny]);
            } else {
                farm["hands"][actor - 1] = json!([nx, ny]);
            }
        }
        return Ok(());
    }
    if op == "PASS" {
        return Ok(());
    }
    if x < 0 || y < 0 {
        return Err("invalid tile coordinate");
    }
    let (x, y) = (x as usize, y as usize);
    let mut tile = farm["tiles"]
        .get(y)
        .and_then(|r| r.get(x))
        .cloned()
        .ok_or("invalid tile coordinate")?;
    let adjacent = shed_adjacent((x as i64, y as i64), size);
    let item = a.get(1).and_then(Value::as_str).unwrap_or("");
    let mut inv = private["inventories"][actor].clone();
    match op {
        "DROP" => {
            if !adjacent {
                return Ok(());
            }
            for (item, n) in inv.as_object().ok_or("inventory object")?.clone() {
                let n = int(&n);
                if n > 0 {
                    let q = n.min((cap - total(&private["shed"])).max(0));
                    if q > 0 {
                        increment(&mut private["shed"], &item, q);
                    }
                }
                remove(&mut inv, &item);
            }
        }
        "PICKUP" => {
            if !adjacent || item.is_empty() {
                return Ok(());
            }
            let n = a
                .get(2)
                .map(int)
                .unwrap_or(1)
                .min(quantity(&private["shed"], item));
            if n <= 0 {
                return Ok(());
            }
            increment(&mut private["shed"], item, -n);
            increment(&mut inv, item, n);
        }
        "PLACE" => {
            if item.is_empty() {
                return Ok(());
            }
            if let Some((structure, _)) = animal_data(item)
                && tile["kind"] == structure
                && tile.get("animal").is_none()
            {
                if take(&mut inv, item, 1) {
                    tile = json!({"kind":structure,"animal":item,"placed_day":day,"yield_units":0,"consecutive_unfed":0,
                            "fed_today":false,"cared_today":false,"fertilizer_available":false,"pending_care_bonus":0});
                }
                private["inventories"][actor] = inv;
                farm["tiles"][y][x] = tile;
                return Ok(());
            }
            if adjacent {
                let n = a
                    .get(2)
                    .map(int)
                    .unwrap_or(1)
                    .min(quantity(&inv, item))
                    .min((cap - total(&private["shed"])).max(0));
                if n <= 0 {
                    return Ok(());
                }
                take(&mut inv, item, n);
                increment(&mut private["shed"], item, n);
            }
        }
        _ if tile == "LOCKED" => return Ok(()),
        "PLANT" => {
            let Some((_, max_day, _, ongoing)) = crop_data(item) else {
                return Ok(());
            };
            if !tile.is_null() || quantity(&private["seeds"], item) <= 0 {
                return Ok(());
            }
            increment(&mut private["seeds"], item, -1);
            tile = json!({"kind":"PLANT","crop":item,"planted_day":day,"watered_today":false,
                "consecutive_unwatered":1,"yield_units":if ongoing {0}else{1},
                "max_lifespan_step":if ongoing {-1}else{(day+max_day+1)*turns},"fertilized_until_day":-1});
        }
        "WATER" => {
            if tile["kind"] != "PLANT" || truth(&tile["watered_today"]) {
                return Ok(());
            }
            let (_, max_day, max_yield, ongoing) =
                crop_data(tile["crop"].as_str().unwrap_or("")).ok_or("unknown plant")?;
            tile["watered_today"] = json!(true);
            let age = day - int(&tile["planted_day"]);
            if !ongoing && (max_day + 1) / 2 <= age && age <= max_day {
                let bonus = if int(&tile["fertilized_until_day"]) >= day {
                    2
                } else {
                    1
                };
                tile["yield_units"] = json!(max_yield.min(int(&tile["yield_units"]) + bonus));
            }
        }
        "HARVEST" => {
            let units = int(&tile["yield_units"]);
            if !tile.is_object() || units <= 0 {
                return Ok(());
            }
            if tile["kind"] == "PLANT" {
                let crop = tile["crop"].as_str().ok_or("missing crop")?.to_string();
                let (first, _, _, ongoing) = crop_data(&crop).ok_or("unknown crop")?;
                if day - int(&tile["planted_day"]) < first {
                    return Ok(());
                }
                tile["yield_units"] = json!(0);
                increment(&mut inv, &crop, units);
                if !ongoing {
                    tile = Value::Null;
                }
            } else if let Some(animal) = tile.get("animal") {
                let (_, product) =
                    animal_data(animal.as_str().unwrap_or("")).ok_or("unknown animal")?;
                tile["yield_units"] = json!(0);
                increment(&mut inv, product, units);
            }
        }
        "FERTILIZE" => {
            if tile["kind"] != "PLANT" || !take(&mut inv, "FERTILIZER", 1) {
                return Ok(());
            }
            tile["fertilized_until_day"] =
                json!(with_default(&tile, "fertilized_until_day", -1).max(day + 2));
        }
        "DIG" => {
            if tile.is_null() || tile.get("animal").is_some() {
                return Ok(());
            }
            tile = Value::Null;
        }
        "BUILD_COOP" | "BUILD_PASTURE" => {
            if !tile.is_null() {
                return Ok(());
            }
            tile = json!({"kind":if op == "BUILD_COOP" {"COOP"}else{"PASTURE"}});
        }
        "FEED" => {
            if tile.get("animal").is_none()
                || truth(&tile["fed_today"])
                || !take(&mut inv, "WHEAT", 1)
            {
                return Ok(());
            }
            tile["fed_today"] = json!(true);
        }
        "COLLECT_FERTILIZER" => {
            if tile.get("animal").is_none() || !truth(&tile["fertilizer_available"]) {
                return Ok(());
            }
            tile["fertilizer_available"] = json!(false);
            increment(&mut inv, "FERTILIZER", 1);
        }
        "CARE" => {
            if tile.get("animal").is_none() || truth(&tile["cared_today"]) {
                return Ok(());
            }
            tile["cared_today"] = json!(true);
        }
        _ => return Ok(()),
    }
    private["inventories"][actor] = inv;
    farm["tiles"][y][x] = tile;
    Ok(())
}
fn decay(farm: &mut Value, step: i64) -> Result<()> {
    for row in farm["tiles"].as_array_mut().ok_or("tile rows")? {
        for tile in row.as_array_mut().ok_or("tile row")? {
            if tile["kind"] != "PLANT" {
                continue;
            }
            let lifespan = tile
                .get("max_lifespan_step")
                .map(int)
                .ok_or("plant lifespan")?;
            if lifespan < 0 || step < lifespan || (step - lifespan) % 2 != 0 {
                continue;
            }
            tile["yield_units"] = json!(int(&tile["yield_units"]) - 1);
            if int(&tile["yield_units"]) <= 0 {
                *tile = json!({"kind":"WEED"});
            }
        }
    }
    Ok(())
}
fn settings(config: &Value) -> Result<(i64, i64, i64, usize)> {
    let size = with_default(config, "boardSize", 10);
    let turns = with_default(config, "turnsPerDay", 24);
    let cap = with_default(config, "shedCapacity", 100);
    let orders = with_default(config, "maxMarketOrdersPerTurn", 10).min(10);
    // These three source fields are compared before integer coercion. In
    // particular, "10" and 10.5 must not silently admit the pinned window.
    if [
        ("boardSize", 10.0),
        ("turnsPerDay", 24.0),
        ("episodeSteps", 720.0),
    ]
    .iter()
    .any(|(key, expected)| {
        config.get(*key).is_some_and(|value| {
            !(value.is_number() || value.is_boolean()) || num(value) != *expected
        })
    }) {
        return Err("requires pinned 10x10/24/720 terminal window");
    }
    if cap < 1 || orders < 1 {
        return Err("invalid capacity/order limit");
    }
    Ok((size, turns, cap, orders as usize))
}
fn validate(schedule: &[Value], orders: usize) -> Result<()> {
    const OPS: [&str; 18] = [
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "PASS",
        "DROP",
        "PICKUP",
        "PLACE",
        "PLANT",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "CARE",
        "COLLECT_FERTILIZER",
    ];
    for action in schedule {
        let object = action.as_object().ok_or("unknown action shape")?;
        if object
            .keys()
            .any(|k| !["farmer", "hands", "market"].contains(&k.as_str()))
        {
            return Err("unknown action shape");
        }
        if action.get("hands").is_some_and(|v| !v.is_array()) {
            return Err("hands must be a list");
        }
        let mut all = vec![action.get("farmer").cloned().unwrap_or(json!(["PASS"]))];
        all.extend(array(&action["hands"]));
        for command in all {
            let a = command.as_array().ok_or("malformed command")?;
            let op = a
                .first()
                .and_then(Value::as_str)
                .ok_or("missing operation")?;
            if !OPS.contains(&op) {
                return Err("unknown unit operation");
            }
            if ["PICKUP", "PLACE", "PLANT"].contains(&op) {
                let item = a.get(1).and_then(Value::as_str).ok_or("missing item")?;
                if !PRODUCTS.contains(&item) && animal_data(item).is_none() {
                    return Err("unknown unit item");
                }
                if a.get(2)
                    .is_some_and(|v| !v.is_i64() && !v.is_u64() && !v.is_boolean())
                {
                    return Err("noninteger unit quantity");
                }
            }
        }
        let market = action.get("market").cloned().unwrap_or(json!([]));
        let market = market.as_array().ok_or("market shape")?;
        if market.len() > orders {
            return Err("market order cap");
        }
        for order in market {
            let o = order.as_array().ok_or("market order shape")?;
            if o.len() != 3
                || o[0] != "SELL"
                || !PRODUCTS.contains(&o[1].as_str().unwrap_or(""))
                || !(o[2].is_i64() || o[2].is_u64() || o[2].is_boolean())
                || int(&o[2]) <= 0
            {
                return Err("baseline market must contain positive integer SELL only");
            }
        }
    }
    Ok(())
}
fn parent_liquidate(farm: &Value, private: &Value, prices: &Value) -> Value {
    shop_terminal(
        &json!({"step":718,"player":0,"farms":[farm],"private":private,"market":{"prices":prices}}),
        json!({"farmer":["PASS"],"hands":[],"market":[]}),
    )
}
fn positive_delta(after: &Value, before: &Value) -> Value {
    let mut delta = json!({});
    if let Some(map) = after.as_object() {
        for (item, amount) in map {
            let n = int(amount) - quantity(before, item);
            if n > 0 {
                delta[item] = json!(n);
            }
        }
    }
    delta
}
fn accumulate(target: &mut Value, delta: &Value) {
    if let Some(map) = delta.as_object() {
        for (item, n) in map {
            increment(target, item, int(n));
        }
    }
}

pub(super) fn simulate(
    obs: &Value,
    config: &Value,
    schedule: &[Value],
    final_liquidate: bool,
    detailed: bool,
    preserve_final_commands: bool,
) -> Result<Value> {
    let (size, turns, cap, order_cap) = settings(config)?;
    let step = with_default(obs, "step", -1);
    if step < START || schedule.is_empty() || step + schedule.len() as i64 - 1 > FINAL {
        return Err("outside terminal window");
    }
    if (step..step + schedule.len() as i64).any(|t| (t + 1) % turns == 0) {
        return Err("day boundary");
    }
    let (mut farm, mut private) = physical(obs)?;
    let n = 1 + farm["hands"].as_array().ok_or("hands list")?.len();
    if private["inventories"]
        .as_array()
        .ok_or("inventories list")?
        .len()
        != n
        || n > 32
    {
        return Err("invalid worker inventory shape");
    }
    validate(schedule, order_cap)?;
    let mut deposited = vec![json!({}); n];
    let mut sold = json!({});
    let (mut snapshots, mut rows, mut events) = (Vec::new(), Vec::new(), Vec::new());
    let mut executed = schedule.to_vec();
    let mut overflow = 0i64;
    for (offset, action) in executed.iter_mut().enumerate() {
        let t = step + offset as i64;
        if detailed {
            snapshots.push(json!([farm, private]));
        }
        if t == FINAL && !preserve_final_commands {
            *action = parent_liquidate(&farm, &private, &obs["market"]["prices"]);
        }
        let mut all = vec![action.get("farmer").cloned().unwrap_or(json!(["PASS"]))];
        all.extend(array(&action["hands"]));
        let mut demand = BTreeMap::<String, i64>::new();
        for command in all {
            if command[0] == "PLANT" {
                *demand
                    .entry(command[1].as_str().unwrap_or("").into())
                    .or_default() += 1;
            }
        }
        let blocked: BTreeSet<String> = demand
            .into_iter()
            .filter_map(|(k, n)| {
                if n > quantity(&private["seeds"], &k) {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();
        for (actor, mut command) in commands(action, n).into_iter().enumerate() {
            if command[0] == "PLANT" && blocked.contains(command[1].as_str().unwrap_or("")) {
                command = json!(["PASS"]);
            }
            let op = command[0].as_str().unwrap_or("");
            let xy = actor_position(&farm, actor);
            let before_inv = private["inventories"][actor].clone();
            let before_shed = private["shed"].clone();
            apply_unit_action(
                &mut farm,
                &mut private,
                actor,
                &command,
                size,
                t / turns,
                turns,
                cap,
            )?;
            if ["DROP", "PLACE"].contains(&op) {
                let delta = positive_delta(&private["shed"], &before_shed);
                accumulate(&mut deposited[actor], &delta);
                if truth(&delta) {
                    events.push(
                        json!({"offset":offset,"actor":actor,"op":op,"xy":xy,"deposited":delta}),
                    );
                }
                if op == "DROP" {
                    for (item, amount) in before_inv.as_object().ok_or("inventory object")? {
                        overflow += (int(amount)
                            - quantity(&private["inventories"][actor], item)
                            - quantity(&delta, item))
                        .max(0);
                    }
                }
            }
            if ["HARVEST", "COLLECT_FERTILIZER"].contains(&op) {
                let delta = positive_delta(&private["inventories"][actor], &before_inv);
                if truth(&delta) {
                    events.push(
                        json!({"offset":offset,"actor":actor,"op":op,"xy":xy,"acquired":delta}),
                    );
                }
            }
        }
        let pre_market = private["shed"].clone();
        if t == FINAL && preserve_final_commands && final_liquidate {
            let mut market: Vec<Value> = PRODUCTS
                .iter()
                .filter_map(|item| {
                    let q = quantity(&pre_market, item);
                    if q > 0 {
                        Some(json!(["SELL", item, q]))
                    } else {
                        None
                    }
                })
                .collect();
            if market.len() > order_cap {
                return Err("actual final stock exceeds order slots");
            }
            market.sort_by_key(|o| {
                -int(&obs["market"]["prices"][o[1].as_str().unwrap()]) * int(&o[2])
            });
            action["market"] = json!(market);
        }
        for order in array(&action["market"]) {
            let item = order[1].as_str().ok_or("market item")?;
            let q = int(&order[2])
                .min(quantity(&private["shed"], item))
                .min(99999);
            if q > 0 {
                increment(&mut private["shed"], item, -q);
                increment(&mut sold, item, q);
            }
        }
        decay(&mut farm, t)?;
        rows.push(json!({"pre_market_shed":pre_market,"post_market_shed":private["shed"],"deposited_by_actor":deposited,"sold":sold}));
    }
    if detailed {
        snapshots.push(json!([farm, private]));
    }
    Ok(
        json!({"rows":rows,"states":snapshots,"events":events,"actions":executed,"overflow_units":overflow,"farm":farm,"private":private,"sold":sold}),
    )
}

pub(super) fn clone_state(farm: &Value, private: &Value) -> (Value, Value) {
    (farm.clone(), private.clone())
}
fn ge(left: &Value, right: &Value) -> bool {
    right
        .as_object()
        .is_some_and(|m| m.iter().all(|(k, v)| quantity(left, k) >= int(v)))
}
fn dominates(candidate: &Value, baseline: &Value) -> bool {
    if truth(&candidate["overflow_units"]) {
        return false;
    }
    array(&candidate["rows"])
        .iter()
        .zip(array(&baseline["rows"]))
        .all(|(new, old)| {
            ge(&new["pre_market_shed"], &old["pre_market_shed"])
                && ge(&new["sold"], &old["sold"])
                && array(&new["deposited_by_actor"])
                    .iter()
                    .zip(array(&old["deposited_by_actor"]))
                    .all(|(a, b)| ge(a, &b))
        })
}
fn value(run: &Value, prices: &Value) -> f64 {
    PRODUCTS
        .iter()
        .map(|item| {
            (quantity(&run["sold"], item) + quantity(&run["private"]["shed"], item)) as f64
                * num(&prices[*item])
        })
        .sum()
}
fn walk(start: (i64, i64), end: (i64, i64)) -> Vec<Value> {
    let mut path = Vec::new();
    for (n, op) in [
        (end.0 - start.0, "EAST"),
        (start.0 - end.0, "WEST"),
        (end.1 - start.1, "SOUTH"),
        (start.1 - end.1, "NORTH"),
    ] {
        for _ in 0..n.max(0) {
            path.push(json!([op]));
        }
    }
    path
}
fn return_route(pos: (i64, i64)) -> Vec<Value> {
    let targets = [(4, 4), (5, 4), (4, 5), (5, 5)];
    let target = *targets
        .iter()
        .min_by_key(|xy| (pos.0 - xy.0).abs() + (pos.1 - xy.1).abs())
        .unwrap();
    let mut path = walk(pos, target);
    path.push(json!(["DROP"]));
    path
}
#[derive(Clone)]
struct Bundle {
    xy: (i64, i64),
    operations: Vec<Value>,
    value: f64,
    distance: usize,
}
#[derive(Clone, PartialEq)]
struct Proposal {
    value: f64,
    offset: usize,
    route: Vec<Value>,
    bundles: usize,
}
fn route_key(route: &[Value]) -> Vec<Vec<String>> {
    route
        .iter()
        .map(|command| {
            array(command)
                .iter()
                .map(|v| v.as_str().unwrap_or("").into())
                .collect()
        })
        .collect()
}
fn proposals(
    run: &Value,
    actor: usize,
    prices: &Value,
    max_per_actor: usize,
) -> Result<Vec<Proposal>> {
    let mut owners: BTreeMap<((i64, i64), String), BTreeSet<usize>> = BTreeMap::new();
    for event in array(&run["events"]) {
        if event.get("acquired").is_some() {
            owners
                .entry((
                    position(&event["xy"]),
                    event["op"].as_str().unwrap_or("").into(),
                ))
                .or_default()
                .insert(int(&event["actor"]) as usize);
        }
    }
    let mut proposals = Vec::new();
    let mut seen = BTreeSet::new();
    let horizon = run["rows"].as_array().ok_or("rows")?.len();
    for offset in 0..horizon {
        let farm = &run["states"][offset][0];
        let private = &run["states"][offset][1];
        let pos = position(&actor_position(farm, actor));
        let inventory = &private["inventories"][actor];
        let carried: f64 = inventory
            .as_object()
            .ok_or("inventory")?
            .iter()
            .map(|(item, n)| num(&prices[item]) * int(n) as f64)
            .sum();
        let prefix = if offset > 0 {
            run["rows"][offset - 1]["deposited_by_actor"][actor].clone()
        } else {
            json!({})
        };
        let future = &run["rows"][horizon - 1]["deposited_by_actor"][actor];
        let obligation: f64 = future
            .as_object()
            .ok_or("deposits")?
            .iter()
            .map(|(item, n)| num(&prices[item]) * (int(n) - quantity(&prefix, item)) as f64)
            .sum();
        let mut bundles = Vec::new();
        for (y, row) in array(&farm["tiles"]).iter().enumerate() {
            for (x, tile) in array(row).iter().enumerate() {
                if !tile.is_object() {
                    continue;
                }
                let xy = (x as i64, y as i64);
                let mut operations = Vec::new();
                let mut val = 0.0;
                let owned_elsewhere = |op: &str| {
                    owners
                        .get(&(xy, op.into()))
                        .is_some_and(|s| s.iter().any(|&a| a != actor))
                };
                if int(&tile["yield_units"]) > 0 {
                    let item = if tile["kind"] == "PLANT" {
                        tile["crop"].as_str()
                    } else {
                        animal_data(tile["animal"].as_str().unwrap_or("")).map(|(_, p)| p)
                    };
                    if let Some(item) = item {
                        let mature = tile.get("animal").is_some()
                            || (START + offset as i64) / 24 - int(&tile["planted_day"])
                                >= crop_data(item).ok_or("unknown crop")?.0;
                        if mature && !owned_elsewhere("HARVEST") {
                            operations.push(json!(["HARVEST"]));
                            val += num(&prices[item]) * int(&tile["yield_units"]) as f64;
                        }
                    }
                }
                if truth(&tile["fertilizer_available"])
                    && tile.get("animal").is_some()
                    && !owned_elsewhere("COLLECT_FERTILIZER")
                {
                    operations.push(json!(["COLLECT_FERTILIZER"]));
                    val += num(&prices["FERTILIZER"]);
                }
                if !operations.is_empty() {
                    let distance = walk(pos, xy).len() + operations.len() + return_route(xy).len();
                    if distance <= horizon - offset {
                        bundles.push(Bundle {
                            xy,
                            operations,
                            value: val,
                            distance,
                        });
                    }
                }
            }
        }
        bundles.sort_by(|a, b| {
            (-a.value / a.distance as f64)
                .total_cmp(&(-b.value / b.distance as f64))
                .then_with(|| (-a.value).total_cmp(&(-b.value)))
                .then(a.xy.cmp(&b.xy))
        });
        let mut variants: Vec<(Vec<Bundle>, f64)> = if carried != 0.0 {
            vec![(vec![], carried)]
        } else {
            vec![]
        };
        for bundle in bundles.iter().take(6) {
            variants.push((vec![bundle.clone()], carried + bundle.value));
        }
        for first in bundles.iter().take(3) {
            for second in bundles.iter().take(3) {
                if first.xy != second.xy {
                    variants.push((
                        vec![first.clone(), second.clone()],
                        carried + first.value + second.value,
                    ));
                }
            }
        }
        for (stops, val) in variants {
            let mut route = Vec::new();
            let mut cursor = pos;
            for bundle in &stops {
                route.extend(walk(cursor, bundle.xy));
                route.extend(bundle.operations.clone());
                cursor = bundle.xy;
            }
            route.extend(return_route(cursor));
            if route.len() > horizon - offset {
                continue;
            }
            route.resize(horizon - offset, json!(["PASS"]));
            if seen.insert((offset, route_key(&route))) {
                proposals.push(Proposal {
                    value: val - obligation,
                    offset,
                    route,
                    bundles: stops.len(),
                });
            }
        }
    }
    proposals.sort_by(|a, b| {
        (-a.value)
            .total_cmp(&(-b.value))
            .then(a.offset.cmp(&b.offset))
            .then_with(|| route_key(&a.route).cmp(&route_key(&b.route)))
    });
    let direct: Vec<_> = proposals
        .iter()
        .filter(|p| p.bundles == 0 && p.value > 0.0)
        .take(2)
        .cloned()
        .collect();
    let mut chosen = direct.clone();
    chosen.extend(proposals.into_iter().filter(|p| !direct.contains(p)));
    chosen.truncate(max_per_actor);
    Ok(chosen)
}
type AcquisitionKey = ((i64, i64), String, usize);
fn acquisition_key(event: &Value) -> AcquisitionKey {
    (
        position(&event["xy"]),
        event["op"].as_str().unwrap_or("").into(),
        int(&event["actor"]) as usize,
    )
}
fn plan_inner(
    obs: &Value,
    config: &Value,
    baseline_remaining: &[Value],
    max_simulations: usize,
    passes: usize,
    proposals_per_actor: usize,
) -> Result<Value> {
    if int(&obs["step"]) != START || baseline_remaining.len() != 7 {
        return Err("planning requires step 712 and exactly seven actions through 718");
    }
    let max_simulations = max_simulations.clamp(1, 256);
    let passes = passes.clamp(1, 2);
    let proposals_per_actor = proposals_per_actor.clamp(1, 16);
    let baseline = simulate(obs, config, baseline_remaining, false, true, false)?;
    let mut prices = json!({});
    for item in PRODUCTS {
        prices[item] = json!(
            obs["market"]["prices"]
                .get(item)
                .map(num)
                .unwrap_or(1.0)
                .max(1.0)
        );
    }
    let mut current = baseline_remaining.to_vec();
    let mut best = baseline.clone();
    let baseline_value = value(&baseline, &prices);
    let mut best_value = baseline_value;
    let mut changes = Vec::new();
    let mut simulations = 0;
    let n = best["private"]["inventories"]
        .as_array()
        .ok_or("inventories")?
        .len();
    for sweep in 0..passes {
        let mut improved = false;
        for actor in 0..n {
            let mut winner = None;
            for proposal in proposals(&best, actor, &prices, proposals_per_actor)? {
                if simulations >= max_simulations {
                    break;
                }
                let mut trial = current.clone();
                for (i, command) in proposal.route.iter().enumerate() {
                    let action = &mut trial[proposal.offset + i];
                    if actor == 0 {
                        action["farmer"] = command.clone();
                    } else {
                        if action.get("hands").is_none() {
                            action["hands"] = json!([]);
                        }
                        let hands = action["hands"].as_array_mut().ok_or("hands")?;
                        while hands.len() < n - 1 {
                            hands.push(json!(["PASS"]));
                        }
                        hands[actor - 1] = command.clone();
                    }
                }
                let evaluated = simulate(obs, config, &trial, false, false, false)?;
                simulations += 1;
                let score = value(&evaluated, &prices);
                if score > best_value && dominates(&evaluated, &baseline) {
                    let mut required = BTreeMap::new();
                    for e in array(&best["events"]) {
                        if e.get("acquired").is_some() && int(&e["actor"]) as usize != actor {
                            // Python dict comprehension retains the last event per key.
                            required.insert(acquisition_key(&e), e["acquired"].clone());
                        }
                    }
                    let mut acquired = BTreeMap::new();
                    for e in array(&evaluated["events"]) {
                        if e.get("acquired").is_some() {
                            accumulate(
                                acquired.entry(acquisition_key(&e)).or_insert(json!({})),
                                &e["acquired"],
                            );
                        }
                    }
                    if required
                        .iter()
                        .all(|(k, v)| ge(acquired.get(k).unwrap_or(&json!({})), v))
                    {
                        winner = Some((trial, proposal.offset, proposal.bundles));
                        best_value = score;
                    }
                }
            }
            if let Some((trial, offset, bundles)) = winner {
                current = trial;
                best = simulate(obs, config, &current, false, true, false)?;
                changes.push(json!({"pass":sweep,"actor":actor,"from_step":START+offset as i64,"resource_bundles":bundles,"estimated_stock_value":best_value}));
                improved = true;
            }
            if simulations >= max_simulations {
                break;
            }
        }
        if !improved || simulations >= max_simulations {
            break;
        }
    }
    if changes.is_empty() || best_value <= baseline_value {
        let mut delta = json!({});
        for item in PRODUCTS {
            delta[item] = json!(0);
        }
        return Ok(
            json!({"accepted":false,"reason":"no positive physical delivery gain","actions":null,"simulations":simulations,
            "changed_workers":[],"changes":[],"certificate":{"stock_value_gain_at_initial_prices":0,"sold_unit_delta":delta}}),
        );
    }
    let final_run = simulate(obs, config, &current, true, true, false)?;
    let physical_run = simulate(obs, config, &current, false, false, false)?;
    if !dominates(&physical_run, &baseline) {
        return Err("no zero-overflow dominating continuation");
    }
    let mut delta = json!({});
    for item in PRODUCTS {
        delta[item] = json!(quantity(&final_run["sold"], item) - quantity(&baseline["sold"], item));
    }
    let last = final_run["rows"].as_array().unwrap().len() - 1;
    let deposited_gain = (0..n).any(|actor| {
        PRODUCTS.iter().any(|item| {
            quantity(&final_run["rows"][last]["deposited_by_actor"][actor], item)
                > quantity(&baseline["rows"][last]["deposited_by_actor"][actor], item)
        })
    });
    let worker_change = array(&final_run["actions"])
        .iter()
        .zip(array(&baseline["actions"]))
        .any(|(new, old)| commands(new, n) != commands(&old, n));
    let accepted = worker_change
        && deposited_gain
        && delta.as_object().unwrap().values().any(|v| int(v) > 0)
        && delta.as_object().unwrap().values().all(|v| int(v) >= 0);
    let mut states = array(&final_run["states"]);
    states.pop();
    let changed_workers: BTreeSet<_> = changes.iter().map(|c| int(&c["actor"])).collect();
    Ok(
        json!({"accepted":accepted,"reason":if accepted{"joint physical dominance"}else{"no improvement"},
        "baseline":baseline_remaining,"actions":final_run["actions"],"expected_states":states,"simulations":simulations,"changes":changes,
        "abandoned":false,"changed_workers":changed_workers,"certificate":{"baseline_rows":baseline["rows"],"physical_rows":physical_run["rows"],
        "baseline_overflow":baseline["overflow_units"],"candidate_overflow":final_run["overflow_units"],"sold_unit_delta":delta,
        "stock_value_gain_at_initial_prices":best_value-baseline_value,"baseline_final_shed":baseline["private"]["shed"],"final_shed":final_run["private"]["shed"],
        "positive_physical_deposit_gain":deposited_gain,"markets_712_717_unchanged":(0..6).all(|i|final_run["actions"][i]["market"]==baseline_remaining[i]["market"])}}),
    )
}
pub(super) fn plan_terminal(
    obs: &Value,
    config: &Value,
    baseline: &[Value],
    max_simulations: usize,
    passes: usize,
    proposals_per_actor: usize,
) -> Value {
    let begun = Instant::now();
    let mut plan = plan_inner(
        obs,
        config,
        baseline,
        max_simulations,
        passes,
        proposals_per_actor,
    )
    .unwrap_or_else(
        |reason| json!({"accepted":false,"reason":reason,"actions":null,"simulations":0}),
    );
    plan["planning_ms"] = json!(begun.elapsed().as_secs_f64() * 1000.0);
    plan
}
fn recover_observed(
    obs: &Value,
    config: &Value,
    parent: &Value,
    plan: &mut Value,
) -> Result<Value> {
    let (farm, private) = physical(obs)?;
    let mut positions = vec![farm["farmer"].clone()];
    positions.extend(array(&farm["hands"]));
    let remaining = FINAL - int(&obs["step"]) + 1;
    let mut room = (with_default(config, "shedCapacity", 100) - total(&private["shed"])).max(0);
    let mut units = Vec::new();
    let mut problems = Vec::new();
    let prices = &obs["market"]["prices"];
    for (actor, (pos, inv)) in positions
        .iter()
        .zip(array(&private["inventories"]))
        .enumerate()
    {
        let mut command = json!(["PASS"]);
        if inv
            .as_object()
            .ok_or("inventory")?
            .values()
            .any(|q| int(q) > 0)
        {
            let route = return_route(position(pos));
            let count: i64 = inv
                .as_object()
                .unwrap()
                .values()
                .map(|q| int(q).max(0))
                .sum();
            if route.len() as i64 > remaining {
                problems.push(json!({"actor":actor,"reason":"unreachable cargo"}));
            } else if route.len() > 1 {
                command = route[0].clone();
            } else if count <= room {
                command = json!(["DROP"]);
                room -= count;
            } else {
                let items: Vec<_> = PRODUCTS
                    .iter()
                    .copied()
                    .filter(|item| quantity(&inv, item) > 0)
                    .collect();
                if room > 0 && !items.is_empty() {
                    // max() is stable on Python's first equal key, hence explicit index tie break.
                    let item = *items
                        .iter()
                        .max_by(|a, b| {
                            let va = prices.get(**a).map(num).unwrap_or(1.0)
                                * quantity(&inv, a).min(room) as f64;
                            let vb = prices.get(**b).map(num).unwrap_or(1.0)
                                * quantity(&inv, b).min(room) as f64;
                            va.total_cmp(&vb).then_with(|| {
                                PRODUCTS
                                    .iter()
                                    .position(|i| i == *b)
                                    .cmp(&PRODUCTS.iter().position(|i| i == *a))
                            })
                        })
                        .unwrap();
                    let q = quantity(&inv, item).min(room);
                    command = json!(["PLACE", item, q]);
                    room -= q;
                } else {
                    problems.push(json!({"actor":actor,"reason":"no shed capacity"}));
                }
            }
        }
        units.push(command);
    }
    if units.is_empty() {
        return Err("no farmer");
    }
    let mut action = json!({"farmer":units[0],"hands":units[1..],"market":parent.get("market").cloned().unwrap_or(json!([]))});
    if int(&obs["step"]) == FINAL {
        action["market"] = json!([]);
        action = simulate(obs, config, &[action], true, false, true)?["actions"][0].clone();
    }
    increment(plan, "recovery_steps", 1);
    if !problems.is_empty() {
        if plan.get("recovery_failures").is_none() {
            plan["recovery_failures"] = json!([]);
        }
        plan["recovery_failures"]
            .as_array_mut()
            .unwrap()
            .push(json!({"step":int(&obs["step"]),"problems":problems}));
    }
    Ok(action)
}
pub(super) fn terminal_action(
    obs: &Value,
    config: &Value,
    parent: &Value,
    plan: &mut Value,
) -> Result<Value> {
    let step = with_default(obs, "step", -1);
    if !truth(&plan["accepted"]) || !(START..=FINAL).contains(&step) {
        return Ok(parent.clone());
    }
    if truth(&plan["abandoned"]) {
        return if truth(&plan["deviated"]) {
            recover_observed(obs, config, parent, plan)
        } else {
            Ok(parent.clone())
        };
    }
    let index = (step - START) as usize;
    let (farm, private) = physical(obs)?;
    let n = 1 + farm["hands"].as_array().ok_or("hands")?.len();
    let mismatch = json!([farm, private]) != plan["expected_states"][index]
        || commands(parent, n) != commands(&plan["baseline"][index], n)
        || parent.get("market").cloned().unwrap_or(json!([]))
            != plan["baseline"][index]
                .get("market")
                .cloned()
                .unwrap_or(json!([]));
    if mismatch {
        plan["abandoned"] = json!(true);
        plan["abandon_step"] = json!(step);
        plan["reason"] = json!("physical observation or effective baseline action diverged");
        plan["safety_failure"] = json!(true);
        return if truth(&plan["deviated"]) {
            recover_observed(obs, config, parent, plan)
        } else {
            Ok(parent.clone())
        };
    }
    let result = if step == FINAL {
        parent_liquidate(&farm, &private, &obs["market"]["prices"])
    } else {
        plan["actions"][index].clone()
    };
    if commands(&result, n) != commands(parent, n) {
        plan["deviated"] = json!(true);
    }
    Ok(result)
}

#[derive(Clone, Debug)]
pub(super) struct Terminal {
    pub plans: Value,
    pub previous: Value,
    pub stats: Value,
}
impl Default for Terminal {
    fn default() -> Self {
        Self {
            plans: json!({}),
            previous: json!({}),
            stats: json!({"planning_calls":0,"accepted":0,"changed_steps":0,"aborted":0,"shadow_declines":0,"errors":0,"max_planning_ms":0.0}),
        }
    }
}
fn pre_terminal(obs: &Value, core: &mut Core) -> Value {
    let action = core.act(obs);
    shop_terminal(obs, action)
}
fn shadow_terminal(
    obs: &Value,
    config: &Value,
    core: &Core,
    committed: bool,
) -> Result<Option<(Vec<Value>, Vec<Value>)>> {
    if committed {
        return Ok(None);
    }
    let seat = int(&obs["player"]) as usize;
    let key = seat.to_string();
    let state = &core.players[&key];
    if !truth(state)
        || int(&state["last_step"]) != 711
        || int(&state["route"]) != 2
        || truth(&state["pending"])
    {
        return Ok(None);
    }
    let mut shadow = core.clone();
    if let Some(map) = shadow.diagnostics.as_object_mut() {
        for value in map.values_mut() {
            *value = json!(0);
        }
    }
    let mut projected = obs.clone();
    let mut baseline = Vec::new();
    let mut states = Vec::new();
    for step in START..=FINAL {
        projected["step"] = json!(step);
        projected["day"] = json!(step / 24);
        projected["hour"] = json!(step % 24);
        states.push(shadow.players[&key].clone());
        let mut action = shadow.act(&projected);
        if step == FINAL {
            action = parent_liquidate(
                &projected["farms"][seat],
                &projected["private"],
                &projected["market"]["prices"],
            );
        } else {
            let market = array(&action["market"]);
            let items: BTreeSet<_> = market.iter().filter_map(|o| o[1].as_str()).collect();
            if market.len() != 9
                || items != PRODUCTS.iter().copied().collect()
                || market.iter().any(|o| {
                    o[0] != "SELL"
                        || o.as_array().map(Vec::len) != Some(3)
                        || !(o[2].is_i64() || o[2].is_u64())
                        || int(&o[2]) < 100
                })
            {
                return Ok(None);
            }
        }
        if shadow
            .diagnostics
            .as_object()
            .is_some_and(|m| m.values().any(truth))
        {
            return Ok(None);
        }
        let run = simulate(&projected, config, &[action.clone()], false, false, false)?;
        if run["actions"][0] != action {
            return Ok(None);
        }
        baseline.push(action);
        let mut farm = run["farm"].clone();
        farm["money"] = projected["farms"][seat]["money"].clone();
        projected["farms"][seat] = farm;
        projected["private"] = run["private"].clone();
    }
    Ok(Some((baseline, states)))
}
impl Terminal {
    pub(super) fn apply(
        &mut self,
        obs: &Value,
        config: &Value,
        core: &mut Core,
        committed: bool,
    ) -> Value {
        let Some(step) = obs.get("step").map(int) else {
            return pre_terminal(obs, core);
        };
        let Some(seat) = obs.get("player").map(int) else {
            return pre_terminal(obs, core);
        };
        let key = seat.to_string();
        let previous = self.previous.get(&key).map(int);
        if step == 0 || previous.is_some_and(|p| step <= p) {
            remove(&mut self.plans, &key);
        }
        self.previous[&key] = json!(step);
        let mut plan = self.plans[&key].clone();
        if truth(&plan["accepted"]) && (START..=FINAL).contains(&step) {
            if previous != Some(step - 1) {
                plan["abandoned"] = json!(true);
                plan["reason"] = json!("nonconsecutive callback");
            }
            let parent = plan["baseline"][(step - START) as usize].clone();
            match terminal_action(obs, config, &parent, &mut plan) {
                Ok(result) => {
                    if truth(&plan["abandoned"]) {
                        if !truth(&plan["abort_counted"]) {
                            plan["abort_counted"] = json!(true);
                            increment(&mut self.stats, "aborted", 1);
                        }
                        if !truth(&plan["deviated"]) {
                            core.players[&key] =
                                plan["parent_states_before"][(step - START) as usize].clone();
                            remove(&mut self.plans, &key);
                            return pre_terminal(obs, core);
                        }
                    }
                    if result != parent {
                        increment(&mut self.stats, "changed_steps", 1);
                    }
                    self.plans[&key] = plan;
                    return result;
                }
                Err(_) => {
                    increment(&mut self.stats, "errors", 1);
                    if truth(&plan["deviated"]) {
                        let result = recover_observed(obs, config, &parent, &mut plan)
                            .unwrap_or_else(|_| {
                                parent_liquidate(
                                    &obs["farms"][seat as usize],
                                    &obs["private"],
                                    &obs["market"]["prices"],
                                )
                            });
                        self.plans[&key] = plan;
                        return result;
                    }
                    self.plans[&key] = plan;
                }
            }
        }
        if step != START {
            return pre_terminal(obs, core);
        }
        increment(&mut self.stats, "planning_calls", 1);
        let Ok(Some((baseline, states))) = shadow_terminal(obs, config, core, committed) else {
            increment(&mut self.stats, "shadow_declines", 1);
            return pre_terminal(obs, core);
        };
        let actual = pre_terminal(obs, core);
        if actual != baseline[0] {
            increment(&mut self.stats, "shadow_declines", 1);
            return actual;
        }
        let mut plan = plan_terminal(obs, config, &baseline, 64, 1, 4);
        self.stats["max_planning_ms"] =
            json!(num(&self.stats["max_planning_ms"]).max(num(&plan["planning_ms"])));
        if !truth(&plan["accepted"]) {
            return actual;
        }
        plan["parent_states_before"] = json!(states);
        increment(&mut self.stats, "accepted", 1);
        let result = match terminal_action(obs, config, &actual, &mut plan) {
            Ok(result) => {
                if result != actual {
                    increment(&mut self.stats, "changed_steps", 1);
                }
                result
            }
            Err(_) => {
                increment(&mut self.stats, "errors", 1);
                actual
            }
        };
        self.plans[&key] = plan;
        result
    }
}
