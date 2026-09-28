//! 2945 Farm HERD2 and COWSWAP layers: species choice at the tape's goose and
//! cow purchases from a day-by-day book forecast.
//! Source: agents/farm2945/main.py (`_hd2_*`, `_cs_*` and their `agent` wrappers).
//! Float operations keep the source's order; `round` is Python's ties-to-even.
use super::race::shop_items;
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use crate::native_agents::v43::production::{market_price, similarity};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const HD2_FROM: i64 = 192;
const HD2_TO: i64 = 360;
const CS_FROM: i64 = 144;
const CS_TO: i64 = 192;
const RATIO: f64 = 1.3;
const MIN_GAIN: f64 = 600.0;
const CARE: f64 = 0.8;
const FUTURE: f64 = 0.0;

struct Spec {
    cost: i64,
    first: i64,
    interval: i64,
    per: i64,
    product: &'static str,
}

fn spec(animal: &str) -> Option<Spec> {
    Some(match animal {
        "GOOSE" => Spec {
            cost: 300,
            first: 4,
            interval: 1,
            per: 2,
            product: "EGG",
        },
        "COW" => Spec {
            cost: 400,
            first: 8,
            interval: 2,
            per: 3,
            product: "MILK",
        },
        "SHEEP" => Spec {
            cost: 500,
            first: 6,
            interval: 3,
            per: 4,
            product: "WOOL",
        },
        _ => return None,
    })
}

const SHOP_TYPES: [&str; 8] = [
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
];

fn shop_demand(products: &[&str]) -> f64 {
    6.0 * if products.len() == 1 { 2.0 } else { 1.0 }
}

/// `_hd2_schedule`: units one animal produces on each day from `day_from`.
fn schedule(animal: &str, placed_day: i64, day_from: i64) -> Vec<(i64, f64)> {
    let Some(s) = spec(animal) else {
        return vec![];
    };
    (day_from.max(placed_day + s.first)..30)
        .filter(|d| (d - placed_day - s.first) % s.interval == 0)
        .map(|d| (d, 1.0 + (s.per - 1) as f64 * CARE))
        .collect()
}

fn add(target: &mut BTreeMap<i64, f64>, units: &[(i64, f64)]) {
    for (d, u) in units {
        *target.entry(*d).or_insert(0.0) += u;
    }
}

/// `_hd2_ev`: margin value of `k` new animals of `option`.
pub fn ev(option: &str, k: i64, obs: &Value, core: &Core) -> Fallible<f64> {
    let s = spec(option).ok_or("hd2: option")?;
    let item = s.product;
    let animal_of = match item {
        "EGG" => "GOOSE",
        "MILK" => "COW",
        _ => "SHEEP",
    };
    let step = int(&obs["step"]);
    let day = step / 24;
    let who = seat(obs);
    let market = &obs["market"];
    let params = market.get("params");
    // existing supply, per farm, per day
    let mut existing = [BTreeMap::new(), BTreeMap::new()];
    for (index, farm) in array(&obs["farms"]).iter().enumerate().take(2) {
        for t in array(&farm["tiles"]).iter().flat_map(array) {
            if t.is_object() && t["animal"] == animal_of {
                let placed = t.get("placed_day").map(int).unwrap_or(day);
                add(&mut existing[index], &schedule(animal_of, placed, day + 1));
            }
        }
    }
    let held = int(&obs["private"]["shed"][animal_of])
        + array(&obs["private"]["inventories"])
            .iter()
            .map(|i| int(&i[animal_of]))
            .sum::<i64>();
    for _ in 0..held {
        add(&mut existing[who], &schedule(animal_of, day + 1, day + 1));
    }
    // Purchases the tape already plans; a family-similar rival runs the same tape.
    let native = &core.players[who.to_string()];
    if truth(native) {
        let similar = similarity(obs) >= 0.9;
        let route = int(&native["route"]);
        for t in step + 1..696 {
            for order in orders(&tape(if t >= 648 { 2 } else { route }, t)) {
                if !is_order(&order, "BUY_ANIMAL", animal_of) {
                    continue;
                }
                for _ in 0..int(&order[2]).max(0) {
                    let units = schedule(animal_of, t / 24 + 1, day + 1);
                    add(&mut existing[who], &units);
                    if similar {
                        add(&mut existing[1 - who], &units);
                    }
                }
            }
        }
    }
    let (ours, rival) = (&existing[who], &existing[1 - who]);
    let new: BTreeMap<i64, f64> = schedule(option, day + 1, day + 1)
        .into_iter()
        .map(|(d, u)| (d, u * k as f64))
        .collect();
    let shops = array(&obs["town"]["unlocked_shops"]);
    let unlocks_left = (8 - shops.len() as i64).max(0);
    let mut base_demand = 0.0;
    for shop in shops {
        let products = shop_items(text(shop));
        if products.contains(&item) {
            base_demand += shop_demand(products);
        }
    }
    base_demand += 1.0;
    let future: f64 = SHOP_TYPES
        .iter()
        .map(|shop| shop_items(shop))
        .filter(|products| products.contains(&item))
        .map(shop_demand)
        .sum::<f64>()
        / SHOP_TYPES.len() as f64;
    let extra_demand = future * FUTURE;
    let get = |m: &BTreeMap<i64, f64>, d: i64| m.get(&d).copied().unwrap_or(0.0);
    let path = |with_new: bool| {
        let mut inventory = num(&market["inventory"][item]);
        let mut prices = BTreeMap::new();
        let mut revenue = 0.0;
        for d in day + 1..30 {
            let opened = unlocks_left.min((d / 3 - day / 3).max(0));
            inventory -= base_demand + extra_demand * opened as f64;
            inventory += get(ours, d) + get(rival, d);
            prices.insert(d, market_price(item, inventory.round_ties_even(), params));
            if with_new {
                for _ in 0..get(&new, d).round_ties_even() as i64 {
                    let price = market_price(item, inventory.round_ties_even(), params);
                    revenue += price as f64;
                    if price > 1 {
                        inventory += 1.0;
                    }
                }
            }
        }
        (prices, revenue)
    };
    let (base_prices, _) = path(false);
    let (new_prices, revenue) = path(true);
    let mut swing = 0.0;
    for (d, base) in &base_prices {
        swing += (new_prices[d] - base) as f64 * (get(ours, *d) - get(rival, *d));
    }
    Ok(revenue + swing - (s.cost * k) as f64)
}

/// `",".join("%s:%d" % (o, v) for o, v in sorted(evs.items()))`.
fn ev_text(evs: &[(&str, f64)]) -> String {
    let mut sorted = evs.to_vec();
    sorted.sort_by_key(|e| e.0);
    sorted
        .iter()
        .map(|(o, v)| format!("{o}:{}", v.trunc() as i64))
        .collect::<Vec<_>>()
        .join(",")
}

fn animal_cost(item: &str) -> i64 {
    spec(item).map(|s| s.cost).unwrap_or(0)
}

fn seed_cost(item: &str) -> i64 {
    match item {
        "WHEAT" => 10,
        "CARROT" => 20,
        "TOMATO" => 50,
        "STRAWBERRY" => 100,
        "MELON" => 80,
        _ => 0,
    }
}

/// Credit of swapped-herd product sold as it reaches the shed (HERD2 and COWSWAP).
fn sell_credit(
    obs: &Value,
    action: &Value,
    market: &mut Vec<Value>,
    item: &str,
    credit: i64,
) -> i64 {
    let stock = Core::projected_shed(action, &View::new(obs));
    let planned: i64 = market
        .iter()
        .filter(|o| is_order(o, "SELL", item))
        .map(|o| int(&o[2]).max(0))
        .sum();
    let extra = credit.min((int(&stock[item]) - planned).max(0));
    if extra <= 0 {
        return 0;
    }
    if !bump_sell(market, item, extra) {
        if market.len() >= MAX_ORDERS {
            return 0;
        }
        market.insert(0, json!(["SELL", item, extra]));
    }
    extra
}

#[derive(Clone, Debug)]
pub struct Herd2 {
    pub herd: Value,
    pub cow: Value,
    pub report: Value,
}
impl Default for Herd2 {
    fn default() -> Self {
        Self {
            herd: json!({}),
            cow: json!({}),
            report: json!({"hd2_decision":"","hd2_ev":"","hd2_rewrites":0,"hd2_credit_units":0,
                "hd2_sold_units":0,"hd2_errors":0,"cs_decision":"","cs_ev":"","cs_rewrites":0,
                "cs_broken":0,"cs_credit":0,"cs_sold":0,"cs_errors":0}),
        }
    }
}

impl Herd2 {
    fn reset(&mut self, keys: &[&str]) {
        for k in keys {
            self.report[*k] = if k.ends_with("decision") || k.ends_with("_ev") {
                json!("")
            } else {
                json!(0)
            };
        }
    }

    /// HERD2: goose / cow / sheep at the tape's goose purchase.
    pub fn herd2(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        match self.herd2_apply(obs, &action, core) {
            Ok(result) => result,
            Err(_) => {
                inc(&mut self.report, "hd2_errors", 1);
                action
            }
        }
    }

    fn herd2_apply(&mut self, obs: &Value, action: &Value, core: &Core) -> Fallible<Value> {
        let step = int(&obs["step"]);
        let who = seat(obs).to_string();
        if self.herd[&who].is_null() || step <= int(&self.herd[&who]["step"]) {
            self.herd[&who] = json!({"step":-1,"inv":{},"decided":false,"mode":null,
                "pending":[],"sites":{},"credit":0});
            if step == 0 {
                self.reset(&[
                    "hd2_decision",
                    "hd2_ev",
                    "hd2_rewrites",
                    "hd2_credit_units",
                    "hd2_sold_units",
                    "hd2_errors",
                ]);
            }
        }
        self.herd[&who]["step"] = json!(step);
        let day = step / 24;
        if self.herd[&who]["inv"][day.to_string()].is_null() {
            self.herd[&who]["inv"][day.to_string()] = obs["market"]["inventory"].clone();
        }
        if truth(&self.herd[&who]["mode"]) {
            self.herd2_confirm(obs);
        } else if !truth(&self.herd[&who]["decided"]) {
            self.herd2_decide(obs, action, core)?;
        }
        if !truth(&self.herd[&who]["mode"]) {
            return Ok(action.clone());
        }
        Ok(self.herd2_rewrite(obs, action.clone()))
    }

    fn herd2_decide(&mut self, obs: &Value, action: &Value, core: &Core) -> Fallible<()> {
        let who = seat(obs).to_string();
        let buys: Vec<Value> = orders(action)
            .into_iter()
            .filter(|o| is_order(o, "BUY_ANIMAL", "GOOSE"))
            .collect();
        if buys.is_empty() {
            return Ok(());
        }
        self.herd[&who]["decided"] = json!(true);
        let step = int(&obs["step"]);
        if !(HD2_FROM..HD2_TO).contains(&step) {
            return Ok(());
        }
        let farm = own_farm(obs);
        if array(&farm["tiles"])
            .iter()
            .flat_map(array)
            .any(|t| t.is_object() && t["kind"] == "COOP")
        {
            self.report["hd2_decision"] = json!("skip:coop_exists");
            return Ok(());
        }
        let k: i64 = buys.iter().map(|o| int(&o[2]).max(0)).sum();
        let mut evs = vec![];
        for option in ["GOOSE", "COW", "SHEEP"] {
            evs.push((option, ev(option, k.max(3), obs, core)?));
        }
        self.report["hd2_ev"] = json!(ev_text(&evs));
        let goose = evs[0].1;
        // `max(_HD2_OPTIONS, key=...)`: COW unless SHEEP is strictly better.
        let best = if evs[2].1 > evs[1].1 { evs[2] } else { evs[1] };
        let gain = best.1 - goose;
        let ok_ratio = best.1 >= RATIO * goose.max(1.0);
        let extra_cost = (animal_cost(best.0) - 300) * k;
        let cash = num(&farm["money"]);
        if gain >= MIN_GAIN && ok_ratio && cash >= (300 * k + extra_cost + 50) as f64 {
            self.herd[&who]["mode"] = json!(best.0);
            self.report["hd2_decision"] = json!(format!("{}@{step}", best.0));
        } else {
            self.report["hd2_decision"] = json!(format!("keep@{step}"));
        }
        Ok(())
    }

    fn herd2_confirm(&mut self, obs: &Value) {
        let st = &mut self.herd[seat(obs).to_string()];
        let farm = own_farm(obs);
        let mut keep = vec![];
        for entry in array(&st["pending"]).to_vec() {
            let (pos, day) = (position(&entry), int(&entry[2]));
            let t = tile(farm, pos);
            if t.is_object() && t["animal"] == st["mode"] && t["placed_day"] == day {
                st["sites"][key(pos)] = json!(day);
            } else if int(&obs["step"]) / 24 <= day + 1 {
                keep.push(entry);
            }
        }
        st["pending"] = json!(keep);
    }

    fn herd2_rewrite(&mut self, obs: &Value, mut action: Value) -> Value {
        let st = &mut self.herd[seat(obs).to_string()];
        let mode = text(&st["mode"]).to_owned();
        let Some(s) = spec(&mode) else {
            return action;
        };
        let farm = own_farm(obs);
        let positions = unit_positions(farm);
        let mut market = orders(&action);
        let mut cash = num(&farm["money"]);
        for order in &mut market {
            let op = text(&order[0]).to_owned();
            if is_order(order, "BUY_ANIMAL", "GOOSE") {
                let n = int(&order[2]).max(0);
                let affordable = floor_div(cash.max(0.0), s.cost as f64) as i64;
                order[1] = json!(mode);
                order[2] = json!(n.min(affordable));
                cash -= (int(&order[2]) * s.cost) as f64;
                inc(&mut self.report, "hd2_rewrites", 1);
            } else if array(order).len() >= 3
                && matches!(op.as_str(), "BUY_ANIMAL" | "BUY_SEED" | "BUY_PRODUCT")
            {
                // Money spent by earlier orders of the same list is not available to the swap.
                let q = int(&order[2]).max(0);
                let item = text(&order[1]);
                cash -= (q * match op.as_str() {
                    "BUY_ANIMAL" => animal_cost(item),
                    "BUY_SEED" => seed_cost(item),
                    _ => int(&obs["market"]["prices"][item]),
                }) as f64;
            } else if op == "BUY_LAND" {
                cash -= 4000.0;
            }
        }
        let mut workers = units(&action);
        for actor in 0..workers.len().min(positions.len()) {
            let work = workers[actor].clone();
            if !truth(&work) {
                continue;
            }
            let pos = positions[actor];
            let t = tile(farm, pos);
            let swapped = |item: &str| {
                let mut command = vec![work[0].clone(), json!(item)];
                command.extend(array(&work).iter().skip(2).cloned());
                json!(command)
            };
            if work[0] == "BUILD_COOP" {
                workers[actor] = json!(["BUILD_PASTURE"]);
                inc(&mut self.report, "hd2_rewrites", 1);
            } else if array(&work).len() >= 2
                && (work[0] == "PICKUP" || work[0] == "PLACE")
                && work[1] == "GOOSE"
            {
                workers[actor] = swapped(&mode);
                inc(&mut self.report, "hd2_rewrites", 1);
                if work[0] == "PLACE"
                    && let Some(pending) = st["pending"].as_array_mut()
                {
                    pending.push(json!([pos.0, pos.1, int(&obs["step"]) / 24]));
                }
            } else if is_pair(&work, "PLACE", "EGG") {
                workers[actor] = swapped(s.product);
                inc(&mut self.report, "hd2_rewrites", 1);
            } else if work == json!(["HARVEST"])
                && !st["sites"][key(pos)].is_null()
                && t.is_object()
                && t["animal"] == mode.as_str()
            {
                let harvested = int(&t["yield_units"]).max(0);
                inc(st, "credit", harvested);
                inc(&mut self.report, "hd2_credit_units", harvested);
            }
        }
        set_commands(&mut action, workers);
        if int(&st["credit"]) > 0 {
            let sold = sell_credit(obs, &action, &mut market, s.product, int(&st["credit"]));
            inc(st, "credit", -sold);
            inc(&mut self.report, "hd2_sold_units", sold);
        }
        set_orders(&mut action, market);
        action
    }

    /// `_cs_plan`: today's pasture builds, cow pickups and placements to rewrite.
    fn cow_plan(obs: &Value, action: &Value, core: &Core, k: i64, mode: &str) -> Option<Value> {
        let step = int(&obs["step"]);
        let farm = own_farm(obs);
        let board = array(&farm["tiles"]).len() as i64;
        let mut positions = unit_positions(farm);
        let (mut builds, mut pickups, mut places) = (vec![], vec![], vec![]);
        for t in step..(step / 24 + 1) * 24 {
            let act = if t == step {
                action.clone()
            } else {
                late_tape(core, obs, t)
            };
            let commands = units(&act);
            for (i, p) in positions.iter_mut().enumerate() {
                let pass = json!(["PASS"]);
                let cmd = commands.get(i).filter(|c| truth(c)).unwrap_or(&pass);
                if let Some((dx, dy)) = move_delta(text(&cmd[0])) {
                    let (nx, ny) = (p.0 + dx, p.1 + dy);
                    if (0..board).contains(&nx) && (0..board).contains(&ny) {
                        *p = (nx, ny);
                    }
                } else if cmd[0] == "BUILD_PASTURE" {
                    builds.push((t, i, *p));
                } else if is_pair(cmd, "PICKUP", "COW") {
                    let q = if array(cmd).len() > 2 {
                        int(&cmd[2])
                    } else {
                        1
                    };
                    pickups.push((t, i, q.max(1)));
                } else if is_pair(cmd, "PLACE", "COW") {
                    places.push((t, i, *p));
                }
            }
            for _ in 0..hires(&act) {
                positions.push(spawn(&positions, board));
            }
        }
        let mut chosen: Vec<(i64, usize, Pos)> = vec![];
        for place in places {
            if !chosen.iter().any(|c| c.2 == place.2) {
                chosen.push(place);
            }
            if chosen.len() as i64 == k {
                break;
            }
        }
        if (chosen.len() as i64) < k {
            return None;
        }
        let mut plan = json!({"builds":{},"pickups":{},"places":{}});
        let buying_land = orders(action).iter().any(|o| o[0] == "BUY_LAND");
        let unlocked: Vec<&str> = if truth(&farm["unlocked_quadrants"]) {
            array(&farm["unlocked_quadrants"])
                .iter()
                .map(text)
                .collect()
        } else {
            vec!["NW"]
        };
        let next_quadrant = ["NE", "SW", "SE"]
            .into_iter()
            .find(|q| !unlocked.contains(q));
        let half = board / 2;
        let slot = |t: i64, i: usize| format!("{t},{i}");
        for (t, i, pos) in chosen {
            let site = tile(farm, pos);
            if mode == "GOOSE" {
                let quadrant = format!(
                    "{}{}",
                    if pos.1 < half { "N" } else { "S" },
                    if pos.0 < half { "W" } else { "E" }
                );
                let locked_but_bought =
                    site == "LOCKED" && buying_land && next_quadrant == Some(quadrant.as_str());
                if !site.is_null() && !locked_but_bought {
                    return None; // already built or occupied: no coop without extra turns
                }
                let (tb, ib, _) = builds.iter().rfind(|b| b.2 == pos && b.0 < t)?;
                plan["builds"][slot(*tb, *ib)] = json!([pos.0, pos.1]);
            }
            let (tp, ip, _) = pickups.iter().rfind(|p| p.1 == i && p.0 < t)?;
            let carried = int(&plan["pickups"][slot(*tp, *ip)]) + 1;
            plan["pickups"][slot(*tp, *ip)] = json!(carried);
            plan["places"][slot(t, i)] = json!([pos.0, pos.1]);
        }
        for (name, n) in plan["pickups"].as_object()? {
            let q = pickups
                .iter()
                .find(|p| slot(p.0, p.1) == *name)
                .map(|p| p.2)?;
            if q != int(n) {
                return None; // a pickup that also carries unswapped cows cannot be split
            }
        }
        Some(plan)
    }

    /// COWSWAP: cow / goose at the tape's first cow purchase.
    pub fn cowswap(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        match self.cowswap_apply(obs, &action, core) {
            Ok(result) => result,
            Err(_) => {
                inc(&mut self.report, "cs_errors", 1);
                action
            }
        }
    }

    fn cowswap_apply(&mut self, obs: &Value, action: &Value, core: &Core) -> Fallible<Value> {
        let step = int(&obs["step"]);
        let who = seat(obs).to_string();
        if self.cow[&who].is_null() || step <= int(&self.cow[&who]["step"]) {
            self.cow[&who] = json!({"step":-1,"decided":false,"mode":null,"plan":null,
                "broken":false,"sites":{},"pending":[],"credit":0});
            if step == 0 {
                self.reset(&[
                    "cs_decision",
                    "cs_ev",
                    "cs_rewrites",
                    "cs_broken",
                    "cs_credit",
                    "cs_sold",
                    "cs_errors",
                ]);
            }
        }
        let st = &mut self.cow[&who];
        st["step"] = json!(step);
        let farm = own_farm(obs);
        let positions = unit_positions(farm);
        let mut market = orders(action);
        let mut result = action.clone();
        // confirm placements
        if truth(&st["pending"]) {
            let mut keep = vec![];
            for entry in array(&st["pending"]).to_vec() {
                let (pos, day) = (position(&entry), int(&entry[2]));
                let t = tile(farm, pos);
                if t.is_object() && t["animal"] == st["mode"] && t["placed_day"] == day {
                    st["sites"][key(pos)] = json!(day);
                } else if step / 24 <= day {
                    keep.push(entry);
                }
            }
            st["pending"] = json!(keep);
        }
        if !truth(&st["decided"]) && (CS_FROM..CS_TO).contains(&step) {
            let k: i64 = market
                .iter()
                .filter(|o| is_order(o, "BUY_ANIMAL", "COW") && int(&o[2]) > 0)
                .map(|o| int(&o[2]))
                .sum();
            let shops = array(&obs["town"]["unlocked_shops"]);
            // `_CS_SHOP_RULE = 'nomilk'`: no milk shop and no yarn store open.
            let shop_ok = !shops.iter().any(|s| {
                matches!(
                    text(s),
                    "PIZZA_SHOP" | "ICE_CREAM_SHOP" | "SMOOTHIE_SHOP" | "YARN_STORE"
                )
            });
            if k > 0 && !shop_ok {
                st["decided"] = json!(true);
                self.report["cs_decision"] = json!(format!("shoprule@{step}"));
            } else if k > 0 {
                st["decided"] = json!(true);
                let cow = ev("COW", k, obs, core)?;
                let goose = ev("GOOSE", k, obs, core)?;
                self.report["cs_ev"] = json!(ev_text(&[("COW", cow), ("GOOSE", goose)]));
                if goose - cow >= MIN_GAIN && goose >= RATIO * cow.max(1.0) {
                    match Self::cow_plan(obs, action, core, k, "GOOSE") {
                        Some(plan) => {
                            st["mode"] = json!("GOOSE");
                            st["plan"] = plan;
                            self.report["cs_decision"] = json!(format!("GOOSE@{step}"));
                            for o in &mut market {
                                if is_order(o, "BUY_ANIMAL", "COW") {
                                    o[1] = json!("GOOSE");
                                    inc(&mut self.report, "cs_rewrites", 1);
                                }
                            }
                        }
                        None => self.report["cs_decision"] = json!(format!("noplan@{step}")),
                    }
                } else {
                    self.report["cs_decision"] = json!(format!("keep@{step}"));
                }
            }
        }
        if truth(&st["mode"]) && truth(&st["plan"]) && !truth(&st["broken"]) {
            let mode = text(&st["mode"]).to_owned();
            let plan = st["plan"].clone();
            let mut commands = units(&result);
            for i in 0..commands.len().min(positions.len()) {
                let pass = json!(["PASS"]);
                let cmd = if truth(&commands[i]) {
                    commands[i].clone()
                } else {
                    pass
                };
                let pos = json!([positions[i].0, positions[i].1]);
                let slot = format!("{step},{i}");
                let swapped = |cmd: &Value| {
                    let mut command = vec![cmd[0].clone(), json!(mode)];
                    command.extend(array(cmd).iter().skip(2).cloned());
                    json!(command)
                };
                if let Some(site) = plan["builds"].get(&slot) {
                    if cmd == json!(["BUILD_PASTURE"]) && pos == *site {
                        commands[i] = json!(["BUILD_COOP"]);
                        inc(&mut self.report, "cs_rewrites", 1);
                    } else {
                        st["broken"] = json!(true);
                    }
                }
                if plan["pickups"].get(&slot).is_some() {
                    if is_pair(&cmd, "PICKUP", "COW") {
                        commands[i] = swapped(&cmd);
                        inc(&mut self.report, "cs_rewrites", 1);
                    } else {
                        st["broken"] = json!(true);
                    }
                }
                if let Some(site) = plan["places"].get(&slot) {
                    if is_pair(&cmd, "PLACE", "COW") && pos == *site {
                        commands[i] = swapped(&cmd);
                        if let Some(pending) = st["pending"].as_array_mut() {
                            pending.push(json!([positions[i].0, positions[i].1, step / 24]));
                        }
                        inc(&mut self.report, "cs_rewrites", 1);
                    } else {
                        st["broken"] = json!(true);
                    }
                }
            }
            if truth(&st["broken"]) {
                inc(&mut self.report, "cs_broken", 1);
            }
            set_commands(&mut result, commands);
        }
        // credit harvests on swapped tiles and sell them as they reach the shed
        if truth(&st["sites"]) {
            let mode = text(&st["mode"]).to_owned();
            let product = spec(&mode).ok_or("cs: mode")?.product;
            let commands = units(&result);
            for i in 0..commands.len().min(positions.len()) {
                let t = tile(farm, positions[i]);
                if commands[i] == json!(["HARVEST"])
                    && !st["sites"][key(positions[i])].is_null()
                    && t.is_object()
                    && t["animal"] == mode.as_str()
                {
                    let n = int(&t["yield_units"]).max(0);
                    inc(st, "credit", n);
                    inc(&mut self.report, "cs_credit", n);
                }
            }
            if int(&st["credit"]) > 0 {
                let sold = sell_credit(obs, &result, &mut market, product, int(&st["credit"]));
                inc(st, "credit", -sold);
                inc(&mut self.report, "cs_sold", sold);
            }
        }
        set_orders(&mut result, market);
        Ok(result)
    }
}
