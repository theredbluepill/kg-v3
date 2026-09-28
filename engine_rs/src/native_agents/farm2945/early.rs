//! 2945 Farm v9 layers closest to the chassis, in source order:
//! COURIER, CARROT, HERD, FERT and OPENING.
//! Source: agents/farm2945/main.py (`_v9_courier`, `_v9_carrot`, `_v9_herd`,
//! `_v9_fert`, `_v9_opening` and their `agent` wrappers).
use super::util::*;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::core::Core;
use serde_json::{Value, json};

const COURIER_ITEMS: [&str; 4] = ["STRAWBERRY", "MILK", "WOOL", "MELON"];
const COURIER_FROM_HOUR: i64 = 12;
const ACCESS: [Pos; 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];

fn idle(command: &Value) -> bool {
    !truth(command)
        || matches!(
            text(&command[0]),
            "PASS" | "NORTH" | "SOUTH" | "EAST" | "WEST" | "DROP"
        )
}

/// Per-seat layer state, reset on a new or repeated step like the source wrappers.
fn layer_state<'a>(states: &'a mut Value, obs: &Value, fresh: Value) -> (&'a mut Value, bool) {
    let step = int(&obs["step"]);
    let key = seat(obs).to_string();
    let reset = states[&key].is_null() || step <= int(&states[&key]["step"]);
    if reset {
        states[&key] = fresh;
    }
    states[&key]["step"] = json!(step);
    (&mut states[&key], reset)
}

#[derive(Clone, Debug)]
pub struct Early {
    pub cha22: bool,
    pub courier: Value,
    pub carrot: Value,
    pub herd: Value,
    pub report: Value,
}
impl Default for Early {
    fn default() -> Self {
        Self {
            cha22: false,
            courier: json!({}),
            carrot: json!({}),
            herd: json!({}),
            report: json!({
                "courier_trips":0,"courier_units":0,"courier_errors":0,
                "carrot_swaps":0,"carrot_seed_swaps":0,"carrot_errors":0,
                "herd_species":"","herd_rewrites":0,"herd_extra_sold":0,"herd_errors":0,
                "fert_applied":0,"fert_errors":0,
            }),
        }
    }
}

fn courier_plan(
    core: &Core,
    route: i64,
    unit: usize,
    pos: Pos,
    commands: &[Value],
    step: i64,
    end: i64,
) -> Option<Vec<Value>> {
    if !idle(&commands[unit]) {
        return None;
    }
    for t in step + 1..=end {
        let future = units(&if core.hybrid_opening || core.cha22 {
            core.configured_route_action(route, t)
        } else {
            tape(route, t)
        });
        if future.get(unit).is_some_and(|c| !idle(c)) {
            return None;
        }
    }
    let target = *ACCESS
        .iter()
        .min_by_key(|a| (a.0 - pos.0).abs() + (a.1 - pos.1).abs())
        .unwrap();
    let mut walk = vec![];
    for (name, n) in [
        ("EAST", target.0 - pos.0),
        ("WEST", pos.0 - target.0),
        ("SOUTH", target.1 - pos.1),
        ("NORTH", pos.1 - target.1),
    ] {
        walk.extend((0..n.max(0)).map(|_| json!([name])));
    }
    if walk.len() as i64 > end - step {
        return None;
    }
    walk.push(json!(["DROP"]));
    Some(walk)
}

impl Early {
    fn reset_at_zero(&mut self, obs: &Value, reset: bool, keys: &[&str]) {
        if reset && int(&obs["step"]) == 0 {
            for k in keys {
                self.report[*k] = if *k == "herd_species" {
                    json!("")
                } else {
                    json!(0)
                };
            }
        }
    }

    /// COURIER: idle tape workers carry premium cargo to the shed before midnight.
    pub fn courier(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let (_, reset) = layer_state(&mut self.courier, obs, json!({"step":-1}));
        self.reset_at_zero(
            obs,
            reset,
            &["courier_trips", "courier_units", "courier_errors"],
        );
        match self.courier_apply(obs, &action, core) {
            Ok(result) => result,
            Err(_) => {
                inc(&mut self.report, "courier_errors", 1);
                action
            }
        }
    }

    fn courier_apply(&mut self, obs: &Value, action: &Value, core: &Core) -> Fallible<Value> {
        let step = int(&obs["step"]);
        if step >= 718 || step % 24 < COURIER_FROM_HOUR {
            return Ok(action.clone());
        }
        let day = step / 24;
        let st = &mut self.courier[seat(obs).to_string()];
        if st["day"] != day {
            st["day"] = json!(day);
            st["plans"] = json!({});
        }
        let Some(route) = native_route(core, obs) else {
            return Ok(action.clone());
        };
        let native = &core.players[seat(obs).to_string()];
        let farm = own_farm(obs);
        let positions = unit_positions(farm);
        let inventories = array(&obs["private"]["inventories"]);
        let mut commands = units(action);
        while commands.len() < positions.len() {
            commands.push(json!(["PASS"]));
        }
        let end = day * 24 + 23;
        let crew = 1
            + (day * 24..tape_len(route).min(end + 1))
                .map(|t| array(&tape(route, t)["hands"]).len())
                .max()
                .ok_or("courier: empty day")?;
        let mut delivered: Vec<(String, i64)> = vec![];
        let mut changed = false;
        for (unit, pos) in positions.iter().enumerate().take(crew) {
            let cargo: Vec<(String, i64)> = inventories
                .get(unit)
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
                .filter(|(k, v)| COURIER_ITEMS.contains(&k.as_str()) && int(v) > 0)
                .map(|(k, v)| (k.clone(), int(v)))
                .collect();
            let slot = unit.to_string();
            if st["plans"][&slot].is_null() {
                if cargo.is_empty() || truth(&native["pending"][&slot]) {
                    continue;
                }
                let Some(plan) = courier_plan(core, route, unit, *pos, &commands, step, end) else {
                    continue;
                };
                st["plans"][&slot] = json!({"route": plan, "start": step});
                inc(&mut self.report, "courier_trips", 1);
            }
            let index = (step - int(&st["plans"][&slot]["start"])) as usize;
            let Some(command) = st["plans"][&slot]["route"].get(index).cloned() else {
                continue;
            };
            if command == json!(["DROP"]) {
                if !ACCESS.contains(pos) {
                    st["plans"][&slot] = json!({"route": [], "start": step});
                    continue;
                }
                for (item, n) in &cargo {
                    match delivered.iter_mut().find(|(k, _)| k == item) {
                        Some(entry) => entry.1 += n,
                        None => delivered.push((item.clone(), *n)),
                    }
                }
            }
            commands[unit] = command;
            changed = true;
        }
        if !changed {
            return Ok(action.clone());
        }
        let mut result = with_units(action, commands);
        if !delivered.is_empty() {
            let mut market = orders(action);
            let prices = &obs["market"]["prices"];
            delivered.sort_by_key(|(item, n)| -int(&prices[item]) * n);
            for (item, n) in delivered {
                if int(&prices[&item]) < 2 {
                    continue;
                }
                if let Some(at) = market.iter().position(|o| is_order(o, "SELL", &item)) {
                    let mut existing = market.remove(at);
                    existing[2] = json!(int(&existing[2]) + n);
                    market.insert(0, existing);
                } else if market.len() < MAX_ORDERS {
                    market.insert(0, json!(["SELL", item, n]));
                }
                inc(&mut self.report, "courier_units", n);
            }
            set_orders(&mut result, market);
        }
        Ok(result)
    }

    /// CARROT: plant carrots instead of wheat while the carrot book pays for it.
    pub fn carrot(&mut self, obs: &Value, action: Value) -> Value {
        let (_, reset) = layer_state(&mut self.carrot, obs, json!({"step":-1}));
        self.reset_at_zero(
            obs,
            reset,
            &["carrot_swaps", "carrot_seed_swaps", "carrot_errors"],
        );
        let step = int(&obs["step"]);
        let day = step / 24;
        let prices = &obs["market"]["prices"];
        let private = &obs["private"];
        let farm = own_farm(obs);
        let positions = unit_positions(farm);
        let mut commands = units(&action);
        let mut market = orders(&action);
        let mut changed = false;
        let st = &mut self.carrot[seat(obs).to_string()];
        if truth(&st["tiles"]) {
            // Swapped carrots die at the start of age 4: harvest on the age-3 visit.
            for (i, c) in commands.iter_mut().enumerate().take(positions.len()) {
                if *c != json!(["WATER"]) || st["tiles"][key(positions[i])] != day - 3 {
                    continue;
                }
                let t = tile(farm, positions[i]);
                let planted = t.get("planted_day").map(int).unwrap_or(-9);
                if t.is_object()
                    && t["crop"] == "CARROT"
                    && planted == day - 3
                    && int(&t["yield_units"]) > 0
                {
                    *c = json!(["HARVEST"]);
                    changed = true;
                    inc(&mut self.report, "carrot_rescues", 1);
                }
            }
        }
        let wheat_held = int(&private["shed"]["WHEAT"])
            + array(&private["inventories"])
                .iter()
                .map(|i| int(&i["WHEAT"]))
                .sum::<i64>();
        let wheat_quote = prices.get("WHEAT").map(int).unwrap_or(99);
        let ratio = int(&prices["CARROT"]) as f64 / wheat_quote.max(1) as f64;
        let boom = ratio >= if self.cha22 { 4.5 } else { 3.5 };
        let reserve = if boom { 10 } else { 40 };
        let window = (10..=23).contains(&day);
        if boom && window && wheat_held < 40 {
            // Carrots are worth several wheat each: buy the feed the swap no longer grows.
            let budget = num(&farm["money"]) - 1500.0;
            let qty =
                (40 - wheat_held).min(floor_div(budget, (wheat_quote + 5).max(1) as f64) as i64);
            if qty > 0
                && market.len() < MAX_ORDERS
                && !market.iter().any(|o| is_pair(o, "BUY_PRODUCT", "WHEAT"))
            {
                market.push(json!(["BUY_PRODUCT", "WHEAT", qty]));
                changed = true;
            }
        }
        if window && wheat_held >= reserve && ratio >= if self.cha22 { 2.0 } else { 1.8 } {
            let mut seeds = int(&private["seeds"]["CARROT"])
                - commands
                    .iter()
                    .filter(|c| is_pair(c, "PLANT", "CARROT"))
                    .count() as i64;
            for (i, c) in commands.iter_mut().enumerate() {
                if is_pair(c, "PLANT", "WHEAT") && seeds > 0 {
                    c[1] = json!("CARROT");
                    seeds -= 1;
                    changed = true;
                    st["swapped"] = json!(true);
                    inc(&mut self.report, "carrot_swaps", 1);
                    if let Some(p) = positions.get(i) {
                        if !st["tiles"].is_object() {
                            st["tiles"] = json!({});
                        }
                        st["tiles"][key(*p)] = json!(day);
                    }
                }
            }
            for o in &mut market {
                if is_order(o, "BUY_SEED", "WHEAT") {
                    o[1] = json!("CARROT");
                    changed = true;
                    st["swapped"] = json!(true);
                    inc(&mut self.report, "carrot_seed_swaps", int(&o[2]));
                }
            }
        }
        let mut result = with_units(&action, commands);
        set_orders(&mut result, market.clone());
        if truth(&st["swapped"]) && day < 24 {
            // Swapped carrots have no planned sale on the tape before its own carrot days.
            let stock = int(&Core::projected_shed(&result, &View::new(obs))["CARROT"]);
            let sold = selling(&market, "CARROT");
            if stock > sold && market.len() < MAX_ORDERS && int(&prices["CARROT"]) >= 2 {
                market.insert(0, json!(["SELL", "CARROT", stock - sold]));
                set_orders(&mut result, market);
                changed = true;
            }
        }
        if changed { result } else { action }
    }

    fn herd_choose(obs: &Value) -> Option<&'static str> {
        let shops = array(&obs["town"]["unlocked_shops"]);
        let prices = &obs["market"]["prices"];
        let count =
            |names: &[&str]| shops.iter().filter(|s| names.contains(&text(s))).count() as i64;
        let egg_shops = count(&["BAKERY", "BRUNCH_SPOT"]);
        if egg_shops <= 1 && count(&["YARN_STORE"]) > 0 && int(&prices["WOOL"]) >= 150 {
            return Some("SHEEP");
        }
        let milk_shops = count(&["PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"]);
        if egg_shops <= 0 && milk_shops >= 3 && int(&prices["MILK"]) >= 150 {
            return Some("COW");
        }
        None
    }

    /// HERD: raise the tape's day-10 geese as sheep or cows when the draw pays.
    pub fn herd(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let (_, reset) = layer_state(&mut self.herd, obs, json!({"step":-1}));
        self.reset_at_zero(
            obs,
            reset,
            &[
                "herd_species",
                "herd_rewrites",
                "herd_extra_sold",
                "herd_errors",
            ],
        );
        if self.cha22 {
            return action;
        }
        let step = int(&obs["step"]);
        let st = &mut self.herd[seat(obs).to_string()];
        let market = orders(&action);
        if st["species"].is_null()
            && !truth(&st["decided"])
            && step >= 216
            && market.iter().any(|o| is_pair(o, "BUY_ANIMAL", "GOOSE"))
        {
            st["decided"] = json!(true);
            let species = Self::herd_choose(obs);
            st["species"] = json!(species);
            self.report["herd_species"] = json!(species.unwrap_or(""));
        }
        let Some(species) = st["species"].as_str().map(str::to_owned) else {
            return action;
        };
        let product = if species == "SHEEP" { "WOOL" } else { "MILK" };
        let mut commands = units(&action);
        for c in &mut commands {
            if c[0] == "BUILD_COOP" {
                c[0] = json!("BUILD_PASTURE");
                inc(&mut self.report, "herd_rewrites", 1);
            } else if array(c).len() >= 2
                && (c[0] == "PICKUP" || c[0] == "PLACE")
                && c[1] == "GOOSE"
            {
                c[1] = json!(species);
                inc(&mut self.report, "herd_rewrites", 1);
            }
        }
        let farm = own_farm(obs);
        let geese = array(&farm["tiles"])
            .iter()
            .flat_map(array)
            .any(|t| t.is_object() && t["animal"] == "GOOSE");
        let mut rewritten = vec![];
        for o in market {
            if is_order(&o, "BUY_ANIMAL", "GOOSE") {
                rewritten.push(json!(["BUY_ANIMAL", species, o[2]]));
            } else if geese || !is_pair(&o, "SELL", "EGG") {
                rewritten.push(o);
            }
        }
        let mut result = with_units(&action, commands);
        set_orders(&mut result, rewritten.clone());
        if let Some(route) = native_route(core, obs)
            && step < 718
        {
            let stock = int(&Core::projected_shed(&result, &View::new(obs))[product]);
            let extra =
                stock - selling(&rewritten, product) - future_sells(route, product, step + 1);
            if extra > 0
                && rewritten.len() < MAX_ORDERS
                && int(&obs["market"]["prices"][product]) >= 2
            {
                rewritten.insert(0, json!(["SELL", product, extra]));
                set_orders(&mut result, rewritten);
                inc(&mut self.report, "herd_extra_sold", extra);
            }
        }
        result
    }

    /// FERT: spend carried fertilizer on one-day-old wheat and carrots from day 16.
    pub fn fert(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        if step == 0 {
            self.report["fert_applied"] = json!(0);
            self.report["fert_errors"] = json!(0);
        }
        let day = step / 24;
        if day < if core.metav4 { 14 } else { 16 } || step >= 700 {
            return action;
        }
        let farm = own_farm(obs);
        let positions = unit_positions(farm);
        let inventories = array(&obs["private"]["inventories"]);
        let mut commands = units(&action);
        let native = &core.players[seat(obs).to_string()];
        let mut planned = std::collections::HashMap::new();
        if truth(native) && !native["route"].is_null() {
            let route = int(&native["route"]);
            if !routes()[route.to_string()].is_null() {
                // Fertilizer this worker's own tape commands still spend today.
                for t in step..tape_len(route).min(day * 24 + 24) {
                    for (u, c) in units(&tape(route, t)).iter().enumerate() {
                        if c[0] == "FERTILIZE" {
                            *planned.entry(u).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        let mut carried = std::collections::HashMap::new();
        let mut targeted = vec![];
        let mut changed = false;
        for unit in 0..commands.len().min(positions.len()) {
            if commands[unit][0] != "WATER" {
                continue;
            }
            let p = positions[unit];
            let t = tile(farm, p);
            if !(t.is_object()
                && t["kind"] == "PLANT"
                && (t["crop"] == "WHEAT" || t["crop"] == "CARROT"))
            {
                continue;
            }
            if day - int(&t["planted_day"]) != 1 || truth(&t["watered_today"]) {
                continue;
            }
            let unwatered = t.get("consecutive_unwatered").map(int).unwrap_or(1);
            let fertilized = t.get("fertilized_until_day").map(int).unwrap_or(-1);
            if unwatered != 0 || fertilized >= day {
                continue;
            }
            let have = *carried.entry(unit).or_insert_with(|| {
                inventories
                    .get(unit)
                    .map(|i| int(&i["FERTILIZER"]))
                    .unwrap_or(0)
                    - planned.get(&unit).copied().unwrap_or(0)
            });
            if have <= 0 || targeted.contains(&p) {
                continue;
            }
            commands[unit] = json!(["FERTILIZE"]);
            carried.insert(unit, have - 1);
            targeted.push(p);
            changed = true;
            inc(&mut self.report, "fert_applied", 1);
        }
        if changed {
            with_units(&action, commands)
        } else {
            action
        }
    }

    /// OPENING: replace the tape's wheat round trip with BUY 20 / SELL 15 at step 0.
    pub fn opening(obs: &Value, action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        if step > 1 || native_route(core, obs).is_none() {
            return action;
        }
        let market = orders(&action);
        let is_wheat = |o: &Value| {
            array(o).len() >= 3 && (o[0] == "BUY_PRODUCT" || o[0] == "SELL") && o[1] == "WHEAT"
        };
        let wheat: Vec<Value> = market
            .iter()
            .filter(|o| is_wheat(o))
            .map(|o| json!([o[0], o[1], int(&o[2])]))
            .collect();
        let expected = if step == 0 {
            json!([
                ["BUY_PRODUCT", "WHEAT", 13],
                ["BUY_PRODUCT", "WHEAT", 30],
                ["SELL", "WHEAT", 30]
            ])
        } else {
            json!([["SELL", "WHEAT", 13], ["BUY_PRODUCT", "WHEAT", 5]])
        };
        if json!(wheat) != expected {
            return action; // the tape's opening was changed upstream; leave it alone
        }
        let mut new = if step == 0 {
            vec![
                json!(["BUY_PRODUCT", "WHEAT", 20]),
                json!(["SELL", "WHEAT", 15]),
            ]
        } else {
            vec![]
        };
        new.extend(market.into_iter().filter(|o| !is_wheat(o)));
        let mut result = action;
        set_orders(&mut result, new);
        result
    }
}
