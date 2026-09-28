//! Native translation of frozen V39 production and reservation overlays.
//! Source: agents/v39/main.py, SHA-256 708c7485…; original attribution retained there.
use super::common::*;
use super::core::Core;
use serde_json::{Value, json};

fn arr(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn name(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
fn qty(v: &Value, k: &str) -> i64 {
    int(&v[k])
}
fn inc(v: &mut Value, k: &str, n: i64) {
    v[k] = json!(qty(v, k) + n);
}
fn remove(v: &mut Value, k: &str) -> Value {
    v.as_object_mut()
        .and_then(|m| m.remove(k))
        .unwrap_or(Value::Null)
}
fn sum_stock(v: &Value) -> i64 {
    v.as_object()
        .map(|m| m.values().map(int).sum())
        .unwrap_or(0)
}
fn pair(c: &Value, op: &str, item: &str) -> bool {
    c[0] == op && c[1] == item
}
fn has_quad(farm: &Value, q: &str) -> bool {
    arr(&farm["unlocked_quadrants"]).iter().any(|v| v == q)
}
fn exactly_three_quads(farm: &Value) -> bool {
    let q = arr(&farm["unlocked_quadrants"]);
    ["NW", "NE", "SW"].iter().all(|s| q.iter().any(|v| v == s))
        && q.iter().all(|v| ["NW", "NE", "SW"].contains(&name(v)))
}
fn tile_at(farm: &Value, p: (i64, i64)) -> &Value {
    &farm["tiles"][p.1 as usize][p.0 as usize]
}
fn animal(v: &Value) -> bool {
    ["COW", "SHEEP", "GOOSE"].contains(&name(v))
}
pub fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1, 1);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}
pub fn walk(p: (i64, i64), t: (i64, i64)) -> Option<Value> {
    if p.0 != t.0 {
        Some(json!([if p.0 < t.0 { "EAST" } else { "WEST" }]))
    } else if p.1 != t.1 {
        Some(json!([if p.1 < t.1 { "SOUTH" } else { "NORTH" }]))
    } else {
        None
    }
}
fn distance(a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
fn home(pos: (i64, i64)) -> (i64, i64) {
    [(4, 4), (5, 4), (4, 5), (5, 5)]
        .into_iter()
        .min_by_key(|p| distance(pos, *p))
        .unwrap()
}
pub fn native_day(native: &Value, day: i64) -> Vec<Value> {
    (day * 24..((day + 1) * 24).min(719))
        .map(|step| route_action(int(&native["route"]), step))
        .collect()
}
fn order_cost(obs: &Value, o: &Value) -> i64 {
    let q = int(&o[2]);
    let item = name(&o[1]);
    match name(&o[0]) {
        "BUY_PRODUCT" => q * (int(&obs["market"]["prices"][item]) + 10),
        "BUY_ANIMAL" => {
            q * match item {
                "COW" => 400,
                "SHEEP" => 500,
                "GOOSE" => 300,
                _ => 0,
            }
        }
        "BUY_SEED" => {
            q * match item {
                "WHEAT" => 10,
                "CARROT" => 20,
                "TOMATO" => 50,
                "STRAWBERRY" => 100,
                "MELON" => 80,
                _ => 0,
            }
        }
        _ => 0,
    }
}

#[derive(Clone, Debug)]
pub struct Production {
    pub v219: Value,
    pub v231: Value,
    pub v233: Value,
    pub report: Value,
    pub r37: Value,
    pub r44: Value,
    pub horizons: Value,
}
impl Default for Production {
    fn default() -> Self {
        Self {
            v219: json!({}),
            v231: json!({}),
            v233: json!({}),
            report: json!({}),
            r37: json!({}),
            r44: json!({}),
            horizons: json!({}),
        }
    }
}

impl Production {
    pub fn before(&mut self, obs: &Value) {
        let seat = int(&obs["player"]);
        let key = seat.to_string();
        let step = int(&obs["step"]);
        if self.r37[&key].is_null() || step <= int(&self.r37[&key]["step"]) {
            self.r37[&key] = json!({"step":-1,"streak":0});
        }
        if self.r44[&key].is_null() || step <= int(&self.r44[&key]["step"]) {
            self.r44[&key] = json!({"step":-1,"money":null,"probe":0,"matched":false});
        }
        if step == 0 {
            for k in [
                "quote_reordered_turns",
                "three_turn_calls",
                "nocturne_errors",
                "probe_matches",
                "probe_four_turn_calls",
                "probe_errors",
                "sale_reserved_units",
                "sale_reservations",
                "sale_errors",
            ] {
                self.report[k] = json!(0);
            }
        }
        let money = json!([
            num(&obs["farms"][seat as usize]["money"]),
            num(&obs["farms"][(1 - seat) as usize]["money"])
        ]);
        let st = &mut self.r44[&key];
        if !st["money"].is_null() && int(&st["probe"]) >= 100 && similarity(obs) >= 0.90 {
            let own = num(&money[0]) - num(&st["money"][0]);
            let rival = num(&money[1]) - num(&st["money"][1]);
            if own > 0.0
                && rival > 0.0
                && (own - rival).abs() <= 5.0_f64.max(0.05 * num(&st["probe"]))
            {
                if !truth(&st["matched"]) {
                    inc(&mut self.report, "probe_matches", 1);
                }
                st["matched"] = json!(true);
            }
        }
        st["step"] = json!(step);
        st["money"] = money;
        st["probe"] = json!(0);
        let st = &mut self.r37[&key];
        st["step"] = json!(step);
        let mut horizon = 2;
        if step < 648 {
            st["streak"] = json!(if similarity(obs) >= 0.90 {
                int(&st["streak"]) + 1
            } else {
                0
            });
            if (336..648).contains(&step) && int(&st["streak"]) >= 6 {
                horizon = 3;
                inc(&mut self.report, "three_turn_calls", 1);
            }
        }
        if horizon == 3 && truth(&self.r44[&key]["matched"]) {
            horizon = 4;
            inc(&mut self.report, "probe_four_turn_calls", 1);
        }
        if (288..696).contains(&step) {
            horizon = 4;
        }
        self.horizons[&key] = json!(horizon);
    }

    /// Called after all inner terminal/room wrappers, before late R46+ overlays.
    pub fn after(
        &mut self,
        obs: &Value,
        mut action: Value,
        core: &mut Core,
        valid_config: bool,
    ) -> Value {
        let key = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        action = self.tomatoes(obs, action, &core.players[&key]);
        if step >= 144 {
            action = self.sales_first(action);
        }
        action = self.cattle(obs, action);
        if valid_config {
            action = self.reserve(obs, action, &mut core.players[&key]);
            if step >= 288 {
                action = self.sales_first(action);
            }
        }
        if (336..648).contains(&step) && !truth(&self.r44[&key]["matched"]) {
            let market = orders(&action);
            if !market.is_empty() && !market.iter().any(|o| truth(o) && o[0] != "SELL") {
                let debt = &core.players[&key]["sell_state"]["r36_debts"][(step + 3).to_string()];
                if let Some(m) = debt.as_object().filter(|m| !m.is_empty()) {
                    let probe: i64 = m
                        .iter()
                        .map(|(item, n)| int(n).max(0) * int(&obs["market"]["prices"][item]))
                        .sum();
                    self.r44[&key]["probe"] = json!(probe);
                }
            }
        }
        if step >= 288 {
            action = self.reorder_sales(obs, action);
        }
        self.sheep(obs, action, &core.players[&key], valid_config)
    }

    pub fn sales_first(&mut self, mut action: Value) -> Value {
        let original: Vec<_> = orders(&action).into_iter().take(10).collect();
        let mut market: Vec<_> = original
            .iter()
            .filter(|o| {
                !arr(o).is_empty()
                    && (["HIRE", "BUY_LAND"].contains(&name(&o[0]))
                        || (arr(o).len() >= 3 && int(&o[2]) > 0))
            })
            .cloned()
            .collect();
        for index in 0..market.len() {
            if market[index][0] != "SELL" {
                continue;
            }
            let mut cursor = index;
            while cursor > 0 {
                let previous = &market[cursor - 1];
                if previous[0] == "SELL"
                    || (["BUY_PRODUCT", "BUY_ANIMAL"].contains(&name(&previous[0]))
                        && previous[1] == market[cursor][1])
                {
                    break;
                }
                market.swap(cursor - 1, cursor);
                cursor -= 1;
            }
        }
        if market != original {
            inc(&mut self.report, "reordered_market_turns", 1);
            set_orders(&mut action, market);
        }
        action
    }

    fn tomato_qualifies(obs: &Value) -> bool {
        let farm = &obs["farms"][int(&obs["player"]) as usize];
        if arr(&farm["tiles"]).len() != 10
            || !exactly_three_quads(farm)
            || num(&farm["money"]) < 12000.0
            || int(&obs["market"]["prices"]["TOMATO"]) < 70
        {
            return false;
        }
        if arr(&obs["town"]["unlocked_shops"])
            .iter()
            .filter(|v| ["PIZZA_SHOP", "FARMERS_MARKET"].contains(&name(v)))
            .count()
            < 3
        {
            return false;
        }
        if (5..7).any(|y| (5..10).any(|x| farm["tiles"][y][x] != "LOCKED"))
            || truth(&obs["private"]["seeds"]["TOMATO"])
            || truth(&obs["private"]["shed"]["TOMATO"])
        {
            return false;
        }
        if arr(&farm["tiles"])
            .iter()
            .flat_map(arr)
            .any(|t| t["crop"] == "TOMATO")
        {
            return false;
        }
        for tape in routes().as_object().into_iter().flat_map(|m| m.values()) {
            for a in arr(tape).iter().skip(432).take(287) {
                if orders(a).iter().any(|o| o[0] == "BUY_LAND")
                    || commands(a).iter().any(|c| *c == json!(["PLANT", "TOMATO"]))
                {
                    return false;
                }
            }
        }
        true
    }

    fn tomato_request(
        &mut self,
        obs: &Value,
        mut action: Value,
        state: &mut Value,
        native: &Value,
    ) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        let offset = step % 24;
        let view = View::new(obs);
        if (!truth(&state["committed"]) && day != 18) || state["requested_day"] == day || offset > 3
        {
            return action;
        }
        let planned = native_day(native, day);
        if planned
            .iter()
            .skip((offset + 1) as usize)
            .any(|a| orders(a).iter().any(|o| o[0] == "HIRE"))
        {
            return action;
        }
        let mut market = orders(&action);
        let parent_hires = market.iter().filter(|o| o[0] == "HIRE").count() as i64;
        let expected = planned
            .iter()
            .map(|a| arr(&a["hands"]).len())
            .max()
            .unwrap_or(0) as i64;
        if view.positions.len() as i64 - 1 + parent_hires != expected {
            return action;
        }
        let fertilizer =
            [24, 27].contains(&day) && super::late::tomato_fertilizer_worthwhile(obs, &action);
        let mut crop_workers = if [19, 20, 21, 22, 23, 25].contains(&day) && offset <= 2 {
            1
        } else if (26..=28).contains(&day) {
            3
        } else {
            2
        };
        let labor = super::late::labor_assignment(obs, &action, fertilizer);
        if let Some(l) = &labor {
            crop_workers = int(&l["workers"]);
        }
        let count = crop_workers + i64::from(fertilizer && day == 27 && labor.is_none());
        let mut extra = vec![];
        if !truth(&state["committed"]) {
            extra.extend([json!(["BUY_LAND"]), json!(["BUY_SEED", "TOMATO", 10])]);
        }
        let fertilizer_quantity = if fertilizer {
            super::late::parent_fert_qty(obs, &action, &planned, offset)
        } else {
            0
        };
        if fertilizer {
            extra.push(json!(["BUY_PRODUCT", "FERTILIZER", fertilizer_quantity]));
        }
        for _ in 0..count {
            extra.push(json!(["HIRE"]));
        }
        if market.len() + extra.len() > 10 {
            return action;
        }
        let mut budget: f64 = (view.hires_today..view.hires_today + parent_hires + count)
            .map(fib)
            .sum::<i64>() as f64;
        if !truth(&state["committed"]) {
            budget += 4500.0;
        }
        if fertilizer {
            budget +=
                fertilizer_quantity as f64 * (num(&obs["market"]["prices"]["FERTILIZER"]) + 5.0);
        }
        budget += market.iter().map(|o| order_cost(obs, o)).sum::<i64>() as f64;
        if view.money < budget + 3000.0 {
            inc(&mut self.report, "budget_declines", 1);
            return action;
        }
        state["pending"] = json!({"step":step,"first_actor":expected+1,"count":count,"crop_workers":crop_workers,"fertilizer":fertilizer,"labor":labor});
        if labor.is_some() {
            inc(&mut self.report, "labor_requests", 1);
            inc(&mut self.report, "labor_hires_avoided", 1);
            inc(&mut self.report, &format!("labor_day{day}"), 1);
        }
        state["requested_day"] = json!(day);
        inc(&mut self.report, "hire_requests", count);
        if !truth(&state["committed"]) {
            state["committed"] = json!(true);
            inc(&mut self.report, "commitments", 1);
        }
        market.extend(extra);
        set_orders(&mut action, market);
        action
    }

    fn tomato_worker(
        &mut self,
        obs: &Value,
        state: &mut Value,
        actor: usize,
        role: &mut Value,
    ) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        let view = View::new(obs);
        let pos = position(&view.positions[actor]);
        let inv = view.inv(actor);
        let targets = arr(&role["targets"]).to_vec();
        let previous = &state["last_work"][actor.to_string()];
        if !previous.is_null()
            && previous["step"] == step - 1
            && previous["command"] == json!(["HARVEST"])
        {
            inc(
                &mut self.report,
                "confirmed_harvest_units",
                (int(&inv["TOMATO"]) - int(&previous["tomatoes"])).max(0),
            );
        }
        if truth(&role["needs_fertilizer"]) && !truth(&role["loaded"]) {
            let dest = home(pos);
            if let Some(c) = walk(pos, dest) {
                return c;
            }
            let desired = role
                .get("fertilizer_quantity")
                .map(int)
                .unwrap_or(if role["kind"] == "fertilizer" { 10 } else { 5 });
            if int(&inv["FERTILIZER"]) >= desired {
                role["loaded"] = json!(true);
            } else if truth(&role["pickup_requested"]) {
                role["loaded"] = json!(true);
                role["fertilizer_available"] = json!(int(&inv["FERTILIZER"]));
            } else if int(&view.shed["FERTILIZER"]) >= desired {
                role["pickup_requested"] = json!(true);
                return json!(["PICKUP", "FERTILIZER", desired]);
            } else {
                role["loaded"] = json!(true);
            }
        }
        let mut todo = vec![];
        for (index, target) in targets.iter().enumerate() {
            let p = position(target);
            let tile = &view.tiles[p.1 as usize][p.0 as usize];
            let tomato = tile["crop"] == "TOMATO";
            if tomato && !arr(&state["seen_plants"]).contains(target) {
                state["seen_plants"]
                    .as_array_mut()
                    .unwrap()
                    .push(target.clone());
                inc(&mut self.report, "confirmed_plants", 1);
            }
            if arr(&state["seen_plants"]).contains(target)
                && !tomato
                && !arr(&state["lost"]).contains(target)
            {
                state["lost"].as_array_mut().unwrap().push(target.clone());
                inc(&mut self.report, "lost_plants", 1);
            }
            let until = tile.get("fertilized_until_day").map(int).unwrap_or(-1);
            let command = if role["kind"] == "fertilizer" {
                if tomato && until < day + 2 && int(&inv["FERTILIZER"]) > 0 {
                    Some(json!(["FERTILIZE"]))
                } else {
                    None
                }
            } else if day == 18 && !tomato {
                if tile.is_null() && int(&obs["private"]["seeds"]["TOMATO"]) > 0 {
                    Some(json!(["PLANT", "TOMATO"]))
                } else if tile["kind"] == "WEED" {
                    Some(json!(["DIG"]))
                } else {
                    None
                }
            } else if tomato {
                if day < 29 && !truth(&tile["watered_today"]) {
                    Some(json!(["WATER"]))
                } else if truth(&role["needs_fertilizer"])
                    && until < day + 2
                    && int(&inv["FERTILIZER"]) > 0
                {
                    Some(json!(["FERTILIZE"]))
                } else if int(&tile["yield_units"]) > 0 {
                    Some(json!(["HARVEST"]))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(c) = command {
                todo.push((distance(pos, p), index, p, c));
            }
        }
        let dest = home(pos);
        let tomatoes = int(&inv["TOMATO"]);
        if step >= 718 - distance(pos, dest) && tomatoes != 0 {
            return walk(pos, dest).unwrap_or_else(|| json!(["PLACE", "TOMATO", tomatoes]));
        }
        if let Some((_, _, p, c)) = todo.into_iter().min_by_key(|v| (v.0, v.1)) {
            return walk(pos, p).unwrap_or(c);
        }
        if tomatoes != 0 {
            return walk(pos, dest).unwrap_or_else(|| json!(["PLACE", "TOMATO", tomatoes]));
        }
        if inv.as_object().is_some_and(|m| m.values().any(truth)) {
            return walk(pos, dest).unwrap_or_else(|| json!(["DROP"]));
        }
        json!(["PASS"])
    }

    fn tomatoes(&mut self, obs: &Value, mut action: Value, native: &Value) -> Value {
        let key = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        let day = step / 24;
        let mut state = remove(&mut self.v219, &key);
        if state.is_null() || step <= int(&state["last_step"]) {
            state = json!({"last_step":step,"day":-1,"workers":{},"last_work":{},"seen_plants":[],"lost":[],"targets":(5..7).flat_map(|y|(5..10).map(move|x|json!([x,y]))).collect::<Vec<_>>()});
        }
        state["last_step"] = json!(step);
        if step == 432 {
            state["eligible"] = json!(Self::tomato_qualifies(obs));
        }
        if !truth(&state["eligible"]) || day < 18 {
            self.v219[&key] = state;
            return action;
        }
        if state["day"] != day {
            state["day"] = json!(day);
            state["workers"] = json!({});
            state["last_work"] = json!({});
        }
        let view = View::new(obs);
        let pending = remove(&mut state, "pending");
        if !pending.is_null() {
            let first = int(&pending["first_actor"]);
            let count = int(&pending["count"]);
            let crop = int(&pending["crop_workers"]);
            if view.positions.len() as i64 >= first + count && has_quad(&view.farm, "SE") {
                for index in 0..count {
                    let fertilizer_worker = index == crop;
                    let targets = if fertilizer_worker || crop == 1 {
                        state["targets"].clone()
                    } else if crop == 2 {
                        json!(arr(&state["targets"])[index as usize * 5..index as usize * 5 + 5])
                    } else {
                        json!([
                            [[5, 5], [6, 5], [7, 5]],
                            [[8, 5], [9, 5], [9, 6], [8, 6]],
                            [[5, 6], [6, 6], [7, 6]]
                        ])[index as usize]
                            .clone()
                    };
                    let mut role = json!({"kind":if fertilizer_worker {"fertilizer"}else{"crop"},"targets":targets,"needs_fertilizer":truth(&pending["fertilizer"])&&(day==24||fertilizer_worker)});
                    if !pending["labor"].is_null() {
                        role["targets"] = pending["labor"]["paths"][index as usize].clone();
                        role["needs_fertilizer"] = pending["labor"]["fertilizer"].clone();
                        role["fertilizer_quantity"] = json!(arr(&role["targets"]).len());
                        if view.positions[(first + index) as usize]
                            != pending["labor"]["spawns"][index as usize]
                        {
                            inc(&mut self.report, "labor_spawn_errors", 1);
                        }
                        if index == 0 {
                            inc(&mut self.report, "labor_confirmed", 1);
                        }
                    }
                    state["workers"][(first + index).to_string()] = role;
                }
                inc(&mut self.report, "confirmed_workers", count);
            } else {
                inc(&mut self.report, "hire_shortfalls", count);
            }
        }
        action = self.tomato_request(obs, action, &mut state, native);
        let roles = state["workers"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !roles.is_empty() {
            let mut work = commands(&action);
            while work.len() < view.positions.len() {
                work.push(json!(["PASS"]));
            }
            for (actor_key, mut role) in roles {
                let actor = actor_key.parse::<usize>().unwrap();
                if actor >= work.len() {
                    continue;
                }
                let command = self.tomato_worker(obs, &mut state, actor, &mut role);
                work[actor] = command.clone();
                let counter = match name(&command[0]) {
                    "PLANT" => Some("plant_requests"),
                    "WATER" => Some("water_requests"),
                    "FERTILIZE" => Some("fertilize_requests"),
                    "HARVEST" => Some("harvest_requests"),
                    "DROP" => Some("drop_requests"),
                    _ => None,
                };
                if let Some(c) = counter {
                    inc(&mut self.report, c, 1);
                }
                state["last_work"][&actor_key] = json!({"step":step,"command":command,"tomatoes":int(&view.inv(actor)["TOMATO"])});
                state["workers"][&actor_key] = role;
            }
            set_commands(&mut action, work);
        }
        let mut market = orders(&action);
        if truth(&state["committed"])
            && market.len() < 10
            && !market.iter().any(|o| pair(o, "SELL", "TOMATO"))
        {
            let quantity = int(&Core::projected_shed(&action, &view)["TOMATO"]);
            if quantity > 0 {
                market.push(json!(["SELL", "TOMATO", quantity]));
                set_orders(&mut action, market);
                inc(&mut self.report, "tomato_sale_requests", quantity);
            }
        }
        self.v219[&key] = state;
        action
    }

    fn cattle(&mut self, obs: &Value, mut action: Value) -> Value {
        let step = int(&obs["step"]);
        let key = int(&obs["player"]).to_string();
        let view = View::new(obs);
        let mut state = remove(&mut self.v231, &key);
        if state.is_null() || step <= int(&state["last"]) {
            state = json!({"last":-1,"confirmed":0,"reserved":0,"pending_buy":null,"carrying":{},"pending_places":[],"sites":{},"milk_credit":0,"requested":0,"failed_purchase_units":0,"picked":0,"placed":0,"failed_placements":0,"extra_milk_harvested":0,"extra_milk_sale_requests":0});
        }
        let pending = remove(&mut state, "pending_buy");
        if !pending.is_null() {
            let gained = (int(&view.shed["COW"]) - int(&pending["before"])).max(0);
            let confirmed = int(&pending["quantity"]).min(gained);
            inc(&mut state, "confirmed", confirmed);
            inc(&mut state, "reserved", confirmed);
            inc(
                &mut state,
                "failed_purchase_units",
                int(&pending["quantity"]) - confirmed,
            );
        }
        state["pending_buy"] = Value::Null;
        for p in arr(&state["pending_places"]).to_vec() {
            let site = position(&p["site"]);
            let site_key = format!("{},{}", site.0, site.1);
            let tile = tile_at(&view.farm, site);
            if tile["animal"] == "COW" && tile["placed_day"] == p["day"] {
                state["sites"][&site_key] = p["day"].clone();
                inc(&mut state, "placed", 1);
                let actor = int(&p["actor"]).to_string();
                state["carrying"][&actor] = json!((int(&state["carrying"][&actor]) - 1).max(0));
            } else {
                inc(&mut state, "failed_placements", 1);
            }
        }
        state["pending_places"] = json!([]);
        state["last"] = json!(step);
        let mut work = commands(&action);
        let mut seen = std::collections::HashSet::new();
        let mut occupied = std::collections::HashSet::new();
        let mut available = int(&view.shed["COW"]);
        for (actor, command) in work.iter_mut().enumerate().take(view.positions.len()) {
            let inv = view.inv(actor);
            let site = position(&view.positions[actor]);
            let sk = format!("{},{}", site.0, site.1);
            let tile = tile_at(&view.farm, site);
            let actor_key = actor.to_string();
            if *command == json!(["HARVEST"])
                && !state["sites"][&sk].is_null()
                && !seen.contains(&site)
                && tile["animal"] == "COW"
                && tile["placed_day"] == state["sites"][&sk]
            {
                let n = int(&tile["yield_units"]).max(0);
                inc(&mut state, "milk_credit", n);
                inc(&mut state, "extra_milk_harvested", n);
                seen.insert(site);
            }
            if pair(command, "PICKUP", "SHEEP") {
                let q = command.get(2).map(int).unwrap_or(1).max(0);
                let center = arr(&view.farm["tiles"]).len() as i64 / 2;
                if q != 0
                    && int(&state["reserved"]) >= q
                    && available >= q
                    && [center - 1, center].contains(&site.0)
                    && [center - 1, center].contains(&site.1)
                    && !["COW", "SHEEP", "GOOSE"].iter().any(|a| truth(&inv[*a]))
                {
                    command[1] = json!("COW");
                    inc(&mut state, "reserved", -q);
                    available -= q;
                    inc(&mut state["carrying"], &actor_key, q);
                    inc(&mut state, "picked", q);
                }
            }
            if pair(command, "PLACE", "SHEEP")
                && int(&state["carrying"][&actor_key]) > 0
                && int(&inv["COW"]) > 0
                && tile["kind"] == "PASTURE"
                && tile.get("animal").is_none()
                && !occupied.contains(&site)
            {
                command[1] = json!("COW");
                state["pending_places"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"actor":actor,"site":site,"day":step/24}));
            }
            if command[0] == "PLACE" && animal(&command[1]) && int(&inv[name(&command[1])]) > 0 {
                occupied.insert(site);
            }
        }
        set_commands(&mut action, work);
        let mut market = orders(&action);
        let animal_orders: Vec<_> = market
            .iter()
            .enumerate()
            .filter(|(_, o)| arr(o).len() >= 3 && o[0] == "BUY_ANIMAL")
            .map(|(i, _)| i)
            .collect();
        let shops = arr(&obs["town"]["unlocked_shops"]);
        let prices = &obs["market"]["prices"];
        let cows = arr(&view.tiles)
            .iter()
            .flat_map(arr)
            .filter(|t| t["animal"] == "COW")
            .count();
        let sheep = arr(&view.tiles)
            .iter()
            .flat_map(arr)
            .filter(|t| t["animal"] == "SHEEP")
            .count();
        let cargo: i64 = view
            .invs
            .iter()
            .map(|inv| {
                ["COW", "SHEEP", "GOOSE"]
                    .iter()
                    .map(|a| int(&inv[*a]))
                    .sum::<i64>()
            })
            .sum();
        let stock_animals: i64 = ["COW", "SHEEP", "GOOSE"]
            .iter()
            .map(|a| int(&view.shed[*a]))
            .sum();
        let milk_shops = shops
            .iter()
            .filter(|s| ["PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"].contains(&name(s)))
            .count();
        if (216..=227).contains(&step)
            && shops.len() >= 3
            && int(&state["confirmed"]) < 4
            && !truth(&state["reserved"])
            && !state["carrying"].as_object().unwrap().values().any(truth)
            && arr(&state["pending_places"]).is_empty()
            && cargo == 0
            && stock_animals == 0
            && animal_orders.len() == 1
            && market[animal_orders[0]][1] == "SHEEP"
            && milk_shops >= 2
            && !shops.iter().any(|s| s == "YARN_STORE")
            && int(&prices["MILK"]) >= int(&prices["WOOL"])
            && cows >= 4
            && sheep >= 2
        {
            let i = animal_orders[0];
            let q = int(&market[i][2]);
            if (1..=2).contains(&q) && q <= 4 - int(&state["confirmed"]) {
                market[i][1] = json!("COW");
                inc(&mut state, "requested", q);
                state["pending_buy"] = json!({"before":int(&view.shed["COW"]),"quantity":q});
            }
        }
        if int(&state["milk_credit"]) > 0 {
            let stock = Core::projected_shed(&action, &view);
            let planned: i64 = market
                .iter()
                .filter(|o| arr(o).len() >= 3 && pair(o, "SELL", "MILK"))
                .map(|o| int(&o[2]).max(0))
                .sum();
            let extra = int(&state["milk_credit"]).min((int(&stock["MILK"]) - planned).max(0));
            if extra != 0
                && let Some(order) = market
                    .iter_mut()
                    .find(|o| arr(o).len() >= 3 && pair(o, "SELL", "MILK") && int(&o[2]) > 0)
            {
                order[2] = json!(int(&order[2]) + extra);
                inc(&mut state, "milk_credit", -extra);
                inc(&mut state, "extra_milk_sale_requests", extra);
            }
        }
        set_orders(&mut action, market);
        for k in [
            "confirmed",
            "reserved",
            "requested",
            "failed_purchase_units",
            "picked",
            "placed",
            "failed_placements",
            "extra_milk_harvested",
            "extra_milk_sale_requests",
            "milk_credit",
        ] {
            self.report[format!("cattle_{k}")] = state[k].clone();
        }
        self.report["cattle_carried_pending"] = json!(sum_stock(&state["carrying"]));
        self.v231[&key] = state;
        action
    }

    fn reserve(&mut self, obs: &Value, mut action: Value, native: &mut Value) -> Value {
        let step = int(&obs["step"]);
        if !(288..696).contains(&step) {
            return action;
        }
        let key = int(&obs["player"]).to_string();
        let horizon = self.horizons.get(&key).map(int).unwrap_or(2);
        let end = 695.min(step + horizon).min((step / 72 + 1) * 72 - 1);
        if end <= step {
            return action;
        }
        let work = commands(&action);
        let view = View::new(obs);
        if work
            .iter()
            .enumerate()
            .take(view.positions.len())
            .any(|(i, c)| c[0] == "PLACE" && animal(&c[1]) && int(&view.inv(i)[name(&c[1])]) > 0)
        {
            return action;
        }
        let stock = Core::projected_shed(&action, &view);
        let mut market = orders(&action);
        let mut blocked = std::collections::HashSet::new();
        for o in &market {
            if arr(o).len() > 1 && ["SELL", "BUY_PRODUCT"].contains(&name(&o[0])) {
                blocked.insert(name(&o[1]).to_owned());
            }
        }
        for c in &work {
            if pair(c, "PICKUP", name(&c[1])) {
                blocked.insert(name(&c[1]).to_owned());
            }
        }
        for queue in native["pending"]
            .as_object()
            .into_iter()
            .flat_map(|m| m.values())
        {
            for pc in arr(queue) {
                let c = &pc[1];
                if c[0] == "PICKUP" && arr(c).len() > 1 {
                    blocked.insert(name(&c[1]).to_owned());
                }
            }
        }
        if native["sell_state"]["r36_debts"].is_null() {
            native["sell_state"]["r36_debts"] = json!({});
        }
        let route = int(&native["route"]);
        let debts = &mut native["sell_state"]["r36_debts"];
        for item in PRODUCTS {
            if ["WHEAT", "FERTILIZER"].contains(&item)
                || blocked.contains(item)
                || int(&view.prices[item]) < 2
            {
                continue;
            }
            let mut available = int(&stock[item]).max(0);
            if available == 0 || market.len() >= 10 {
                continue;
            }
            let mut reservations = vec![];
            for due in step + 1..=end {
                let future = route_action(route, due);
                if commands(&future).iter().any(|c| pair(c, "PICKUP", item))
                    || orders(&future).iter().any(|o| pair(o, "BUY_PRODUCT", item))
                {
                    break;
                }
                let planned: i64 = orders(&future)
                    .iter()
                    .filter(|o| arr(o).len() >= 3 && pair(o, "SELL", item))
                    .map(|o| int(&o[2]).max(0))
                    .sum();
                let amount = available.min((planned - int(&debts[due.to_string()][item])).max(0));
                if amount != 0 {
                    reservations.push((due, amount));
                    available -= amount;
                }
                if available == 0 {
                    break;
                }
            }
            let q: i64 = reservations.iter().map(|(_, q)| q).sum();
            if q != 0 {
                market.push(json!(["SELL", item, q]));
                for (due, n) in reservations {
                    if debts[due.to_string()].is_null() {
                        debts[due.to_string()] = json!({});
                    }
                    inc(&mut debts[due.to_string()], item, n);
                }
                inc(&mut self.report, "sale_reserved_units", q);
                inc(&mut self.report, "sale_reservations", 1);
            }
        }
        set_orders(&mut action, market);
        action
    }

    fn reorder_sales(&mut self, obs: &Value, mut action: Value) -> Value {
        let stock = Core::projected_shed(&action, &View::new(obs));
        let mut market = orders(&action);
        let original = market.clone();
        let mut start = 0;
        while start < market.len() {
            if market[start][0] != "SELL" {
                start += 1;
                continue;
            }
            let mut end = start;
            while end < market.len() && market[end][0] == "SELL" {
                end += 1;
            }
            let distinct: std::collections::HashSet<_> =
                market[start..end].iter().map(|o| name(&o[1])).collect();
            if distinct.len() == end - start {
                market[start..end]
                    .sort_by_key(|o| std::cmp::Reverse(quote_priority(obs, o, &stock)));
            }
            start = end;
        }
        if market != original {
            inc(&mut self.report, "quote_reordered_turns", 1);
            set_orders(&mut action, market);
        }
        action
    }

    fn sheep_eligible(obs: &Value, native: &Value) -> bool {
        let farm = &obs["farms"][int(&obs["player"]) as usize];
        let prices = &obs["market"]["prices"];
        if arr(&farm["tiles"]).len() != 10
            || !exactly_three_quads(farm)
            || arr(&obs["town"]["unlocked_shops"])
                .iter()
                .filter(|s| *s == "YARN_STORE")
                .count()
                < 2
            || num(&prices["WOOL"]) < 220.0
            || num(&prices["WHEAT"]) > 45.0
        {
            return false;
        }
        if (5..7).any(|y| (5..8).any(|x| farm["tiles"][y][x] != "LOCKED"))
            || truth(&obs["private"]["shed"]["SHEEP"])
            || arr(&obs["private"]["inventories"])
                .iter()
                .any(|i| truth(&i["SHEEP"]))
        {
            return false;
        }
        for day in 12..30 {
            for a in native_day(native, day) {
                if orders(&a)
                    .iter()
                    .any(|o| o[0] == "BUY_LAND" || pair(o, "BUY_ANIMAL", "SHEEP"))
                    || commands(&a)
                        .iter()
                        .any(|c| ["PICKUP", "PLACE"].contains(&name(&c[0])) && c[1] == "SHEEP")
                {
                    return false;
                }
            }
        }
        true
    }

    fn sheep_request(
        &mut self,
        obs: &Value,
        mut action: Value,
        state: &mut Value,
        native: &Value,
    ) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        let hour = step % 24;
        let committed = truth(&state["committed"]);
        if hour > if committed { 2 } else { 1 } || state["requested_day"] == day {
            return action;
        }
        if !committed && (day != 12 || !Self::sheep_eligible(obs, native)) {
            return action;
        }
        let planned = native_day(native, day);
        if planned
            .iter()
            .skip((hour + 1) as usize)
            .any(|a| orders(a).iter().any(|o| o[0] == "HIRE"))
        {
            return action;
        }
        let view = View::new(obs);
        let mut market = orders(&action);
        let parent_hires = market.iter().filter(|o| o[0] == "HIRE").count() as i64;
        let expected = planned
            .iter()
            .map(|a| arr(&a["hands"]).len())
            .max()
            .unwrap_or(0) as i64;
        if view.positions.len() as i64 - 1 + parent_hires != expected {
            return action;
        }
        let initial = !committed;
        let mut extra = vec![];
        if initial {
            extra.extend([json!(["BUY_LAND"]), json!(["BUY_ANIMAL", "SHEEP", 6])]);
        }
        extra.extend([
            json!(["BUY_PRODUCT", "WHEAT", 6]),
            json!(["HIRE"]),
            json!(["HIRE"]),
        ]);
        if market.len() + extra.len() > 10 {
            return action;
        }
        let stock = Core::projected_shed(&action, &view);
        let mut incoming = 6 + 6 * i64::from(initial);
        let mut budget =
            7000 * i64::from(initial) + 6 * (int(&obs["market"]["prices"]["WHEAT"]) + 10);
        budget += (view.hires_today..view.hires_today + parent_hires + 2)
            .map(fib)
            .sum::<i64>();
        for o in &market {
            if o[0] == "BUY_LAND" {
                return action;
            }
            if ["BUY_PRODUCT", "BUY_ANIMAL"].contains(&name(&o[0])) {
                incoming += int(&o[2]);
            }
            budget += order_cost(obs, o);
        }
        if sum_stock(&stock) + incoming > 100 {
            inc(&mut self.report, "sheep_capacity_declines", 1);
            return action;
        }
        if view.money < (budget + if initial { 3000 } else { 1000 }) as f64 {
            inc(&mut self.report, "sheep_budget_declines", 1);
            return action;
        }
        state["requested_day"] = json!(day);
        state["pending"] = json!({"first":expected+1,"initial":initial});
        inc(&mut self.report, "sheep_hire_requests", 2);
        inc(&mut self.report, "sheep_feed_buy_requests", 6);
        if initial {
            inc(&mut self.report, "sheep_commit_requests", 1);
        }
        market.extend(extra);
        set_orders(&mut action, market);
        action
    }

    fn sheep_worker(obs: &Value, actor: usize, targets: &Value) -> Value {
        let view = View::new(obs);
        let step = int(&obs["step"]);
        let pos = position(&view.positions[actor]);
        let inv = view.inv(actor);
        let dest = [(4, 4), (5, 4), (4, 5), (5, 5)]
            .into_iter()
            .min_by_key(|p| (distance(pos, *p), *p))
            .unwrap();
        let cargo: Vec<_> = ["WOOL", "FERTILIZER"]
            .into_iter()
            .filter(|i| truth(&inv[*i]))
            .collect();
        if !cargo.is_empty()
            && step % 24 >= if step / 24 == 29 { 22 } else { 23 } - distance(pos, dest)
        {
            return walk(pos, dest).unwrap_or_else(|| json!(["PLACE", cargo[0], inv[cargo[0]]]));
        }
        let missing = arr(targets)
            .iter()
            .filter(|p| tile_at(&view.farm, position(p))["animal"] != "SHEEP")
            .count() as i64;
        if missing != 0 && !truth(&inv["SHEEP"]) && truth(&view.shed["SHEEP"]) {
            return walk(pos, dest).unwrap_or_else(|| {
                json!(["PICKUP", "SHEEP", missing.min(int(&view.shed["SHEEP"]))])
            });
        }
        let hungry = arr(targets)
            .iter()
            .filter(|p| !truth(&tile_at(&view.farm, position(p))["fed_today"]))
            .count() as i64;
        if hungry != 0 && !truth(&inv["WHEAT"]) && truth(&view.shed["WHEAT"]) {
            return walk(pos, dest).unwrap_or_else(|| {
                json!(["PICKUP", "WHEAT", hungry.min(int(&view.shed["WHEAT"]))])
            });
        }
        let mut tasks = vec![];
        for (index, target) in arr(targets).iter().enumerate() {
            let p = position(target);
            let t = tile_at(&view.farm, p);
            let c = if t.is_null() {
                Some(json!(["BUILD_PASTURE"]))
            } else if t["kind"] == "WEED" {
                Some(json!(["DIG"]))
            } else if t["kind"] == "PASTURE" && !truth(&t["animal"]) {
                if truth(&inv["SHEEP"]) {
                    Some(json!(["PLACE", "SHEEP"]))
                } else {
                    None
                }
            } else if t["animal"] == "SHEEP" {
                if !truth(&t["fed_today"]) && truth(&inv["WHEAT"]) {
                    Some(json!(["FEED"]))
                } else if !truth(&t["cared_today"]) {
                    Some(json!(["CARE"]))
                } else if truth(&t["yield_units"]) {
                    Some(json!(["HARVEST"]))
                } else if truth(&t["fertilizer_available"]) {
                    Some(json!(["COLLECT_FERTILIZER"]))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(c) = c {
                tasks.push((distance(pos, p), index, p, c));
            }
        }
        if let Some((_, _, p, c)) = tasks.into_iter().min_by_key(|v| (v.0, v.1)) {
            return walk(pos, p).unwrap_or(c);
        }
        if !cargo.is_empty() {
            return walk(pos, dest).unwrap_or_else(|| json!(["PLACE", cargo[0], inv[cargo[0]]]));
        }
        json!(["PASS"])
    }

    fn sheep_rescue(&mut self, obs: &Value, mut action: Value, state: &mut Value) -> Value {
        if !truth(&state["workers"]) || int(&obs["step"]) % 24 > 14 {
            return action;
        }
        let mut market = orders(&action);
        if market.len() >= 10
            || market.iter().any(|o| {
                ["HIRE", "BUY_LAND", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_SEED"].contains(&name(&o[0]))
                    || o[1] == "WHEAT"
            })
        {
            return action;
        }
        let view = View::new(obs);
        let work = commands(&action);
        let mut hungry = 0;
        let mut carried = 0;
        for (actor, targets) in state["workers"].as_object().unwrap() {
            let actor = actor.parse::<usize>().unwrap();
            let c = &work[actor];
            if *c == json!(["FEED"]) || pair(c, "PICKUP", "WHEAT") {
                return action;
            }
            carried += int(&view.inv(actor)["WHEAT"]);
            hungry += arr(targets)
                .iter()
                .filter(|p| {
                    let t = tile_at(&view.farm, position(p));
                    t["animal"] == "SHEEP" && !truth(&t["fed_today"])
                })
                .count() as i64;
        }
        let stock = Core::projected_shed(&action, &view);
        let shortage = hungry - carried - int(&stock["WHEAT"]);
        if !(1..=6).contains(&shortage) || int(&state["rescue_today"]) + shortage > 6 {
            return action;
        }
        let quote = int(&obs["market"]["prices"]["WHEAT"]);
        if quote < 1
            || view.money < (1000 + shortage * (quote + 10)) as f64
            || sum_stock(&stock) + shortage > 100
        {
            return action;
        }
        market.push(json!(["BUY_PRODUCT", "WHEAT", shortage]));
        set_orders(&mut action, market);
        inc(state, "rescue_today", shortage);
        inc(&mut self.report, "sheep_rescue_feed_requests", shortage);
        action
    }

    fn sheep(
        &mut self,
        obs: &Value,
        mut action: Value,
        native: &Value,
        valid_config: bool,
    ) -> Value {
        let step = int(&obs["step"]);
        let day = step / 24;
        let key = int(&obs["player"]).to_string();
        let mut state = remove(&mut self.v233, &key);
        if state.is_null() || step <= int(&state["last_step"]) {
            state = json!({"last_step":step,"day":-1,"workers":{},"work":{},"credit":{"WOOL":0,"FERTILIZER":0}});
        }
        state["last_step"] = json!(step);
        if !valid_config || day < 12 {
            self.v233[&key] = state;
            return action;
        }
        let view = View::new(obs);
        if state["day"] != day {
            state["day"] = json!(day);
            state["workers"] = json!({});
            state["work"] = json!({});
            state["rescue_today"] = json!(0);
        }
        for (actor, previous) in state["work"].as_object().unwrap().clone() {
            let actor = actor.parse::<usize>().unwrap();
            if previous["step"] != step - 1 || actor >= view.invs.len() {
                continue;
            }
            let item = match name(&previous["command"][0]) {
                "HARVEST" => Some("WOOL"),
                "COLLECT_FERTILIZER" => Some("FERTILIZER"),
                _ => None,
            };
            if let Some(item) = item {
                let gained =
                    (int(&view.inv(actor)[item]) - int(&previous["inventory"][item])).max(0);
                inc(&mut state["credit"], item, gained);
                inc(
                    &mut self.report,
                    if item == "WOOL" {
                        "sheep_wool_harvested"
                    } else {
                        "sheep_fert_collected"
                    },
                    gained,
                );
            }
        }
        let pending = remove(&mut state, "pending");
        if !pending.is_null() {
            let initial = truth(&pending["initial"]);
            let first = int(&pending["first"]);
            if !has_quad(&view.farm, "SE") || (initial && int(&view.shed["SHEEP"]) < 6) {
                inc(&mut self.report, "sheep_purchase_shortfalls", 1);
            } else if (view.positions.len() as i64 - 1) < first + 1 {
                inc(&mut self.report, "sheep_hire_shortfalls", 1);
            } else {
                for i in 0..2 {
                    state["workers"][(first + i).to_string()] =
                        json!((5..8).map(|x| json!([x, 5 + i])).collect::<Vec<_>>());
                }
                inc(&mut self.report, "sheep_workers_confirmed", 2);
                if initial {
                    state["committed"] = json!(true);
                    inc(&mut self.report, "sheep_committed", 1);
                }
            }
        }
        action = self.sheep_request(obs, action, &mut state, native);
        if !truth(&state["committed"]) {
            self.v233[&key] = state;
            return action;
        }
        let mut work = commands(&action);
        while work.len() < view.positions.len() {
            work.push(json!(["PASS"]));
        }
        state["work"] = json!({});
        for (actor, targets) in state["workers"].as_object().unwrap().clone() {
            let index = actor.parse::<usize>().unwrap();
            let c = Self::sheep_worker(obs, index, &targets);
            work[index] = c.clone();
            state["work"][actor] = json!({"step":step,"command":c,"inventory":view.inv(index)});
        }
        set_commands(&mut action, work);
        action = self.sheep_rescue(obs, action, &mut state);
        let stock = Core::projected_shed(&action, &view);
        let mut market = orders(&action);
        for item in ["WOOL", "FERTILIZER"] {
            let scheduled: i64 = market
                .iter()
                .filter(|o| pair(o, "SELL", item))
                .map(|o| int(&o[2]))
                .sum();
            let count = int(&state["credit"][item]).min((int(&stock[item]) - scheduled).max(0));
            if count != 0 && market.len() < 10 {
                market.push(json!(["SELL", item, count]));
                inc(&mut state["credit"], item, -count);
                inc(
                    &mut self.report,
                    if item == "WOOL" {
                        "sheep_extra_wool_sales"
                    } else {
                        "sheep_extra_fert_sales"
                    },
                    count,
                );
            }
        }
        set_orders(&mut action, market);
        self.v233[&key] = state;
        action
    }

    pub fn committed(&self, seat: i64) -> bool {
        let key = seat.to_string();
        truth(&self.v219[&key]["committed"]) || truth(&self.v233[&key]["committed"])
    }
}

pub fn similarity(obs: &Value) -> f64 {
    let seat = int(&obs["player"]) as usize;
    let own = &obs["farms"][seat];
    let rival = &obs["farms"][1 - seat];
    if own["unlocked_quadrants"] != rival["unlocked_quadrants"] {
        return 0.0;
    }
    let mut matches = 0;
    let mut total = 0;
    for (a, b) in arr(&own["tiles"])
        .iter()
        .flat_map(arr)
        .zip(arr(&rival["tiles"]).iter().flat_map(arr))
    {
        let sa = (&a["crop"], &a["animal"]);
        let sb = (&b["crop"], &b["animal"]);
        if !sa.0.is_null() || !sa.1.is_null() || !sb.0.is_null() || !sb.1.is_null() {
            total += 1;
            if sa == sb {
                matches += 1;
            }
        }
    }
    if total >= 8 {
        matches as f64 / total as f64
    } else {
        0.0
    }
}

fn shape(func: &str, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match func {
        "linear" => x,
        "sq" => x * x,
        "sqrt" => x.sqrt(),
        "log" => (1.0 + x).ln(),
        "log10" => (1.0 + x).log10(),
        "hinge" => {
            if t <= 0.0 {
                x
            } else {
                let u = x / t;
                u + 8.0 * (u - 1.0).max(0.0).powi(2)
            }
        }
        _ => x,
    }
}
pub fn market_price(item: &str, inventory: f64, params: Option<&Value>) -> i64 {
    let (base, t, below, bt, above, at) = match item {
        "WHEAT" => (25.0, 400.0, "sqrt", 0.8, "log", 0.2),
        "CARROT" => (35.0, 450.0, "hinge", 1.0, "sqrt", 0.7),
        "TOMATO" => (60.0, 200.0, "hinge", 0.4, "sqrt", 0.6),
        "STRAWBERRY" => (120.0, 100.0, "sqrt", 0.7, "linear", 1.6),
        "MELON" => (250.0, 300.0, "log", 0.2, "sq", 3.6),
        "EGG" => (50.0, 332.0, "hinge", 0.4, "log", 0.2),
        "MILK" => (160.0, 122.0, "sqrt", 0.6, "linear", 1.6),
        "WOOL" => (200.0, 105.0, "log", 0.2, "sq", 3.2),
        "FERTILIZER" => (100.0, 200.0, "linear", 0.4, "linear", 0.4),
        _ => return 1,
    };
    let p = params.and_then(|m| m.get(item));
    let get_num = |k: &str, default: f64| p.and_then(|v| v.get(k)).map(num).unwrap_or(default);
    let get_str = |k: &str, default: &str| {
        p.and_then(|v| v.get(k))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| default.to_owned())
    };
    let base = get_num("base", base);
    let i0 = get_num("I0", 10000.0);
    let t = get_num("T", t);
    let price = if inventory < i0 {
        let f = get_str("below_func", below);
        base + get_num("below_target", bt) * base / shape(&f, t, t) * shape(&f, i0 - inventory, t)
    } else {
        let f = get_str("above_func", above);
        base - get_num("above_target", at) * base / shape(&f, t, t) * shape(&f, inventory - i0, t)
    };
    (price.round_ties_even() as i64).max(1)
}

fn quote_priority(obs: &Value, order: &Value, stock: &Value) -> i64 {
    let item = name(&order[1]);
    let quantity = int(&order[2]).max(0).min(int(&stock[item]));
    if quantity == 0 || !PRODUCTS.contains(&item) {
        return 0;
    }
    let inventory = num(&obs["market"]["inventory"][item]);
    let rival = &obs["farms"][1 - int(&obs["player"]) as usize];
    let crop = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"].contains(&item);
    let animal = match item {
        "EGG" => Some("GOOSE"),
        "MILK" => Some("COW"),
        "WOOL" => Some("SHEEP"),
        _ => None,
    };
    let standing: i64 = arr(&rival["tiles"])
        .iter()
        .flat_map(arr)
        .filter(|t| (crop && t["crop"] == item) || animal.is_some_and(|a| t["animal"] == a))
        .map(|t| int(&t["yield_units"]).max(0))
        .sum();
    let batch = standing.clamp(8, 24);
    let params = obs["market"].get("params");
    let now: i64 = (0..quantity)
        .map(|j| market_price(item, inventory + j as f64, params))
        .sum();
    let later: i64 = (0..quantity)
        .map(|j| market_price(item, inventory + (batch + j) as f64, params))
        .sum();
    now - later
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_python_production_helper_oracle() {
        let path=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/v39-production-parity.json");
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(
            fixture["source_sha256"],
            "708c7485fa964853b193f175dcd83020602c005e159ce82e38dc350b22e970c8"
        );
        assert_eq!(fixture["engine_transitions"], 0);
        let mut count = 0;
        for case in arr(&fixture["cases"]) {
            let mut policy = Production::default();
            let obs = &case["observation"];
            let label = name(&case["name"]);
            let actual = match name(&case["kind"]) {
                "price" => json!(market_price(
                    name(&case["item"]),
                    num(&case["inventory"]),
                    None
                )),
                "sales_first" => policy.sales_first(case["action"].clone()),
                "sheep_worker" => {
                    Production::sheep_worker(obs, int(&case["actor"]) as usize, &case["targets"])
                }
                "tomato_worker" => {
                    let mut state = case["state"].clone();
                    let mut role = case["role"].clone();
                    let result = policy.tomato_worker(
                        obs,
                        &mut state,
                        int(&case["actor"]) as usize,
                        &mut role,
                    );
                    assert_eq!(state, case["next_state"], "{label}: tomato state");
                    assert_eq!(role, case["next_role"], "{label}: tomato role");
                    result
                }
                "reserve" => {
                    let key = int(&obs["player"]).to_string();
                    policy.horizons[&key] = json!(4);
                    let mut native = case["native"].clone();
                    let result = policy.reserve(obs, case["action"].clone(), &mut native);
                    assert_eq!(native, case["next_native"], "{label}: reservation debts");
                    result
                }
                "cattle" => {
                    let key = int(&obs["player"]).to_string();
                    policy.v231[&key] = case["state"].clone();
                    let result = policy.cattle(obs, case["action"].clone());
                    assert_eq!(
                        policy.v231[&key], case["next_state"],
                        "{label}: cattle custody"
                    );
                    result
                }
                kind => panic!("unsupported oracle kind {kind}"),
            };
            assert_eq!(actual, case["expected"], "{label}: action/quote");
            count += 1;
        }
        assert_eq!(count, 104);
    }
}
