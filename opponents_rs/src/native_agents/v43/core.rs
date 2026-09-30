//! Active chassis and storage wrappers from Ahmed Berat Ozer's frozen V43.
//! Source attribution and Apache-2.0 notices remain in agents/v43/main.py.
//! The public `_SETTINGS` disables the generic budget, room, clamp, dead-stock,
//! terminal and opponent-plan layers. Only the actual enabled policy is executed.

use super::common::*;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Core {
    pub players: Value,
    pub diagnostics: Value,
    /// Thomas Tschinkel "2945 Farm" (v9) lineage: V39 opening, yarn worlds on
    /// route 9 (one rival-keyed route 128), lead sales from step 288 and the
    /// RACEPX glut gate. False for the frozen V43/V47/V48 controllers.
    pub farm2945: bool,
    /// Dmitrii Gluzdov's Smaller Market Shock opening overlay.  This is kept
    /// separate from `farm2945` so the frozen parent remains byte-for-byte
    /// unchanged when the overlay is disabled.
    pub small_market: bool,
    /// Metav4 source constants and optional Pipe16 hybrid route-zero opening.
    pub metav4: bool,
    pub hybrid_opening: bool,
    /// V56 mature-wheat route-zero overlay atop Pipe16 HybridOpening.
    pub v56: bool,
    /// Frozen cha22 settings and EarlyCycle mature-wheat opening.
    pub cha22: bool,
    pub racepx_report: Value,
}

impl Default for Core {
    fn default() -> Self {
        Self {
            players: json!({}),
            diagnostics: json!({"layer_fallbacks":0,"entry_fallbacks":0,
                "terminal_rescue_errors":0,"v28_entry_errors":0}),
            farm2945: false,
            small_market: false,
            metav4: false,
            hybrid_opening: false,
            v56: false,
            cha22: false,
            racepx_report: json!({"racepx_skipped":0,"racepx_sold":0,"racepx_errors":0}),
        }
    }
}

fn animal_structure(item: &str) -> Option<&'static str> {
    match item {
        "GOOSE" => Some("COOP"),
        "COW" | "SHEEP" => Some("PASTURE"),
        _ => None,
    }
}

pub fn is_noop(
    act: &Value,
    tile: &Value,
    inv: &Value,
    seeds: &Value,
    pos: &Value,
    board: i64,
) -> bool {
    if !truth(act) {
        return true;
    }
    let op = text(&act[0]);
    let (x, y) = position(pos);
    let movement = match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    };
    if let Some((dx, dy)) = movement {
        return !(0 <= x + dx && x + dx < board && 0 <= y + dy && y + dy < board);
    }
    let adjacent = shed_adjacent(pos, board);
    let item = text(&act[1]);
    match op {
        "PASS" => return true,
        "DROP" => return !adjacent || !truth(inv),
        "PICKUP" => return !adjacent,
        "PLACE" => {
            if animal_structure(item).is_some_and(|kind| text(&tile["kind"]) == kind)
                && tile.is_object()
                && tile["animal"].is_null()
            {
                return int(&inv[item]) <= 0;
            }
            return !adjacent || int(&inv[item]) <= 0;
        }
        _ => {}
    }
    if tile == "LOCKED" {
        return true;
    }
    let kind = text(&tile["kind"]);
    let animal = tile.is_object() && !tile["animal"].is_null();
    match op {
        "PLANT" => !tile.is_null() || int(&seeds[item]) <= 0,
        "WATER" => kind != "PLANT" || truth(&tile["watered_today"]),
        "HARVEST" => !tile.is_object() || int(&tile["yield_units"]) <= 0,
        "FERTILIZE" => kind != "PLANT" || int(&inv["FERTILIZER"]) <= 0,
        "DIG" => tile.is_null() || animal,
        "BUILD_COOP" | "BUILD_PASTURE" => !tile.is_null(),
        "FEED" => !animal || truth(&tile["fed_today"]) || int(&inv["WHEAT"]) <= 0,
        "COLLECT_FERTILIZER" => !animal || !truth(&tile["fertilizer_available"]),
        "CARE" => !animal || truth(&tile["cared_today"]),
        _ => true,
    }
}

/// V39 opening kept by the 2945 Farm (`_R42_OPENING`); the route fixture holds V43's.
fn farm2945_opening() -> Value {
    json!([
        ["BUY_PRODUCT", "WHEAT", 13],
        ["BUY_PRODUCT", "WHEAT", 30],
        ["SELL", "WHEAT", 30]
    ])
}

/// The Smaller Market Shock source mutates route 0's first 96 rows before
/// invoking the complete 2945 Farm parent.  Apply those same route edits at
/// the route-action boundary, before hand alignment and all parent layers.
/// The source's tape assertions are true for the frozen fixture; conditional
/// writes keep malformed external observations fail-closed instead of
/// inventing a new route.
fn smaller_market_route(mut action: Value, step: i64) -> Value {
    let mut set_hand = |index: usize, command: Value| {
        if let Some(hands) = action["hands"].as_array_mut()
            && let Some(slot) = hands.get_mut(index)
        {
            *slot = command;
        }
    };
    match step {
        0 => action["market"] = json!([["BUY_PRODUCT", "WHEAT", 5], ["BUY_SEED", "WHEAT", 1]]),
        1 => {
            let kept: Vec<Value> = array(&action["market"])
                .iter()
                .filter(|order| {
                    !(array(order).len() >= 3
                        && (order[0] == "BUY_PRODUCT" || order[0] == "SELL")
                        && order[1] == "WHEAT")
                })
                .cloned()
                .collect();
            action["market"] = json!(kept);
        }
        2 => set_hand(1, json!(["WEST"])),
        3 => set_hand(1, json!(["WEST"])),
        4 => set_hand(1, json!(["WEST"])),
        5 => set_hand(1, json!(["PLANT", "WHEAT"])),
        6 => set_hand(1, json!(["WATER"])),
        29 => set_hand(2, json!(["WATER"])),
        49 => set_hand(0, json!(["WEST"])),
        50 => set_hand(0, json!(["WEST"])),
        51 => set_hand(0, json!(["WEST"])),
        52 => set_hand(0, json!(["WATER"])),
        53 => set_hand(0, json!(["HARVEST"])),
        54 => set_hand(0, json!(["BUILD_PASTURE"])),
        55 => set_hand(0, json!(["EAST"])),
        56 => set_hand(0, json!(["EAST"])),
        57 => set_hand(0, json!(["DROP"])),
        _ => {}
    }
    action
}

/// Controller-local route overlay: never mutates the shared frozen fixtures.
fn configured_route_action(
    route: i64,
    step: i64,
    farm: bool,
    smaller: bool,
    hybrid: bool,
    v56: bool,
) -> Value {
    let mut action = route_action(route, step);
    if smaller && route == 0 && (0..96).contains(&step) {
        action = smaller_market_route(action, step);
    }
    if farm && !smaller && step == 0 && action["market"].is_array() {
        action["market"] = farm2945_opening();
    }
    if hybrid && route == 0 && (0..96).contains(&step) {
        if step == 0 {
            action["market"]
                .as_array_mut()
                .unwrap()
                .push(json!(["BUY_SEED", "WHEAT", 1]));
        } else if step > 1 {
            action = smaller_market_route(action, step);
        }
    }
    if v56 && route == 0 {
        let command = match step {
            53..=57 => Some("PASS"),
            84 | 85 | 89 | 90 => Some("EAST"),
            86 => Some("WATER"),
            87 => Some("HARVEST"),
            88 => Some("BUILD_PASTURE"),
            91 => Some("DROP"),
            _ => None,
        };
        if let Some(command) = command {
            action["hands"][0] = json!([command]);
        }
    }
    action
}

const SHOP_NAMES: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];

/// RACEPX/RACEGATE base quotes (`V9_RACEPX_BASE`, `V9_RACEGATE_BASE`).
pub fn farm2945_base_price(item: &str) -> Option<i64> {
    Some(match item {
        "WHEAT" => 25,
        "CARROT" => 35,
        "TOMATO" => 60,
        "STRAWBERRY" => 120,
        "MELON" => 250,
        "EGG" => 50,
        "MILK" => 160,
        "WOOL" => 200,
        "FERTILIZER" => 100,
        _ => return None,
    })
}

/// 2945 Farm `_router`: turn-2 rival key, then `_V92_TABLE` / `_V93_ROUTE_BY_RIVAL`.
fn router_farm2945(observation: &Value, step: i64, state: &mut Value, cha22: bool) -> i64 {
    if step == 2 {
        let rival = &observation["farms"][(1 - int(&observation["player"])) as usize];
        let wheat = &observation["market"]["inventory"]["WHEAT"];
        state["rkey"] = if rival["money"].is_number() && wheat.is_number() {
            json!([
                crate::native_agents::farm2945::util::py_round3(num(&rival["money"])),
                int(wheat)
            ])
        } else {
            Value::Null
        };
    }
    if step >= 144 && !truth(&state["day6"]) {
        let all = array(&observation["town"]["unlocked_shops"]);
        let shops = &all[..all.len().min(2)];
        let yarn = shops.iter().any(|shop| shop == "YARN_STORE");
        state["expert"] = json!(if yarn { "V39" } else { "EXP240" });
        let mut route = if !yarn {
            shop_route(shops)
        } else if shops.len() == 2 && shops.iter().all(|s| SHOP_NAMES.contains(&text(s))) {
            9
        } else {
            0
        };
        if !cha22 && yarn && state["rkey"] == json!([229.0, 9989]) {
            route = 128;
        }
        state["route"] = json!(route);
        state["day6"] = json!(true);
    }
    if step >= 648 && !truth(&state["day27"]) {
        state["route"] = json!(2);
        state["day27"] = json!(true);
    }
    int(&state["route"])
}

fn router(observation: &Value, step: i64, state: &mut Value) -> i64 {
    if step >= 144 && !truth(&state["day6"]) {
        let shops = &observation["town"]["unlocked_shops"];
        let old_route = match (text(&shops[0]), text(&shops[1])) {
            ("BAKERY", "YARN_STORE") => 3,
            ("BRUNCH_SPOT", "YARN_STORE") => 4,
            ("FARMERS_MARKET" | "PET_CAFE", "YARN_STORE") => 5,
            ("ICE_CREAM_SHOP", "YARN_STORE") => 6,
            ("PIZZA_SHOP", "YARN_STORE") => 7,
            ("SMOOTHIE_SHOP", "YARN_STORE") => 8,
            ("YARN_STORE", "BAKERY" | "BRUNCH_SPOT" | "ICE_CREAM_SHOP") => 9,
            ("YARN_STORE", "FARMERS_MARKET") => 1,
            ("YARN_STORE", "PET_CAFE") => 10,
            ("YARN_STORE", "PIZZA_SHOP") => 6,
            ("YARN_STORE", "SMOOTHIE_SHOP") => 11,
            ("YARN_STORE", "YARN_STORE") => 12,
            _ => 0,
        };
        let shops = &array(shops)[..array(shops).len().min(2)];
        let use_new = !shops.iter().any(|shop| shop == "YARN_STORE");
        state["expert"] = json!(if use_new { "EXP240" } else { "V39" });
        state["route"] = json!(if use_new {
            shop_route(shops)
        } else {
            old_route
        });
        state["day6"] = json!(true);
    }
    if step >= 648 && !truth(&state["day27"]) {
        state["route"] = json!(2);
        state["day27"] = json!(true);
    }
    int(&state["route"])
}

impl Core {
    pub fn player_state(&self, player: usize) -> &Value {
        &self.players[player.to_string()]
    }

    pub fn set_player_state(&mut self, player: usize, state: Value) {
        self.players[player.to_string()] = state;
    }

    pub fn clear_diagnostics(&mut self) {
        if let Some(entries) = self.diagnostics.as_object_mut() {
            for value in entries.values_mut() {
                *value = json!(0);
            }
        }
    }

    pub fn has_diagnostic_errors(&self) -> bool {
        self.diagnostics
            .as_object()
            .is_some_and(|m| m.values().any(|v| int(v) != 0))
    }

    pub fn route_action(route: i64, step: i64) -> Value {
        route_action(route, step)
    }

    pub fn configured_route_action(&self, route: i64, step: i64) -> Value {
        configured_route_action(
            route,
            step,
            self.farm2945,
            self.small_market || (self.cha22 && route == 0),
            self.hybrid_opening,
            self.v56 || self.cha22,
        )
    }

    pub fn act(&mut self, observation: &Value) -> Value {
        let step = step_of(observation);
        let key = int(&observation["player"]).to_string();
        if array(&observation["farms"]).len() < 2 {
            increment(&mut self.diagnostics, "entry_fallbacks", 1);
            let route = int(&self.players[&key]["route"]);
            return if self.v56 {
                self.configured_route_action(route, step)
            } else {
                route_action(route, step)
            };
        }
        if self.players[&key].is_null()
            || step == 0
            || step <= int(&self.players[&key]["last_step"])
        {
            self.players[&key] = json!({"last_step":-1,"route":null,"router_state":{},
                "pending":{},"sell_state":{"due_step":-1,"suppress":{}}});
        }
        let state = &mut self.players[&key];
        state["last_step"] = json!(step);
        let view = View::new(observation);
        let route = if self.farm2945 {
            router_farm2945(observation, step, &mut state["router_state"], self.cha22)
        } else {
            router(observation, step, &mut state["router_state"])
        };
        state["route"] = json!(route);
        let mut action = configured_route_action(
            route,
            step,
            self.farm2945,
            self.small_market || (self.cha22 && route == 0),
            self.hybrid_opening,
            self.v56 || self.cha22,
        );
        Self::hand_align(&mut action, &view);
        Self::weed_repair(
            &mut action,
            &view,
            state,
            &configured_route_action(
                route,
                step + 1,
                self.hybrid_opening,
                self.cha22,
                self.hybrid_opening,
                self.v56 || self.cha22,
            ),
        );
        Self::apply_suppression(&mut action, &mut state["sell_state"], step);
        let mut projected = Self::projected_shed(&action, &view);
        let debts = state["sell_state"]["r36_debts"]
            .as_object()
            .map(|m| Value::Object(m.clone()))
            .unwrap_or_else(|| json!({}));
        let mut next_sup = json!({"due_step":-1,"suppress":{},"r36_debts":debts});
        if self.farm2945 {
            // `_v9_racepx_lead` over `_r36_native_lead` (native window ends at step 288).
            let blocked: Vec<&str> = PRODUCTS
                .into_iter()
                .filter(|i| farm2945_base_price(i).is_some_and(|b| int(&view.prices[*i]) <= b))
                .collect();
            if blocked.is_empty() {
                increment(&mut self.racepx_report, "racepx_sold", 1);
                if !(288..696).contains(&step) {
                    Self::sell_lead(
                        &mut action,
                        &view,
                        &mut projected,
                        route,
                        step,
                        &mut next_sup,
                        None,
                    );
                }
            } else {
                let added = Self::sell_lead(
                    &mut action,
                    &view,
                    &mut projected,
                    route,
                    step,
                    &mut next_sup,
                    Some(&blocked),
                );
                increment(&mut self.racepx_report, "racepx_skipped", added);
            }
        } else if !(216..696).contains(&step) {
            Self::sell_lead(
                &mut action,
                &view,
                &mut projected,
                route,
                step,
                &mut next_sup,
                None,
            );
        }
        state["sell_state"] = next_sup;
        if let Some(market) = action["market"].as_array_mut() {
            market.truncate(10);
        }
        action
    }

    pub fn hand_align(action: &mut Value, view: &View) {
        let expected = view.positions.len().saturating_sub(1);
        let mut hands = array(&action["hands"]).to_vec();
        hands.resize_with(expected, || json!(["PASS"]));
        action["hands"] = json!(hands);
    }

    fn weed_repair(action: &mut Value, view: &View, state: &mut Value, next_action: &Value) {
        let mut units = commands(action);
        let next_units = commands(next_action);
        for index in 0..units.len().min(view.positions.len()) {
            let pos = &view.positions[index];
            if !pos.is_array() {
                continue;
            }
            let tile = tile_at(&view.tiles, pos);
            let mut act = units[index].clone();
            let key = index.to_string();
            let mut queue = array(&state["pending"][&key]).to_vec();
            if !queue.is_empty() && queue[0][0] != *pos {
                state["pending"].as_object_mut().unwrap().shift_remove(&key);
                queue.clear();
            }
            let is_weed = text(&tile["kind"]) == "WEED";
            let noop = is_noop(&act, tile, view.inv(index), &view.seeds, pos, view.board);
            let next_op = next_units.get(index).map(|a| text(&a[0])).unwrap_or("PASS");
            if matches!(text(&act[0]), "PLANT" | "BUILD_COOP" | "BUILD_PASTURE") && is_weed {
                queue.push(json!([pos, act]));
                state["pending"][&key] = json!(queue);
                act = json!(["DIG"]);
            } else if !queue.is_empty() && noop {
                let replay = queue[0][1].clone();
                if replay[0] == "PLANT" && MOVES.contains(&next_op) {
                    state["pending"].as_object_mut().unwrap().shift_remove(&key);
                } else {
                    queue.remove(0);
                    if truth(&act) && act[0] != "PASS" && !MOVES.contains(&text(&act[0])) {
                        queue.push(json!([pos, act]));
                    }
                    act = replay;
                    if queue.is_empty() {
                        state["pending"].as_object_mut().unwrap().shift_remove(&key);
                    } else {
                        state["pending"][&key] = json!(queue);
                    }
                }
            } else if is_weed && noop {
                act = json!(["DIG"]);
            }
            units[index] = act;
        }
        set_commands(action, units);
    }

    pub fn projected_shed(action: &Value, view: &View) -> Value {
        let mut projected = json!({});
        for item in PRODUCTS {
            projected[item] = json!(int(&view.shed[item]));
        }
        if let Some(entries) = view.shed.as_object() {
            for (item, amount) in entries {
                if projected.get(item).is_none() {
                    projected[item] = amount.clone();
                }
            }
        }
        let mut total = stock_total(&projected);
        for (index, act) in commands(action)
            .iter()
            .enumerate()
            .take(view.positions.len())
        {
            if !shed_adjacent(&view.positions[index], view.board) {
                continue;
            }
            let item = text(&act[1]);
            let inv = view.inv(index);
            match text(&act[0]) {
                "PICKUP" if projected.get(item).is_some() => {
                    let qty = int(&projected[item]).min(quantity(act, 1).max(0));
                    increment(&mut projected, item, -qty);
                    total -= qty;
                }
                "DROP" => {
                    if let Some(entries) = inv.as_object() {
                        for (item, held) in entries {
                            let take = int(held).max(0).min((100 - total).max(0));
                            if take > 0 {
                                increment(&mut projected, item, take);
                                total += take;
                            }
                        }
                    }
                }
                "PLACE" if act.get(1).is_some() && animal_structure(item).is_none() => {
                    let take = quantity(act, 1)
                        .max(0)
                        .min(int(&inv[item]).max(0))
                        .min((100 - total).max(0));
                    if take > 0 {
                        increment(&mut projected, item, take);
                        total += take;
                    }
                }
                _ => {}
            }
        }
        projected
    }

    pub fn apply_suppression(action: &mut Value, sell_state: &mut Value, step: i64) {
        let mut market = orders(action);
        if int(&sell_state["due_step"]) == step {
            let mut remaining = sell_state["suppress"].clone();
            for order in &mut market {
                let item = text(&order[1]).to_owned();
                if order[0] == "SELL" && array(order).len() >= 3 && int(&remaining[&item]) > 0 {
                    let removed = int(&order[2]).max(0).min(int(&remaining[&item]));
                    order[2] = json!(int(&order[2]) - removed);
                    increment(&mut remaining, &item, -removed);
                }
            }
        }
        let mut due = sell_state["r36_debts"]
            .as_object_mut()
            .and_then(|m| m.shift_remove(&step.to_string()))
            .unwrap_or_else(|| json!({}));
        for order in &mut market {
            if array(order).len() >= 3 && order[0] == "SELL" {
                let item = text(&order[1]).to_owned();
                let removed = int(&order[2]).max(0).min(int(&due[&item]));
                order[2] = json!(int(&order[2]) - removed);
                increment(&mut due, &item, -removed);
            }
        }
        set_orders(action, market);
    }

    pub fn add_sell(
        action: &mut Value,
        item: &str,
        qty: i64,
        max_orders: usize,
        merge: bool,
    ) -> bool {
        if !action["market"].is_array() {
            action["market"] = json!([]);
        }
        let market = action["market"].as_array_mut().unwrap();
        if merge && let Some(order) = market.iter_mut().find(|o| o[0] == "SELL" && o[1] == item) {
            order[2] = json!(int(&order[2]) + qty);
            return true;
        }
        if market.len() >= max_orders {
            return false;
        }
        market.push(json!(["SELL", item, qty]));
        true
    }

    fn sell_lead(
        action: &mut Value,
        view: &View,
        projected: &mut Value,
        route: i64,
        step: i64,
        next_sup: &mut Value,
        racepx_blocked: Option<&[&str]>,
    ) -> i64 {
        let next = step + 1;
        if next > 718 || next % 72 == 0 || step % 4 == 0 {
            return 0;
        }
        let mut added = 0;
        let mut planned = json!({});
        for order in orders(&route_action(route, next)) {
            if order[0] == "SELL" && array(&order).len() >= 3 && PRODUCTS.contains(&text(&order[1]))
            {
                increment(&mut planned, text(&order[1]), int(&order[2]).max(0));
            }
        }
        let already: Vec<String> = orders(action)
            .iter()
            .filter(|o| o[0] == "SELL" && array(o).len() > 1)
            .map(|o| text(&o[1]).to_owned())
            .collect();
        for item in PRODUCTS {
            // The RACEPX variant gates on the glut set instead of WHEAT/FERTILIZER.
            let skipped = match racepx_blocked {
                Some(blocked) => blocked.contains(&item),
                None => matches!(item, "WHEAT" | "FERTILIZER"),
            };
            if skipped || int(&planned[item]) <= 0 || already.iter().any(|x| x == item) {
                continue;
            }
            let qty = int(&projected[item]).min(int(&planned[item]));
            if qty <= 0 || int(&view.prices[item]) < 2 {
                continue;
            }
            if !Self::add_sell(action, item, qty, 10, false) {
                break;
            }
            increment(projected, item, -qty);
            increment(&mut next_sup["suppress"], item, qty);
            added += 1;
        }
        if truth(&next_sup["suppress"]) {
            next_sup["due_step"] = json!(next);
        }
        added
    }
}

/// Public wrapper at Python line 976, called before the physical terminal planner.
pub fn shop_terminal(observation: &Value, action: Value) -> Value {
    if step_of(observation) < 718 {
        return action;
    }
    let view = View::new(observation);
    let units = view
        .positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            json!([if shed_adjacent(pos, view.board) && truth(view.inv(i)) {
                "DROP"
            } else {
                "PASS"
            }])
        })
        .collect();
    let mut result = pass_action();
    set_commands(&mut result, units);
    let projected = Core::projected_shed(&result, &view);
    let mut market: Vec<Value> = PRODUCTS
        .iter()
        .filter(|p| int(&projected[**p]) > 0)
        .map(|p| json!(["SELL", p, int(&projected[*p])]))
        .collect();
    market.sort_by_key(|o| -(int(&view.prices[text(&o[1])]) * int(&o[2])));
    set_orders(&mut result, market);
    result
}

/// Public day-close storage wrapper at Python line 1104. Its estimate is retained.
pub fn room_guard(observation: &Value, action: Value) -> Value {
    if step_of(observation) % 24 != 23 {
        return action;
    }
    let view = View::new(observation);
    let carried: i64 = view
        .invs
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|m| m.values())
        .map(|v| int(v).max(0))
        .sum();
    let mut needed = stock_total(&view.shed) + carried - 99;
    if needed <= 0 {
        return action;
    }
    let mut planned = json!({});
    for order in orders(&action) {
        if order[0] == "SELL" && array(&order).len() >= 3 {
            increment(&mut planned, text(&order[1]), int(&order[2]).max(0));
        }
    }
    let mut priority = PRODUCTS.to_vec();
    priority.sort_by_key(|item| -int(&view.prices[*item]));
    let mut result = action;
    let mut market = orders(&result);
    for item in priority {
        let qty = needed.min((int(&view.shed[item]) - int(&planned[item])).max(0));
        if qty <= 0 {
            continue;
        }
        if market.len() >= 10 {
            break;
        }
        market.push(json!(["SELL", item, qty]));
        needed -= qty;
        if needed <= 0 {
            break;
        }
    }
    set_orders(&mut result, market);
    result
}
