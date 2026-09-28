//! cha22 weed recovery, ready-stock advance, terminal clear and EarlyCycle wrappers.
//! Faithful translation of agents/cha22/main.py; upstream Apache notices retained there.
use crate::native_agents::farm2945::util::{is_order, late_tape, own_farm, units, with_units};
use crate::native_agents::metav4::Metav4Controller;
use crate::native_agents::pipe16::sell_extra;
use crate::native_agents::v43::{common::*, core::Core};
use crate::native_agents::v47::{advance::Advance, race::standard};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Early {
    pub weedlag: Value,
    pub pipe: Value,
    pub mirror: Value,
    pub weed_report: Value,
    pub advance: Advance,
    pub terminal_report: Value,
    pub pipe_report: Value,
    pub mirror_report: Value,
    pub buy_report: Value,
}
impl Default for Early {
    fn default() -> Self {
        let mut advance = Advance {
            subtract_debts: true,
            ..Default::default()
        };
        advance.report["front_turns"] = json!(0);
        Self {
            weedlag: json!({}),
            pipe: json!({}),
            mirror: json!({"prior":null,"mirror_like":false}),
            weed_report: json!({"events":0,"replays":0,"errors":0}),
            advance,
            terminal_report: json!({"term_sells":0,"term_units":0}),
            pipe_report: json!({}),
            mirror_report: json!({"classified":0,"mirror_like":0,"market_drop":0,"errors":0}),
            buy_report: json!({"reordered":0}),
        }
    }
}
impl Early {
    pub fn before(&mut self, obs: &Value) {
        let step = int(&obs["step"]);
        let seat = int(&obs["player"]).to_string();
        if step == 0 {
            self.mirror = json!({"prior":null,"mirror_like":false});
            self.mirror_report = json!({"classified":0,"mirror_like":0,"market_drop":0,"errors":0});
            self.weedlag = json!({});
            self.weed_report = json!({"events":0,"replays":0,"errors":0});
        }
        if step == 1 {
            self.mirror["prior"] = json!(int(&obs["market"]["inventory"]["WHEAT"]));
        }
        if step == 2 && !self.mirror["prior"].is_null() {
            let fall = int(&self.mirror["prior"]) - int(&obs["market"]["inventory"]["WHEAT"]);
            self.mirror["mirror_like"] = json!(fall >= 15);
            self.mirror_report = json!({"classified":1,"mirror_like":i64::from(fall>=15),"market_drop":fall,"errors":int(&self.mirror_report["errors"])});
        }
        if self.pipe[&seat].is_null() || step <= int(&self.pipe[&seat]["step"]) {
            self.pipe[&seat] = json!({"step":-1,"mode":"EarlyCycle"});
            self.pipe_report = json!({"selected_opening":"EarlyCycle","temporary_crop_seen":0,"temporary_crop_harvested":0,"restored_pasture_seen":0,"delivered_extra_wheat":0,"tomato_seen":0,"tomato_water_requests":0,"tomato_harvest_requests":0,"tomato_units_harvest_requested":0,"extension_errors":0});
        }
    }
    pub fn weed_apply(&mut self, obs: &Value, action: Value, core: &Core) -> Value {
        let step = int(&obs["step"]);
        let seat = int(&obs["player"]).to_string();
        let st = &mut self.weedlag[&seat];
        if st.is_null() {
            *st = json!({"step":-1,"active":{}});
        }
        if step <= int(&st["step"]) || step % 24 == 0 {
            st["active"] = json!({});
        }
        st["step"] = json!(step);
        let farm = own_farm(obs);
        let mut positions = vec![farm["farmer"].clone()];
        positions.extend(array(&farm["hands"]).iter().cloned());
        let mut commands = units(&action);
        let now = units(&late_tape(core, obs, step));
        let prev = units(&late_tape(core, obs, step - 1));
        let mut changed = false;
        let active = st["active"].as_object().cloned().unwrap_or_default();
        for (key, tx) in active {
            let k = key.parse::<usize>().unwrap_or(usize::MAX);
            if k >= commands.len() || k >= positions.len() {
                st["active"].as_object_mut().unwrap().shift_remove(&key);
                continue;
            }
            let age = step - int(&tx["start"]);
            if age == 1 {
                commands[k] = tx["intended"].clone();
                changed = true;
                increment(&mut self.weed_report, "replays", 1);
            } else if (2..=9).contains(&age) {
                commands[k] = prev
                    .get(k)
                    .filter(|v| v.is_array() && truth(v))
                    .cloned()
                    .unwrap_or(json!(["PASS"]));
                changed = true;
                increment(&mut self.weed_report, "replays", 1);
            } else {
                st["active"].as_object_mut().unwrap().shift_remove(&key);
            }
        }
        for k in 0..commands.len().min(positions.len()).min(now.len()) {
            let key = k.to_string();
            if st["active"].get(&key).is_some() {
                continue;
            }
            if !matches!(text(&now[k][0]), "BUILD_PASTURE" | "BUILD_COOP") {
                continue;
            }
            if tile_at(&farm["tiles"], &positions[k])["kind"] != "WEED" {
                continue;
            }
            if commands[k] != json!(["DIG"]) {
                commands[k] = json!(["DIG"]);
                changed = true;
            }
            st["active"][key] = json!({"start":step,"intended":now[k]});
            increment(&mut self.weed_report, "events", 1);
        }
        if changed {
            with_units(&action, commands)
        } else {
            action
        }
    }
    pub fn weed(&mut self, obs: &Value, config: &Value, action: Value, core: &Core) -> Value {
        if standard(config, false) {
            self.weed_apply(obs, action, core)
        } else {
            action
        }
    }
    pub fn advance_apply(&mut self, obs: &Value, action: Value, core: &mut Core) -> Value {
        if int(&obs["step"]) < 216 || core.players[int(&obs["player"]).to_string()].is_null() {
            return action;
        }
        match self.advance.apply(obs, action.clone(), core) {
            Ok(a) => a,
            Err(_) => {
                increment(&mut self.advance.report, "adv_errors", 1);
                action
            }
        }
    }
    pub fn advance(
        &mut self,
        obs: &Value,
        config: &Value,
        action: Value,
        base: &mut Metav4Controller,
    ) -> Value {
        if int(&obs["step"]) == 0 {
            self.advance.report =
                json!({"adv_turns":0,"adv_units":0,"adv_errors":0,"front_turns":0});
        }
        let action = if standard(config, false) {
            self.advance_apply(obs, action, &mut base.base.base.core)
        } else {
            action
        };
        let st = &mut base.race.states[int(&obs["player"]).to_string()];
        if !st.is_null() && !st["prev_action"].is_null() && st["step"] == obs["step"] {
            st["prev_action"] = action.clone();
        }
        action
    }
    pub fn terminal(&mut self, obs: &Value, mut action: Value) -> Value {
        if int(&obs["step"]) < 712 || !action.is_object() {
            return action;
        }
        let prices = &obs["market"]["prices"];
        let mut items: Vec<_> = obs["private"]["shed"]
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(k, v)| int(v) > 0 && int(&prices[*k]) >= 1)
            .collect();
        items.sort_by_key(|(k, _)| -int(&prices[*k]));
        if items.is_empty() {
            return action;
        }
        let mut market = vec![];
        for (item, q) in items.into_iter().take(10) {
            market.push(json!(["SELL", item, int(q)]));
            increment(&mut self.terminal_report, "term_units", int(q));
        }
        action["market"] = json!(market);
        increment(&mut self.terminal_report, "term_sells", 1);
        action
    }
    pub fn pipe(&mut self, obs: &Value, mut action: Value) -> Value {
        let step = int(&obs["step"]);
        let key = int(&obs["player"]).to_string();
        let farm = own_farm(obs);
        let site = &farm["tiles"][4][2];
        if step == 6 {
            self.pipe_report["temporary_crop_seen"] =
                json!(i64::from(site.is_object() && site["crop"] == "WHEAT"));
        }
        if step == 54 {
            if obs["private"]["inventories"].get(1).is_some() {
                self.pipe_report["temporary_crop_harvested"] =
                    json!(int(&obs["private"]["inventories"][1]["WHEAT"]));
            } else {
                increment(&mut self.pipe_report, "extension_errors", 1);
            }
        }
        if step == 55 {
            self.pipe_report["restored_pasture_seen"] =
                json!(i64::from(site.is_object() && site["kind"] == "PASTURE"));
        }
        if matches!(step, 57 | 91)
            && !array(&farm["hands"]).is_empty()
            && farm["hands"][0] == json!([4, 4])
            && action["hands"][0] == json!(["DROP"])
        {
            if obs["private"]["inventories"].get(1).is_some() {
                let n = int(&obs["private"]["inventories"][1]["WHEAT"]);
                action = sell_extra(action, "WHEAT", n);
                self.pipe_report["delivered_extra_wheat"] = json!(n);
            } else {
                increment(&mut self.pipe_report, "extension_errors", 1);
            }
        }
        if self.pipe[&key].is_null() {
            self.pipe[&key] = json!({"mode":"EarlyCycle","step":step});
        } else {
            self.pipe[&key]["step"] = json!(step);
        }
        action
    }
    pub fn buyfirst(&mut self, obs: &Value, mut action: Value) -> Value {
        let shops = array(&obs["town"]["unlocked_shops"]);
        let pair = shops.iter().take(2).map(text).collect::<Vec<_>>();
        let target = pair == ["BRUNCH_SPOT", "BRUNCH_SPOT"]
            || pair == ["BAKERY", "BRUNCH_SPOT"]
            || (truth(&self.mirror["mirror_like"]) && pair == ["PIZZA_SHOP", "SMOOTHIE_SHOP"]);
        if !target {
            return action;
        }
        let market = orders(&action);
        if market.len() < 2 {
            return action;
        }
        let buys: Vec<_> = market
            .iter()
            .filter(|o| is_order(o, "BUY_PRODUCT", "WHEAT"))
            .cloned()
            .collect();
        if buys.is_empty() || buys.contains(&market[0]) {
            return action;
        }
        let mut out = buys.clone();
        out.extend(market.into_iter().filter(|o| !buys.contains(o)));
        action["market"] = json!(out);
        increment(&mut self.buy_report, "reordered", 1);
        action
    }
    pub fn states(&self) -> Value {
        json!({"weedlag":self.weedlag,"pipe":self.pipe,"mirror":self.mirror})
    }
    pub fn reports(&self) -> Value {
        json!({"weedlag":self.weed_report,"advance":self.advance.report,"terminal_clear":self.terminal_report,"pipe":self.pipe_report,"mirror":self.mirror_report,"buyfirst":self.buy_report})
    }
    pub fn inject(&mut self, state: &Value) {
        for (name, dest) in [
            ("weedlag", &mut self.weedlag),
            ("pipe", &mut self.pipe),
            ("mirror", &mut self.mirror),
        ] {
            if let Some(v) = state.get(name) {
                *dest = v.clone();
            }
        }
    }
}
