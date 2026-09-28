//! V56 active last-callable entry point: CL, price guard, EXP402 and EXP410.
//! Exact upstream attribution and source are retained in agents/v56/.
use crate::Game;
use crate::native_agents::farm2945::{
    carrot2::{self, Growth},
    util::*,
};
use crate::native_agents::pipe16::{Pipe16Controller, sell_extra};
use crate::native_agents::v43::{
    common::{array, int, num, truth},
    terminal,
};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct V56Controller {
    pub base: Pipe16Controller,
    pub cl_report: Value,
    pub seed_report: Value,
    pub fert_report: Value,
    pub seed_cache: Value,
    stages: Value,
}
impl Default for V56Controller {
    fn default() -> Self {
        let mut base = Pipe16Controller::default();
        base.base.base.base.core.v56 = true;
        Self {
            base,
            cl_report: json!({"cl_harvested":0,"cl_pasture":0,"cl_delivered":0,"cl_errors":0}),
            seed_report: json!({"cut_units":0,"saved_cost":0,"changed_turns":0,"errors":0}),
            fert_report: json!({"skips":0,"covered":0,"capped":0,"errors":0}),
            seed_cache: json!({}),
            stages: Value::Null,
        }
    }
}
impl V56Controller {
    pub fn cl(&mut self, obs: &Value, mut action: Value) -> Value {
        let step = int(&obs["step"]);
        if step == 0 {
            for v in self.cl_report.as_object_mut().unwrap().values_mut() {
                *v = json!(0);
            }
        }
        let result = (|| -> Result<(), ()> {
            let farm = obs.get("farms").and_then(|v| v.get(seat(obs))).ok_or(())?;
            let private = obs.get("private").ok_or(())?;
            if step == 88 {
                let inv = private
                    .get("inventories")
                    .and_then(|v| v.get(1))
                    .and_then(Value::as_object)
                    .ok_or(())?;
                self.cl_report["cl_harvested"] = json!(inv.get("WHEAT").map(int).unwrap_or(0));
            }
            if step == 89 {
                let tile = farm
                    .get("tiles")
                    .and_then(|v| v.get(4))
                    .and_then(|v| v.get(2))
                    .ok_or(())?;
                self.cl_report["cl_pasture"] =
                    json!(i64::from(tile.is_object() && tile["kind"] == "PASTURE"));
            }
            if step == 91 && truth(&farm["hands"]) && farm["hands"][0] == json!([4, 4]) {
                let cmd = action.get("hands").and_then(|v| v.get(0)).ok_or(())?;
                if *cmd == json!(["DROP"]) {
                    let inv = private
                        .get("inventories")
                        .and_then(|v| v.get(1))
                        .and_then(Value::as_object)
                        .ok_or(())?;
                    let n = inv.get("WHEAT").map(int).unwrap_or(0);
                    action = sell_extra(action.clone(), "WHEAT", n);
                    self.cl_report["cl_delivered"] = json!(n);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            inc(&mut self.cl_report, "cl_errors", 1);
        }
        action
    }
    pub fn price(&self, obs: &Value, mut action: Value) -> Value {
        if int(&obs["step"]) == 91 && num(&obs["market"]["prices"]["WHEAT"]) < 31.0 {
            action["market"] = json!(
                array(&action["market"])
                    .iter()
                    .filter(|o| !is_order(o, "SELL", "WHEAT"))
                    .cloned()
                    .collect::<Vec<_>>()
            );
        }
        action
    }
    pub fn remaining(&mut self, route: i64, step: i64) -> i64 {
        let k = format!("{route},{step}");
        if let Some(v) = self.seed_cache.get(&k) {
            return int(v);
        }
        let core = &self.base.base.base.base.core;
        let n = (step + 1..719)
            .flat_map(|t| units(&core.configured_route_action(if t >= 648 { 2 } else { route }, t)))
            .filter(|c| is_pair(c, "PLANT", "WHEAT") || is_pair(c, "PLANT", "CARROT"))
            .count() as i64;
        self.seed_cache[&k] = json!(n);
        n
    }
    pub fn seeds(&mut self, obs: &Value, mut action: Value) -> Value {
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
        let native = self.base.base.base.base.core.players[&who].clone();
        if !native.is_object() {
            inc(&mut self.seed_report, "errors", 1);
            return action;
        }
        let mut remaining = self.remaining(int(&native["route"]), step);
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
                            let st = &mut self.base.base.base.carrot2.states[&who];
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
    pub fn fertilizer(&mut self, obs: &Value, mut action: Value) -> Value {
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
            let core = &self.base.base.base.base.core;
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
            let reactive = &self.base.base.base.base.late.input[&who]["workers"];
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
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        let pipe = self.base.act(obs, config);
        let cl = self.cl(obs, pipe.clone());
        let price = self.price(obs, cl.clone());
        let seeds = self.seeds(obs, price.clone());
        let final_action = self.fertilizer(obs, seeds.clone());
        self.stages =
            json!({"pipe16":pipe,"cl":cl,"price":price,"seeds":seeds,"final":final_action});
        final_action
    }
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid v56 seat {seat}"));
        }
        let mut obs = serde_json::to_value(game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(&game.config).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }
    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut d = self.base.debug();
        for (k, v) in self.stages.as_object().unwrap() {
            d["stages"][k] = v.clone();
        }
        d["reports"]["cl"] = self.cl_report.clone();
        d["reports"]["seed"] = self.seed_report.clone();
        d["reports"]["fert"] = self.fert_report.clone();
        d
    }
    pub fn states(&self) -> Value {
        let mut s = self.base.states();
        s["reports"]["cl"] = self.cl_report.clone();
        s["reports"]["seed"] = self.seed_report.clone();
        s["reports"]["fert"] = self.fert_report.clone();
        s["seed_cache"] = self.seed_cache.clone();
        s["debug"] = self.debug();
        s
    }
    pub fn oracle_request(&mut self, v: &Value) -> Value {
        if v["reset"] == true {
            *self = Self::default();
        }
        if let Some(s) = v.get("state") {
            self.base.base.inject(s);
            if let Some(input) = s.get("input") {
                self.base.base.base.base.late.input = input.clone();
            }
            if let Some(p) = s.get("pipe16") {
                self.base.opening = p.clone();
            }
            if let Some(p) = s.get("seed_cache") {
                self.seed_cache = p.clone();
            }
            for (k, target) in [
                ("pipe16", &mut self.base.report),
                ("cl", &mut self.cl_report),
                ("seed", &mut self.seed_report),
                ("fert", &mut self.fert_report),
            ] {
                if let Some(r) = s.get("reports").and_then(|r| r.get(k)) {
                    *target = r.clone();
                }
            }
        }
        let obs = &v["observation"];
        let a = v["action"].clone();
        let config = v.get("configuration").cloned().unwrap_or(json!({}));
        let result = match v["mode"].as_str().unwrap_or("act") {
            "act" => self.act(obs, &config),
            "clone_act" => {
                let mut fork = self.clone();
                let a = self.act(obs, &config);
                let b = fork.act(obs, &config);
                if a != b || self.states() != fork.states() {
                    return json!({"id":v["id"],"error":"cloned controller diverged"});
                }
                a
            }
            "v56_cl" => self.cl(obs, a),
            "v56_price" => self.price(obs, a),
            "v56_seed" => self.seeds(obs, a),
            "v56_fert" => self.fertilizer(obs, a),
            "v56_remaining" => json!(self.remaining(int(&v["route"]), int(&v["step"]))),
            _ => {
                let mut req = v.clone();
                req["reset"] = json!(false);
                let mut response = self.base.oracle_request(&req);
                response["state"] = self.states();
                return response;
            }
        };
        json!({"id":v["id"],"result":result,"state":self.states(),"engine_transitions":0})
    }
}
