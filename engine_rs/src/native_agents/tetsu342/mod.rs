//! Native port of tetsutani's 2026-09-20 V104 public agent.
//!
//! The frozen source is the V48 controller with its opening attack changed
//! from 30 to 22, followed by the 2945 Farm's reusable V9/EXP342 layers
//! (courier, carrot, herd, fertilizer, CARROT2, ORDERPRI2, CAPHARV,
//! SHEDROOM, HERD2 and COWSWAP).  The source's final EXP343 recovery arm is
//! kept behind the small `weedlag` layer below; the six-sheep day-11 arm is
//! represented by the existing V233 production state and remains a separate
//! bounded direct-case concern.

use crate::native_agents::farm2945::util::{inc, with_units};
use crate::native_agents::farm2945::{capharv, carrot2, early, herd2, market, orderpri2};
use crate::native_agents::v43::common::*;
use crate::native_agents::v48::V48Controller;
use crate::Game;
use serde_json::{json, Value};

#[derive(Clone, Debug, Default)]
pub struct WeedLag {
    pub state: Value,
    pub report: Value,
}

impl WeedLag {
    fn fresh_report() -> Value {
        json!({"events":0,"replays":0,"errors":0})
    }

    /// The EXP343 weed recovery is only reachable when the V48 chassis has
    /// already converted a tape build/plant into DIG on a weed.  Preserve the
    /// one-step replay and bounded lag without consulting hidden state.
    pub fn apply(
        &mut self,
        obs: &Value,
        action: Value,
        core: &crate::native_agents::v43::core::Core,
    ) -> Value {
        let step = int(&obs["step"]);
        let seat = int(&obs["player"]) as usize;
        if self.state.is_null()
            || self.state[seat.to_string()].is_null()
            || step <= int(&self.state[seat.to_string()]["step"])
        {
            if step == 0 {
                self.report = Self::fresh_report();
            }
            self.state[seat.to_string()] = json!({"step":-1,"active":{}});
        }
        let st = &mut self.state[seat.to_string()];
        st["step"] = json!(step);
        if step % 24 == 0 {
            st["active"] = json!({});
        }
        let farm = &obs["farms"][seat];
        let positions = {
            let mut p = vec![farm["farmer"].clone()];
            p.extend(array(&farm["hands"]).iter().cloned());
            p
        };
        let mut units = commands(&action);
        while units.len() < positions.len() {
            units.push(json!(["PASS"]));
        }
        let native = &core.players[seat.to_string()];
        let route = int(&native["route"]);
        let tape = |t: i64| -> Vec<Value> {
            let r = if t >= 648 { 2 } else { route };
            let a = crate::native_agents::v43::common::routes()[r.to_string()]
                .get(t as usize)
                .cloned()
                .unwrap_or_else(|| json!({}));
            commands(&a)
        };
        let now = tape(step);
        let prev = tape(step - 1);
        let mut changed = false;
        for (key, tx) in st["active"].as_object().cloned().unwrap_or_default() {
            let k: usize = key.parse().unwrap_or(usize::MAX);
            if k >= units.len() || k >= positions.len() {
                continue;
            }
            let start = int(&tx["start"]);
            let age = step - start;
            if age == 1 {
                units[k] = tx["intended"].clone();
                changed = true;
                inc(&mut self.report, "replays", 1);
            } else if (2..=9).contains(&age) {
                units[k] = prev.get(k).cloned().unwrap_or_else(|| json!(["PASS"]));
                changed = true;
                inc(&mut self.report, "replays", 1);
            } else {
                st["active"].as_object_mut().map(|m| m.remove(&key));
            }
        }
        for k in 0..units.len().min(positions.len()).min(now.len()) {
            if !st["active"][k.to_string()].is_null() {
                continue;
            }
            let intent = &now[k];
            let op = text(&intent[0]);
            if !matches!(op, "BUILD_PASTURE" | "BUILD_COOP" | "PLANT") {
                continue;
            }
            let (x, y) = position(&positions[k]);
            let tile = farm["tiles"]
                .get(y as usize)
                .and_then(|r| r.get(x as usize))
                .unwrap_or(&Value::Null);
            if !(tile.is_object() && tile["kind"] == "WEED") {
                continue;
            }
            if units[k] != json!(["DIG"]) {
                units[k] = json!(["DIG"]);
                changed = true;
            }
            st["active"][k.to_string()] = json!({"start":step,"intended":intent});
            inc(&mut self.report, "events", 1);
        }
        if changed {
            with_units(&action, units)
        } else {
            action
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tetsu342Controller {
    pub base: V48Controller,
    pub early: early::Early,
    pub carrot2: carrot2::Carrot2,
    pub market: market::Market,
    pub orderpri2: orderpri2::OrderPri2,
    pub capharv: capharv::CapHarv,
    pub herd2: herd2::Herd2,
    pub weedlag: WeedLag,
    pub stages: Value,
}

impl Default for Tetsu342Controller {
    fn default() -> Self {
        let mut base = V48Controller::default();
        base.inner.open.attack = 22;
        Self {
            base,
            early: Default::default(),
            carrot2: Default::default(),
            market: Default::default(),
            orderpri2: Default::default(),
            capharv: Default::default(),
            herd2: Default::default(),
            weedlag: WeedLag {
                state: json!({}),
                report: WeedLag::fresh_report(),
            },
            stages: Value::Null,
        }
    }
}

impl Tetsu342Controller {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        if !obs["farms"].is_array() || array(&obs["farms"]).len() < 2 {
            return pass_action();
        }
        let v48 = self.base.act(obs, config);
        let core = &self.base.inner.base.core;
        let courier = self.early.courier(obs, v48.clone(), core);
        let carrot = self.early.carrot(obs, courier.clone());
        let herd = self.early.herd(obs, carrot.clone(), core);
        let fert = self.early.fert(obs, herd.clone(), core);
        let carrot2 = self.carrot2.layer(obs, fert.clone(), core);
        let orderpri2 = self
            .orderpri2
            .layer(obs, carrot2.clone(), &mut self.base.inner.base.core);
        let core = &self.base.inner.base.core;
        let capharv = self.capharv.layer(obs, orderpri2.clone(), core);
        let shedroom = self.market.shedroom(obs, capharv.clone(), config, core);
        let herd2 = self.herd2.herd2(obs, shedroom.clone(), core);
        let cowswap = self.herd2.cowswap(obs, herd2.clone(), core);
        let standard = [
            ("boardSize", 10),
            ("turnsPerDay", 24),
            ("shedCapacity", 100),
            ("maxMarketOrdersPerTurn", 10),
        ]
        .iter()
        .all(|(key, default)| {
            config.get(*key).and_then(Value::as_i64).unwrap_or(*default) == *default
        });
        let final_action = if standard {
            self.weedlag.apply(obs, cowswap.clone(), core)
        } else {
            cowswap.clone()
        };
        self.stages = json!({"v48":v48,"courier":courier,"carrot":carrot,"herd":herd,"fert":fert,
            "carrot2":carrot2,"orderpri2":orderpri2,"capharv":capharv,"shedroom":shedroom,
            "herd2":herd2,"cowswap":cowswap,"weedlag":final_action,"final":final_action});
        final_action
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid tetsu342 seat {seat}"));
        }
        let mut obs = serde_json::to_value(game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(&game.config).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }

    pub fn reports(&self) -> Value {
        json!({"v48":self.base.inner.debug(),"early":self.early.report,"carrot2":self.carrot2.report,
            "market":{"sr_turns":self.market.report["sr_turns"],"sr_units":self.market.report["sr_units"],"sr_errors":self.market.report["sr_errors"]},
            "orderpri2":self.orderpri2.report,"capharv":self.capharv.report,"herd2":self.herd2.report,"weedlag":self.weedlag.report})
    }
    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut d = self.base.debug();
        d["stages"] = self.stages.clone();
        d["reports"] = self.reports();
        d
    }
    pub fn states(&self) -> Value {
        let mut s = self.base.states();
        s["early_courier"] = self.early.courier.clone();
        s["early_carrot"] = self.early.carrot.clone();
        s["early_herd"] = self.early.herd.clone();
        s["carrot2"] = self.carrot2.states.clone();
        s["orderpri2"] = self.orderpri2.states();
        s["capharv"] = self.capharv.states.clone();
        s["herd2"] = self.herd2.herd.clone();
        s["cowswap"] = self.herd2.cow.clone();
        s["weedlag"] = self.weedlag.state.clone();
        s["debug"] = self.debug();
        s
    }
    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            *self = Self::default();
        }
        let obs = &value["observation"];
        let config = value.get("configuration").cloned().unwrap_or(json!({}));
        let result = match value["mode"].as_str().unwrap_or("act") {
            "act" => Ok(self.act(obs, &config)),
            "clone_act" => {
                let mut fork = self.clone();
                let a = self.act(obs, &config);
                let b = fork.act(obs, &config);
                if a != b || self.states() != fork.states() {
                    Err("cloned controller diverged".to_owned())
                } else {
                    Ok(a)
                }
            }
            _ => Err(format!("unknown mode {}", value["mode"])),
        };
        match result {
            Ok(r) => {
                json!({"id":value["id"],"result":r,"state":self.states(),"engine_transitions":0})
            }
            Err(e) => json!({"id":value["id"],"error":e}),
        }
    }
}
