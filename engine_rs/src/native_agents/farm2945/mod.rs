//! Native port of Thomas Tschinkel's frozen "The 2945 Farm" public agent (v9/4).
//! SHA-256 and full upstream Apache-2.0 notices live in agents/farm2945/.
//!
//! The source is the V39/V40 route replayer (the V43 modules here without the
//! R124-R148 contracts) with in-place changes — V39 opening, yarn worlds on
//! route 9, reservation from step 192 under the RACE horizon, RACEPX/RACEGATE
//! glut gates, day-11 sheep commitment (SL/VE/VT) — wrapped by the layers
//! below in source order. PREDICT2 has no stream library in the published
//! agent and is inert; ORDERPRI2's disabled sell-now block is not ported.
pub mod capharv;
pub mod carrot2;
pub mod early;
pub mod herd2;
pub mod market;
pub mod orderpri2;
pub mod race;
pub mod sheep;
pub mod util;

use self::util::*;
use crate::Game;
use crate::native_agents::v43::V43Controller;
use crate::native_agents::v43::common::*;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Farm2945Controller {
    pub base: V43Controller,
    pub early: early::Early,
    pub predict: race::Predict,
    pub race: race::Race,
    pub market: market::Market,
    pub carrot2: carrot2::Carrot2,
    pub orderpri2: orderpri2::OrderPri2,
    pub capharv: capharv::CapHarv,
    pub herd2: herd2::Herd2,
    /// Telemetry for the optional Smaller Market Shock suffix. It is emitted
    /// only when that overlay is enabled, preserving Farm2945 parity states.
    pub small_report: Value,
    stages: Value,
}
impl Default for Farm2945Controller {
    fn default() -> Self {
        let mut base = V43Controller::default();
        base.core.farm2945 = true;
        base.production.farm2945 = true;
        Self {
            base,
            early: Default::default(),
            predict: Default::default(),
            race: Default::default(),
            market: Default::default(),
            carrot2: Default::default(),
            orderpri2: Default::default(),
            capharv: Default::default(),
            herd2: Default::default(),
            small_report: json!({
                "selected_opening":"EarlyCycle",
                "temporary_crop_seen":0,
                "temporary_crop_harvested":0,
                "restored_pasture_seen":0,
                "delivered_extra_wheat":0,
                "tomato_seen":0,
                "tomato_water_requests":0,
                "tomato_harvest_requests":0,
                "tomato_units_harvest_requested":0,
                "extension_errors":0
            }),
            stages: Value::Null,
        }
    }
}

impl Farm2945Controller {
    /// Enable Dmitrii Gluzdov's Smaller Market Shock route-0 opening overlay.
    pub fn configure_smaller_market(&mut self) {
        self.base.core.small_market = true;
        self.market.disable_ctrtable = true;
    }

    fn reset_small_report(&mut self) {
        self.small_report = json!({
            "selected_opening":"EarlyCycle",
            "temporary_crop_seen":0,
            "temporary_crop_harvested":0,
            "restored_pasture_seen":0,
            "delivered_extra_wheat":0,
            "tomato_seen":0,
            "tomato_water_requests":0,
            "tomato_harvest_requests":0,
            "tomato_units_harvest_requested":0,
            "extension_errors":0
        });
    }

    /// Apply the source's post-parent step-57 extra wheat SELL wrapper.
    fn smaller_suffix(&mut self, obs: &Value, mut action: Value) -> Value {
        if !self.base.core.small_market {
            return action;
        }
        let step = int(&obs["step"]);
        if step == 0 {
            self.reset_small_report();
        }
        if step == 6 {
            let site = own_farm(obs)
                .get("tiles")
                .and_then(|tiles| tiles.get(4))
                .and_then(|row| row.get(2));
            if site.is_some_and(|tile| tile.is_object() && tile["crop"] == "WHEAT") {
                self.small_report["temporary_crop_seen"] = json!(1);
            }
        }
        if step == 54 {
            let n = obs["private"]["inventories"]
                .get(1)
                .map(|inventory| int(&inventory["WHEAT"]))
                .unwrap_or(0);
            self.small_report["temporary_crop_harvested"] = json!(n);
        }
        if step == 55 {
            let site = own_farm(obs)
                .get("tiles")
                .and_then(|tiles| tiles.get(4))
                .and_then(|row| row.get(2));
            if site.is_some_and(|tile| tile.is_object() && tile["kind"] == "PASTURE") {
                self.small_report["restored_pasture_seen"] = json!(1);
            }
        }
        if step != 57 {
            return action;
        }
        let first_hand_at_shed = own_farm(obs)
            .get("hands")
            .and_then(|hands| hands.get(0))
            .is_some_and(|position| position == &json!([4, 4]));
        let first_hand_drop = array(&action["hands"])
            .first()
            .is_some_and(|command| command == &json!(["DROP"]));
        if !first_hand_at_shed || !first_hand_drop {
            return action;
        }
        let extra = obs["private"]["inventories"]
            .get(1)
            .map(|inventory| int(&inventory["WHEAT"]).max(0))
            .unwrap_or(0);
        if extra <= 0 {
            return action;
        }
        let mut market = orders(&action);
        if let Some(order) = market
            .iter_mut()
            .find(|order| is_order(order, "SELL", "WHEAT"))
        {
            order[2] = json!(int(&order[2]) + extra);
            self.small_report["delivered_extra_wheat"] = json!(extra);
        } else if market.len() < MAX_ORDERS {
            market.push(json!(["SELL", "WHEAT", extra]));
            self.small_report["delivered_extra_wheat"] = json!(extra);
        }
        set_orders(&mut action, market);
        action
    }

    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        if !obs["farms"].is_array() || array(&obs["farms"]).len() < 2 {
            return pass_action();
        }
        if int(&obs["step"]) == 0 {
            for k in ["racepx_skipped", "racepx_sold", "racepx_errors"] {
                self.base.core.racepx_report[k] = json!(0);
            }
            self.base.production.report["racegate_reserved_units"] = json!(0);
        }
        self.race
            .before(obs, &self.base.core, &mut self.base.production.item_hz);
        let late = self.base.act_farm2945(obs, config);
        let core = &self.base.core;
        let courier = self.early.courier(obs, late.clone(), core);
        let carrot = self.early.carrot(obs, courier.clone());
        let herd = self.early.herd(obs, carrot.clone(), core);
        let fert = self.early.fert(obs, herd.clone(), core);
        let opening = early::Early::opening(obs, fert.clone(), core);
        let predict = if core.cha22 {
            opening.clone()
        } else {
            self.predict.layer(obs, opening.clone(), core, &self.race)
        };
        self.race.after(obs, &predict);
        let ctrtable = self.market.ctrtable(obs, predict.clone());
        let overflow = self.market.overflow(obs, ctrtable.clone());
        let carrot2 = self.carrot2.layer(obs, overflow.clone(), core);
        let orderpri2 = self
            .orderpri2
            .layer(obs, carrot2.clone(), &mut self.base.core);
        let core = &self.base.core;
        let capharv = self.capharv.layer(obs, orderpri2.clone(), core);
        let shedroom = self.market.shedroom(obs, capharv.clone(), config, core);
        let herd2 = self.herd2.herd2(obs, shedroom.clone(), core);
        let cowswap = self.herd2.cowswap(obs, herd2.clone(), core);
        let final_action = self.smaller_suffix(obs, cowswap.clone());
        self.stages = json!({
            "late": late, "courier": courier, "carrot": carrot, "herd": herd, "fert": fert,
            "opening": opening, "predict": predict, "ctrtable": ctrtable, "overflow": overflow,
            "carrot2": carrot2, "orderpri2": orderpri2, "capharv": capharv,
            "shedroom": shedroom, "herd2": herd2, "cowswap": cowswap, "final": final_action,
        });
        final_action
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid farm2945 seat {seat}"));
        }
        let mut obs = serde_json::to_value(game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(&game.config).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }

    /// All layer report counters, keyed like the source's telemetry dictionaries.
    pub fn reports(&self) -> Value {
        let mut reports = json!({
            "production": self.base.production.report,
            "racepx": self.base.core.racepx_report,
            "early": self.early.report,
            "predict": self.predict.report,
            "race": self.race.report,
            "market": self.market.report,
            "carrot2": self.carrot2.report,
            "orderpri2": self.orderpri2.report,
            "capharv": self.capharv.report,
            "herd2": self.herd2.report,
        });
        if self.base.core.small_market {
            reports["smaller"] = self.small_report.clone();
        }
        reports
    }

    /// Chassis debug view plus every layer output of the last `act`.
    /// Null until the first action, so a fresh FFI slot reads as untouched.
    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut debug = self.base.debug().clone();
        if let Some(stages) = self.stages.as_object() {
            for (k, v) in stages {
                debug["stages"][k] = v.clone();
            }
        }
        debug["reports"] = self.reports();
        debug
    }

    /// Oracle state snapshot: chassis memory plus every layer state.
    pub fn states(&self) -> Value {
        let base = &self.base;
        let mut states = json!({
            "core": base.core.players,
            "diagnostics": base.core.diagnostics,
            "v219": base.production.v219,
            "v231": base.production.v231,
            "v233": base.production.v233,
            "r37": base.production.r37,
            "r44": base.production.r44,
            "horizons": base.production.horizons,
            "item_hz": base.production.item_hz,
            "input": base.late.input,
            "courier": self.early.courier,
            "carrot": self.early.carrot,
            "herd": self.early.herd,
            "predict": self.predict.states(),
            "race": self.race.states,
            "ctrtable": self.market.ctrtable,
            "carrot2": self.carrot2.states,
            "orderpri2": self.orderpri2.states(),
            "capharv": self.capharv.states,
            "herd2": self.herd2.herd,
            "cowswap": self.herd2.cow,
            "reports": self.reports(),
            "debug": self.debug(),
        });
        if self.base.core.small_market {
            states["smaller"] = self.small_report.clone();
        }
        states
    }

    pub fn inject(&mut self, state: &Value) {
        let slots: [(&str, &mut Value); 10] = [
            ("core", &mut self.base.core.players),
            ("v233", &mut self.base.production.v233),
            ("courier", &mut self.early.courier),
            ("carrot", &mut self.early.carrot),
            ("herd", &mut self.early.herd),
            ("ctrtable", &mut self.market.ctrtable),
            ("carrot2", &mut self.carrot2.states),
            ("capharv", &mut self.capharv.states),
            ("herd2", &mut self.herd2.herd),
            ("cowswap", &mut self.herd2.cow),
        ];
        for (name, slot) in slots {
            if let Some(v) = state.get(name) {
                *slot = v.clone();
            }
        }
        if self.base.core.small_market
            && let Some(v) = state.get("smaller")
        {
            self.small_report = v.clone();
        }
    }

    /// JSONL oracle request for the differential harness (no engine transitions).
    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            let smaller = self.base.core.small_market;
            *self = Self::default();
            if smaller {
                self.configure_smaller_market();
            }
        }
        if let Some(state) = value.get("state") {
            self.inject(state);
        }
        let obs = &value["observation"];
        let config = value.get("configuration").cloned().unwrap_or(json!({}));
        let action = value.get("action").cloned().unwrap_or(Value::Null);
        let core = &self.base.core;
        let result: Result<Value, String> = match value["mode"].as_str().unwrap_or("act") {
            "act" => Ok(self.act(obs, &config)),
            "clone_act" => {
                let mut fork = self.clone();
                let actual = self.act(obs, &config);
                let cloned = fork.act(obs, &config);
                if actual != cloned || self.states() != fork.states() {
                    Err("cloned controller diverged".to_owned())
                } else {
                    Ok(actual)
                }
            }
            "courier" => Ok(self.early.courier(obs, action, core)),
            "carrot" => Ok(self.early.carrot(obs, action)),
            "herd" => Ok(self.early.herd(obs, action, core)),
            "fert" => Ok(self.early.fert(obs, action, core)),
            "opening" => Ok(early::Early::opening(obs, action, core)),
            "ctrtable" => Ok(self.market.ctrtable(obs, action)),
            "overflow" => Ok(self.market.overflow(obs, action)),
            "carrot2" => Ok(self.carrot2.layer(obs, action, core)),
            "capharv" => Ok(self.capharv.layer(obs, action, core)),
            "shedroom" => Ok(self.market.shedroom(obs, action, &config, core)),
            "herd2" => Ok(self.herd2.herd2(obs, action, core)),
            "cowswap" => Ok(self.herd2.cowswap(obs, action, core)),
            "orderpri2" => Ok(self.orderpri2.layer(obs, action, &mut self.base.core)),
            "hd2_ev" => herd2::ev(text(&value["option"]), int(&value["k"]), obs, core)
                .map(|v| json!(v))
                .map_err(str::to_owned),
            "sheep_request" => {
                let key = int(&obs["player"]).to_string();
                let mut state = self.base.production.v233[&key].clone();
                let native = self.base.core.players[&key].clone();
                let result = self
                    .base
                    .production
                    .sheep_request(obs, action, &mut state, &native);
                self.base.production.v233[&key] = state;
                Ok(result)
            }
            "sheep_worker" => Ok(sheep::worker(
                obs,
                int(&value["actor"]) as usize,
                &value["targets"],
                &mut self.base.production.report,
            )),
            "sl_path" => {
                let targets: Vec<_> = array(&value["targets"]).iter().map(position).collect();
                let path = sheep::sl_path(position(&value["start"]), &targets);
                Ok(json!(
                    path.iter().map(|p| json!([p.0, p.1])).collect::<Vec<_>>()
                ))
            }
            mode => Err(format!("unknown mode {mode}")),
        };
        match result {
            Ok(result) => {
                json!({"id":value["id"],"result":result,"state":self.states(),"engine_transitions":0})
            }
            Err(error) => json!({"id":value["id"],"error":error}),
        }
    }
}
