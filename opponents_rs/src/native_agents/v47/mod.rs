//! Native port of Ahmed Berat Özer's frozen Kaggriculture V47 public agent.
//! SHA-256 and full upstream Apache-2.0 notices live in agents/v47/.
//!
//! V47 is the frozen V43 controller (identical route data, contracts and
//! scratch engine; its terminal planner rewrite is a caching-only change)
//! wrapped by seven layers in source order: race horizon, opening round trip,
//! sale advance/front-load, pre-guard, clone lockstep reorder and the
//! shop-aware herd controller.
pub mod advance;
pub mod herd;
pub mod lockstep;
pub mod opening;
pub mod preguard;
pub mod race;

use crate::Game;
use crate::native_agents::v43::V43Controller;
use crate::native_agents::v43::common::*;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct V47Controller {
    pub base: V43Controller,
    pub race: race::Race,
    pub open: opening::Open,
    pub adv: advance::Advance,
    pub preguard: preguard::Preguard,
    pub lockstep: lockstep::Lockstep,
    pub herd: herd::Herd,
    stages: Value,
}

impl V47Controller {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        let (key, has_state) = self.race.before(
            obs,
            config,
            &self.base.core,
            &mut self.base.production.race_horizons,
        );
        let snapshot = self.race.snapshot_phase(obs, has_state);
        let v43 = self.base.act(obs, config);
        self.race.after(&key, has_state, snapshot, &v43);
        let open = self.open.apply(obs, config, v43.clone());
        let adv = self.adv.layer(
            obs,
            config,
            open.clone(),
            &mut self.base.core,
            &mut self.race,
        );
        let pre = self.preguard.layer(obs, config, adv.clone());
        let lock = self.lockstep.layer(obs, pre.clone(), &self.race);
        let herd = self.herd.layer(obs, lock.clone());
        // The race layer returns its parent's action unchanged.
        self.stages = json!({
            "v43": v43,
            "race": v43,
            "open": open,
            "adv": adv,
            "preguard": pre,
            "lockstep": lock,
            "herd": herd,
            "final": herd,
        });
        herd
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid V47 seat {seat}"));
        }
        let mut obs = serde_json::to_value(&game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates()[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(game.configuration()).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }

    /// Layer outputs of the last `act`, keyed by the producing layer.
    pub fn stages(&self) -> &Value {
        &self.stages
    }

    /// Complete controller state: the V43 debug view plus every V47 layer.
    /// Null until the first action, so a fresh FFI slot reads as untouched.
    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut debug = self.base.debug().clone();
        if let Some(stages) = self.base.contracts.stages.as_object() {
            for (k, v) in stages {
                debug["stages"][k] = v.clone();
            }
        }
        if let Some(stages) = self.stages.as_object() {
            for (k, v) in stages {
                debug["stages"][k] = v.clone();
            }
        }
        debug["race"] = self.race.states.clone();
        debug["race_report"] = self.race.report.clone();
        debug["open"] = json!({"open_attack": int(&self.open.report["open_attack"])});
        debug["open_report"] = self.open.report.clone();
        debug["adv_report"] = self.adv.report.clone();
        debug["preguard_report"] = self.preguard.report.clone();
        debug["lockstep_report"] = self.lockstep.report.clone();
        debug["herd"] = self.herd.states.clone();
        debug["herd_report"] = self.herd.report.clone();
        debug
    }

    /// Oracle state snapshot: V43 fields plus V47 layer states.
    pub fn states(&self) -> Value {
        let base = &self.base;
        json!({
            "core": base.core.players,
            "diagnostics": base.core.diagnostics,
            "v219": base.production.v219,
            "v231": base.production.v231,
            "v233": base.production.v233,
            "r37": base.production.r37,
            "r44": base.production.r44,
            "horizons": base.production.horizons,
            "race_horizons": base.production.race_horizons,
            "input": base.late.input,
            "r124": base.contracts.states124,
            "r127": base.contracts.states127,
            "r128": base.contracts.states128,
            "r148": base.contracts.pending148,
            "contract_report": base.contracts.report,
            "race": self.race.states,
            "open": {"open_attack": int(&self.open.report["open_attack"])},
            "herd": self.herd.states,
            "debug": self.debug(),
        })
    }

    pub fn inject(&mut self, state: &Value) {
        if let Some(v) = state.get("core") {
            self.base.core.players = v.clone();
        }
        if let Some(v) = state.get("r124") {
            self.base.contracts.states124 = v.clone();
        }
        if let Some(v) = state.get("r127") {
            self.base.contracts.states127 = v.clone();
        }
        if let Some(v) = state.get("r128") {
            self.base.contracts.states128 = v.clone();
        }
        if let Some(v) = state.get("r148") {
            self.base.contracts.pending148 = v.clone();
        }
        if let Some(v) = state.get("race") {
            self.race.states = v.clone();
        }
        if let Some(v) = state.get("open") {
            self.open.report["open_attack"] = v["open_attack"].clone();
        }
        if let Some(v) = state.get("herd") {
            self.herd.states = v.clone();
        }
    }

    /// JSONL oracle request for the differential harness (no engine transitions).
    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            *self = Self::default();
        }
        if let Some(state) = value.get("state") {
            self.inject(state);
        }
        let obs = &value["observation"];
        let config = value.get("configuration").cloned().unwrap_or(json!({}));
        let action = value.get("action").cloned().unwrap_or(Value::Null);
        let mode = value["mode"].as_str().unwrap_or("act");
        let result: Result<Value, String> = match mode {
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
            "open" => Ok(self.open.apply(obs, &config, action)),
            "race" => {
                let (key, has_state) = self.race.before(
                    obs,
                    &config,
                    &self.base.core,
                    &mut self.base.production.race_horizons,
                );
                let snapshot = self.race.snapshot_phase(obs, has_state);
                self.race.after(&key, has_state, snapshot, &action);
                Ok(action)
            }
            "race_lost" => {
                let key = int(&obs["player"]).to_string();
                race::lost(obs, &self.race.states[&key], &self.base.core)
                    .map(|v| json!(v))
                    .map_err(str::to_owned)
            }
            "clone_gate" => Ok(json!(self.race.clone_gate(obs))),
            "adv_apply" => self
                .adv
                .apply(obs, action, &mut self.base.core)
                .map_err(str::to_owned),
            "adv_front" => Ok(self.adv.frontload(obs, action)),
            "preguard" => self.preguard.apply(obs, action).map_err(str::to_owned),
            "lockstep" => self.lockstep.reorder(obs, action).map_err(str::to_owned),
            "herd" => {
                let key = int(&obs["player"]).to_string();
                if self.herd.states[&key].is_null() {
                    self.herd.states[&key] = herd::new_state();
                }
                herd::controller(
                    obs,
                    &action,
                    &mut self.herd.states[&key],
                    &mut self.herd.report,
                )
                .map_err(str::to_owned)
            }
            _ => Err(format!("unknown mode {mode}")),
        };
        match result {
            Ok(result) => {
                json!({"id":value["id"],"result":result,"state":self.states(),"engine_transitions":0})
            }
            Err(error) => json!({"id":value["id"],"error":error}),
        }
    }
}
