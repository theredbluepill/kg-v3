//! Native Metav4 Farm submission v13. Exact upstream source and notices:
//! agents/metav4/main.py (SHA-256 9d63494603f88219857a3101d7dc19cc750ded04e5886732ee428580326967d9).
//! Reuses the unchanged Farm2945 layers and V47/V48 suffix mechanisms, with
//! controller-local source constants, sale library and V13 tomato skip.
pub mod library_lengths;

use crate::Game;
use crate::native_agents::farm2945::Farm2945Controller;
use crate::native_agents::v43::{common::*, contracts::Contracts, late::cfg_ok};
use crate::native_agents::v47::{herd, lockstep, preguard, race};
use crate::native_agents::v48::compact;
use serde_json::{Value, json};

const PRIORITY_REPORT: [&str; 6] = [
    "last_hour_plants_dropped",
    "priority_grain_orders",
    "priority_grain_units",
    "priority_grain_confirmed",
    "priority_grain_shortfalls",
    "priority_contract_errors",
];

#[derive(Clone, Debug)]
pub struct Metav4Controller {
    pub base: Farm2945Controller,
    pub race: race::Race,
    pub priority: Contracts,
    pub preguard: preguard::Preguard,
    pub lockstep: lockstep::Lockstep,
    pub herd: herd::Herd,
    pub compact_report: Value,
    stages: Value,
}

impl Default for Metav4Controller {
    fn default() -> Self {
        let mut base = Farm2945Controller::default();
        base.base.core.metav4 = true;
        base.base.production.metav4 = true;
        Self {
            base,
            race: Default::default(),
            priority: Default::default(),
            preguard: Default::default(),
            lockstep: Default::default(),
            herd: Default::default(),
            compact_report: json!({"changed":0,"removed":0,"errors":0}),
            stages: Value::Null,
        }
    }
}

impl Metav4Controller {
    pub fn configure_pipe16(&mut self) {
        self.base.base.core.hybrid_opening = true;
    }

    pub fn priority_layer(&mut self, obs: &Value, config: &Value, mut action: Value) -> Value {
        let key = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        let mut state = self.priority.states127[&key].clone();
        if state.is_null() || step <= int(&state["step"]) {
            state = json!({"step":-1});
            for k in PRIORITY_REPORT {
                self.priority.report[k] = json!(0);
            }
        }
        let pending = state
            .as_object_mut()
            .and_then(|m| m.remove("pending_grain"))
            .unwrap_or(Value::Null);
        if truth(&pending) && step == int(&pending[0]) {
            let k = if int(&obs["private"]["shed"]["WHEAT"]) >= int(&pending[1]) {
                "priority_grain_confirmed"
            } else {
                "priority_grain_shortfalls"
            };
            increment(
                &mut self.priority.report,
                k,
                if k == "priority_grain_confirmed" {
                    int(&pending[2])
                } else {
                    1
                },
            );
        }
        state["step"] = json!(step);
        if cfg_ok(config, true) {
            let result = self.priority.last_hour(obs, action.clone()).and_then(|a| {
                self.priority
                    .priority(obs, a, &self.base.base.core, &mut state)
            });
            match result {
                Ok(a) => action = a,
                Err(_) => increment(&mut self.priority.report, "priority_contract_errors", 1),
            }
        }
        self.priority.states127[&key] = state;
        action
    }

    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        if !obs["farms"].is_array() || array(&obs["farms"]).len() < 2 {
            return pass_action();
        }
        if int(&obs["step"]) == 0 {
            self.compact_report = json!({"changed":0,"removed":0,"errors":0});
            self.base.base.production.skip_report =
                json!({"v_skipped":0,"v_blocked":0,"v_errors":0});
        }
        let (key, exists) = self.race.before(
            obs,
            config,
            &self.base.base.core,
            &mut self.base.base.production.race_horizons,
        );
        let snapshot = self.race.snapshot_phase(obs, exists);
        let farm = self.base.act(obs, config);
        self.race.after(&key, exists, snapshot, &farm);
        let priority = self.priority_layer(obs, config, farm.clone());
        let preguard = self.preguard.layer(obs, config, priority.clone());
        // Metav4 sets _V44Y_REORDER_GATE=False, so every qualifying turn is ordered.
        let lockstep = if int(&obs["step"]) >= 216 {
            match self.lockstep.reorder(obs, preguard.clone()) {
                Ok(a) => a,
                Err(_) => {
                    increment(&mut self.lockstep.report, "v44y_errors", 1);
                    preguard.clone()
                }
            }
        } else {
            preguard.clone()
        };
        let herd = self.herd.layer(obs, lockstep.clone());
        let compact = match compact::compact(obs, &herd, &mut self.compact_report) {
            Ok(a) => a,
            Err(_) => {
                increment(&mut self.compact_report, "errors", 1);
                herd.clone()
            }
        };
        self.stages = json!({"farm2945":farm,"race47":farm,"r127":priority,"preguard":preguard,"lockstep":lockstep,"shopherd":herd,"compact":compact,"final":compact});
        compact
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid metav4 seat {seat}"));
        }
        let mut obs = serde_json::to_value(&game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates()[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(game.configuration()).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }

    pub fn reports(&self) -> Value {
        let mut reports = self.base.reports();
        let mut priority = json!({});
        for k in PRIORITY_REPORT {
            priority[k] = self.priority.report[k].clone();
        }
        reports["race47"] = self.race.report.clone();
        reports["r127"] = priority;
        reports["preguard"] = self.preguard.report.clone();
        reports["lockstep"] = self.lockstep.report.clone();
        reports["shopherd"] = self.herd.report.clone();
        reports["compact"] = self.compact_report.clone();
        reports["skip"] = self.base.base.production.skip_report.clone();
        reports
    }

    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut debug = self.base.debug();
        for (k, v) in self.stages.as_object().unwrap() {
            debug["stages"][k] = v.clone();
        }
        debug["reports"] = self.reports();
        debug
    }

    pub fn states(&self) -> Value {
        let mut states = self.base.states();
        states["race47"] = self.race.states.clone();
        states["r127"] = self.priority.states127.clone();
        states["shopherd"] = self.herd.states.clone();
        states["compact_report"] = self.compact_report.clone();
        states["skip_report"] = self.base.base.production.skip_report.clone();
        states["reports"] = self.reports();
        states["debug"] = self.debug();
        states
    }

    pub fn inject(&mut self, state: &Value) {
        self.base.inject(state);
        if let Some(v) = state.get("predict") {
            self.base.predict.inject(v);
        }
        if let Some(v) = state.get("race") {
            self.base.race.states = v.clone();
        }
        if let Some(v) = state.get("race47") {
            self.race.states = v.clone();
        }
        if let Some(v) = state.get("r127") {
            self.priority.states127 = v.clone();
        }
        if let Some(v) = state.get("shopherd") {
            self.herd.states = v.clone();
        }
        if let Some(v) = state.get("v219") {
            self.base.base.production.v219 = v.clone();
        }
    }

    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            let pipe = self.base.base.core.hybrid_opening;
            *self = Self::default();
            if pipe {
                self.configure_pipe16();
            }
        }
        if let Some(state) = value.get("state") {
            self.inject(state);
        }
        let obs = &value["observation"];
        let config = value.get("configuration").cloned().unwrap_or(json!({}));
        let action = value.get("action").cloned().unwrap_or(Value::Null);
        let result: Result<Value, String> = match value["mode"].as_str().unwrap_or("act") {
            "act" => Ok(self.act(obs, &config)),
            "clone_act" => {
                let mut clone = self.clone();
                let a = self.act(obs, &config);
                let b = clone.act(obs, &config);
                if a != b || self.states() != clone.states() {
                    Err("cloned controller diverged".into())
                } else {
                    Ok(a)
                }
            }
            "library" => Ok(crate::native_agents::farm2945::race::library_pair(
                true,
                int(&value["pair"]) as usize,
            )),
            "predict" => {
                Ok(self
                    .base
                    .predict
                    .layer(obs, action, &self.base.base.core, &self.base.race))
            }
            "route" => Ok(self
                .base
                .base
                .core
                .configured_route_action(int(&value["route"]), int(&value["step"]))),
            "race47" => {
                let (key, exists) = self.race.before(
                    obs,
                    &config,
                    &self.base.base.core,
                    &mut self.base.base.production.race_horizons,
                );
                let snapshot = self.race.snapshot_phase(obs, exists);
                self.race.after(&key, exists, snapshot, &action);
                Ok(action)
            }
            "race_lost" => race::lost(
                obs,
                &self.race.states[int(&obs["player"]).to_string()],
                &self.base.base.core,
            )
            .map(|v| json!(v))
            .map_err(str::to_owned),
            "clone_gate" => Ok(json!(self.race.clone_gate(obs))),
            "r127" => Ok(self.priority_layer(obs, &config, action)),
            "priority" => {
                let key = int(&obs["player"]).to_string();
                let mut state = self.priority.states127[&key].clone();
                if state.is_null() {
                    state = json!({"step":-1});
                }
                let result = self
                    .priority
                    .priority(obs, action, &self.base.base.core, &mut state)
                    .map_err(str::to_owned);
                self.priority.states127[&key] = state;
                result
            }
            "last_hour" => self.priority.last_hour(obs, action).map_err(str::to_owned),
            "preguard" => self.preguard.apply(obs, action).map_err(str::to_owned),
            "lockstep" => self.lockstep.reorder(obs, action).map_err(str::to_owned),
            "shopherd" => {
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
            "compact" => {
                compact::compact(obs, &action, &mut self.compact_report).map_err(str::to_owned)
            }
            "compact_original" => {
                compact::original(obs, &action, &mut self.compact_report).map_err(str::to_owned)
            }
            "tomato_request" => {
                let key = int(&obs["player"]).to_string();
                let mut state = self.base.base.production.v219[&key].clone();
                let native = self.base.base.core.players[&key].clone();
                let result = self
                    .base
                    .base
                    .production
                    .tomato_request(obs, action, &mut state, &native);
                self.base.base.production.v219[&key] = state;
                Ok(result)
            }
            _ => {
                let mut req = value.clone();
                req["reset"] = json!(false);
                let response = self.base.oracle_request(&req);
                if let Some(e) = response.get("error") {
                    Err(e.as_str().unwrap_or("base oracle error").to_owned())
                } else {
                    Ok(response["result"].clone())
                }
            }
        };
        match result {
            Ok(result) => {
                json!({"id":value["id"],"result":result,"state":self.states(),"engine_transitions":0})
            }
            Err(error) => json!({"id":value["id"],"error":error}),
        }
    }
}
