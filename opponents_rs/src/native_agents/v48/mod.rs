//! Native port of Ahmed Berat Özer's frozen Kaggriculture V48 public agent.
//! SHA-256 and full upstream Apache-2.0 notices live in agents/v48/.
//!
//! V48 is the V47 controller plus the EXP334/335 market compaction layer
//! (`_e334_agent` / `_e335_agent`); routes and production are unchanged.
pub mod compact;

use crate::Game;
use crate::native_agents::v43::common::*;
use crate::native_agents::v47::V47Controller;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct V48Controller {
    pub inner: V47Controller,
    pub report: Value,
    stages: Value,
}
impl Default for V48Controller {
    fn default() -> Self {
        Self {
            inner: V47Controller::default(),
            report: json!({"changed":0,"removed":0,"errors":0}),
            stages: Value::Null,
        }
    }
}

impl V48Controller {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        if int(&obs["step"]) == 0 {
            for k in ["changed", "removed", "errors"] {
                self.report[k] = json!(0);
            }
        }
        let herd = self.inner.act(obs, config);
        let result = match compact::compact(obs, &herd, &mut self.report) {
            Ok(result) => result,
            Err(_) => {
                increment(&mut self.report, "errors", 1);
                herd.clone()
            }
        };
        self.stages = json!({"compact": result, "final": result});
        result
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid V48 seat {seat}"));
        }
        let mut obs = serde_json::to_value(&game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates()[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(game.configuration()).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }

    pub fn debug(&self) -> Value {
        let mut debug = self.inner.debug();
        if debug.is_null() {
            return debug;
        }
        if let Some(stages) = self.stages.as_object() {
            for (k, v) in stages {
                debug["stages"][k] = v.clone();
            }
        }
        debug["compact_report"] = self.report.clone();
        debug
    }

    pub fn states(&self) -> Value {
        let mut states = self.inner.states();
        states["debug"] = self.debug();
        states["compact_report"] = self.report.clone();
        states
    }

    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            *self = Self::default();
        }
        let obs = &value["observation"];
        let config = value.get("configuration").cloned().unwrap_or(json!({}));
        let action = value.get("action").cloned().unwrap_or(Value::Null);
        let mode = value["mode"].as_str().unwrap_or("act");
        let result: Result<Value, String> = match mode {
            "act" => {
                if let Some(state) = value.get("state") {
                    self.inner.inject(state);
                }
                Ok(self.act(obs, &config))
            }
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
            "compact" => compact::compact(obs, &action, &mut self.report).map_err(str::to_owned),
            "compact_original" => {
                compact::original(obs, &action, &mut self.report).map_err(str::to_owned)
            }
            _ => {
                let mut request = value.clone();
                request["reset"] = json!(false);
                let response = self.inner.oracle_request(&request);
                if response.get("error").is_some() {
                    return response;
                }
                return json!({"id":value["id"],"result":response["result"],"state":self.states(),"engine_transitions":0});
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
