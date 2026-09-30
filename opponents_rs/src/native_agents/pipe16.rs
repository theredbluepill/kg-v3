//! Pure Rust port of Nathan Jacob's Pipe16 Idle Workers wrapper.
//!
//! The frozen source appends a fixed `HybridOpening` to the complete Metav4
//! agent. Upstream sources, notices and checksums remain in agents/pipe16/.
//! Unselected developer alternatives (Mixed, EarlyCycle, TomatoInsteadOfCow)
//! are not activated by the published entry point.

use crate::Game;
use crate::native_agents::metav4::Metav4Controller;
use crate::native_agents::v43::common::{array, int};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Pipe16Controller {
    pub base: Metav4Controller,
    /// Python `_ALT_STATE`, keyed by seat; repeats/rewinds reset that seat.
    pub opening: Value,
    /// Python `_ALT_REPORT`; shared by the seats in a module namespace.
    pub report: Value,
    stages: Value,
}

impl Default for Pipe16Controller {
    fn default() -> Self {
        let mut base = Metav4Controller::default();
        base.configure_pipe16();
        Self {
            base,
            opening: json!({}),
            report: json!({}),
            stages: Value::Null,
        }
    }
}

/// Source `_alt_sell_extra`: merge only the first matching sale, otherwise
/// append if the order list has space. A full list leaves the action unchanged.
pub fn sell_extra(mut action: Value, item: &str, n: i64) -> Value {
    if n <= 0 {
        return action;
    }
    let mut orders = array(&action["market"]).to_vec();
    if let Some(order) = orders
        .iter_mut()
        .find(|order| array(order).len() >= 3 && order[0] == "SELL" && order[1] == item)
    {
        order[2] = json!(int(&order[2]) + n);
    } else if orders.len() < 10 {
        orders.push(json!(["SELL", item, n]));
    } else {
        return action;
    }
    action["market"] = json!(orders);
    action
}

impl Pipe16Controller {
    fn begin(&mut self, obs: &Value) {
        let seat = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        if self.opening[&seat].is_null() || step <= int(&self.opening[&seat]["step"]) {
            self.opening[&seat] = json!({"step":-1,"mode":"HybridOpening"});
            self.report = json!({
                "selected_opening":"HybridOpening",
                "temporary_crop_seen":0,"temporary_crop_harvested":0,
                "restored_pasture_seen":0,"delivered_extra_wheat":0,
                "tomato_seen":0,"tomato_water_requests":0,
                "tomato_harvest_requests":0,"tomato_units_harvest_requested":0,
                "extension_errors":0
            });
        }
    }

    /// Post-parent part of the active HybridOpening wrapper. Keep exception
    /// telemetry and assignment semantics, including zero stock/full markets.
    fn suffix(&mut self, obs: &Value, mut action: Value) -> Value {
        let seat = int(&obs["player"]) as usize;
        let step = int(&obs["step"]);
        let result = (|| -> Result<(), ()> {
            let farm = obs.get("farms").and_then(|v| v.get(seat)).ok_or(())?;
            let private = obs.get("private").ok_or(())?;
            let site = farm
                .get("tiles")
                .and_then(|v| v.get(4))
                .and_then(|v| v.get(2))
                .ok_or(())?;
            if step == 6 {
                self.report["temporary_crop_seen"] =
                    json!(i64::from(site.is_object() && site["crop"] == "WHEAT"));
            }
            if step == 54 {
                let inventory = private
                    .get("inventories")
                    .and_then(|v| v.get(1))
                    .and_then(Value::as_object)
                    .ok_or(())?;
                self.report["temporary_crop_harvested"] =
                    json!(inventory.get("WHEAT").map(int).unwrap_or(0));
            }
            if step == 55 {
                self.report["restored_pasture_seen"] =
                    json!(i64::from(site.is_object() && site["kind"] == "PASTURE"));
            }
            if step == 57 {
                let hands = farm.get("hands").and_then(Value::as_array).ok_or(())?;
                if !hands.is_empty() && hands[0] == json!([4, 4]) {
                    // Python defaults missing hands to [[]], but a present
                    // empty list raises inside the extension's try block.
                    let drops = match action.get("hands") {
                        None => false,
                        Some(v) => {
                            v.as_array().and_then(|v| v.first()).ok_or(())? == &json!(["DROP"])
                        }
                    };
                    if drops {
                        let inventory = private
                            .get("inventories")
                            .and_then(|v| v.get(1))
                            .and_then(Value::as_object)
                            .ok_or(())?;
                        let n = inventory.get("WHEAT").map(int).unwrap_or(0);
                        action = sell_extra(action.clone(), "WHEAT", n);
                        // Source reports delivery requested even if market
                        // capacity prevents appending a SELL, and when n<=0.
                        self.report["delivered_extra_wheat"] = json!(n);
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.report["extension_errors"] = json!(int(&self.report["extension_errors"]) + 1);
        }
        self.opening[seat.to_string()]["step"] = json!(step);
        action
    }

    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        self.begin(obs);
        let parent = self.base.act(obs, config);
        let action = self.suffix(obs, parent.clone());
        self.stages = json!({"parent":parent,"final":action});
        action
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid pipe16 seat {seat}"));
        }
        let mut obs = serde_json::to_value(&game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates()[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(game.configuration()).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }

    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut debug = self.base.debug();
        for (key, value) in self.stages.as_object().expect("stage mapping") {
            debug["stages"][key] = value.clone();
        }
        debug["reports"]["pipe16"] = self.report.clone();
        debug
    }

    pub fn states(&self) -> Value {
        let mut state = self.base.states();
        state["pipe16"] = self.opening.clone();
        state["reports"]["pipe16"] = self.report.clone();
        state["debug"] = self.debug();
        state
    }

    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            *self = Self::default();
        }
        if let Some(state) = value.get("state") {
            self.base.inject(state);
            if let Some(opening) = state.get("pipe16") {
                self.opening = opening.clone();
            }
            if let Some(report) = state.get("reports").and_then(|v| v.get("pipe16")) {
                self.report = report.clone();
            }
        }
        let obs = &value["observation"];
        let config = value.get("configuration").cloned().unwrap_or(json!({}));
        let mode = value["mode"].as_str().unwrap_or("act");
        let result = match mode {
            "act" => self.act(obs, &config),
            "clone_act" => {
                let mut fork = self.clone();
                let result = self.act(obs, &config);
                let cloned = fork.act(obs, &config);
                if result != cloned || self.states() != fork.states() {
                    return json!({"id":value["id"],"error":"cloned controller diverged"});
                }
                result
            }
            "pipe_suffix" => {
                self.begin(obs);
                self.suffix(obs, value["action"].clone())
            }
            "pipe_sell_extra" => sell_extra(
                value["action"].clone(),
                value["item"].as_str().unwrap_or(""),
                int(&value["n"]),
            ),
            "pipe_route" => self
                .base
                .base
                .base
                .core
                .configured_route_action(int(&value["route"]), int(&value["step"])),
            _ => {
                // Forward all Metav4 direct-layer oracle modes; resetting here
                // already recreated the correctly configured parent.
                let mut request = value.clone();
                request["reset"] = json!(false);
                let mut response = self.base.oracle_request(&request);
                response["state"] = self.states();
                return response;
            }
        };
        json!({"id":value["id"],"result":result,"state":self.states(),"engine_transitions":0})
    }
}
