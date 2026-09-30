//! Native port of Ahmed Berat Özer's frozen Kaggriculture V43 public agent.
//! SHA-256 and full upstream Apache-2.0 notices live in agents/v43/.
pub mod common;
pub mod contracts;
pub mod core;
pub mod late;
pub mod production;
pub mod terminal;

use crate::Game;
use common::*;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct V43Controller {
    pub core: core::Core,
    pub contracts: contracts::Contracts,
    pub production: production::Production,
    terminal: terminal::Terminal,
    pub late: late::Late,
    debug: Value,
}
impl V43Controller {
    /// An interrupted terminal plan must never replay its projected future.
    pub(crate) fn cancel_terminal_plan(&mut self, seat: &str) {
        if let Some(plans) = self.terminal.plans.as_object_mut() {
            plans.remove(seat);
        }
    }
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        if !obs["farms"].is_array() || array(&obs["farms"]).len() < 2 {
            return common::pass_action();
        }
        let [terminal_action, room_action, production_action, late_action] =
            self.base_stages(obs, config);
        let action = self
            .contracts
            .after(obs, late_action.clone(), &self.core, config);
        self.debug = json!({"core":self.core.players,"production":{"v219":self.production.v219,"v231":self.production.v231,"v233":self.production.v233},"input":self.late.input,"contracts":{"r124":self.contracts.states124,"r127":self.contracts.states127,"r128":self.contracts.states128,"r148":self.contracts.pending148},"stages":{"terminal":terminal_action,"room":room_action,"production":production_action,"late":late_action,"final":action}});
        if let Some(stages) = self.contracts.stages.as_object() {
            for label in ["r124", "r127", "r128"] {
                if let Some(value) = stages.get(label) {
                    self.debug["stages"][label] = value.clone();
                }
            }
        }
        action
    }
    /// Terminal, room, production and late stages shared with the 2945 Farm chassis.
    fn base_stages(&mut self, obs: &Value, config: &Value) -> [Value; 4] {
        let seat = int(&obs["player"]);
        self.production.before(obs);
        let committed = truth(&self.production.v219[seat.to_string()]["committed"])
            || truth(&self.production.v233[seat.to_string()]["committed"]);
        let terminal = self.terminal.apply(obs, config, &mut self.core, committed);
        let room = core::room_guard(obs, terminal.clone());
        let valid = late::cfg_ok(config, false);
        let production = self
            .production
            .after(obs, room.clone(), &mut self.core, valid);
        let late = self.late.after(
            obs,
            production.clone(),
            &self.core,
            &self.production,
            config,
        );
        [terminal, room, production, late]
    }

    /// The 2945 Farm chassis: this controller without the R124-R148 contracts.
    /// The caller sets `core.farm2945` and `production.farm2945`.
    pub fn act_farm2945(&mut self, obs: &Value, config: &Value) -> Value {
        let [terminal, room, production, late] = self.base_stages(obs, config);
        self.debug = json!({"core":self.core.players,"production":{"v219":self.production.v219,"v231":self.production.v231,"v233":self.production.v233},"input":self.late.input,"stages":{"terminal":terminal,"room":room,"production":production,"late":late}});
        late
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid V43 seat {seat}"));
        }
        let mut obs = serde_json::to_value(&game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates()[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(game.configuration()).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }
    pub fn debug(&self) -> &Value {
        &self.debug
    }
}
