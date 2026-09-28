//! Native port of Ahmed Berat Özer's frozen Kaggriculture V39 public agent.
//! SHA-256 and full upstream Apache-2.0 notices live in agents/v39/.
pub mod common;
pub mod core;
pub mod late;
pub mod production;
pub mod terminal;

use crate::Game;
use common::*;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct V39Controller {
    pub core: core::Core,
    pub production: production::Production,
    terminal: terminal::Terminal,
    pub late: late::Late,
    debug: Value,
}
impl V39Controller {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        if !obs["farms"].is_array() || array(&obs["farms"]).len() < 2 {
            return common::pass_action();
        }
        let seat = int(&obs["player"]);
        self.production.before(obs);
        let committed = truth(&self.production.v219[seat.to_string()]["committed"])
            || truth(&self.production.v233[seat.to_string()]["committed"]);
        let action = self.terminal.apply(obs, config, &mut self.core, committed);
        let terminal_action = action.clone();
        let action = core::room_guard(obs, action);
        let room_action = action.clone();
        let valid = late::cfg_ok(config, false);
        let action = self.production.after(obs, action, &mut self.core, valid);
        let production_action = action.clone();
        let action = self
            .late
            .after(obs, action, &self.core, &self.production, config);
        self.debug = json!({"core":self.core.players,"production":{"v219":self.production.v219,"v231":self.production.v231,"v233":self.production.v233},"input":self.late.input,"stages":{"terminal":terminal_action,"room":room_action,"production":production_action,"final":action}});
        action
    }
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid V39 seat {seat}"));
        }
        let mut obs = serde_json::to_value(game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(&game.config).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }
    pub fn debug(&self) -> &Value {
        &self.debug
    }
}
