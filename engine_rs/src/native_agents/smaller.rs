//! Native port of Dmitrii Gluzdov's "Kaggriculture: A Smaller Market Shock".
//!
//! The public source is a Farm2945/V39 chassis with one bounded `EarlyCycle`
//! opening overlay.  The implementation keeps the complete Farm2945 parent in
//! one place and enables its route-0, CT_TABLE and step-57 suffix hooks.

use crate::native_agents::farm2945::Farm2945Controller;
use crate::Game;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct SmallerController {
    pub base: Farm2945Controller,
}

impl Default for SmallerController {
    fn default() -> Self {
        let mut base = Farm2945Controller::default();
        base.configure_smaller_market();
        Self { base }
    }
}

impl SmallerController {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        self.base.act(obs, config)
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        self.base.action(game, seat)
    }

    pub fn debug(&self) -> Value {
        self.base.debug()
    }

    pub fn oracle_request(&mut self, value: &Value) -> Value {
        self.base.oracle_request(value)
    }
}
