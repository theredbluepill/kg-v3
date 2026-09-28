//! Native port of tetsutani's frozen "Demand-Preserving Turn Sale Timing"
//! public agent (variant `demand_preserving_sale_lookahead_14_v65`).
//! SHA-256 and full upstream Apache-2.0 notices live in agents/tetsu65/.
//!
//! The source is Ahmed Berat Özer's V48 with one changed layer: the EXP293 sale
//! advance looks 14 turns ahead instead of 3 (3 is kept while two bakeries are
//! open) and leaves WOOL sales five or more turns out alone while a yarn store
//! is open. Routes, production, race, herd and compaction are V48's.
use crate::Game;
use crate::native_agents::v48::V48Controller;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Tetsu65Controller {
    pub inner: V48Controller,
}
impl Default for Tetsu65Controller {
    fn default() -> Self {
        let mut inner = V48Controller::default();
        inner.inner.adv.demand_preserving = true;
        Self { inner }
    }
}

impl Tetsu65Controller {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        self.inner.act(obs, config)
    }

    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid tetsu65 seat {seat}"));
        }
        self.inner.action(game, seat)
    }

    pub fn debug(&self) -> Value {
        self.inner.debug()
    }

    pub fn states(&self) -> Value {
        self.inner.states()
    }

    /// JSONL oracle request; a reset must restore the v65 advance setting.
    pub fn oracle_request(&mut self, value: &Value) -> Value {
        if value["reset"] == true {
            *self = Self::default();
        }
        let mut request = value.clone();
        request["reset"] = json!(false);
        self.inner.oracle_request(&request)
    }
}
