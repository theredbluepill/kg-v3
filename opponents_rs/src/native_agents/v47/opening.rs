//! V47 EXP284/293 opening arm: step-0 wheat round trip and step-1/2 attack.
//! Source: agents/v47/main.py (`_OPEN_*`, opening `agent`).
use super::race::standard;
use crate::native_agents::v43::common::*;
use serde_json::{Value, json};

pub const ATTACK: i64 = 30;
pub const ATTACK_MIN_CASH: f64 = 2860.0;

#[derive(Clone, Debug)]
pub struct Open {
    pub report: Value,
    /// V47/V48 use the 30-unit attack.  The tetsu342 source changes only
    /// this literal to 22, so it is carried as per-controller state.
    pub attack: i64,
}
impl Default for Open {
    fn default() -> Self {
        Self {
            report: json!({"open_turns":0,"open_errors":0}),
            attack: ATTACK,
        }
    }
}

impl Open {
    pub fn apply(&mut self, obs: &Value, config: &Value, action: Value) -> Value {
        let mut result = action.clone();
        let outcome: Result<(), &'static str> = (|| {
            let step = int(obs.get("step").ok_or("open: step")?);
            if step == 0 {
                self.report["open_turns"] = json!(0);
                self.report["open_errors"] = json!(0);
                self.report["open_attack"] = json!(0);
            }
            let standard = standard(config, true);
            if standard
                && step == 0
                && result["market"]
                    == json!([
                        ["BUY_PRODUCT", "WHEAT", 5],
                        ["BUY_PRODUCT", "WHEAT", 10],
                        ["SELL", "WHEAT", 60]
                    ])
            {
                result["market"] = json!([["BUY_PRODUCT", "WHEAT", 7], ["SELL", "WHEAT", 2]]);
                increment(&mut self.report, "open_turns", 1);
            } else if standard && step == 1 {
                let mut market = orders(&result);
                if market.len() >= 2
                    && market[0] == json!(["SELL", "WHEAT", 13])
                    && market[1] == json!(["BUY_PRODUCT", "WHEAT", 5])
                {
                    market.drain(..2);
                    increment(&mut self.report, "open_turns", 1);
                    let player = int(&obs["player"]) as usize;
                    let money = obs["farms"]
                        .get(player)
                        .and_then(|f| f.get("money"))
                        .ok_or("open: money")?;
                    if num(money) >= ATTACK_MIN_CASH {
                        market.insert(0, json!(["BUY_PRODUCT", "WHEAT", self.attack]));
                        self.report["open_attack"] = json!(1);
                    }
                    set_orders(&mut result, market);
                }
            } else if standard && step == 2 && truth(&self.report["open_attack"]) {
                let mut market = vec![json!(["SELL", "WHEAT", self.attack])];
                market.extend(orders(&result).into_iter().take(9));
                set_orders(&mut result, market);
                increment(&mut self.report, "open_turns", 1);
            }
            Ok(())
        })();
        if outcome.is_err() {
            increment(&mut self.report, "open_errors", 1);
            return action;
        }
        result
    }
}
