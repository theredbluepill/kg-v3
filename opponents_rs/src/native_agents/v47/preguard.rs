//! V47 v44y pre-guard: quote at hours 21/22 what the hour-23 storage guard would dump.
//! Source: agents/v47/main.py (`_y_preguard`, `agent_v44y_preguard`), after Seyit Kaan Gunes.
use super::race::standard;
use crate::native_agents::v43::common::*;
use crate::native_agents::v43::contracts::fields;
use crate::native_agents::v43::late::market_stock;
use serde_json::{Value, json};

pub const HOURS: [i64; 2] = [21, 22];
pub const ITEMS: [&str; 5] = ["MILK", "STRAWBERRY", "MELON", "WOOL", "TOMATO"];
pub const MIN_DAY: i64 = 1;
pub const MARGIN: i64 = -6;

#[derive(Clone, Debug)]
pub struct Preguard {
    pub report: Value,
}
impl Default for Preguard {
    fn default() -> Self {
        Self {
            report: json!({"preguard_turns":0,"preguard_units":0,"preguard_errors":0}),
        }
    }
}

impl Preguard {
    pub fn apply(&mut self, obs: &Value, action: Value) -> Result<Value, &'static str> {
        let step = int(obs.get("step").ok_or("preguard: step")?);
        if !HOURS.contains(&step.rem_euclid(24)) || step / 24 < MIN_DAY || step >= 696 {
            return Ok(action);
        }
        let os = orders(&action);
        if os.len() >= 10 {
            return Ok(action);
        }
        let (_, private) = fields(obs, &action)?;
        let (stock, _, _) = market_stock(&private["shed"], &os);
        let mut carried = 0;
        for bag in array(&private["inventories"]) {
            let items = bag.as_object().ok_or("preguard: inventory")?;
            carried += items.values().map(|n| int(n).max(0)).sum::<i64>();
        }
        let held: i64 = stock
            .as_object()
            .map(|m| m.values().map(|v| int(v).max(0)).sum())
            .unwrap_or(0);
        let mut needed = held + carried - 99 - MARGIN;
        if needed <= 0 {
            return Ok(action);
        }
        let prices = obs["market"].get("prices").ok_or("preguard: prices")?;
        let mut items: Vec<&str> = PRODUCTS.to_vec();
        items.sort_by_key(|it| -int(&prices[*it]));
        let mut extra: Vec<Value> = vec![];
        for item in items {
            let avail = int(&stock[item]).max(0);
            let qty = needed.min(avail);
            if qty <= 0 {
                continue;
            }
            if ITEMS.contains(&item) && int(&prices[item]) >= 2 {
                extra.push(json!(["SELL", item, qty]));
            }
            needed -= qty;
            if needed <= 0 {
                break;
            }
        }
        if extra.is_empty() || os.len() + extra.len() > 10 {
            return Ok(action);
        }
        increment(&mut self.report, "preguard_turns", 1);
        let units: i64 = extra.iter().map(|o| int(&o[2])).sum();
        increment(&mut self.report, "preguard_units", units);
        let mut result = action;
        let mut market = os;
        market.extend(extra);
        set_orders(&mut result, market);
        Ok(result)
    }

    pub fn layer(&mut self, obs: &Value, config: &Value, action: Value) -> Value {
        if !standard(config, false) {
            return action;
        }
        match self.apply(obs, action.clone()) {
            Ok(result) => result,
            Err(_) => {
                increment(&mut self.report, "preguard_errors", 1);
                action
            }
        }
    }
}
