//! Native frozen cha22 full ig_agent entry (SHA256 127ed3e6…).
//! Attribution and exact upstream source: agents/cha22/main.py.
pub mod early;
mod execution;
#[cfg(test)]
mod execution_tests;
pub mod market;
pub mod tail;
use crate::Game;
use crate::native_agents::metav4::Metav4Controller;
use crate::native_agents::v43::common::*;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Cha22Controller {
    pub base: Metav4Controller,
    pub early: early::Early,
    pub market: market::Market,
    pub tail: tail::Tail,
    stages: Value,
    execution: Value,
    recovery: Value,
}
impl Default for Cha22Controller {
    fn default() -> Self {
        let mut base = Metav4Controller::default();
        base.base.base.core.cha22 = true;
        base.base.base.production.cha22 = true;
        base.base.early.cha22 = true;
        base.base.market.empty_ctrtable = true;
        Self {
            base,
            early: Default::default(),
            market: Default::default(),
            tail: Default::default(),
            stages: Value::Null,
            execution: Value::Null,
            recovery: json!({}),
        }
    }
}
fn merge(dest: &mut Value, other: Value) {
    for (k, v) in other.as_object().unwrap() {
        dest[k] = v.clone();
    }
}
impl Cha22Controller {
    pub fn act(&mut self, obs: &Value, config: &Value) -> Value {
        let recovery_before = if self.recovery.as_object().is_some_and(|m| !m.is_empty()) {
            Some(self.clone())
        } else {
            None
        };
        self.tail.before(obs);
        self.market.before(obs);
        self.early.before(obs);
        let parent = self.base.act(obs, config);
        let weedlag = self
            .early
            .weed(obs, config, parent.clone(), &self.base.base.base.core);
        let advance = self
            .early
            .advance(obs, config, weedlag.clone(), &mut self.base);
        let terminal_clear = self.early.terminal(obs, advance.clone());
        // T62A overwrites its parent's market in place. RACE's prev_action points
        // to that same Python object after ADV, before later copy-on-write layers.
        if terminal_clear != advance {
            let st = &mut self.base.race.states[int(&obs["player"]).to_string()];
            if !st.is_null() && st["step"] == obs["step"] && !st["prev_action"].is_null() {
                st["prev_action"] = terminal_clear.clone();
            }
        }
        let pipe = self.early.pipe(obs, terminal_clear.clone());
        let buyfirst = self.early.buyfirst(obs, pipe.clone());
        let core = &self.base.base.base.core;
        let flow = self.market.fx_ev(
            obs,
            buyfirst.clone(),
            core,
            truth(&self.early.mirror["mirror_like"]),
        );
        let dawn = self.market.dawn(obs, flow.clone(), core);
        let midday = self.market.midday(obs, dawn.clone(), core);
        let bankdrip = self.market.bankdrip(obs, midday.clone());
        let premium = self.market.premium(obs, bankdrip.clone(), core);
        let shield = self.market.shield(obs, premium.clone());
        let reorder = self.tail.reorder(obs, shield.clone());
        let fertilizer = self.tail.fertilizer(obs, reorder.clone(), &mut self.base);
        let seeds = self.tail.seeds(obs, fertilizer.clone(), &mut self.base);
        let merged = self.tail.merge(obs, seeds.clone());
        let final_action = self.tail.queue(obs, merged.clone());
        self.stages = json!({"parent":parent,"weedlag":weedlag,"advance":advance,"terminal_clear":terminal_clear,"pipe":pipe,"mirror":pipe,"buyfirst":buyfirst,"flow":flow,"dawn":dawn,"midday":midday,"bankdrip":bankdrip,"premium":premium,"shield":shield,"reorder":reorder,"fertilizer":fertilizer,"seeds":seeds,"merge":merged,"final":final_action});
        if let Some(before) = recovery_before {
            let recovered = self.recover_movement(obs, final_action.clone());
            if recovered != final_action {
                self.reconcile_interruption(&before, obs, &final_action, &recovered);
            }
            recovered
        } else {
            final_action
        }
    }
    pub fn action(&mut self, game: &Game, seat: usize) -> Result<Value, String> {
        if seat > 1 {
            return Err(format!("invalid cha22 seat {seat}"));
        }
        let mut obs = serde_json::to_value(&game.snapshot().public).map_err(|e| e.to_string())?;
        obs["player"] = json!(seat);
        obs["private"] = serde_json::to_value(&game.privates()[seat]).map_err(|e| e.to_string())?;
        let config = serde_json::to_value(game.configuration()).map_err(|e| e.to_string())?;
        Ok(self.act(&obs, &config))
    }
    pub fn reports(&self) -> Value {
        let mut reports = self.base.reports();
        reports.as_object_mut().unwrap().shift_remove("predict");
        merge(&mut reports, self.early.reports());
        merge(&mut reports, self.market.reports());
        merge(
            &mut reports,
            json!({"cxd":self.tail.cxd_report,"fert":self.tail.fert_report,"seed":self.tail.seed_report,"mg":self.tail.mg_report,"ig":self.tail.ig_report}),
        );
        reports
    }
    pub fn debug(&self) -> Value {
        if self.stages.is_null() {
            return Value::Null;
        }
        let mut debug = self.base.debug();
        merge(&mut debug["stages"], self.stages.clone());
        debug["reports"] = self.reports();
        debug
    }
    pub fn states(&self) -> Value {
        let mut states = self.base.states();
        states.as_object_mut().unwrap().shift_remove("predict");
        merge(&mut states, self.early.states());
        merge(&mut states, self.market.states());
        merge(
            &mut states,
            json!({"seed_cache":self.tail.seed_cache,"cxd_models":self.tail.cxd_models,"cxd_parent_orders":self.tail.cxd_parent_orders}),
        );
        states["reports"] = self.reports();
        states["debug"] = self.debug();
        if !self.execution.is_null() {
            states["execution"] = self.execution.clone();
            states["execution_recovery"] = self.recovery.clone();
        }
        states
    }
    pub fn inject(&mut self, state: &Value) {
        self.base.inject(state);
        self.early.inject(state);
        self.market.inject(state);
        if let Some(v) = state.get("input") {
            self.base.base.base.late.input = v.clone();
        }
        for (k, slot) in [
            ("seed_cache", &mut self.tail.seed_cache),
            ("cxd_models", &mut self.tail.cxd_models),
            ("cxd_parent_orders", &mut self.tail.cxd_parent_orders),
        ] {
            if let Some(v) = state.get(k) {
                *slot = v.clone();
            }
        }
    }
    pub fn oracle_request(&mut self, v: &Value) -> Value {
        if v["reset"] == true {
            *self = Self::default();
        }
        if let Some(state) = v.get("state") {
            self.inject(state);
        }
        let obs = &v["observation"];
        let config = v.get("configuration").cloned().unwrap_or(json!({}));
        let a = v.get("action").cloned().unwrap_or(Value::Null);
        let result = match v["mode"].as_str().unwrap_or("act") {
            "act" => self.act(obs, &config),
            "drive_executed" => {
                let before = self.clone();
                let proposal = self.act(obs, &config);
                let executed = v.get("executed").unwrap_or(&proposal);
                self.commit_executed(&before, obs, &proposal, executed);
                proposal
            }
            "clone_act" => {
                let mut copy = self.clone();
                let a = self.act(obs, &config);
                let b = copy.act(obs, &config);
                if a != b || self.states() != copy.states() {
                    return json!({"id":v["id"],"error":"clone divergence"});
                }
                a
            }
            "weedlag" => self.early.weed_apply(obs, a, &self.base.base.base.core),
            "advance" => self
                .early
                .advance_apply(obs, a, &mut self.base.base.base.core),
            "terminal_clear" => self.early.terminal(obs, a),
            "pipe" => {
                self.early.before(obs);
                self.early.pipe(obs, a)
            }
            "buyfirst" => self.early.buyfirst(obs, a),
            "flow_apply" => self.market.flow_apply(
                obs,
                a,
                &self.base.base.base.core,
                truth(&self.early.mirror["mirror_like"]),
            ),
            "evening" => self.market.evening(obs, a, &self.base.base.base.core),
            "dawn" => self.market.dawn(obs, a, &self.base.base.base.core),
            "midday" => self.market.midday(obs, a, &self.base.base.base.core),
            "bankdrip" => self.market.bankdrip(obs, a),
            "premium" => self.market.premium(obs, a, &self.base.base.base.core),
            "shield" => self.market.shield(obs, a),
            "reorder" => match self.tail.reorder_apply(obs, a) {
                Ok(a) => a,
                Err(e) => return json!({"id":v["id"],"error":e}),
            },
            "fertilizer" => self.tail.fertilizer(obs, a, &mut self.base),
            "seeds" => self.tail.seeds(obs, a, &mut self.base),
            "merge" => match self.tail.merge_apply(a) {
                Ok(a) => a,
                Err(e) => return json!({"id":v["id"],"error":e}),
            },
            "queue" => match self.tail.queue_apply(obs, a) {
                Ok(a) => a,
                Err(e) => return json!({"id":v["id"],"error":e}),
            },
            "remaining" => json!(self.tail.remaining(
                int(&v["route"]),
                int(&v["step"]),
                &self.base
            )),
            "sheep_worker" => crate::native_agents::farm2945::sheep::worker_configured(
                obs,
                int(&v["actor"]) as usize,
                &v["targets"],
                &mut self.base.base.base.production.report,
                true,
            ),
            _ => {
                let mut req = v.clone();
                req["reset"] = json!(false);
                let result = self.base.oracle_request(&req);
                if result.get("error").is_some() {
                    return result;
                }
                result["result"].clone()
            }
        };
        json!({"id":v["id"],"result":result,"state":self.states(),"engine_transitions":0})
    }
}
