//! Execution-aware interruption policy, deliberately separate from frozen act().
//!
//! A divergent command cancels contingent plans. It does not rewind observed
//! prices, rival movement/stock, route classification, or confirmed plant facts.
//! This defines the off-policy teacher; it does not claim the frozen tape can
//! recover an arbitrary farm or that an emitted request succeeded in the engine.
use super::Cha22Controller;
use crate::native_agents::v43::{common::*, core::Core};
use serde_json::{Value, json};

fn assign(state: &mut Value, fields: Value) {
    if !state.is_object() {
        *state = json!({});
    }
    for (k, v) in fields.as_object().unwrap() {
        state[k] = v.clone();
    }
}
fn remove(state: &mut Value, names: &[&str]) {
    if let Some(object) = state.as_object_mut() {
        for name in names {
            object.remove(*name);
        }
    }
}
fn observed_tiles(obs: &Value, kind: &str, born: &str) -> Value {
    let mut out = json!({});
    let farm = &obs["farms"][int(&obs["player"]) as usize];
    for (y, row) in array(&farm["tiles"]).iter().enumerate() {
        for (x, tile) in array(row).iter().enumerate() {
            if tile["crop"] == kind || tile["animal"] == kind {
                out[format!("{x},{y}")] = tile[born].clone();
            }
        }
    }
    out
}
fn sales(obs: &Value, action: &Value) -> Value {
    let mut stock = Core::projected_shed(action, &View::new(obs));
    let mut own = json!({});
    for order in orders(action).iter().take(10) {
        if array(order).len() >= 3 && order[0] == "SELL" {
            let item = text(&order[1]);
            let n = int(&order[2]).max(0).min(int(&stock[item]).max(0));
            increment(&mut own, item, n);
            increment(&mut stock, item, -n);
        }
    }
    own
}

impl Cha22Controller {
    /// Commit a proposal clone using the action submitted for this seat.
    ///
    /// Equality is an exact no-op, including telemetry and intermediate-layer
    /// alias quirks. On interruption, observations already absorbed by act()
    /// survive while contingent worker/market plans are canceled or rebased.
    /// `obs` is the PRE-step observation; success is learned next observation.
    pub fn commit_executed(
        &mut self,
        before: &Self,
        obs: &Value,
        proposal: &Value,
        executed: &Value,
    ) {
        if proposal == executed {
            return;
        }
        let view = View::new(obs);
        let proposed = commands(proposal);
        let actual = commands(executed);
        for (actor, command) in proposed.iter().enumerate().take(view.positions.len()) {
            if actual.get(actor) == Some(command) {
                continue;
            }
            let (x, y) = position(&view.positions[actor]);
            let target = match text(&command[0]) {
                "WEST" => Some((x - 1, y)),
                "EAST" => Some((x + 1, y)),
                "NORTH" => Some((x, y - 1)),
                "SOUTH" => Some((x, y + 1)),
                _ => None,
            };
            let k = actor.to_string();
            if let Some((x, y)) = target
                && (0..view.board).contains(&x)
                && (0..view.board).contains(&y)
                && self.recovery[&k].is_null()
            {
                self.recovery[k] =
                    json!({"day":int(&obs["step"])/24,"target":[x,y],"deferred":null});
            }
        }
        self.reconcile_interruption(before, obs, proposal, executed);
    }

    /// Bounded spatial repair: return to the destination of a missed move and
    /// retain one interrupted site operation. Position and applicability are
    /// checked on every callback; the queue expires at midnight or despawn.
    pub(super) fn recover_movement(&mut self, obs: &Value, mut action: Value) -> Value {
        let view = View::new(obs);
        let day = int(&obs["step"]) / 24;
        let mut work = commands(&action);
        let entries = self.recovery.as_object().cloned().unwrap_or_default();
        for (key, mut entry) in entries {
            let actor = key.parse::<usize>().unwrap_or(usize::MAX);
            if actor >= view.positions.len() || actor >= work.len() || int(&entry["day"]) != day {
                self.recovery.as_object_mut().unwrap().remove(&key);
                continue;
            }
            let (x, y) = position(&view.positions[actor]);
            let (tx, ty) = position(&entry["target"]);
            if (x, y) != (tx, ty) {
                if entry["deferred"].is_null()
                    && matches!(
                        text(&work[actor][0]),
                        "PLANT"
                            | "BUILD_COOP"
                            | "BUILD_PASTURE"
                            | "WATER"
                            | "FEED"
                            | "HARVEST"
                            | "FERTILIZE"
                            | "CARE"
                            | "COLLECT_FERTILIZER"
                    )
                {
                    entry["deferred"] = work[actor].clone();
                }
                work[actor] = json!([if x > tx {
                    "WEST"
                } else if x < tx {
                    "EAST"
                } else if y > ty {
                    "NORTH"
                } else {
                    "SOUTH"
                }]);
                self.recovery[&key] = entry;
            } else {
                let deferred = &entry["deferred"];
                let tile = tile_at(&view.tiles, &view.positions[actor]);
                let current = json!({"tile":tile,"inventory":view.inv(actor)});
                let previous = &entry["issued_state"];
                // Advance only from the local operation's observed effect;
                // unrelated seed purchases or inventory changes are no proof.
                let observed_effect = !previous.is_null()
                    && match text(&deferred[0]) {
                        "PLANT" => tile["crop"] == deferred[1],
                        "BUILD_COOP" => tile["kind"] == "COOP",
                        "BUILD_PASTURE" => tile["kind"] == "PASTURE",
                        "WATER" => truth(&tile["watered_today"]),
                        "FEED" => truth(&tile["fed_today"]),
                        "CARE" => truth(&tile["cared_today"]),
                        "HARVEST" => {
                            int(&tile["yield_units"]) < int(&previous["tile"]["yield_units"])
                        }
                        "FERTILIZE" => {
                            int(&tile["fertilized_until_day"])
                                > int(&previous["tile"]["fertilized_until_day"])
                        }
                        "COLLECT_FERTILIZER" => !truth(&tile["fertilizer_available"]),
                        _ => false,
                    };
                if deferred.is_null()
                    || observed_effect
                    || crate::native_agents::v43::core::is_noop(
                        deferred,
                        tile,
                        view.inv(actor),
                        &view.seeds,
                        &view.positions[actor],
                        view.board,
                    )
                {
                    self.recovery.as_object_mut().unwrap().remove(&key);
                } else {
                    work[actor] = deferred.clone();
                    entry["issued_state"] = current;
                    self.recovery[&key] = entry;
                }
            }
        }
        set_commands(&mut action, work);
        action
    }

    pub(super) fn reconcile_interruption(
        &mut self,
        before: &Self,
        obs: &Value,
        proposal: &Value,
        executed: &Value,
    ) {
        let key = int(&obs["player"]).to_string();
        let step = int(&obs["step"]);
        let day = step / 24;
        let own = sales(obs, executed);
        let view = View::new(obs);
        let work = commands(executed);

        // Race inference uses observed previous book/farm plus actual requests.
        let race = &mut self.base.race.states[&key];
        if race.is_object() && !race["prev"].is_null() {
            race["prev_action"] = executed.clone();
        }
        self.base.base.race.after(obs, executed);
        self.base.base.orderpri2.observe_executed(obs, executed);
        for state in [
            &mut self.market.fx_states[&key],
            &mut self.market.mpx_hist[&key],
        ] {
            if state["prev"].is_object() {
                state["prev"]["own"] = own.clone();
            }
        }
        // BUYDIP's pending is a deferred purchase plan, not owned inventory.
        // Its quote history remains; an interrupted plan has no future debt.
        if self.market.bd_states[&key].is_object() {
            assign(
                &mut self.market.bd_states[&key],
                json!({"pending":0,"since":null}),
            );
        }
        assign(
            &mut self.early.weedlag[&key],
            json!({"step":step,"active":{}}),
        );

        let farm = &mut self.base.base;
        let core = &mut farm.base.core.players[&key];
        // Route/router_state are observation classifications. Queues and sales
        // suppression instead encode commands that the frozen tape issued.
        assign(
            core,
            json!({"pending":{},"sell_state":{"due_step":-1,"suppress":{},"r36_debts":{}}}),
        );
        farm.base.cancel_terminal_plan(&key);
        if self.base.priority.states127[&key].is_object() {
            remove(&mut self.base.priority.states127[&key], &["pending_grain"]);
        }
        let production = &mut farm.base.production;
        if production.r44[&key].is_object() {
            production.r44[&key]["probe"] = json!(0);
        }
        let tomato = &mut production.v219[&key];
        if tomato.is_object() {
            // eligible, seen_plants, lost, targets and confirmed prior commitment
            // survive. New requests/role progress cannot claim to have executed.
            let committed = truth(&before.base.base.base.production.v219[&key]["committed"]);
            assign(
                tomato,
                json!({"workers":{},"last_work":{},"committed":committed}),
            );
            remove(tomato, &["pending", "requested_day"]);
        }
        let cattle = &mut production.v231[&key];
        if cattle.is_object() {
            // Rebase animal accounting to visible inventory, keeping cumulative
            // observed confirmation counters. No predicted PICKUP/PLACE credit.
            let mut carrying = json!({});
            for (i, inventory) in array(&obs["private"]["inventories"]).iter().enumerate() {
                carrying[i.to_string()] = json!(int(&inventory["COW"]));
            }
            assign(
                cattle,
                json!({"pending_buy":null,"pending_places":[],"carrying":carrying,
                "reserved":int(&obs["private"]["shed"]["COW"]),"sites":observed_tiles(obs,"COW","placed_day"),"milk_credit":0}),
            );
        }
        let sheep = &mut production.v233[&key];
        if sheep.is_object() {
            // committed is set by observed confirmation of an older request.
            // Keep it; reject speculative new hire/rescue/worker progression.
            assign(
                sheep,
                json!({"workers":{},"work":{},"rescue_today":0,"credit":{"WOOL":0,"FERTILIZER":0}}),
            );
            remove(sheep, &["pending", "requested_day"]);
            for (actor, command) in work.iter().enumerate().take(view.invs.len()) {
                sheep["work"][actor.to_string()] =
                    json!({"step":step,"command":command,"inventory":view.inv(actor)});
            }
        }
        assign(
            &mut farm.base.late.input[&key],
            json!({"step":step,"day":day,"workers":{},"pending":null,"placed":[]}),
        );
        assign(
            &mut farm.early.courier[&key],
            json!({"step":step,"day":day,"plans":{}}),
        );
        assign(
            &mut farm.early.carrot[&key],
            json!({"step":step,"swapped":false,"tiles":observed_tiles(obs,"CARROT","planted_day")}),
        );
        // Spare/harvest credits are scheduling allocations, not observed stock.
        // Cancel them; the ordinary policy can still inspect actual stock.
        assign(
            &mut farm.carrot2.states[&key],
            json!({"step":step,"tiles":observed_tiles(obs,"CARROT","planted_day"),"spare_wheat":0,"spare_carrot":0,"credit":0}),
        );
        assign(
            &mut farm.capharv.states[&key],
            json!({"step":step,"credit":{}}),
        );
        for state in [&mut farm.herd2.herd[&key], &mut farm.herd2.cow[&key]] {
            // Daily market inventory history (`inv`) remains in herd2. A
            // substitution route that was interrupted is not an executed fact.
            assign(
                state,
                json!({"step":step,"decided":false,"mode":null,"plan":null,"broken":false,"pending":[],"sites":{},"credit":0}),
            );
        }
        assign(
            &mut self.base.herd.states[&key],
            json!({"last":step,"pending":[],"credit":{},"sites":{},"sale":{},"coop_swap":0}),
        );
        // reports/stages describe proposals, explicitly not engine outcomes.
        // Keep full observed inputs available for audit and reconstruction.
        self.execution = json!({"policy":"cancel-contingent-plans-v1","step":step,
            "proposal":proposal,"executed":executed,"observation":obs});
    }
}
