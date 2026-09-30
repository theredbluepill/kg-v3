//! Independent invariants for interrupted-proposal memory. Full saved-game
//! action replay is qualified by the Python execution-teacher harness.
use super::Cha22Controller;
use crate::{Config, Game};
use serde_json::{Value, json};

fn observation(step: i64) -> (Value, Value) {
    let game = Game::new(Config::default(), 0, 2).unwrap();
    let mut obs = serde_json::to_value(&game.snapshot().public).unwrap();
    obs["player"] = json!(0);
    obs["private"] = serde_json::to_value(&game.privates()[0]).unwrap();
    obs["step"] = json!(step);
    obs["day"] = json!(step / 24);
    obs["hour"] = json!(step % 24);
    (obs, serde_json::to_value(game.configuration()).unwrap())
}

fn different_action(proposal: &Value) -> Value {
    let mut action = proposal.clone();
    action["farmer"] = if proposal["farmer"] == json!(["PASS"]) {
        json!(["NORTH"])
    } else {
        json!(["PASS"])
    };
    action
}

#[test]
fn equality_commit_preserves_every_controller_field() {
    let (obs, config) = observation(0);
    let before = Cha22Controller::default();
    let mut teacher = before.clone();
    let proposal = teacher.act(&obs, &config);
    // Debug covers private caches as well as the public states() diagnostic.
    let expected = format!("{teacher:?}");
    teacher.commit_executed(&before, &obs, &proposal, &proposal);
    assert_eq!(format!("{teacher:?}"), expected);
}

#[test]
fn divergence_retains_new_mirror_observation_and_quote_history() {
    let (mut obs, config) = observation(1);
    obs["market"]["inventory"]["WHEAT"] = json!(100);
    let mut teacher = Cha22Controller::default();
    teacher.act(&obs, &config);
    assert_eq!(teacher.early.mirror["prior"], json!(100));
    assert_eq!(teacher.early.mirror["mirror_like"], json!(false));
    let old_quote = teacher.market.fx_states["0"]["quotes"]["1"].clone();

    obs["step"] = json!(2);
    obs["hour"] = json!(2);
    obs["market"]["inventory"]["WHEAT"] = json!(80);
    obs["market"]["prices"]["WOOL"] = json!(177);
    let before = teacher.clone();
    let proposal = teacher.act(&obs, &config);
    assert_eq!(teacher.early.mirror["mirror_like"], json!(true));
    let executed = different_action(&proposal);
    teacher.commit_executed(&before, &obs, &proposal, &executed);
    assert_eq!(teacher.early.mirror["prior"], json!(100));
    assert_eq!(teacher.early.mirror["mirror_like"], json!(true));
    assert_eq!(teacher.market.fx_states["0"]["quotes"]["1"], old_quote);
    assert_eq!(teacher.market.fx_states["0"]["quotes"]["2"]["WOOL"], json!(177));
    assert_eq!(teacher.market.fx_states["0"]["prev"]["inventory"], obs["market"]["inventory"]);
}

#[test]
fn divergence_records_executed_market_for_next_rival_inference() {
    let (mut obs, config) = observation(215);
    obs["private"]["shed"]["WOOL"] = json!(9);
    let before = Cha22Controller::default();
    let mut teacher = before.clone();
    let proposal = teacher.act(&obs, &config);
    let executed = json!({"farmer":["PASS"],"hands":[],"market":[["SELL","WOOL",3]]});
    assert_ne!(proposal, executed);
    teacher.commit_executed(&before, &obs, &proposal, &executed);
    assert_eq!(teacher.base.race.states["0"]["prev_action"], executed);
    assert_eq!(teacher.base.race.states["0"]["prev"]["step"], json!(215));
    assert_eq!(teacher.market.fx_states["0"]["prev"]["own"]["WOOL"], json!(3));
    assert_eq!(teacher.base.base.race.states["0"]["prev"]["own"]["WOOL"], json!(3));
}

#[test]
fn divergence_cancels_nested_intentions_but_keeps_confirmed_facts() {
    let (mut obs, _) = observation(433);
    obs["private"]["inventories"][0]["COW"] = json!(2);
    obs["private"]["shed"]["COW"] = json!(3);
    let mut before = Cha22Controller::default();
    before.base.base.base.production.v219 = json!({"0":{"committed":false}});
    let mut teacher = before.clone();
    // These distinguish facts absorbed on this callback from contingent effects
    // of its proposed HIRE/PICKUP/SELL, independently of the hook's plumbing.
    teacher.base.base.base.core.players = json!({"0":{
        "route":2,"router_state":{"first":"YARN_STORE"},"pending":{"0":[["PLANT","TOMATO"]]},
        "sell_state":{"due_step":434,"suppress":{"WOOL":9},"r36_debts":{"436":{"WOOL":9}}}
    }});
    let production = &mut teacher.base.base.base.production;
    production.v219 = json!({"0":{"last_step":433,"day":18,"eligible":true,
        "seen_plants":[[5,5]],"lost":[[6,5]],"targets":[[5,5],[6,5]],
        "committed":true,"requested_day":18,"pending":{"first_actor":2},
        "workers":{"1":{"loaded":true}},"last_work":{"1":{"command":["HARVEST"]}}
    }});
    production.v233 = json!({"0":{"committed":true,"last_step":433,"day":18,
        "pending":{"first":2},"requested_day":18,"workers":{"1":[[5,5]]},
        "work":{"0":{"command":["HARVEST"]}},"credit":{"WOOL":9},"rescue_today":6
    }});
    production.v231 = json!({"0":{"confirmed":4,"pending_buy":{"quantity":2},
        "pending_places":[{"actor":0}],"carrying":{"0":8},"reserved":8,"milk_credit":30
    }});
    production.r37 = json!({"0":{"step":433,"streak":6}});
    production.r44 = json!({"0":{"step":433,"money":[9000,8000],"matched":true,"probe":1000}});
    teacher.base.base.base.late.input = json!({"0":{"workers":{"1":{"loaded":true,"path":[]}},"pending":{"2":{}},"placed":[[5,5]]}});
    teacher.market.bd_states = json!({"0":{"hist":[100,101],"pending":4,"since":432}});
    let proposal = json!({"farmer":["HARVEST"],"hands":[],"market":[["HIRE"]]});
    let executed = json!({"farmer":["PASS"],"hands":[],"market":[]});
    teacher.commit_executed(&before, &obs, &proposal, &executed);

    let core = &teacher.base.base.base.core.players["0"];
    assert_eq!(core["route"], json!(2));
    assert_eq!(core["router_state"]["first"], json!("YARN_STORE"));
    assert_eq!(core["pending"], json!({}));
    assert_eq!(core["sell_state"]["r36_debts"], json!({}));
    let production = &teacher.base.base.base.production;
    let tomato = &production.v219["0"];
    assert_eq!(tomato["eligible"], json!(true));
    assert_eq!(tomato["seen_plants"], json!([[5,5]]));
    assert_eq!(tomato["lost"], json!([[6,5]]));
    assert_eq!(tomato["committed"], json!(false));
    assert!(tomato.get("pending").is_none());
    assert!(tomato.get("requested_day").is_none());
    assert_eq!(tomato["workers"], json!({}));
    assert_eq!(production.v233["0"]["committed"], json!(true));
    assert_eq!(production.v233["0"]["work"]["0"]["command"], json!(["PASS"]));
    assert_eq!(production.v233["0"]["work"]["0"]["inventory"], obs["private"]["inventories"][0]);
    assert_eq!(production.v231["0"]["confirmed"], json!(4));
    assert_eq!(production.v231["0"]["carrying"]["0"], json!(2));
    assert_eq!(production.v231["0"]["reserved"], json!(3));
    assert_eq!(production.r37["0"]["streak"], json!(6));
    assert_eq!(production.r44["0"]["money"], json!([9000,8000]));
    assert_eq!(production.r44["0"]["matched"], json!(true));
    assert_eq!(production.r44["0"]["probe"], json!(0));
    assert_eq!(teacher.base.base.base.late.input["0"]["workers"], json!({}));
    assert_eq!(teacher.market.bd_states["0"]["hist"], json!([100,101]));
    assert_eq!(teacher.market.bd_states["0"]["pending"], json!(0));
}

#[test]
fn movement_recovery_survives_repeated_missed_correction_until_observed_plant() {
    let (mut obs, _) = observation(9);
    obs["farms"][0]["farmer"] = json!([4,0]);
    obs["farms"][0]["tiles"][0][3] = Value::Null;
    obs["private"]["seeds"]["MELON"] = json!(2);
    let before = Cha22Controller::default();
    let mut teacher = before.clone();
    let west = json!({"farmer":["WEST"],"hands":[],"market":[]});
    let pass = json!({"farmer":["PASS"],"hands":[],"market":[]});
    teacher.commit_executed(&before, &obs, &west, &pass);
    assert_eq!(teacher.recovery["0"]["target"], json!([3,0]));

    obs["step"] = json!(10);
    let plant = json!({"farmer":["PLANT","MELON"],"hands":[],"market":[]});
    let correction = teacher.recover_movement(&obs, plant.clone());
    assert_eq!(correction["farmer"], json!(["WEST"]));
    assert_eq!(teacher.recovery["0"]["deferred"], json!(["PLANT","MELON"]));
    let before_second = teacher.clone();
    teacher.commit_executed(&before_second, &obs, &correction, &pass);
    assert_eq!(teacher.recovery["0"]["target"], json!([3,0]));
    assert_eq!(teacher.recovery["0"]["deferred"], json!(["PLANT","MELON"]));

    obs["step"] = json!(11);
    obs["farms"][0]["farmer"] = json!([3,0]);
    assert_eq!(teacher.recover_movement(&obs, pass.clone())["farmer"], plant["farmer"]);
    obs["step"] = json!(12);
    // An unrelated seed purchase is not evidence that the site operation ran.
    obs["private"]["seeds"]["MELON"] = json!(3);
    assert_eq!(teacher.recover_movement(&obs, pass.clone())["farmer"], plant["farmer"]);
    assert!(teacher.recovery.get("0").is_some());
    obs["step"] = json!(13);
    obs["farms"][0]["tiles"][0][3] = json!({"kind":"PLANT","crop":"MELON","planted_day":0});
    assert_eq!(teacher.recover_movement(&obs, pass.clone()), pass);
    assert!(teacher.recovery.get("0").is_none());
}

#[test]
fn market_only_divergence_does_not_create_spatial_recovery() {
    let (obs, _) = observation(9);
    let before = Cha22Controller::default();
    let mut teacher = before.clone();
    let proposed = json!({"farmer":["WEST"],"hands":[],"market":[["BUY_PRODUCT","WHEAT",9]]});
    let executed = json!({"farmer":["WEST"],"hands":[],"market":[["BUY_PRODUCT","WHEAT",5]]});
    teacher.commit_executed(&before, &obs, &proposed, &executed);
    assert_eq!(teacher.recovery, json!({}));
}

#[test]
fn spatial_recovery_expires_at_midnight() {
    let (obs, _) = observation(24);
    let mut teacher = Cha22Controller::default();
    teacher.recovery = json!({"0":{"day":0,"target":[0,0],"deferred":["PLANT","MELON"]}});
    let pass = json!({"farmer":["PASS"],"hands":[],"market":[]});
    assert_eq!(teacher.recover_movement(&obs, pass.clone()), pass);
    assert_eq!(teacher.recovery, json!({}));
}
