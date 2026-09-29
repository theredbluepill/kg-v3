use super::replay_export::*;
use kaggriculture_engine::Config;
use serde_json::{json, Value};

fn header_json() -> Value {
    let mut configuration = serde_json::to_value(Config::default()).unwrap();
    configuration["episodeSteps"] = json!(7);
    configuration["turnsPerDay"] = json!(2);
    configuration["seed"] = Value::Null;
    json!({"seed":123, "configuration":configuration,
        "provenance":{"source":"unit-test", "engine":"pinned-native", "schema_version":1,
            "target_framework_version":"1.32.7"},
        "specification":{"observation":{
            "remainingOverageTime":{"default":60}, "step":{"shared":true},
            "player":{},"farms":{"shared":true},"private":{},
            "market":{"shared":true},"town":{"shared":true},
            "day":{"shared":true},"hour":{"shared":true}},
            "action":{"default":{"farmer":["PASS"],"hands":[],"market":[]}},
            "reward":{"default":0}},
        "envelope":{"id":"native-test","name":"kaggriculture","title":"Kaggriculture",
            "description":"test schema only","version":"0.1.0","module_version":"1.32.7","schema_version":1}})
}

fn action() -> Value {
    json!({"farmer":["PASS"],"hands":[],"market":[]})
}
fn tape_json(n: usize, complete: bool) -> Value {
    json!({"complete":complete,"transitions": (0..n).map(|_|json!({"actions":[action(), action()]})).collect::<Vec<_>>()})
}

#[test]
fn replay_seed_only_poison_and_exact_wide_seed() {
    let mut raw = header_json();
    raw["seed"] = serde_json::from_str("1267650600228229401496703205377").unwrap();
    let header = SeedHeader::parse(&raw).unwrap();
    let tape = ActionTape::parse(&tape_json(6, true)).unwrap();
    let expected = replay_from_seed(&header, &tape).unwrap();
    assert_eq!(expected.len(), 7);
    assert!(expected[6].done);
    assert_eq!(header.seed.to_string(), "1267650600228229401496703205377");
    let mut poison = header.trace_header().unwrap();
    poison.initial.public.farms[0].money = -99999.;
    poison.initial.privates[1].shed.insert("WHEAT".into(), 999);
    poison.shop_schedule = vec!["POISON".into()];
    poison.rng_schedule = vec![kaggriculture_engine::RngDay {
        day: 1,
        weed_rolls: vec![1.; 200],
    }];
    poison.terminal_banks = vec![-1., -2.];
    poison.transitions = 9999;
    let mut game = kaggriculture_engine::Game::from_seed_header(&poison).unwrap();
    assert_eq!(
        serde_json::to_value(game.snapshot()).unwrap(),
        serde_json::to_value(&expected[0]).unwrap()
    );
    for row in tape.transitions {
        game.step(&row.actions).unwrap();
    }
    assert_eq!(
        serde_json::to_value(game.snapshot()).unwrap(),
        serde_json::to_value(&expected[6]).unwrap()
    );
}

#[test]
fn replay_rejects_invalid_seed_config_counts_and_completion() {
    for seed in [
        json!(null),
        json!(-1),
        json!(1.0),
        json!(1.5),
        json!("1"),
        json!(true),
    ] {
        let mut h = header_json();
        h["seed"] = seed;
        assert!(SeedHeader::parse(&h).unwrap_err().pointer.contains("seed"));
    }
    let mut h = header_json();
    h.as_object_mut().unwrap().shift_remove("seed");
    assert!(SeedHeader::parse(&h).is_err());
    for (field, value) in [
        ("boardSize", json!(9)),
        ("marketParams", json!({"WHEAT":{}})),
        ("extra", json!(1)),
        ("maxMarketOrdersPerTurn", json!(11)),
        ("startingMoney", json!(9223372036854775808_u64)),
    ] {
        let mut h = header_json();
        h["configuration"][field] = value;
        assert!(SeedHeader::parse(&h).is_err(), "{field}");
    }
    let h = SeedHeader::parse(&header_json()).unwrap();
    let mut t = tape_json(1, false);
    t["transitions"][0]["actions"] = json!([action()]);
    assert!(ActionTape::parse(&t).is_err());
    for (n, complete, needle) in [(5, true, "shorter"), (7, true, "DONE")] {
        let err =
            replay_from_seed(&h, &ActionTape::parse(&tape_json(n, complete)).unwrap()).unwrap_err();
        assert!(err.to_string().contains(needle), "{err}");
    }
}

#[test]
fn token_tape_uses_pre_step_native_decode() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let p = super::grammar::plan(1, 10, 21).unwrap();
    let mut tokens = vec![0; super::grammar::TOKENS_PER_SEAT];
    let length = super::grammar::encode(&p, &action(), &mut tokens).unwrap();
    let mut t = tape_json(6, true);
    t["transitions"][0]["tokens"] =
        json!([{"tokens":tokens,"length":length},{"tokens":tokens,"length":length}]);
    assert!(replay_from_seed(&h, &ActionTape::parse(&t).unwrap()).is_ok());
    t["transitions"][0]["actions"][0]["farmer"] = json!(["NORTH"]);
    let err = replay_from_seed(&h, &ActionTape::parse(&t).unwrap()).unwrap_err();
    assert_eq!(err.transition, Some(0));
    assert!(err.pointer.contains("actions/0"));
}

#[test]
fn episode_timing_shared_privacy_quantities_and_byte_roundtrip() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let mut raw = tape_json(6, true);
    raw["transitions"][0]["actions"][0] =
        json!({"farmer":["PICKUP","WHEAT"],"hands":[],"market":[[],["SELL","WHEAT",0]]});
    raw["transitions"][1]["actions"][1] =
        json!({"farmer":["PASS"],"hands":[],"market":[["BUY_SEED","CARROT",3]]});
    let tape = ActionTape::parse(&raw).unwrap();
    let episode = export_kaggle_episode(&h, &tape).unwrap();
    assert_eq!(
        episode["steps"][1][0]["action"],
        raw["transitions"][0]["actions"][0]
    );
    assert_ne!(
        episode["steps"][2][0]["observation"]["private"],
        episode["steps"][2][1]["observation"]["private"]
    );
    assert_eq!(episode["steps"][0][1]["observation"]["step"], Value::Null);
    assert_eq!(
        episode["steps"][0][1]["observation"]["farms"],
        episode["steps"][0][0]["observation"]["farms"]
    );
    let report = verify_round_trip(&episode.to_string(), None).unwrap();
    assert_eq!(report.mode, "byte");
    assert_eq!(report.canonical_json, episode.to_string());
    let mut omitted = episode.clone();
    omitted["info"]
        .as_object_mut()
        .unwrap()
        .shift_remove("v3_native_replay");
    for step in omitted["steps"].as_array_mut().unwrap() {
        for field in ["farms", "market", "town", "day", "hour"] {
            step[1]["observation"]
                .as_object_mut()
                .unwrap()
                .shift_remove(field);
        }
    }
    assert!(verify_round_trip(&omitted.to_string(), None).is_ok());
    let mut bad = episode.clone();
    bad["steps"][6][0]["reward"] = json!(0.8);
    let err = verify_round_trip(&bad.to_string(), None).unwrap_err();
    assert_eq!(err.pointer, "/steps/6/0/reward");
    assert_eq!(err.transition, Some(5));
    eprintln!("mutation shaped reward: {err}");
    let mut bad = episode.clone();
    let seeds = bad["steps"][1][0]["observation"]["private"]["seeds"]
        .as_object_mut()
        .unwrap();
    let key = seeds.keys().next().unwrap().clone();
    let value = seeds.shift_remove(&key).unwrap();
    seeds.insert(key, value);
    let err = verify_round_trip(&bad.to_string(), None).unwrap_err();
    assert!(err.pointer.ends_with("/private/seeds"));
    assert_eq!(err.transition, Some(0));
    eprintln!("mutation order: {err}");
}

#[test]
fn import_rejections_and_captured_evidence_are_not_circular() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let t = ActionTape::parse(&tape_json(6, true)).unwrap();
    let e = export_kaggle_episode(&h, &t).unwrap();
    for (pointer, value) in [
        ("/info/seed", Value::Null),
        ("/steps/1/0/observation/step", json!(3)),
        ("/steps/6/1/status", json!("ACTIVE")),
        ("/statuses", json!(["ACTIVE", "ACTIVE"])),
        ("/steps/1/0/action", Value::Null),
    ] {
        let mut bad = e.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(import_kaggle_episode(&bad).is_err(), "{pointer}");
    }
    let mut bad = e.clone();
    bad["steps"].as_array_mut().unwrap().pop();
    assert!(import_kaggle_episode(&bad).is_err());
    let mut bad = e.clone();
    bad["steps"][1].as_array_mut().unwrap().pop();
    assert!(import_kaggle_episode(&bad).is_err());
    let snapshots = replay_from_seed(&h, &t).unwrap();
    let banks: Vec<_> = snapshots[1..]
        .iter()
        .map(|s| s.public.farms.iter().map(|f| f.money).collect::<Vec<_>>())
        .collect();
    let mut captured = json!({"initial":snapshots[0],"terminal":snapshots[6],"banks":banks,"snapshots":[{"transition":2,"snapshot":snapshots[3]}]});
    assert!(verify_round_trip(&e.to_string(), Some(&captured)).is_ok());
    captured["banks"][2][1] = json!(2999.);
    let err = verify_round_trip(&e.to_string(), Some(&captured)).unwrap_err();
    assert_eq!(err.pointer, "/captured/banks/2/1");
    assert_eq!(err.transition, Some(2));
    eprintln!("mutation captured bank: {err}");
}

#[test]
fn semantic_numbers_are_exact_without_float_seed_rounding() {
    let number = |s: &str| serde_json::from_str::<Value>(s).unwrap();
    assert!(compare_tree(&number("0.00001"), &number("1e-5"), "/number").is_ok());
    assert!(compare_tree(
        &number("9007199254740993"),
        &number("9007199254740992.0"),
        "/number"
    )
    .is_err());
}

#[test]
fn native_step_error_is_explicit_and_never_shortened_success() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let mut raw = tape_json(6, true);
    raw["transitions"][2]["actions"][0]["farmer"] = json!(["PICKUP", "WHEAT", null]);
    let err = replay_from_seed(&h, &ActionTape::parse(&raw).unwrap()).unwrap_err();
    assert_eq!(err.transition, Some(2));
    assert!(err.message.contains("native step error"), "{err}");
    eprintln!("native error: {err}");
}

#[test]
fn native_byte_numeric_spelling_reports_the_first_pointer() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let t = ActionTape::parse(&tape_json(6, true)).unwrap();
    let mut e = export_kaggle_episode(&h, &t).unwrap();
    e["steps"][1][0]["observation"]["farms"][0]["money"] = json!(3000);
    let err = verify_round_trip(&e.to_string(), None).unwrap_err();
    assert_eq!(err.pointer, "/steps/1/0/observation/farms/0/money");
    assert_eq!(err.transition, Some(0));
}

#[test]
fn typed_tape_rejects_token_seat_count_without_panicking() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    for n in [0, 1, 3] {
        let mut tape = ActionTape::parse(&tape_json(1, false)).unwrap();
        let p = super::grammar::plan(1, 10, 21).unwrap();
        let mut tokens = vec![0; super::grammar::TOKENS_PER_SEAT];
        let length = super::grammar::encode(&p, &action(), &mut tokens).unwrap();
        tape.transitions[0].tokens = Some(vec![ExecutedTokens { tokens, length }; n]);
        let result = replay_from_seed(&h, &tape);
        assert!(result.is_err(), "token seats={n}");
    }
}

#[test]
fn shifted_action_and_dropped_transition_are_detected() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let mut raw = tape_json(6, true);
    raw["transitions"][1]["actions"][0]["market"] = json!([["BUY_PRODUCT", "WHEAT", 2]]);
    let mut e = export_kaggle_episode(&h, &ActionTape::parse(&raw).unwrap()).unwrap();
    e["steps"][1][0]["action"] = e["steps"][2][0]["action"].clone();
    let err = verify_round_trip(&e.to_string(), None).unwrap_err();
    assert_eq!(err.transition, Some(0));
    assert!(err.pointer.ends_with("/farms/0/money"));
    eprintln!("mutation action timing: {err}");
    e["steps"].as_array_mut().unwrap().remove(3);
    let err = verify_round_trip(&e.to_string(), None).unwrap_err();
    assert_eq!(err.pointer, "/steps/3/0/observation/step");
    assert_eq!(err.transition, Some(2));
    eprintln!("mutation dropped transition: {err}");
}

#[test]
fn importer_rejects_incomplete_terminal_state_before_replay() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let t = ActionTape::parse(&tape_json(6, true)).unwrap();
    let e = export_kaggle_episode(&h, &t).unwrap();
    for field in ["private", "town", "market"] {
        let mut bad = e.clone();
        bad["steps"][6][1]["observation"][field] = json!({});
        assert!(
            import_kaggle_episode(&bad).is_err(),
            "malformed terminal {field}"
        );
    }
}

#[test]
fn captured_empty_replay_returns_error() {
    assert!(compare_captured(&[], &json!({"initial":{}})).is_err());
}

// Claude review: Kaggle writes the schema reward default (integer 0) until DONE,
// then float(money) (pinned kaggriculture.py:963). Native f64 0.0 is not that.
#[test]
fn active_rewards_use_the_kaggle_integer_default_and_done_rewards_are_floats() {
    let h = SeedHeader::parse(&header_json()).unwrap();
    let t = ActionTape::parse(&tape_json(6, true)).unwrap();
    let e = export_kaggle_episode(&h, &t).unwrap();
    for step in 0..6 {
        for seat in 0..2 {
            assert_eq!(
                e["steps"][step][seat]["reward"].to_string(),
                "0",
                "step {step}"
            );
        }
    }
    assert_eq!(e["steps"][6][0]["reward"].to_string(), "3000.0");
    assert_eq!(e["rewards"].to_string(), "[3000.0,3000.0]");
}

// Claude review: semantic equality may normalize decimal spelling but must not
// conflate a JSON integer with a float (Kaggle rewards/money distinguish them).
#[test]
fn semantic_numbers_keep_integer_and_float_kinds_distinct() {
    let number = |s: &str| serde_json::from_str::<Value>(s).unwrap();
    for (a, b) in [("0", "0.0"), ("3000", "3000.0"), ("1", "1e0")] {
        let err = compare_tree(&number(a), &number(b), "/number").unwrap_err();
        assert_eq!(err.pointer, "/number", "{a} vs {b}");
    }
    for (a, b) in [("0.00001", "1e-5"), ("1e16", "1e+16"), ("12.50", "12.5")] {
        assert!(
            compare_tree(&number(a), &number(b), "/number").is_ok(),
            "{a} vs {b}"
        );
    }
}

// Claude review: replay derives the grammar hire cap as turnsPerDay * orders + 1.
// Hands clear at end of day, so that is the day's maximum actors + hires; a day
// that hires on every order of every turn must still decode, and the cap is
// reached exactly (actors + hires = cap - 1 before the last HIRE).
#[test]
fn derived_hire_cap_accepts_a_day_of_maximal_hiring() {
    let mut raw = header_json();
    raw["configuration"]["turnsPerDay"] = json!(2);
    raw["configuration"]["maxMarketOrdersPerTurn"] = json!(2);
    raw["configuration"]["episodeSteps"] = json!(4);
    let h = SeedHeader::parse(&raw).unwrap();
    let hire = |actors: usize| json!({"farmer":["PASS"],"hands":vec![json!(["PASS"]); actors - 1],"market":[["HIRE"],["HIRE"]]});
    let mut tape = tape_json(2, false);
    tape["transitions"][0]["actions"][0] = hire(1);
    tape["transitions"][1]["actions"][0] = hire(3);
    let snapshots = replay_from_seed(&h, &ActionTape::parse(&tape).unwrap()).unwrap();
    assert_eq!(
        snapshots[1].public.farms[0].hands.len(),
        2,
        "both day-one hires ran"
    );
    assert_eq!(
        snapshots[2].public.farms[0].hands.len(),
        0,
        "hands clear at day end"
    );
    let derived = 2 * 2 + 1;
    for t in 0..2 {
        let actors = snapshots[t].public.farms[0].hands.len() + 1;
        let row = |action: &Value, actors: usize| {
            let plan = super::grammar::plan(i64::try_from(actors).unwrap(), 2, 241).unwrap();
            let mut tokens = vec![0; super::grammar::TOKENS_PER_SEAT];
            let length = super::grammar::encode(&plan, action, &mut tokens).unwrap();
            json!({"tokens":tokens,"length":length})
        };
        let program = tape["transitions"][t]["actions"][0].clone();
        tape["transitions"][t]["tokens"] = json!([row(&program, actors), row(&action(), 1)]);
        if t == 1 {
            assert_eq!(actors + 1, derived - 1, "last HIRE sits at the derived cap");
        }
    }
    assert!(replay_from_seed(&h, &ActionTape::parse(&tape).unwrap()).is_ok());
}
