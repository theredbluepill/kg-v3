use super::*;
use serde_json::json;

#[test]
fn debug_does_not_expose_hidden_engine_fields() {
    let game = Game::new(Config::default(), 123456789, 2).unwrap();
    let debug = format!("{game:?}");
    for hidden in [
        "seed:",
        "123456789",
        "econ_counters",
        "econ_land_used",
        "attrib:",
    ] {
        assert!(
            !debug.contains(hidden),
            "debug leaked hidden field {hidden}"
        );
    }
    assert!(debug.contains("farms:"));
}

#[test]
fn fibonacci_and_config_are_exact() {
    assert_eq!(
        (0..8).map(fib).collect::<Vec<_>>(),
        [1, 1, 2, 3, 5, 8, 13, 21].map(BigInt::from)
    );
    assert_eq!(fib(100).to_string(), "573147844013817084101");
    let config = Config {
        farm_hand_cost_mult: 37_i64.into(),
        ..Config::default()
    };
    assert_eq!(
        ControllerConfig::from_config(&config)
            .unwrap()
            .farm_hand_cost_mult
            .0,
        BigInt::from(37)
    );
}

#[test]
fn snapshot_refreshes_only_after_success() {
    let config = Config::default();
    let engine = kaggriculture_engine::Game::new(config.clone(), 17, 2).unwrap();
    let mut view = Game::from_engine(engine, &config).unwrap();
    let before = serde_json::to_string(view.snapshot()).unwrap();
    assert!(view.step(&[]).is_err());
    assert_eq!(serde_json::to_string(view.snapshot()).unwrap(), before);
    view.step(&[json!({}), json!({})]).unwrap();
    assert_eq!(view.step_index(), 1);
    assert_eq!(
        serde_json::to_string(view.snapshot()).unwrap(),
        serde_json::to_string(&view.engine.snapshot()).unwrap()
    );
}

fn visibility(kind: OpponentKind) {
    let mut game = Game::new(Config::default(), 17, 2).unwrap();
    let mut controllers = [
        SeatController::new(kind, 0, &game).unwrap(),
        SeatController::new(kind, 1, &game).unwrap(),
    ];
    let mut positive = [0; 2];
    for step in 0..719 {
        let mut actions = Vec::new();
        for seat in 0..2 {
            let mut expected = controllers[seat].clone();
            let action = expected.action(&game, seat).unwrap();
            if [0, 1, 23, 24, 250, 696, 718].contains(&step) {
                let mut hidden = game.clone();
                let rival = &mut hidden.snapshot.privates[1 - seat];
                rival.shed.insert("CARROT".into(), 987);
                rival.seeds.insert("WHEAT".into(), 432);
                for inv in &mut rival.inventories {
                    inv.insert("MILK".into(), 765);
                }
                assert_eq!(
                    controllers[seat].clone().action(&hidden, seat).unwrap(),
                    action,
                    "{} step {step} seat {seat}: rival private leak",
                    kind.key()
                );
                let mut own = game.clone();
                own.snapshot.public.farms[seat].money = 0.0;
                own.snapshot.privates[seat].shed.insert("CARROT".into(), 99);
                own.snapshot.privates[seat].seeds.clear();
                let altered = controllers[seat].clone().action(&own, seat).unwrap();
                positive[seat] += usize::from(altered != action);
            }
            assert_eq!(controllers[seat].action(&game, seat).unwrap(), action);
            actions.push(action);
        }
        game.step(&actions).unwrap();
    }
    assert!(
        positive.iter().all(|&n| n > 0),
        "{} positive controls {positive:?}",
        kind.key()
    );
    eprintln!(
        "visibility {}: 7 checkpoints per seat; positive changes {positive:?}",
        kind.key()
    );
}

#[test]
fn starter_visibility() {
    visibility(OpponentKind::Starter);
}
#[test]
fn r04_visibility() {
    visibility(OpponentKind::R04);
}
#[test]
fn ecobot_visibility() {
    visibility(OpponentKind::Ecobot);
}
#[test]
fn e776_visibility() {
    visibility(OpponentKind::E776);
}
#[test]
fn cha22_visibility() {
    visibility(OpponentKind::Cha22);
}
