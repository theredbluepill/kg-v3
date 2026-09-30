//! A host-owned engine driving `HostedSeat`s reproduces `play_match` exactly.
use kaggriculture_opponents::{Config, HostedSeat, OpponentKind, play_match};

fn hosted_actions(seed: i64, kinds: [OpponentKind; 2]) -> (Vec<[serde_json::Value; 2]>, [f64; 2]) {
    let config = Config::default();
    let mut engine = kaggriculture_engine::Game::new(config.clone(), seed, 2).unwrap();
    let mut seats = [
        HostedSeat::new(kinds[0], 0, &config, engine.snapshot()).unwrap(),
        HostedSeat::new(kinds[1], 1, &config, engine.snapshot()).unwrap(),
    ];
    let mut actions = Vec::new();
    while !engine.snapshot().done {
        let pair = [
            seats[0].action(engine.snapshot()).unwrap(),
            seats[1].action(engine.snapshot()).unwrap(),
        ];
        engine.step_with_market_metrics(&pair).unwrap();
        actions.push(pair);
    }
    let banks = engine.seat_money();
    // A completed game refuses another turn.
    assert!(
        seats[0]
            .action(engine.snapshot())
            .unwrap_err()
            .contains("completion")
    );
    (actions, [banks[0], banks[1]])
}

#[test]
fn hosted_seats_reproduce_play_match_for_every_bot_in_both_seats() {
    let pairs = [
        (20260941, [OpponentKind::Starter, OpponentKind::R04]),
        (20260942, [OpponentKind::E776, OpponentKind::Ecobot]),
        (20260944, [OpponentKind::R04, OpponentKind::Starter]),
        (20260945, [OpponentKind::Ecobot, OpponentKind::E776]),
        // One mirror game hosts Cha22 (the anchor bot) in both seats.
        (20260943, [OpponentKind::Cha22, OpponentKind::Cha22]),
    ];
    // Every registered bot is hosted in each seat by some game.
    for kind in OpponentKind::ALL {
        for seat in 0..2 {
            assert!(
                pairs.iter().any(|(_, kinds)| kinds[seat] == kind),
                "{kind:?} is not hosted in seat {seat}"
            );
        }
    }
    for (seed, kinds) in pairs {
        let reference = play_match(Config::default(), seed, kinds).unwrap();
        assert!(reference.completed);
        let (actions, banks) = hosted_actions(seed, kinds);
        assert_eq!(actions.len(), reference.steps.len());
        for (step, (pair, record)) in actions.iter().zip(&reference.steps).enumerate() {
            for (seat, (action, recorded)) in pair.iter().zip(&record.seats).enumerate() {
                assert_eq!(
                    Some(action),
                    recorded.action.as_ref(),
                    "step {step} seat {seat}"
                );
            }
        }
        assert_eq!(banks, reference.final_banks);
    }
}

#[test]
fn hosted_seat_rejects_repeats_and_reports_its_identity() {
    let config = Config::default();
    let mut engine = kaggriculture_engine::Game::new(config.clone(), 5, 2).unwrap();
    for kind in OpponentKind::ALL {
        let mut seat = HostedSeat::new(kind, 1, &config, engine.snapshot()).unwrap();
        assert_eq!((seat.kind(), seat.seat()), (kind, 1));
        seat.action(engine.snapshot()).unwrap();
        // The same pre-step snapshot again is a repeated turn.
        assert!(seat.action(engine.snapshot()).unwrap_err().contains("step"));
    }
    engine
        .step_with_market_metrics(&[serde_json::json!({}), serde_json::json!({})])
        .unwrap();
    // A fresh controller needs a step-zero snapshot.
    assert!(
        HostedSeat::new(OpponentKind::Starter, 0, &config, engine.snapshot())
            .unwrap_err()
            .contains("step zero")
    );
    assert!(
        HostedSeat::new(OpponentKind::Starter, 2, &config, engine.snapshot())
            .unwrap_err()
            .contains("invalid seat")
    );
}
