use kaggriculture_opponents::{Config, Game, OpponentKind, SeatController, play_match};

#[test]
fn registry_and_lifecycle_fail_explicitly() {
    assert!(
        "unknown"
            .parse::<OpponentKind>()
            .unwrap_err()
            .contains("unknown")
    );
    for kind in OpponentKind::ALL {
        assert_eq!(kind.key().parse::<OpponentKind>().unwrap(), kind);
        let mut game = Game::new(Config::default(), 17, 2).unwrap();
        let mut seat = SeatController::new(kind, 0, &game).unwrap();
        assert!(seat.reset(&game).unwrap_err().contains("new episode"));
        assert!(seat.action(&game, 1).unwrap_err().contains("seat"));
        let first = seat.action(&game, 0).unwrap();
        assert!(seat.action(&game, 0).unwrap_err().contains("step"));
        let other = Game::new(Config::default(), 17, 2).unwrap();
        assert!(seat.action(&other, 0).unwrap_err().contains("episode"));
        game.step(&[first.clone(), serde_json::json!({})]).unwrap();
        game.step(&[serde_json::json!({}), serde_json::json!({})])
            .unwrap();
        assert!(seat.action(&game, 0).unwrap_err().contains("step"));
        seat.reset(&other).unwrap();
        let mut fresh = SeatController::new(kind, 0, &other).unwrap();
        assert_eq!(
            seat.action(&other, 0).unwrap(),
            fresh.action(&other, 0).unwrap()
        );
        assert_eq!(
            first,
            SeatController::new(kind, 0, &other)
                .unwrap()
                .action(&other, 0)
                .unwrap()
        );
    }
}

#[test]
fn matches_reject_nondefault_configuration() {
    let config = Config {
        episode_steps: 3_i64.into(),
        ..Config::default()
    };
    assert!(
        play_match(config, 0, [OpponentKind::Starter; 2])
            .unwrap_err()
            .contains("default")
    );
}

fn hashes(result: &kaggriculture_opponents::MatchResult) -> Vec<String> {
    use sha2::{Digest, Sha256};
    result
        .steps
        .iter()
        .map(|row| {
            let actions = row
                .seats
                .each_ref()
                .map(|seat| seat.action.as_ref().unwrap());
            format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&actions).unwrap())
            )
        })
        .collect()
}

#[test]
fn same_seed_has_identical_action_hashes_banks_and_execution() {
    let a = play_match(
        Config::default(),
        17,
        [OpponentKind::R04, OpponentKind::E776],
    )
    .unwrap();
    let b = play_match(
        Config::default(),
        17,
        [OpponentKind::R04, OpponentKind::E776],
    )
    .unwrap();
    assert!(a.completed && b.completed);
    assert_eq!(a.steps.len(), 719);
    assert_eq!(hashes(&a), hashes(&b));
    assert_eq!(a.final_banks, b.final_banks);
    for row in &a.steps {
        assert!(
            row.seats
                .iter()
                .all(|seat| seat.engine_accepted == Some(true) && seat.controller_error.is_none())
        );
        assert!(row.engine_error.is_none());
        assert!(row.market_metrics.is_some());
    }
}

#[test]
fn changed_seed_changes_actions() {
    let a = play_match(
        Config::default(),
        17,
        [OpponentKind::R04, OpponentKind::E776],
    )
    .unwrap();
    let b = play_match(
        Config::default(),
        18,
        [OpponentKind::R04, OpponentKind::E776],
    )
    .unwrap();
    assert!(a.completed && b.completed);
    assert_ne!(hashes(&a), hashes(&b));
}

#[test]
fn independent_seats_and_mid_episode_replay() {
    for kind in OpponentKind::ALL {
        let mut game = Game::new(Config::default(), 17, 2).unwrap();
        let mut seats = [
            SeatController::new(kind, 0, &game).unwrap(),
            SeatController::new(kind, 1, &game).unwrap(),
        ];
        let mut recorded = Vec::new();
        for _ in 0..48 {
            // Calling seat zero cannot alter seat one's controller state.
            let mut independent = seats[1].clone();
            let a = seats[0].action(&game, 0).unwrap();
            let b = seats[1].action(&game, 1).unwrap();
            assert_eq!(b, independent.action(&game, 1).unwrap());
            recorded.push([a.clone(), b.clone()]);
            game.step(&[a, b]).unwrap();
        }
        let mut replay = Game::new(Config::default(), 17, 2).unwrap();
        for seat in &mut seats {
            seat.reset(&replay).unwrap();
        }
        for actions in &recorded {
            for seat in 0..2 {
                assert_eq!(seats[seat].action(&replay, seat).unwrap(), actions[seat]);
            }
            replay.step(actions).unwrap();
        }
        assert_eq!(
            serde_json::to_string(game.snapshot()).unwrap(),
            serde_json::to_string(replay.snapshot()).unwrap()
        );
        assert!(SeatController::new(kind, 0, &replay).is_err());
    }
}
