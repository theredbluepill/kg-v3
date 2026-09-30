//! Fixed-opponent collection (`env.opponent_mix`) inside the native step.
//!
//! Each check replays the same games on an independent reference: a bare
//! kernel `Game` stepped with the learner's decoded program and a fresh
//! `HostedSeat` for the scripted seat. Equal state snapshots after every
//! transition show that the scripted seat's action came from the bot and the
//! learner's action was executed unchanged.
use super::env::{learner_seat, EnvError, FaultPoint, NativeEnv, OpponentMix};
use super::env_tests::{reward_config, Output};
use super::grammar;
use kaggriculture_engine::Config;
use kaggriculture_opponents::{HostedSeat, OpponentKind};
use serde_json::{json, Value};

fn mixed_env(n: usize, seed: i64, config: Config, mix: Option<OpponentMix>) -> NativeEnv {
    NativeEnv::new(n, seed, 1, config, reward_config(), 1, 241, mix).unwrap()
}

fn short_config(episode_steps: i64) -> Config {
    serde_json::from_value(json!({ "episodeSteps": episode_steps })).unwrap()
}

/// Learner rows get an all-PASS program; scripted rows the absent program.
fn learner_program(env: &NativeEnv) -> (Vec<i64>, Vec<i64>, Vec<[Option<Value>; 2]>) {
    let mask = env.learner_mask();
    let mut tokens = vec![0; env.n_envs() * 2 * grammar::TOKENS_PER_SEAT];
    let mut lengths = vec![0; env.n_envs() * 2];
    let mut decoded = Vec::new();
    for e in 0..env.n_envs() {
        let snapshot: Value = serde_json::from_str(&env.state_snapshot(e).unwrap()).unwrap();
        let mut pair = [None, None];
        for seat in 0..2 {
            if !mask[e * 2 + seat] {
                continue;
            }
            let actors = snapshot["public"]["farms"][seat]["hands"]
                .as_array()
                .unwrap()
                .len()
                + 1;
            let action = json!({
                "farmer": ["PASS"],
                "hands": vec![json!(["PASS"]); actors - 1],
                "market": [],
            });
            let plan = grammar::plan(actors as i64, 10, 241).unwrap();
            let row = (e * 2 + seat) * grammar::TOKENS_PER_SEAT;
            let slice = &mut tokens[row..row + grammar::TOKENS_PER_SEAT];
            let length = grammar::encode(&plan, &action, slice).unwrap();
            lengths[e * 2 + seat] = length;
            pair[seat] = Some(grammar::decode(&plan, slice, length).unwrap());
        }
        decoded.push(pair);
    }
    (tokens, lengths, decoded)
}

/// One env's independent replay: a bare kernel plus a fresh hosted controller
/// for each new game.
struct Reference {
    engine: kaggriculture_engine::Game,
    bot: Option<HostedSeat>,
    bot_actions: Vec<Value>,
}
impl Reference {
    fn new(config: &Config, seed: i64, kind: Option<OpponentKind>, bot_seat: usize) -> Self {
        let engine =
            kaggriculture_engine::Game::new_with_seed_decimal(config.clone(), &seed.to_string(), 2)
                .unwrap();
        let bot =
            kind.map(|kind| HostedSeat::new(kind, bot_seat, config, engine.snapshot()).unwrap());
        Self {
            engine,
            bot,
            bot_actions: Vec::new(),
        }
    }
    fn step(&mut self, learner: &[Option<Value>; 2]) {
        let mut joint = Vec::new();
        for (seat, action) in learner.iter().enumerate() {
            joint.push(match action {
                Some(action) => action.clone(),
                None => {
                    let bot = self.bot.as_mut().unwrap();
                    assert_eq!(bot.seat(), seat);
                    let action = bot.action(self.engine.snapshot()).unwrap();
                    self.bot_actions.push(action.clone());
                    action
                },
            });
        }
        self.engine.step_with_market_metrics(&joint).unwrap();
    }
    fn state(&self) -> String {
        serde_json::to_string(&self.engine.snapshot()).unwrap()
    }
}

#[test]
fn scripted_seat_plays_the_bot_and_the_learner_its_own_program() {
    for kind in OpponentKind::ALL {
        let config = Config::default();
        let mut env = mixed_env(2, 4100, config.clone(), Some(OpponentMix { kind, envs: 2 }));
        // Env 0 learns seat 0 and env 1 seat 1 in their first games.
        assert_eq!(env.learner_mask(), vec![true, false, false, true]);
        let mut out = Output::new(2);
        out.observe(&env);
        let mut references: Vec<Reference> = (0..2)
            .map(|e| Reference::new(&config, 4100 + e as i64, Some(kind), 1 - learner_seat(e, 0)))
            .collect();
        for _ in 0..40 {
            let (tokens, lengths, decoded) = learner_program(&env);
            out.step(&mut env, &tokens, &lengths).unwrap();
            for (e, reference) in references.iter_mut().enumerate() {
                reference.step(&decoded[e]);
                assert_eq!(
                    env.state_snapshot(e).unwrap(),
                    reference.state(),
                    "{kind:?} env {e}"
                );
            }
        }
        // Non-vacuous: the bot did something other than the learner's PASS
        // program, so a PASS substitute for its seat would diverge.
        for reference in &references {
            assert!(
                reference
                    .bot_actions
                    .iter()
                    .any(|action| action["farmer"] != json!(["PASS"])
                        || action["market"] != json!([])),
                "{kind:?}: the bot only passed in the checked window"
            );
        }
    }
}

#[test]
fn learner_seat_is_never_played_by_the_bot() {
    // Mutation guard for "the learner's action is never overridden": a bot
    // that also played the learner seat would reproduce the two-bot game.
    let config = Config::default();
    let kind = OpponentKind::Starter;
    let mut env = mixed_env(1, 4200, config.clone(), Some(OpponentMix { kind, envs: 1 }));
    let mut out = Output::new(1);
    out.observe(&env);
    let mut both_bots = Reference::new(&config, 4200, Some(kind), 1);
    let mut learner_bot = HostedSeat::new(kind, 0, &config, both_bots.engine.snapshot()).unwrap();
    let mut diverged = false;
    for _ in 0..40 {
        let (tokens, lengths, _decoded) = learner_program(&env);
        out.step(&mut env, &tokens, &lengths).unwrap();
        let seat0 = learner_bot.action(both_bots.engine.snapshot()).unwrap();
        both_bots.step(&[Some(seat0), None]);
        diverged |= env.state_snapshot(0).unwrap() != both_bots.state();
    }
    assert!(
        diverged,
        "the learner seat followed the bot instead of its tokens"
    );
}

#[test]
fn seats_alternate_by_env_and_episode_and_every_reset_restarts_the_bot() {
    let config = short_config(3);
    let kind = OpponentKind::Starter;
    let mut env = mixed_env(3, 4300, config.clone(), Some(OpponentMix { kind, envs: 2 }));
    let mut out = Output::new(3);
    out.observe(&env);
    let mut episodes = [0u64; 3];
    let expected_mask = |episodes: &[u64; 3]| -> Vec<bool> {
        (0..3)
            .flat_map(|e| {
                if e == 2 {
                    [true, true]
                } else {
                    let mut mask = [false, false];
                    mask[learner_seat(e, episodes[e])] = true;
                    mask
                }
            })
            .collect()
    };
    assert_eq!(
        env.learner_mask(),
        vec![true, false, false, true, true, true]
    );
    let new_references = |env: &NativeEnv, episodes: &[u64; 3]| -> Vec<Reference> {
        let seeds = env.seed_state().1;
        (0..3)
            .map(|e| {
                let bot = (e < 2).then_some(kind);
                Reference::new(&config, seeds[e], bot, 1 - learner_seat(e, episodes[e]))
            })
            .collect()
    };
    let mut references = new_references(&env, &episodes);
    // Three games per env: the engine completes at step episodeSteps - 1 = 2.
    for game in 0..3 {
        for transition in 0..2 {
            let (tokens, lengths, decoded) = learner_program(&env);
            let before = env.learner_mask();
            let pending = env.prepare_step(&tokens, &lengths).unwrap();
            let terminal: Vec<_> = pending.metrics().to_vec();
            out.step_pending(&mut env, pending);
            for (e, reference) in references.iter_mut().enumerate() {
                reference.step(&decoded[e]);
            }
            if transition == 1 {
                // Completed games report the learned seat of the game that ended.
                assert_eq!(
                    terminal.iter().map(|m| m.3).collect::<Vec<_>>(),
                    vec![
                        Some(learner_seat(0, episodes[0])),
                        Some(learner_seat(1, episodes[1])),
                        None
                    ],
                    "game {game}"
                );
                for (e, reference) in references.iter().enumerate() {
                    let record = env.terminal_metrics(e).unwrap().unwrap();
                    let final_state: Value = serde_json::from_str(&reference.state()).unwrap();
                    let money = |seat: usize| {
                        final_state["public"]["farms"][seat]["money"]
                            .as_f64()
                            .unwrap()
                    };
                    assert_eq!(record.banks, [money(0), money(1)], "game {game} env {e}");
                    assert_eq!(
                        record.learner_seat,
                        (e < 2).then(|| learner_seat(e, episodes[e]))
                    );
                }
                for episode in &mut episodes {
                    *episode += 1;
                }
                // Independent of `learner_seat`: each bot env's seats swap.
                let after = env.learner_mask();
                for row in 0..4 {
                    assert_eq!(after[row], !before[row], "game {game} row {row}");
                }
                assert!(after[4] && after[5]);
                // Auto-reset: every env starts a new game, and each bot env's
                // learned seat flips; a stale controller would refuse step 0.
                assert_eq!(env.learner_mask(), expected_mask(&episodes), "game {game}");
                references = new_references(&env, &episodes);
            } else {
                for (e, reference) in references.iter().enumerate() {
                    assert_eq!(env.state_snapshot(e).unwrap(), reference.state());
                }
            }
        }
    }
    // A full reset starts a new episode everywhere; a truncation only in the
    // selected envs.
    out.reset(&mut env, &[true, true, true], false).unwrap();
    for episode in &mut episodes {
        *episode += 1;
    }
    assert_eq!(env.learner_mask(), expected_mask(&episodes));
    out.reset(&mut env, &[false, true, false], true).unwrap();
    episodes[1] += 1;
    assert_eq!(env.learner_mask(), expected_mask(&episodes));
    let mut references = new_references(&env, &episodes);
    let (tokens, lengths, decoded) = learner_program(&env);
    out.step(&mut env, &tokens, &lengths).unwrap();
    for (e, reference) in references.iter_mut().enumerate() {
        reference.step(&decoded[e]);
        assert_eq!(env.state_snapshot(e).unwrap(), reference.state());
    }
}

#[test]
fn scripted_rows_must_submit_the_absent_program_and_failures_roll_back() {
    let config = Config::default();
    let mix = Some(OpponentMix {
        kind: OpponentKind::R04,
        envs: 1,
    });
    let mut env = mixed_env(2, 4400, config.clone(), mix);
    let mut out = Output::new(2);
    out.observe(&env);
    let (tokens, lengths, _) = learner_program(&env);
    let before = out.bytes();
    let states: Vec<_> = (0..2).map(|e| env.state_snapshot(e).unwrap()).collect();
    // Env 0's scripted seat is 1: any submitted program there is refused.
    let bot_row = 1;
    let mut bad_length = lengths.clone();
    bad_length[bot_row] = 2;
    let mut bad_token = tokens.clone();
    bad_token[bot_row * grammar::TOKENS_PER_SEAT + 5] = 1;
    for (t, l) in [(&tokens, &bad_length), (&bad_token, &lengths)] {
        let error = out.step(&mut env, t, l).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("env=0 seat=1 is played by the fixed opponent"),
            "{error}"
        );
    }
    // Learner rows keep the ordinary 1..=252 length admission.
    let mut empty_learner = lengths.clone();
    empty_learner[0] = 0;
    let error = out.step(&mut env, &tokens, &empty_learner).unwrap_err();
    assert!(error.to_string().contains("length outside"), "{error}");
    // A failed batch after the bot acted keeps the committed controller: the
    // retry acts at the same step instead of refusing a repeated turn.
    env.set_fault(FaultPoint::StepResult, 1);
    assert!(out.step(&mut env, &tokens, &lengths).is_err());
    assert_eq!(env.fault_hits(), 1);
    env.clear_fault();
    assert_eq!(out.bytes(), before);
    assert_eq!(
        (0..2)
            .map(|e| env.state_snapshot(e).unwrap())
            .collect::<Vec<_>>(),
        states
    );
    let mut control = mixed_env(2, 4400, config, mix);
    let mut control_out = Output::new(2);
    control_out.observe(&control);
    for _ in 0..3 {
        let (t, l, _) = learner_program(&env);
        out.step(&mut env, &t, &l).unwrap();
        control_out.step(&mut control, &t, &l).unwrap();
    }
    assert_eq!(out.bytes(), control_out.bytes());
}

#[test]
fn scripted_envs_are_deterministic_across_thread_counts() {
    let mix = Some(OpponentMix {
        kind: OpponentKind::Ecobot,
        envs: 3,
    });
    let mut outputs = Vec::new();
    for threads in [1, 2] {
        let mut env = NativeEnv::new(
            4,
            4500,
            1,
            Config::default(),
            reward_config(),
            threads,
            241,
            mix,
        )
        .unwrap();
        let mut out = Output::new(4);
        out.observe(&env);
        for _ in 0..8 {
            let (tokens, lengths, _) = learner_program(&env);
            out.step(&mut env, &tokens, &lengths).unwrap();
        }
        outputs.push((
            out.bytes(),
            (0..4)
                .map(|e| env.state_snapshot(e).unwrap())
                .collect::<Vec<_>>(),
        ));
    }
    assert!(outputs[0] == outputs[1]);
}

#[test]
fn observations_carry_no_opponent_identity() {
    // At construction and after a reset every env is at step zero, so a
    // scripted env's rows must equal the self-play env's rows byte for byte,
    // whichever bot is hosted.
    let mut baseline = None;
    for kind in [None, Some(OpponentKind::Starter), Some(OpponentKind::E776)] {
        let mix = kind.map(|kind| OpponentMix { kind, envs: 2 });
        let mut env = mixed_env(2, 4600, Config::default(), mix);
        let mut out = Output::new(2);
        out.observe(&env);
        let constructed = out.bytes();
        out.reset(&mut env, &[true, true], false).unwrap();
        let reset = out.bytes();
        match &baseline {
            None => baseline = Some((constructed, reset)),
            Some(expected) => assert!(*expected == (constructed, reset), "{kind:?}"),
        }
    }
}

#[test]
fn opponent_env_count_is_admitted() {
    for envs in [0, 3] {
        let mix = Some(OpponentMix {
            kind: OpponentKind::Starter,
            envs,
        });
        let error = NativeEnv::new(2, 1, 1, Config::default(), reward_config(), 1, 241, mix)
            .err()
            .unwrap();
        assert!(matches!(error, EnvError::Value(ref s) if s.contains("opponent envs")));
    }
}
