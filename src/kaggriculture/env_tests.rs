use super::env::SeedStream;
use super::reward::{RewardConfig, RewardMode};

fn reward_config() -> RewardConfig {
    RewardConfig {
        reward_mode: RewardMode::WinLoss,
        econ_shaping: 0.02,
        econ_starvation_weight: 4.0,
        econ_drought_weight: 1.0,
        econ_cap: 0.25,
        econ_ineffective_weight: 0.001,
        econ_ineffective_cap: 0.10,
    }
}

#[test]
fn seed_construction_reset_stride_and_overflow() {
    let stream = SeedStream::new(11, 2).unwrap();
    let (seeds, next) = stream.reserve(&[true, true]).unwrap();
    assert_eq!(seeds, vec![Some(11), Some(13)]);
    assert_eq!(stream.next, 11);
    assert_eq!(next.next, 15);
    let (seeds, next) = next.reserve(&[true, true]).unwrap();
    assert_eq!(seeds, vec![Some(15), Some(17)]);
    assert_eq!(next.next, 19);
    let stream = SeedStream::new(i64::MAX - 1, 1).unwrap();
    assert!(stream.reserve(&[true, true]).is_err());
    assert_eq!(stream.next, i64::MAX - 1);
    assert_eq!(
        stream.reserve(&[false, true]).unwrap().0,
        vec![None, Some(i64::MAX - 1)]
    );
    assert!(SeedStream::new(-1, 1).is_err());
    assert!(SeedStream::new(0, 0).is_err());
}

#[test]
fn reward_uses_only_own_counters_and_raw_banks() {
    let cfg = reward_config();
    cfg.validate().unwrap();
    let before = [[0; 32]; 2];
    let mut after = before;
    after[0][0] = 1;
    assert_eq!(
        cfg.transition(&before, &after, [10., 5.], false).unwrap(),
        [-0.08f32, 0.08f32]
    );
    let expected = cfg.transition(&before, &after, [10., 5.], true).unwrap();
    after[0][3..].fill(i64::MAX);
    after[1][3..].fill(i64::MAX);
    assert_eq!(
        cfg.transition(&before, &after, [10., 5.], true).unwrap(),
        expected
    );
    assert_eq!(expected[0], (f64::from(-0.08f32) + 0.65) as f32);
    assert_eq!(
        cfg.transition(&before, &before, [5., 5.], true).unwrap(),
        [0.; 2]
    );
}

#[test]
fn reward_admission_predicate_cases() {
    for (w, s, d, cap, iw, ic, accept) in [
        (0.2, 4., 1., 0.25, 0., 0., true),
        (0.2, 0., 1., 0.25, 0., 0., true),
        (0.2, 0., 0., 0.25, 0., 0., false),
        (0.2, 4., 1., 0., 0., 0., false),
        (1e-300, 1e-300, 1e-300, 0.25, 0., 0., false),
        (1e-300, 1e-300, 1., 0.25, 0., 0., true),
        (0., 0., 0., 0., 0., 0., true),
        (0., 4., 1., 0.25, 0., 0., true),
        (0., 0., 0., 0., 0.001, 0., false),
        (0., 0., 0., 0., 0.001, 0.1, true),
    ] {
        let c = RewardConfig {
            reward_mode: RewardMode::WinLoss,
            econ_shaping: w,
            econ_starvation_weight: s,
            econ_drought_weight: d,
            econ_cap: cap,
            econ_ineffective_weight: iw,
            econ_ineffective_cap: ic,
        };
        assert_eq!(c.validate().is_ok(), accept, "{c:?}");
    }
}

#[test]
fn reward_rejects_invalid_coefficients_and_handles_huge_counts() {
    let cfg = reward_config();
    for bad in [f64::NAN, f64::INFINITY, -1.] {
        let mut c = cfg.clone();
        c.econ_shaping = bad;
        assert!(c.validate().is_err());
    }
    let mut c = cfg.clone();
    c.econ_cap = 0.9;
    assert!(c.validate().is_err());
    c = cfg.clone();
    c.econ_shaping = f64::MAX;
    c.econ_starvation_weight = f64::MAX;
    c.validate().unwrap();
    assert_eq!(c.penalty(&[i64::MAX; 32]), 0.35);
    c.econ_shaping = 0.;
    c.econ_ineffective_weight = 0.;
    assert_eq!(c.penalty(&[i64::MAX; 32]), 0.);
}

fn half_ulp32(value: f64) -> f64 {
    let rounded = value as f32;
    ((f64::from(rounded.next_up()) - f64::from(rounded))
        .abs()
        .max((f64::from(rounded) - f64::from(rounded.next_down())).abs()))
        / 2.
}
fn half_ulp64(value: f64) -> f64 {
    (value.next_up() - value)
        .abs()
        .max((value - value.next_down()).abs())
        / 2.
}
#[test]
fn reward_episode_telescopes_with_explicit_output_rounding_budget() {
    let cfg = reward_config();
    for banks in [[5., 10.], [10., 5.], [5., 5.]] {
        let mut before = [[0; 32]; 2];
        let mut sum64 = [0.0_f64; 2];
        let mut actual_sum = [0.0_f64; 2];
        let mut expected_f32_sum = [0.0_f64; 2];
        let mut arithmetic_budget = [0.0_f64; 2];
        let mut output_budget = [0.0_f64; 2];
        for step in 0..719 {
            let mut after = before;
            after[0][0] += i64::from(step % 91 == 0);
            after[1][1] += i64::from(step % 67 == 0);
            after[0][2] += 1;
            let delta: [f64; 2] =
                std::array::from_fn(|s| cfg.penalty(&after[s]) - cfg.penalty(&before[s]));
            let rewards = cfg.transition(&before, &after, banks, step == 718).unwrap();
            for s in 0..2 {
                let econ = delta[1 - s] - delta[s];
                let sign = if banks[s] > banks[1 - s] {
                    1.
                } else if banks[s] < banks[1 - s] {
                    -1.
                } else {
                    0.
                };
                let terminal = if step == 718 {
                    cfg.terminal_scale() * sign
                } else {
                    0.
                };
                let full64 = econ + terminal;
                sum64[s] += full64;
                arithmetic_budget[s] += half_ulp64(delta[0])
                    + half_ulp64(delta[1])
                    + half_ulp64(econ)
                    + half_ulp64(full64)
                    + half_ulp64(sum64[s]);
                let expected_econ = econ as f32;
                let terminal_input = f64::from(expected_econ) + terminal;
                let expected_reward = if step == 718 {
                    terminal_input as f32
                } else {
                    expected_econ
                };
                expected_f32_sum[s] += f64::from(expected_reward);
                // Independent budgets use expected arithmetic only. Wrong native
                // output cannot enlarge its own tolerance (zero-output mutation).
                output_budget[s] += half_ulp32(econ) + half_ulp64(expected_f32_sum[s]);
                if step == 718 {
                    output_budget[s] += half_ulp64(terminal_input) + half_ulp32(terminal_input);
                }
                actual_sum[s] += f64::from(rewards[s]);
            }
            before = after;
        }
        for s in 0..2 {
            let sign = if banks[s] > banks[1 - s] {
                1.
            } else if banks[s] < banks[1 - s] {
                -1.
            } else {
                0.
            };
            let penalty_diff = cfg.penalty(&before[1 - s]) - cfg.penalty(&before[s]);
            let endpoint = penalty_diff + cfg.terminal_scale() * sign;
            let budget64 = arithmetic_budget[s] + half_ulp64(penalty_diff) + half_ulp64(endpoint);
            assert!((sum64[s] - endpoint).abs() <= budget64);
            assert!(endpoint.abs() <= 1.);
            assert!((actual_sum[s] - endpoint).abs() <= output_budget[s] + budget64);
        }
    }
}

use super::env::{EnvError, FaultPoint, NativeEnv, TransitionCache};
use super::{grammar, ObsRowMut, ObsStaging, ObservationGame};
use kaggriculture_engine::{Config, TraceHeader};
use serde_json::{json, Value};

struct Output {
    obs: ObsStaging,
    transition: TransitionCache,
}
impl Output {
    fn new(n: usize) -> Self {
        Self {
            obs: ObsStaging::new(n).unwrap(),
            transition: TransitionCache::test_zeros(n),
        }
    }
    fn observe(&mut self, env: &NativeEnv) {
        env.publish_observe(
            env.prepare_observe().unwrap(),
            &mut self.obs.buffers_mut(),
            self.transition.buffers_mut(),
        )
        .unwrap();
    }
    fn step(
        &mut self,
        env: &mut NativeEnv,
        tokens: &[i64],
        lengths: &[i64],
    ) -> Result<(), EnvError> {
        let pending = env.prepare_step(tokens, lengths)?;
        env.commit(
            pending,
            &mut self.obs.buffers_mut(),
            self.transition.buffers_mut(),
        )
    }
    fn reset(
        &mut self,
        env: &mut NativeEnv,
        mask: &[bool],
        truncate: bool,
    ) -> Result<(), EnvError> {
        let pending = env.prepare_reset(mask, truncate)?;
        env.commit(
            pending,
            &mut self.obs.buffers_mut(),
            self.transition.buffers_mut(),
        )
    }
    fn bytes(&mut self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for rows in self.obs.buffers_mut().envs_mut() {
            for row in rows.seats {
                bytes.extend(row_bytes(row));
            }
        }
        for x in &self.transition.rewards {
            bytes.extend(x.to_ne_bytes());
        }
        for x in &self.transition.dones {
            bytes.push(u8::from(*x));
        }
        for values in [
            &self.transition.transition_banks_before,
            &self.transition.transition_banks_after,
        ] {
            for x in values {
                bytes.extend(x.to_ne_bytes());
            }
        }
        for values in [
            &self.transition.transition_econ_before,
            &self.transition.transition_econ_after,
        ] {
            for x in values {
                bytes.extend(x.to_ne_bytes());
            }
        }
        bytes
    }
}
fn row_bytes(row: ObsRowMut<'_>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for x in row.tile_kind.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.tile_crop.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.tile_animal.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.tile_cell.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.tile_role.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.tiles_int.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.tiles_float.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.actor_slot.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.actor_cell.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.actor_role.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.actor_mask.iter() {
        bytes.extend([u8::from(*x)]);
    }
    for x in row.actor_inventory.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.actor_inventory_rank.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.actors_float.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.player_features.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.storage_counts.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.storage_rank.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.banks.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.shop_type.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.shop_slot.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.shop_mask.iter() {
        bytes.extend([u8::from(*x)]);
    }
    for x in row.market_product.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.market_float.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.market_int.iter().flatten() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.global_features.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    for x in row.globals_int.iter() {
        bytes.extend(x.to_ne_bytes());
    }
    bytes.extend([u8::from(*row.still_playing)]);
    bytes.extend(row.order_limits.to_ne_bytes());
    for x in row.can_act.iter() {
        bytes.extend([u8::from(*x)]);
    }
    bytes
}

fn actions(env: &NativeEnv, farmer: Value, market: Value) -> (Vec<i64>, Vec<i64>) {
    let mut tokens = vec![0; env.n_envs() * 2 * grammar::TOKENS_PER_SEAT];
    let mut lengths = vec![];
    for e in 0..env.n_envs() {
        let snapshot: Value = serde_json::from_str(&env.state_snapshot(e).unwrap()).unwrap();
        for seat in 0..2 {
            let actors = snapshot["public"]["farms"][seat]["hands"]
                .as_array()
                .unwrap()
                .len()
                + 1;
            let action =
                json!({"farmer":farmer,"hands":vec![json!(["PASS"]);actors-1],"market":market});
            let row = (e * 2 + seat) * grammar::TOKENS_PER_SEAT;
            lengths.push(
                grammar::encode(
                    &grammar::plan(actors as i64, 10, 241).unwrap(),
                    &action,
                    &mut tokens[row..row + grammar::TOKENS_PER_SEAT],
                )
                .unwrap(),
            );
        }
    }
    (tokens, lengths)
}
fn fixture(seed: i64, episode: i64, terminal: bool) -> (NativeEnv, Output) {
    let mut env = NativeEnv::new(
        2,
        seed,
        1,
        serde_json::from_value(json!({"episodeSteps":episode})).unwrap(),
        reward_config(),
        1,
        241,
    )
    .unwrap();
    let mut out = Output::new(2);
    out.observe(&env);
    if terminal {
        let (t, l) = actions(&env, json!(["PASS"]), json!([]));
        out.step(&mut env, &t, &l).unwrap();
    }
    (env, out)
}
#[derive(PartialEq)]
struct Snapshot {
    bytes: Vec<u8>,
    fresh: Vec<u8>,
    games: Vec<String>,
    seeds: (i64, Vec<i64>),
    terminals: Vec<Option<super::env::TerminalRecord>>,
}
fn capture(env: &NativeEnv, out: &mut Output) -> Snapshot {
    let mut fresh = Output::new(env.n_envs());
    fresh.observe(env);
    Snapshot {
        bytes: out.bytes(),
        fresh: fresh.bytes(),
        games: (0..env.n_envs())
            .map(|i| env.state_snapshot(i).unwrap())
            .collect(),
        seeds: env.seed_state(),
        terminals: (0..env.n_envs())
            .map(|i| env.terminal_metrics(i).unwrap().cloned())
            .collect(),
    }
}
#[test]
fn batch_failure_preserves_every_published_byte() {
    for fault in [
        None,
        Some(FaultPoint::StepResult),
        Some(FaultPoint::StepPanic),
        Some(FaultPoint::AutoReset),
        Some(FaultPoint::StepPrepare),
    ] {
        let expected = {
            let (mut env, mut out) = fixture(1000, 2, true);
            let (t, l) = actions(&env, json!(["PASS"]), json!([]));
            out.step(&mut env, &t, &l).unwrap();
            capture(&env, &mut out)
        }; // Drop control before creating the two live subject lanes.
        let (mut env, mut out) = fixture(1000, 2, true);
        let before = capture(&env, &mut out);
        let (mut t, l) = actions(&env, json!(["PASS"]), json!([]));
        if let Some(point) = fault {
            env.set_fault(point, 1);
        } else {
            t[2 * grammar::TOKENS_PER_SEAT + 1] = 19;
        }
        let error = out
            .step(&mut env, &t, &l)
            .expect_err("missing rollback failure");
        if fault.is_some() {
            assert_eq!(env.fault_hits(), 1, "injection not reached: {error}");
        }
        assert_eq!(capture(&env, &mut out), before, "{fault:?}: {error}");
        env.clear_fault();
        let (t, l) = actions(&env, json!(["PASS"]), json!([]));
        out.step(&mut env, &t, &l).unwrap();
        assert_eq!(capture(&env, &mut out), expected, "retry {fault:?}");
    }
}

#[test]
fn reset_and_truncate_failures_preserve_every_published_byte() {
    for (truncate, mask, point) in [
        (false, [true, true], FaultPoint::ResetConstruct),
        (false, [true, true], FaultPoint::ResetPrepare),
        (true, [true, true], FaultPoint::ResetConstruct),
        (true, [true, true], FaultPoint::ResetPrepare),
        (true, [false, true], FaultPoint::ResetPrepare),
        (false, [true, true], FaultPoint::ResetPanic),
    ] {
        let expected = {
            let (mut env, mut out) = fixture(1000, 2, true);
            out.reset(&mut env, &mask, truncate).unwrap();
            capture(&env, &mut out)
        };
        let (mut env, mut out) = fixture(1000, 2, true);
        assert!((0..2).all(|i| env.terminal_metrics(i).unwrap().is_some()));
        let before = capture(&env, &mut out);
        env.set_fault(point, 1);
        let error = out
            .reset(&mut env, &mask, truncate)
            .expect_err("missing reset/truncate failure");
        assert_eq!(env.fault_hits(), 1, "injection not reached: {error}");
        assert_eq!(capture(&env, &mut out), before, "{point:?}, mask={mask:?}");
        env.clear_fault();
        out.reset(&mut env, &mask, truncate).unwrap();
        assert_eq!(capture(&env, &mut out), expected);
    }
    let (mut env, mut out) = fixture(i64::MAX - 5, 2, true);
    let before = capture(&env, &mut out);
    assert_eq!(env.seed_state().0, i64::MAX - 1);
    for truncate in [false, true] {
        assert!(matches!(
            out.reset(&mut env, &[true, true], truncate),
            Err(EnvError::Overflow(_))
        ));
        assert_eq!(capture(&env, &mut out), before);
    }
    out.reset(&mut env, &[false, true], true).unwrap();
    assert_eq!(
        env.seed_state(),
        (i64::MAX, vec![i64::MAX - 3, i64::MAX - 1])
    );
    assert_eq!(env.state_snapshot(0).unwrap(), before.games[0]);
    assert_eq!(
        env.terminal_metrics(0).unwrap(),
        before.terminals[0].as_ref()
    );
    assert!(env.terminal_metrics(1).unwrap().is_none());
}

fn header(config: Config) -> TraceHeader {
    let initial = kaggriculture_engine::Game::new(config.clone(), 42, 2)
        .unwrap()
        .snapshot();
    TraceHeader {
        format: kaggriculture_engine::TRACE_FORMAT.into(),
        seed: 42.into(),
        configuration: config,
        shop_schedule: vec![],
        rng_schedule: vec![],
        initial: kaggriculture_engine::InitialState {
            public: initial.public,
            privates: initial.privates,
        },
        terminal_banks: vec![],
        transitions: 0,
    }
}
#[test]
fn release_dependency_overflow_is_caught() {
    let config: Config = serde_json::from_value(json!({"episodeSteps":2})).unwrap();
    let mut header = header(config.clone());
    header.initial.privates[0]
        .shed
        .insert("WHEAT".into(), i64::MAX);
    header.initial.privates[0]
        .shed
        .insert("FERTILIZER".into(), 1);
    let mut env = NativeEnv::new(1, 42, 1, config, reward_config(), 1, 241).unwrap();
    env.replace_game(0, ObservationGame::from_header(&header).unwrap());
    let mut out = Output::new(1);
    out.observe(&env);
    let before = capture(&env, &mut out);
    let (t, l) = actions(&env, json!(["PASS"]), json!([]));
    assert!(
        matches!(out.step(&mut env, &t, &l), Err(EnvError::Panic(_))),
        "engine overflow must panic inside the caught candidate step"
    );
    assert_eq!(capture(&env, &mut out), before);
}
#[test]
fn executed_unbounded_hire_cast_rejects_before_publication() {
    let config: Config = serde_json::from_value(json!({"farmHandCostMult":i64::MAX})).unwrap();
    let mut header = header(config.clone());
    header.initial.public.farms[0].hires_today = 2;
    header.initial.public.farms[0].money = 3e19;
    let mut env = NativeEnv::new(1, 42, 1, config, reward_config(), 1, 241).unwrap();
    env.replace_game(0, ObservationGame::from_header(&header).unwrap());
    let mut out = Output::new(1);
    out.observe(&env);
    let before = capture(&env, &mut out);
    let (t, l) = actions(&env, json!(["PASS"]), json!([["HIRE"]]));
    let error = out
        .step(&mut env, &t, &l)
        .expect_err("unbounded hire must reject");
    assert!(error.to_string().contains("uint64 cash"), "{error}");
    assert_eq!(capture(&env, &mut out), before);
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::hash::{Hash, Hasher};
        let digest = |bytes: &Vec<u8>| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        f.debug_struct("Snapshot")
            .field("bytes_hash", &digest(&self.bytes))
            .field("fresh_hash", &digest(&self.fresh))
            .field("seeds", &self.seeds)
            .field("terminals", &self.terminals)
            .finish()
    }
}

#[test]
fn default_horizon_terminates_at_719() {
    let mut env = NativeEnv::new(1, 17000, 1, Config::default(), reward_config(), 1, 241).unwrap();
    let mut out = Output::new(1);
    out.observe(&env);
    for transition in 1..=719 {
        let (t, l) = actions(&env, json!(["PASS"]), json!([]));
        out.step(&mut env, &t, &l).unwrap();
        assert_eq!(
            out.transition.dones,
            vec![transition == 719; 2],
            "transition {transition}"
        );
        assert_eq!(
            env.seed_state(),
            if transition == 719 {
                (17002, vec![17001])
            } else {
                (17001, vec![17000])
            }
        );
        let mut rows = out.obs.buffers_mut();
        for row in rows.envs_mut().next().unwrap().seats {
            assert_eq!(
                row.globals_int[0],
                if transition == 719 { 0 } else { transition }
            );
            assert!(*row.still_playing);
        }
        if transition < 719 {
            assert!(env.terminal_metrics(0).unwrap().is_none());
        }
    }
    let record = env.terminal_metrics(0).unwrap().unwrap();
    assert_eq!(record.episode_steps, 719);
    assert_eq!(
        record.banks.as_slice(),
        out.transition.transition_banks_after
    );
    assert_eq!(record.econ.concat(), out.transition.transition_econ_after);
    assert_eq!(
        record.winner,
        if record.banks[0] > record.banks[1] {
            0
        } else if record.banks[0] < record.banks[1] {
            1
        } else {
            -1
        }
    );
    // The completed game's full snapshot is captured before auto-reset; the
    // live slot already holds the replacement game.
    let finished: serde_json::Value = serde_json::from_str(&record.snapshot).unwrap();
    assert_eq!(finished["done"], json!(true));
    assert_eq!(finished["public"]["step"], json!(719));
    assert_eq!(finished["statuses"], json!(["DONE", "DONE"]));
    for seat in 0..2 {
        assert_eq!(
            finished["public"]["farms"][seat]["money"].as_f64(),
            Some(record.banks[seat])
        );
    }
    let live: serde_json::Value = serde_json::from_str(&env.state_snapshot(0).unwrap()).unwrap();
    assert_eq!(live["public"]["step"], json!(0));
    assert_eq!(live["done"], json!(false));
    let mut fresh = Output::new(1);
    fresh.observe(&env);
    assert_eq!(fresh.transition, out.transition);
}

fn economic_step(env: &mut NativeEnv, out: &mut Output) {
    let (mut t, mut l) = actions(
        env,
        json!(["HARVEST"]),
        json!([["BUY_PRODUCT", "WHEAT", 1]]),
    );
    for e in 0..env.n_envs() {
        let start = (2 * e + 1) * grammar::TOKENS_PER_SEAT;
        l[2 * e + 1] = grammar::encode(
            &grammar::plan(1, 10, 241).unwrap(),
            &json!({"farmer":["PASS"],"hands":[],"market":[]}),
            &mut t[start..start + grammar::TOKENS_PER_SEAT],
        )
        .unwrap();
    }
    out.step(env, &t, &l).unwrap();
    assert!(out.transition.rewards.iter().any(|x| *x != 0.));
    assert!(out
        .transition
        .transition_banks_after
        .iter()
        .any(|x| *x != 0.));
    assert!(out.transition.transition_econ_after.iter().any(|x| *x != 0));
}
#[test]
fn truncate_preserves_transition_and_no_winner() {
    let (mut env, mut out) = fixture(42, 720, false);
    economic_step(&mut env, &mut out);
    let transition = out.transition.clone();
    let bootstrap: Vec<_> = out
        .obs
        .buffers_mut()
        .envs_mut()
        .map(|rows| (rows.seats[0].globals_int[0], *rows.seats[0].banks))
        .collect();
    assert_eq!(bootstrap[1].0, 1);
    out.reset(&mut env, &[false, true], true).unwrap();
    assert_eq!(out.transition, transition);
    assert!((0..2).all(|i| env.terminal_metrics(i).unwrap().is_none()));
    let mut rows = out.obs.buffers_mut();
    let reset = rows.envs_mut().nth(1).unwrap();
    assert_eq!(reset.seats[0].globals_int[0], 0);
    assert_ne!(*reset.seats[0].banks, bootstrap[1].1);
    // The saved pre-reset rows are the distinguishable native bootstrap fixture;
    // trainer copying/evaluation order is owned by Tasks 3.2 and 1.5.
    assert_eq!(bootstrap[1].0, 1);
}
fn sentinel(row: ObsRowMut<'_>) {
    row.tile_kind.fill(i64::from_ne_bytes([0x5A; 8]));
    row.tile_crop.fill(i64::from_ne_bytes([0x5A; 8]));
    row.tile_animal.fill(i64::from_ne_bytes([0x5A; 8]));
    row.tile_cell.fill(i64::from_ne_bytes([0x5A; 8]));
    row.tile_role.fill(i64::from_ne_bytes([0x5A; 8]));
    row.actor_slot.fill(i64::from_ne_bytes([0x5A; 8]));
    row.actor_cell.fill(i64::from_ne_bytes([0x5A; 8]));
    row.actor_role.fill(i64::from_ne_bytes([0x5A; 8]));
    row.storage_counts.fill(i64::from_ne_bytes([0x5A; 8]));
    row.storage_rank.fill(i64::from_ne_bytes([0x5A; 8]));
    row.shop_type.fill(i64::from_ne_bytes([0x5A; 8]));
    row.shop_slot.fill(i64::from_ne_bytes([0x5A; 8]));
    row.market_product.fill(i64::from_ne_bytes([0x5A; 8]));
    row.globals_int.fill(i64::from_ne_bytes([0x5A; 8]));
    for values in row.tiles_int.iter_mut() {
        values.fill(i64::from_ne_bytes([0x5A; 8]));
    }
    for values in row.actor_inventory.iter_mut() {
        values.fill(i64::from_ne_bytes([0x5A; 8]));
    }
    for values in row.actor_inventory_rank.iter_mut() {
        values.fill(i64::from_ne_bytes([0x5A; 8]));
    }
    for values in row.market_int.iter_mut() {
        values.fill(i64::from_ne_bytes([0x5A; 8]));
    }
    row.global_features.fill(f32::from_ne_bytes([0x5A; 4]));
    for values in row.tiles_float.iter_mut() {
        values.fill(f32::from_ne_bytes([0x5A; 4]));
    }
    for values in row.actors_float.iter_mut() {
        values.fill(f32::from_ne_bytes([0x5A; 4]));
    }
    for values in row.player_features.iter_mut() {
        values.fill(f32::from_ne_bytes([0x5A; 4]));
    }
    for values in row.market_float.iter_mut() {
        values.fill(f32::from_ne_bytes([0x5A; 4]));
    }
    row.banks.fill(f64::from_ne_bytes([0x5A; 8]));
    row.actor_mask.fill(true);
    row.shop_mask.fill(true);
    row.can_act.fill(true);
    *row.still_playing = true;
    *row.order_limits = i64::from_ne_bytes([0x5A; 8]);
}
fn obs_bytes(out: &mut Output) -> Vec<Vec<u8>> {
    out.obs
        .buffers_mut()
        .envs_mut()
        .map(|rows| rows.seats.into_iter().flat_map(row_bytes).collect())
        .collect()
}
#[test]
fn truncate_commits_only_selected_rows() {
    for terminal in [false, true] {
        for mask in [[false, true], [true, false], [false, false]] {
            let (mut env, mut out) = fixture(1000, if terminal { 2 } else { 720 }, false);
            economic_step(&mut env, &mut out);
            assert_eq!(out.transition.dones, vec![terminal; 4]);
            for (i, rows) in out.obs.buffers_mut().envs_mut().enumerate() {
                if !mask[i] {
                    for row in rows.seats {
                        sentinel(row);
                    }
                }
            }
            let before = obs_bytes(&mut out);
            let transition = out.transition.clone();
            let state = capture(&env, &mut out);
            out.reset(&mut env, &mask, true).unwrap();
            let after = obs_bytes(&mut out);
            let mut fresh = Output::new(2);
            fresh.observe(&env);
            let expected = obs_bytes(&mut fresh);
            assert_eq!(out.transition, transition);
            assert_eq!(
                fresh.transition, transition,
                "truncate must retain cached transitions too"
            );
            assert_eq!(
                env.seed_state().0,
                state.seeds.0 + mask.iter().filter(|x| **x).count() as i64
            );
            for i in 0..2 {
                if mask[i] {
                    assert_eq!(after[i], expected[i], "selected row and all padding");
                    assert!(env.terminal_metrics(i).unwrap().is_none());
                } else {
                    assert!(
                        after[i] == before[i],
                        "unselected observation bytes changed: env={i}, terminal={terminal}"
                    );
                    assert_eq!(env.state_snapshot(i).unwrap(), state.games[i]);
                    assert_eq!(
                        env.terminal_metrics(i).unwrap(),
                        state.terminals[i].as_ref()
                    );
                    assert_eq!(env.seed_state().1[i], state.seeds.1[i]);
                }
            }
            if mask == [false, false] {
                assert_eq!(capture(&env, &mut out), state);
            }
        }
    }
}
