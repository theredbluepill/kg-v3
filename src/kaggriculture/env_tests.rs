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
        econ_bank_weight: 0.,
        econ_bank_scale: 100_000.,
        econ_bank_cap: 0.,
        econ_margin_weight: 0.,
        econ_margin_scale: 50_000.,
        econ_margin_cap: 0.,
    }
}
/// Owner term A at unit weight (w_b 1, S 100,000, cap_b .25), so a score reads
/// as bank / S and saturates at 25,000. Not the presets' weight: see
/// `preset_bank_config`.
fn bank_reward_config() -> RewardConfig {
    RewardConfig {
        econ_shaping: 0.2,
        econ_ineffective_weight: 0.,
        econ_bank_weight: 1.,
        econ_bank_cap: 0.25,
        ..reward_config()
    }
}
/// The bank presets' values (w_b .25, S 100,000, cap_b .25): the score is
/// `.25 * min(1, bank / 100,000)`, saturating at 100,000.
fn preset_bank_config() -> RewardConfig {
    RewardConfig {
        econ_bank_weight: 0.25,
        ..bank_reward_config()
    }
}
/// Only the bank term: relative penalties and their caps disabled.
fn bank_only_config() -> RewardConfig {
    RewardConfig {
        econ_shaping: 0.,
        econ_ineffective_weight: 0.,
        ..bank_reward_config()
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
        cfg.transition(&before, &after, [10., 5.], [10., 5.], false)
            .unwrap(),
        [-0.08f32, 0.08f32]
    );
    let expected = cfg
        .transition(&before, &after, [10., 5.], [10., 5.], true)
        .unwrap();
    after[0][3..].fill(i64::MAX);
    after[1][3..].fill(i64::MAX);
    assert_eq!(
        cfg.transition(&before, &after, [10., 5.], [10., 5.], true)
            .unwrap(),
        expected
    );
    assert_eq!(expected[0], (f64::from(-0.08f32) + 0.65) as f32);
    assert_eq!(
        cfg.transition(&before, &before, [5., 5.], [5., 5.], true)
            .unwrap(),
        [0.; 2]
    );
}

#[test]
fn reward_admission_predicate_cases() {
    // The original eleven rows run with bank shaping off; the bank rows follow
    // in the same order as the Python/native tables.
    let off = (0., 100_000., 0.);
    for ((w, s, d, cap, iw, ic), (wb, bs, bc), accept) in [
        ((0.2, 4., 1., 0.25, 0., 0.), off, true),
        ((0.2, 0., 1., 0.25, 0., 0.), off, true),
        ((0.2, 0., 0., 0.25, 0., 0.), off, false),
        ((0.2, 4., 1., 0., 0., 0.), off, false),
        ((1e-300, 1e-300, 1e-300, 0.25, 0., 0.), off, false),
        ((1e-300, 1e-300, 1., 0.25, 0., 0.), off, true),
        ((0.2, 0., 0., 0.25, 0.001, 0.1), off, false),
        ((0., 0., 0., 0., 0., 0.), off, true),
        ((0., 4., 1., 0.25, 0., 0.), off, true),
        ((0., 0., 0., 0., 0.001, 0.), off, false),
        ((0., 0., 0., 0., 0.001, 0.1), off, true),
        ((0.2, 4., 1., 0.25, 0., 0.1), (1., 100_000., 0.25), true),
        ((0.2, 4., 1., 0.25, 0., 0.1), (1., 0., 0.25), false),
        ((0.2, 4., 1., 0.25, 0., 0.1), (1., 100_000., 0.), false),
        ((0.2, 4., 1., 0.25, 0.001, 0.1), (1., 100_000., 0.65), false),
        ((0.2, 4., 1., 0.25, 0.001, 0.1), (1., 100_000., 0.64), true),
        ((0., 0., 0., 0., 0., 0.), (1., 100_000., 1.), false),
        ((0., 0., 0., 0., 0., 0.), (0., 0., 2.), true),
        ((0., 0., 0., 0., 0., 0.), (1e-300, 1e300, 0.25), true),
    ] {
        let c = RewardConfig {
            reward_mode: RewardMode::WinLoss,
            econ_shaping: w,
            econ_starvation_weight: s,
            econ_drought_weight: d,
            econ_cap: cap,
            econ_ineffective_weight: iw,
            econ_ineffective_cap: ic,
            econ_bank_weight: wb,
            econ_bank_scale: bs,
            econ_bank_cap: bc,
            econ_margin_weight: 0.,
            econ_margin_scale: 50_000.,
            econ_margin_cap: 0.,
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
        for field in 0..3 {
            let mut c = bank_reward_config();
            *[
                &mut c.econ_bank_weight,
                &mut c.econ_bank_scale,
                &mut c.econ_bank_cap,
            ][field] = bad;
            assert!(c.validate().is_err(), "{c:?}");
        }
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
            let rewards = cfg
                .transition(&before, &after, banks, banks, step == 718)
                .unwrap();
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

// --- Owner term A: absolute own-bank shaping (2026-09-30) -------------------

#[test]
fn bank_growth_pays_spending_costs_and_the_cap_binds() {
    let cfg = bank_only_config();
    cfg.validate().unwrap();
    assert_eq!(cfg.terminal_scale(), 0.75);
    let zero = [[0; 32]; 2];
    // 3,000 -> 5,000 is +.02; spending 5,000 -> 4,000 is -.01.
    assert_eq!(
        cfg.transition(&zero, &zero, [3_000., 3_000.], [5_000., 3_000.], false)
            .unwrap(),
        [(0.05_f64 - 0.03) as f32, 0.]
    );
    let spend = cfg
        .transition(&zero, &zero, [5_000., 3_000.], [4_000., 3_000.], false)
        .unwrap();
    assert!(spend[0] < 0.);
    assert_eq!(spend[0], (0.04_f64 - 0.05) as f32);
    assert_eq!(spend[1], 0.);
    // The score saturates at cap_b = .25, reached at bank 25,000.
    assert_eq!(cfg.bank_score(25_000.), 0.25);
    assert_eq!(cfg.bank_score(1e308), 0.25);
    assert_eq!(cfg.bank_score(-500.), 0.);
    assert_eq!(
        cfg.transition(&zero, &zero, [20_000., 0.], [30_000., 0.], false)
            .unwrap()[0],
        (0.25_f64 - 0.2) as f32
    );
    assert_eq!(
        cfg.transition(&zero, &zero, [30_000., 0.], [90_000., 0.], false)
            .unwrap()[0],
        0.
    );
    // Negative banks score zero, so debt carries no extra penalty.
    assert_eq!(
        cfg.transition(&zero, &zero, [0., 0.], [-400., 0.], false)
            .unwrap()[0],
        0.
    );
    // An overflowing product saturates at the cap instead of turning NaN.
    let mut huge = cfg.clone();
    huge.econ_bank_weight = f64::MAX;
    huge.econ_bank_scale = f64::MIN_POSITIVE;
    assert_eq!(huge.bank_score(f64::MAX), 0.25);
    assert_eq!(huge.bank_score(0.), 0.);
}

#[test]
fn bank_term_is_own_seat_only() {
    let cfg = bank_reward_config();
    let zero = [[0; 32]; 2];
    let seat1 = |seat0_after: f64| {
        cfg.transition(&zero, &zero, [3_000., 3_000.], [seat0_after, 7_000.], false)
            .unwrap()[1]
    };
    let expected = (cfg.bank_score(7_000.) - cfg.bank_score(3_000.)) as f32;
    for seat0_after in [0., 3_000., 12_345., 1e9] {
        assert_eq!(seat1(seat0_after), expected);
    }
    // Relative penalties still move both seats; the bank term does not.
    let mut after = zero;
    after[0][0] = 1;
    let r = cfg
        .transition(&zero, &after, [3_000., 3_000.], [3_000., 8_000.], false)
        .unwrap();
    assert_eq!(r[0], -0.25_f64 as f32);
    assert_eq!(r[1], (0.25 + (0.08 - 0.03_f64)) as f32);
}

#[test]
fn bank_term_reduces_the_terminal_scale() {
    let cfg = bank_reward_config();
    assert_eq!(cfg.terminal_scale(), 1. - 0.25 - 0.25);
    let mut both = cfg.clone();
    both.econ_ineffective_weight = 0.001;
    both.validate().unwrap();
    assert_eq!(both.terminal_scale(), 1. - 0.25 - 0.1 - 0.25);
    // Inactive: the cap does not count, whatever its value.
    let mut off = cfg.clone();
    off.econ_bank_weight = 0.;
    off.econ_bank_cap = 0.9;
    off.validate().unwrap();
    assert_eq!(off.terminal_scale(), 0.75);
    let zero = [[0; 32]; 2];
    let r = cfg
        .transition(&zero, &zero, [40_000., 10_000.], [40_000., 10_000.], true)
        .unwrap();
    assert_eq!(r, [0.5, -0.5]);
    let r = cfg
        .transition(&zero, &zero, [20_000., 10_000.], [30_000., 10_000.], true)
        .unwrap();
    let economic = (0.25_f64 - 0.2) as f32;
    assert_eq!(r[0], (f64::from(economic) + 0.5) as f32);
    assert_eq!(r[1], -0.5);
}

#[test]
fn preset_bank_values_match_the_accepted_consequences() {
    // The proposal's stated consequences: a 70k final bank earns +.175 and
    // 100k+ earns the full +.25; the reset bank of 3,000 scores .0075.
    let cfg = preset_bank_config();
    cfg.validate().unwrap();
    assert_eq!(cfg.terminal_scale(), 0.5);
    assert_eq!(cfg.bank_score(3_000.), 0.0075);
    assert_eq!(cfg.bank_score(25_000.), 0.0625);
    assert_eq!(cfg.bank_score(70_000.), 0.175);
    assert!(cfg.bank_score(99_999.) < 0.25);
    assert_eq!(cfg.bank_score(100_000.), 0.25);
    assert_eq!(cfg.bank_score(250_000.), 0.25);
    // A 73k -> 25k slide costs the sliding seat .12 (the run-J economy).
    let zero = [[0; 32]; 2];
    let r = cfg
        .transition(&zero, &zero, [73_000., 73_000.], [25_000., 73_000.], false)
        .unwrap();
    assert_eq!(r, [(0.0625_f64 - 0.1825) as f32, 0.]);
}

#[test]
fn bank_score_takes_the_product_then_the_quotient_then_the_cap() {
    // The parity contract's binary64 order. At a non-power-of-two weight the two
    // associations differ in the last bit: (.1 * 70,000) / 100,000 is .07, but
    // .1 * (70,000 / 100,000) is .06999999999999999.
    let cfg = RewardConfig {
        econ_bank_weight: 0.1,
        ..preset_bank_config()
    };
    cfg.validate().unwrap();
    assert_eq!(cfg.bank_score(70_000.).to_bits(), 0.07_f64.to_bits());
    for bank in [3_003., 3_006., 12_345., 70_000.] {
        let product_first = (0.1_f64 * bank) / 100_000.;
        let quotient_first = 0.1_f64 * (bank / 100_000.);
        assert_ne!(
            product_first.to_bits(),
            quotient_first.to_bits(),
            "row must discriminate"
        );
        assert_eq!(cfg.bank_score(bank).to_bits(), product_first.to_bits());
    }
}

#[test]
fn relative_and_bank_parts_are_summed_in_f64_and_rounded_to_f32_once() {
    // Seat 0 records one drought death, so seat 1's relative term is +.2, and
    // seat 1's own bank grows. Rounding each part to f32 separately and adding
    // in f32 gives a different bit pattern on these rows.
    let mut after = [[0; 32]; 2];
    after[0][1] = 1;
    let zero = [[0; 32]; 2];
    for (cfg, bank_after) in [
        (bank_reward_config(), 3_006.),
        (preset_bank_config(), 3_024.),
    ] {
        cfg.validate().unwrap();
        let bank = cfg.bank_score(bank_after) - cfg.bank_score(3_000.);
        let relative = cfg.penalty(&after[0]);
        assert_eq!(relative, 0.2);
        let once = (relative + bank) as f32;
        let separately = relative as f32 + bank as f32;
        assert_ne!(
            once.to_bits(),
            separately.to_bits(),
            "row must discriminate"
        );
        let r = cfg
            .transition(&zero, &after, [3_000., 3_000.], [3_000., bank_after], false)
            .unwrap();
        assert_eq!(r[1].to_bits(), once.to_bits());
        assert_eq!(r[0].to_bits(), (-0.2_f64 as f32).to_bits());
    }
}

#[test]
fn transition_rejects_nonfinite_banks_before_and_after() {
    let zero = [[0; 32]; 2];
    for cfg in [reward_config(), bank_reward_config()] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for seat in 0..2 {
                let mut banks = [0.; 2];
                banks[seat] = bad;
                assert!(cfg
                    .transition(&zero, &zero, banks, [0., 0.], false)
                    .is_err());
                assert!(cfg.transition(&zero, &zero, [0., 0.], banks, true).is_err());
            }
        }
    }
}

#[test]
fn disabled_bank_term_is_byte_identical_to_the_relative_reward() {
    // With w_b = 0 the reward ignores banks_before and any inactive scale/cap:
    // bit-equal to the relative-only formula computed independently here.
    let mut before = [[0; 32]; 2];
    before[1][0] = 2;
    let mut after = before;
    after[0][0] = 3;
    after[0][2] = 50;
    after[1][1] = 1;
    for (scale, cap) in [(100_000., 0.), (0., 0.), (1., 0.9)] {
        let mut cfg = reward_config();
        cfg.econ_bank_scale = scale;
        cfg.econ_bank_cap = cap;
        cfg.validate().unwrap();
        for banks_after in [[10., 5.], [5., 10.], [7., 7.], [-3., 1e300]] {
            for banks_before in [banks_after, [0., 0.], [1e9, -1e9]] {
                for done in [false, true] {
                    let delta: [f64; 2] =
                        std::array::from_fn(|s| cfg.penalty(&after[s]) - cfg.penalty(&before[s]));
                    let legacy: [f32; 2] = std::array::from_fn(|s| {
                        let economic = (delta[1 - s] - delta[s]) as f32;
                        let sign = f64::from(i8::from(banks_after[s] > banks_after[1 - s]))
                            - f64::from(i8::from(banks_after[s] < banks_after[1 - s]));
                        if done {
                            (f64::from(economic) + cfg.terminal_scale() * sign) as f32
                        } else {
                            economic
                        }
                    });
                    let actual = cfg
                        .transition(&before, &after, banks_before, banks_after, done)
                        .unwrap();
                    assert_eq!(actual.map(f32::to_bits), legacy.map(f32::to_bits));
                }
            }
        }
    }
}

#[test]
fn bank_episode_telescopes_to_final_minus_start_score() {
    let cfg = bank_reward_config();
    cfg.validate().unwrap();
    let steps = 719;
    // A deterministic bank walk: spend early, earn late, cross the cap.
    let bank = |seat: usize, t: usize| -> f64 {
        let t = t as f64;
        if seat == 0 {
            3_000. - 20. * t.min(100.) + 60. * (t - 100.).max(0.)
        } else {
            3_000. + 7.5 * t - 2. * (t * 0.37).sin() * 100.
        }
    };
    let mut before = [[0; 32]; 2];
    let mut sum = [0.0_f64; 2];
    let mut budget = [0.0_f64; 2];
    for t in 0..steps {
        let mut after = before;
        after[1][0] += i64::from(t % 150 == 0);
        let b0 = [bank(0, t), bank(1, t)];
        let b1 = [bank(0, t + 1), bank(1, t + 1)];
        let done = t + 1 == steps;
        let r = cfg.transition(&before, &after, b0, b1, done).unwrap();
        for s in 0..2 {
            let relative = (cfg.penalty(&after[1 - s]) - cfg.penalty(&before[1 - s]))
                - (cfg.penalty(&after[s]) - cfg.penalty(&before[s]));
            let economic = relative + (cfg.bank_score(b1[s]) - cfg.bank_score(b0[s]));
            budget[s] += half_ulp32(economic) + 4. * f64::EPSILON;
            if done {
                budget[s] += half_ulp32(f64::from(economic as f32) + 1.);
            }
            sum[s] += f64::from(r[s]);
        }
        before = after;
    }
    let final_banks = [bank(0, steps), bank(1, steps)];
    let start = [bank(0, 0), bank(1, 0)];
    for s in 0..2 {
        let relative = cfg.penalty(&before[1 - s]) - cfg.penalty(&before[s]);
        let sign = if final_banks[s] > final_banks[1 - s] {
            1.
        } else {
            -1.
        };
        let endpoint = relative
            + (cfg.bank_score(final_banks[s]) - cfg.bank_score(start[s]))
            + cfg.terminal_scale() * sign;
        assert!(
            (sum[s] - endpoint).abs() <= budget[s],
            "{s}: {sum:?} {endpoint}"
        );
        assert!(endpoint.abs() <= 1.);
    }
    // The walk crossed the cap for seat 0 and spent below the start first.
    assert_eq!(cfg.bank_score(final_banks[0]), 0.25);
    assert!(bank(0, 100) < start[0]);
}

#[test]
fn autoreset_never_spans_two_games_in_the_bank_term() {
    let config: Config = serde_json::from_value(json!({"episodeSteps":4})).unwrap();
    let cfg = bank_only_config();
    let mut env = NativeEnv::new(1, 17000, 1, config, cfg.clone(), 1, 241).unwrap();
    let mut out = Output::new(1);
    out.observe(&env);
    let mut spent = false;
    let mut after_terminal = None;
    for transition in 1..=6 {
        let market = if transition == 1 {
            json!([["BUY_SEED", "WHEAT", 2]])
        } else {
            json!([])
        };
        let (t, l) = actions(&env, json!(["PASS"]), market);
        out.step(&mut env, &t, &l).unwrap();
        let before: [f64; 2] = out.transition.transition_banks_before[..]
            .try_into()
            .unwrap();
        let after: [f64; 2] = out.transition.transition_banks_after[..]
            .try_into()
            .unwrap();
        spent |= after[0] < 3_000.;
        let done = out.transition.dones[0];
        let expected = cfg
            .transition(&[[0; 32]; 2], &[[0; 32]; 2], before, after, done)
            .unwrap();
        // Counters are zero-weight here, so only the bank and terminal terms act.
        assert_eq!(out.transition.rewards, expected, "transition {transition}");
        if after_terminal == Some(transition) {
            // The first transition of the next game starts from its reset bank,
            // never the previous game's final bank.
            assert_eq!(before, [3_000., 3_000.]);
        }
        if done && after_terminal.is_none() {
            // The first game ends below its reset bank (the seed purchase), so
            // a carried-over bank would pay +.0002 on the next transition.
            assert!(after[0] < 3_000., "the bank walk must be discriminating");
            after_terminal = Some(transition + 1);
        }
    }
    assert!(spent);
    assert!(after_terminal.is_some_and(|t| t <= 6));
}

// --- Owner term M: cash-difference (margin) potential (2026-09-30) ---------

/// The margin preset's reward: term M (w_m .5, S_m 50,000, c_m .5) plus the
/// terminal sign at scale .5; death, ineffective and own-bank shaping off.
fn margin_config() -> RewardConfig {
    RewardConfig {
        econ_shaping: 0.,
        econ_ineffective_weight: 0.,
        econ_margin_weight: 0.5,
        econ_margin_scale: 50_000.,
        econ_margin_cap: 0.5,
        ..reward_config()
    }
}

#[test]
fn margin_validation_budget_and_terminal_scale() {
    let cfg = margin_config();
    cfg.validate().unwrap();
    assert_eq!(cfg.terminal_scale(), 0.5);
    for bad in [f64::NAN, f64::INFINITY, -1.] {
        for field in 0..3 {
            let mut c = margin_config();
            *[
                &mut c.econ_margin_weight,
                &mut c.econ_margin_scale,
                &mut c.econ_margin_cap,
            ][field] = bad;
            assert!(c.validate().is_err(), "{c:?}");
        }
    }
    let mut c = margin_config();
    c.econ_margin_scale = 0.;
    assert!(c.validate().unwrap_err().contains("econ_margin_scale"));
    c = margin_config();
    c.econ_margin_cap = 0.;
    assert!(c.validate().unwrap_err().contains("econ_margin_cap"));
    // The margin cap joins the active-cap budget with every other enabled cap.
    c = margin_config();
    c.econ_margin_cap = 1.;
    assert!(c.validate().is_err());
    c.econ_margin_cap = 0.99;
    c.validate().unwrap();
    c = RewardConfig {
        econ_shaping: 0.2,
        ..margin_config()
    };
    assert_eq!(c.terminal_scale(), 1. - 0.25 - 0.5);
    c.econ_margin_cap = 0.75;
    assert!(c.validate().is_err());
    c = RewardConfig {
        econ_bank_weight: 0.25,
        econ_bank_cap: 0.25,
        ..margin_config()
    };
    c.validate().unwrap();
    assert_eq!(c.terminal_scale(), 0.25);
    c.econ_margin_cap = 0.75;
    assert!(c.validate().is_err());
    // Disabled: any scale/cap is inert, even ones that would break the budget.
    c = reward_config();
    c.econ_margin_scale = 0.;
    c.econ_margin_cap = 5.;
    c.validate().unwrap();
    assert_eq!(c.terminal_scale(), reward_config().terminal_scale());
    assert_eq!(c.margin_score(1e9), 0.);
}

#[test]
fn margin_score_is_linear_then_clamps_at_both_signs() {
    let cfg = margin_config();
    assert_eq!(cfg.margin_score(0.), 0.);
    assert_eq!(cfg.margin_score(10_000.), 0.1);
    assert_eq!(cfg.margin_score(-10_000.), -0.1);
    assert_eq!(cfg.margin_score(25_000.), 0.25);
    assert!(cfg.margin_score(49_999.) < 0.5);
    assert_eq!(cfg.margin_score(50_000.), 0.5);
    assert_eq!(cfg.margin_score(-50_000.), -0.5);
    assert_eq!(cfg.margin_score(1e300), 0.5);
    assert_eq!(cfg.margin_score(-1e300), -0.5);
    // Product, quotient, then clamp: a product that overflows saturates.
    let mut huge = margin_config();
    huge.econ_margin_weight = f64::MAX;
    huge.econ_margin_scale = f64::MIN_POSITIVE;
    assert_eq!(huge.margin_score(f64::MAX), 0.5);
    assert_eq!(huge.margin_score(-f64::MAX), -0.5);
    assert_eq!(huge.margin_score(0.), 0.);
    // At weight .1 and scale 50,000 the two associations differ in the last bit
    // on every margin below, so a quotient-first score fails.
    let odd = RewardConfig {
        econ_margin_weight: 0.1,
        ..margin_config()
    };
    for margin in [3_003., 12_345., 20_000., 40_000.] {
        let product_first = (0.1_f64 * margin) / 50_000.;
        let quotient_first = 0.1_f64 * (margin / 50_000.);
        assert_ne!(product_first.to_bits(), quotient_first.to_bits());
        assert_eq!(odd.margin_score(margin).to_bits(), product_first.to_bits());
        assert_eq!(
            odd.margin_score(-margin).to_bits(),
            (-product_first).to_bits()
        );
    }
}

#[test]
fn margin_term_is_zero_sum_bit_for_bit() {
    let mut before = [[0; 32]; 2];
    before[1][0] = 2;
    let mut after = before;
    after[0][0] = 3;
    after[1][1] = 1;
    for cfg in [
        margin_config(),
        RewardConfig {
            econ_shaping: 0.2,
            econ_margin_cap: 0.4,
            ..margin_config()
        },
    ] {
        cfg.validate().unwrap();
        for (b0, b1) in [
            ([3_000., 3_000.], [2_100., 4_700.]),
            ([12_345.5, 7.25], [60_000., 1.]),
            ([-40., 90_000.], [1e6, -1e6]),
            ([5., 5.], [5., 5.]),
        ] {
            for done in [false, true] {
                let r = cfg.transition(&before, &after, b0, b1, done).unwrap();
                // Exact negation (a +0.0 / -0.0 pair counts as zero-sum).
                assert_eq!(r[0], -r[1], "{b0:?} {b1:?} {done}");
            }
        }
    }
    // The own-bank term is not zero-sum, so only term M must be.
}

#[test]
fn margin_transition_adds_the_potential_difference_before_f32() {
    let cfg = RewardConfig {
        econ_shaping: 0.2,
        econ_margin_cap: 0.4,
        ..margin_config()
    };
    let before = [[0; 32]; 2];
    let mut after = before;
    after[0][0] = 1;
    // A 12,345 lead (score .12345) with a .25 death penalty: summing in f64
    // and rounding once differs in the last bit from rounding the margin
    // separately and adding in f32, so a double rounding fails.
    let (b0, b1) = ([3_000., 3_000.], [15_345., 3_000.]);
    let r = cfg.transition(&before, &after, b0, b1, false).unwrap();
    for s in 0..2 {
        let relative = (cfg.penalty(&after[1 - s]) - cfg.penalty(&before[1 - s]))
            - (cfg.penalty(&after[s]) - cfg.penalty(&before[s]));
        let margin = cfg.margin_score(b1[s] - b1[1 - s]) - cfg.margin_score(b0[s] - b0[1 - s]);
        let single = ((relative + margin) as f32).to_bits();
        let double = ((relative as f32) + (margin as f32)).to_bits();
        assert_ne!(
            single, double,
            "seat {s}: the inputs must separate the orders"
        );
        assert_eq!(r[s].to_bits(), single);
    }
    assert_eq!(r[0], (-0.25_f64 + 0.12345) as f32);
    let done = cfg.transition(&before, &after, b0, b1, true).unwrap();
    assert_eq!(done[0], (f64::from(r[0]) + 0.35) as f32);
}

#[test]
fn disabled_margin_term_is_byte_identical_to_the_previous_reward() {
    // w_m = 0 with any inactive scale/cap reproduces the pre-M formula
    // (relative, then own bank when enabled, one f32 rounding) bit for bit.
    let mut before = [[0; 32]; 2];
    before[1][0] = 2;
    let mut after = before;
    after[0][0] = 3;
    after[0][2] = 50;
    after[1][1] = 1;
    for base in [reward_config(), bank_reward_config()] {
        for (scale, cap) in [(50_000., 0.), (0., 0.), (1., 0.9)] {
            let mut cfg = base.clone();
            cfg.econ_margin_scale = scale;
            cfg.econ_margin_cap = cap;
            cfg.validate().unwrap();
            for banks_after in [[10., 5.], [5., 10.], [7., 7.], [-3., 1e300]] {
                for banks_before in [banks_after, [0., 0.], [1e9, -1e9]] {
                    for done in [false, true] {
                        let legacy: [f32; 2] = std::array::from_fn(|s| {
                            let relative = (cfg.penalty(&after[1 - s])
                                - cfg.penalty(&before[1 - s]))
                                - (cfg.penalty(&after[s]) - cfg.penalty(&before[s]));
                            let economic = if cfg.econ_bank_weight > 0. {
                                (relative
                                    + (cfg.bank_score(banks_after[s])
                                        - cfg.bank_score(banks_before[s])))
                                    as f32
                            } else {
                                relative as f32
                            };
                            let sign = f64::from(i8::from(banks_after[s] > banks_after[1 - s]))
                                - f64::from(i8::from(banks_after[s] < banks_after[1 - s]));
                            if done {
                                (f64::from(economic) + cfg.terminal_scale() * sign) as f32
                            } else {
                                economic
                            }
                        });
                        let actual = cfg
                            .transition(&before, &after, banks_before, banks_after, done)
                            .unwrap();
                        assert_eq!(actual.map(f32::to_bits), legacy.map(f32::to_bits));
                    }
                }
            }
        }
    }
}

#[test]
fn margin_episode_telescopes_to_the_final_margin_score() {
    let cfg = margin_config();
    let steps = 719;
    // Equal reset banks (the engine gives both farms startingMoney). Seat 0
    // spends first, then out-earns seat 1 and crosses the +50k clamp late; the
    // lead also swings sign mid-game.
    let bank = |seat: usize, t: usize| -> f64 {
        let t = t as f64;
        if seat == 0 {
            3_000. - 20. * t.min(100.) + 150. * (t - 100.).max(0.)
        } else {
            3_000. + 40. * t - 2. * (t * 0.37).sin() * 100.
        }
    };
    let zero = [[0; 32]; 2];
    let mut sum = [0.0_f64; 2];
    let mut budget = [0.0_f64; 2];
    let mut saw_lead = [false; 2];
    for t in 0..steps {
        let b0 = [bank(0, t), bank(1, t)];
        let b1 = [bank(0, t + 1), bank(1, t + 1)];
        let done = t + 1 == steps;
        let r = cfg.transition(&zero, &zero, b0, b1, done).unwrap();
        for s in 0..2 {
            saw_lead[s] |= b1[s] > b1[1 - s];
            let economic =
                cfg.margin_score(b1[s] - b1[1 - s]) - cfg.margin_score(b0[s] - b0[1 - s]);
            budget[s] += half_ulp32(economic) + 4. * f64::EPSILON;
            if done {
                budget[s] += half_ulp32(f64::from(economic as f32) + 1.);
            }
            sum[s] += f64::from(r[s]);
        }
    }
    assert_eq!(bank(0, 0), bank(1, 0));
    assert_eq!(saw_lead, [true, true], "the lead must change hands");
    let final_banks = [bank(0, steps), bank(1, steps)];
    assert!(final_banks[0] - final_banks[1] > 50_000.);
    for s in 0..2 {
        let sign = if final_banks[s] > final_banks[1 - s] {
            1.
        } else {
            -1.
        };
        let endpoint =
            cfg.margin_score(final_banks[s] - final_banks[1 - s]) + cfg.terminal_scale() * sign;
        assert!(
            (sum[s] - endpoint).abs() <= budget[s],
            "{s}: {sum:?} {endpoint}"
        );
        assert!(endpoint.abs() <= 1.);
    }
    // Saturated win: +1 and -1 exactly in f64 endpoints.
    assert_eq!(cfg.margin_score(final_banks[0] - final_banks[1]), 0.5);
}

#[test]
fn autoreset_never_spans_two_games_in_the_margin_term() {
    let config: Config = serde_json::from_value(json!({"episodeSteps":4})).unwrap();
    let cfg = margin_config();
    let mut env = NativeEnv::new(1, 17000, 1, config, cfg.clone(), 1, 241).unwrap();
    let mut out = Output::new(1);
    out.observe(&env);
    let mut after_terminal = None;
    let mut first = true;
    for transition in 1..=6 {
        let market = if transition == 1 {
            json!([["BUY_SEED", "WHEAT", 2]])
        } else {
            json!([])
        };
        // Only seat 0 buys seed; seat 1 always passes, so the banks diverge.
        let (mut t, mut l) = actions(&env, json!(["PASS"]), market);
        let (pass_t, pass_l) = actions(&env, json!(["PASS"]), json!([]));
        let seat1 = grammar::TOKENS_PER_SEAT..2 * grammar::TOKENS_PER_SEAT;
        t[seat1.clone()].copy_from_slice(&pass_t[seat1]);
        l[1] = pass_l[1];
        out.step(&mut env, &t, &l).unwrap();
        let before: [f64; 2] = out.transition.transition_banks_before[..]
            .try_into()
            .unwrap();
        let after: [f64; 2] = out.transition.transition_banks_after[..]
            .try_into()
            .unwrap();
        if first {
            // Equal reset banks: the potential starts at zero.
            assert_eq!(before, [3_000., 3_000.]);
            first = false;
        }
        let done = out.transition.dones[0];
        let expected = cfg
            .transition(&[[0; 32]; 2], &[[0; 32]; 2], before, after, done)
            .unwrap();
        assert_eq!(out.transition.rewards, expected, "transition {transition}");
        assert_eq!(expected[0], -expected[1]);
        if after_terminal == Some(transition) {
            // The next game's first transition starts from equal reset banks,
            // never the previous game's final margin.
            assert_eq!(before, [3_000., 3_000.]);
        }
        if done && after_terminal.is_none() {
            // The first game ends with seat 0 behind (the seed purchase), so a
            // carried-over margin would pay seat 0 on the next transition.
            assert!(
                after[0] < after[1],
                "the margin walk must be discriminating"
            );
            after_terminal = Some(transition + 1);
        }
    }
    assert!(after_terminal.is_some_and(|t| t <= 6));
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
