# Native core independent verification — Task 1.4, r3

Head `1e63597957ed4995dc3eac47e996349a485670a9`, three-dot base `e197528`. No native production finding. The audit read the brief and native buffer Reference, then inspected staged reset/step/observe, seed reservation, selected-row commit, terminal capture, reward arithmetic/admission, HIRE cast admission, and corresponding test code. Vendored engine bytes were not changed.

## Findings

None. Native scope supports APPROVE, subject to the parent review's other lanes.

## Independent mutation results

All **25 source mutants killed** across **19 named test oracles**. Every mutation was confined to `.codex-tmp/verify-env-r3-core`, a `git archive HEAD` copy. Its Cargo target was clone-copied as independent files; it never pointed at the primary target. No primary source or installed extension was edited. Every mutation restores original source bytes in `finally`, verifies the SHA-256, and checks primary source equality. `source-custody-after.json` confirms all seven relevant source/manifests remain byte-exact with the scratch copy; `native-mutation-summary.json` contains each before/mutant/after hash, exact command, exit code and bounded receipt.

Each subprocess used two Cargo jobs, offline resolution, 115 seconds and a 960 MiB sampled RSS stop. All 25 mutants finished within the guard; maximum sampled group RSS was 857,128,960 bytes. They are failed assertion tests, not compile errors or guard stops.

| Mutation | Named oracle | Discriminating failure |
|---|---|---|
| hire-admission-wiring | `executed_unbounded_hire_cast_rejects_before_publication` | killed; unbounded hire must reject: () |
| hire-later-seat | `hire_cash_checks_later_executed_hire_and_both_seats` | killed; named test failure; see log |
| hire-negative-multiplier | `hire_cash_rejects_negative_multiplier_explicitly` | killed; named test failure; see log |
| hire-no-execution | `hire_cash_no_execution_does_not_cap_configuration` | killed; assertion left == right failed |
| hire-oversized | `hire_cash_rejects_oversized_exact_cost_without_float_saturation` | killed; named test failure; see log |
| hire-rounded-threshold | `hire_cash_rejects_cost_that_rounds_up_to_two_to_64` | killed; named test failure; see log |
| hire-safe-sequences | `hire_cash_admits_safe_executed_sequences` | killed; assertion left == right failed |
| horizon-off-by-one | `default_horizon_terminates_at_719` | killed; called Result::unwrap() on an Err value: Value("env=0 terminal seed prediction disagrees with kernel") |
| observe-holds-gil | `native_observe_releases_gil_with_controlled_latch` | killed; test kaggriculture::bindings::detached_test::native_observe_releases_gil_with_controlled_latch ... pyo3_runtime.PanicException: Python thread could not progress while real observe work held the test latch |
| reset-live-seed | `reset_and_truncate_failures_preserve_every_published_byte` | killed; assertion left == right failed: ResetConstruct, mask=[true, true] |
| reset-terminal-clear | `reset_and_truncate_failures_preserve_every_published_byte` | killed; assertion left == right failed: ResetConstruct, mask=[true, true] |
| reward-active-cap-sum | `reward_rejects_invalid_coefficients_and_handles_huge_counts` | killed; assertion failed: c.validate().is_err() |
| reward-death-cap | `reward_admission_predicate_cases` | killed; assertion left == right failed: RewardConfig { reward_mode: WinLoss, econ_shaping: 0.2, econ_starvation_weight: 4.0, econ_drought_weight: 1.0, econ_cap: 0.0, econ_ineffective_weight: 0.0, econ_ineffective_cap: 0.0 } |
| reward-ineffective-cap | `reward_admission_predicate_cases` | killed; assertion left == right failed: RewardConfig { reward_mode: WinLoss, econ_shaping: 0.0, econ_starvation_weight: 0.0, econ_drought_weight: 0.0, econ_cap: 0.0, econ_ineffective_weight: 0.001, econ_ineffective_cap: 0.0 } |
| reward-negative | `reward_rejects_invalid_coefficients_and_handles_huge_counts` | killed; assertion failed: c.validate().is_err() |
| reward-nonfinite | `reward_rejects_invalid_coefficients_and_handles_huge_counts` | killed; assertion failed: c.validate().is_err() |
| reward-own-counters | `reward_uses_only_own_counters_and_raw_banks` | killed; assertion left == right failed |
| reward-telescoping | `reward_episode_telescopes_with_explicit_output_rounding_budget` | killed; assertion failed: (actual_sum[s] - endpoint).abs() <= output_budget[s] + budget64 |
| reward-underflow-products | `reward_admission_predicate_cases` | killed; assertion left == right failed: RewardConfig { reward_mode: WinLoss, econ_shaping: 1e-300, econ_starvation_weight: 1e-300, econ_drought_weight: 1e-300, econ_cap: 0.25, econ_ineffective_weight: 0.0, econ_ineffective_cap: 0.0 } |
| seed-overflow | `seed_construction_reset_stride_and_overflow` | killed; assertion failed: stream.reserve(&[true, true]).is_err() |
| seed-stride | `seed_construction_reset_stride_and_overflow` | killed; assertion left == right failed |
| step-live-seed | `batch_failure_preserves_every_published_byte` | killed; assertion left == right failed: None: env=1 seat=0 grammar support: frame 0 slot unit_kind value 19 |
| timing-fixture-density | `lifecycle_timing_fixtures_step_and_prepare` | killed; assertion left == right failed |
| truncate-transition-clear | `truncate_preserves_transition_and_no_winner` | killed; assertion left == right failed |
| truncate-whole-publish | `truncate_commits_only_selected_rows` | killed; unselected observation bytes changed: env=0, terminal=false |

## Release overflow

Canonical fat-LTO release build: stopped at the RSS guard after 20.81 seconds, sampled 1,018,511,360 bytes. The test body did not run. This is a check limitation, not a failing overflow oracle.

Reduced-LTO release (`profile.release.lto=false`, `profile.release.codegen-units=16`) passed `release_dependency_overflow_is_caught`: **1 passed**, 49.47 seconds, sampled 826,523,648 bytes. Removing only the engine overflow policy using Cargo `--config profile.release.package.kaggriculture-engine.overflow-checks=false` produced **1 failed**, at `src/kaggriculture/env_tests.rs:558`: `engine overflow must panic inside the caught candidate step`. The mutant completed in 41.37 seconds at 860,880,896 bytes. Restoring the configuration passed **1 test** again. This configuration mutation never changed a source file.

The independent reduced-LTO proof does not establish fat-LTO equivalence or new performance data. `pod-source-comparison.json` confirms all ten recorded source/lockfile hashes from the existing canonical pod release receipt equal current HEAD; `src/kaggriculture/env_tests.rs` also equals the pod source `9dc2d02` byte-for-byte. The existing canonical proof remains applicable source-bound evidence, and was not rerun on the pod in this verification.

## Restored verification

After all source mutations, `cargo test --locked --offline --lib kaggriculture:: -- --test-threads=1` passed **117 tests, 3 ignored**, with 159 filtered out. It completed in 29.60 seconds including compilation at 555,581,440 bytes sampled peak RSS. The reduced-LTO release proof also returned green after removing the override-off config.

## Scope

The 1.5 Python adapter, pinned CUDA reuse fence, distributed trainer behavior and complete-update throughput are outside Task 1.4 and unclaimed. Main baseline and binding/oracle suites are owned by the parent verifier and sibling lanes.
