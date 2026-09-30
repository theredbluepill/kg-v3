Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 3.1 remainder: test non-vacuity and coverage review (r1)

## Scope

- Target: branch `kg/rebuild-3-1` at `7a55e81` (Codex `15ea55f` plus Claude review commit), diff `49a4835...HEAD`.
- Lens: whether the new and un-skipped tests actually discriminate. This covers teacher T18/T19b, the native two-update smoke, native policy evaluation, Kaggriculture and Orbit mapping, W&B identity, seed budget and startup guards. It also checks the doc and cookbook claims against tests, and the stateless-policy constraints.
- Inputs read: the full code, test, doc and cookbook diff; `ops/rebuild-2026-09-29/codex/task-3.1-rest2-report.md`; and the adapter `step` admission in `python/owl/kaggriculture/env.py`.
- Method: I made a scratch worktree `git worktree add --detach /tmp/cv-3.1-tests-r1 HEAD`, copied the fixtures in from `kg-v3/tests/fixtures`, and copied in `python/owl/rs.abi3.so` from `kg-v3-t31`. That build dates from 23:37, after the last Rust change `d0d65f7` at 22:17, and no Rust changes in `49a4835...HEAD`. I ran the `kg-v3-t31` venv interpreter with `PYTHONPATH` set to the scratch `python/`, and confirmed that `owl`, `owl.rs` and `owl.train.ppo` resolved to the scratch copy. The scratch worktree has been removed. `kg-v3-t31` has 0 tracked modifications and is still at `7a55e81`.

## Checks (scratch copy, CPU, OMP_NUM_THREADS=2)

| Check | Result |
|---|---|
| Smoke, mapping, logging, benchmark files | 241 passed, 0.42 s, peak RSS 346 MB |
| `tests/scripts/test_run_ppo.py` and `tests/kaggriculture/test_configs.py` | 172 passed, 1.47 s, 379 MB |
| The four un-skipped teacher tests (T18, trainer checkpoint, T19b resume, T19b fresh launch) | 4 passed, 0 skipped, 379 MB |
| Non-Kaggriculture suite (`-m "not slow" tests --ignore=tests/kaggriculture`) | 1276 passed, 3 skipped, 31.6 s, 687 MB |
| Kaggriculture suite, per-file shards (27 files) | 1092 passed, 3 skipped (1 CUDA fence, 2 observe). This matches Codex's 1092/3. |
| Shard RSS | 25 shards ran under 1 GB. `test_model_heads.py` (1.087 GB) and `test_teacher.py` (1.025 GB) went slightly over the 1 GB budget; I didn't split them further. |
| `ruff check` / `ruff format --check` on python, scripts, tests | Passed / 137 files already formatted |
| `mypy python/ scripts/` | No issues in 70 source files |
| `scripts/check_doc_freshness.py` | rc 0 |
| Probe, not committed: smoke with `truncation_step=1, truncation_prob=1.0, episodeSteps=6` through the native env | Passed. Every row truncated, and bootstrap values were finite. |

Totals: 1276 + 1092 = 2368 passed and 6 skipped, which matches the report's 2368/6. I didn't run the monolithic `just prepare`, because Codex's attempt hit the 1 GB watchdog.

## Mutations (19 applied, each reverted via `git checkout`)

The target set was `test_ppo_observation_mapping.py`, `test_training_smoke.py`, the 4 teacher tests, `test_run_ppo.py`, `test_logging.py`, `test_benchmark_checkpoints.py` and `test_configs.py`. `-x` stopped each run at the first failure.

| # | Mutation | Result | First killing test |
|---|---|---|---|
| M1 | `_copy_actions_time_step`: drop the Kaggriculture `lengths` copy | CAUGHT | smoke only. The mapping test never compares `lengths` after `write_step`. |
| M2 | `_actions_to_cpu`: drop `.contiguous()` | CAUGHT | `test_kaggriculture_actions_to_cpu_materializes_contiguous_int64` |
| M3 | `_map_action_bundle`: leave `lengths` unmapped | CAUGHT | same |
| M4 | Remove the `isinstance(batch_segment_obs, ObsBatch)` winner-reshape guard | CAUGHT | smoke, T18 and the trainer-checkpoint test (3 tests, run without `-x`) |
| M5 | Eval terminal dict: swap `bank_0`/`bank_1` | CAUGHT | `test_kaggriculture_policy_evaluation_runs_native_games` |
| M5b | Eval terminal dict: drop the `winner` key, which turns off the `terminal_seat_banks` cross-check | **SURVIVED** | none |
| M6 | Seed budget: reserve only one construction/reset batch | CAUGHT | `test_kaggriculture_seed_budget_covers_resets_truncation_and_update_overshoot` |
| M6b | Seed budget: drop the update-overshoot term | CAUGHT | same |
| M7 | `create_env(rank=0)` | CAUGHT | `test_main_kaggriculture_rollout_factory_uses_rank_seed_and_transfer_device[cpu]` |
| M8 | Kaggriculture `_select_actions` tokens use `~use_a` | CAUGHT | eval native-games test and `test_select_kaggriculture_actions_preserves_seats_and_native_storage` |
| M9 | Drop the `kaggriculture-v3` W&B tag | CAUGHT | `test_kaggriculture_wandb_init_uses_v3_identity_and_mode[online]` |
| M10 | `_evaluate_games` Kaggriculture replay guard disabled | **SURVIVED** | none (the `main` guard runs first, so this one is defence in depth) |
| M11 | `train/max_entities` uses `still_playing` instead of `actor_mask` | **SURVIVED** | none |
| M12 | Offline+resume guard in `WandbLogger` disabled | CAUGHT | `test_offline_wandb_rejects_resume_before_initialization` |
| M13 | `main` Kaggriculture replay guard disabled | CAUGHT | `test_main_rejects_kaggriculture_replay_before_allocation` |
| M14 | Rollout factory `transfer_device=cpu` | CAUGHT | factory test `[cuda]` |
| M15 | Eval Kaggriculture action-type guard disabled | **SURVIVED** | none |
| M16 | Orbit `DiscreteTargetBinActions` time-step copy drops `fleet_bin` | CAUGHT | `test_orbit_storage_and_actions_equal_base_49a4835[action_spec2-...]` |
| M17 | `main` passes `args.max_env_steps` instead of the computed safe ceiling to `_run_training_session` | **SURVIVED** | none (176 runner, teacher and smoke tests passed) |
| M18 | Kaggriculture actions always written to step 0 | CAUGHT | `test_kaggriculture_storage_and_every_mapping_preserve_schema` |
| M19 | `_kaggriculture_step_limit` returns `None` when no limit is given | CAUGHT | seed-budget test |

Result: 14 of 19 caught. Every core storage, mapping, transport, seat-selection, seed-budget, factory, W&B and teacher-path mutation was killed. The 5 survivors are defensive guards, one telemetry metric, and one documented default that isn't wired through to a test.

## Stateless-policy constraints

- The smoke asserts `trainer._hidden_state is None` after two native updates. The rollout buffer adds no hidden or history field. Kaggriculture storage is the native one-row allocation with a leading horizon dimension added, and nothing more.
- Evaluation feeds both models the same device observation and mixes their actions by seat. The seat assignment never reaches model inputs, embeddings, heads, losses or checkpoints. W&B tags and names carry no opponent identity.
- The four teacher tests assert that teacher and table state are absent from checkpoint key sets and state dicts.
- I found no violation.

## Findings

No P1 or P2 findings. The P3 findings below are test gaps and polish.

- **P3: the documented default "an omitted `--max-env-steps` uses the safe ceiling" is untested.**
  - Where: `scripts/run_ppo.py:176-178` and `:319`.
  - Evidence: M17 survived. The ceiling is about 2^61 steps, so this is not a practical risk. It is still a claim in README, `docs/rl-api-specs.md` and `docs/kaggriculture-contract.md` that no test checks.
  - Fix: in the factory-style `main` test, patch `_run_training_session` to capture `max_env_steps`, and assert that it equals `_kaggriculture_step_limit(cfg, ctx, max_env_steps=None)`.
- **P3: the Kaggriculture `train/max_entities` metric is untested.**
  - Where: `python/owl/train/ppo.py:750-758`.
  - Evidence: M11 survived. A wrong mask (for example `still_playing`) would log a plausible but meaningless number to W&B.
  - Fix: in the smoke, assert that `metrics["train/max_entities"]` equals the maximum of `actor_mask.sum(-1)` over the rollout's `still_playing` seats.
- **P3: the defensive guards are untested.**
  - Where: the `_evaluate_games` Kaggriculture replay guard and the eval action-type `TypeError`s at `scripts/run_ppo.py:1465-1469` and `:1536-1544`. Also the `_PPORolloutBuffer` obs/action family-mismatch `ValueError`s at `ppo.py:173-177` and `:216-217`.
  - Evidence: M10 and M15 survived. No test matches either rollout-buffer message.
  - Fix: add direct unit tests, one per message, or remove the unreachable duplicates.
- **P3: the eval `winner` cross-check has no test.**
  - Where: `scripts/run_ppo.py:630-636`.
  - Evidence: M5b survived. Dropping `"winner"` silently disables `terminal_seat_banks`'s consistency check.
  - Fix: assert in the native-eval test that the per-game `metrics["winner"]` is present and agrees with the sign of the margin.
- **P3: part of the native-eval test is a tautology.**
  - Where: `tests/scripts/test_run_ppo.py`, `test_kaggriculture_policy_evaluation_runs_native_games`.
  - Evidence: `candidate - incumbent == margin` restates `_candidate_bank_metrics`, and `expected_wins` is derived from the same margins. The test kills M5 and M8 only through the cross-check and through native admission.
  - Fix: add an independent oracle. For example, replay the same evaluation seeds with a recorded action stream and compare final banks, or assert the per-seat banks against `env.terminal_metrics`.
- **P3: `_copy_actions_time_step` has an implicit no-op fallthrough.**
  - Where: `python/owl/train/ppo.py:2305-2328`.
  - Evidence: the old generic field loop always copied. The new `if/elif` chain has no `else: raise`, so any future action type would store zeros silently. That goes against the repo's fail-fast rule.
  - Fix: add `else: raise TypeError(...)`.
- **P3: the Kaggriculture mapping test doesn't check `lengths` after `write_step`, and writes identical actions at every step.**
  - Where: `tests/owl/train/test_ppo_observation_mapping.py:614-673`.
  - Evidence: M1 was caught only by the smoke.
  - Fix: vary actions per step and assert `rollout.actions.lengths[step]` and `tokens[step]` for each step.
- **P3: the Orbit-parity oracle depends on git history and the working directory.**
  - Where: `tests/owl/train/test_ppo_observation_mapping.py:689-701`.
  - Evidence: `subprocess.check_output(["git","show","49a4835:..."])` has no `cwd`. It errors in a shallow clone, an rsync'd tree without `.git`, or a run from another directory.
  - Fix: pass `cwd=` the repo root, and either vendor a frozen reference module or state the history requirement.
- **P3: Kaggriculture truncation through `PPOTrainer` is not in the suite.**
  - Evidence: the cookbook already discloses this. My uncommitted probe passed on CPU, which suggests a regression test would be cheap. The presets don't enable truncation, but the seed budget reserves for it.
  - Fix: add the probe as a test, with `truncation_step=1`, `truncation_prob=1.0`, and assertions on `rollout.truncated` and finite `bootstrap_values`.
- **P3: the shipped 2/4/8-rank Kaggriculture configs fail startup by default** (`eval_replay_games: 8`).
  - Where: `configs/kaggriculture_{2,4,8}rank.yaml:69-70`.
  - Evidence: README and the cookbook disclose the `-o rl.eval_replay_games=0` override. The canonical multi-rank presets still can't launch unmodified.
  - Fix: set 0 in the YAMLs until Task 7.3, or keep it as a deliberate, documented stop.
- **P3: resource note.**
  - Evidence: `test_model_heads.py` (1.087 GB) and `test_teacher.py` (1.025 GB) each exceed 1 GB when run as a single file. The four newly un-skipped teacher tests alone peak at 379 MB, so the excess comes from pre-existing tests.
  - Fix: split those files in bounded runs.

## Claims checked against tests

- Held: the rl-api-specs and contract storage shapes and dtypes, the C-contiguous CPU int64 action transport, Orbit equality with base, the factory's seed, rank and transfer device, seed range and budget, the replay stop, and W&B identity with offline mode and the resume rejection. The cookbook's "unmapped `can_act` / int32 tokens fail T18" is consistent with my M1, M4 and M18 results.
- Unverified by any test: "omitting it uses the safe ceiling" (M17) and Kaggriculture truncation through the trainer. The second is disclosed.

VERDICT: APPROVE WITH EDITS
