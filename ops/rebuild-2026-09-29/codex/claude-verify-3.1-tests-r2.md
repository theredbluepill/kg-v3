Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 3.1 remainder: test non-vacuity and coverage review (r2)

## Scope

- Target: branch `kg/rebuild-3-1` at `2413c9e` (the verify r1 fix commit on top of `7a55e81` and Codex's `15ea55f`), diff `49a4835...HEAD`. I read the whole fix commit (code, tests, configs, docs) and re-read the r1 code paths it touches.
- Spec read: plan Task 3.1, `docs/rl-api-specs.md` (trainer seam), `docs/kaggriculture-contract.md` (rollout/evaluation seeds), README launch section, the r1 reports, and the fix logs under `ops/rebuild-2026-09-29/3.1-rest2/verify-r1-fixes/`.
- Lens: whether the new and un-skipped tests discriminate: teacher T18/T19b, the native smoke and truncation, evaluation seat crediting, mapping/storage, resume seeds, W&B mode and presets. Also whether any doc or cookbook claim goes untested, and whether the stateless-policy rules hold.
- Method: I created a scratch worktree with `git -C kg-v3-t31 worktree add --detach /tmp/cv-3.1-tests-r2 HEAD` and copied the fixtures in from `kg-v3/tests/fixtures`. I also copied in `python/owl/rs.abi3.so` from `kg-v3-t31`, built 23:37 on 29 Sep. `49a4835...HEAD` contains no Rust change, so no rebuild was needed. I ran the `kg-v3-t31` venv interpreter with `PYTHONPATH` set to the scratch `python/`, and confirmed that `owl`, `owl.rs`, `owl.train.ppo` and `scripts.run_ppo` all resolved to the scratch copy. Everything ran on CPU with OMP_NUM_THREADS=2. No training, GPU or W&B was used.
- Cleanup: the scratch worktree has been removed. `kg-v3-t31` is still at `2413c9e` with 0 tracked modifications; its one pre-existing untracked directory, `ops/.../verify-3.1-rest2-r1/`, is not mine.
- Harness incident, disclosed: the session scratchpad is shared with sibling verifiers, and another agent overwrote my first harness file (`scratchpad/mut.py`) between two runs. One invocation therefore executed a foreign script, and its "V*" output is discarded here. All mutation results below come from my own harness: N1–N15 ran before the overwrite, and N16–N20 ran from the uniquely named `scratchpad/cv31tests-r2/mutate_cv31tests.py`. Afterwards I confirmed that `/tmp/cv-3.1-runppo-r2`, `/tmp/cv-3.1-storage-r2`, my scratch copy and `kg-v3-t31` all had 0 tracked modifications.

## Checks (scratch copy, 2413c9e)

| Check | Result |
|---|---|
| `uvx ruff check python scripts tests tools` | All checks passed |
| `uvx ruff format --check` | 137 files already formatted |
| `scripts/check_python_311_syntax.py` | rc 0 |
| `mypy python/ scripts/` | No issues in 70 source files |
| `scripts/check_doc_freshness.py` | rc 0 |
| Targeted files: mapping, smoke, `test_run_ppo`, logging, benchmark, configs | 430 passed, 1.83 s, peak RSS 417 MB |
| The four un-skipped teacher tests (T18, trainer checkpoint, T19b resume, T19b fresh launch) | 4 passed, 0 skipped, 383 MB |
| Mutation baseline set: the above plus evaluation, training semantics, `test_ppo.py` and the four teacher tests | 549 passed |
| Non-Kaggriculture suite (`-m "not slow" tests --ignore=tests/kaggriculture`) | 1291 passed, 3 skipped, 37.7 s, 714 MB |
| Kaggriculture suite, 25 files, with `test_model_heads.py` and `test_teacher.py` split in halves | 1094 passed, 3 skipped (1 CUDA fence, 2 observe) |
| Shard RSS | Maximum was 1,028,407,296 B (981 MiB), in `test_teacher.py` first half. Every other shard was at or below 996 MB. |

Total: 1291 + 1094 = **2385 passed, 6 skipped**, which matches the fix commit's claim of 2,385/6. I didn't run the monolithic `just prepare`, which trips the 1 GB watchdog.

## Mutations (20 applied, each reverted; `-x` on the 549-test set)

| # | Mutation | Result | First killing test |
|---|---|---|---|
| N1 | Disable the "checkpoint changed during startup" guard (`run_ppo.py:282`) | **SURVIVED** | none |
| N2 | Resume base `seed + 3*S` instead of `4*S` | CAUGHT | `test_kaggriculture_resume_bases_clear_worst_case_launches` |
| N3 | Seed budget drops the `- 2*start_env_steps` term | CAUGHT | `test_kaggriculture_resume_budget_counts_the_resumed_start` |
| N4 | Drop the "resume past the budget" raise | CAUGHT | same |
| N5 | Candidate bank always seat 0 and incumbent seat 1 | CAUGHT | `test_kaggriculture_policy_evaluation_runs_native_games` |
| N6 | Kaggriculture eval scored by shaped returns instead of raw banks | CAUGHT | `test_kaggriculture_evaluation_decides_winners_by_raw_banks` |
| N7 | `main` passes `start_env_steps=0` to `_kaggriculture_step_limit` on resume (`run_ppo.py:188`) | **SURVIVED** | none |
| N8 | Truncation drops Kaggriculture's transition reward (`keep_transition_reward=False`, `ppo.py:1070`) | **SURVIVED** | none |
| N9 | Truncation bootstrap zeroed | CAUGHT | `test_truncation_bootstraps_through_the_native_trainer` |
| N10 | 2-rank preset back to `eval_replay_games: 8` | CAUGHT | `test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation[kaggriculture_2rank...]` |
| N11 | `PPOTrainer` `winner_ce` guard off | CAUGHT | `test_trainer_rejects_winner_ce_for_kaggriculture` |
| N12 | Offline mode accepted with `--log-mode debug` | CAUGHT | `test_validate_args_rejects_offline_wandb_with_debug_logging` |
| N13 | Eval terminal dict drops `winner` (r1's M5b survivor) | CAUGHT | `test_kaggriculture_evaluation_credits_the_candidates_seat_bank` |
| N14 | Resume pre-read uses the last-best checkpoint's steps | CAUGHT | `test_main_kaggriculture_resume_starts_a_disjoint_seed_stream` |
| N15 | Candidate margin sign flipped | CAUGHT | `test_kaggriculture_evaluation_decides_winners_by_raw_banks` |
| N16 | `main` drops the `wandb_mode` kwarg (r1 M6) | CAUGHT | `test_main_forwards_wandb_mode_and_the_default_step_limit` |
| N17 | Fresh launch rollout start 1 instead of 0 | CAUGHT | same |
| N18 | `train/max_entities` from `still_playing` (r1 M11) | CAUGHT | `test_max_entity_count_is_one_seats_actors_for_kaggriculture` |
| N19 | Kaggriculture action tokens always copied to step 0 | CAUGHT | `test_kaggriculture_storage_and_every_mapping_preserve_schema` |
| N20 | Resume skips activating the last-best teacher | CAUGHT | `test_run_ppo_resume_restores_the_teacher_from_checkpoint_last_best` (T19b) |

Result: 17 of 20 caught. Every r1 survivor I re-applied (M5b as N13, M6 as N16, M11 as N18) is now killed. So are the new resume base, budget and pre-read, the seat crediting, the preset replay stop, T19b and the truncation bootstrap. The three survivors are listed as P3 findings below.

## Prior findings re-checked

| Prior finding | Status | Evidence |
|---|---|---|
| P2 Resume replays the first launch's worlds | **RESOLVED** | See the note after this table. |
| P2 No test of which seat's bank evaluation credits | **RESOLVED** | See the note after this table. |
| P2 `main` does not forward `--wandb-mode` | **RESOLVED** | See the note after this table. |
| P2 Presets set `eval_replay_games: 8`; stale config headers | **RESOLVED** | See the note after this table. |

1. **Resume replays the first launch's worlds.** `main` reads the checkpoint's `env_steps` before allocation (`run_ppo.py:179`) and builds the env at `seed + 4*S0`.
   - Arithmetic check: I re-derived the bound independently. A launch from S0 to S1 draws, per rank, at most `2*n_envs + 2*(S1-S0)/ws` seeds at stride `ws`, so every seed stays below `base + 2*global_envs + 2*(S1-S0)`. `env_steps_per_iteration = horizon*n_envs*world_size` is global (`run_ppo.py:332`), so any checkpoint written after at least one update satisfies `S1 - S0 >= global_envs`, which gives `4*S1 >= 4*S0 + 2*ge + 2*(S1-S0)`.
   - Checkpoint targets: resume targets are only final or numbered checkpoints, written after updates. The last-best checkpoint's `env_steps=start_env_steps` is never a resume target.
   - Tests: the native-stream disjointness, arithmetic-chain and main-forwarding tests all pass. N2, N3, N4 and N14 are caught.
   - Residual gaps: N1 and N7 survive (P3 below).
2. **No test of which seat's bank evaluation credits.** `test_kaggriculture_evaluation_credits_the_candidates_seat_bank` pins the candidate to seat 1 with `bank_0 < bank_1`. It asserts wins `[1,0]`, `candidate_bank == bank_1` and margin `+3000`, and checks that `winner` reaches the scorer. N5, N13 and N15 are caught. The fix log records r1's M12b as caught.
3. **`main` does not forward `--wandb-mode`.** `test_main_forwards_wandb_mode_and_the_default_step_limit` runs the real tiny CPU launch up to `_run_training_session` and asserts `wandb_mode == "offline"` and the default step limit. N16 is caught. A new `_validate_args` rule (offline requires `--log-mode wandb`) is tested, and N12 is caught.
4. **Presets set `eval_replay_games: 8`; stale config headers.** `configs/kaggriculture_{2,4,8}rank.yaml` now set `eval_replay_games: 0` with a Task 7.3 comment, and all four headers now say "builds the native env and trains through the shared PPO path". The startup test runs the unmodified presets, with no `-o` override, and N10 is caught. README drops the override from the launch example.

## Stateless-policy constraints

- The rollout buffer adds no hidden-state or history field.
- The smoke asserts `trainer._hidden_state is None` after two native updates. The new truncation test runs the stateless-only truncation path; `PPOTrainer` raises if a hidden state exists with truncation enabled.
- Evaluation (`_eval_actions_for_assignments_and_hidden`) gives both models the same device observation. It uses the seat assignment only to select actions (`_select_actions`), never as a model input, embedding, head, loss or checkpoint field.
- In the diff's added lines, "hidden", "opponent", "identity", "assignment", "seat_id" and "player_id" appear only in one widened return-type annotation.
- The W&B identity is `kg-v3`/`ppo`, with no opponent label.
- The resume seed rule uses only the checkpoint's `env_steps`, not policy state.
- I found no violation.

## Findings

There are no P1 or P2 findings. The P3 findings are test gaps and polish.

- **P3: the "checkpoint changed during startup" guard is untested.**
  - Where: `scripts/run_ppo.py:280-289`.
  - Evidence: N1 survived. The fix report and commit message present this guard as a startup check. If it regressed, a checkpoint rewritten between the pre-read and `trainer.load_checkpoint` could still slip through. In that case the seed base would come from a different step than the weights.
  - Fix: in the `main` resume test, patch `_checkpoint_env_steps` to return 1,000 and the (fake) trainer's `load_checkpoint` metadata to return 2,000, then assert `RuntimeError` matching `changed during startup`.
- **P3: `main`'s forwarding of the resume start into the seed budget is untested.**
  - Where: `scripts/run_ppo.py:184-189`.
  - Evidence: N7 survived. `_kaggriculture_step_limit(start_env_steps=...)` is tested directly, but nothing checks that `main` passes the resumed step. If that forwarding were dropped, the documented rule "a resume past the budget fails at startup" (contract, rl-api-specs) would stop holding. The limit would also be computed from 0 rather than from the resumed base. That only matters near about 2^61 steps, so it isn't a practical risk.
  - Fix: extend `test_main_kaggriculture_resume_starts_a_disjoint_seed_stream`, which uses `env_steps=1_000`. Either capture `max_env_steps` and compare it against `_kaggriculture_step_limit(..., start_env_steps=1_000)`, or add a case with a saved `env_steps` past the budget and assert `ValueError` matching `resumed at env step`.
- **P3: the end-to-end Kaggriculture truncation test does not check the transition reward.**
  - Where: `tests/kaggriculture/test_training_smoke.py:138` and `python/owl/train/ppo.py:1070`.
  - Evidence: N8 survived. The rule "truncation keeps the transition's economic reward" (rl-api-specs, lesson L2) is unit-tested in `test_training_semantics.py` through `_cut_truncated_envs_`. The trainer-path wiring that this diff first opens to Kaggriculture (`_apply_truncation` widened to `GameObsBatch`) is not tested. Passing `False` there would silently drop the economic reward on every cut row.
  - Fix: in `test_truncation_bootstraps_through_the_native_trainer`, record the env's raw step rewards, for example with a thin wrapper around `env.step`. Assert that `rollout.rewards` on truncated rows equals them, and that at least one is nonzero.
- **P3: the resume pre-read loads the whole checkpoint a second time on every rank.**
  - Where: `scripts/run_ppo.py:179`, via `_checkpoint_env_steps` at `:781`.
  - Evidence: `_checkpoint_env_steps` runs `torch.load(..., weights_only=False)` of the full checkpoint (model, optimizer, scheduler) on CPU. Every rank does this before allocation, then `trainer.load_checkpoint` loads it again. `_latest_resume_checkpoint` already uses the same pattern, so this is a startup cost and not a correctness issue.
  - Fix: optional. Use a `mmap=True` load, or store `env_steps` in a small sidecar, if resume startup time or host RAM on 8-rank pods becomes visible.
- **P3: resource note.**
  - Evidence: `test_teacher.py` first half peaked at 981 MiB (1.028 GB decimal) and `test_base_generics_typing.py` at 996 MB. Both are close to the 1 GB bound. This is pre-existing and not caused by this diff; the four un-skipped teacher tests alone peak at 383 MB.

## Claims checked against tests

- **Held, with a killing mutation:**
  - resume base `seed + 4*S0`, including chains and world-size changes (arithmetic);
  - the native seed-stream disjointness;
  - the resume budget counting from the resumed base;
  - the fresh launch starting at `cfg.env.seed`;
  - seat-1 bank crediting and winner pass-through;
  - raw-bank winners;
  - offline W&B forwarding, and offline requiring `--log-mode wandb`;
  - the default safe step limit;
  - presets launching unmodified;
  - the `winner_ce` rejection;
  - `train/max_entities` as one seat's actors;
  - `{n}p_rate` up to the seat count;
  - the mixed-spec rollout-buffer guards;
  - unknown action bundles raising;
  - per-step action storage;
  - seat-preserving GAE;
  - T18 and T19b teacher paths;
  - the truncation bootstrap being finite and nonzero through the native trainer.
- **Untested:**
  - the startup-change guard (N1);
  - `main`'s forwarding of the resume start into the budget (N7);
  - Kaggriculture keeping the transition reward through the trainer's truncation path (N8).
- **By construction, not tested:** "a `--load-model-weights` fresh launch starts at `cfg.env.seed`". The pre-read branch applies only to `ResumeLaunch`.

VERDICT: APPROVE WITH EDITS
