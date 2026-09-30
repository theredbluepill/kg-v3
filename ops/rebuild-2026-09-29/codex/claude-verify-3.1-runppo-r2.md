Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 3.1 remainder, verify r2: `scripts/run_ppo.py` lens

## Scope

- Branch `kg/rebuild-3-1` at `2413c9e` (worktree `/Users/poonszesen/kg-v3-t31`, not modified). Diff `49a4835...HEAD`, with a focus on the r1-fix commit `2413c9e`.
- Lens: env construction (seed, rank, world, transfer_device); the training seed range below 2**62 against the evaluation band; evaluation winner by raw banks; the teacher lifecycle; W&B project, identifiers and the offline flag; startup guards (eval_replay_games > 0); resume semantics; removed stops.
- Spec read: plan Task 3.1, `docs/rl-api-specs.md` (trainer seam and environment rows), `docs/kaggriculture-contract.md` (rollout and evaluation paragraphs), README training section and the four Kaggriculture presets.
- Scratch copy: `/tmp/cv-3.1-runppo-r2` (detached at `2413c9e`). It used its own uv venv plus the `owl.rs` .so copied from the t31 worktree. There is no Rust change in `49a4835...HEAD`, and the last `src/` commit is `d0d65f7`, so the copied .so matches the source. The scratch copy was removed afterwards. `git -C /Users/poonszesen/kg-v3-t31 status --untracked-files=no` shows no tracked modifications, and HEAD is still `2413c9e`.

## Checks (all run in the scratch copy, CPU only)

| Check | Result |
|---|---|
| `scripts/check_python_311_syntax.py` | rc 0 |
| `ruff check python scripts tests` | All checks passed |
| `ruff format --check python scripts tests` | 137 files already formatted |
| `mypy python/owl scripts` | Success, 69 source files (peak 849 MiB) |
| `scripts/check_doc_freshness.py` | rc 0 |
| `pytest -m "not slow" tests --ignore=tests/kaggriculture` | 1,291 passed, 3 skipped (peak 673 MiB) |
| `tests/kaggriculture/test_*.py`, one file per process | 1,094 passed, 3 skipped (CUDA fence 1, observe 2) |
| Total | 2,385 passed / 6 skipped, the same as the fix commit's claim |
| Lens shard (run_ppo, logging, observation mapping, smoke, evaluation, configs, semantics, teacher) | 483 passed (mutation baseline) |

No training, GPU or live W&B run was done.

## Mutations (17 in the batch + 2 targeted re-checks)

A first batch run was contaminated and thrown away. The session scratchpad is shared with sibling agents, and another agent's `mut.py` overwrote mine mid-run: failures were attributed to the wrong mutations and one mutation was left applied in the scratch copy. I restored the scratch files with `git checkout` and reran everything serially from a uniquely named script. Only the clean rerun is reported.

| ID | Mutation | Result | Killing test |
|---|---|---|---|
| V1 | Startup "checkpoint env_steps changed" RuntimeError disabled | **SURVIVED** | none |
| V2 | `main` passes `start_env_steps=0` to `_kaggriculture_step_limit` | **SURVIVED** | none |
| V3 | "resumed past budget" guard removed | caught | `test_kaggriculture_resume_budget_counts_the_resumed_start` |
| V4 | `start_env_steps < 0` guard removed | survived (see note) | none |
| V5 | candidate/incumbent bank index swapped | caught | `test_kaggriculture_evaluation_decides_winners_by_raw_banks` |
| V6 | eval scores from shaped returns, not banks | caught | same |
| V7 | eval env seeded with the training seed | caught | `test_kaggriculture_eval_factory_arguments[cpu]` |
| V8 | W&B Kaggriculture mode forced online | caught | `test_kaggriculture_wandb_init_uses_v3_identity_and_mode[offline]` |
| V9 | W&B project `orbit-wars` for Kaggriculture | caught | `...identity_and_mode[online]` |
| V10 | `main` eval-replay startup guard off | caught | `test_main_rejects_kaggriculture_replay_before_allocation` |
| V11 | rollout `rank=0` | caught | `test_main_kaggriculture_rollout_factory_uses_rank_seed_and_transfer_device[cpu]` |
| V12 | resume + offline guard removed | caught | `test_resume_offline_fails_before_reading_checkpoint` |
| V13 | resume base `seed + 3*S0` | caught | `test_kaggriculture_resume_bases_clear_worst_case_launches` |
| V14 | budget subtracts `1*S0` not `2*S0` | caught | `test_kaggriculture_resume_budget_counts_the_resumed_start` |
| V15 | W&B name without `ppo-` | caught | `...identity_and_mode[online]` |
| V16 | `min(seed_steps, 2**61-1)` cap dropped | survived, **equivalent** | n/a |
| V17 | `main` never reads the resume step | caught | `test_main_kaggriculture_resume_starts_a_disjoint_seed_stream` |
| M12b | r1 seat swap: banks swapped, margin negated, winner remapped | caught | `test_kaggriculture_evaluation_credits_the_candidates_seat_bank` (the only failure in 175) |
| M6 | `main` drops `wandb_mode=` | caught | `test_main_forwards_wandb_mode_and_the_default_step_limit` (the only failure in 142) |

Notes:
- V16 is equivalent. With `seed >= 0` and `global_envs >= 1`, `seed_steps = (2**62 - seed - 2G - 2S0)//2 <= 2**61 - 1`, so the `min` can never pick its second argument.
- V4 is covered upstream: `_checkpoint_nonnegative_int` rejects a negative saved `env_steps`, so the guard is defensive only.

## Seed arithmetic (checked by hand against the native code)

- The native `SeedStream` (`src/kaggriculture/env.rs:35-57`) hands rank r the seeds `base + r + j*W`. Seeds are drawn only at construction (n), at `reset` (n) and for terminal or truncated rows.
- `_apply_truncation` excludes rows that ended naturally (`& ~env_done`), so each env draws at most one reseed per transition. The documented bound of two per transition is conservative.
- `env_steps` counts global steps (`horizon * n_envs * world_size`). With per-rank draws `k_r <= 2n + 2T_r`, the highest seed is `base + r + (k_r - 1)W < base + 2G + 2(S1 - S0)`.
- The next resume base `seed + 4*S1` clears that bound whenever `S1 - S0 >= G`. Checkpoints are written only after a full update (`>= G*horizon` steps). The one exception is `checkpoint_final.pt` at `S0` after a zero-update launch, and that launch trained on nothing.
- The budget check `S_end <= seed_steps` keeps every seed strictly below 2**62 and keeps the evaluation `env_steps` below 2**61. The evaluation band is `[2**62, 2**62 + 2**61)`, so it is disjoint from training.

## Prior findings (r1)

1. **P2: resume replays the first launch's worlds. RESOLVED.** `scripts/run_ppo.py:176-189` reads the checkpoint's `env_steps` before any allocation, and `:214-219` builds the env with `base_seed = seed + 4*env_steps`.
   - The budget counts from the resume start (`:1719-1735`).
   - Tests: `test_kaggriculture_resume_bases_clear_worst_case_launches` (chain arithmetic), `test_kaggriculture_resume_seeds_follow_every_first_launch_seed` (real native stream, disjoint sets) and `test_main_kaggriculture_resume_starts_a_disjoint_seed_stream` (main forwards the base). Mutations V13, V14 and V17 are killed.
   - Documented in `docs/kaggriculture-contract.md:253-267`, `docs/rl-api-specs.md` (trainer seam) and README.
   - Two parts of the fix are untested (P3-1 below).
2. **P2: no test of which seat's bank evaluation credits. RESOLVED.** `_SeatPinnedNativeEvalEnv` pins the candidate to seat 1 with `bank_0 < bank_1`. The test asserts `wins == [1,0]`, `candidate_bank == 4000`, `candidate_bank_margin == 3000` and that the winner reaches the scorer. The consistent seat swap M12b is now killed.
3. **P2: `--wandb-mode` forwarding untested. RESOLVED.** `test_main_forwards_wandb_mode_and_the_default_step_limit` runs `main` with `--wandb-mode offline` through the real tiny CPU env and model construction, then asserts `wandb_mode == "offline"`. M6 is killed. `_run_training_session` → `create_logger` forwarding is covered by `test_kaggriculture_session_forwards_offline_mode_and_shared_metrics`. `--wandb-mode offline` now also requires `--log-mode wandb` (`:691-692`), and V8 is killed.
4. **P2: presets set `eval_replay_games: 8` and headers are stale. RESOLVED.**
   - The 2-, 4- and 8-rank presets now set `eval_replay_games: 0` (lines 70, 70 and 71), with a comment that Task 7.3 restores 8.
   - All four headers now say run_ppo "builds the native env and trains through the shared PPO path (Task 3.1)".
   - `tests/kaggriculture/test_configs.py:198-199` pins 0 against scaling_6m's 8.
   - Nothing in `python/ scripts/ docs/ README.md configs/ tests/` still contains "stops until", "cannot run Kaggriculture yet", "Task 3.1 lands" or `require_orbit_env`. The helper was deleted, and `benchmark_checkpoints.py` has its own explicit Orbit-only error, so no stop is left half-wired.

## Other lens items checked with no finding

- **Env construction:** `create_env` gets `rank=distributed.rank`, `world_size=distributed.world_size`, `pin_memory` from the config and `transfer_device=device` (V11 killed). Evaluation uses rank 0, world 1 and `_evaluation_seed(cfg.env.seed, env_steps)` (V7 killed). Orbit keeps its `VectorizedEnv` call.
- **Evaluation winner:** decided by `terminal_seat_banks` on raw banks, and equal banks give a draw (V5 and V6 killed). The action-type guards are on both branches.
- **Startup guard:** `eval_replay_games > 0` fails in `main` before the run dir, env or model exists (V10 killed). `_evaluate_games` guards it again before building the eval env.
- **Teacher lifecycle:** `require_orbit_env` was removed from both teacher loaders. `_validate_teacher_specs` still requires matching action specs and a compatible observation spec. Resume re-arms a `last_best` teacher from `checkpoint_last_best.pt`, and a fresh launch keeps Isaiah's semantics. The teacher test file passes (52).
- **W&B:** project `kg-v3`, job_type and group `ppo`, tags `[kaggriculture-v3, ppo]`, name `ppo-<run_dir>`. The explicit `mode=` is passed for Kaggriculture. Resume requires online, and `WandbLogger` also refuses offline with a resume id (V8, V9, V12 and V15 killed).

## Findings

**P3-1: two new resume guards in `main` are unverified; mutations V1 and V2 survive.**
- Where: `scripts/run_ppo.py:280-288` and `:184-189`.
- The startup check "resume checkpoint changed during startup" (V1) can be disabled with all 483 lens tests passing. So can passing `start_env_steps=0` to the budget in `main` (V2).
- The fix commit and README say "a resume with no budget left fails at startup", but only the helper is tested, not `main`'s wiring.
- The practical risk is low: overrunning the budget needs about 2**60 env steps.
- Fix: in `test_main_kaggriculture_resume_starts_a_disjoint_seed_stream`'s pattern, write a metadata checkpoint whose `env_steps` exceeds the fresh safe limit and assert `main` raises "resumed at env step" before `create_env`. Add a test that patches `PPOTrainer.load_checkpoint` or `_checkpoint_env_steps` to return different `env_steps` and asserts the `RuntimeError`.

**P3-2: a `--load-model-weights` fresh launch replays the source checkpoint's worlds, and the docs do not warn about it.**
- Where: `scripts/run_ppo.py:176-182` and `:305-318`. Docs: `docs/rl-api-specs.md` (trainer seam: "A `--load-model-weights` fresh launch starts at `cfg.env.seed`").
- A fresh launch that loads weights keeps the checkpoint's `env_steps` (`:318`) but seeds rollouts from `cfg.env.seed`. With `--load-model-weights-mode model_and_optimizer`, a run that is in effect a continuation retrains on the exact world sequence its weights already saw, unless the operator changes `env.seed`.
- The seed budget stays safe (the launch consumes less than a fresh launch), so this is a documented-behaviour gap, not a correctness bug.
- Fix: either apply `_kaggriculture_rollout_base_seed(seed, start_env_steps=checkpoint.env_steps)` to load-weights launches too, or state the consequence in the contract and README with the remedy (`-o env.seed=<new>`).

**P3-3: a resume loads the full checkpoint up to three times on every rank.**
- Where: `scripts/run_ppo.py:781-783` (`_checkpoint_env_steps`, `torch.load(..., weights_only=False)` of the whole file). It is called from `_latest_resume_checkpoint` when a final checkpoint exists, then again at `:179`, and then `trainer.load_checkpoint` loads it a third time.
- This is harmless at the 6M-parameter scale and only a startup cost.
- Fix (optional): reuse the metadata that `_resolve_resume_launch` already read, or load only once and pass the metadata through.

No P1 or P2 findings.

VERDICT: APPROVE WITH EDITS
