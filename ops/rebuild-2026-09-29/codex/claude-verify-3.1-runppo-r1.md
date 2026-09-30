Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 3.1 remainder: run_ppo lens, adversarial verification r1

## Scope

- Branch `kg/rebuild-3-1` at `7a55e81df9fb28268bf8da1ce3e45966fe2ae49e`, diff `49a4835...HEAD`
  (two commits: `15ea55f` Codex implementation, `7a55e81` Claude review).
- Lens: `scripts/run_ppo.py`. Checked env construction (seed, rank, world, `transfer_device`),
  the training seed budget below `2**62` against the evaluation band, the evaluation winner from
  raw banks, the teacher lifecycle, W&B project/identifiers and `--wandb-mode`, startup guards
  (`eval_replay_games > 0`), resume semantics, and whether removed stops were left half-wired. I
  also read `python/owl/train/logging.py`, `config.py`, `__init__.py` and
  `scripts/benchmark_checkpoints.py`.
- Spec read: plan Task 3.1 (and 3.3/3.5 context); `docs/rl-api-specs.md` and
  `docs/kaggriculture-contract.md` diffs; the 1.5 brief's "Exact 1.5 / 3.1 boundary" rows
  (243–252) and smoke section; Codex report `codex/task-3.1-rest2-report.md`. I read the native
  seed stream (`src/kaggriculture/env.rs`: construction, `reset(mask)`/truncate and terminal
  autoreset each reserve at most one seed per selected env) to check the budget arithmetic.
- Out of lens: ppo.py storage and mapping internals (reviewed by the sibling storage verifier).

## Checks (scratch worktree `/tmp/cv-3.1-runppo-r1`, detached at HEAD, CPU only)

| Check | Result |
|---|---|
| `uv sync` + `maturin develop --release` (CARGO_BUILD_JOBS=2) | built in 34 s |
| Fixtures | copied `generation/`, `orbit_wars_replays/` from main checkout |
| `pytest tests/scripts/test_run_ppo.py tests/owl/train/test_logging.py` | **133 passed** in 0.55 s (max RSS 365 MB) |
| `pytest` smoke + teacher + mapping + benchmark + configs (5 files) | **332 passed** in 8.76 s (max RSS 1.05 GB, slightly above the 1 GB target for one combined shard; no watchdog trip) |
| `ruff format --check` / `ruff check` (python scripts tests) | 137 files formatted / all checks passed |
| `mypy python/owl scripts` | Success, 69 source files (Codex reported 70 with `python/ scripts/` paths) |
| `just docs-fresh` | "No doc updates required" (the committed tree was clean, so this check was trivial and does not independently confirm freshness) |
| Monolithic `just py-prepare` | not run (known 1 GB watchdog trip; replaced by the shards above) |

Seed-budget arithmetic, checked by hand against `env.rs`: rank `r` reserves seeds
`base + r + k*ws`. Per rank there are at most `2n` construction+reset seeds plus 2 per env per
transition. The maximum seed is therefore `< base + 2*global_envs + 2*S`, where `S` counts
global env steps. With `S <= safe_limit + update_steps - 1`, `_kaggriculture_step_limit` keeps
this below `2**62` and keeps `env_steps < 2**61` for `_evaluation_seed`. Confirmed correct.

## Mutations (each applied alone, tests rerun, source restored)

| # | Mutation | Result |
|---|---|---|
| M1 | drop the `- (update_steps - 1)` overshoot | CAUGHT (`test_kaggriculture_seed_budget_covers_...`) |
| M2 | seeds per step 2 → 1 (`// 2` removed) | CAUGHT (same) |
| M3 | base-seed bound `2**61` → `2**62` | CAUGHT (`test_main_rejects_kaggriculture_seed_before_allocation[2**61]`) |
| M4 | remove main's `eval_replay_games > 0` guard | CAUGHT (`test_main_rejects_kaggriculture_replay_before_allocation`) |
| M5 | rollout `rank=0` | CAUGHT (`test_main_kaggriculture_rollout_factory_uses_rank_seed_and_transfer_device`) |
| M6 | `main` stops forwarding `wandb_mode=args.wandb_mode` | **SURVIVED** (133 passed) |
| M7 | `main` passes `args.max_env_steps` instead of the computed limit | **SURVIVED** (133 passed) |
| M8 | remove the WandbLogger offline+resume guard | CAUGHT (`test_offline_wandb_rejects_resume_before_initialization`) |
| M9 | Kaggriculture W&B `mode` not forwarded | CAUGHT (`test_kaggriculture_wandb_init_uses_v3_identity_and_mode`) |
| M10 | remove the `_validate_args` offline-resume guard | CAUGHT (`test_resume_offline_fails_before_reading_checkpoint`) |
| M11 | eval seed ignores `env_steps` | CAUGHT (`test_kaggriculture_eval_factory_arguments`) |
| M12 | swap bank_0/bank_1 and negate margin in the eval terminal dict | CAUGHT, but only through `terminal_seat_banks`' winner cross-check |
| M12b | same swap plus a consistently remapped `winner` | **SURVIVED** (133 passed; also 174 passed with `test_evaluation.py`, `test_training_smoke.py`, `test_training_semantics.py` added) |
| M13 | eval env `transfer_device=cpu` | CAUGHT (`test_kaggriculture_eval_factory_arguments[cuda]`) |
| M14 | rollout `pin_memory=True` forced | CAUGHT (factory test) |
| M15 | `_kaggriculture_step_limit` always passes through | CAUGHT (5 tests) |
| M16 | W&B name drops the `ppo-` prefix | CAUGHT (2 tests) |
| M17 | remove the `_evaluate_games` replay guard | **SURVIVED** (133 passed) |

17 mutations: 13 caught and 4 survived (M6, M7, M12b, M17). The survivors appear below as
findings.

## Findings

### P1

None found.

### P2

1. **Resume replays the first launch's training worlds.** `scripts/run_ppo.py:206`. Every
   launch, resume included, builds the rollout env with `base_seed=env_config.seed`, and the
   native seed counter starts again at `k=0`. Checkpoints carry no seed state:
   `python/owl/train/ppo.py` has no `seed`/`seed_state` in checkpoint I/O. A resumed Kaggriculture
   run therefore trains on exactly the same world sequence as the original launch did from step
   0. With a changed world size, the sequence is permuted but still overlaps. Isaiah's Orbit env
   samples its own games, so this is new, Kaggriculture-only behavior. No test or doc covers it.
   `rl-api-specs.md` only says the budget holds "throughout the launch". This is not a crash, but
   it silently reduces data diversity across resumes and is a reproducibility claim nobody
   checks. **Fix:** store `env.seed_state()` (or the per-rank `k`) in the checkpoint and restart
   the stream from it on resume. Alternatively, derive a disjoint resume base from
   `start_env_steps` inside the budget. Either way, include `start_env_steps` in
   `_kaggriculture_step_limit`, add a resume test that asserts the first post-resume seeds differ
   from the first-launch seeds, and document the rule in `docs/kaggriculture-contract.md`
   "Trainer seam".
2. **No test checks which seat's bank is credited in evaluation.** `scripts/run_ppo.py:1575`
   (terminal dict) with `tests/scripts/test_run_ppo.py:3176`. M12b swaps seat banks, negates the
   margin and remaps `winner` consistently, and it survives every test in scope. The native
   evaluation test uses a candidate and last-best with identical weights and random seats. Its
   oracle (`expected_wins` from `candidate_bank_margin`) comes from the same possibly-wrong data,
   so it cannot detect crediting the wrong seat. That is the L1 raw-bank outcome, and getting it
   wrong would invert promotion decisions. **Fix:** add a test with a fake or monkeypatched
   Kaggriculture eval env (or a patched `_assign_eval_models`) that pins the candidate to seat 1
   and returns terminal banks `bank_0 < bank_1`. Assert `stats.wins == [1, 0]`,
   `candidate_bank == bank_1` and `candidate_bank_margin == bank_1 - bank_0`.
3. **Nothing tests that `main` forwards `--wandb-mode`.** `scripts/run_ppo.py:317`. M6 (dropping
   the kwarg) survives. `test_kaggriculture_session_forwards_offline_mode_and_shared_metrics`
   calls `_run_training_session` directly, and the parse tests stop at `_parse_args`. If the flag
   is dropped, an offline launch on a keyless host silently goes online, against the documented
   telemetry contract. **Fix:** extend the startup-sentinel test so `main` with
   `--wandb-mode offline` reaches a patched `_run_training_session`/`create_logger`, and assert
   `wandb_mode == "offline"`.
4. **The shipped multi-rank presets fail at startup, and their headers are stale.**
   `configs/kaggriculture_2rank.yaml:69`, `_4rank.yaml:69` and `_8rank.yaml:70` set
   `eval_replay_games: 8`, which the new guard rejects (`run_ppo.py:179`). Every production
   preset therefore needs `-o rl.eval_replay_games=0`. The README documents this, but the early
   smoke and Phase 6 launches read these files. All four Kaggriculture configs (lines 6/11/11/12)
   still say "run_ppo checks its GEMM workload, then stops until the ... trainer game seam
   (Task 3.1) land[s]". That is now false. **Fix:** set `eval_replay_games: 0` in the three
   presets, with a comment that Task 7.3 restores it, and rewrite the four headers. The YAML was
   out of Codex's allowed surface, so this is a Claude follow-up.

### P3

5. **Nothing tests that an omitted `--max-env-steps` defaults to the safe ceiling.**
   `run_ppo.py:319` / `:1683`. M7 survives. The practical impact is nil, because the ceiling is
   about `2**61` steps, but the README and contract state the behavior. **Fix:** assert in the
   factory startup test that `_run_training_session` receives the computed limit.
6. **The duplicate replay guard in `_evaluate_games` is untested.** `run_ppo.py:1465`. M17
   survives because main's guard is tested. Without this guard, the recorder path reaches the
   bare `assert not isinstance(env, KaggricultureVectorizedEnv)` statements, which disappear
   under `python -O`. **Fix:** call `_evaluate_games(..., replay_games=1)` on a Kaggriculture
   config in a test, or replace the asserts with explicit `TypeError`s.
7. **`--wandb-mode offline` with `--log-mode debug` is silently ignored.** `run_ppo.py`
   `_validate_args`. **Fix:** reject the combination, or document that the flag applies only to
   W&B.
8. **The Orbit rollout and evaluation still call `VectorizedEnv` directly.** `run_ppo.py:213`,
   `_create_eval_env`. The 1.5 brief row 243 says "3.1 replaces both with create_env". The
   implementation keeps the game branch in the runner, and `rl-api-specs.md` documents this. It
   is behavior-neutral, but it deviates from the brief. The Codex report line "Both native
   factories receive transfer_device" refers to the two Kaggriculture constructions only.
   **Fix:** either route Orbit through `create_env(base_seed=0, ...)` as the brief specifies, or
   record the deviation in the Task 3.1 cookbook record.
9. **Evaluation seats come from the unseeded global torch RNG.** `run_ppo.py:1826`. This is
   inherited Isaiah behavior, not a 3.1 regression. With small `n_envs` (for example 32 per rank
   at 8 ranks → 32 eval games), the candidate's seat split is unbalanced and not reproducible,
   although `_evaluation_seed` reproduces the worlds. **Fix (optional, later):** give
   `_assign_eval_models` a generator seeded from the evaluation seed, or alternate seats by env
   index, and report the per-seat denominator.

## What held up

- Rollout `create_env` receives `base_seed=cfg.env.seed`, the distributed rank and world size,
  `pin_memory` and `transfer_device=device`, and tests pin all of them. The evaluation env uses
  rank 0 / world 1, `_evaluation_seed`, and the eval device.
- The seed budget math is correct and conservative. It is enforced before the run directory,
  env and model exist, and tests cover the base-seed bound and excess limits.
- Evaluation scores Kaggriculture games from terminal raw banks through `terminal_seat_banks`,
  which cross-checks margin and winner. Ties are draws.
- Teacher lifecycle: the fixed teacher, the last-best init (validated against the student spec),
  resume restore, and warm-start activation all go through the shared path. The four previously
  skipped teacher tests now run and pass. Mixing game families raises `TypeError`.
- W&B: Kaggriculture runs use project `kg-v3`, job type and group `ppo`, tags
  `kaggriculture-v3`/`ppo`, and name `ppo-<run_dir>`, with the explicit mode. Orbit's init
  arguments are unchanged. Offline resume is rejected both in the CLI and in the logger.
- Removed stops are fully removed in the code. `require_orbit_env` is gone, its benchmark uses
  are replaced by explicit local guards, and no code path still raises a Task 3.1 stop. The only
  stale text left is in the YAML headers (finding 4).

## Cleanup

The scratch worktree `/tmp/cv-3.1-runppo-r1` was removed with `git worktree remove --force`.
`/Users/poonszesen/kg-v3-t31` shows 0 tracked modifications at `7a55e81`. Nothing was committed
or pushed.

VERDICT: APPROVE WITH EDITS
