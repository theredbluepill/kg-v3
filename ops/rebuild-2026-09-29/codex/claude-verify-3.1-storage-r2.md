Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 3.1 remainder: storage and mapping verification, round 2

## Scope

- Branch `kg/rebuild-3-1` at `2413c9e`, worktree `/Users/poonszesen/kg-v3-t31`.
- I reviewed the diff `49a4835...HEAD` in a detached scratch worktree at
  `/tmp/cv-3.1-storage-r2`, which has since been removed.
- Lens: rollout storage and mapping in `python/owl/train/ppo.py`. That covers:
  - shapes and dtypes
  - mask mapping
  - the seat and actor layout
  - contiguous CPU transfer
  - the Orbit byte-equality claim against `49a4835`
  - the GAE and advantage path for 2 seats
- I also re-checked the four r1 findings from the other lenses: resume seeds,
  evaluation seat credit, `--wandb-mode` forwarding and the presets.
- Specs read:
  - `docs/rl-api-specs.md` (the Task 3.1 and trainer-seam sections)
  - `docs/kaggriculture-contract.md` (the resume seed rule at lines 255-265)
  - the Codex report `task-3.1-rest2-report.md`
  - my r1 report `claude-verify-3.1-storage-r1.md`
  - fix commit `2413c9e`

## What I confirmed by reading the code

- **Storage.**
  - `_PPORolloutBuffer` builds Kaggriculture observation storage from a one-row
    `allocate_observation_buffers` prototype. The tensor shapes are:
    - tokens: int64 `[H,E,2,252,12]`
    - lengths: int64 `[H,E,2]`
    - `can_act`: bool `[H,E,2,252]`
    - logp, values, rewards, dones, truncated and bootstrap: `[H,E,2]`
    - entity_logp: `[H,E,2,252]`
  - These match the model's `_policy_outputs` shapes.
  - The two cross-game spec guards now have tests.
  - `_copy_actions_time_step` now raises `TypeError` on an unknown bundle.
- **Masks.**
  - `_policy_mask` gives `[N,T,2]`.
  - `_policy_entity_mask` takes the `ndim + 1` branch and gives `[N,T,2,252]`.
- **Aliasing and truncation.**
  - On CPU, `rewards.to(device)` and `dones.to(device)` alias the adapter's
    persistent `_rewards` and `_dones` buffers, and `_cut_truncated_envs_`
    writes into them in place.
  - This is safe because native `truncate_envs` never republishes transitions.
    The comment at `src/kaggriculture/env.rs:653` says "truncate never
    publishes transitions" (`pending.transition` is `None` there).
    `write_step` copies before the next `step`.
  - `truncate_envs` rewrites the observation buffers in place, so the
    following `_copy_obs_to_device_` picks up the reset rows.
- **Per-seat dones.** Kaggriculture dones are per game. The native code fills
  both seats (`env.rs:621`), and truncation sets whole rows.
- **Resume seed arithmetic (checked by hand).**
  - Rank `r` consumes `base + r + k*ws`.
  - Per rank, `k < 2n + 2·T_rank`, and `env_steps` is global
    (`horizon * n_envs * world_size`). A launch from `S0` to `S1` therefore
    stays below `base + 2g + 2(S1 - S0)`.
  - With `base = seed + 4·S0`, the next base `seed + 4·S1` clears that bound
    whenever `S1 - S0 >= g_prev`. One update at horizon ≥ 1 guarantees this, so
    the rule holds when the world size changes.
  - A checkpoint saved with no update (`Δ = 0`) reuses a base that trained
    nothing, so it is still safe.
  - The budget `(FLOOR - seed - 2g - 2·S0)//2` matches the bound
    `seed + 2g + 2·S0 + 2·S_end`.
  - Orbit is unchanged, because `rollout_start_env_steps = 0` for non-Kaggriculture runs.

## Checks, run in the scratch copy on Mac CPU

I built `owl.rs` with `uv sync` (maturin, release) and copied the missing
fixtures (`generation`, `orbit_wars_replays`) from `/Users/poonszesen/kg-v3/tests/fixtures`.

| Check | Result | Max RSS |
|---|---|---|
| `uvx ruff check python scripts tests` | All checks passed | – |
| `uvx ruff format --check python scripts tests` | 137 files already formatted | – |
| `uv run mypy python scripts` | Success, no issues in 70 source files | 893,517,824 B |
| mapping + smoke + semantics | 227 passed | 351,387,648 B |
| `test_run_ppo` + `test_ppo` + `test_logging` + `test_evaluation` | 230 passed | 418,234,368 B |
| `test_teacher` + `test_configs` | 101 passed | 1,057,947,648 B (over 1 GB) |
| `-m "not slow" tests/owl tests/scripts tests/tools` | 1291 passed, 3 skipped | 700,514,304 B |
| `-m "not slow" tests/kaggriculture`, minus native_env, teacher and model_heads | 632 passed, 3 skipped | 1,101,496,320 B (over 1 GB) |
| `tests/kaggriculture/test_native_env.py` | 345 passed | 314,605,568 B |
| `tests/kaggriculture/test_model_heads.py` | 65 passed | 1,102,282,752 B (over 1 GB) |
| `-k base_49a4835` (the Orbit byte-equality oracle, run and not skipped) | 18 passed | – |

- Unique total: 1291 + 632 + 345 + 65 + 52 (teacher) = **2385 passed, 6 skipped**
  out of 2391 collected. No tests are marked slow. This matches the commit's
  claim of 2,385 passed and 6 skipped.
- Three shards measured 1.06 to 1.10 GB under `/usr/bin/time`. That RSS
  includes uv and the pytest parent. It exceeded the 1 GB guidance slightly;
  no watchdog tripped.
- I did not run the monolithic `just py-prepare`.

## Mutations

Each mutation was a one-string replacement, reverted after its run. Each run
used `pytest -x` over these test files:

- `test_ppo_observation_mapping`
- `test_training_smoke`
- `test_training_semantics`
- `test_ppo`
- `test_run_ppo`
- `test_evaluation`
- `test_logging`

A "seat swap" means `x.flip(-1) if x.shape[-1] == 2 else x`.

| # | Mutation | Result |
|---|---|---|
| S1 | Swap seats in stored rewards (`write_step`) | CAUGHT (`test_kaggriculture_rollout_keeps_each_seat_and_step_through_gae`) |
| S2 | Swap seats in stored values | CAUGHT (same GAE test) |
| S3 | Swap seats in stored dones | SURVIVED. **Equivalent**: Kaggriculture dones are equal for both seats (`env.rs:621`) |
| S4 | Zero the stored truncated flags | CAUGHT (`test_truncation_bootstraps_through_the_native_trainer`) |
| S5 | Swap seats in stored bootstrap values | **SURVIVED** |
| G1 | Remove the `winner_ce` guard in `PPOTrainer` | CAUGHT (`test_trainer_rejects_winner_ce_for_kaggriculture`) |
| G2 | Make the unknown-bundle `TypeError` a no-op | CAUGHT (`test_copy_actions_time_step_rejects_an_unknown_action_bundle`) |
| G3 | Compute `train/max_entities` from `shop_mask` | CAUGHT (`test_max_entity_count_is_one_seats_actors_for_kaggriculture`) |
| G4 | `_player_count_rates` back to `OUTER_PLAYER_SLOTS` | CAUGHT (`test_player_count_rates_follow_the_games_seats`) |
| G5 | `_policy_mask` ignores `can_act` | CAUGHT (`test_ppo.py::test_discrete_target_rollout_buffer_and_policy_mask`) |
| R1 | Resume base = `seed` | CAUGHT (`test_kaggriculture_resume_bases_clear_worst_case_launches`) |
| R2 | Resume base = `seed + 2*S` | CAUGHT (same test) |
| R3 | Budget ignores `start_env_steps` | CAUGHT (`test_kaggriculture_resume_budget_counts_the_resumed_start`) |
| R4 | Remove the "checkpoint changed during startup" check (`run_ppo.py:282`) | **SURVIVED** |
| R5 | Remove the "resumed past the budget" check | CAUGHT (budget test) |
| E1 | r1 M12b: swap `bank_0`/`bank_1`, negate the margin and flip the winner in the eval terminal dict | CAUGHT (`test_kaggriculture_evaluation_credits_the_candidates_seat_bank`) |
| E2 | r1 M6: stop `main` forwarding `wandb_mode` | CAUGHT (`test_main_forwards_wandb_mode_and_the_default_step_limit`) |

Totals: 17 mutations. 14 were caught. Of the 3 survivors, 1 is equivalent
(S3), which leaves 2 real survivors (S5, R4).

## Re-check of the r1 findings

1. **P2: resume replays the first launch's worlds. RESOLVED.**
   - `run_ppo.py:178-190` reads the checkpoint's `env_steps` before
     allocation.
   - `run_ppo.py:217` passes `_kaggriculture_rollout_base_seed(seed, S0)` to
     `create_env`.
   - The budget counts from that base, and a resume past it raises at startup.
   - Evidence:
     - The hand-checked arithmetic above.
     - A native-stream disjointness test
       (`test_kaggriculture_resume_seeds_follow_every_first_launch_seed`).
     - A test that `main` passes the resumed base
       (`test_main_kaggriculture_resume_starts_a_disjoint_seed_stream`).
     - Mutations R1, R2, R3 and R5 were caught.
     - The rule is documented at `docs/kaggriculture-contract.md:255-265`
       and in `docs/rl-api-specs.md:871,1020`.
2. **P2: no test of which seat's bank evaluation credits. RESOLVED.**
   - `test_kaggriculture_evaluation_credits_the_candidates_seat_bank` pins the
     candidate to seat 1 with `bank_0 < bank_1`.
   - It asserts wins `[1,0]`, `candidate_bank == bank_1`, margin `+3000` and
     that the native winner reaches the scorer.
   - The consistent seat swap (E1, the old M12b) is now caught.
3. **P2: `--wandb-mode` forwarding untested. RESOLVED.**
   - `test_main_forwards_wandb_mode_and_the_default_step_limit` runs the real
     tiny launch with `--wandb-mode offline` and asserts
     `wandb_mode == 'offline'` at `_run_training_session`.
   - E2 (the old M6) is caught.
   - Also added: `--wandb-mode offline` now requires `--log-mode wandb`, and
     `test_validate_args_rejects_offline_wandb_with_debug_logging` covers that.
4. **P2: presets set `eval_replay_games: 8` and headers are stale. RESOLVED.**
   - `configs/kaggriculture.yaml:59`, `_2rank.yaml:70`, `_4rank.yaml:70` and
     `_8rank.yaml:71` all set `eval_replay_games: 0`.
   - All four headers now say the config "trains through the shared PPO path
     (Task 3.1)". The stale stop is gone.

My r1 P3 items are resolved too:

- The cross-game guards now have tests.
- `_copy_actions_time_step` has an `else: raise`.
- `PPOTrainer` rejects `winner_ce`.
- The base oracle uses `git -C` and skips with an explicit reason.
- `train/max_entities` and `{n}p_rate` are documented, tested and follow the
  game's seats.
- The storage and time tests now use distinct actions for each step and seat.

## Findings

No P1. No P2.

### P3 findings

1. **No seat-order oracle for truncation bootstrap values (S5 survived).**
   - Where: `python/owl/train/ppo.py:2873` (`bootstrap_values[rows] = row_values`),
     tested by `tests/kaggriculture/test_training_smoke.py`
     (`test_truncation_bootstraps_through_the_native_trainer`).
   - Issue:
     - The smoke test checks only that bootstrap values are finite and
       non-zero.
     - The GAE seat test writes no truncation.
     - Swapping the two seats' bootstrap values survives every test.
     - With the swap, each seat would bootstrap from the opponent's critic
       value at a time-limit cut.
   - Risk: low. The code is shared and does not reorder seats.
   - Fix: in `test_kaggriculture_rollout_keeps_each_seat_and_step_through_gae`,
     add one step with `truncated` set and per-seat `bootstrap_values`, for
     example `[0.25, -0.75]`. Assert the hand GAE for each seat. Or assert in
     the smoke test that `rollout.bootstrap_values[t]` equals
     `model.compute_value(cut_obs)` seat by seat.
2. **The new "checkpoint changed during startup" guard has no test (R4 survived).**
   - Where: `scripts/run_ppo.py:280-288`.
   - Issue: the fix commit claims this guard. Removing it passes all runner
     tests.
   - Fix: add a `main`-level test that patches `_checkpoint_env_steps`, or
     `trainer.load_checkpoint`'s metadata, so the two reads disagree (for
     example 1000 then 2000). Assert `RuntimeError` with
     `match="changed during startup"`.
3. **A Kaggriculture resume loads the whole checkpoint twice on every rank.**
   - Where: `scripts/run_ppo.py:781-783` (`_checkpoint_env_steps` uses
     `torch.load(..., weights_only=False)` on the full checkpoint), called from
     `run_ppo.py:179`. `trainer.load_checkpoint` then loads it again.
   - Issue: the first load reads only `env_steps`, but it pulls the full model
     and optimizer state into CPU memory on every rank. For large checkpoints
     on 8 ranks, that is a transient spike in host memory and startup time.
     It is correct, just wasteful.
   - Fix: none needed now. Optionally reuse the first loaded dict for
     `trainer.load_checkpoint`, or record `env_steps` in a small sidecar.
     Either way, document the cost.

### Byte-equality claim

This still holds. The 18 `test_orbit_storage_and_actions_equal_base_49a4835`
cases ran against `git show 49a4835:python/owl/train/ppo.py`; none were
skipped. The Orbit-facing changes in `2413c9e` keep Orbit behaviour the same:

- `_max_entity_count` still uses `entity_mask.sum(-1).max()` for Orbit.
- `_player_count_rates` still gives `range(1, 5)` for Orbit's 4-seat
  `still_playing`.

The r1 limit still applies: the base module runs against the current `owl.rl`
and `owl.model` imports.

## Process notes

- `git -C /Users/poonszesen/kg-v3-t31 worktree remove --force /tmp/cv-3.1-storage-r2`
  succeeded. After that, `git -C /Users/poonszesen/kg-v3-t31 status --short`
  showed only the untracked `ops/rebuild-2026-09-29/codex/verify-3.1-rest2-r1/`,
  which was there before I started. HEAD is still `2413c9e`. No tracked
  modifications.
- Not covered: CUDA and pinned-memory DMA, compiled execution, multi-rank
  training (the seed-stream proof for world size > 1 is arithmetic plus a
  one-rank native test), and a live W&B sync.

VERDICT: APPROVE WITH EDITS
