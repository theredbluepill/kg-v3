Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Task 3.1 remainder: verification of rollout storage and mapping (r1)

## Scope

- Branch `kg/rebuild-3-1` at `7a55e81` (worktree `/Users/poonszesen/kg-v3-t31`).
  I reviewed the diff `49a4835...HEAD` in a detached scratch worktree at
  `/tmp/cv-3.1-storage-r1`, which has since been removed.
- Lens: rollout storage and mapping in `python/owl/train/ppo.py`. That covers
  shapes and dtypes, mask mapping, the seat and actor layout, contiguous CPU
  transfer, the Orbit byte-equality claim against `49a4835`, and the GAE and
  advantage path for 2 seats. I also checked the seat mixing and bank scoring
  in `scripts/run_ppo.py` evaluation, because they share the seat layout.
- Specs read:
  - `ops/rebuild-2026-09-29/plan.md` Task 3.1 (I11: schema-generic helpers, no shims)
  - `docs/rl-api-specs.md`, the Task 3.1 sections
  - `docs/kaggriculture-contract.md`, the Trainer seam section
  - the Codex report `task-3.1-rest2-report.md`

## What I confirmed by reading the code

- **Kaggriculture storage.** `_PPORolloutBuffer` builds the observation storage
  from a one-row `allocate_observation_buffers` prototype, so the native
  allocator stays the single owner of the schema. It adds only the leading
  horizon dimension. The shapes and dtypes are:
  - `can_act`: bool `[H,E,2,252]`
  - tokens: int64 `[H,E,2,252,12]`
  - lengths: int64 `[H,E,2]`
  - logp, values, rewards, dones, truncated and bootstrap: `[H,E,2]`
  - entity_logp: `[H,E,2,252]`
  These match `docs/rl-api-specs.md` and the model's output shapes
  (`per_player_entity [E,2,252]`, `values [E,2]`).
- **Masks.**
  - `_policy_mask`: `can_act [N,T,2,252]` passes through `flatten(3)` and
    `any(-1)` to become `[N,T,2]`. It is correct for 2 seats.
  - `_policy_entity_mask`: takes the `ndim == still_playing.ndim + 1` branch
    and gives `[N,T,2,252]`, which is also correct.
- **Aliasing and off-by-one in collection.** `_obs_to_device` clones on CPU.
  So `self._obs` never aliases the adapter's reusable buffers, and
  `write_step(obs=self._obs)` after `env.step` stores the pre-step observation.
  There is no off-by-one there.
- **CPU transfer.** `_actions_to_cpu` for `KaggricultureActions` keeps the
  dtype (no cast) and makes the tensors C-contiguous. Evaluation's
  `_select_actions` also returns contiguous CPU int64 tensors.
- **GAE.** GAE and the value/policy masks are shape-generic over the last
  (seat) axis. No Kaggriculture-specific seat indexing is added in the update.
- **Winner layout.**
  - The Kaggriculture winner log-probs are `[B*T,2,2]`.
  - The update skips the Orbit `view_as` for them. That is correct under the
    admitted `value_loss='mse'`, and `config.py:68` rejects `winner_ce` for
    Kaggriculture.
- **Evaluation seats.**
  - Candidate/last-best actions are mixed per `[env, seat]` over the whole
    `[252,12]` program.
  - Terminal raw banks are read by seat index through `terminal_seat_banks`
    and `_candidate_bank_metrics`.

## Checks, run in the scratch copy on Mac CPU

I built `owl.rs` with `uv sync --frozen` (maturin). The ruff, format and mypy
commands are the static phases of `just py-prepare`. I did not run the
monolithic `just py-prepare`: Codex found that its single pytest run trips the
1 GB watchdog, so I ran the tests as shards.

| Check | Result |
|---|---|
| `uvx ruff check python scripts tests` | All checks passed |
| `uvx ruff format --check python scripts tests` | 137 files already formatted |
| `uv run mypy python scripts` | Success, no issues in 70 source files (max RSS 890,650,624 B) |
| `pytest tests/owl/train/test_ppo_observation_mapping.py tests/kaggriculture/test_training_smoke.py` | 192 passed |
| `pytest -m "not slow" tests/owl tests/scripts tests/tools` | 1276 passed, 3 skipped (flash-attn CUDA x2, qnnpack) |
| `pytest -m "not slow" tests/kaggriculture` minus native_env/teacher/model_heads | 630 passed, 3 skipped (CUDA fence, pinned memory x2) |
| `pytest tests/kaggriculture/test_native_env.py` | 345 passed |
| `pytest tests/kaggriculture/test_teacher.py tests/kaggriculture/test_model_heads.py` | 117 passed |
| Shard total | **2368 passed, 6 skipped**. This matches the Codex report's 2368/6. Each shard's max RSS was about 296 to 320 MB. |

## Mutations

Each mutation was a one-line source replacement, reverted after its run. The
test set for the `ppo.py` mutations was:

- `tests/owl/train/test_ppo_observation_mapping.py`
- `tests/kaggriculture/test_training_smoke.py`
- `tests/owl/train/test_ppo.py`
- `tests/scripts/test_run_ppo.py`
- `tests/kaggriculture/test_teacher.py`
- `tests/kaggriculture/test_training_semantics.py`

The `run_ppo.py` mutations used `tests/scripts/test_run_ppo.py` and
`tests/kaggriculture/test_evaluation.py`.

| # | Mutation | Result |
|---|---|---|
| M1 | Remove the "Kaggriculture observations require Kaggriculture actions" guard (`ppo.py:286`) | **SURVIVED** (469 passed) |
| M2 | Remove the "Orbit observations require Orbit actions" guard (`ppo.py:315`) | **SURVIVED** (469 passed) |
| M3 | Don't copy `lengths` in `_copy_actions_time_step` | CAUGHT (smoke: GrammarReplayError) |
| M4 | Write tokens to `(step+1) % H` (time off-by-one) | CAUGHT (smoke: GrammarReplayError) |
| M5 | Swap seats in stored tokens (`flip(1)`) | CAUGHT (mapping test :670) |
| M6 | Drop `.contiguous()` in `_actions_to_cpu` | CAUGHT (contiguity test) |
| M7 | Remove the `isinstance(..., ObsBatch)` winner-view guard | CAUGHT (smoke: RuntimeError at view_as) |
| M8 | `train/max_entities` from `shop_mask` instead of `actor_mask` | **SURVIVED** |
| M9 | Swap seats in stored rewards for 2-seat games | **SURVIVED** |
| M10 | Remove the `KaggricultureActionMask` branch of `_map_action_mask` | CAUGHT (mapping: ValidationError) |
| M11 | Leave `lengths` unmapped in `_map_action_bundle` | CAUGHT (contiguity test) |
| M12 | Swap seats in stored `entity_logp` for 2-seat games | **SURVIVED** (the field is unused under the required per_player clip) |
| M13 | Orbit `fleet_target` fill -1 changed to 0 | CAUGHT (base-49a4835 equality test) |
| M14 | Swap seats in stored values for 2-seat games | **SURVIVED** |
| M15 | Swap seats in stored logp for 2-seat games | CAUGHT (smoke: target_kl_exceeded_total) |
| M16 | `lengths + 1` in storage | CAUGHT (smoke: GrammarReplayError) |
| E1 | Swap the candidate and incumbent bank seats in `_candidate_bank_metrics` | CAUGHT |
| E2 | Invert the seat selection for Kaggriculture tokens in `_select_actions` | CAUGHT |
| E3 | Flip the seat selection for Kaggriculture lengths in `_select_actions` | CAUGHT |

Totals: 19 mutations, 13 caught, 6 survived. The survivors are:

- M1 and M2: two new construction guards with no test.
- M8: telemetry.
- M9, M12 and M14: synthetic seat swaps that fire only on 2-seat tensors, in
  shared code that preserves seat indices. No test gives the 2-seat trainer
  path a seat-identity oracle.

## Findings

No P1. No P2.

### P3 findings

1. **Untested new guards.**
   - Where: `python/owl/train/ppo.py:285-288` and `ppo.py:314-315`.
   - Issue: the two cross-game spec guards in `_PPORolloutBuffer.__init__`
     have no test. M1 and M2 both survived.
   - Fix: add two `pytest.raises(ValueError, match=...)` cases to
     `test_ppo_observation_mapping.py`. One pairs `KaggricultureObsConfig`
     with `ActionPureConfig`, the other pairs `EntityBasedConfig` with
     `KaggricultureActionConfig`.
2. **No seat-identity oracle on the 2-seat trainer path.**
   - Where: `tests/kaggriculture/test_training_smoke.py`,
     `tests/owl/train/test_ppo_observation_mapping.py:612`.
   - Issue:
     - The smoke test checks finiteness and counters only.
     - The mapping test writes the same observation and action at every step,
       so it has no time or seat oracle for rewards, values or dones through
       `write_step` → `segment_major` → `_compute_gae`.
     - M9 and M14 survived.
   - Risk: the risk is low today because the code is shared and seat-generic.
   - Fix: write distinct per-seat and per-step reward/value marks into a
     Kaggriculture `_PPORolloutBuffer`. Then assert that `segment_major()` and
     `compute_gae` give the hand-computed per-seat advantages, with seat 0 ≠
     seat 1.
3. **Silent no-op fallthrough in `_copy_actions_time_step`.**
   - Where: `python/owl/train/ppo.py:2305-2328`.
   - Issue: the generic `__dataclass_fields__` loop was replaced by an
     explicit `if/elif` chain with no final `else`. An action bundle of a new
     type that matches itself would pass the type check and copy nothing.
     That contradicts "unsupported ... types fail explicitly"
     (`docs/rl-api-specs.md`) and is less schema-generic than the base (I11).
   - Fix: add `else: raise TypeError(f"unsupported action bundle {type(dst).__name__}")`.
4. **Winner layout is guarded only by the config.**
   - Where: `python/owl/train/ppo.py:1378-1384`.
   - Issue: the winner `view_as` is skipped based on the observation type. For
     Kaggriculture the tensor is `[B*T,2,2]`. If `PPOTrainer` is built
     directly with `value_loss='winner_ce'`, bypassing the `FullConfig` guard
     at `config.py:68`, the loss can broadcast silently when `B=1, T=2`.
   - Fix: in `PPOTrainer.__init__`, reject `value_loss == 'winner_ce'` when
     `env.obs_spec` is `KaggricultureObsConfig`. Alternatively, reshape to
     `[B,T,2,2]` and raise.
5. **The base-equality oracle needs git history at test time.**
   - Where: `tests/owl/train/test_ppo_observation_mapping.py:694-706`.
   - Issue: the `base_ppo` fixture shells out to
     `git show 49a4835:python/owl/train/ppo.py` from the pytest cwd. In a tree
     without `.git` or without that commit (a shallow clone, or an rsynced
     pod copy), all 18 `test_orbit_storage_and_actions_equal_base_49a4835`
     cases error.
   - Fix: pass `-C` with the repo root derived from `__file__`. Then either
     `pytest.skip` with an explicit reason when the commit is unavailable, or
     vendor a frozen copy of the base helpers under `tests/fixtures`.
6. **Telemetry and storage nits.**
   - Where: `python/owl/train/ppo.py:750-758` and `ppo.py:502`.
   - Issues:
     - `train/max_entities` for Kaggriculture counts the own and rival
       `actor_mask` only, not the tiles, shops or market tokens. The metric's
       meaning differs from Orbit's `entity_mask`, and no test pins it (M8).
     - `entity_logp [H,E,2,252]` float32 is stored and transposed every
       iteration. Kaggriculture's required `per_player` clip never reads it:
       about 16.5 MB per rank at 128 envs × horizon 64. This is harmless.
     - `_player_count_rates` still logs `3p_rate`/`4p_rate` (always 0).
   - Fix:
     - Document or rename the metric's Kaggriculture meaning.
     - Optionally skip entity_logp storage when the clip mode is not
       `per_entity`.

### Byte-equality claim

This claim is verified. The test runs the actual `49a4835` module source and
compares the following for all three Orbit action families × cross-attention
on/off × seeds:

- field-by-field dtype, shape, contiguity/stride and values of the storage;
- `_copy_*_time_step`, segment-major, index, flatten, `_obs_to_device` and
  `_actions_to_cpu`.

M13 confirms that the test detects a one-value storage change. Its limit: the
base module runs against the current `owl.rl` and `owl.model` imports, so the
claim covers `ppo.py` helper behaviour only. The CUDA `non_blocking` transfer
path is not exercised.

## Process notes

- `uv sync` in the scratch copy briefly created an untracked build directory,
  `/Users/poonszesen/kg-v3-t31/target-cv-none`. I passed a mistaken
  `CARGO_TARGET_DIR`. I deleted it right after the build. No tracked file in
  the target worktree was touched.
- After I removed the scratch worktree, `git -C /Users/poonszesen/kg-v3-t31
  status --short` showed only the untracked
  `ops/rebuild-2026-09-29/codex/verify-3.1-rest2-r1/`, which was there before
  I started. HEAD is still `7a55e81`.
- Not covered: CUDA or pinned-memory DMA, compiled execution, multi-rank
  training, and a live W&B sync. None of these can run on a Mac CPU.

VERDICT: APPROVE
