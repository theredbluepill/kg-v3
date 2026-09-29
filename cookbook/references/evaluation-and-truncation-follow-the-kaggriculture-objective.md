---
type: "Reference"
title: "Evaluation and truncation follow the Kaggriculture objective"
description: "Rebuild Tasks 3.2/3.3: raw-bank evaluation winners, truncation that keeps the economic reward, joint per-player clipping and value-mode guards, a per-evaluation seed in a checked int64 band (distinct per step within a run, not a global stream separator), and promotion telemetry logged only after promotion completes, all proven on the trainer seam with CPU TDD, a fake env and 1,344 Python passes; since the Task 3.4 merge the guards run through the registered Kaggriculture env config and every `configs/kaggriculture*.yaml`, and since Task 1.5 the evaluation env is the seeded native adapter, and since the Task 3.1 remainder (15ea55f) policy evaluation runs native games through the shared mapper and startup keeps rollout seeds below the evaluation band; a real (GPU) Kaggriculture run is still pending."
tags: ["kaggriculture-v3", "adaptation", "evaluation", "rewards"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.4.md"
  - resource: "repository:ops/rebuild-2026-09-29/plan.md"
  - resource: "repository:docs/kaggriculture-contract.md"
  - resource: "repository:python/owl/kaggriculture/evaluation.py"
  - resource: "repository:python/owl/train/config.py"
  - resource: "repository:python/owl/train/ppo.py"
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:engine_rs/src/lib.rs"
  - resource: "repository:configs/scaling_6m.yaml"
  - resource: "repository:tests/kaggriculture/test_evaluation.py"
  - resource: "repository:tests/kaggriculture/test_training_semantics.py"
  - resource: "repository:tests/owl/train/test_ppo.py"
  - resource: "repository:tests/scripts/test_run_ppo.py"
  - resource: "repository:configs/kaggriculture.yaml"
  - resource: "repository:python/owl/kaggriculture/config.py"
  - resource: "repository:README.md"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:ops/rebuild-2026-09-29/3.2-3.3-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.2-3.3-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.2-3.3-py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.2-3.3-r1-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.2-3.3-r1-py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-3.2-3.3-independent/review.md"
  - resource: "reference-branch:kg/reference-2026-09-29/scripts/run_ppo.py"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/train/ppo.py"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/train/config.py"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/gap-closure-2026-09-29/plan.md"
  - resource: "repository:python/owl/game.py"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.5.md"
---

# Evaluation and truncation follow the Kaggriculture objective

This note records rebuild Tasks 3.2 (game semantics) and 3.3 (evaluation) from `ops/rebuild-2026-09-29/plan.md`, serving the [[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]] and the [[../decisions/evaluation-preserves-generality-and-evidence|evaluation Decision]]. They implement reference lessons L1 (raw-bank winners), L2 (truncation reward) and L12 (a fresh seed per evaluation). The reference branch's `run_ppo.py`, `ppo.py` and `config.py` diffs and its gap-closure plan served as the oracle; the code was rebuilt, not copied. A cookbook search found the historical [[reward-reuse-preserves-objective-and-critic-semantics|reward-reuse Reference]] and [[shared-ppo-adapts-game-batches-without-a-second-loop|shared-PPO Reference]], which state the same rules for the reference branch. The [[ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|trainer-seams Reference]] covers the schema-generic mapping these changes sit on.

When these tasks were built, the native Kaggriculture environment (Tasks 1.4/1.5) didn't exist. So each rule is a game-dispatched function on the trainer seam, proven with Kaggriculture tensors or a fake environment. When these tasks were built, the Kaggriculture env and model configs weren't registered either; Task 3.1 registered the model and Task 3.4 the env (see "Merge with Tasks 3.1 and 3.4" below). Task 1.5 Stage 2 later un-skipped the integration test (see Limits).

## Contract and adaptation inventory

**L1: raw-bank evaluation outcome** (`python/owl/kaggriculture/evaluation.py`, `scripts/run_ppo.py`).
- `terminal_seat_banks(terminal_metrics)` returns `[bank_0, bank_1]` as float64. It requires the contract's `bank_0`, `bank_1` and `margin_0` to be present, finite and consistent, and checks the optional `winner` extension against the banks.
- `_evaluation_scores_and_metrics` decides each evaluation game. A Kaggriculture config (`cfg.env.obs_spec` is `KaggricultureObsConfig`) uses the raw banks. Orbit returns Isaiah's accumulated returns unchanged.
- `_record_eval_terminal_result` now takes `scores` (renamed from `returns`) with the same tie rule, so equal banks count as a draw.
- Kaggriculture evaluations also log candidate-relative `eval/candidate_bank`, `eval/last_best_bank` and `eval/candidate_bank_margin`.
- `_evaluate_games` sizes its seat assignments and returns from `obs.still_playing` rather than a literal 4. Orbit still gets 4 slots, and a two-seat game gets 2.

**L2: truncation keeps the economic reward** (`python/owl/train/ppo.py`).
- `_apply_truncation` now calls `_cut_truncated_envs_`. It marks the cut rows done and bootstraps them from the critic's value of the cut state. Its output shapes follow `dones`, so a two-seat game works; before, `OUTER_PLAYER_SLOTS = 4` was hard-coded.
- `_truncation_keeps_transition_reward(obs)` sets the reward rule: `ObsBatch` (Orbit) returns `False`, so Isaiah's zeroing stays; `KaggricultureObsBatch` returns `True`; any other type raises `TypeError`.

**Joint per-player clipping and value-mode guards** (`python/owl/train/config.py`).
- `_validate_kaggriculture_training` runs first in `FullConfig`'s cross-config validator when the env is Kaggriculture. It requires:
  - `env.reward_mode: win_loss`, because the critic reads `2p(self) − 1`
  - `rl.gamma: 1.0`, because `terminal_scale` bounds the shaped return only undiscounted
  - `rl.value_loss: mse`, because `winner_ce` needs `win_only`
  - `rl.ppo_clip_mode: per_player`, because a seat's turn is one joint autoregressive action
- These are `configs/scaling_6m.yaml`'s settings, so the guards encode Isaiah's recipe; `vf_clip_coef` stays free.

**L12 and promotion telemetry** (`scripts/run_ppo.py`).
- `_evaluation_seed(base_seed, env_steps)` mixes the base seed with a bijective 61-bit xor-shift/odd-multiply mix, xors in `env_steps`, mixes again and adds `2**62`. Both inputs must be in `[0, 2**61)`.
  - The result lies in `[2**62, 2**62 + 2**61)`. That band is non-negative (contract v4) and fits `engine_rs` `Game::new`'s `i64` seed, leaving `2**61` seeds of headroom for the consecutive seeds one evaluation env consumes.
  - Repeating an evaluation repeats its seed. For a fixed base seed, distinct evaluation steps get distinct seeds; for a fixed step, distinct base seeds do. That is what Task 3.3 requires.
  - It is not injective over `(base_seed, env_steps)` pairs, and it does not keep the consecutive seed ranges of different evaluations or runs apart: `(0, 0)` and `(1, 2131737497183550101)` collide, and `(0, 787325655728545358)` starts one seed later than `(0, 0)` (Codex verify-3.2-3.3-r1; pinned by a test).
  - Training streams (`base_seed + rank + k·world_size`) stay below the band only while that value is below `2**62`. Contract v4 requires only non-negative seeds. The Task 1.4 native ABI deliberately admits every nonnegative i64 seed with checked consumption (brief Q2), and it adds no training-only cap. The Task 1.5 factory adds no cap; the Task 3.1 remainder (`15ea55f`) enforces the bound at `run_ppo` startup: `env.seed` in `[0, 2**61)` and a step budget covering construction/reset, an autoreset and a truncation per step and one update of overshoot, failing before allocation ([[ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|trainer seams Reference]]).
- `_run_training_loop` passes `env_steps` to `_evaluate_against_last_best`, which passes it to `_evaluate_games` and the new `_create_eval_env`.
  - `_create_eval_env` builds Isaiah's unseeded Orbit `VectorizedEnv` with the same arguments as before.
  - For a Kaggriculture config it raised `NotImplementedError` at Task 3.3. Since Task 1.5 Stage 2 it returns an independent native adapter from `owl.game.create_env` with `base_seed=_evaluation_seed(cfg.env.seed, env_steps)`, rank 0 and world 1.
- Isaiah's evaluation count is kept: `cfg.env.n_envs` games on the main process, with no `eval_n_games` knob.
- Every evaluation log now carries `eval/games`, `eval/promoted` (1 or 0) and `eval/promotion_threshold` (0.7). The evaluation record is logged once, after the incumbent refresh, teacher update, promoted-checkpoint write and barrier complete (the reference's order), so a failed promotion leaves no `eval/promoted` record. Isaiah logged the evaluation before promoting; this moves that log, and a promotion failure now loses that step's evaluation metrics along with the run.

Docs: `README.md` (training) describes the telemetry, the Kaggriculture winner rule, the guards and the truncation difference. The Kaggriculture section of `docs/rl-api-specs.md` records the seed band.

## Verification

All checks ran on this version on the owner's Mac, CPU-only with `OMP_NUM_THREADS=2`.

| Step | Result | Receipt |
|---|---|---|
| Red | the two new Kaggriculture test modules fail at import; 17 run_ppo tests fail; the Orbit truncation characterization passes before the change | `3.2-3.3-red.log` |
| Green | 253 passed, 2 skipped (the explicit integration placeholders) over the new modules, `test_run_ppo.py`, `test_ppo.py` and `test_config.py` | `3.2-3.3-green.log` |
| Mutation | making Kaggriculture scores the shaped returns fails the L1 test (`[0.0, 2.0] != [1.5, 0.5]`); restored afterwards | `3.2-3.3-green.log` |
| `just py-prepare` | ruff, format, 3.11 syntax, mypy over 59 files, 1,341 passed / 6 skipped, docs freshness | `3.2-3.3-py-prepare.log` |
| Review repair red | after Codex verify-3.2-3.3-r1: the new promotion-failure test fails for an injected refresh failure and an injected promoted-checkpoint write failure (`eval/promoted` = 1 already logged) | `3.2-3.3-r1-red.log` |
| Review repair `just py-prepare` | ruff, format, 3.11 syntax, mypy over 59 files, 1,344 passed / 6 skipped, docs freshness | `3.2-3.3-r1-py-prepare.log` |

What the named tests show:
- **L1:** a two-seat fake env whose shaped return favors the incumbent in both games, while the banks give the candidate one win and one draw. Evaluation credits the candidate with 1.5 of 2 games and logs margins of 1,000 and 0. The candidate's seat is read from its actions, not assumed.
- **L2:** `make_obs` contract batches keep a 0.02/−0.02 economic reward on a cut row, and the GAE return there equals that reward plus the bootstrap. Orbit zeroes it. A `PPOTrainer` rollout on a truncating tiny Orbit env pins Isaiah's behavior end to end: the cut step has zero reward, done, a truncated flag and a bootstrap equal to `compute_value` of the cut state.
- **Clipping:** with 40 acting frames each moved by 0.01 nats, `per_player` clips the joint ratio e^0.4 (clip fraction 1, loss −1.2), while `per_entity` would clip none.
- **Guards:** each rejected setting fails with its own message. `scaling_6m` settings pass. An Orbit config with gamma 0.99 and `per_entity` still validates, while a Kaggriculture config with gamma 0.99 fails (originally an unvalidated Orbit copy carrying `KaggricultureObsConfig`; the shipped `configs/kaggriculture.yaml` since the Task 3.4 merge).
- **Seed:** 1,000 checkpoint steps and 4 base seeds × 256 adjacent steps get distinct starting seeds. Extreme inputs stay in the band with headroom, and out-of-band inputs are rejected. A characterization test pins the pair collision and the adjacent-start counterexample above, so the limit stays documented.
- **Telemetry:** a two-evaluation training loop logs `eval/games`, `eval/promoted` and `eval/promotion_threshold` at 0.7 and 0.69, and it promotes only at 0.7. When the incumbent refresh or the promoted-checkpoint write raises, the loop logs only the iteration's training metrics and no `eval/promoted`.

`cargo test` wasn't run, because no Rust changed. No training, evaluation or GPU run was performed.

## Merge with Tasks 3.1 and 3.4

These tasks forked before Task 3.1 registered `KaggricultureTransformerConfig` and Task 3.4 added the typed `KaggricultureEnvConfig` behind `FullConfig.env`'s `GameEnvConfig` discriminator. The integration merge (`kg/merge-trainer-lanes`) keeps both sides' guards:
- A validated Kaggriculture env now runs Task 3.4's `_validate_kaggriculture_rl` (gamma 1, no `winner_ce`) and then `_validate_kaggriculture_training` above (adds `rl.value_loss: mse` and `rl.ppo_clip_mode: per_player`; `env.reward_mode` is also fixed to `win_loss` by the env schema). Orbit configs run Isaiah's checks in `_validate_orbit_constraints`. The earlier `isinstance(env.obs_spec, KaggricultureObsConfig)` branch on Orbit's `EnvConfig` was unreachable for validated configs and was dropped.
- `_create_eval_env` still rejects Kaggriculture first, then narrows the Orbit env with `require_orbit_env` before building `VectorizedEnv`.
- The tests use the shipped configs instead of unvalidated `model_copy` configs: the full-config guard test and the raw-bank evaluation tests load `configs/kaggriculture.yaml`, and the formerly skipped `test_kaggriculture_yaml_configs_load_through_the_training_guards` now checks every `configs/kaggriculture*.yaml` loads and rejects gamma 0.99, `winner_ce`, `win_only` and `per_entity`.

## Limits and gaps

- **Config registration landed with Tasks 3.1 and 3.4** (see below). The guards and the Kaggriculture evaluation branches are now reachable through validated configs; `EnvConfig` itself stays Orbit-only.
- **Policy evaluation runs on CPU.** Since Task 1.5 Stage 2, `_create_eval_env` builds the seeded native adapter, and `test_kaggriculture_native_evaluations_draw_fresh_reproducible_worlds` runs: different `env_steps` draw different worlds, and a repeated evaluation reproduces its worlds and final banks with fixed legal actions. Since the Task 3.1 remainder (`15ea55f`), `_evaluate_games` maps Kaggriculture batches with the shared `ppo._obs_to_device` and `_select_actions`, and `test_kaggriculture_policy_evaluation_runs_native_games` plays native games with policies and scores raw banks; swapping `bank_0`/`bank_1` fails it (Claude's review mutation). Replay export stays Orbit-only: a Kaggriculture `rl.eval_replay_games > 0` fails at startup until Task 7.3.
- **Truncation through the trainer is untested for Kaggriculture.** Since `15ea55f`, `_obs_index` dispatches `KaggricultureActionMask`, but only the pure cut rule is tested; no test drives a Kaggriculture `_apply_truncation` through the native env. The Kaggriculture `truncate_envs` must also keep the transition buffers (contract).
- **Orbit keeps a quirk.** Its truncation zeroes the whole cut row, including a 4-player elimination reward (−1) earned on that same step. That is Isaiah's behavior, recorded here, not changed.
- **Telemetry is new for Orbit too.** Orbit runs now log `eval/games`, `eval/promoted` and `eval/promotion_threshold`. No selection behavior changed.
- **The seed is unexercised natively.** The seed band is checked against the `i64` signature and the contract's non-negativity. No native game has consumed a seed in `[2**62, 2**63)` yet. Per-game seed custody for replays remains Task 7.3's.
