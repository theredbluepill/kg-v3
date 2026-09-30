Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Review r2: reward term M (cash-difference potential) on `kg/rebuild-reward-margin`

- Branch HEAD reviewed: `25412a7`. That is the core `87beaf0`, the docs and cookbook in `8b45577`, and the r1 fixes in `25412a7`. The base is `kg/isaiah-gap-closure` `111ae7d`.
- I worked in a scratch detached worktree (`<scratchpad>/margin-r2`) with its own `.venv` and its own `maturin develop --release` build. I edited nothing in `/Users/poonszesen/kg-v3-margin` except this file, which is not committed.
- Owner requests, verbatim, 2026-09-30:
  - "can we relaunch (before anchor) run, fix the reward, 0.5 Cash Diff (add this in) + 0.5 (Terminal loss 1/-1/0)?"
  - "implement the new rewrad first before we revisit the cha22 anchor setup."
  - "can we accelerrate?"

## Bottom line

There are no P1 or P2 findings. The reward code is unchanged since r1 and is correct as specified. I re-verified all of the following against this version:
- the math and operation order
- zero-sum
- telescoping and the equal start
- the reset and auto-reset boundary
- validation and `terminal_scale`
- byte-identity with the term disabled
- Python/Rust parity
- the preset against J/2's effective config

All four r1 findings (P2-1, P2-2, P3-1, P3-2) are resolved.

Two P3 items remain. Neither changes a reward bit under the preset, and neither blocks the launch or the landing:
- a non-discriminating Rust test (the same class as r1's M9);
- one missing gap line in the Decision.

The relaunch is safe on this code, and the branch can land.

## Checks I ran on `25412a7`

| Check | Result |
| --- | --- |
| `cargo test --lib kaggriculture::` | 134 passed, 3 ignored. This includes the 7 `margin_*` / autoreset tests. |
| `pytest tests/kaggriculture -m "not slow"` | 1327 passed, 4 skipped (CUDA and BC-env skips). This is one more than r1, from the new order-pin test. |
| `cargo fmt --check`, `ruff check`, `ruff format --check`, `mypy python/ scripts/` (77 files), `scripts/check_doc_freshness.py` | All clean. `docs-fresh` reports "No doc updates required". |
| Cookbook lint (`.claude/hooks/cookbook-lint.mjs`) | The Decision and both edited References lint clean. |
| Cookbook `repository:` sources | Every `repository:` source in those three notes exists at this HEAD. |
| Cookbook contract | `cookbook/log.md` is prepended for both commits, and `decisions/index.md` and `references/index.md` are updated. No v3 board exists yet, so no board edit is due. |
| Resolved preset vs J/2's effective config (`/Users/poonszesen/kg-v3-runs/control-J2-4rank-20260930/rundir/config.yaml`), compared with a recursive walk of `FullConfig.from_file(...).model_dump()` | Only differences: `econ_shaping` 0.2→0.0, bank 0/100000/0 and margin 0.5/50000/0.5 (all absent in J/2), and `runtime.n_runtime_gpus` 4 vs 1. The trainer fills `n_runtime_gpus` from `world_size`. Everything else is equal: muon 1e-4, adamw 5e-6, `checkpoint_freq` 10M, `native_threads` 4, `gamma` 1.0, `gae_lambda` 0.9, `vf_coef` 2, teacher, model and seed. |
| J/2 warm-start checkpoint | `sha256(checkpoint_final.pt)` = `80e5667e…32e161dd`, which matches the brief. |

## Verified properties

- **Formula and order.** `src/kaggriculture/reward.rs:135-143` computes `(w_m * margin / S_m).max(-c_m).min(c_m)`. The product comes first, then the quotient, then the clamp, and the doc comment states that order. The Python `margin_score` at `python/owl/kaggriculture/rewards.py:209-220` uses the same order. Both sides are now pinned on inputs where the two associations differ: `env_tests.rs:684-702` and `test_rewards.py:878-889`. Both pins assert `product_first != quotient_first` before comparing.
- **Accumulation order.** `reward.rs:179-190` keeps `economic` in f64 and adds relative, then bank, then margin, with one f32 rounding. The terminal term is added in f64 and rounded again. With the bank term alone, `economic += A - B` has the same expression tree as `111ae7d`'s `relative + (A - B)`. With both terms off it is `relative as f32`, so the disabled path is byte-identical. The Rust `disabled_margin_term_is_byte_identical_to_the_previous_reward` test and the recorded 16-game `test_env_reference.py` replay both confirm this.
- **Zero-sum.** Each part of the reward is an exact negation between the seats:
  - `relative` is an exact negation between seats.
  - `M` is odd bit for bit: IEEE `*` and `/` are sign-symmetric, and `max(-c).min(c)` on `±x` gives `∓` the same value.
  - The after and before margins are exact negations.
  - The terminal term `f64(e) + ts·sign` negates exactly.
  
  So `r0 == -r1` exactly on every transition.
- **Telescoping and equal start.** `engine_rs/src/lib.rs:1245-1248` builds every farm with `money: starting_money`. `src/kaggriculture/env.rs:501` reads `before_banks = game_banks(&game)` from the pre-step game, which is the reset game on a new game's first step. `after_banks` are read before any auto-reset. `gamma` is 1.0, so the discounted and undiscounted sums agree, and a game's term M sum equals `M(final margin)` up to one f32 rounding per step. The Rust telescope test (719 steps, with a lead change and clamp saturation) and the autoreset test (`[3000, 3000]` on both games' first transition) pin this.
- **Validation and `terminal_scale`.** Rust (`reward.rs:33-82, 83-100`) and Python (`rewards.py:51-103`) agree:
  - every coefficient must be finite and nonnegative;
  - `w_m > 0` requires `S_m > 0` and `c_m > 0`;
  - the caps sum as `((death + ineffective) + bank) + margin < 1`;
  - `terminal_scale` subtracts `c_m` last.
  
  For the preset, `terminal_scale` is exactly 0.5.
- **Critic.** Kaggriculture requires `value_loss: mse` (`python/owl/train/config.py:66`), so there is no winner-CE target that a return outside [0, 1] could invalidate. The whole-game return from reset, `M(final) + .5·sign`, lies in [−1, 1]. The mid-game return-to-go can reach ±1.5. The docs now state that correctly:
  - `docs/rl-api-specs.md:1473-1488`
  - `docs/kaggriculture-contract.md:237`
  - the preset header
  - the Decision's "Critic range" paragraph
  
  I checked the doc's worked example: trailing by 30k and winning by 25k gives 0.25 + 0.3 + 0.5 = +1.05. That is correct.
- **PyO3 and schema.** `bindings.rs:331-345` expects exactly thirteen keys, `rs.pyi` is updated, and every preset states term M explicitly.
- **Warm start.** In `scripts/run_ppo.py:369-395`, `--load-model-weights` loads the weights, and the optimizer with `model_and_optimizer`. It then builds the last-best teacher from those weights and the student config. It never reads J/2's sibling `config.yaml`, which now fails validation because the three new keys are required. So the relaunch path is unaffected.
- **Stateless policy.** Term M is computed natively from game banks and enters only `rewards`. The telemetry (`reward_margin_abs_mean`, `env.py:458-471`) goes only to the metric logger. There are no model inputs, no checkpoint state and no opponent identity.

## Mutations (9 run: 7 killed, 1 equivalent, 1 survived the Rust suite but killed by the Python native suite)

The harness is `<scratchpad>/mut_margin_r2.py`. For each mutation it replaces one exact string in the source, runs the command (rebuilding the extension with `maturin develop --release` where Python must see a native change), restores the source and reports `git status`. The worktree was clean after every mutation, and the clean build passes again afterwards: 617 Python and 134 Rust.

| # | Mutation | Command | Outcome |
| --- | --- | --- | --- |
| N2 | Rust `transition`: margin sign flipped (`after[1-s] - after[s]`, and the same for before). It stays zero-sum. | `cargo test --lib kaggriculture::` | **killed**: `margin_transition_adds_the_potential_difference_before_f32`, `margin_episode_telescopes_to_the_final_margin_score` |
| N3 | Rust `transition`: before-margin read from the after banks, so the increment is always 0 | same | **killed**: same two tests |
| N8 | Rust `validate`: drops the `c_m > 0` requirement | same | **killed**: `margin_validation_budget_and_terminal_scale` |
| N5 | Python validator: drops the `econ_margin_cap <= 0` check | `pytest test_rewards/test_configs/test_env/test_native_env/test_env_reference` | **killed**: `test_margin_admission_budget_and_terminal_scale` |
| N6 | Python oracle: margin added after the f32 rounding | same | **killed**: `test_extreme_value_native_rewards_match_independent_oracle[margin-with-bank]` |
| N4 | PyO3 binding: `econ_margin_weight` read from the `"econ_margin_cap"` key. The preset has both at 0.5, so this is invisible under the preset. | rebuild + pytest | **killed**: `[margin-with-bank]` live parity, `test_margin_admission_budget_and_terminal_scale` |
| N7 | Rust `transition`: margin rounded to f32 separately and added in f32 | `cargo test --lib kaggriculture::` | **SURVIVED** (134 passed). See P3-1. |
| N7b | Same as N7, through the native binding | rebuild + pytest | **killed**: `[margin-with-bank]` live parity |
| N1 | Rust `transition`: the `econ_margin_weight > 0` guard removed, so a disabled term adds `+0.0 - 0.0` | cargo, then rebuild + pytest | Survived both, but it is an **equivalent mutant**. `relative` is never `-0.0` in f64: penalties are `>= +0`, `x - x = +0`, and a tiny negative `relative` stays nonzero in f64. So adding `+0.0` changes no bit. The guard is redundant, not untested. |

## Findings

### P3-1: The Rust "before f32" test does not discriminate the rounding order (mutation N7 survives the Rust suite)

- `src/kaggriculture/env_tests.rs:738-760` (`margin_transition_adds_the_potential_difference_before_f32`)

The test's only nonzero-relative case is `relative = -0.25` with a margin of 5,000 (score +0.05). Rounded once in f64→f32 and rounded twice in f32, these give the same bits (`-1102263091` both ways; I checked in numpy). So the test name claims a property that its inputs cannot detect. N7, which rounds the margin separately and adds in f32, passes all 134 Rust tests.

The property is still pinned end to end: N7b is killed by the Python native-parity row `[margin-with-bank]`. Under the preset, `relative` is exactly 0 and the bank term is off, so `(0 + m) as f32 == m as f32` and live rewards cannot differ. This is test-quality only.

This is the same class as r1's M9, where a Rust pin's inputs gave identical bits. Requested: use a margin where the two orders differ and assert that they do, as the r1 fix did. For example, margin 12,345 at `relative = -0.25`, `c_m` 0.4: `f32(e)` is `-1107192237` and the double rounding gives `-1107192238`. Add `assert_ne!` on the two roundings.

### P3-2: The Decision's warm-start gap omits the last-best teacher's value targets

- `cookbook/decisions/replace-the-reward-with-half-cash-difference-and-half-terminal-sign.md:73`
- `configs/kaggriculture_4rank_margin.yaml:178` (`teacher_value_coef: 0.005`)

The new gap line correctly says that the warm-started critic was trained on J/2's reward. The last-best teacher is also built from the same J/2 weights (`run_ppo.py:390-395`). Its value distillation (`teacher_value_coef` 0.005, against `vf_coef` 2.0) keeps pulling the critic toward J/2-reward values until the first promotion (win rate ≥ 0.7 at a 10M-step checkpoint).

The pull is small, but it is a second, persistent source of the same confound, and a reader of the curves should know about it. Requested: extend the gap line with one clause. Do not change the config: the teacher settings are part of "J/2 apart from the reward".

## Not findings (checked and accepted)

- **Telemetry name.** The brief suggested `train/reward_margin_mean`. The branch logs `train/reward_margin_abs_mean`, which is justified because the signed mean is exactly 0 for a zero-sum term. It is documented in `env.py:5-10` and in the Decision.
- **ops artifacts in two folders.** The core's prepare log is at `ops/rebuild-2026-09-29/reward-margin/prepare.log`, and the r1 artifacts are at `ops/reward-margin/`. The Decision cites both paths, so provenance is intact. This is cosmetic.
- **Saturated wins.** A saturated win returns exactly 1.0, which is at the open boundary of `2p − 1`. This is the same as the pre-existing terminal-only reward (win = 1.0 when every cap is off), so it is not new.
- **|R| > 1 telemetry.** It was not added. The Decision records this as a gap with a reopening condition, which r1 marked optional.
- **The r1 "not findings".** These still hold at this HEAD: `n_runtime_gpus`, the telemetry cost, `return_common_mean` staying 0, truncation (`truncation_prob` 0) and the warm-start path.

VERDICT: APPROVE
