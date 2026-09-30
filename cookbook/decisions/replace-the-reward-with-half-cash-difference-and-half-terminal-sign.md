---
type: "Decision"
title: "Replace the reward with half cash difference and half terminal sign"
description: "Owner decision (2026-09-30): relaunch the pre-anchor J/2 run with the reward 0.5 x cash difference + 0.5 x terminal sign (+1/-1/0), before revisiting the anchor setup. Adds reward term M, a zero-sum clamped margin score M(d) = clamp(w_m * d / S_m, -c_m, c_m) of d = bank_self - bank_opp whose per-step change is paid to each seat, with bit-exact Rust/Python parity, three required fields, the cap in the below-one budget and terminal_scale reduced by c_m. configs/kaggriculture_4rank_margin.yaml is J/2's effective config with starvation/drought shaping and term A off and w_m .5, S_m 50,000, c_m .5 (terminal_scale .5); the .5/.5 split is the owner's, the 50,000 scale is agent-proposed. Every other config sets term M off, bit-identically. The reading of '(before anchor) run' as J/2 is the main agent's interpretation. Docs and cookbook landed after the launch at the owner's 'can we accelerrate?'. CPU checks only."
tags: ["kaggriculture-v3", "rewards", "training", "decisions", "adaptation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
decider: "Owner, 2026-09-30: \"can we relaunch (before anchor) run, fix the reward, 0.5 Cash Diff (add this in) + 0.5 (Terminal loss 1/-1/0)?\""
sources: [{"resource": "user-directive:2026-09-30:relaunch-before-anchor-run-half-cash-diff-half-terminal"}, {"resource": "user-directive:2026-09-30:implement-the-new-reward-first"}, {"resource": "user-directive:2026-09-30:can-we-accelerate"}, {"resource": "external-url:https://wandb.ai/spoon/kg-v3/runs/nw3klj2s"}, {"resource": "repository:src/kaggriculture/reward.rs"}, {"resource": "repository:src/kaggriculture/bindings.rs"}, {"resource": "repository:src/kaggriculture/env_tests.rs"}, {"resource": "repository:python/owl/kaggriculture/rewards.py"}, {"resource": "repository:python/owl/kaggriculture/env.py"}, {"resource": "repository:python/owl/rs.pyi"}, {"resource": "repository:configs/kaggriculture_4rank_margin.yaml"}, {"resource": "repository:configs/kaggriculture_4rank.yaml"}, {"resource": "repository:tests/kaggriculture/test_rewards.py"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:tests/kaggriculture/test_env.py"}, {"resource": "repository:tests/kaggriculture/test_env_reference.py"}, {"resource": "repository:tests/kaggriculture/test_native_env.py"}, {"resource": "repository:tests/kaggriculture/test_game.py"}, {"resource": "repository:tests/kaggriculture/test_training_smoke.py"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/reward-margin/prepare.log"}]
---

# Replace the reward with half cash difference and half terminal sign

## Decision

The owner, verbatim, on 2026-09-30:

1. **"can we relaunch (before anchor) run, fix the reward, 0.5 Cash Diff (add this in) + 0.5 (Terminal loss 1/-1/0)?"**
2. **"implement the new rewrad first before we revisit the cha22 anchor setup."**
3. About the launch plan: **"can we accelerrate?"**

The owner adopted a reward of one half cash difference plus one half terminal win/loss/draw sign, a relaunch of the pre-anchor run with it, and the order: the reward before the anchor setup. The owner asked for the relaunch, so replacing the live run on the running pod was approved.

**Interpretation, not owner adoption.** The main agent read the owner's words as follows; the owner has not confirmed these readings:
- "(before anchor) run" is J/2 (`control-J2-4rank-20260930`, W&B `spoon/kg-v3` `nw3klj2s`, run dir `20260930-010131`), the run that the live `hz4bpjnq` continued. "Relaunch" is a fresh warm start from J/2's `checkpoint_final.pt` (sha256 `80e5667ecbf36966a3edc94d2e831ee5372cd7e791b5d235cf30cbfc32e161dd`) with `--load-model-weights-mode model_and_optimizer`, as `hz4bpjnq` was launched, with J/2's LRs (muon 1e-4, adamw 5e-6), the 10M checkpoint and evaluation cadence and `native_threads` 4.
- "fix the reward" replaces the reward: starvation/drought shaping off (`econ_shaping` 0) and term A off (`econ_bank_weight` 0), leaving only term M and the terminal sign.
- "can we accelerrate?" puts the launch on the critical path: the reward core (`87beaf0`) was launched first, and these docs, this note and the full `just prepare` follow it, before landing.

## Why

The main agent reported that `hz4bpjnq` (J/2 continued under 0.2 starvation/drought shaping plus 0.8 terminal sign) was sliding, with its own bank falling from about 63k to about 38k over 875 iterations. This note has not checked those numbers. The earlier reward never paid for earning. The penalty reads only the starvation and drought counters, and it saturates at its 0.2 cap after the first event. Banks entered only as `sign(bank_self − bank_opp)` at game end. Term M pays the bank margin on every step.

## Design (as implemented, commit `87beaf0`)

- **Score.** `M(d) = clamp(w_m × d / S_m, −c_m, c_m)` in binary64, evaluated as the product, then the quotient, then the clamp. An overflowing product or quotient saturates at ±`c_m` (`S_m > 0`, so it is never NaN). `M` is odd in `d` bit for bit, because IEEE negation, multiplication and division round sign-symmetrically.
- **Per-step reward.** Seat `s` adds `M(after_s − after_o) − M(before_s − before_o)` to its economic term. The before and after banks are those of term A: the state the action was taken in, and the completed transition before any auto-reset, so no delta spans two games. The two seats' increments are exact negations, so the term is **zero-sum**. Unlike term A, the zero-sum winner critic can represent it.
- **Telescoping.** Both farms reset to the same `startingMoney` (`engine_rs/src/lib.rs`), so the margin starts at 0 and a complete game's term telescopes to `M(final margin)`.
- **Rounding.** In float64, the relative penalty term is added first, then the bank term, then the margin term. The sum is rounded to f32 once, and the terminal term is added as before. With `w_m = 0` nothing is added, so rewards are byte-identical to the reward without it.
- **Budget.** `terminal_scale = 1 − (enabled caps)`, now including `c_m` when `w_m > 0`. Validation requires every coefficient to be finite and nonnegative, requires `S_m > 0` and `c_m > 0` when `w_m > 0`, and requires the active caps, summed as death, then ineffective, then bank, then margin, to be below 1. The Rust `validate()` and `KaggricultureRewardConfig` apply the same predicate. The three fields are required everywhere, and the native reward dict now has thirteen keys.
- **Values.** `configs/kaggriculture_4rank_margin.yaml` sets `econ_margin_weight` 0.5, `econ_margin_scale` 50000.0 and `econ_margin_cap` 0.5, with `econ_shaping` 0 and the bank term off. So `terminal_scale` is 0.5, and a game returns `.5 × clamp(final margin / 50,000, −1, 1) + .5 × sign(final margin)`, within [−1, 1]. Every other preset sets 0.0 / 50000.0 / 0.0 explicitly.
- **Stateless policy.** The term is computed natively from game banks and enters only the rewards. It never feeds model inputs, and no opponent identity is involved.

## Values: provenance

The owner gave the 0.5 / 0.5 split. "0.5 Cash Diff" does not say how cash is normalised, so the agent chose the scale. `S_m = 50,000` is agent-proposed from `hz4bpjnq`'s self-play margins as the main agent reported them (mean |margin| 10–18k, median 8–13k). At that scale typical games stay in the linear range, and the term saturates only at |margin| ≥ 50k. The scale is not owner-given and not a measured optimum. The cap `c_m = .5` makes the dense term's whole-game weight equal to the terminal term's.

## Adaptation inventory

- **Native.** `src/kaggriculture/reward.rs` adds the three fields, `margin_score`, the validation and budget rules, and the margin increment in `transition`. `src/kaggriculture/bindings.rs` requires the thirteen-key dict. `python/owl/rs.pyi` updates `KaggricultureRewardDict`.
- **Python.** `python/owl/kaggriculture/rewards.py` adds the fields, the shared predicate, and the `margin_score` and `margin_rewards` oracles, and `transition_rewards` gains the margin term. `python/owl/kaggriculture/env.py` adds the `reward_margin_abs_mean` step metric (the mean absolute component over seats; the signed mean is zero), logged as `train/reward_margin_abs_mean`.
- **Configs.** Every Kaggriculture preset states term M off. The new `configs/kaggriculture_4rank_margin.yaml` reproduces J/2's effective config (`kaggriculture_4rank.yaml` with muon 1e-4, adamw 5e-6, `checkpoint_freq` 10M and `native_threads` 4) except for the reward. Its header quotes the owner.
- **Tests** (from the core commit). Rust `env_tests.rs` covers validation, the clamp, zero-sum, telescoping, byte-identity with the term off, and autoreset. Python covers the oracle, native parity, the telemetry value and the preset (`test_rewards.py`, `test_configs.py`, `test_env.py`, `test_env_reference.py`, `test_native_env.py`, `test_game.py`, `test_training_smoke.py`).
- **Docs** (this follow-up). `docs/rl-api-specs.md` documents the fields, the validation, the formula, the term M paragraph and the migration. `docs/kaggriculture-contract.md` adds `ΔM_s` to the reward and to `terminal_scale`. `README.md` describes the margin preset.
- **Migration.** A `config.yaml` written before this change, such as J/2's or `hz4bpjnq`'s, no longer validates as `rl.teacher_init` or for a full resume. Adding `econ_margin_weight: 0.0`, `econ_margin_scale: 50000.0` and `econ_margin_cap: 0.0` reproduces its rewards exactly. A `--load-model-weights` warm start builds the last-best teacher from the loaded weights and the student config (`scripts/run_ppo.py`). It does not read the source's sibling config, so the relaunch is unaffected.

## Checks

- Full `just prepare` on the core commit `87beaf0` plus this follow-up's docs and cookbook exits 0: Rust root 291 passed with 5 ignored, plus the other crates; Python 2,852 passed with 18 skipped; ruff, mypy, docs-lint and docs-fresh are clean, and the formatters changed no file (`ops/rebuild-2026-09-29/reward-margin/prepare.log`). Before the run, the git-ignored Orbit fixtures were copied in, identical by `diff -rq`.
- `just docs-fresh` passes. None of the core commit's code paths fall under a docs-fresh rule, and the reward docs were updated anyway.
- Nothing was trained by this follow-up. The relaunch itself belongs to its own run episode.

## Gaps and reopening conditions

- One seed, untrained at commit time. The relaunch cannot separate the new reward from the fresh warm start's reset LR schedule and optimizer step counter.
- The 50,000 scale is unmeasured. Reopen it if `train/reward_margin_abs_mean` or the terminal margins show that most games saturate the clamp (|margin| ≥ 50k), or that the dense term is negligible.
- The owner has not confirmed that "(before anchor) run" means J/2, or that "fix the reward" turns the starvation/drought shaping off. Reopen if the owner reads either differently.
- The anchor setup ("cha22") is deferred by the owner's own ordering and is not addressed here.
- No independent verification of this change has run yet.
