---
type: "Decision"
title: "Add a per-seat critic offset for the own-bank reward"
description: "Owner direction (2026-09-30, \"per-player critic might be the way out?\") after self-play was re-adopted with a non-zero-sum own-bank reward: model.critic_offset (default off) adds a head on each seat row's own critic-value token whose zero-initialised output o is added to the value, V = 2p(self) - 1 + o. The value MSE trains the sum; teacher value distillation still targets the winner softmax. The loader accepts a checkpoint that omits exactly all critic_offset_head.* keys and zeroes the head's output; critic_offset_detach_trunk stops the head's trunk gradient. The rollout logs train/value_offset_mean, train/value_offset_abs_mean and train/ev_common. configs/kaggriculture_4rank_bank_critic.yaml is the margin preset with the owner-approved reward .25 own bank (/150k) + .25 cash difference (/100k) + .5 terminal sign and the head on. Default off is byte-identical to 3e89425 (native, trainer, model and config digests); 10 of 10 source mutations are killed. Untrained."
tags: ["kaggriculture-v3", "model", "training", "rewards", "decisions", "adaptation"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-30"}
decider: "Owner, 2026-09-30 (as relayed verbatim by the orchestrating agent): \"per-player critic might be the way out?\"; the reward numbers were agent-proposed and approved with \"OK go ahead.\""
sources: [{"resource": "user-directive:2026-09-30:per-player-critic-might-be-the-way-out"}, {"resource": "user-directive:2026-09-30:switch-back-to-self-play-earn-money-for-real"}, {"resource": "user-directive:2026-09-30:ok-go-ahead-bank-margin-sign-reward"}, {"resource": "local-untracked:kg-v3-int/ops/earn-money-2026-09-30/plan.md"}, {"resource": "local-untracked:kg-v3-int/ops/earn-money-2026-09-30/design-signal.md"}, {"resource": "local-untracked:kg-v3-int/ops/earn-money-2026-09-30/critique-signal.md"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/model/base.py"}, {"resource": "repository:python/owl/model/lora.py"}, {"resource": "repository:python/owl/train/ppo.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:configs/model/kaggriculture_critic_offset.yaml"}, {"resource": "repository:configs/kaggriculture_4rank_bank_critic.yaml"}, {"resource": "repository:tests/kaggriculture/test_critic_offset.py"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:docs/kaggriculture-contract.md"}, {"resource": "repository:docs/kaggriculture-model.md"}, {"resource": "repository:README.md"}, {"resource": "repository:ops/critic-offset-2026-09-30/baseline_digest.py"}, {"resource": "repository:ops/critic-offset-2026-09-30/baseline-digest-3e89425.json"}, {"resource": "repository:ops/critic-offset-2026-09-30/post-digest.json"}, {"resource": "repository:ops/critic-offset-2026-09-30/mutations.py"}, {"resource": "repository:ops/critic-offset-2026-09-30/mutations.log"}, {"resource": "repository:ops/critic-offset-2026-09-30/prepare.log"}, {"resource": "repository:ops/critic-offset/review-r1.md"}, {"resource": "repository:ops/critic-offset/r1-followup-mutation.log"}]
---

# Add a per-seat critic offset for the own-bank reward

## Decision

The owner, verbatim, on 2026-09-30, as relayed by the orchestrating agent (this note's author did not see the conversation itself):

1. **"let's switch back to self play no matter what, and think about how do we get the agent to earn moneny for real?"**
2. After the main agent explained that the Kaggriculture critic is a zero-sum winner head that cannot represent the shared (common-mode) part of a non-zero-sum own-bank reward: **"per-player critic might be the way out?"**
3. On the agent-proposed reward numbers (0.25 own bank /150k + 0.25 cash difference /100k + 0.5 terminal sign): **"OK go ahead."**

**Interpretation, not owner adoption.** The orchestrating agent read (2) as a request to build the design's per-seat critic offset (plan F4, design C2 in `kg-v3-int/ops/earn-money-2026-09-30/`), behind a default-off flag, and to ship a preset with the approved reward and the head on. The owner has not confirmed that reading, the head's form, or the choice to leave the trunk gradient attached in the preset.

## Why

Under term A (`econ_bank_*`) both seats can gain or lose together, so the return has a common mode (the mean over both seats). The winner critic's value `2 p(self) − 1` is bounded to (−1, 1) and is trained toward a consistent winner distribution by the winner and teacher cross-entropies. It can represent the common mode only by distorting those winner probabilities. What it cannot explain enters the advantages as unexplained return. The design files report the one v2 mirror recipe that raised its own bank: an own-bank target with a per-seat critic (m17, 82.0k → 86.8k). This note has not checked that figure.

**Correction.** The earlier wording "the two seat values always sum to 0" (term A Decision, `docs/rl-api-specs.md`, `docs/kaggriculture-contract.md`) was too strong. `p(self) + p(opponent) = 1` holds within one seat's row, but the two seats' rows are separate softmaxes over different views. Their values sum to zero only for a critic consistent across the views. The hard limits are the range and the winner semantic. Those three places now say so.

## Design (as implemented)

- **Head.** `KaggricultureTransformer.critic_offset_head` is an `OutputProjectionMLP(trunk, 1)`, registered last. It reads only `critic_value_hidden[:, 0]`, the row's own (self) critic-value token: no opponent token, no other row, and no opponent identity. It is stateless. Its output is cast to FP32 (its Linear layers run in the autocast dtype, bf16 in the presets) and masked to 0 on non-live rows.
- **Value.** `_values` returns `2 p(self) − 1 + o`. So every place the critic value is used sees the sum: rollout values for GAE, the truncation bootstrap, `last_values`, the MSE value loss (`winner_ce` stays refused for Kaggriculture) and `train/explained_variance`. `ModelOutput.value_offsets` carries `o`, and it is `None` without the head.
- **Zero init.** `reset_parameters` zeroes the output layer's weight and bias instead of applying a gain. The hidden layer takes the normal init, so the head can train. At step 0, values, actions and log-probabilities are bit-identical to the headless model on the same weights. The output layer is in `get_output_layers` (AdamW, no int8). Its hidden layer uses Muon.
- **Teacher.** Value distillation is unchanged: a CE between the student's and the teacher's winner softmaxes (`teacher_winner_probabilities` / `student_winner_log_probabilities`). The offset is outside it and trained only by the value MSE, so the CE stays well-defined whether or not either model has a head. MSE sees only the sum. The winner part is also pulled toward the teacher's winner distribution, and the offset takes the rest. The split between the two is not otherwise identified.
- **`critic_offset_detach_trunk`** (requires `critic_offset`) feeds the head a detached token. The offset's gradient then stops at the head, while the winner part still trains the trunk.
- **Loader rule.** `BaseModelAPI.optional_state_keys()` (default empty) and `reset_optional_state()`. `load_model_state_dict_allowing_lora`, the loader behind every `--load-model-weights` mode, `rl.teacher_init`, last-best and the resume's last-best, accepts a checkpoint that omits **all** `critic_offset_head.*` keys. It then zeroes the head's output. A partial head, any other missing key and any unexpected key still fail. With the flag off, a head checkpoint fails on its unexpected keys. `model_and_optimizer` from a headless checkpoint fails on the optimizer param count, so warm starts use `model_only`. The resume model itself loads strictly, so a headless resume into a head config fails.
- **Architecture parity.** The last-best teacher and the promotion refresh are built from the student config, so they carry the same head. A fixed teacher (`teacher_mode: fixed`) keeps its own config. Only its winner probabilities are read, so this is well-defined either way.
- **Telemetry**, only with the head. The rollout buffer stores each offset. `train/value_offset_mean` and `train/value_offset_abs_mean` are taken over the value mask. `train/ev_common` is the explained variance of the mean of both seats' GAE returns by the mean of both seats' offsets, over steps where both seats are trained. It is omitted when no such step exists (fixed opponent at fraction 1.0). Learner rows scatter their offsets like their values.
- **Kaggle agent.** The policy never reads the head, which a test shows: random head weights change no action or log-probability. The Orbit `Agent` does not load Kaggriculture checkpoints yet. The rule for Task 7.4's packaging is to build the model from the checkpoint's own `config.yaml`, which records `critic_offset`, so the head's keys load strictly.
- **Default off.** No head is built. The config serializer omits both fields when `critic_offset` is false, so `config.yaml` and `v3/config_sha256` are unchanged.

## Preset and values

`configs/kaggriculture_4rank_bank_critic.yaml` is `configs/kaggriculture_4rank_margin.yaml` (J/2's recipe, 4 ranks) with two changes:

- **Reward.** `econ_bank_weight` .25, `econ_bank_scale` 150,000, `econ_bank_cap` .25; `econ_margin_weight` .25, `econ_margin_scale` 100,000, `econ_margin_cap` .25; `econ_shaping` 0. So `terminal_scale` is .5.
- **Model.** `model: kaggriculture_critic_offset`, which is `configs/model/kaggriculture.yaml` plus `critic_offset: true` and `critic_offset_detach_trunk: false`.

The reward numbers were proposed by the agent and approved by the owner. They are not owner-derived or measured optima. The head adds 66,049 parameters (6,318,272 in all).

## Adaptation inventory

- `python/owl/model/kaggriculture.py`: the config fields, validator and serializer; the head, zero init, `optional_state_keys` / `reset_optional_state`, `_value_parts` / `_critic_offsets`, and `forward`'s `value_offsets`.
- `python/owl/model/base.py`: `ModelOutput.value_offsets` and the two optional-state hooks.
- `python/owl/model/lora.py`: `_admit_missing_optional_state` in the shared loader.
- `python/owl/train/ppo.py`: the rollout's `value_offsets` buffer, `LearnerRowsOutput.value_offsets`, and `_value_offset_metrics`.
- `scripts/run_ppo.py`: the `--load-model-weights-mode` help text only.
- The configs: `configs/model/kaggriculture_critic_offset.yaml` and `configs/kaggriculture_4rank_bank_critic.yaml`.
- The tests: `tests/kaggriculture/test_critic_offset.py` (new), `tests/kaggriculture/test_configs.py` (the preset registry, its equality to the margin preset and its reward), and `tests/scripts/test_run_ppo.py` (a two-update run through `main`).
- The docs: `docs/model-architecture.md`, `docs/rl-api-specs.md`, `docs/kaggriculture-contract.md`, `docs/kaggriculture-model.md` and `README.md`.
- The term A Decision's common-mode limit is revised (the correction above).

## Checks (this tree, Apple M5, CPU, `OMP_NUM_THREADS=2`)

- **Default-off identity.** `ops/critic-offset-2026-09-30/baseline_digest.py` was run on 3e89425 and on this change. The native env digest (`257eae38…`), the trainer's two-update digest (`3ffd53a0…`, the receipt of the opponent-mix landing), the tiny model's forward, evaluate and compute_value digest, and the config hash plus `config.yaml` bytes of all 14 existing Kaggriculture presets are all identical. The only difference is the new preset's line (`baseline-digest-3e89425.json` against `post-digest.json`).
- **`tests/kaggriculture/test_critic_offset.py`** (21 tests) covers:
  - the default dump and the pinned hash of `configs/kaggriculture.yaml`;
  - no head with the flag off, and the zero-output / nonzero-hidden init;
  - bit-for-bit values, actions and log-probabilities at step 0 on the off model's weights;
  - value = winner + offset, with actions independent of the head;
  - a zero offset on non-live rows, and the head's input being exactly token 0 with row independence;
  - a supervised fit of returns with common mode 1.2: the off critic stays at MSE ≥ .125, the head reaches below 1e-3;
  - the detached trunk gets no gradient, with the attached control reaching the trunk;
  - the loader rule: headless loads, partial, other-missing and extra keys fail, a head checkpoint into an off model fails, strict round trip through its own config, and the trainer's `model_only` versus `model_and_optimizer`;
  - off-trainer and on-trainer runs over two updates with term A on;
  - learner-row offsets under a fixed opponent, and `ev_common` omitted at fraction 1.0;
  - the telemetry formula.
- **`test_critic_offset_with_the_own_bank_reward_two_update_run_through_main`** runs a tiny CPU run with term A and term M at the preset values plus `model.critic_offset=true`. The last-best teacher loads the headless `rl.teacher_init` by the loader rule. The telemetry is finite, the head trains, `config.yaml` records the flag, and every checkpoint reloads strictly.
- **Mutations.** 10 of 10 source mutations are killed (`mutations.log`): detach removed, no zero init, no reset on load, partial head accepted, opponent token read, live mask removed, offset not added, fields dumped, learner-row offsets dropped, and `ev_common` not scored by the offsets.
- **`just prepare`** (`CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`, `uvx --from rust-just just prepare`) exits 0 on the working tree before commit. It runs build, fmt, clippy, ruff, docs-lint and mypy (77 files, no issues). Rust reports 407 passed and 5 ignored across the three crates. Python reports 2,969 passed and 9 skipped (full suite), and docs-fresh passes (`ops/critic-offset-2026-09-30/prepare.log`).

## Gaps and reopening conditions

- **Nothing is trained.** The preset's first run is unlaunched, and no effect on own bank is measured. The critic-signal review (`critique-signal.md`) makes three points:
  - It recommends a detached first arm to separate value-gradient interference in the trunk from common-mode credit. The preset leaves the trunk attached. That was the building agent's choice under its task text; neither the owner's quoted words nor plan F4 ask for it, and F4 asks for a detached first arm. The detached arm is `-o model.critic_offset_detach_trunk=true`. Even detached, the head's gradient enters the global `max_grad_norm` clip, so a detached arm is not free of the clip-share part of M5 (review r1 P2-2: shared-update Σ|Δ| −21% at clip 1e-3 on a tiny model). Reopen by logging the head's gradient norm or clipping it separately before reading a detached arm as M5-free.
  - `ev_common > 0.3` can be met by fitting the common mode's phase-of-game trend without valuing assets.
  - The mean δ on purchase rows is not logged. Reopen with that measurement if a run's own bank moves.
- **The value split is not identified.** The winner part and the offset share one MSE target. Only the teacher CE (`teacher_value_coef` 0.005) and the (−1, 1) bound separate them. Their split is unmeasured.
- **Slow head output.** The head's output layer is on AdamW at the preset's `adamw_lr` 5e-6, the same as `critic_head.out`. How fast the offset learns is unmeasured.
- **BC.** BC trains only the winner CE, so a BC model with the flag would keep a zero offset. No BC config sets it.
- **RNG stream.** Building the head draws from the global RNG, so a head run and a flag-off twin at the same seed do not share sampled trajectories (review r1 P3-4).
- **No optimizer carry-over.** `model_and_optimizer` from a headless checkpoint is refused, not adapted.
- **Review.** `ops/critic-offset/review-r1.md` reviewed `732438c` (VERDICT: APPROVE, no P1). Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict. Author follow-ups, tests and docs only: P2-1, `test_value_is_the_winner_value_plus_the_offset_and_actions_ignore_it` now checks `compute_value` and `evaluate_actions` against the sum at a nonzero head, and the review's surviving mutation R1 is killed (`ops/critic-offset/r1-followup-mutation.log`); P2-2, P3-1, P3-3 and P3-4 are corrected or stated as limits in this note and the docs; P3-2's provenance is stated above. Not changed: P2-2's separate clip or head grad-norm metric, P3-3's out-of-autocast head, P3-4's `fork_rng`, and P3-5's `ops/` lint nits and quantizer name match. The follow-ups were not independently re-reviewed. The owner has not confirmed the interpretation above.
