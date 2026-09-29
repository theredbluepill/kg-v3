---
type: "Reference"
title: "Kaggriculture teacher distills per-slot KL and per-seat winner CE"
description: "Phase 4.1: the grammar core returns replay-conditioned per-slot masked logits and a liveness-weighted per-slot KL(teacher || student) in the log-prob frame layout, and Isaiah's categorical KL helper now promotes instead of demoting FP64; CPU TDD with a brute-force oracle and four killed mutations; targets, cache, trainer wiring and configs are pending."
tags: ["kaggriculture-v3", "model", "training", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "repository:python/owl/model/kaggriculture_actor.py"
  - resource: "repository:python/owl/model/kaggriculture.py"
  - resource: "repository:python/owl/model/actor/common.py"
  - resource: "repository:tests/kaggriculture/test_teacher.py"
  - resource: "repository:tests/kaggriculture/helpers.py"
  - resource: "repository:tests/kaggriculture/test_model_heads.py"
  - resource: "repository:docs/model-architecture.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/4-teacher.md"
  - resource: "repository:ops/rebuild-2026-09-29/codex/brief-4-review.md"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.1-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.1-mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.1-py-prepare.log"
---

# Kaggriculture teacher distills per-slot KL and per-seat winner CE

Branch `kg/rebuild-trainer-model`. This note records Phase 4 of `ops/rebuild-2026-09-29/plan.md` as specified by the v2 brief `ops/rebuild-2026-09-29/briefs/4-teacher.md` (Codex brief review REVISE, all findings applied). It serves the [[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]]. It builds on the [[kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|grammar heads]] and the `TeacherTargets` protocol in the [[ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|PPO trainer seams Reference]]. A cookbook search found no current note on Kaggriculture teacher distillation; the reference branch's [[shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]] lists the teacher as a gap.

## Adaptation inventory

**4.1 distributions and KL.**
- `python/owl/model/kaggriculture_actor.py`: `policy_core` gains `teacher_logits` and `collect_logits`. With `collect_logits`, each policy slot's density-dtype logits are returned, masked with the replay-conditioned mask the density uses and filled with `finfo(dtype).min` outside it (`market_kind` uses the final HIRE-capacity mask). With `teacher_logits`, each slot adds `categorical_kl_from_logits(teacher, student, mask)` weighted by `unit_live` or `market_live`, placed with the log-probs' `place` scatter into `[B, 252, 12]`. `GrammarPolicyResult` gains `slot_logits` and `kl` (both `None` unless requested).
- `python/owl/model/kaggriculture.py`: `_policy` slices `teacher_logits` per head chunk and joins `slot_logits` and `kl`. `_actor_inputs` (the actor projection) and `_evaluation_from` (the replay `ModelEvaluation`) are extracted so the teacher paths and the oracle test share them. `forward` and `evaluate_actions` behave as before.
- `python/owl/model/actor/common.py`: Isaiah's `categorical_kl_from_logits` computes in `promote_types(teacher, student, float32)` instead of `.float()`. BF16, FP16 and FP32 inputs compute in FP32 as before; FP64 is no longer demoted. The brief review found the demotion turns an FP64 `finfo.min` fill into `-inf` and breaks FP64 gradient checks.
- `tests/kaggriculture/helpers.py`: the heads tests' tiny models, programs and replay cases, moved unchanged in behavior (65 heads tests pass before and after).

The estimator is Isaiah's: teacher-forced conditionals at the behavior policy's replayed prefix, summed per slot, not an unbiased joint-program KL. Both sides must share the grammar tables and `hire_limit`; replay admission does not detect a mismatch that still admits the program.

## Verification (this version, CPU only)

- Red: all 19 new tests failed on the missing `collect_logits` argument (`4.1-red.log`).
- Green, `tests/kaggriculture/test_teacher.py` 4.1 block:
  - **T1:** a teacher equal to the student gives `torch.equal` zeros on four replay cases. A perturbed teacher gives finite values that are exactly 0 at slots 0/2/11, dead unit frames, positions after STOP, the forced sentinel, unavailable positions, item/digit slots at STOP and inactive rows, and positive elsewhere.
  - **T2:** `log_softmax(slot_logits)` at the replayed token is `torch.equal` to `evaluate_actions` log-probs in FP32 and FP64 on all four cases.
  - **T3:** a brute-force oracle that runs `policy_core` once per admissible value (sets built in Python from the tables and the HIRE rule, each checked to carry probability 1) matches the KL within 1e-6 on the base programs and a capacity-exhausted HIRE case.
  - **T4:** head chunking (6 calls of 2 rows) and trunk chunking (6 calls of 2 rows) match the unchunked result with `assert_close`.
  - **T5:** a `fullgraph` eager-backend capture of the core matches eager for the logits, the KL and its gradients.
  - **T6:** KL gradients reach every student head and no teacher parameter. In FP64, autograd matches a central finite difference within `rel=1e-6`.
- Non-vacuity (`4.1-mutations.log`), each applied, run and reverted:

| Mutation | Result |
|---|---|
| (a) unit KL against unmasked student logits | 7 failed (T1, T3, T6 FD) |
| (b) market-kind KL not weighted by `market_live` | 2 failed (T1) |
| (c) KL helper demotes to FP32 again | 1 failed (T6 FD) |
| (d) slot logits collected with `-inf` masking | 8 failed (T2) |

- `just py-prepare` (`4.1-py-prepare.log`): format, lint, mypy and 1,343 passed with 4 skipped (3 hardware/backend, 1 native grammar binding). docs-fresh first flagged `docs/model-architecture.md`; after the Kaggriculture teacher bullet and the KL dtype note were added, it passes.

## Deviations from the brief (4.1)

- **T1 non-negativity:** FP32 rounding gives values down to −5.6e-8 where the true KL is of order 1e-5, so the test bounds `kl ≥ −1e-6` instead of `≥ 0`.
- **T4 head chunking:** the brief asked for `torch.equal`, but head-chunked market-kind logits differ in the last bits (CPU GEMM blocking varies with row count). The test uses `assert_close`, as the heads chunking test already did. T2's FP32 exactness, which the brief made a stop condition, holds.
- **T3 HIRE case:** the oracle uses a small `hire_limit = 5` case (budgets 2, 3, 1, 1, all exhausted mid-queue) instead of the 238–241-actor `hire_capacity` case. The one-value-per-row oracle would need about 5,000 variant rows of 241 frames there. The capacity-blocked HIRE sites are asserted to be exercised.

## Limits and gaps

- Everything ran on CPU with tiny models and the synthetic grammar tables (Task 1.4 binding pending). BF16, compile on CUDA and GPU memory/time are unmeasured.
- The per-seat KL sums up to 241 × 5 + 11 × 4 conditional KLs with Isaiah's coefficient 0.005. This scale difference is recorded, not tuned.
- 4.2 (targets and cache), 4.3 (model methods and trainer wiring) and 4.4 (configs) are not recorded here yet.
