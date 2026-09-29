---
type: "Reference"
title: "Kaggriculture teacher distills per-slot KL and per-seat winner CE"
description: "Phases 4.1-4.2: the grammar core returns replay-conditioned per-slot masked logits and a liveness-weighted per-slot KL(teacher || student), Isaiah's categorical KL helper promotes instead of demoting FP64, and KaggricultureTeacherTargets caches them (102,208 B per seat row) under the TeacherTargets protocol with symmetric concat, nbytes and a grammar signature; CPU TDD, a brute-force oracle and eight killed mutations; trainer wiring and configs are pending."
tags: ["kaggriculture-v3", "model", "training", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "repository:python/owl/model/kaggriculture_actor.py"
  - resource: "repository:python/owl/model/kaggriculture.py"
  - resource: "repository:python/owl/model/actor/common.py"
  - resource: "repository:python/owl/model/kaggriculture_teacher.py"
  - resource: "repository:python/owl/model/teacher_targets.py"
  - resource: "repository:python/owl/model/stateless_transformer_v1.py"
  - resource: "repository:python/owl/model/base.py"
  - resource: "repository:python/owl/model/__init__.py"
  - resource: "repository:python/owl/kaggriculture/gpu_grammar.py"
  - resource: "repository:python/owl/train/ppo.py"
  - resource: "repository:python/owl/train/distributed.py"
  - resource: "repository:tests/kaggriculture/test_teacher.py"
  - resource: "repository:tests/kaggriculture/helpers.py"
  - resource: "repository:tests/kaggriculture/test_model_heads.py"
  - resource: "repository:docs/model-architecture.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/4-teacher.md"
  - resource: "repository:ops/rebuild-2026-09-29/codex/brief-4-review.md"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.1-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.1-mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.1-py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.2-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.2-mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.2-py-prepare.log"
---

# Kaggriculture teacher distills per-slot KL and per-seat winner CE

Branch `kg/rebuild-trainer-model`. This note records Phase 4 of `ops/rebuild-2026-09-29/plan.md` as specified by the v2 brief `ops/rebuild-2026-09-29/briefs/4-teacher.md` (Codex brief review REVISE, all findings applied). It serves the [[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]]. It builds on the [[kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection|grammar heads]] and the `TeacherTargets` protocol in the [[ppo-trainer-seams-map-any-schema-and-alarm-on-replay-drift|PPO trainer seams Reference]]. A cookbook search found no current note on Kaggriculture teacher distillation; the reference branch's [[shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]] lists the teacher as a gap.

## Adaptation inventory

**4.1 distributions and KL.**
- `python/owl/model/kaggriculture_actor.py`: `policy_core` gains `teacher_logits` and `collect_logits`. With `collect_logits`, each policy slot's density-dtype logits are returned, masked with the replay-conditioned mask the density uses and filled with `finfo(dtype).min` outside it (`market_kind` uses the final HIRE-capacity mask). With `teacher_logits`, each slot adds `categorical_kl_from_logits(teacher, student, mask)` weighted by `unit_live` or `market_live`, placed with the log-probs' `place` scatter into `[B, 252, 12]`. `GrammarPolicyResult` gains `slot_logits` and `kl` (both `None` unless requested).
- `python/owl/model/kaggriculture.py`: `_policy` slices `teacher_logits` per head chunk and joins `slot_logits` and `kl`. `_actor_inputs` (the actor projection) and `_evaluation_from` (the replay `ModelEvaluation`) are extracted so the teacher paths and the oracle test share them. `forward` and `evaluate_actions` behave as before.
- `python/owl/model/actor/common.py`: Isaiah's `categorical_kl_from_logits` computes in `promote_types(teacher, student, float32)` instead of `.float()`. BF16, FP16 and FP32 inputs compute in FP32 as before; FP64 is no longer demoted. The brief review found the demotion turns an FP64 `finfo.min` fill into `-inf` and breaks FP64 gradient checks.
- `tests/kaggriculture/helpers.py`: the heads tests' tiny models, programs and replay cases, moved unchanged in behavior (65 heads tests pass before and after).

**4.2 targets and cache.**
- `python/owl/model/kaggriculture_teacher.py` (new): `KaggricultureTeacherTargets(slot_logits, winner_probabilities, grammar)` implements `TeacherTargets`. `index` slices dim 0 and carries `grammar`. `concat` validates every chunk symmetrically: optional-field presence, slot keys and grammar must match, and each failure raises `ValueError` naming the field; one chunk is returned without a copy. `nbytes` sums tensor metadata. `GrammarSignature(tables_sha256, hire_limit)` and `TEACHER_TARGET_BYTES_PER_ROW` (102,208, derived from `kt.SLOT_WIDTHS`, 241 unit frames and 11 market positions, plus 8 B of winner probabilities) are also defined there.
- `python/owl/kaggriculture/gpu_grammar.py`: `grammar_tables_digest` (SHA-256 over names, shapes and values). `KaggricultureGrammarActor` takes it once at construction; the buffers are never reassigned, so the signature needs no device sync.
- `python/owl/model/kaggriculture.py`: `grammar_signature()` and `compute_teacher_distillation_targets`, which runs one guarded encode, `_policy(collect_logits=True)`, `check_replay_flags`, a reshape to the lead and the winner probabilities, all under `no_grad`.
- **Protocol typing (brief §3.2, moved from 4.3 because mypy required it with the new return type):** `BaseModelAPI.compute_teacher_distillation_targets -> TeacherTargets` and `evaluate_actions_with_cached_teacher(teacher_targets: TeacherTargets)`. `StatelessTransformerV1` narrows with `isinstance` and raises `TypeError` before any kernel. `ppo.py` and the DDP adapter (`python/owl/train/distributed.py`) annotate the protocol. `teacher_targets.py` adds `nbytes`, and `CachedTeacherDistillationTargets.nbytes` sums its tensors.

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
| 4.2 (a) `concat` checks winner presence only against the first chunk | 1 failed (T9) |
| 4.2 (b) `index` slices dim 1 | 6 failed (T9) |
| 4.2 (c) grammar dropped from `concat`'s check | 1 failed (T9) |
| 4.2 (d) `nbytes` omits winner probabilities | 1 failed (T10) |

- 4.2 red: collection `ImportError` (missing module); the Isaiah foreign-targets test failed with an `AttributeError` after encoding (`4.2-red.log`).
- 4.2 green: T7 lead layout, dtypes and optional fields on a segment-major `[3, 2, 2]` batch; T8 a non-canonical program and a HIRE count beyond a `hire_limit = 3` teacher's capacity both raise `GrammarReplayError` (support group named); T9 index-then-concat over three layouts and chunk sizes 1–3 is exact, a single chunk is returned as is, and mismatched presence, keys and grammar raise in both orders; T9b the signature tracks a flipped table entry and `hire_limit`; T10 `nbytes` equals the tensor sum and `rows × 102,208`, and Isaiah's type counts its optional continuation logits; T10b pins 1,674,575,872 B and 837,287,936 B; T11 per-segment chunks joined by `concat` match one call (`assert_close`, since batch sizes differ); Isaiah's model rejects Kaggriculture targets with `TypeError` before any encode.
- `just py-prepare` (`4.1-py-prepare.log`): format, lint, mypy and 1,343 passed with 4 skipped (3 hardware/backend, 1 native grammar binding). docs-fresh first flagged `docs/model-architecture.md`; after the Kaggriculture teacher bullet and the KL dtype note were added, it passes. For 4.2 (`4.2-py-prepare.log`): format, lint, mypy over 59 files and 1,360 passed with the same 4 skips. The model doc gained the targets bullet and the protocol typing. `ppo.py` and `distributed.py` changed annotations only and README names no concrete target type, so docs-fresh was acknowledged with `DOCS_CURRENT=1`.

## Deviations from the brief (4.1)

- **T1 non-negativity:** FP32 rounding gives values down to −5.6e-8 where the true KL is of order 1e-5, so the test bounds `kl ≥ −1e-6` instead of `≥ 0`.
- **T4 head chunking:** the brief asked for `torch.equal`, but head-chunked market-kind logits differ in the last bits (CPU GEMM blocking varies with row count). The test uses `assert_close`, as the heads chunking test already did. T2's FP32 exactness, which the brief made a stop condition, holds.
- **T3 HIRE case:** the oracle uses a small `hire_limit = 5` case (budgets 2, 3, 1, 1, all exhausted mid-queue) instead of the 238–241-actor `hire_capacity` case. The one-value-per-row oracle would need about 5,000 variant rows of 241 frames there. The capacity-blocked HIRE sites are asserted to be exercised.

## Limits and gaps

- Everything ran on CPU with tiny models and the synthetic grammar tables (Task 1.4 binding pending). BF16, compile on CUDA and GPU memory/time are unmeasured.
- The per-seat KL sums up to 241 × 5 + 11 × 4 conditional KLs with Isaiah's coefficient 0.005. This scale difference is recorded, not tuned.
- The grammar digest is taken at construction. An in-place edit of a table buffer after that would not change the signature; the combined path (4.3) also compares tables with `torch.equal`.
- The cache estimate (1.56 GiB per 2-rank rollout) is arithmetic; GPU peak allocated and reserved memory are unmeasured until Task 6.1.
- 4.3 (model methods and trainer wiring) and 4.4 (configs) are not recorded here yet.
