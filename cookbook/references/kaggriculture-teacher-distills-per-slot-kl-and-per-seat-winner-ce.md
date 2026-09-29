---
type: "Reference"
title: "Kaggriculture teacher distills per-slot KL and per-seat winner CE"
description: "Phases 4.1-4.3 on CPU: replay-conditioned per-slot masked logits and a liveness-weighted per-slot KL(teacher || student), cached as KaggricultureTeacherTargets (102,208 B per seat row) with a grammar signature, a cached path bit-for-bit equal to the combined path, a live-seat-mean winner CE owned by the model, stateless PPO teacher dispatch and teacher/cache_bytes; eleven killed mutations. Since the Task 3.1 remainder (15ea55f, merged with Task 4.4 in kg/merge-3-1-3-5-c) the trainer and run_ppo launch/resume tests run and pass on CPU (T18 and the trainer-checkpoint test on a fake Kaggriculture env, T19b with create_env patched and a teacher checkpoint source). Task 4.4 has pinned the configs' teacher settings, reported the cache bytes at startup and required a teacher checkpoint at a fresh Kaggriculture launch (configs Reference). A later GPU component bundle at 8fde43c (synthetic inputs, fresh weights, BF16 and compiled trunk) found self KL near zero, perturbed KL positive, cached and combined paths bit-identical and a finite teacher-term loss backward, and timed target precompute; cross-chunk/minibatch equality, multi-rank, a native or GPU teacher-on trainer run stay open."
tags: ["kaggriculture-v3", "model", "training", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "repository:python/owl/model/kaggriculture_actor.py"
  - resource: "repository:ops/rebuild-2026-09-29/gpu-checks-2026-09-29/README.md"
  - resource: "repository:ops/rebuild-2026-09-29/gpu-checks-2026-09-29/summary.json"
  - resource: "repository:ops/rebuild-2026-09-29/results.md"
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
  - resource: "repository:scripts/run_ppo.py"
  - resource: "repository:tests/owl/train/test_loss.py"
  - resource: "repository:docs/kaggriculture-model.md"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:README.md"
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
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.3-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.3-mutations.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.3-py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/trainer-model/4.3-isaiah-suites.log"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-4-teacher-r1/report.md"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-r1-fixes/t19b_dryrun.py"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-r1-fixes/t19b-dryrun.log"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-r1-fixes/py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-r1-fixes/isaiah-suites.log"
  - resource: "repository:ops/rebuild-2026-09-29/teacher-r1-fixes/test-teacher-skips.log"
  - resource: "repository:ops/rebuild-2026-09-29/merge-teacher/skip-probe.log"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1.md"
  - resource: "repository:python/owl/model/kaggriculture_workload.py"
  - resource: "repository:tests/kaggriculture/test_configs.py"
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

**4.3 model methods and trainer wiring.**
- `KaggricultureTransformer`:
  - `supports_cached_teacher_distillation()` and `supports_cached_value_distillation()` return `True`.
  - `evaluate_actions_with_cached_teacher` admits before any kernel: stateless checks, the target type (`TypeError`), required targets, the grammar signature, slot keys, FP32/FP64 dtypes and shapes (`ValueError`). It then encodes the student once and returns `evaluate_actions`'s evaluation plus `action_kl`: `event [*lead,252,12]`, `per_player_entity`, zero `launch`, per-slot `components`, `target=None`.
  - `evaluate_actions_with_teacher` is the combined path. It checks the teacher type, `action_spec`, signature and per-table `torch.equal`, then feeds the no-grad teacher's row-layout logits straight to the student.
  - `teacher_value_cross_entropy` takes the per-seat CE over (self, opponent) and averages it over live seats.
- `BaseModelAPI.teacher_value_cross_entropy(student_log, teacher_probs, *, value_mask)`: its default is Isaiah's formula, moved verbatim from the removed `ppo._teacher_value_cross_entropy`. The `supports_cached_teacher_distillation` docstring is model-generic.
- `ppo.py`:
  - The value CE goes through `unwrap_model(self.model).teacher_value_cross_entropy(..., value_mask=batch_value_mask)`, without the two `.view_as` calls; those were no-ops for Orbit's segment-major cached tensors.
  - `_model_evaluate_actions_with_teacher` and `_model_evaluate_actions_with_cached_teacher` dispatch statelessly: with no hidden state they pass neither `hidden_state` nor `dones` (review P1-1).
  - `teacher/cache_bytes` is logged every iteration (0 without targets).
  - `set_teacher_model`'s error names cached action-KL support.
- `scripts/run_ppo.py`: `_teacher_obs_spec_for_student` dispatches by game with `isinstance`. Orbit keeps the `max_entities` rule, Kaggriculture requires equality, and a cross-game pair raises `TypeError`.
- `tests/owl/train/test_loss.py`: the value-CE test calls the base method (call syntax only).
- Docs: `docs/model-architecture.md` (its game-generic Teacher Distillation section scopes the discrete_targets-only cached action KL and the fixed-teacher launch-mode rule to Orbit, and names Kaggriculture's policy-core KL and grammar-signature rule; verification r1 P3), `docs/kaggriculture-model.md` (a Teacher conformance row), `docs/rl-api-specs.md` (target shapes), `README.md` (`teacher/cache_bytes`, the model-owned value CE, stateless dispatch).

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
| 4.3 (e) cached admission without the signature check | 2 failed (T15, grammar mismatch) |
| 4.3 (f) cached-teacher wrapper always passes `dones` | 1 failed (T15b) |
| 4.3 (g) value CE sums seats instead of the live-seat mean | 2 failed (T13, T14) |

- 4.2 red: collection `ImportError` (missing module); the Isaiah foreign-targets test failed with an `AttributeError` after encoding (`4.2-red.log`).
- 4.2 green: T7 lead layout, dtypes and optional fields on a segment-major `[3, 2, 2]` batch; T8 a non-canonical program and a HIRE count beyond a `hire_limit = 3` teacher's capacity both raise `GrammarReplayError` (support group named); T9 index-then-concat over three layouts and chunk sizes 1–3 is exact, a single chunk is returned as is, and mismatched presence, keys and grammar raise in both orders; T9b the signature tracks a flipped table entry and `hire_limit`; T10 `nbytes` equals the tensor sum and `rows × 102,208`, and Isaiah's type counts its optional continuation logits; T10b pins 1,674,575,872 B and 837,287,936 B; T11 per-segment chunks joined by `concat` match one call (`assert_close`, since batch sizes differ); Isaiah's model rejects Kaggriculture targets with `TypeError` before any encode.
- 4.3 red: 12 failed on missing methods, 2 skipped (`4.3-red.log`). Green:
  - **T12:** cached equals combined with `torch.equal` on every field (KL event, per-entity and per-slot components; teacher winner probabilities; student winner log-probs; value CE; student log-probs, entropies and values) for a segment-major batch with a dense row and an inactive seat.
  - **T13:** a `load_state_dict` copy gives exactly zero KL and the value CE equals the live-seat mean of the student's winner entropy.
  - **T14:** the CE is the mean over two live seats (not the sum), excludes a non-live seat, is 0 with no live seat (state weight 0), and sends no gradient to the teacher.
  - **T15:** cached admission rejects Isaiah's target type, missing targets, missing keys, a wrong shape, BF16, a wrong winner shape, a foreign grammar, `hidden_state` and `dones`, all with zero encodes. The combined path rejects an Orbit teacher, a different `action_spec`, different tables and an in-place table edit (same signature), also with zero encodes.
  - **Grammar mismatch that replay admits:** a teacher without HIRE in `market_kind` computes targets for a HIRE-free program without error. The student rejects them by signature. Re-stamped with the student's signature, they pass and change the `market_kind` KL, so the signature is the only guard.
  - **T15b:** both PPO wrappers accept a non-`None` `dones` for the Kaggriculture student and equal the direct call; on Isaiah's segment-major rollout they equal his direct calls with `dones`.
  - **T16:** the base value CE is `torch.equal` to Isaiah's formula on that rollout, and `test_loss.py` passes.
  - **T17:** seat 0's targets and KL are unchanged when seat 1's observation changes, while seat 1's KL changes. Unrelated calls in between leave the targets equal.
  - **T19a:** the obs-spec dispatch (equality, a constructed schema-4 mismatch, cross-game `TypeError` both ways).
  - **Refresh:** `_refresh_eval_model_from_weights` keeps the last-best's tables and signature, and gives zero KL against the student.
  - **Checkpoint keys:** the state dict holds no table or teacher keys.
- **T19b (verification r1 P2).** The r1 placeholder, a docstring and an unconditional `AssertionError`, is replaced by three skipped tests:
  - **Resume:** `run_ppo.main` resumes a run directory whose `checkpoint_final.pt` holds the student and whose `checkpoint_last_best.pt` holds a different model. The only `set_teacher_model` call is active and passes the session's last-best model, which is not the student, is in eval mode, has the saved last-best weights and the grammar's tables, and gives targets `torch.equal` to the saved model's. Both files have exactly `PPOTrainer.write_checkpoint`'s key set, with no teacher or table keys in the model state.
  - **Fresh launch from weights:** `--load-model-weights` activates a last-best teacher with the loaded weights, separate from the student, and zero KL against it.
  - **Trainer checkpoint:** after a teacher iteration fills the cache, `write_checkpoint` for the student and for the teacher writes exactly the fixed key set, `_checkpoint_metadata` accepts it and the model state holds no teacher keys. Its body mirrors T18 and has not run.
  - The launch pair follows Isaiah's `tests/scripts/test_run_ppo.py` pattern: a fake env and trainer, with last-best construction and loading left real. It loads the merged `configs/kaggriculture.yaml` (Task 3.4; it was on `kg/rebuild-configs` when the test was written) with the tiny model and `model_compile: none`. The config now loads and passes the workload headroom check, then `require_orbit_env` raises run_ppo's explicit "cannot run Kaggriculture yet" error, so the skip (`NEEDS_RUN_PPO_GAME_SEAM`) named the Task 3.1 run_ppo game seam and the Task 1.4 native env. The Task 3.1 remainder (`15ea55f`) removed the skip; the pair now patches `run_ppo.create_env` instead of `VectorizedEnv` and sets `eval_replay_games: 0`, with its assertions unchanged.
  - **Dry run (`t19b-dryrun.log`):** with `FullConfig` validation bypassed (`t19b_dryrun.py`, copied into `tests/kaggriculture/` for the run and removed afterwards), the launch pair passes on CPU. Four `run_ppo.py` mutations each fail one of them: resume loading last-best into a discarded model, resume activating the student, fresh launch building last-best from initialization, and fresh launch never activating. `run_ppo.py`'s SHA-1 is unchanged afterwards. The dry run bypasses the real config schema and the env constructor, so the first unbypassed run after the merges remains the check.
- r1 fixes `just py-prepare` (`teacher-r1-fixes/py-prepare.log`): format, lint, mypy over 59 files, 1,372 passed and 8 skipped (the 4 hardware/backend/binding skips, T18, the trainer-checkpoint test and the T19b pair). docs-fresh passes. Isaiah's suites: 1,048 passed, 3 skipped (`isaiah-suites.log`).
- 4.3 `just py-prepare` (`4.3-py-prepare.log`): format, lint, mypy over 59 files, 1,372 passed and 6 skipped (the 4 above plus T18 and T19b). docs-fresh passes. Isaiah's suites on their own: 1,048 passed, 3 skipped (`4.3-isaiah-suites.log`), the same count as the Task 3.1 baseline.
- `just py-prepare` (`4.1-py-prepare.log`): format, lint, mypy and 1,343 passed with 4 skipped (3 hardware/backend, 1 native grammar binding). docs-fresh first flagged `docs/model-architecture.md`; after the Kaggriculture teacher bullet and the KL dtype note were added, it passes. For 4.2 (`4.2-py-prepare.log`): format, lint, mypy over 59 files and 1,360 passed with the same 4 skips. The model doc gained the targets bullet and the protocol typing. `ppo.py` and `distributed.py` changed annotations only and README names no concrete target type, so docs-fresh was acknowledged with `DOCS_CURRENT=1`.

## Deviations from the brief

- **T1 non-negativity:** FP32 rounding gives values down to −5.6e-8 where the true KL is of order 1e-5, so the test bounds `kl ≥ −1e-6` instead of `≥ 0`.
- **T4 head chunking:** the brief asked for `torch.equal`, but head-chunked market-kind logits differ in the last bits (CPU GEMM blocking varies with row count). The test uses `assert_close`, as the heads chunking test already did. T2's FP32 exactness, which the brief made a stop condition, holds.
- **Order:** the protocol typing of §3.2 (`base.py`, `ppo.py`, the DDP adapter) landed with 4.2 because mypy rejected the new return type otherwise. Its Isaiah-side `TypeError` test was written first.
- **T19 split:** the obs-spec dispatch (T19a) and the last-best refresh are pure functions, so they run now. The resume and fresh-launch part (T19b) was written while the configs were unmerged; it ran once the Task 3.1 run_ppo game seam landed (`15ea55f`). Its checkpoint key-set check needs a real `PPOTrainer`, which the trainer-checkpoint test now supplies.
- **T3 HIRE case:** the oracle uses a small `hire_limit = 5` case (budgets 2, 3, 1, 1, all exhausted mid-queue) instead of the 238–241-actor `hire_capacity` case. The one-value-per-row oracle would need about 5,000 variant rows of 241 frames there. The capacity-blocked HIRE sites are asserted to be exercised.

## Limits and gaps

- The Phase 4.1–4.3 checks above ran on CPU with tiny models and the synthetic grammar tables (Task 1.4 binding pending).
- **Later GPU component evidence** (GPU checks bundle at `8fde43c`, one RTX PRO 6000; `ops/rebuild-2026-09-29/results.md`, "GPU checks bundle (component)", checks 3 and 4; evidence in `gpu-checks-2026-09-29/`). Preset model with fresh weights, synthetic observations and grammar tables, BF16 autocast and the compiled trunk:
  - **C3 smoke** (mid and dense, ATEN-only and default GEMM backends): self KL per-row mean 1.4e-7–9.8e-7 and max ≤ 8.3e-6; a +5 % noise copy gave positive finite KL (per-row mean 7.1e-5 mid, 5.4e-4 dense); cached and combined paths bit-identical for self and perturbed teachers, with targets and replay on the same 256-row batch; a 1,024-row PPO-shaped loss with both teacher terms gave a finite loss and 210/210 finite gradients. The executed value term there was 0.25·MSE, not the declared 0.5·MSE.
  - **C4 timing** (ATEN-only): `compute_teacher_distillation_targets` on a rank's rollout took 1,012 ms (mid) and 1,741 ms (dense) at 16,384 rows, with trunk chunk counts matching the guard; model-only component timing.
  - The small fresh-weight head gain does not qualify the KL or log-ratio margins for trained policies.
- The per-seat KL sums up to 241 × 5 + 11 × 4 conditional KLs with Isaiah's coefficient 0.005. This scale difference is recorded, not tuned.
- The grammar digest is taken at construction. An in-place edit of a table buffer after that would not change the signature; the combined path (4.3) also compares tables with `torch.equal`.
- The cache estimate (1.56 GiB per 2-rank rollout) is arithmetic. C4 recorded component peaks at the 2-rank shape (40.70 GiB allocated in the dense cached-teacher PPO step, 69.21 GiB reserved in mid target precompute), but not the cache's own footprint or a trainer-integrated rollout; those wait for Task 6.1.
- **Phase 4's CPU scope closes with the 3.1/3.5 merge** (brief §6, review P2-4 kept it open until T18 and T19b ran). The integration merge (base `b8747b6`, with Tasks 3.1 model side, 3.2/3.3, 3.4 configs and 1.3) landed the configs and the Task 3.2 value-mode guards. The last two open items are now closed:
  - **T18, the trainer-checkpoint test and T19b's launch/resume pair run since `15ea55f`.** The earlier probe (`merge-teacher/skip-probe.log`) failed all four on the missing `KaggricultureActionMask` mapping and run_ppo's stop. With the Task 3.1 remainder they pass unskipped inside full `just prepare` (`3.1-rest2/claude-review/prepare.log`); an unmapped Kaggriculture `can_act` or int32 rollout token storage fails T18 and the trainer-checkpoint test (Claude's review mutations). They run on CPU with a fake env and tiny model, not a native or GPU teacher run.
  - **Task 4.4** (configs) is merged (`kg/rebuild-4-4` `4a662ad`, landed through `kg/merge-4-4-c`); see the [[kaggriculture-configs-follow-isaiahs-scaling-6m-recipe|configs Reference]]'s teacher section. It pins the four configs' teacher fields to `scaling_6m`, keeping the 128-segment chunk undivided per rank as Isaiah does. It reports `TEACHER_TARGET_BYTES_PER_ROW` × rollout rows in the startup headroom line, which matches T10b's totals. It also makes the teacher checkpoint (`--load-model-weights` or `rl.teacher_init`) a required input for a fresh Kaggriculture `last_best` launch. The 3.1/3.5 merge (`kg/merge-3-1-3-5-c`) runs both together.
- The `teacher/cache_bytes` metric is this rank's value, not reduced across ranks.
- Equality of teacher targets across the teacher chunk and the minibatch under BF16 autocast and the compiled trunk, real multi-rank runs and full trainer integration are unverified until Phase 6. The C3 cached-versus-combined equality used one batch for both.
