---
type: "Reference"
title: "PPO trainer seams map any schema and alarm on replay drift"
description: "Rebuild Tasks 3.1, 3.6 and Phase 4 prep: schema-generic observation mapping, a default-on first-minibatch log-ratio alarm whose units follow ppo_clip_mode, and a TeacherTargets protocol whose concat keeps an inherited first-chunk asymmetry, proven on Orbit types with CPU TDD and 941 Python passes; GPU noise, real multi-rank runs and Kaggriculture use remain unverified."
tags: ["kaggriculture-v3", "adaptation", "diagnostics"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "repository:python/owl/train/ppo.py"
  - resource: "repository:python/owl/model/teacher_targets.py"
  - resource: "repository:python/owl/model/stateless_transformer_v1.py"
  - resource: "repository:python/owl/model/__init__.py"
  - resource: "repository:tests/owl/train/test_ppo_observation_mapping.py"
  - resource: "repository:tests/owl/train/test_ppo.py"
  - resource: "repository:tests/owl/model/test_teacher_targets.py"
  - resource: "repository:README.md"
  - resource: "repository:docs/model-architecture.md"
  - resource: "repository:ops/rebuild-2026-09-29/plan.md"
  - resource: "repository:ops/rebuild-2026-09-29/3.1-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.1-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.1-python-suite.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.6-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.6-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.6-python-suite.log"
  - resource: "repository:ops/rebuild-2026-09-29/3.6-review-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/stream-c-review-python-suite.log"
  - resource: "repository:ops/rebuild-2026-09-29/stream-c-r2-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/stream-c-r2-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/stream-c-r2-python-suite.log"
  - resource: "repository:ops/rebuild-2026-09-29/4-prep-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/4-prep-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/4-prep-teacher-tests.log"
  - resource: "repository:ops/rebuild-2026-09-29/4-prep-python-suite.log"
---

# PPO trainer seams map any schema and alarm on replay drift

This note covers three game-neutral changes to Isaiah's trainer on the clean base, serving the
[[../decisions/restart-the-port-from-isaiahs-clean-base|restart Decision]]. They are rebuild Task 3.1, Task 3.6 and the Phase 4 `TeacherTargets` refactor in `ops/rebuild-2026-09-29/plan.md`. They follow principle I11 (refactor, don't add shims). Kaggriculture types don't exist yet, so each change is proven on Isaiah's Orbit types. A cookbook search found the requirement in the
[[compiled-gemm-template-overflows-above-2-21-rows|compiler overflow Reference]], and the historical
[[shared-ppo-adapts-game-batches-without-a-second-loop|shared PPO Reference]] describes the reference branch's own `_map_observation`. No current rebuild contract covered these seams.

## Contract and adaptation inventory

**Task 3.1: schema-generic observation mapping** (`python/owl/train/ppo.py`).
- `_map_observation(obs, fn)` iterates `type(obs).model_fields` and rebuilds `type(obs)`. Tensors map through `fn` and unset optionals stay `None`. Action masks map through an explicit `isinstance` dispatch (`_map_action_mask`), and any other field type raises `TypeError`.
- `_copy_observation_(dst, src, copy, context=...)` handles the in-place copies. It requires matching batch types, and it fails when an optional buffer is present on only one side or the action-mask types differ.
- `_obs_segment_major`, `_obs_index`, `_flatten_obs_time`, `_obs_to_device`, `_copy_obs_time_step` and `_copy_obs_to_device_` are now one-line specializations. `_OBS_TENSOR_FIELDS`, `_OBS_OPTIONAL_TENSOR_FIELDS`, `_map_optional_obs_tensors` and the per-operation action-mask wrappers are removed.
- CPU `_obs_to_device` still clones. Accelerator transfers still do not.
- `_PPORolloutBuffer` allocation stays Orbit-specific, as the task scope requires.

**Task 3.6: first-minibatch log-ratio alarm** (`ppo.py`, `README.md`).
- `PPOConfig.first_minibatch_logratio_limit` defaults to `0.05` nats. It must be finite and positive; `None` disables it.
- `PPOTrainer._update` checks the first minibatch of every update, after its backward and before any optimizer step. It reads the existing policy-weighted `logratio_mean` loss metric, which is already all-reduced under DDP.
- The raised `RuntimeError` reports the observed mean, the limit, the rollout `[segments, horizon, players]` shape, the minibatch width and every set observation tensor shape (the token counts), including action-mask tensors as `action_mask.can_act` and `action_mask.max_launch`, and it points to the compiler overflow Reference. Unset optional tensors are omitted.
- The limit's units follow `ppo_clip_mode`, because the alarm reads the loss's own log-ratio. `per_player` sums entity log-probs, so the limit bounds the joint action: a coherent drift of `d` nats on each of `K` acting entities reads as `K * d`. `per_entity` averages entity log-ratios, so the same drift reads as `d`. The `PPOConfig` field comment and README say so.

**Phase 4 prep: `TeacherTargets` protocol** (`python/owl/model/teacher_targets.py`).
- The protocol is `index(indices) -> Self` plus classmethod `concat(chunks) -> Self`, both along the segment dimension.
- `CachedTeacherDistillationTargets` subclasses it explicitly, so mypy checks the signatures. The bodies of `index_teacher_distillation_targets` and `concat_teacher_distillation_targets` moved into the methods unchanged, apart from `self`/`cls` construction.
- The free functions and their `owl.model` exports are removed. `ppo.py` calls `teacher_targets.index(idx)` and `type(chunks[0]).concat(chunks)`, and `owl.model` exports `TeacherTargets`.
- `docs/model-architecture.md` describes the protocol. It and the docstrings state the inherited `concat` behavior exactly: the first chunk's layout wins, a later chunk missing one of its optional targets raises, and a target carried only by later chunks is silently dropped. `python/owl/model/base.py` is untouched because another stream owns it.

Future game batches can pass through the trainer's mapping helpers if they are pydantic models whose fields are tensors, `None` or the listed action-mask types. A new mask type must be added to the explicit dispatch in both `_map_observation_value` and `_copy_observation_`. A future teacher-target type implements the protocol instead of adding free functions.

## Verification

Each change went test-first on this version:

| Change | Red | Green | Full suite |
|---|---|---|---|
| 3.1 | 6 failed, 150 passed | 218 passed (new file + `test_ppo.py`) | 886 passed, 3 skipped |
| 3.6 | 9 failed, 4 passed | 13 passed | 899 passed, 3 skipped |
| Phase 4 prep | collection `ImportError` | 25 passed; Isaiah's 18 teacher tests pass | 924 passed, 3 skipped |
| Codex review edits | 7 failed (strengthened config tests, alarm field removed) | 28 teacher-target and 20 log-ratio tests pass | 934 passed, 3 skipped |
| Codex re-verification edits | 7 failed (shape diagnostics omitted action-mask tensors) | 27 shape-diagnostic and log-ratio tests pass | 941 passed, 3 skipped |

The full suite is `uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q`; the repository has no slow-marked tests.

- **3.1:** the oracle is an extracted successful-path copy of Isaiah's 32b3ec9 helpers inside the test file, not a literal freeze: its tensor field list derives from the current `ObsBatch` schema, and its copy helper omits the original validation. It checks successful calls only. The Orbit cases (3 seeds × optional fields set/unset × 3 action-mask types) passing against the old helpers in the red run is what validates the oracle. A test-local batch type with different field names round-trips through every helper, and a monkeypatched accelerator transfer shows the no-clone path.
- **Alarm shape diagnostics:** Codex's re-verification found the message listed only top-level tensors, so a mask-related mismatch would hide mask shapes. `_observation_tensor_shapes` now also lists each action-mask tensor; tests cover all three Orbit mask types with optionals set and unset, a schema with different field names, and the alarm message itself (`stream-c-r2-red.log`, `stream-c-r2-green.log`).
- **3.6:** the original 4 red passes were invalid-limit cases that unknown-field rejection already satisfied. After Codex's review they assert the exact Pydantic error (`greater_than` for 0 and negatives, `finite_number` for NaN and ±inf) on this field, and a custom positive limit is accepted; with the field removed all 7 fail (`3.6-review-red.log`). A replay-shifted fake model (±0.2 nats, with and without gradient accumulation) raises after one evaluation, with zero optimizer steps and bit-identical parameters. No existing PPO test trips the default-on alarm. With all 44 action entities acting, a 0.002-nat drift per entity aborts `per_player` at +0.0880 nats but passes `per_entity`, and a 0.06-nat drift aborts `per_entity` at +0.0600. A two-fake-rank unit test of `_distributed_ppo_loss_metrics` (monkeypatched sum and max reductions) shows unequal active-player counts reduce to one global weighted mean in both clip modes: 0.3 nats on 1 of 4 players gives 0.075 on both ranks, not 0.15.
- **Phase 4 prep:** index matches direct slicing, and index-then-concat is exact for five optional-target layouts and three chunk sizes. Both chunk orders are pinned for action params, continuation logits and winner probabilities: a later chunk missing a target raises, while a first chunk missing it drops the later chunk's target.
- The Phase 4 prep `just py-prepare` run (superseded as the latest result by the rows above) passed formatting, lint, 3.11 syntax, mypy over 49 source files, 924 tests with 3 hardware skips, and docs freshness. That freshness check only diffs the working tree against `HEAD`, so it was vacuous after the commits. At branch level, README (train) and `docs/model-architecture.md` (model) were both updated.
- The latest `just py-prepare`, run before committing the re-verification edits, passes formatting, lint, 3.11 syntax, mypy over 49 source files, 941 tests with 3 hardware skips, and docs freshness; that freshness check was not vacuous, because `ppo.py` and README were both modified in the working tree.
- No Rust source changed and `cargo test` was not run. No training, evaluation or GPU run was performed.

## Limits and gaps

- **Orbit behaviour.** Orbit output is unchanged except in failure paths. When several optional fields mismatch, the copy error now names the first one in declaration order (`fleet_target` before `player_features`). The device-copy action-mask error now says `destination action-mask type mismatch` instead of `obs action-mask ...`. The alarm itself is new default-on behaviour for every Orbit config: a run that previously trained on corrupted replay now aborts.
- **The 0.05-nat threshold is unmeasured.** It is the plan's value; BF16/compiled replay noise on GPU has not been measured here. Under `per_player` it bounds the summed joint action, so coherent small per-entity noise over many entities could trip it; Codex's synthetic probe gives 0.0723 nats from 241 entities at 0.0003 each. Real false positives have been neither shown nor excluded. The multi-rank reduction is tested with fake ranks only, not real processes. Signed cancellation across samples or ranks can hide drift. The check adds one host sync per update.
- **Script-local copies remain.** `scripts/run_ppo.py` and `scripts/benchmark_checkpoints.py` keep Orbit-specific `_obs_to_device` copies on the evaluation path. `AGENTS.md` still cites the removed `_OBS_TENSOR_FIELDS` as its example of sanctioned dynamic access.
- **Pre-existing concat asymmetry, preserved and pinned.** `concat` raises when a later chunk lacks an optional target that the first chunk has. When the first chunk lacks it and a later one has it, the target is silently dropped. Tests pin both orders; rebuild Phase 4 decides whether to validate symmetrically.
- **Trainer typing is still concrete.** Annotations still name `CachedTeacherDistillationTargets` because `base.py` types the model methods concretely.

Reopen the threshold when the Phase 6 GPU smoke records first-minibatch log-ratios under BF16 and compile in each clip mode used, and reopen the mapping dispatch when Task 1.5 defines the Kaggriculture batch and mask types.
