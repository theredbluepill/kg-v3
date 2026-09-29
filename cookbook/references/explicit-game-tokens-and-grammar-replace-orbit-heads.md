---
type: "Reference"
title: "Explicit game tokens and grammar replace Orbit heads"
description: "Restored independent player context and entity/player/plan batched heads pass local checks; earlier GPU measurements remain scoped to the previous model."
tags: ["kaggriculture-v3", "adaptation", "model"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources: [{"resource": "user-directive:2026-09-29:restore-player-token"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/player-token-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/player-token-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/player-token-2026-09-29/model-count.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/player-token-2026-09-29/py-prepare-final.log"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/player-token-2026-09-29/before/manifest.json"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/scratch-plan-alignment.md"}, {"resource": "external:https://github.com/IsaiahPressman/kaggle-orbit-wars/blob/32b3ec900ad406eedd965f53a1a0f4490d31c589/python/owl/model/stateless_transformer_v1.py"}, {"resource": "user-directive:2026-09-29:reject-python-autoregressive-pipeline"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/rejected-decoder/cookbook-reference.md.txt"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_model.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:record-every-adaptation"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/model/config.py"}, {"resource": "repository:python/owl/model/factory.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/types.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/gpu_sampling_grammar.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/model/kaggriculture.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/docs/kaggriculture-model.md"}, {"resource": "repository:docs/model-architecture.md"}]
---

# Explicit game tokens and grammar replace Orbit heads

## Current model and adopted repair

The owner rejects Python-driven autoregressive heads and directs efficient reuse
of the whole Isaiah pipeline. After the scratch/plan audit found a missing
independent player readout, the owner explicitly requests:

> 修復差異吧 player token

Source: `user-directive:2026-09-29:restore-player-token`, the direct reply to the
alignment discussion. The current `batched_entities_v2` model restores that
readout: one independently learned player token joins each legal seat view's
scratch, plan, critic, entity and resource tokens in the reused Isaiah trunk.
Initial token parameters are shared across seat views; hidden representations
depend only on the relevant legal observation. No opponent identity or
between-turn state is introduced.

The actor input is now `[entity, player, plan]`, restoring the three Isaiah
readouts. One `3D -> D` projection is evaluated as an entity term plus a shared
player/plan term, with the bias included once. Market inputs use
`[plan, player, plan]` before existing ordinal/field conditioning. This keeps
one trunk pass and batched heads, without a GRU, frame scan, or sampled U/A
coordinator. The player parameter is explicitly initialized with standard
deviation `D**-0.5`, registered as an input layer, and included in token masks
and work counts.

The default constructor has **8,353,727 parameters**, measured with Torch 2.9.0:
**65,792** more than the prior model (256 token values plus 65,536 projection
weights). Explicit `batched_entities_v1` configs are rejected. Old weights fail
strict loading for the missing player token and changed input-projection shape.
Fresh configs may omit the decoder version and use the current default; this
does not migrate old weights. The schema version marks incompatibility, not a
staged product delivery.

## Integration inventory and semantic boundaries

The repair changes `python/owl/model/kaggriculture.py`,
`configs/model/kaggriculture.yaml` and `tests/kaggriculture/test_model.py`.
README, model docs and the optional coordination design/reference are reconciled
to current parameter/schema/token facts and historical benchmark scope.
The prior dirty files are preserved under `ops/player-token-2026-09-29/before`.

The full-suite run encountered a concurrently changed evaluation logger.
`tests/scripts/test_run_ppo.py` had an exact-dictionary expectation missing the
new `eval/promoted` and `eval/promotion_threshold` fields; only that assertion
is repaired here, checking 0.0 and 0.7 for the 25%-win fixture. The runner's
telemetry implementation is separate work and was not changed by this repair.

Preserve independent legal seat observations, full actor/market/STOP capacity,
queue semantics, native masks, sampling/replay densities and critic objectives.
Rust buffers, action ABI, rewards, canonical PPO, GAE, optimizer and selection
rules remain outside this repair. V2 supplies engine/schema/grammar/reward
semantics, never the neural implementation. A seat-local player query is not an
opponent identity.

## What is aligned and what remains adapted

Scratch/plan roles, normal initialization, shared attention and the three-part
actor readout follow pinned Isaiah
`32b3ec900ad406eedd965f53a1a0f4490d31c589`. Kaggriculture still uses independent
private-view encoding and shared initial query weights, rather than putting
four absolute-player plan rows into one sequence. Game stems and market heads
remain adaptations. This is not a claim of whole-model or learned-policy
equivalence.

The earlier missing-player audit is retained in
`ops/coordination-design-2026-09-29/scratch-plan-alignment.md` as evidence about
its hashed pre-repair source. The repair removes that specific omission; it
does not establish improved playing strength or a need for the optional U/A
coordinator.

## Actual checks and remaining qualification

`uv run --no-sync pytest tests/kaggriculture/test_model.py -q` passes **33 cases**.
New checks isolate direct player-readout effects on actor/market probabilities,
verify finite nonzero policy gradients, compare factored and concatenated
three-part projections, check real attention masks/token counts with mixed
padding and zero/four scratch tokens, and reject old configs/weights. Existing
private-seat isolation, sampling/replay, full-capacity, inactive-rank backward,
fullgraph capture and checkpoint reload checks pass. Capture uses Dynamo's
eager backend, not an Inductor/CUDA performance measurement.

The actual canonical recipe, invoked through the available runner as
`uvx --from rust-just just py-prepare`, passes **811 tests, with three declared
hardware/backend skips**, formatting, Ruff, Python 3.11 syntax, strict mypy over
58 files and documentation freshness. This whole-workspace result includes
concurrent evaluation tests; it is not a count of tests added by this task.
An independent source review found no blocker in the repair diff.
Logs, count, source custody, scope and retained first-attempt failure are in
`ops/player-token-2026-09-29/results.md`.

No new GPU benchmark, dense-game capacity qualification, paid learner run or
policy-strength evaluation was performed. Prior GPU numbers do not qualify the
changed model.

## Historical decoder and GPU evidence

The previous `batched_entities_v1` model had 8,287,935 parameters and 29 focused
CPU tests. Its heads-only two-PRO6000 run measured 1,774.752 game SPS over 5,120
transitions and 20 optimizer steps/rank, with finite gradients on all 198
parameter tensors, native admission and synthetic 17/241-actor checks. The
fused-pipeline runs reached 4,096 environments/GPU; 8,192 failed with CUDA OOM.
These short-workload receipts and their BF16/actor-population limits remain in
`ops/gpu-sps-2026-09-29/results.md`, bound to that previous source.

The rejected 8,294,450-parameter Python-frame/GRU model measured 310.819 and
282.253 game SPS in separate two-PRO6000 runs. Its original contract/source/docs
are frozen under `ops/gpu-sps-2026-09-29/rejected-decoder`. Prior CPU correctness,
reset-per-observation state and transformer ancestry did not satisfy the later
efficient-pipeline requirement. Those historical measurements do not prove
learning improvement or provide a controlled comparison for the player repair.
