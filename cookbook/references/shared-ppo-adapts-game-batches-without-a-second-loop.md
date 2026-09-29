---
type: "Reference"
title: "Shared PPO adapts game batches without a second loop"
description: "One shared Isaiah PPO trainer carries Kaggriculture batches; its core is upstream-equivalent, the recipe is partially aligned, and teacher, cadence, CUDA-fault and GPU-qualification gaps are tracked in ops."
tags: ["kaggriculture-v3", "adaptation"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-28"}
sources: [{"resource": "reference-branch:kg/reference-2026-09-29/ops/gap-closure-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/cookbook-cleanup-2026-09-29/before.sha256"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/infrastructure-audit.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/infrastructure-audit.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/semantic-audit.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/verification.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-from-bc-2026-09-29/upstream-audit/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/continuous30-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/interruption.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/default4096-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-promotions-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-promotions-2026-09-29/wandb-verification.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-promotions-2026-09-29/ddp-contract.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-promotions-2026-09-29/verify_promotions.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-promotions-2026-09-29/py-prepare.log"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/final-checks/py-prepare.log"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank_32env_fixedppo.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/analyze_components.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/profile_components.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/cuda_contract_probe.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_training.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:tests/owl/train/test_distributed.py"}, {"resource": "repository:tests/owl/model/test_model_config_files.py"}, {"resource": "repository:scripts/regenerate_test_fixtures.sh"}, {"resource": "repository:justfile"}, {"resource": "repository:docs/containerization.md"}, {"resource": "repository:Dockerfile.kaggle"}, {"resource": "repository:Dockerfile"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_4rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:record-every-adaptation"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/game.py"}, {"resource": "repository:python/owl/rl.py"}, {"resource": "repository:python/owl/model/base.py"}, {"resource": "repository:python/owl/model/config.py"}, {"resource": "repository:python/owl/model/factory.py"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "repository:python/owl/train/ppo.py"}, {"resource": "repository:python/owl/train/distributed.py"}, {"resource": "repository:python/owl/train/optimizer.py"}, {"resource": "repository:python/owl/train/utils.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_last_best.py"}, {"resource": "repository:python/owl/utils.py"}, {"resource": "repository:python/owl/train/logging.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/scripts/benchmark_kaggriculture.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/model/kaggriculture.yaml"}, {"resource": "repository:.gitignore"}, {"resource": "repository:pyproject.toml"}, {"resource": "repository:uv.lock"}, {"resource": "repository:python/owl/agent/agent.py"}, {"resource": "repository:python/owl/model/lora.py"}, {"resource": "repository:python/owl/int8_emulation.py"}, {"resource": "repository:scripts/benchmark_checkpoints.py"}, {"resource": "repository:python/owl/model/stateless_transformer_v1.py"}, {"resource": "repository:python/owl/model/recurrent_transformer_v1.py"}]
---

# Shared PPO adapts game batches without a second loop

## Current state

`scripts/run_ppo.py` and `python/owl/train/ppo.py` remain the one trainer. Kaggriculture batches flow through Isaiah's collection, storage, GAE, minibatch, optimizer and incumbent lifecycle; no second PPO loop exists.

The recipe is partially aligned to pinned upstream `scaling_6m.yaml` (`32b3ec9`). Aligned so far: Muon/AdamW optimizer and warmup/cosine scheduler, compatible PPO coefficients, compiled GAE/loss, trunk compile mode and the 20M-step checkpoint cadence. Those alignments pass CPU optimizer/config checks only. Known gaps:
- The GPU YAMLs still run the historical 4096 env/rank, spm 1, accumulation 2 cadence: 2,048 optimizer steps per iteration against upstream's 16, with four-times-faster scheduler exposure per env step. The [[../decisions/recipe-choices-align-to-isaiah-without-owner-escalation|alignment Decision]] resolves it toward Isaiah's multi-GPU rule.
- Teacher action/cache/value adapters are missing and config rejects them. Kaggriculture replay export is also missing, so `eval_replay_games` is 0.
- Muon grouping and the FlashAttention requirement differ from upstream.
- The CUDA illegal memory access in the BC-initialized rollout is now explained: a Torch Inductor GEMM template overflows 32-bit offsets above 2^21 rows ([[compiled-gemm-template-overflows-above-2-21-rows|compiler overflow Reference]]).
- W&B reports a crashed run as finished, a behavior inherited from upstream.
- Every evaluation reuses the same seed (`seed=cfg.env.seed`).

The itemized inventory is `ops/isaiah-alignment-2026-09-29/results.md`; the repair order is `ops/gap-closure-2026-09-29/plan.md`. Copying YAML keys does not establish training equivalence: compare resolved configs and derived workload (`semantic-audit.json`).

## Retained core versus integration

Against upstream `32b3ec9`, the GAE, attention and shared actor-projection modules are byte-identical. The PPO loss, update/minibatch/normalization, optimizer/scheduler and `TransformerBlock` bodies match after annotation normalization (`ops/isaiah-alignment-2026-09-29/infrastructure-audit.md`). No upstream file is deleted, and upstream tests are only extended. On 2026-09-29 the full suites pass on current source: 814 Python (3 hardware skips) and 155 Rust (2 ignored); Isaiah's Orbit tests are included. These are CPU/software checks, not GPU or learning qualification. Game-specific storage, model dispatch, evaluation adapters and config semantics are the integration surface.

## What the sources record

- `python/owl/game.py` defines explicit game batch/mask/environment unions and a factory. `python/owl/rl.py` adds supported config unions and rejects mismatched families, non-two-player Kaggriculture and Orbit-only reward modes.
- Rollout storage gains Kaggriculture feature/context/entity/action tensors. Observation/action mapping helpers carry them through the existing collection, storage, replay, optimization and device-transfer path; `_map_observation` is equivalent to the upstream helpers.
- Config guards require joint per-player clipping and the MSE critic, and reject teacher distillation, Orbit replay export, per-frame clipping and winner-CE for Kaggriculture. The model factory rejects cross-game specs.
- Typed generic model API (`BaseModelAPI[...]`) across model, distributed, optimizer, LoRA/int8 and agent helpers. The Orbit agent explicitly refuses Kaggriculture checkpoints; Kaggriculture serving is unqualified.
- `python/owl/train/logging.py` selects W&B project `kg-v3` for Kaggriculture and keeps `orbit-wars` for Orbit.
- `configs/kaggriculture.yaml` is the CPU functional recipe (64-step horizon, two environments, FP32), not a measured optimum.
- `.gitignore` excludes run output, weights, traces and credentials. `pyproject.toml` moves `kaggle`/`kaggle-environments` into an optional `reference` extra with `kaggle-environments==1.32.7`; `justfile` test recipes request that extra. Orbit reference fixtures on this host were regenerated with 1.32.7. Native-build cache keys include `engine_rs/src/**/*.rs` and `engine_rs/Cargo.toml`, so vendored engine edits rebuild the extension.

## Game-specific semantics in the shared path

Source review found three integration defects; the current source repairs them. These are correctness repairs, not policy-improvement evidence:
- Kaggriculture evaluation decides W/L/D from final native banks. Summed shaped reward cannot pick the winner: an allowed .8 economic cap could otherwise score the real bank winner `.2-.8=-.6`.
- Artificial truncation keeps the real economic reward earned on that transition and adds the critic bootstrap; Orbit keeps upstream's `rewards=0`.
- Rank seed streams use `seed+rank` with `seed_stride=world_size` through every reset; a `rank*n_envs` offset produced overlapping streams.

## Incumbent selection, resume and execution controls

The previous-best logic stays intact: evaluate against last best, promote at win rate **>= .7**, keep separate current/best checkpoints, and resume the current model/optimizer separately from the last-best opponent. The only runner change moves evaluation logging after the promotion write/barrier and adds `eval/promoted`, `eval/promotion_threshold` and `eval/games`. `tests/kaggriculture/test_last_best.py` exercises native evaluation, the .699/.7 branches and current/incumbent resume pairing. Periodic evaluation now follows upstream's 20M-step `checkpoint_freq`; short diagnostics need an explicit override. `OWL_ALLOW_CPU_DDP=1` enables Gloo/CPU diagnostics only; normal multi-rank runs require CUDA.

Evaluation runs on rank 0 while the other ranks wait for the broadcast, with no per-turn progress log. An idle or collective-waiting GPU during evaluation is not a hang. A 4,096-game evaluation did exceed the 600 s NCCL broadcast timeout; optional `rl.eval_n_games` now sets the evaluation count independently (unset keeps upstream behavior). A smaller count changes selection variance.

## Measurement harness

`scripts/benchmark_kaggriculture.py` launches the canonical runner with warmup and measured updates. It records source/binary hashes, effective config, driver/device/CUDA metadata, phase times, game-transition and learner-seat denominators, synchronized optimizer-step deltas and KL-stopped updates, and a measured-window `nvidia-smi` series. It rejects nonfinite metrics. Only the benchmark invocation disables periodic checkpoints/evaluation. Profiling wrappers in `ops/gpu-sps-2026-09-29/` are historical to the old environment.

## Historical measurements

These measure earlier models or the historical 4096/spm 1/accumulation 2 recipe. They do not qualify the aligned recipe or dense BC positions.
- Rejected Python-frame/GRU decoder: 310.8 / 282.3 game SPS. Heads-only replacement: 1,774.8 game SPS (`ops/gpu-sps-2026-09-29/results.md`).
- Fixed-minibatch environment sweep on two PRO6000: 3,627 / 5,703 / 6,363 / 6,947 / 7,167 learner-seat SPS at 8 / 32 / 128 / 2,048 / 4,096 envs/GPU; 8,192 OOMs (`ops/gpu-sps-2026-09-29/env-sweep/`).
- Two-rank promotion/resume contract probe and online run `spoon/kg-v3/vw4ohx4a` (`ops/ppo-promotions-2026-09-29/results.md`).
- Uninterrupted half-hour continuation: nine full updates, 3,380.64 game SPS over the last six. All nine 128-game evaluations ended at zero banks, 50% and no promotion, so this is execution evidence, not strength (`ops/continuous30-2026-09-29/results.md`, `ops/default4096-2026-09-29/results.md`).

## Interpretation and consequence

Reuse the starter's PPO machinery, DDP, scheduling, optimizer grouping and telemetry without hard-coding Orbit entity/action field names at transport boundaries. Evaluate the two-seat game with its own reward/termination definitions. Preserve behavior action densities from collection through update; changing the action representation must not silently turn joint probabilities into per-slot clipping.

## Container and remaining gaps

`Dockerfile` keeps the starter CUDA/uv/Rust build with kg-v3 paths and copies `engine_rs/` before Cargo resolution. `Dockerfile.kaggle` includes that path dependency, but its Orbit serving path does not support Kaggriculture (`docs/containerization.md`). Neither image has run on an NVIDIA host. Still unqualified: container builds, measured FlashAttention kernel use, Kaggriculture serving/export, the aligned recipe on GPU, and competitive strength. `verified-scoped` covers the source and check evidence above, not these targets. The pre-trim text of this note is preserved byte-exact in `ops/cookbook-cleanup-2026-09-29/before/`.
