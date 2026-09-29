---
type: "Reference"
title: "Kaggriculture configs follow Isaiah's scaling_6m recipe"
description: "Task 3.4 configs deliberately align to Isaiah: 2-rank 128 envs/spm 8 and 4-rank 64/spm 4 keep scaling_6m's global batch, 16 optimizer steps and 16,384 env steps per iteration, with its optimizer, PPO, teacher and compile settings; run_ppo startup checks every forward against the model's GEMM chunking before allocation and prints bounded headroom. Section-schema and CPU tested only; FullConfig loading waits for Task 3.1."
tags: ["kaggriculture-v3", "training", "adaptation"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:configs/kaggriculture_2rank.yaml"}, {"resource": "repository:configs/kaggriculture_4rank.yaml"}, {"resource": "repository:configs/kaggriculture.yaml"}, {"resource": "repository:configs/model/kaggriculture_cpu.yaml"}, {"resource": "repository:configs/model/kaggriculture.yaml"}, {"resource": "repository:configs/scaling_6m.yaml"}, {"resource": "repository:configs/scaling_6m_halfbatch.yaml"}, {"resource": "repository:configs/winner_ce_6m_4x5090.yaml"}, {"resource": "repository:python/owl/model/kaggriculture_workload.py"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:tests/scripts/test_run_ppo.py"}, {"resource": "repository:python/owl/model/kaggriculture.py"}, {"resource": "repository:tests/kaggriculture/test_configs.py"}, {"resource": "repository:tests/owl/train/test_config.py"}, {"resource": "repository:docs/model-architecture.md"}, {"resource": "repository:ops/rebuild-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/rewards.py"}]
---

# Kaggriculture configs follow Isaiah's scaling_6m recipe

These configs **deliberately align to Isaiah**, as the [[../decisions/recipe-choices-align-to-isaiah-without-owner-escalation|recipe Decision]] requires: every recipe choice resolves toward upstream `scaling_6m`, and only per-rank shapes change with the world size.

## What changed (rebuild Task 3.4)

- **`configs/kaggriculture_2rank.yaml`** (128 envs/rank, `segments_per_minibatch` 8, accumulation 1) and **`configs/kaggriculture_4rank.yaml`** (64/4/1). This is Isaiah's own multi-GPU rule from `winner_ce_6m_4x5090.yaml`: divide `n_envs` and `segments_per_minibatch` by the world size and keep the global batch. Everything else is `scaling_6m`'s:
  - the Muon optimizer and the linear-warmup cosine schedule (1,000 / 400,000 steps; the global batch is unchanged, so nothing is rescaled);
  - the PPO coefficients (gamma 1, λ 0.9, clip 0.2, no value clip, `vf_coef` 2, `ent_coef` 1e-6, grad norm 10, `target_kl: null`, `per_player`, normalized advantages);
  - the last-best teacher (0.005 / 0.005, `teacher_segments_per_minibatch` 128, not divided, as in Isaiah's per-rank configs), 8 evaluation replays, `compile_mode: default`, BF16, the trunk-compile defaults and the 20M checkpoint cadence.
- **Kaggriculture-only settings:** the env section (`obs_spec` schema 3, `hire_limit` 241, `reward_mode: win_loss`, `native_threads` 2 until Task 6.1 measures it) and the owner's economic shaping 0.2. The other reward coefficients are the reference branch's defaults, named as in its `rewards.py`, which plan C8 ports. The model is the existing `configs/model/kaggriculture.yaml`, which already forces FlashAttention. It is not renamed to the plan's `kaggriculture_gpu.yaml`, because model tests and docs cite that path.
- **Local CPU configs:** `configs/kaggriculture.yaml` (2 envs, spm 1, FP32, no compilation, no replays) and the tiny `configs/model/kaggriculture_cpu.yaml` (width 16, depth 1, FlashAttention off) are for Task 3.5. They use the same optimizer and PPO coefficients.
- **Startup workload check** (`python/owl/model/kaggriculture_workload.py`, from the GEMM-limit audit in the [[compiled-gemm-template-overflows-above-2-21-rows|overflow Reference]]):
  - `ppo_forward_workloads` derives each forward's per-rank seat rows: rollout `n_envs × 2`, minibatch `spm × horizon × 2`, teacher chunk `min(teacher_spm, n_envs) × horizon × 2`, and evaluation `n_envs × 2`. A BC caller adds its own `ForwardWorkload`.
  - `check_workload_headroom` divides these by the model's own `rows_per_chunk` (709 padded tokens × width 512, giving 5,915 rows) and `head_rows_per_chunk` (11,096). It raises when one row cannot fit either limit or a workload is empty. Head calls are exact. Trunk figures assume full padding, so trunk calls are an **upper bound** and trunk headroom a **lower bound**: the packed path plans chunks from actual token counts (16,384 rows of 300 tokens need 2 packed chunks, not 3).
  - **At startup:** `scripts/run_ppo.py` calls `_check_model_workload` once the per-rank shapes are final (after runtime-GPU adaptation, fresh or resume) and before the run directory, env or model exist. The teacher chunk is omitted when `teacher_mode` is unset. The main process prints one `GEMM workload headroom …` line per workload. Isaiah's Orbit models do not chunk, so they are skipped.
- **Tests:** `tests/scripts/test_run_ppo.py` drives `run_ppo.main` with each ranked config's model, `n_envs` and PPO sections. An over-wide model fails before any allocation step. Valid configs print the headroom lines, and a resume run checks the runtime-adapted `n_envs`. `FullConfig` cannot hold these sections before Task 3.1, so the tests build the config with `model_construct` and patch only its loader. `tests/owl/train/test_config.py` skips `kaggriculture*.yaml` in its FullConfig glob, with a reason naming Task 3.1.

| Workload (preset) | 2 ranks: rows, max trunk / exact head calls | 4 ranks |
|---|---|---|
| rollout | 256, 1/1 (≥ 23.1× trunk headroom) | 128, 1/1 |
| minibatch | 1,024, 1/1 (≥ 5.78×) | 512, 1/1 |
| teacher chunk | 16,384, ≤ 3/2 | 8,192, ≤ 2/1 |
| evaluation | 256, 1/1 | 128, 1/1 |

## Checks

- The first version was rejected by Codex (`ops/rebuild-2026-09-29/codex/verify-3.4-r1.md`): the check had no startup caller, the calls were called exact, and the log overstated schema coverage. The startup wiring, bound labels and narrowed claims answer it. Removing the `run_ppo` call fails the four `main` tests.
- `tests/kaggriculture/test_configs.py`: 20 pass and 3 are skipped (the FullConfig-and-`create_model` test, which names Task 3.1). The tests cover:
  - global envs, optimizer steps per iteration (counted through Isaiah's `_minibatch_indices`), global segments per step and env steps per iteration, which equal `scaling_6m` (256, 16, 16, 16,384) at both world sizes;
  - an optimizer config equal to `scaling_6m`'s, and `PPOConfig` equal except `segments_per_minibatch`;
  - env sections that differ only in `n_envs`;
  - the exact rows and calls in the table, the exact log lines, and packed chunk counts that never exceed the padded bound;
  - fail-fast cases: a model too wide for one row, empty shapes and empty workloads.
- `uvx --from rust-just just py-prepare` passed after the fix: 1,331 passed and 10 skipped; ruff, format, mypy (59 files) and docs-fresh passed, with `DOCS_CURRENT=1` after README was reviewed and still current. No GPU, training or Rust check ran.

## Limits and handoffs

- **Loading is unverified.** No Kaggriculture config loads through `FullConfig` yet: `EnvConfig` and `ModelConfig` lack Kaggriculture until Task 3.1. Only the observation, action, model, optimizer and PPO sections are schema-validated. The env keys and reward-shaping values get exact key and value assertions only. The startup tests bypass the loader, so an end-to-end launch from these files is unproven until Task 3.1. Its `_check_model_workload` signature can then narrow to `ModelConfig`.
- **BC is not checked.** The BC trainer (Task 5.2) does not exist yet; it must add its batch as a `ForwardWorkload` through the same check.
- **The trunk bounds are conservative.** Trunk headroom is a lower bound and trunk calls an upper bound, from full padding. Evaluation uses the `n_envs × 2` upper bound.
- **Chunking is only CPU-tested.** Teacher chunks rely on the model's chunking, which has not been measured on the GPU.
- **Pending dependencies:** `eval_replay_games: 8` needs Task 7.3's Kaggriculture replay export, and the last-best teacher needs Phase 4.
- **Memory is unmeasured.** The spm 8 / accumulation 1 split holds only if the Task 6.1 memory smoke confirms it; otherwise the plan moves to 4 / 2 with the same product.
