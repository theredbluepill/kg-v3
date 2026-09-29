---
type: "Reference"
title: "Compiled GEMM template overflows above 2^21 rows"
description: "Torch 2.9 Inductor max-autotune Triton GEMM templates wrap 32-bit offsets once rows × inner dim exceeds 2^31, silently corrupting and then faulting; this caused the reference PPO-from-BC crash and its log-ratio mismatch."
tags: ["kaggriculture-v3", "cuda", "compile"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/results.md"}, {"resource": "repository:ops/rebuild-2026-09-29/run-statements/cuda-repro-blocking.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-from-bc-2026-09-29/train.log"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/cuda-repro-2026-09-29/train.log"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/cuda-repro-2026-09-29/int32_probe.py"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/cuda-repro-2026-09-29/int32_probe.log"}]
---

# Compiled GEMM template overflows above 2^21 rows

## What was observed

The reference branch's PPO run from the BC checkpoint crashed with a CUDA illegal memory access after 3 iterations. Rerunning the exact setup with `CUDA_LAUNCH_BLOCKING=1` reproduced it on both ranks. Both tracebacks end in the Inductor **max-autotune Triton GEMM template** `triton_tem_fused__to_copy_addmm_gelu_t_view_9`, the transformer trunk's MLP with `M` = batch rows × tokens, K = 1024, N = 256. The kernel declares `INDEX_DTYPE = tl.int64` but forms `xindex = k + 1024*idx_m` from the 32-bit program id, so once `M > 2^21` the offset passes 2³¹ and wraps.

A controlled probe confirmed the threshold. It used a synthetic BF16 MLP (256→1024, GELU, 1024→256) with `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)`, compiled at a small shape first. At M = 2,088,960 the compiled output matches eager within BF16 noise (max diff 0.0078). At M = 2,105,344 it faults. Environment: torch 2.9.0+cu128, RTX PRO 6000 Blackwell, driver 595.91.07. Hashes and logs are in `ops/rebuild-2026-09-29/results.md`.

## Why it matters

- Below the fault, wrapped offsets land in valid memory and **silently corrupt** activations. In the reference run, rollout forwards (8,192 rows × more than 256 padded tokens) exceeded the threshold, while training minibatches (128 rows) did not. That produced the first-update log-ratio mean of −3.77 before any weight change. The earlier "BC→PPO deterioration" was infrastructure corruption, not evidence against the BC policy.
- The bug is in compiler-generated code, so it is invisible in the model source. It can reappear for any compiled GEMM whose `rows × inner_dim` exceeds 2³¹: rollouts, evaluation, BC, and other models with larger batches.

## Consequences

- Keep compiled forwards below the threshold: Isaiah's multi-GPU cadence (128 envs/rank) does this for training. Guard the compiled trunk so it fails fast or chunks when `rows × tokens × max_inner_dim ≥ 2^31`, with a boundary test.
- Treat a non-zero log-ratio at the first minibatch (before any optimizer step) as a correctness failure and abort, recording the batch shape.
- Don't upgrade torch or drivers as a remedy without a separate measured decision; rerun the probe after any upgrade.

## Limits

This is verified for one torch/Triton/driver/GPU combination and one template family (`mm`/`addmm` with a fused epilogue). Other templates and versions are unchecked. The probe proves the threshold for K = 1024; for inner dimension K it's `M > 2^31 / K`.
