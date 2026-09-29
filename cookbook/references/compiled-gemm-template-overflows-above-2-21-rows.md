---
type: "Reference"
title: "Compiled GEMM template overflows above 2^21 rows"
description: "Torch 2.9 Inductor max-autotune Triton GEMM templates wrap the 32-bit A-load offset once rows × GEMM input width exceeds 2^31 while the size argument stays int32 (measured at widths 4096, 768 and 512 and silent in the real trunk; output-wide GEMMs stayed correct to 2^32); this caused the reference PPO-from-BC crash. The rebuilt trunk guard is measured correct at L−1 and rejects L; production compliance is unproven."
tags: ["kaggriculture-v3", "cuda", "compile"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/results.md"}, {"resource": "repository:ops/rebuild-2026-09-29/run-statements/cuda-repro-blocking.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/ppo-from-bc-2026-09-29/train.log"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/cuda-repro-2026-09-29/train.log"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/cuda-repro-2026-09-29/int32_probe.py"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/cuda-repro-2026-09-29/int32_probe.log"}, {"resource": "repository:ops/rebuild-2026-09-29/run-statements/gemm-limits-probe.md"}, {"resource": "repository:ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/driver.jsonl"}, {"resource": "pod-artifact:w7ia3zvxqsvs3g:/workspace/kg-v3/runs/gemm-limits-2026-09-29/"}, {"resource": "repository:ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/kernel_analysis.txt"}, {"resource": "repository:ops/rebuild-2026-09-29/codex/verify-gemm-limits-r1.md"}]
---

# Compiled GEMM template overflows above 2^21 rows

## What was observed

The reference branch's PPO run from the BC checkpoint crashed with a CUDA illegal memory access after 3 iterations. Rerunning the exact setup with `CUDA_LAUNCH_BLOCKING=1` reproduced it on both ranks. Both tracebacks end in the Inductor **max-autotune Triton GEMM template** `triton_tem_fused__to_copy_addmm_gelu_t_view_9`, the transformer trunk's MLP with `M` = batch rows × tokens, K = 1024, N = 256. The kernel declares `INDEX_DTYPE = tl.int64`, which the mm template does not use, and forms the input-side offset `xindex = k + 1024*idx_m` in 32-bit arithmetic. Once `M > 2^21`, that offset passes 2³¹ and wraps. Here K = 1024 is the input width of the MLP down-projection.

A controlled probe confirmed the threshold. It used a synthetic BF16 MLP (256→1024, GELU, 1024→256) with `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)`, compiled at a small shape first. At M = 2,088,960 the compiled output matches eager within BF16 noise (max diff 0.0078). At M = 2,105,344 it faults. Environment: torch 2.9.0+cu128, RTX PRO 6000 Blackwell, driver 595.91.07. Hashes and logs are in `ops/rebuild-2026-09-29/results.md`.

## Why it matters

- Below the fault, wrapped offsets land in valid memory and **silently corrupt** activations. In the reference run, rollout forwards (8,192 rows × more than 256 padded tokens) exceeded the threshold, while training minibatches (128 rows) did not. That produced the first-update log-ratio mean of −3.77 before any weight change. The earlier "BC→PPO deterioration" was infrastructure corruption, not evidence against the BC policy.
- The bug is in compiler-generated code, so it is invisible in the model source. It can reappear for any compiled GEMM whose `rows × input width` exceeds 2³¹: rollouts, evaluation, BC, and other models with larger batches. In training, backward reads forward outputs as inputs, so the forward output width counts too (below).

## Measured limits at the rebuild's shapes (2026-09-29)

A second bounded probe compared compiled output against eager at every point. It covered synthetic compiled Linears and MLPs, plus the real `KaggricultureTransformer` trunk from `kg/rebuild-model` @ `1ddc71d`, compiled directly through its own `compile_transformer_trunk`. Precision matched `run_ppo`: fp32 parameters under bf16 autocast, `max-autotune-no-cudagraphs`, `dynamic=True`, compiled small first. Codex verified the evidence and requested edits, applied here. The table, the generated-code attribution and the source citations are in `ops/rebuild-2026-09-29/results.md`, "GEMM limits at our shapes".

Measured (L_in = 2³¹ / GEMM input width; trunk L = 2³¹/512 = 4,194,304):
- **The overflow is input-side at these shapes.** Input-wide GEMMs (4096→16, 768→256) were correct at L_in and failed at L_in+1. Output-wide GEMMs (16→4096, 256→512, 256→256) stayed correct up to M·N = 2³². The corrected kernel summaries show why:
  - Every failing graph launches the template with a 32-bit size argument (`ks0: i32`) and an A-load of `K*idx_m`.
  - Graphs whose output exceeds int32 get `ks0: i64` and stayed correct.
  - In the real trunk, the corrupting kernel is the 512→256 MLP down-projection. Above L it kept `ks0: i32`, while the 256→512 up-projection had been promoted.

  The earlier universal rule, "overflow iff M·max(K, N) > 2³¹", is withdrawn.
- **In the real trunk the corruption is silent.** With the guard patched off:
  - 2 tokens were wrong at L+1 = 4,194,305 packed tokens, and 4,665 at 4,198,400.
  - The padded path was wrong on half its tokens at 8.39M, with no fault.
- **Above the bound, nothing is reliable.** Whether a given M corrupts, faults or happens to be correct depends on the recompile hint and on autotune's backend choice. For example, cuBLAS won for a lone 512→256, and the trunk was correct again at 8.39M tokens after an i64 recompile. Recompilation is not a safety guarantee.
- **The guard, measured endpoints:** with the guard on, `_run_trunk` is correct at L−1 = 4,194,303 packed tokens. It raises at L with zero trunk calls, and chunks the padded path into calls of 4,193,735 tokens, which are also correct.

Inferred, not measured:
- **Unguarded trunk at exactly L.** It was not run, so whether the guard's rejection at equality is stricter than necessary is unknown.
- **Real-trunk backward.** It was not run at any size. The synthetic MLP backward faulted while compiling at L+1, which does not localize a backward-kernel threshold.
- **Mechanism.** The source path and the generated code agree that the size-arg dtype follows a storage-size check on the template output, and that the mm template ignores `INDEX_DTYPE`. Promotion of the A-load offset through `M → grid_m → group_size → pid_m` in the Triton IR is unverified.

## Consequences

- **Design bound: keep every compiled region at M × max(in, out) ≤ 2³¹ over its Linear layers.** This follows from the measured input-side rule plus backward: dX = dY·W reads dY at the forward output width, and weight-gradient GEMMs reduce over M. It is not justified by an output-store overflow.
  - For the trunk that width is 512, which is `gemm_kmax`.
  - Nothing enforces the bound outside the trunk. The stems are eager today, but the full stem is 371→512→256. The heads and the planned 768-wide `actor_input_proj` are not implemented yet; compiled, `actor_input_proj` would exceed L_in at 2,796,203 rows × frames.
- **Production compliance is unproven.** Kaggriculture is not yet in `ModelConfig` or the model factory, and `configure_model_compile` rejects it for `model_compile="trunk"`. Integrated rollout, PPO recompute, evaluation, BC and teacher workloads stay unverified until Task 3.1 and Phase 6.
- **Size margins against 2³¹/width, not against observed faults.** A passing test above the bound is not evidence of safety.
- **Workload margins against the trunk L:**
  - Rollout at 256 rows: 23.1× headroom, forward measured correct.
  - PPO minibatch at 1,024 rows: 5.78×, arithmetic only.
  - Planned 16,384-row teacher precompute: reaches L at a mean of 256 tokens per row, so the safe mean is strictly below 256. It needs `teacher_segments_per_minibatch ≤ 46` at horizon 64, or chunking. The guard would raise, not corrupt.
- **Log-ratio alarm:** treat a non-zero log-ratio at the first minibatch (before any optimizer step) as a correctness failure and abort, recording the batch shape.
- **Upgrades:** don't upgrade torch or drivers as a remedy without a separate measured decision. Rerun both probes after any upgrade.

## Limits

- **Stack:** measured on torch 2.9.0+cu128, driver 595.91.07, RTX PRO 6000 Blackwell (sm_120). Triton 3.5.0 is the lockfile pin; its installed version was not captured.
- **Templates:** `mm`/`addmm` templates with fused prologues and epilogues, dynamic shapes. Unmeasured:
  - the real trunk's backward;
  - the unguarded trunk at exactly L;
  - `bmm`, persistent-TMA and decompose-K paths (decompose-K is on by default and can apply to weight-gradient GEMMs);
  - static-shape compiles;
  - other versions.
- **Tolerances:** the checks detect the reported corruption. They do not prove BF16 equivalence for arbitrary inputs.
- **FlashAttention:** the packed-path runs used torch's ATen varlen flash shim, because the pod venv has no `flash-attn`; the pod's run config also had `force_flash_attn: false`. The real flash-attn kernel is untested, and Phase 6 must install and verify it.
- **Other versions:** torch 2.8 showed no overflow at width 1024 (Task 0.2).
