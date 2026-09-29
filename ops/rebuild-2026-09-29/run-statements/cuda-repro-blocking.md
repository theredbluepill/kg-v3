# Run statement — CUDA illegal memory access, blocking reproduction (Task 0.2, step 1)

- **Question:** which kernel raises the illegal memory access seen in `ops/ppo-from-bc-2026-09-29` (reference branch)? The async error surfaced at `.cpu()` in `KaggricultureTransformer._check_flags` during rollout, so that location isn't necessarily the cause. Hypotheses:
  - **H1:** a dense-state index overflow in a compiled kernel.
  - **H2:** an async host→device copy from a native buffer that the next step overwrites.
  - **H3:** another kernel (attention, sampling, NCCL).
- **Inputs:**
  - **Code:** the pod checkout `/workspace/kg-v3`, which is the code that crashed (commit `0d01234` plus 16 uncommitted files, equivalent to reference-branch content).
  - **Config:** the crashed run's own `ops/ppo-from-bc-2026-09-29/config.yaml`.
  - **Checkpoint:** weights from `runs/ppo-from-bc-input-2026-09-29/checkpoint_best.pt`, SHA-256 `ffd7d9e46f04d8d0e6330d05c6ef14f6f8eda3abc94e30a8602303295032e140`, loaded `model_only`.
  - **Hardware and software:** 2 ranks on pod `w7ia3zvxqsvs3g` (2× RTX PRO 6000 Blackwell, driver 595.91.07, torch 2.9.0+cu128).
- **Change from the crashed run:** `CUDA_LAUNCH_BLOCKING=1`; `--log-mode debug` (no W&B); a separate output dir `runs/cuda-repro-2026-09-29/`; a hard wall-clock limit.
- **Discriminating observation:** with blocking launches, the Python traceback names the op that faults.
  - An indexing/gather/embedding op, or a compiled kernel reading an index tensor → H1. Record the actor/frame counts at the fault.
  - The fault at or right after a host→device copy of observation tensors → H2.
  - Anything else → H3.
- **Stopping condition:** the first CUDA error, or 60 minutes wall time, whichever comes first. If there's no fault in 60 minutes, record "not reproduced under blocking launches". The next step is then a rollout-only `compute-sanitizer` repro (Codex script) with synchronous copies as the H2 control.
- **Budget:** ≤ 60 min on an already-running pod ($4.18/h, no new billing).
- **Artifacts:** pod `/workspace/kg-v3/runs/cuda-repro-2026-09-29/` (log `train.log`, exit code); a copy of the log and the conclusion in `ops/rebuild-2026-09-29/results.md`.
- **Safety:** the pod showed 0% GPU utilization and no training process before launch. Nothing else is running.

## Follow-up run statement — int32 GEMM-template overflow confirmation (single GPU, ≤ 10 min)

- **Question:** does an Inductor `max-autotune-no-cudagraphs` dynamic-shape MLP (Linear 256→1024, GELU, Linear 1024→256, BF16) produce wrong results or fault once the row count M = batch × tokens exceeds 2²¹, as the faulting kernel's 32-bit `1024*idx_m` offset predicts?
- **Inputs:** synthetic random tensors; the pod `.venv` (torch 2.9.0+cu128); GPU 0 only. Compile at a small shape first (as the reference run did), then evaluate at M = 2²¹ − 256 and M = 2²¹ + 4,096.
- **Discriminating observation:** max |compiled − eager| is at BF16 noise below the threshold and large (or a fault) above it → confirmed. Equal behavior on both sides → refuted; look elsewhere.
- **Stop:** both shapes run once each, or the first fault. **Artifacts:** `runs/cuda-repro-2026-09-29/int32_probe.log`.
