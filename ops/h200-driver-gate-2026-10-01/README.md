# H200 driver gate — ATEN-only GEMM correctness re-probe (2026-09-30 17:01–17:04Z)

Run statement: `run-statement.md` (written before launch). Pod: 8× NVIDIA H200
(143,771 MiB, compute capability 9.0), driver 570.211.01; GPU 0 only
(`CUDA_VISIBLE_DEVICES=0`), GPUs 1–7 idle before and after
(`receipts/idle_nvidia_smi_*.txt`). Source: `/root/kg-v3` detached at
`07c8fc9972297f28e0414da752700194a90d4fc3`, porcelain empty before and after
(`receipts/git_pre.txt`, `git_post.txt`). Stack (`receipts/versions.txt`):
torch 2.9.0+cu128 (git `0fabc3ba…`), triton 3.5.0, flash-attn 2.8.3, CUDA 12.8.
Driver 179.1 s, exit 0; launch 17:01:20Z → 17:04:25Z.

## Results (all ATEN stages pass; the control reproduced)

| stage | points | result |
|---|---|---|
| `aten_trunk_packed_bypass` (real trunk, packed/flash, guard bypassed) | 4,194,305 / 4,198,400 packed; dense 5,916 rows (4,194,444) and 11,830 rows (8,387,470) tokens | 0 wrong tokens, 0 non-finite at every point; max \|Δ\| 0.1528 / 0.1728 / 0.1465 / 0.1624; one trunk call of all tokens each (M·512/2³¹ up to 1.9997) |
| `aten_lin_768_256` | 2,796,203 rows | 0 bad rows, 0 non-finite, max \|Δ\| 0.0, no clobber |
| `aten_lin_512_256` | 4,198,401 and 8,388,608 rows | 0 bad rows, 0 non-finite, max \|Δ\| 0.0, no clobber |
| `aten_mlpbwd_256_512_256` | 4,194,305 rows | fwd 0 bad rows (max \|Δ\| 0.00049), dX 0 bad rows (0.0); param-grad rel max 0.0 / 0.00263 / 0.0 / 0.00222 (threshold 0.05) |
| `ctl_default_lin_768_256` (default backends, control) | 2,796,203 rows | **reproduced**: `CUDA error: an illegal memory access` during Triton mm autotuning (19 of 21 choices Triton), rc 1 |

Backend records: every ATEN stage `"ATEN"` at start and end; the control
`"ATEN,TRITON,CPP"`. The trunk process shows the real stack check rejecting
driver 570.211.01 (`real_stack_check`), the probe's explicit bypass, and the
claim `gemm_backends='ATEN'` at the end.

**Template census** (`template_census.txt`, `kernel_analysis.txt`): every ATEN
cache has 0 `triton_tem_` definitions and 0 launches, no decompose-K,
persistent-TMA, contiguous-subgraph or `triton_mm_` text; GEMMs are
`extern_kernels.bias_addmm`/`addmm`/`mm` (trunk 48 + 48; MLP fwd/bwd 4 + 8).
All 10 ATEN autotune lines log `num_triton_choices: 0`. The control cache has
3 `triton_tem_` definitions and 1 launch (analyzer positive control).

## Adaptations (all declared in the run statement)

- `probe_trunk_07c8fc9.py` from `probe_trunk_e1458d2.py` (`f700787b…`): ROOT
  `/root/kg-v3`; a required `--stack-check-bypass` replacing
  `owl.model.compile_gemm.check_compile_stack` (07c8fc9's trunk compile claims
  the backends and rejects this driver); extra process-record fields.
- `driver.py`, `launch.sh`: paths, 25/26-min limits, timing stages and the
  default-backend trunk control dropped. `probe_linear.py`,
  `gemm_backend_wrap.py`, `analyze_kernels.py`, `check_templates.py` unchanged
  (sha256 in `receipts/scripts.sha256`).

## Custody and limits

- `receipts/idle_ps_*.txt`: the pod's Jupyter `--IdentityProvider.token` value
  is redacted to `<REDACTED>`; nothing else is edited.
- Not copied: the Inductor/Triton caches (70 MB, pod `/root/aten-h200/`) and
  their per-file hash list `receipts/compile_caches.sha256` (3,310 lines,
  sha256 `9f4ecb73…d8f1`, pod only).
- Not measured: timing/SPS on H200, the real-trunk backward element-wise above
  the bound, bmm, static-shape compiles, GPUs 1–7 individually, multi-rank.
  One run; no nsys (`receipts/nsys_check.txt`: absent).
