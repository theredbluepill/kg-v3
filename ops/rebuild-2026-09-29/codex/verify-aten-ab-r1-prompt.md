You are Codex. READ-ONLY review of the ATEN-only GEMM A/B: ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md, ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/, results.md section 'ATEN-only GEMM A/B'.

Summary (from the orchestrator; verify against the files, do not trust):

ran: true

run_statement: /Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md, committed together with the scripts in 9904121 on kg/rebuild-model at 07:06:00Z, before the 07:06:29Z launch. It states the question, the hypothesis and the source-reading support (torch 2.9 kernel/mm.py adds the mm, persistent-TMA and decompose-K templates only inside use_triton_template, which needs 'TRITON' in max_autotune_gemm_backends). It also covers the stage order, pass criteria, idle rule, stop-at-first-failure rule and the 45-min hard limit. The 45-min limit is enforced twice: a 44-min internal deadline in the driver and `timeout -k 20 2700` around the whole driver. The pod checkout was /workspace/kg-v3-rebuild at e1458d2a717d9d731a367cbb78b98616ee6649f4, which equals kg-v3-int HEAD, so no bundle was needed. Porcelain was empty before and after. Stack: torch 2.9.0+cu128 (git 0fabc3ba), triton 3.5.0, real flash-attn 2.8.3, driver 595.91.07. Idle gate: GPU 0 had no compute processes and 0 % utilisation, with 0 MiB in use at prelaunch. The gate passed after 0 s, with raw receipts captured. The driver ran 07:06:31Z to 07:14:55Z (503.9 s) and exited 0. No first-failure stop fired. Both default-backend controls ran last, each in its own subprocess, and both reproduced the failure.

correctness: Supported for the hypothesis at these shapes on this stack. With ATEN-only and the guard bypassed, every case that corrupts or faults under the default backends matched eager.

- **Real trunk, packed path, real flash-attn:** tested at 4,194,305 (L+1), 4,198,400, 4,194,444 and 8,387,470 tokens, one trunk call each. There were 0 tokens with |Δ| > 0.25 and 0 non-finite values; max |Δ| ≤ 0.1699, which is below the 0.1703 passing level seen earlier.
- **Linear 768→256 at 2,796,203 rows, and 512→256 at 4,198,401 and 8,388,608 rows:** 0 bad rows, max |Δ| 0.0. The compiled graph now calls the same cuBLAS bias_addmm that eager uses.
- **MLP 256→512→256 forward and backward at 4,194,305 rows:** forward and dX had 0 bad rows, and the largest relative parameter-gradient error was 0.0026. The pre-declared threshold was 0.05.

The controls in the same run, with default backends, reproduced the bug. The trunk had 2 and 4,665 wrong tokens (max |Δ| 45.95, 1,194,240 non-finite values), identical to the earlier probe. Linear 768→256 hit an illegal memory access during autotune at 2,796,203 rows. So the pass is discriminating.

Inductor's own 32-bit-indexing guard on the pointwise kernels (can_use_32bit_indexing, simd.py:1440) triggered a recompile in each ATEN stage, and the recompiled graphs were correct.

timing_table: Component timing only, conditional on the synthetic A/B/C/D schedule. Medians over 20 CUDA-event iterations, same unchanged bench, same process order (mid then dense for each setting).

| density | workload | default median / p90 (ms) | ATEN median / p90 (ms) | ATEN ÷ default |
|---|---|---|---|---|
| mid (303 tok) | A | 12.904 / 12.926 | 15.228 / 15.235 | 1.180 |
| mid | B | 177.815 / 177.998 | 186.177 / 186.401 | 1.047 |
| mid | C | 986.390 / 986.567 | 1004.654 / 1004.973 | 1.019 |
| mid | D | 9.373 / 9.378 | 11.838 / 11.847 | 1.263 |
| dense (709 tok) | A | 25.389 / 25.394 | 29.106 / 29.118 | 1.146 |
| dense | B | 326.702 / 327.109 | 332.705 / 332.927 | 1.018 |
| dense | C | 1687.409 / 1687.614 | 1731.732 / 1732.191 | 1.026 |
| dense | D | 21.965 / 21.977 | 25.753 / 25.891 | 1.173 |

Update wall = 64·t_A + t_C + 16·t_B + t_D:
- **mid:** 4.667 s → 4.970 s, **+6.5 %**; ceiling 1,755 → 1,648 SPS per rank.
- **dense:** 8.562 s → 8.944 s, **+4.5 %**; ceiling 957 → 916 SPS per rank.

The default arm reproduces the earlier SPS-ceiling run (4.662 s and 8.564 s). Every p90 is within 0.53 % of its median, and every max sample is within 1.95 %. Allocated memory is unchanged. The cold first compile is faster under ATEN (mid B: 39.6 s → 26.0 s).

kernel_evidence: Sources: the unchanged revision-2 analyzer (pod/kernel_analysis.txt and .json), a template census from scripts/check_templates.py (pod/template_census.txt), and per-wrapper Triton kernel counts (post-run/wrapper_kernel_counts.txt).

**ATEN-only caches have 0 triton_tem_ definitions and 0 launches:**
- trunk: 48 bias_addmm and 48 mm calls in each of its 2 wrappers
- lin 768: 2 bias_addmm
- lin 512: 2 bias_addmm
- MLP backward: 4 bias_addmm and 8 mm
- bench_aten: 57 bias_addmm and 135 mm, including the trunk backward graph of B (ke wrapper, 96 mm)

Every ATEN autotune log line reports 'num_triton_choices': 0.

**Other fusions stay in Triton.** The ATEN trunk forward wrappers keep 4 pointwise and 5 persistent-reduction Triton kernels each. The B backward wrapper keeps 3 pointwise, 4 persistent-reduction and 8 reduction kernels.

**Positive control:** the default caches from this same run launch the vulnerable templates:
- bench_default: 25 definitions and 128 launches, e.g. triton_tem_fused__to_copy_addmm_gelu_t_9 .

Check: run statement committed before launch; idle rule and receipts; driver stop/limit enforcement; that the correctness cases really bypass the guard as the original probe did and compare every element (and gradients); that the kernel evidence shows extern cuBLAS GEMMs and no Triton mm templates in the ATEN runs, with the analyzer used correctly; timing method and that claims stay component/conditional; the recommendation follows from evidence; whether 'immune' is overclaimed (other templates like bmm, flex, static shapes). Verify against the installed torch 2.9 source in /Users/poonszesen/kg-v3/.venv that max_autotune_gemm_backends='ATEN' excludes Triton mm/addmm/bmm and decompose-K/persistent-TMA choices (cite file:line). End with VERDICT: APPROVE / APPROVE WITH EDITS / REJECT, with findings.
