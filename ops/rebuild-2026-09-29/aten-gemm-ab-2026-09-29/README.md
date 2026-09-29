# ATEN-only GEMM backend A/B, 2026-09-29 07:06–07:15Z (pod `w7ia3zvxqsvs3g`, GPU 0 only)

**Infrastructure diagnostic.** One Inductor setting changed: `torch._inductor.config.max_autotune_gemm_backends` from torch 2.9.0's default `"ATEN,TRITON,CPP"` to `"ATEN"`. No model, loss, reward or learning code changed. The timing half is a **component measurement conditional on the A/B/C/D synthetic schedule** of the SPS-ceiling probe, not end-to-end SPS.

- Pre-run statement: `../run-statements/aten-gemm-ab.md`, committed with the scripts in `9904121` at 07:06:00Z. Launch at 07:06:29Z; driver 07:06:31Z–07:14:55Z (**503.9 s**, exit 0; `pod/launch.out`, `pod/driver.jsonl`).
- Source: pod checkout `/workspace/kg-v3-rebuild` at `e1458d2a717d9d731a367cbb78b98616ee6649f4` (tree `16b79ba5…`), equal to `/Users/poonszesen/kg-v3-int` HEAD; `git status --porcelain` empty before and after (`pod/receipts/git_pre.txt`, `git_post.txt`); the run dir is gitignored (`.gitignore:27:runs/`). No bundle, no install.
- Stack (`pod/receipts/versions.txt`): torch 2.9.0+cu128 (git `0fabc3ba…`), triton 3.5.0, flash-attn 2.8.3, CUDA runtime 12.8, driver 595.91.07, RTX PRO 6000 Blackwell Server Edition. Default `max_autotune_gemm_backends` = `'ATEN,TRITON,CPP'`; `TORCHINDUCTOR_*`/`TRITON_*` env unset.
- Scripts: `scripts/` (sha256 in `pod/receipts/scripts.sha256`, identical to the committed copies). `probe_linear.py`, `bench_model_sps.py` and `analyze_kernels.py` (rev 2) are byte-identical to the earlier runs (`0a7af363…`, `4a7ede9a…`, `ee26698a…`). Every stage ran through `gemm_backend_wrap.py`, whose per-stage record (`pod/*.backend.json`) shows the value at start and end: `ATEN` for ATEN stages, `ATEN,TRITON,CPP` for default stages.
- Idle rule: GPU 0 no compute processes and 0 % utilization. Prelaunch capture (07:06:31Z) **0 MiB, 0 %, P8, no processes** on both GPUs; gate passed after 0 s (`pod/receipts/idle_nvidia_smi_prelaunch.txt`). Post (07:14:56Z): 0 MiB, 0 %, no processes; again at 07:16:29Z after the copy (`post-run/idle_after_copy.txt`). The pod was left running and idle.
- nsys: absent (`pod/receipts/nsys_check.txt`: `which nsys` rc 1, no `/usr/local/cuda*/bin/nsys`, no `/opt/nvidia/nsight-systems*`). Timing is CUDA-event totals only.

## 1. Correctness above the bound, ATEN-only (guard bypassed)

All four stages passed; the driver's first-failure stop never fired.

| Stage (ATEN-only) | Points | Result (vs eager) | Default backends, earlier probe / this run's control |
|---|---|---|---|
| Real trunk, packed, real flash-attn, guard bypassed (`_GEMM_ELEMENT_LIMIT = 2**62`) | 4,194,305 (L+1); 4,198,400; 4,194,444 (5,916 dense rows); 8,387,470 (11,830 dense rows) packed tokens, one trunk call each | **0** tokens with \|Δ\| > 0.25, 0 non-finite; max \|Δ\| 0.1565 / 0.1638 / 0.1657 / 0.1699 (warm 0.1033) | 2 / 4,665 (1.19M non-finite) / 236 wrong tokens; 8,387,470 illegal memory access. **Control this run:** 2 and 4,665 wrong (max 45.95, 1,194,240 non-finite), identical counts |
| Linear 768→256 | 2,796,203 (L_in+1) | 0 bad rows, max \|Δ\| **0.0** | fault in autotune benchmarking. **Control this run:** illegal memory access in `select_algorithm.py` autotune at 2,796,203 (rc 1) |
| Linear 512→256 | 4,198,401; 8,388,608 (2·L_in) | 0 bad rows, max \|Δ\| 0.0 at both | 8,388,608 faulted in autotune (earlier probe) |
| MLP 256→512→256, forward + backward | 4,194,305 (L+1) | fwd 0 bad rows (max 0.000488), dX 0 bad rows (max 0.0), grad rel max: up.weight 0.0, up.bias 0.00229, down.weight 0.0, down.bias 0.00260 (threshold 0.05) | fault **while compiling** at L+1 (earlier probe) |

Tolerances as in the earlier probe (single GEMMs: `|Δ| ≤ 0.02 + 0.02|ref|` per element; trunk: a valid token is wrong if any channel exceeds 0.25). Raw: `pod/aten_*.jsonl`, `pod/ctl_default_*.jsonl`, logs `pod/*.log`.

The single-Linear results are bit-identical to eager (max |Δ| 0.0) because the compiled graph now calls the same cuBLAS `bias_addmm` that eager `F.linear` uses; that is the mechanism, not an independent test of cuBLAS. Every ATEN autotune log line reports `"num_choices": 2, "num_triton_choices": 0` (`bias_addmm` vs `addmm`).

The Triton pointwise kernels' own int32 protection engaged as designed: each ATEN stage recompiled on a `can_use_32bit_indexing` guard (`simd.py:1440`), e.g. `512*x.size()[0] <= 2147483647` in the trunk (`pod/aten_trunk_packed_bypass.log:13`), and the recompiled graphs were correct.

## 2. Generated-code evidence

`pod/kernel_analysis.{txt,json}` (analyzer rev 2, unchanged) and `pod/template_census.{txt,json}` (`scripts/check_templates.py`), both run on the pod over every per-stage cache; `post-run/wrapper_kernel_counts.txt` counts Triton kernel definitions per wrapper.

| Cache | Wrappers | extern GEMM calls | `triton_tem_` defs / launches |
|---|---|---|---|
| aten_trunk_packed_bypass | 2 | 48 `bias_addmm`, 48 `mm` | **0 / 0** |
| aten_lin_768_256 | 2 | 2 `bias_addmm` | **0 / 0** |
| aten_lin_512_256 | 2 | 2 `bias_addmm` | **0 / 0** |
| aten_mlpbwd_256_512_256 | 6 | 4 `bias_addmm`, 8 `mm` | **0 / 0** |
| bench_aten (A/B/C/D incl. B's trunk backward) | 4 | 57 `bias_addmm`, 135 `mm` | **0 / 0** |
| bench_default | 12 | 64 `mm` | 25 / 128 |
| ctl_default_trunk_packed_bypass | 6 | 0 | 24 / 96 |
| ctl_default_lin_768_256 | 2 | 1 `bias_addmm` | 2 / 1 |

- **ATEN-only: every GEMM is an extern cuBLAS call and no Triton mm template exists**, including the trunk backward graph of the bench's B (`bench_aten/ke/…`: 96 `mm`, 0 templates).
- **Other fusions stay Triton.** The ATEN trunk forward wrappers still define 4 pointwise + 5 persistent-reduction Triton kernels each; the B backward wrapper 3 pointwise + 4 persistent-reduction + 8 reduction kernels (`post-run/wrapper_kernel_counts.txt`).
- **Default backends in the same run still produce the vulnerable class**: e.g. `bench_default/lu/…` launches `triton_tem_fused__to_copy_addmm_gelu_t_9` (A-load `512*idx_m`, `ks0: i32`) and the backward `bench_default/yd/…` launches `triton_tem_fused_mm_3/_11` with `ks0: i32`. This is the analyzer's positive control.
- **Decompose-K, persistent-TMA, contiguous-subgraph templates:** 0 marker files in every cache. This census is **not positive-controlled** for them (the default caches have 0 as well, so these templates were not selected here either). Their absence under ATEN-only rests on source gating: in the installed `kernel/mm.py`, `tuned_mm` (760–783) and `tuned_addmm` (997–1029) add them only inside `use_triton_template(...)`, which requires `"TRITON"` in `max_autotune_gemm_backends` (`utils.py:1640–1665`), plus the logged `num_triton_choices: 0`.

## 3. Timing A/B (component, conditional on the synthetic schedule)

Same unchanged bench, same iteration counts (first call, 5 warmup, 20 CUDA-event iterations), process order mid-default, mid-ATEN, dense-default, dense-ATEN; one cold cache per setting (mid cold, dense warm for both). All four passed their checks (`use_flash_attn` true on every call; trunk chunks A/B/D 1, C 2 at mid and 3 at dense). Summary: `ab_summary.json` (from `summarize_ab.py`).

| density | workload | default median / p90 (ms) | ATEN median / p90 (ms) | ATEN / default (median) |
|---|---|---|---|---|
| mid (303 tok) | A fwd 256 | 12.904 / 12.926 | 15.228 / 15.235 | 1.180 |
| | B train 1,024 | 177.815 / 177.998 | 186.177 / 186.401 | 1.047 |
| | C teacher-proxy 16,384 | 986.390 / 986.567 | 1,004.654 / 1,004.973 | 1.019 |
| | D value 256 | 9.373 / 9.378 | 11.838 / 11.847 | 1.263 |
| dense (709 tok) | A | 25.389 / 25.394 | 29.106 / 29.118 | 1.146 |
| | B | 326.702 / 327.109 | 332.705 / 332.927 | 1.018 |
| | C | 1,687.409 / 1,687.614 | 1,731.732 / 1,732.191 | 1.026 |
| | D | 21.965 / 21.977 | 25.753 / 25.891 | 1.173 |

| density | update wall default | update wall ATEN | relative cost | ceiling SPS/rank default → ATEN |
|---|---|---|---|---|
| mid | 4.667 s | 4.970 s | **+6.5 %** | 1,755 → 1,648 |
| dense | 8.562 s | 8.944 s | **+4.5 %** | 957 → 916 |

- update wall = 64·t_A + t_C + 16·t_B + t_D (medians); ceiling = 8,192 / wall. The default arm reproduces the SPS-ceiling run (mid 4.662 s, dense 8.564 s there).
- Variability: every p90 is within 0.53 % of its median (largest: ATEN dense D); every max sample (`cuda_event_ms`) is within 1.95 % of its median; no tail sample like the earlier 545 ms dense-B outlier occurred.
- Where the cost sits: small-batch no-grad calls lose most (A +14.6–18.0 %, D +17.3–26.3 %), because cuBLAS cannot take Inductor's fused prologues/epilogues (casts, bias + GELU, residual + LayerNorm), so they become separate Triton kernels. The large calls lose little (B +1.8–4.7 %, C +1.9–2.6 %). Attribution beyond these totals needs a timeline; none exists.
- Peak `max_memory_allocated` is unchanged (e.g. dense B 40.27 vs 40.28 GiB).
- First call (compile/autotune), cold mid: A 20.8 s → 15.9 s, B 39.6 s → 26.0 s (fewer autotune candidates); warm dense similar (A 8.2/8.4 s, B 10.0/9.8 s).

## Limits

- **One stack and these shapes:** torch 2.9.0+cu128, triton 3.5.0, driver 595.91.07, sm_120. Largest tested GEMM input 8,388,608 × 512 = 2³² elements; beyond that is untested. cuBLAS 64-bit indexing is the vendor library's behavior, measured here only to 2³².
- **Real-trunk backward above the bound was not element-compared.** The bench's B ran the trunk backward only below the bound (1,024 rows), with no numerical check; above the bound, backward correctness rests on the synthetic MLP (correct at L+1) plus kernel evidence (no templates in the ATEN backward graph).
- **Non-GEMM Triton kernels remain.** Pointwise/reduction kernels are protected by Inductor's own 32-bit-indexing guards, which recompiled here and were correct at the tested sizes; that is not a general proof. `bmm` was not exercised (attention runs in flash-attn); it is gated by the same backend list (`kernel/bmm.py:200–211`), unmeasured.
- **Decompose-K / TMA absence** is source-gated plus zero-marker, not positive-controlled.
- **Timing is component-only**, conditional on the synthetic A/B/C/D schedule (uniform densities, tiled batches, surrogate B and C, no finite-loss or replay-equality check, heads eager). No nsys timeline; one run per arm; no repeat to measure run-to-run spread of the A/B difference.
- **Process-global setting.** `max_autotune_gemm_backends` applies to every `torch.compile` in the process (including a compiled Isaiah model in the same process). `configure_model_compile` still rejects this model at `e1458d2`; the compiled path is not reachable from `run_ppo` config yet.
- **No change in this repo's code.** Adoption would need a wired setting and a test; nothing was implemented here.

## Large files kept on the pod only (not copied, > 1 MB)

- `inductor_cache/` (269,569,900 B) and `triton_cache/` (194,098,115 B): 11,432 files. Per-file manifest `receipts/compile_caches.sha256` (2,142,259 B, 11,432 lines, sha256 `0df334497dbcab41c7600d238b5bf7bc691e7cfdb991a046848111baceac77a9`).

Local manifest: `MANIFEST.sha256`.
