# Independent verification: GEMM-limit audit and probe (round 1)

You are an independent, read-only verifier. Do not edit files. Do not run GPU work. Reading source and running cheap local arithmetic, grep or python-without-torch-execution is fine.

## The owner's question

"How do you know the limits are true, and how do you ensure we are under that limit?"

Two artifacts claim to answer that: (1) a source-reading audit of torch 2.9.0 Inductor; (2) a measured pod probe. Verify both against primary evidence and decide whether the conclusions are supported.

## Primary sources to check against

- Installed compiler: `/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor` (torch 2.9.0; confirm version/git in `torch/version.py`). Triton is not installed on the Mac.
- Model/guard: `/Users/poonszesen/kg-v3/python/owl/model/kaggriculture.py` (`_run_trunk`, `gemm_kmax`, `compile_transformer_trunk`, stems/heads), `/Users/poonszesen/kg-v3/python/owl/train/utils.py`, `/Users/poonszesen/kg-v3/python/owl/train/ppo.py`, `/Users/poonszesen/kg-v3/scripts/run_ppo.py`, `/Users/poonszesen/kg-v3/configs/`.
- Probe run statement: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/run-statements/gemm-limits-probe.md`
- Probe code: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/` (probe_linear.py, probe_trunk.py, driver.py, analyze_kernels.py)
- Probe outputs: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/` (per-case JSONL, attempt logs incl. recompile/guard output, driver.jsonl, driver.out, kernel_analysis.json/.txt, inductor_cache/ generated kernels, sha256.txt)
- Results write-up: section "GEMM limits at our shapes" in `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md`
- Cookbook note: `/Users/poonszesen/kg-v3/cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md`

## Claims under review (condensed; the full text is in results.md and the cookbook note)

### Audit (source reading, no GPU)

1. In the mm/addmm Triton template with dynamic M, `INDEX_DTYPE` is never applied to rm/rn/xindex; it is only read by flex templates (select_algorithm.py:1533-1539; grep kernel/flex/templates/*.jinja). rm = pid_m*BLOCK_M + tl.arange from raw tl.program_id(0) (kernel/mm.py:80, 93, 129). The template's codegen_range_tree is a no-op (select_algorithm.py:1212-1213), so the int64 pid cast of normal kernels (codegen/triton.py:4480-4481) is skipped.
2. Two independent dtype decisions: (a) INDEX_DTYPE via can_use_32bit_indexing over M*N, M*K, K*N (select_algorithm.py:1530-1536; simd.py:1407-1443); (b) the kernel's own index_dtype via SIMDKernelFeatures([], numel) (select_algorithm.py:389-395) -> select_index_dtype (simd_kernel_features.py:133-150) using only output numel M*N. (b) sets ks* size-arg types for templates (select_algorithm.py:580-583; codegen/triton_utils.py:141-151, 78-81); non-template kernels force ks* to int64.
3. Regimes from the compile hint M_h (utils.py:3034-3065; guards via sizevars.py:404-426, 444-445; symbolic_shapes.py:7709-7714): R1 all int32 with guards (recompile, safe); R2 M_h*N <= int_max < M_h*K: ks i32, A-load xindex = idx_n + K*idx_m (select_algorithm.py:902-906) wraps once M*K > 2^31; R3 M_h*N > int_max: ks i64, A-load safe (INFERRED promotion), store xindex = idx_n + N*idx_m (select_algorithm.py:1057; mm.py:129-136) wraps once M*N > 2^31.
4. Pointwise/reduction kernels are protected (simd_kernel_features.py:133-150, simd.py:1435-1442, triton_utils.py:141-147, codegen/triton.py:4451-4452, 4480-4481). Unprotected surface is GEMM templates (mm/addmm, bmm).
5. Predicted combined rule: overflow iff M*max(K, N_out, fused-epilogue row stride) > 2^31. Limits: 256->256: 8,388,608; 256->512: 4,194,304; 512->256: 4,194,304; 768->256: 2,796,202; 371->256 about 5.7-5.8M.
6. Only the trunk is compiled (kaggriculture.py:394-398 compiles _forward_transformer_trunk). Stems, actor_input_proj (768->256), 9 heads and critic run eager. Guard constant gemm_kmax=512 (kaggriculture.py:125-132, 359-373); comment at kaggriculture.py:53-55 calls it "inner dim".
7. Workload 256 rows x 709 tokens = 181,504 token rows, 23.1x below 4,194,304.
Audit's own gaps: Triton promotion rules INFERRED; bmm path not checked; static-shape compiles; train/utils.py:79-88 model_compile='trunk' raises unless StatelessTransformerV1 while KaggricultureTransformer subclasses BaseModelAPI (kaggriculture.py:139), so how run_ppo compiles the trunk is unverified; largest-M forwards (PPO recompute, eval, BC) not audited; persistent-TMA/decompose-K off by default (config.py:1422-1424).

### Probe (measured, pod GPU 0, torch 2.9.0, sm_120, 514 s)

Precision as run_ppo: fp32 params, bf16 autocast, TF32, torch.compile(max-autotune-no-cudagraphs, dynamic=True), compiled small first; every point compared with eager. Model source = git archive of kg/rebuild-model @ 1ddc71d. L = 2^31 / GEMM input width.

| Case | Correct at | First failure |
|---|---|---|
| Real trunk packed, guard on | 62,003 / 181,504 tokens; 5,914 and 5,915 dense rows; 4,194,303 tokens (L-1); max abs diff <= 0.17 | Guard ValueError at 4,194,304 tokens, 5,916 and 11,830 rows, 0 trunk calls |
| Real trunk padded, guard on | 5,916 rows -> calls [4,193,735, 709]; 11,830 -> [4,193,735 x2]; max abs diff <= 0.18 | none |
| Real trunk packed, guard off | - | 4,194,305: 2 tokens wrong (max abs diff 45.8); 4,194,444: 236 wrong; 4,198,400: 4,665 wrong, non-finite; 8,387,470: illegal memory access; 8,388,888: correct (int64 recompile) |
| Real trunk padded, guard off | - | 4,194,444: 287 wrong; 8,387,470: all tokens past row 5,915 wrong, no fault |
| Linear 4096->16 | 524,288 (L) | 524,289: fault; ks0 'i32', A-load xindex = idx_n + 4096*idx_m |
| Linear 16->4096 | 1,048,576 (2L) | none |
| Linear 768->256 | 2,796,202 (L) | 2,796,203: fault in recompile/autotune |
| Linear 512->256 | 4,198,400 | 8,388,608: fault; at hint L autotune picked cuBLAS bias_addmm |
| Linear 256->512 | 8,388,608 | none |
| Linear 256->256 | 16,777,216 | none |
| MLP 256->512->256 fwd | 4,198,400 | 8,388,608 fault |
| MLP 256->512->256 fwd+bwd | 4,190,208 | 4,194,305 fault while compiling |

Probe conclusions: the limit is on the GEMM input side (output-only-wide GEMMs correct to 2L, refuting the audit's output-store half at these shapes); for training, backward makes it M x max(in, out) of each compiled Linear, = 512 for the trunk, so gemm_kmax=512 is "measured exact"; only M <= L is safe (above L depends on hint and autotune backend); the _run_trunk guard admits <= 4,194,303 packed tokens, raises at >= 4,194,304 before any trunk call, chunks padded rows correctly; rollout 256 rows correct and 23.1x under L; PPO minibatch 1,024 rows (726,016 tokens) 5.78x under L (arithmetic only); planned teacher precompute of 16,384 rows exceeds L at mean 256 tokens/row (needs chunking or teacher_segments_per_minibatch <= 46 at horizon 64); compiled heads/stems would need their own bound (768 input -> 2,796,203). Anomalies: pod venv has no flash-attn and pod config force_flash_attn: false, so packed path used aten._flash_attention_forward varlen instead (max abs diff 0.0022 vs per-sequence SDPA); copied branch tree had no built Rust extension (symlinked).

## What to verify

1. Kernel-source claims: open each cited file:line in the installed torch and confirm or refute it, citing exact file:line. Flag any cite that is wrong or does not say what is claimed. Specifically resolve whether the audit's R3 store-overflow prediction is actually contradicted by the probe or whether the probe simply never reached R3 for the output-wide cases (check compile hints, the ks* signature types in inductor_cache kernels, and whether autotune chose Triton or cuBLAS for those cases).
2. Recompute every limit and margin (L values, 181,504 = 256 x 709, 23.1x, 5.78x, 726,016, chunk sizes 4,193,735 and 5,915 x 709, the "<= 46 segments" teacher figure, 2,796,202, 371/376 stride figures).
3. Probe method: read probe code, driver, run statement and logs. Check for (a) corruption detected only by absence of faults (is there a real compiled-vs-eager numeric comparison, over all tokens, with a sound tolerance?); (b) single-subprocess CUDA poisoning after a fault contaminating later points (are post-fault points in fresh processes?); (c) wrong compile mode/dynamic setting vs run_ppo; (d) not the real model (is the trunk the real kaggriculture trunk at 1ddc71d, and is 1ddc71d's trunk/guard identical to the current working tree?); (e) whether the guard-bypass mechanism changes anything else; (f) whether the "exact" claim is actually measured at both L-1 and L for the real trunk, including backward; (g) whether the pod venv differs (torch/triton versions) from the run_ppo training environment; (h) sha256 integrity of copied artifacts vs sha256.txt and the run statement's recorded hashes.
4. Guard coverage: against kaggriculture.py (current working tree), check each coverage-gap claim: which modules are compiled, whether backward goes through the same guarded path, whether any forward (rollout, PPO recompute, eval, BC, teacher precompute) can reach the trunk bypassing `_run_trunk`, whether the train/utils.py model_compile wiring actually compiles the Kaggriculture trunk, and whether the guard rationale comment is accurate.
5. The owner's question: state plainly whether the evidence supports (a) "the limit is true" and (b) "we are under it", and what remains unproven.

## Output

Write findings as a numbered list, each labelled BLOCKING or NON-BLOCKING, with file:line evidence. Separate CONFIRMED (you checked it) from UNVERIFIED (you could not). End with exactly one line:

VERDICT: APPROVE | APPROVE WITH EDITS | REJECT
