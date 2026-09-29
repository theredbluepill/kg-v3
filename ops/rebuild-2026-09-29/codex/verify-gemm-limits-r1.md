1. **BLOCKING — The evidence does not yet establish that the production trainer is “under the limit.”**

   **CONFIRMED:** The current encoder enforces a conservative trunk bound, and the proposed rollout/PPO sizes fit it arithmetically. But Kaggriculture is absent from [ModelConfig](/Users/poonszesen/kg-v3/python/owl/model/config.py:10) and the [model factory](/Users/poonszesen/kg-v3/python/owl/model/factory.py:18). Moreover, [configure_model_compile](/Users/poonszesen/kg-v3/python/owl/train/utils.py:84) rejects Kaggriculture for `model_compile="trunk"`; [run_ppo](/Users/poonszesen/kg-v3/scripts/run_ppo.py:171) calls that helper. The probe deliberately bypasses this wiring by calling the model’s compile method directly.

   **UNVERIFIED:** Integrated Kaggriculture rollout, PPO recompute, evaluation, BC and teacher workloads. Thus the owner’s answer is: **the overflow is supported by evidence; the current encoder’s admission bound is enforced; production-wide compliance remains unproven.**

2. **BLOCKING — “Measured exact” overstates the boundary and backward evidence.**

   **CONFIRMED:** [trunk_packed.jsonl:13](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/trunk_packed.jsonl:13) measures correct forward output at **L−1 = 4,194,303**. Line 15 measures rejection at L with **zero trunk calls**. [trunk_packed_bypass.jsonl:5](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/trunk_packed_bypass.jsonl:5) measures corruption at L+1.

   **UNVERIFIED:** Unguarded real-trunk correctness at L, and real-trunk backward at any boundary. The [driver](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/driver.py:59) tests synthetic backward only at L−4096, L+1 and 2L. Its L+1 failure occurs during compilation, so it does not independently localize a backward-kernel threshold.

   Revise [results.md:69–72](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:69) and the [cookbook’s “exact” claims](/Users/poonszesen/kg-v3/cookbook/references/compiled-gemm-template-overflows-above-2-21-rows.md:28) to distinguish the **measured admitted forward endpoint**, the failing L+1 point, and inferred training bounds. “One token more conservative than needed” is not directly measured for the real trunk.

3. **BLOCKING — The audit’s R3 store-overflow prediction is genuinely contradicted by the probe.**

   **CONFIRMED:** The output-wide cases reached R3 and selected **Triton**, rather than avoiding the regime through cuBLAS:

   - **16→4096:** [attempt log:88–96](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/lin_16_4096.attempt0.log:88) records recompilation at M=524,288 and selection of `triton_mm_22`. The [generated wrapper:173](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/inductor_cache/lin_16_4096/s2/cs2raflcaujfiojvsiieiuixqqwlno52kc7ovgrg726jr3olpkqr.py:173) declares `ks0:i64`; line 304 launches that template. [JSONL:7–13](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/lin_16_4096.jsonl:7) reports correct output through M=1,048,576, where M·N=2³².
   - **256→512:** [attempt log:92](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/lin_256_512.attempt0.log:92) selects Triton at M=4,194,304; its [wrapper:173](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/inductor_cache/lin_256_512/5i/c5ilzhu47rp7cik4ldmoytyaelkmplcmrp5a3oyebfea6zliheik.py:173) likewise has `ks0:i64`. It passes through M=8,388,608.
   - **256→256:** the [large-shape wrapper:173](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/inductor_cache/lin_256_256/7v/c7vl2ehudho5i3rdc7pcdly3kgcipxmvj3veockxefxb7eqbtks6.py:173) has `ks0:i64`, and [JSONL:13](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/lin_256_256.jsonl:13) passes at 16,777,216.

   The missed dependency is `M → grid_m → group_size → pid_m`, visible in installed [mm.py:81–89](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/mm.py:81). Rematerializing `rm` therefore does not establish that it is int32.

   **UNVERIFIED:** Exact Triton IR promotion, because lowered IR is unavailable. The numerical contradiction is confirmed. The audit’s universal **“overflow iff M·max(…) > 2³¹”** rule must be withdrawn; a conservative admission rule is a different claim.

4. **BLOCKING — The generated-kernel summaries misreport signatures and omit cuBLAS calls.**

   **CONFIRMED:** [analyze_kernels.py:34](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/analyze_kernels.py:34) finds the first signature in the entire file, rather than the signature belonging to each template. Line 52 prints only the first variant. Consequently [kernel_analysis.txt:4](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/kernel_analysis.txt:4) reports a pointwise `xnumel:i32` signature as the GEMM signature and obscures the i64 variants.

   The regex at [line 28](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/analyze_kernels.py:28) also omits `bias_addmm`, yielding zero extern GEMMs for wrappers that actually call [extern_kernels.bias_addmm](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/inductor_cache/lin_512_256/ke/ckeabf3uauhket4dovmchx2xyricmf6pfjm7eui5vwmwjrlkqkfs.py:187).

   Correct the summaries before using them as dtype/backend evidence. The raw kernels remain usable.

5. **NON-BLOCKING — The core source audit is substantially confirmed, with precise qualifications.**

   **CONFIRMED:** Installed [torch/version.py:4–7](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/version.py:4) identifies **2.9.0**, git **`0fabc3ba44823f257e70ce397d989c8de5e362c1`**, with no CUDA build. Triton is absent locally.

   In the installed `_inductor` source:

   - `select_algorithm.py:1530–1539` chooses and emits `INDEX_DTYPE`; standard `kernel/mm.py:80–136` does not consume it. Consumers found by search are flex templates, including `kernel/flex/templates/flex_attention.py.jinja:48–50`. ADDMM reuses the MM template at `kernel/mm.py:997–1006`.
   - `select_algorithm.py:1212–1213` disables ordinary range-tree code generation.
   - `select_algorithm.py:388–395` constructs `SIMDKernelFeatures([], output_numel)`. `codegen/simd_kernel_features.py:133–150` therefore makes this separate decision using output numel. `select_algorithm.py:578–585` and `codegen/triton_utils.py:78–81` propagate it into template size-argument types.
   - `codegen/simd.py:1413–1443` checks **storage sizes**, not abstract matrix dimensions. M·N, M·K and K·N are the contiguous-layout specialization. Guards are installed only on the successful int32 branch.
   - The hint citation must include **`utils.py:3066`**, where the operative return occurs. The cited `sizevars.py:404–426,444–445` and `torch/fx/experimental/symbolic_shapes.py:7709–7714` support the guard path.

   R1 guards protect its current specialization; **recompilation itself is not a safety guarantee**, because it can enter R2.

6. **NON-BLOCKING — Two source-scope corrections are needed.**

   **CONFIRMED:** Ordinary pointwise/reduction kernels have the described safeguards: `codegen/simd_kernel_features.py:133–150`, `codegen/simd.py:1435–1442`, and `codegen/triton.py:4451–4459,4480–4481`. However, forcing non-template `ks*` arguments to int64 is conditional on **`use_block_ptr=False`**, explicitly shown in [triton_utils.py:141–147](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton_utils.py:141).

   Persistent TMA is disabled by default, but **decompose-K is not**: [config.py:1434–1442](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/config.py:1434) defaults to 10 splits and threshold 32. [utils.py:1825–1840](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/utils.py:1825) allows it when K≥32M and K≥32N, subject to additional conditions. This can matter for weight-gradient GEMMs.

   **UNVERIFIED:** A general BMM boundary. [bmm.py:85–110](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/kernel/bmm.py:85) confirms raw batch pids and stride-dependent offsets without `INDEX_DTYPE` casts, but does not establish that the MM formula covers its batch indexing. Static compilation and persistent-TMA behavior remain unmeasured.

7. **NON-BLOCKING — The probe detects numerical corruption and isolates post-fault processes.**

   **CONFIRMED:** [probe_linear.py:56–82](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/probe_linear.py:56) compares every output row against eager using `0.02 + 0.02|ref|`, counts nonfinite values, and records bad-row ranges. Its backward branch also compares forward output, input gradients and parameter-gradient differences.

   [probe_trunk.py:126–146](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/probe_trunk.py:126) checks every channel of every **valid token**, reporting thresholds of 0.25 and 1.0. This separates the observed normal differences, approximately 0.18 or less, from the L+1 corruption reaching 45.8. It is substantially stronger than absence of faults.

   [driver.py:108–138](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/driver.py:108) starts fresh subprocesses after failures; the [driver log](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/driver.jsonl:14) confirms those restarts and 514-second completion.

   **UNVERIFIED:** That the tolerances exclude every subtle numerical error across arbitrary inputs. They convincingly detect the reported corruption, but do not prove universal BF16 equivalence. Compiler/autotuner faults also do not identify the precise faulting candidate without further traces.

8. **NON-BLOCKING — Model identity, guard dispatch and compilation scope are confirmed.**

   **CONFIRMED:** Current `kaggriculture.py` is byte-identical to `1ddc71d`, SHA-256 **`92f4df615656f761262ecc62a1219336e5b7904bcde3a4876ccd2cf4bc109dee`**. The [compile hook](/Users/poonszesen/kg-v3/python/owl/model/kaggriculture.py:394) covers blocks and final normalization. Stems and critic are outside it; **`actor_input_proj` and the nine action heads are absent**, rather than implemented eager modules.

   [The guard](/Users/poonszesen/kg-v3/python/owl/model/kaggriculture.py:347) rejects packed M·512≥2³¹ and chunks padded rows. The bypass changes only [the limit constant](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/probe_trunk.py:172), thereby disabling rejection/chunking.

   Current encoder/value calls enter `_run_trunk`; no production direct bypass was found. However, [forward/evaluate_actions](/Users/poonszesen/kg-v3/python/owl/model/kaggriculture.py:409) and [teacher APIs](/Users/poonszesen/kg-v3/python/owl/model/base.py:154) are unfinished. Backward follows the accepted forward’s autograd graph; it **does not re-enter the Python guard**.

   The [comment at line 53](/Users/poonszesen/kg-v3/python/owl/model/kaggriculture.py:53) should distinguish observed overflow **above** the bound from conservative rejection **at** equality.

9. **NON-BLOCKING — Arithmetic checks pass, with notation and equality corrections.**

   **CONFIRMED:** Recomputed without torch:

   | Width/stride | floor(2³¹ / width) |
   |---|---:|
   | 256 | 8,388,608 |
   | 512 | 4,194,304 |
   | 768 | 2,796,202 |
   | 4096 | 524,288 |
   | 371 | 5,788,365 |
   | 376 | 5,711,392 |

   For 768, the exact quotient is **2,796,202⅔**; the first exceeding integer is 2,796,203. The 256→512 and 512→256 conservative comparison bounds are both 4,194,304.

   `256×709=181,504`, giving **23.1086×** headroom. `1024×709=726,016`, giving **5.77715×**. `5915×709=4,193,735`; 5,916 rows split into `[4,193,735,709]`, and 11,830 into two equal chunks. These match [padded results:13–15](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/trunk_padded.jsonl:13).

   Teacher rows are `128×64×2=16,384`. Mean **256** tokens already reaches L and triggers the guard; safe mean must be **below 256**. With 709 tokens/row, 46 segments produce **4,174,592** tokens; 47 produce **4,265,344**, so ≤46 is correct.

   [results.md:51](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/results.md:51) defines L using input width, but its output-wide “2L” entries use output width. Use separate input/output bounds or consistently define the table’s comparison bound.

10. **NON-BLOCKING — The 371/376 figures are conditional, not a measured stem limit.**

    **CONFIRMED:** The actor stem input is 371, but [ObservationInputStem](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:2750) makes the actual topology **371→512→256**. A compiled full stem would also include the 512-wide projection.

    BF16 alignment can round 371 to 376 under [pad_mm.py:69–71,121–130](/Users/poonszesen/kg-v3/.venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/pad_mm.py:69), but padding is conditional at line 728.

    **UNVERIFIED:** Any generated 376 stride for these currently eager stems. The arithmetic figures are valid conditional stride bounds, not observed kernel limits.

11. **NON-BLOCKING — Precision settings match, but full environment equivalence does not.**

    **CONFIRMED:** Probe code uses fp32 parameters, bf16 autocast, TF32 and dynamic `max-autotune-no-cudagraphs`, consistent with [training utilities](/Users/poonszesen/kg-v3/python/owl/train/utils.py:35). Runtime records identify torch **2.9.0+cu128** and generated metadata identifies **sm_120**.

    Packed attention uses the disclosed ATen shim, whereas [the model preset](/Users/poonszesen/kg-v3/configs/model/kaggriculture.yaml:11) requires FlashAttention. The measured shim difference is **0.00225955**, from [trunk_packed.jsonl:1](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/trunk_packed.jsonl:1). Also, packed maximum difference is **0.17025495**, so the table’s literal “≤0.17” needs rounding qualification.

    **UNVERIFIED:** Complete pod/training package equivalence, pod torch git identity, and independently captured Triton/driver versions. Triton 3.5.0 matches [the lockfile](/Users/poonszesen/kg-v3/uv.lock:4227), but the supplied process records do not emit its installed version. The Rust-extension symlink and missing FlashAttention are disclosed deviations, not evidence of a full production-path check.

12. **NON-BLOCKING — Listed checksums pass; custody coverage is incomplete.**

    **CONFIRMED:** All **14 entries** in [sha256.txt](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/gemm-limits-2026-09-29/pod-run/sha256.txt:1) match. All four probe-source hashes match [the run statement](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/run-statements/gemm-limits-probe.md:12). Recorded git tree IDs and model-file hash also match.

    **UNVERIFIED:** The manifest does not cover generated kernels, attempt logs, `driver.out` or `kernel_analysis.txt`. Reproducing the documented `git archive 1ddc71d python/owl` command yields SHA-256 **`29cfaafec3f899fa8a7d84e020a38f5372baae9bc3d9ba6a94c975fbc427996b`**, rather than the recorded **`47293ba8…`**. The original archive is absent, so this discrepancy needs reconciliation; it does not negate the matching model blob and tree identities.

VERDICT: APPROVE WITH EDITS