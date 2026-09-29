# Rebuild results log

Working evidence for `ops/rebuild-2026-09-29/plan.md`. Durable conclusions move to the cookbook when a task closes.

## Task 0.2 — CUDA illegal memory access (reference branch, pod `w7ia3zvxqsvs3g`)

Run statement: `run-statements/cuda-repro-blocking.md`. Launched 2026-09-29T03:34:45Z with `CUDA_LAUNCH_BLOCKING=1`, reusing the crashed run's own config and the BC best checkpoint (`ffd7d9e4…`), 2 ranks.

**Observation while running (iteration 1, before any fault):** the first PPO update on BC-policy rollouts shows `policy/logratio_mean −3.77`, `approx_kl 3.43`, `ratio_mean 0.66`, `logratio_abs_max 12.08`, and a KL stop after 2 optimizer steps. At the first minibatch the weights haven't changed, so replayed log-probs should equal rollout log-probs (log-ratio 0). A mean of −3.77 nats means the reference model's **sampling path and evaluation path disagree** on the states the BC policy reaches. Rollout was 4,878 env steps/s even with blocking launches; evaluation: 128 games, win rate 8.6% vs the BC incumbent.

Interim interpretation (superseded by the result below): I read the mismatch as an H1 index overflow in the model's own code. The actual cause is a compiler-generated GEMM overflow (H3), which corrupts the rollout activations the same way.

### Result — root cause established (2026-09-29, 03:42–03:55Z)

- **Reproduced:** the blocking run faulted after 3 iterations on both ranks (`exit=1`, 7m43s wall), matching the original crash. With blocking launches, both tracebacks end in the same compiled kernel, `triton_tem_fused__to_copy_addmm_gelu_t_view_9`. This is an Inductor **max-autotune Triton GEMM template** in the transformer trunk MLP (`M = ks0·ks1` = batch rows × tokens, K = 1024, N = 256), called from `KaggricultureTransformer._run_trunk`. It is not the model's own indexing, not the async copies (H2), and not attention.
- **Mechanism:** the kernel declares `INDEX_DTYPE = tl.int64` but computes `rm = pid_m*BLOCK_M + arange` and `xindex = k + 1024*idx_m` in int32. Once M > 2²¹ = 2,097,152, `1024·idx_m` passes 2³¹ and wraps. Wrapped addresses first land in valid memory (silent corruption), then eventually outside it (fault). The generated source is `/tmp/torchinductor_root/2e/c2ezrgfnyc…py` on the pod.
- **Controlled confirmation (`int32_probe.py`, sha256 `78dd05d3…`; log `ea88948b…`):** a synthetic MLP (256→1024, GELU, 1024→256), BF16, `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)`, compiled at a small shape first. At M = 2,088,960 (< 2²¹) max |compiled − eager| = 0.0078 (BF16 noise). At M = 2,105,344 (> 2²¹): **CUDA illegal memory access**. torch 2.9.0+cu128, RTX PRO 6000 Blackwell, driver 595.91.07.
- **Why it hit the reference:** a rollout forward used 4096 envs × 2 seats = 8,192 rows, so any padded sequence above 256 tokens crossed the threshold. Tiles alone are 200 tokens, so the extra actors a BC policy hires push dense states over. Training minibatches (128 rows) stay far below it. That explains **both** symptoms: the fault, and the iteration-1 log-ratio of −3.77 (sampling ran on corrupted activations; replay didn't).
- **Consequence for the reference's PPO-from-BC failure:** its KL spikes and falling win rates came from corrupted rollout activations. They are not evidence that the BC policy or the BC→PPO handoff is bad.

### Design requirements for the rebuild (from this result)

1. **Workload:** Isaiah's cadence (Task 3.4: 128 envs/rank → 256 rows per forward) keeps M far below the threshold. Even at 710 padded tokens, M ≈ 182k.
2. **Guard (Task 2.1):** the model's compiled trunk path fails fast when `rows × tokens × max_inner_dim ≥ 2³¹`, or chunks the forward, with a test at the boundary. This protects evaluation, BC and any future larger batch.
3. **Silent-corruption alarm (Task 3.x):** at the first minibatch of every update, before any optimizer step, |mean log-ratio| must be ≈ 0. A threshold breach aborts the run with the offending batch shape, because this is how the corruption was first visible.
4. **Upstream:** record the torch version. Don't upgrade torch or drivers as a fix without a separate, measured decision. Retest the probe after any upgrade.

### Torch 2.8 comparison (ran 03:54Z; the owner rejected the command after it had already started)

The unchanged probe under the pod's system torch 2.8.0+cu128 on GPU 0 gives max |compiled − eager| = 0.0078 both below (M = 2,088,960) and above (M = 2,105,344) the threshold: **no overflow in torch 2.8** for this template. (The "out of resource" autotune lines are candidate configs exceeding sm_120 shared memory, which autotune discards.) Log: pod `runs/cuda-repro-2026-09-29/int32_probe_torch28.log`.

**Owner decision: stay on torch 2.9** (Isaiah's pin, with its locked flash-attn 2.8.3 / triton 3.5.0). The 2.9 overflow is handled by the design requirements above: Isaiah's cadence keeps rows far below the threshold, the trunk chunks and guards, and the first-minibatch log-ratio alarm catches silent corruption. Torch 2.8 remains a measured fallback if 2.9 ever needs replacing.

## GEMM limits at our shapes (2026-09-29, 05:23–05:32Z, pod `w7ia3zvxqsvs3g`, GPU 0 only)

Revised after Codex verification (`codex/verify-gemm-limits-r1.md`, APPROVE WITH EDITS). The first draft overstated three things: it called the trunk bounds "measured exact", stated a universal `M·max(K, N)` overflow rule, and cited generated-kernel summaries from a buggy analyzer. This text replaces it.

**Scope: production compliance is unproven.** The probe compiled the trunk directly through the model's own `compile_transformer_trunk`. The production wiring does not reach it yet:
- Kaggriculture is absent from `ModelConfig` (`python/owl/model/config.py`) and the model factory (`python/owl/model/factory.py`).
- `configure_model_compile` (`python/owl/train/utils.py:84`, called from `scripts/run_ppo.py:171`) rejects this model for `model_compile="trunk"`.
- `forward`/`evaluate_actions` and the teacher APIs are unfinished. `actor_input_proj` and the nine action heads are **not implemented yet**; they are absent, not eager.

What this run supports: the overflow is real at our widths, and the current encoder's `_run_trunk` admission bound is enforced. Integrated Kaggriculture rollout, PPO recompute, evaluation, BC and teacher workloads remain **unverified** until Task 3.1 wires the model into the trainer and Phase 6 runs them on the pod.

**Anomaly: no FlashAttention on the pod.** The pod venv has **no `flash-attn` package**, and the pod's run config had `force_flash_attn: false`. That config lives on the pod and was not copied here. The model preset (`configs/model/kaggriculture.yaml`) and Isaiah's recipe (plan I8) require `force_flash_attn: true` with flash-attn 2.8.3. Consequences:
- The packed path in this probe ran on a shim: torch's own varlen `aten._flash_attention_forward` in place of `flash_attn_varlen_func`. The shim was checked against per-sequence SDPA: max |Δ| 0.0022 in the pre-launch smoke check, and 0.00225955 in-run (`pod-run/trunk_packed.jsonl:1`). Every GEMM, the pack/unpack and the guard are branch code, but the packed results say nothing about the real flash-attn kernel.
- **Phase 6 must install or build flash-attn 2.8.3 on the pod (`uv sync --extra flash-attn`) and verify that the real FlashAttention path ran before any qualification or throughput claim** (plan, Phase 6).

Run statement: `run-statements/gemm-limits-probe.md`. Driver wall time: 514 s, within the 45-minute budget. Inputs:
- Probes: `gemm-limits-2026-09-29/probe/`, sha256 `0a7af363…` (linear), `5ce3f09a…` (trunk), `aa42c12b…` (driver).
- Model source: `kg/rebuild-model` @ `1ddc71d`; `kaggriculture.py` sha256 `92f4df61…`, byte-identical to the current tree (Codex).
- The source tar's sha256 `47293ba8…` is the output of `git archive --prefix=gemm-limits-src-1ddc71d/ 1ddc71d python/owl`. Without `--prefix`, the same pathspec gives `29cfaafe…`, which is the hash Codex reproduced from the run statement's command. Both hashes were reproduced locally with git 2.50.1. The run statement had omitted the prefix and is now corrected. The tree IDs and model-file hash match either way.
- Stack: torch 2.9.0+cu128 (runtime records), sm_120 (generated metadata), RTX PRO 6000 Blackwell, driver 595.91.07.
- **Triton version not captured:** 3.5.0 is the `uv.lock` pin (`uv.lock:4227`), but the process records do not emit Triton's installed version. The pod's torch git identity was not captured either.
- The Rust-extension symlink and the missing flash-attn are disclosed deviations. This run is not a full production-path check.

Setup, matching `run_ppo`'s precision:
- fp32 parameters under `torch.autocast(bfloat16)`, TF32 on.
- `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)`, compiled at a small shape first. For `lin_4096_16`, the logs show the warm compile specialized M: the recompile guard reads "expected 4096". Dynamic-M graphs come from the first recompile.
- Every point is compared against the eager (cuBLAS) forward, so silent corruption is measured, not just faults.
- Each case runs in a fresh process, and the driver restarts after a fault.

**Custody:**
- Logs, JSONL, generated kernels: `gemm-limits-2026-09-29/pod-run/`; the pod copy is at `/workspace/kg-v3/runs/gemm-limits-2026-09-29/`.
- `pod-run/sha256.txt` has 14 entries, all re-verified by Codex. It covers the 13 JSONL files and the *first* `kernel_analysis.json` (`c295ce49…`), which is superseded and kept only on the pod.
- The pod manifest did not cover the generated kernels, attempt logs, `driver.out` or `kernel_analysis.txt`. `pod-run/sha256-local.txt` hashes every file in the local copy, including the regenerated analysis.

**Generated-code attribution** comes from `probe/analyze_kernels.py` revision 2 (sha256 `ee26698a…`). Revision 1 gave every template the first signature in its file (usually a pointwise `xnumel: i32`), printed one variant per name and missed `extern_kernels.bias_addmm`. Revision 2 was rerun locally on the copied caches (text parsing only) and produced `pod-run/kernel_analysis.{json,txt}`. The analyzer cannot map a wrapper to its compile hint. The mappings below come from the recompile hints in the attempt logs, by elimination.

Definitions:
- **L_in** = ⌊2³¹ / GEMM input width K⌋; **L_out** = ⌊2³¹ / output width N⌋.
- For the trunk, L = 2³¹ / 512 = 4,194,304 is the design bound (below). M = token rows: padded rows × 709, or packed tokens.
- For single GEMMs, "correct" means every element satisfies `|Δ| ≤ 0.02 + 0.02|ref|`.
- For the trunk, "correct" means no valid token with |Δ| > 0.25 in any channel. Passing points reached max |Δ| 0.1703 packed (0.17025495 at 5,915 dense rows) and 0.1792 padded; the L+1 corruption reached 45.8.
- These tolerances detect the corruption reported here. They do not prove BF16 equivalence for arbitrary inputs.

| Case | L_in / L_out | Correct at (measured) | First failure (measured) | Kind |
|---|---|---|---|---|
| **Real trunk, packed, guard on** | trunk L = 4,194,304 | 256 rows sparse (62,003 tok), 256 dense (181,504), 5,914 and 5,915 dense rows, **4,194,303 tok = L−1** (`trunk_packed.jsonl:13`) | none admitted. The guard raises `ValueError` at 4,194,304 tok = L (line 15), 5,916 rows and 11,830 rows, with **zero trunk calls** | — |
| **Real trunk, padded, guard on** | trunk L | 256, 5,914 and 5,915 rows in one call. 5,916 rows → calls [4,193,735, 709]. 11,830 rows → [4,193,735, 4,193,735] | none | — |
| Real trunk, packed, **guard off** | trunk L | (L−1 above; L itself was not run unguarded) | 4,194,305 = **L+1** (`trunk_packed_bypass.jsonl:5`): 2 tokens wrong (max \|Δ\| 45.8, rows 46 and 5,915). 4,194,444: 236 tokens. 4,198,400: 4,665 tokens, 1.19M non-finite values. **All silent.** 8,387,470: illegal memory access. 8,388,888: correct again | silent, then fault |
| Real trunk, padded, **guard off** | trunk L | — | 4,194,444: 287 tokens wrong. 8,387,470: every token past row 5,915 wrong or non-finite (4,193,735 tokens). **No fault at either size** | silent |
| Linear 4096→16 (wide input only) | 524,288 / 134,217,728 | 524,288 = L_in | 524,289 = L_in+1: run-time fault | fault |
| Linear 16→4096 (wide output only) | 134,217,728 / 524,288 | 1,048,576 = 2·L_out (M·N = 2³²) | none | — |
| Linear 768→256 (planned `actor_input_proj`) | 2,796,202 / 8,388,608 | 2,796,202 = L_in (2³¹/768 = 2,796,202⅔) | 2,796,203 = L_in+1: fault in autotune benchmarking at the new shape (`lin_768_256.attempt0.log:91`) | fault |
| Linear 512→256 | 4,194,304 / 8,388,608 | 4,198,400 = L_in+4,096. At hint L_in, autotune picked extern cuBLAS `bias_addmm` (`lin_512_256.attempt0.log:92`) | 8,388,608 = 2·L_in: fault in autotune benchmarking (`…attempt0.log:179`) | fault |
| Linear 256→512 | 8,388,608 / 4,194,304 | 8,388,608 = L_in = 2·L_out | none | — |
| Linear 256→256 (q/k/v/out) | 8,388,608 / 8,388,608 | 16,777,216 = 2·L | none | — |
| MLP 256→512→256, forward | max width 512: 4,194,304 | 4,198,400 | 8,388,608: fault during the recompile | fault |
| MLP 256→512→256, forward + backward | max width 512: 4,194,304 | 4,190,208 (L−4,096) | 4,194,305 (L+1): fault **while compiling** at that shape. Only L−4,096, L+1 and 2L were tested (`probe/driver.py:59`) | fault |

Generated code, from the corrected summaries (`pod-run/kernel_analysis.txt`):
- **Failing input-wide graphs keep a 32-bit size argument.**
  - 4096→16: the graph recompiled at hint 520,192 (`inductor_cache/lin_4096_16/rv/…`) has `ks0: i32` with A-load `xindex = idx_n + 4096*idx_m`. It ran correctly at L_in and faulted at L_in+1.
  - The recompile at hint 528,384, where M·4096 > 2³¹ (`…/mp/…`), still has `ks0: i32`. It declares `INDEX_DTYPE = tl.int64`, which the standard mm template does not consume. It faulted at 528,384 and at 1,048,576.
  - 768→256: 3 of its 4 wrappers call extern `bias_addmm`. The Triton wrapper has `ks0: i32` with A-load `768*idx_m`.
- **The trunk's corrupting kernel is the MLP down-projection.** In the packed bypass's first recompile above L (`trunk_packed_bypass/ar/…`), the down-projection `…addmm_gelu_t_9` (A-load `512*idx_m`, store `256*idx_m`) kept `ks0: i32`. The up-projection `…add_addmm_native_layer_norm_t_7` (A-load `256*idx_m`, store `512*idx_m`) got `ks0: i64`. The padded bypass shows the same split (`trunk_padded_bypass/2b/…`). The recompile at 8,388,888 tokens, where M·256 > 2³¹ as well (`trunk_packed_bypass/jg/…`), has `ks0: i64` in every template and was correct. (The first draft called `…native_layer_norm_t_7` the down-projection; that was wrong.)
- **Output-wide graphs that passed above L_out have `ks0: i64`.** The 16→4096 recompile at M = 524,288 selected Triton `triton_mm_22` (`lin_16_4096.attempt0.log:88–96`; wrapper `…/s2/…:173`, launched at line 304). The same holds for 256→512 at 4,194,304 (`…/5i/…:173`) and 256→256 at 16,777,216 (`…/7v/…:173`).
- Extern cuBLAS: 2 of 3 `lin_512_256` wrappers and several trunk-padded and MLP wrappers call `bias_addmm` or `mm`.

What is **measured** (this stack, these shapes):
1. **Real trunk forward, guard on:** correct at L−1 = 4,194,303 packed tokens. The guard rejects L with zero trunk calls. The padded path chunks to 4,193,735 tokens per call, and the chunks are correct.
2. **Real trunk forward, guard off:** silent corruption at L+1 = 4,194,305. It stayed silent up to at least 4,198,400 packed tokens. The padded path corrupted half its tokens at 8.39M with no fault. A crash is not a detector, and a corrupted token appeared at row 46, far from the wrap point.
3. **Single GEMMs:** input-wide GEMMs (4096→16, 768→256) were correct at L_in and failed at L_in+1. Output-wide GEMMs (16→4096, 256→512, 256→256) stayed correct up to M·N = 2³² or 2·L. **The measured overflow at these shapes is input-side: the A-load, M·K.** The audit's universal "overflow iff M·max(K, N) > 2³¹" rule is **withdrawn**.
4. **Above the bound, behavior is not a usable threshold.** It depends on the recompile hint and on autotune's backend choice. 512→256 stayed correct to L+4,096 only because autotune picked cuBLAS. The real trunk corrupted at L+1 and was correct again at 8,388,888 after an i64 recompile. A passing test above the bound proves nothing. **Recompilation is not a safety guarantee:** R1's int32 guards protect only their own specialization, and a recompile can land in the unsafe regime (R2).

**Mechanism** (source reading plus generated code; partly unverified). Codex confirmed these points in the installed torch 2.9.0 (`0fabc3ba…`) `_inductor`:
- The emitted `INDEX_DTYPE` comes from an all-buffer **storage-size** check (`codegen/simd.py:1413–1443`), not from abstract matrix dimensions; M·N, M·K and K·N are the contiguous-layout specialization.
- The template's **size-arg** dtype is decided separately: `select_algorithm.py:388–395` builds `SIMDKernelFeatures([], output_numel)` (`codegen/simd_kernel_features.py:133–150`). `select_algorithm.py:578–585` and `codegen/triton_utils.py:78–81` carry the result into the template's size-arg types.
- The hint path returns at `utils.py:3066`, with guards via `sizevars.py:404–426,444–445` and `symbolic_shapes.py:7709–7714`. Guards are installed only on the successful int32 branch.
- `INDEX_DTYPE` is emitted (`select_algorithm.py:1530–1539`) but not consumed by the standard mm/addmm template (`kernel/mm.py:80–136`; ADDMM reuses it at `kernel/mm.py:997–1006`). That is why `INDEX_DTYPE = tl.int64` next to `ks0: i32` did not help.

With an i64 `ks0`, the chain `M → grid_m → group_size → pid_m` (`kernel/mm.py:81–89`) starts from a 64-bit value. That plausibly promotes `idx_m`, and with it the A-load offset. With an i32 `ks0`, `K·idx_m` wraps once M·K > 2³¹. This matches every failing and passing graph whose kernel can be attributed from the retained generated code. The 768→256 and 512→256 failures happened during autotuning and do not identify the failing candidate kernel; that localization gap remains. **Triton IR promotion is unverified**, because no lowered IR was captured. Rematerializing `rm` does not by itself establish int32.

What is **inferred, not measured**:
- **The unguarded real trunk at exactly L** was not run. The synthetic single GEMMs were correct at exactly L_in, so the guard's `≥ 2³¹` rejection may be stricter than necessary. For the real trunk that is unmeasured, so the guard is a conservative admission rule, not a measured edge.
- **Real-trunk backward** was not measured at any size. The synthetic MLP backward was tested only at L−4,096 (correct), L+1 and 2L. Its L+1 failure happened during compilation, so no backward-kernel threshold is localized.
- **Design bound for training: M × max(in, out) over every Linear in a compiled region.** Backward consumes forward outputs as GEMM inputs: dX = dY·W reads dY at the forward *output* width. Under the measured input-side rule, a backward GEMM can therefore wrap at M × (forward output width) even when the forward's input side is safe. Weight-gradient GEMMs reduce over M and index offsets up to M × width.
  - For the trunk, every Linear has max(in, out) ≤ 512, so `gemm_kmax = 512` is that design bound. It is justified by the input-side rule plus backward, not by any output-store overflow.
  - Decompose-K is **enabled by default**: 10 splits, threshold 32 (`config.py:1434–1442`). It is allowed when K ≥ 32M and K ≥ 32N, subject to more conditions (`utils.py:1825–1840`), which weight-gradient GEMMs can meet. Unmeasured.
- **Other templates:** outside mm/addmm, pointwise and reduction kernels have their own safeguards (`codegen/simd_kernel_features.py:133–150`, `codegen/simd.py:1435–1442`, `codegen/triton.py:4451–4459,4480–4481`). Forcing non-template `ks*` arguments to int64 applies only when `use_block_ptr=False` (`codegen/triton_utils.py:141–147`).
- `bmm` stays unmeasured. `kernel/bmm.py:85–110` uses raw batch program ids and stride-dependent offsets without `INDEX_DTYPE` casts, and the MM rule does not establish its batch indexing. Static-shape compiles and persistent TMA (disabled by default) are also unmeasured.

Where our workloads sit, as rows × 709 tokens against L = 4,194,304 (2³¹/512):
- **Measured correct (forward):**
  - Rollout, 256 rows: 181,504 tokens, 23.1086× headroom. Compiled vs eager was measured at this shape, dense and sparse.
- **Arithmetic only:**
  - PPO minibatch, 1,024 rows (spm 8 × horizon 64 × 2): 726,016 tokens, 5.77715×. Its backward sits under the design bound; that is inferred, not measured.
- **Over the limit as planned:**
  - Teacher precompute, 128 segments × 64 × 2 = 16,384 rows. A mean of **256** tokens per row already reaches L (16,384 × 256 = 2²²) and triggers the guard, so the safe mean is **strictly below 256**.
  - At the worst case of 709 tokens per row, 46 segments give 4,174,592 tokens (under L) and 47 give 4,265,344 (over). So `teacher_segments_per_minibatch ≤ 46` at horizon 64, or chunk.
  - The packed guard would **raise**, not corrupt.
- **Uncovered by the trunk guard:**
  - The heads are not implemented yet. If the head core is ever compiled, the planned `actor_input_proj` (768 in) exceeds L_in at M = 2,796,203 rows × frames.
  - Stems are eager today. `ObservationInputStem` is 371→512→256 (`stateless_transformer_v1.py:2750`), so a compiled full stem would include a 512-wide projection, with the same L as the trunk.
  - The audit's 371/376 figures are conditional stride bounds, not observed kernel limits: ⌊2³¹/371⌋ = 5,788,365, and ⌊2³¹/376⌋ = **5,711,392**, if BF16 alignment padding rounds 371 to 376 (`fx_passes/pad_mm.py:69–71,121–130`). That padding is conditional (line 728). No 376 stride was generated for these eager stems.

**Open and handoffs:**
- Real-trunk backward and the unguarded real trunk at L: not measured.
- flash-attn: absent on the pod. Install and verify it in Phase 6 (above).
- bmm, persistent TMA, decompose-K, static compiles, and other torch/Triton versions: unchecked.
- **For Task 2.3:** the comment at `python/owl/model/kaggriculture.py:53` should distinguish the observed overflow *above* the bound from the conservative rejection *at* equality. It is part of the docstring correction in brief §9 item 1 and out of scope for this evidence commit.

## Phase 6.0 — flash-attn on the pod (2026-09-29, 06:16–06:27Z, pod `w7ia3zvxqsvs3g`, GPU 0 only)

Run statement: `run-statements/pod-flash-attn-setup.md`, which now has a post-run addendum on custody and corrections. Receipts are in `flash-attn-setup-2026-09-29/`; the README has identities and hashes. Codex reviewed them in `codex/verify-flash-attn-r1.md` (APPROVE WITH EDITS), and this section applies those edits.

Wall time was about 11 min of the 90-min budget, roughly $0.8 at $4.18/h, with no new billable resource. The operator transcript's first and last pod commands are at 06:16:21Z and 06:28:14Z, so the wall time and cost are estimates. The pod was left running and idle: idle checks are retained at 06:16:25, 06:24:26, 06:26:39 and 06:28:14Z (`flash-attn-setup-2026-09-29/post-run/operator_transcript_excerpts.txt`) and at 06:36:54Z (`flash-attn-setup-2026-09-29/post-run/git_idle_state_post_run.txt`).

**Outcome: the blocker is cleared for the forward path. The trunk numerics are not qualified (see (b)).** A separate v3 environment, `/workspace/kg-v3-rebuild` (`kg/isaiah-gap-closure` @ `69397da`, via git bundle; `.venv` from `uv sync --frozen --group dev --extra flash-attn`), has torch 2.9.0+cu128, triton 3.5.0 and flash-attn 2.8.3, and the model's forced packed FlashAttention path runs on sm_120.
- **Install:** the extra's sdist build (`FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`, no build isolation) downloaded the prebuilt release wheel `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` in 7.5 s. No compile, so no `MAX_JOBS`/arch settings were needed. Wheel sha256 `4e2f9e39…0810` equals the GitHub release digest. The installed `flash_attn_2_cuda` `.so` (sha256 `8ca052bf…5807`) is byte-identical to the wheel's member: the member was extracted from the retained wheel, both files have the same sha256, and `cmp` exits 0. That receipt was produced post-run at 06:37Z and is kept as `flash-attn-setup-2026-09-29/pod/wheel_member_compare.txt`, locally and on the pod.
- **sm_120 support:** flash-attn 2.8.3's `setup.py` includes `120` in its default arch list (CUDA ≥ 12.8). The installed `.so` carries 72 sm_120 cubins, alongside sm_80/90/100 (`cuobjdump --list-elf`). Not a blocker.
- **Rust extension:** `maturin develop` (justfile `build`, dev profile) → `owl.rs` sha256 `35239d1b…93b8`. `owl` and `owl.model.kaggriculture` import from the new checkout, and `flash_attn_available()` is True.

**(a) Kernel vs SDPA.** `flash_attn_varlen_func` ran on BF16 packed q/k/v with 8 heads × head_dim 32, 256 sequences of 214–709 tokens, 173,518 tokens in total. The reference is per-sequence `F.scaled_dot_product_attention` on fp32 inputs, with the backend chosen automatically. The math backend was not selected explicitly, and the backend actually used was not profiled.
- Against that reference, flash has max |Δ| 0.00359 and mean |Δ| 1.07e-4. BF16 SDPA against the same reference gives the same max |Δ| (0.00359, mean 1.06e-4). BF16 output rounding is a plausible explanation that fits this. Equal maxima do not prove it.
- Flash differs from BF16 SDPA by at most 0.00390625 = 2⁻⁸. Near |ref| ≈ 1.38, the largest reference magnitude, BF16 spacing is 2⁻⁷ = 0.0078125, so this is half a spacing there. It would be one spacing for values in [0.5, 1). The magnitude of the element with the largest error was not retained.
- No element falls outside `0.02 + 0.02|ref|`. The profiler records `flash::flash_fwd_kernel<…bfloat16…>`.

**(b) Model trunk.** Setup: preset `configs/model/kaggriculture.yaml` (width 256, depth 8, 8 heads, `force_flash_attn: true`), fp32 params, `autocast(bfloat16)`, TF32 via `configure_torch()`. Batch: `make_obs` with 128 envs = 256 rows, 173,108 present tokens, lengths 221–709, padded length 709.

**The trunk smoke completed with outliers.** In every comparison, 0.017–0.021 % of present-token elements fall outside `0.02 + 0.02|ref|`. The script reports these differences but asserts no numerical acceptance, so this run does not claim trunk numerical agreement.

| Comparison (present tokens) | elements | max \|Δ\| | mean \|Δ\| | outside tol |
|---|---|---|---|---|
| compiled flash vs eager flash | 44.3M | 0.0872 | 0.00542 | 0.018 % |
| eager flash vs eager padded SDPA | 44.3M | 0.0805 | 0.00498 | 0.021 % |
| compiled flash vs eager padded SDPA | 44.3M | 0.0775 | 0.00546 | 0.019 % |
| compiled vs eager, second shape (64 rows) | 10.9M | 0.0872 | 0.00542 | 0.017 % |

Reference output magnitudes reach 4.03, or 3.86 for the 64-row shape. BF16 spacing is 0.03125 in [4, 8) and 0.015625 in [2, 4). The magnitudes of the elements with the largest errors, and the locations of the outliers, were not retained.

The three paths differ from one another by similar amounts. BF16 rounding accumulated over 8 layers is a plausible explanation. It is not proven: similar pairwise differences do not rule out an error specific to one path.

**What would discriminate (future Phase 6 work, not run now):** run the same weights and batch through an fp32 reference with autocast off and TF32 off (padded SDPA). Compare both BF16 paths (flash and padded) against it, and retain the magnitudes and locations of the largest-error elements and of the outliers.
- If both paths show similar error distributions against fp32, and their outliers sit on large-magnitude or long-sequence elements, that supports rounding.
- If flash errors are larger, or cluster at pack boundaries or particular sequences, that indicates a path-specific error.

All outputs are finite, masked positions are exactly zero, and repeated compiled calls agree exactly (Δ = 0).

Evidence that the flash path ran:
- `use_flash_attn(x)` is True with x in bfloat16.
- `pack_sequence` was called exactly once per forward, as (173,108 tokens, max_seqlen 709), in both eager and compiled.
- In eager mode, the Python-level `varlen_attention` ran 8 times, one per block.
- The CUDA profiler shows `flash::flash_fwd_kernel` in both the eager and the compiled forward. The compiled forward also shows 12 Triton kernels.
- Compile: `compile_transformer_trunk("max-autotune-no-cudagraphs")`, dynamic. The first call took 20.6 s, including autotune. Autotune picked Triton templates for the 173,108-row mms.
- Control: the padded path (dispatch patched off) made no pack calls and launched no flash kernel. It used PyTorch's mem-efficient `fmha_cutlassF_bf16…sm80` kernel. The JSON field `eager_padded_use_flash_attn_x: true` reports the unpatched `owl.model.attn.use_flash_attn`, not the patched dispatch.
- Peak allocated memory was 1.72 GiB.

**(c) Tests.** On the pod, `uv run --frozen pytest tests/owl/model/test_attn.py` passed 7 of 7 with 0 skipped. That includes `test_varlen_attention_flash_backend` and `test_varlen_attention_matches_torch_sdpa_per_sequence`, which skip on the Mac. `tests/kaggriculture` passed 161 of 161.

**Limits and handoffs:**
- **Forward only.** The flash backward kernels, and compiled backward through the packed trunk, were not exercised. The actor heads don't exist yet (Task 2.3), and the model is not yet wired into `run_ppo`. So 6.1 still has to confirm flash in rollout, PPO update and evaluation, with backward included.
- The timings above are single profiled forwards. They are not throughput evidence.
- The Rust extension is a dev-profile (unoptimized) build. Build `--release` before any env-throughput measurement.
- **Custody:** the torch 2.9 wheel is a release asset added on 2025-12-17, after the v2.8.3 tag. `setup.py` resolves it by URL, and `uv.lock` pins only the sdist hash. Record the asset digest above. A rebuilt pod that resolves a different asset would change the kernel without changing the lock.
- **Pod state:**
  - The new venv and cache take about 3 GB; disk is at 20 GB free (62 %).
  - `/workspace/kg-v3-rebuild/runs/flash-attn-setup-2026-09-29/` also holds the 243 MB wheel re-download and a 55 MB Inductor cache.
  - `/workspace/transfer-flash-attn-2026-09-29/v3.bundle` is kept for custody.
  - A post-run read-only check at 06:38Z (`flash-attn-setup-2026-09-29/post-run/untouched_paths_post_run*.txt`) found:
    - `/workspace/kg-v3`: no mtime changes.
    - `/workspace/gemm-limits-src-1ddc71d`: no mtime or ctime changes.
    - `/workspace/kg-v3/.venv`: 19,796 entries have a new ctime. All are hard-linked regular files, and the sampled ones share an inode with the new venv (link count 3), which fits uv hard-linking from its cache. Their mtimes are unchanged, so no content write is recorded, but the two venvs now **share inodes**. An in-place edit of a file in either venv would change the other.
  - Source state:
    - Detached HEAD at `69397da` with a clean checkout: `git checkout` output at 06:18:38Z (transcript excerpts).
    - Zero tracked-file changes at 06:26:39Z.
    - Fresh post-run receipt at 06:36:54Z: HEAD `69397da`, detached, `git status --porcelain` empty, `runs/` gitignored.
  - Still **operator-reported**, with no retained receipt: "no credentials copied" and "nothing pushed". The only evidence is the clone's remote list, which shows just the bundle path.

## Model-only SPS ceiling (component) (2026-09-29, 06:45–06:49Z, pod `w7ia3zvxqsvs3g`, GPU 0 only)

**Component measurement, not end-to-end SPS.** These numbers are **conditional estimates for the specified A/B/C/D synthetic schedule**: model-only env steps/s if the engine, host copies, GAE, logging and DDP cost nothing. B (PPO-shaped loss) and C (teacher proxy) are surrogates, so their costs are not proven bounds on the eventual trainer. **Timing evidence only:** no finite-loss/gradient check and no sampling-versus-replay equality check were retained. At `e1458d2`, `configure_model_compile` (`model_compile="trunk"`) still rejects the Kaggriculture model; the probe compiled the trunk directly. (Revised after Codex review `codex/verify-sps-ceiling-r1.md`, APPROVE WITH EDITS.)

- Pre-run statement: `run-statements/model-sps-ceiling.md`, committed in `ed61770` at 06:44:12Z. Launch was at 06:45:05Z.
- Source: `e1458d2` (full model with heads).
- Settings: preset config, bf16 autocast over fp32 params, TF32, trunk compiled with `max-autotune-no-cudagraphs` (dynamic), heads eager, flash forced.
- Inputs: synthetic `make_obs` observations and synthetic grammar tables.
- Evidence and limits: `model-sps-ceiling-2026-09-29/README.md`.

| density (tokens/row) | t_A fwd 256 | t_B train 1,024 | t_C teacher-proxy 16,384 | t_D value 256 | update wall | ceiling SPS/rank | ≈2 ranks |
|---|---|---|---|---|---|---|---|
| sparse (222) | 11.09 ms | 147.5 ms | 852.0 ms | 7.48 ms | 3.93 s | 2,085 | ~4,170 |
| mid (303) | 12.87 ms | 177.6 ms | 987.7 ms | 9.50 ms | 4.66 s | 1,757 | ~3,514 |
| dense (709) | 25.41 ms | 326.6 ms | 1,690.6 ms | 22.04 ms | 8.56 s | 957 | ~1,913 |

**Formulas.**
- update wall = 64·t_A + t_C + 16·t_B + t_D, using medians over 20 CUDA-event iterations.
- Variability: p90/median ≤ +0.14 % everywhere except sparse B (p90 148.678 ms vs 147.472 ms median, **+0.818 %**). Dense B has one tail sample of **545.106 ms** vs a 326.579 ms median (cause unknown; p90 327.007 ms). The weighted sum of component p90s is not a measured whole-update p90; no whole update was timed.
- ceiling = 8,192 / update wall.
- Two ranks give about 2× that, assuming the all-reduce is small. The all-reduce was not measured.

**What dominates.**
- The 16 train steps take about 60 % of the synthetic schedule's update wall at every density.
- Teacher-proxy precompute takes 20–22 % and rollout sampling takes 18–19 %.
- This is a share of the schedule only. It does not attribute B's cost to heads backward, Muon or any other sub-phase, and A − D is the incremental sampling-path cost, not isolated head kernels.

**Engine budget.**
- For the engine to cost no more than the model, which halves the ceiling, it must step 8,192 env steps in at most 3.9 s (sparse) to 8.6 s (dense) per rank. That is about 0.48–1.05 ms per env step.
- To lose at most 10 % of the ceiling SPS, engine time E must satisfy 8,192/(M + E) ≥ 0.9 · 8,192/M, i.e. E ≤ M/9: 0.437 s (sparse), 0.518 s (mid), 0.952 s (dense) per update, or 53 / 63 / 116 µs per env step. (The earlier 0.1 × M figures, 0.39–0.86 s and 48–105 µs, are conservative.)
- The 64 rollout steps run serially with the engine. Each batched engine step of 128 envs therefore competes with an 11–25 ms sampling forward.

**Flash and chunks.**
- `use_flash_attn` was True on every call.
- Trunk chunks in C were **counted** as 1, 2 and 3 for sparse, mid and dense, as predicted (sparse is under the **packed**-token bound, 3,637,248 × 512 < 2³¹). The 2 head chunks were **calculated from source** (`ceil(16,384 / 11,096)`), not counted.

**Memory.**
- Peak allocated memory was at most **40.275 GiB** (B at dense), 42.4 % of 94.97 GiB, within the plan's 85 % rule, which is defined on `max_memory_allocated`. Complete-trainer memory fit is not qualified by this.
- Caching-allocator **reserved** peak is a separate observation: **84.994 GiB (89.5 %)** during dense C. The explanation that C reused B's cached blocks (carryover/fragmentation) is a **hypothesis**; no allocator snapshot or phase-order control was run.

**Compile.**
- On a cold cache the first call took 20 s for A and 40 s for B.
- With a warm cache it took 7–11 s.

**Profiling.** `nsys` absence on the pod is operator-reported (06:41:21Z pre-check; no command-output receipt retained). Nothing was installed and no timeline exists. In-step phase attribution is unresolved. Host/event median agreement cannot show the calls are intrinsically synchronous, because the timer calls `end.synchronize()` before reading the host clock.

**Run conduct.** The prelaunch capture showed 563 MiB on both GPUs (0 % util, no compute processes), not 0 MiB, and launch followed one second later. The driver did not enforce the statement's first-failure stop or 45-minute aggregate limit (three independent 25-min timeouts, no `set -e`). All densities exited 0 in 227 s total, so neither gap affected the timings.

## ATEN-only GEMM A/B (2026-09-29, 07:06–07:15Z, pod `w7ia3zvxqsvs3g`, GPU 0 only)

**Infrastructure diagnostic, not a learning change.** One setting changed: `torch._inductor.config.max_autotune_gemm_backends` from torch 2.9.0's default `"ATEN,TRITON,CPP"` to `"ATEN"`, so every compiled mm/addmm lowers to extern cuBLAS instead of Inductor's Triton mm templates. **Scope:** this excludes only the identified GEMM template paths (mm, addmm, bmm, decompose-K, persistent-TMA). It is not general immunity: FlexAttention generates its own Triton templates independently of this setting (torch `flex_attention.py:348, 776`; unused here), and static-shape compiles, bmm numerics and the real-trunk backward above the bound were not measured. The timing half is a **component measurement conditional on the A/B/C/D synthetic schedule** ("Model-only SPS ceiling (component)" above), not end-to-end SPS.

- Pre-run statement: `run-statements/aten-gemm-ab.md`, committed with the scripts in `9904121` at 07:06:00Z. Driver 07:06:31Z–07:14:55Z, 503.9 s, exit 0; no first-failure stop fired.
- Source: `e1458d2` (integration HEAD, unchanged); the pod checkout's porcelain was empty before and after.
- Stack: torch 2.9.0+cu128, triton 3.5.0, **real flash-attn 2.8.3**, driver 595.91.07.
- Idle gate passed at 0 MiB / 0 % / no processes. The pod was left running and idle.
- Evidence, custody and limits: `aten-gemm-ab-2026-09-29/README.md`.

**Correctness above the int32 bound (guard bypassed, compared element-wise to eager):** every case that failed under default backends was correct under ATEN-only.
- Real trunk, packed, at 4,194,305 / 4,198,400 / 4,194,444 / 8,387,470 tokens: 0 wrong tokens, max |Δ| ≤ 0.1699.
- Linear 768→256 at 2,796,203 rows and Linear 512→256 at 4,198,401 and 8,388,608 rows: 0 bad rows, max |Δ| 0.0 (the same cuBLAS call as eager).
- MLP 256→512→256 forward and backward at 4,194,305 rows: forward and dX clean; parameter-gradient relative max ≤ 0.0026.

The default-backend controls in the same run reproduced the failure: the trunk had 2 and 4,665 wrong tokens (the second with 1,194,240 non-finite values), identical to the earlier probe, and Linear 768→256 hit an illegal memory access during autotuning. The A/B therefore discriminates.

**Kernel evidence** comes from the analyzer (rev 2, unchanged) and a template census:
- Every ATEN-only cache has **zero `triton_tem_` definitions or launches**. GEMMs are `extern_kernels.bias_addmm`/`mm`, including the trunk backward graph of the bench's B (96 `mm`).
- Every ATEN autotune line logs `num_triton_choices: 0`.
- Pointwise and reduction fusions stay Triton (for example, 4 pointwise and 5 persistent-reduction kernels per trunk-forward wrapper).
- In the same run, the default caches launch the vulnerable templates (`…addmm_gelu_t_9`, A-load `512*idx_m`, `ks0: i32`; backward `triton_tem_fused_mm_*`, `ks0: i32`).
- Decompose-K and persistent-TMA text is absent everywhere, but the default caches had none either. Their exclusion under ATEN rests on source gating: `kernel/mm.py` adds them only inside `use_triton_template`, which requires `TRITON` in the backend list.

**Timing cost (medians, CUDA events, 20 iterations, same order and counts):**

| density | A fwd 256 | B train 1,024 | C teacher-proxy 16,384 | D value 256 | update wall default → ATEN | cost | ceiling SPS/rank |
|---|---|---|---|---|---|---|---|
| mid (303) | 12.90 → 15.23 ms | 177.8 → 186.2 ms | 986.4 → 1,004.7 ms | 9.37 → 11.84 ms | 4.667 → 4.970 s | **+6.5 %** | 1,755 → 1,648 |
| dense (709) | 25.39 → 29.11 ms | 326.7 → 332.7 ms | 1,687.4 → 1,731.7 ms | 21.97 → 25.75 ms | 8.561 → 8.944 s | **+4.5 %** | 957 → 916 |

- The default arm reproduces the earlier ceiling run (4.662 s and 8.564 s).
- Every p90 is within about 0.53 % of its median (largest 0.534 %).
- The cost is concentrated in the small no-grad calls (A +15–18 %, D +17–26 %). This is consistent with lost prologue/epilogue fusion (casts, bias+GELU, residual+LayerNorm), an inference from the generated code; there is no timeline. B and C lose 1.8–4.7 %.
- Overall allocated peak is approximately unchanged (dense B about 40.27 GiB in both arms; dense D drops 0.950 → 0.862 GiB). Cold compile is faster (mid B first call: 39.6 s → 26.0 s).

**Limits:**
- One stack; inputs up to 2³² elements (8,388,608 × 512).
- The real-trunk backward was not element-compared above the bound. That rests on the synthetic MLP backward plus kernel evidence.
- Non-GEMM Triton kernels rely on Inductor's own 32-bit-indexing guards (they recompiled here and were correct).
- bmm was not exercised.
- No nsys timeline, and one run per arm.
- The setting is process-global.
- The compiled path is still not reachable from `run_ppo` config at `e1458d2`.
- Nothing was wired into repo code.

## GPU checks bundle (component) (2026-09-29, 09:15–09:29Z, pod `w7ia3zvxqsvs3g`, GPUs 0 and 1)

**Diagnostics plus a component measurement at `8fde43c`** (`kg/rebuild-trainer-model`, tree `70d50fa3…`: full model with encoder, masked critic, grammar heads and the Phase 4 teacher KL). No repo code changed. Check 4 is **model-only**: engine, host copies, GAE, logging and all-reduce are excluded, so its SPS numbers are not end-to-end.

- Run statement `run-statements/gpu-checks-bundle.md`: committed with the scripts in `5f2ee2d` before launch. Amendment 1 was committed in `a4c75e0` before the relaunch.
- Evidence, identities, hashes and attempts: `gpu-checks-2026-09-29/README.md`. Every number below is in `gpu-checks-2026-09-29/summary.json` (`summarize.py`).
- Pod checkout moved `e1458d2` → `8fde43c` by git bundle and detached checkout. Porcelain was empty throughout and nothing was pushed. The Rust extension was not rebuilt, since no Rust, Cargo or lock change was involved (`rs.abi3.so` sha256 unchanged).
- Stack: torch 2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3, driver 595.91.07.
- Settings: preset config, fp32 params under BF16 autocast, TF32 on, trunk compiled through the registered `configure_model_compile` path (`max-autotune-no-cudagraphs`, dynamic), heads eager. Inputs were synthetic `make_obs` at mid (303 tokens/row), dense (709) and, for check 1, mixed (222–709) densities, with fresh weights.
- **Attempts.**
  - Attempt 1 stopped after 55 s. The only failure was `c2_aten_bwd`'s `blocks.0.attn.k.bias` gradient (relative error 0.81–1.16), and it failed equally at the warm point **below** the bound.
  - The key bias adds a per-query constant to the attention logits, so its true gradient is identically zero. An fp64 CPU check gave 3.9e-14 against 30.8 for the query bias, so a relative error there divides rounding residue by ~0.
  - Amendment 1 judged that one parameter against the query-bias scale instead. Attempt 2 then reran the whole bundle and every stage passed: driver 480 s, exit 0.
  - Aggregate driver wall was 535 s of the 60-min limit, about $0.6 at $4.18/h. The pod session was ~35 min, 08:55–09:30Z.
  - Idle gate: 0 MiB, 0 %, no processes on both GPUs before and after. The pod was left running and idle.
  - **Outer-timeout cleanup (post-run fix).** Both recorded attempts ran a driver that started each stage in its own session with no signal handler, so launch.sh's outer `timeout` could not have terminated running stages. Neither attempt came near that timeout, and attempt 1's ordinary stop terminated its running stage. The driver and launcher were revised after the run: SIGTERM/SIGINT/SIGHUP and exit handlers terminate every stage group, and `timeout -s TERM` signals the driver first. A local dummy-stage test passed (`gpu-checks-2026-09-29/post-run/driver_cleanup_test_local.txt`). The fix has not run on the pod.
  - **Stage-error and judge guards (post-run fix, Codex review `verify-merge-gpu-receipts-r1`).** The driver now stops with exit 5 and bounded cleanup when starting or judging a stage raises (it previously let Phase 2 start and returned 0), C3 checks student values on all four teacher paths (it checked two), and the C1 `> 2×` median rule also applies at median 0. None of these fired in the recorded attempts: the retained C3 records pass the full check and every retained C1 median is positive. Local test: `gpu-checks-2026-09-29/post-run/driver_guards_test_local.txt`. Not run on the pod.
  - **Launcher idle gate and C2 reference non-finite values (post-run fix, Codex review `verify-merge-gpu-receipts-r2`).** `launch.sh`'s idle gate now treats a failed `nvidia-smi` query as not idle (a failed compute-apps query beside 0 % utilization used to pass), and C2's comparators count non-finite values in the eager reference as well as the compiled side (a reference NaN used to read `nonfinite=0` and could hide a 10 % gradient error in its 512-row chunk). Neither fired in the recorded attempts: both prelaunch receipts show an answering compute-apps query with no process at 0 %, and every numeric field in the retained C2 records is finite. Local test: `gpu-checks-2026-09-29/post-run/driver_guards_test_local_r4.txt` (both new checks fail against the previous scripts: `driver_guards_test_prefix_scripts_r4.txt`). Not run on the pod.

**1. fp32 discriminating check (Phase 6.0 (b) open item): BF16 rounding is supported; no path exceeded the pre-declared aggregate threshold. A shared per-channel concentration of outliers (channel 229 at mid) is unexplained.**

Setup:
- Same weights and the same BF16-rounded input for every path.
- Reference: autocast off, TF32 off, padded MATH SDPA. The auto-backend fp32 reference differs from it by ≤ 4.5e-6.
- Metric: present-token elements outside `0.02 + 0.02|ref|`.

| density (tokens) | compiled-ATEN | compiled-default | eager flash | padded SDPA |
|---|---|---|---|---|
| mid (77,568 tok) max / mean / outside | 0.0431 / 0.00541 / 0.0059 % | 0.0452 / 0.00532 / 0.0046 % | 0.0642 / 0.00628 / 0.0267 % | 0.0670 / 0.00629 / 0.0264 % |
| dense (181,504) | 0.0459 / 0.00502 / 0.0018 % | 0.0480 / 0.00495 / 0.0013 % | 0.0682 / 0.00592 / 0.0131 % | 0.0685 / 0.00592 / 0.0130 % |
| mixed (118,916) | 0.0468 / 0.00518 / 0.0037 % | 0.0474 / 0.00510 / 0.0028 % | 0.0634 / 0.00607 / 0.0194 % | 0.0659 / 0.00607 / 0.0195 % |

- **Pre-declared rule.** A path is an outlier if its mean |Δ| or outside-tol fraction exceeds 2× the median of the three paths. The result is **similar** at every density under both backends; no path is flagged.
- **Compiled is the most accurate path**, not the least: 0.84–0.86× the median mean |Δ| and 0.10–0.22× the median outside-tol fraction. That fused kernels keep fp32 intermediates is a hypothesis; the generated code was not read for it.
- Eager flash and padded SDPA are indistinguishable (mean |Δ| equal to 3 significant figures).
- **Outlier location.**
  - Outliers sit at small |ref|, where the absolute 0.02 term dominates the tolerance. Example: mid eager has 4,684 of 5,305 outliers at |ref| < 0.25, 504 in [0.25, 0.5), 114 in [0.5, 1), 3 in [1, 2) and 0 at |ref| ≥ 2.
  - The largest errors are 0.043–0.069 at |ref| 2.5–3.4, which is **2.8–4.4 BF16 ulps** of the reference.
  - Across token groups, outliers follow the element share (for example, mid tile tokens hold 66 % of elements and 69 % of eager outliers).
  - **Across channels they do not** (`summary.json` `c1_channels`, computed from every retained outlier coordinate in `c1_*.outliers.json`; no path reached the 50,000 cap). Each of the 256 channels holds **0.391 %** of elements. At mid, **channel 229 holds 700 / 5,305 = 13.2 %** of eager outliers, 663 / 5,248 = 12.6 % of padded, and 176 / 1,171 = 15.0 % (ATEN) and 138 / 910 = 15.2 % (default) of compiled. It is the top channel for every mid path.
  - At mixed, channel 229 is again the top eager and padded channel (8.7 % and 8.1 %) and holds 10.4 % / 12.7 % of compiled outliers (ATEN / default). At dense it holds only 1.0–3.9 %. There the top channels are 236, 47 and 184 for eager and padded, and 135 and 189 for compiled, each at 8–17 %.
  - Outlier sets barely overlap between paths element by element (Jaccard 0.004–0.048). Low Jaccard alone does not establish independent noise, because the paths' outliers share channels.
  - **Unexplained, to recheck in Phase 6.** The largest errors are 2.8–4.4 BF16 ulps and the outliers are small-|ref| dominated, which supports BF16 rounding. The shared channel concentration, strongest at channel 229, is not explained by this check. Phase 6 should recheck it, for example with trained weights, before treating it as a property of fresh initialization.
- **Reproduction of Phase 6.0.** The Phase 6.0 pairwise comparisons re-measure at 0.009–0.021 % (compiled vs eager 0.009–0.015 %, eager vs padded 0.012–0.021 %). The earlier 0.017–0.021 % is therefore the tail of this common error distribution under a tolerance tighter than 8-layer BF16 accumulation near zero.
- Repeated compiled calls were bit-identical.

**2. Real-trunk backward above the bound, ATEN-only: correct at 4,194,305 and 4,198,400 packed tokens.**

Setup and deviation:
- Guard bypassed as in the forward probe, so each point was exactly one trunk call of all packed tokens.
- Deviation: **depth 1** (one real block plus final norm at preset width). Depth 8 needs more activation memory than one GPU has.
- Measured peak `max_memory_allocated` was 43.1 GiB compiled and 58.4 GiB eager at depth 1.

Compiled vs eager:

| point | output | dX | param grads (excl. k.bias) | k.bias |
|---|---|---|---|---|
| 4,194,305 | 0 wrong tokens, max \|Δ\| 0.034 | rel_max 0.0087, rel_fro 0.0045, max token rel-L2 0.0074, 0 non-finite, 0 at masked | rel_max ≤ 0.0113 | worst \|value\| or \|Δ\| is 0.0039 × \|q.bias grad\| |
| 4,198,400 | 0 wrong, max 0.032 | rel_max 0.0085, token rel-L2 ≤ 0.0076 | ≤ 0.0142 | 0.0035 × |

- The eager-vs-eager floor is dX 0.0044 and params ≤ 0.0040, from nondeterministic flash backward.
- The warm point (5,672 tokens) gives the same picture.
- **Kernel census:** the ATEN caches, including the backward graphs (26 `extern_kernels.mm`, 10 `bias_addmm`), contain **0** `triton_tem_` definitions or launches.
- **Default-backend control reproduced the failure:** an illegal memory access during default-backend autotuning at 4,194,305 (rc 1). It failed while compiling the first target point, before any gradient was compared. The control therefore discriminates the compile/forward template path, and this run does not show backward-specific corruption under default backends.

**3. Full-model GPU smoke: all four processes (mid/dense × ATEN/default) passed.**

- **Sampling and replay.**
  - 256-row sampling was replayed by `evaluate_actions` (grad enabled) at 256 rows and at 1,024 tiled rows. The model's replay validation accepted its own samples every time.
  - **The log-ratio the first-minibatch alarm reads** (signed mean over rows of Σ replay − Σ sampled event log-probs) had |mean| ≤ **1.4e-4 nats** against the 0.05 limit.
  - Mid ATEN +5.2e-6 / +5.4e-6 (256 / 1,024); dense ATEN −1.37e-4 / −1.28e-4; mid default −1.8e-5; dense default −7.3e-5.
  - Per-row |log-ratio| max was ~2.0e-3 at mid (0.001983642578125, mid default; 1.85e-3 mid ATEN) and 5.6e-3 at dense. Row joint log-probs are −165 (mid) and −908 (dense) nats.
  - Per-slot event max |Δ| was ≤ 4.1e-4.
- **Teacher KL (Phase 4).**
  - Self: per-row mean 1.4e-7–9.8e-7, max ≤ 8.3e-6, per-event min ≥ −4.0e-7.
  - Perturbed copy (+5 % std noise): per-row mean 7.1e-5 (mid) and 5.4e-4 (dense), all positive and finite.
  - Cached and combined paths are **bit-identical** (max |Δ| 0), both for self and perturbed.
- **Values and loss.** Values are finite in [−1, 1], and `compute_value` equals the sampled values exactly. The PPO-shaped loss backward at 1,024 rows with the teacher terms gives finite loss and 210/210 finite gradients (norm 3.52 mid, 7.73 dense).
  - As executed, the value term was **0.25·MSE**, not the declared 0.5·MSE: `c3_smoke.py` computes `v_loss = 0.5·mean((v − ret)²)` and then adds `0.5 · v_loss`. The recorded `v_loss` (0.165 mid, 0.171 dense) is that half-MSE. This does not affect the pass criteria (finiteness).
- `use_flash_attn` was true on all 14 calls per process.
- Peak allocated memory was 21.3 GiB (mid) and 40.6 GiB (dense).
- **Unplanned observation, unattributed.** Values from grad-enabled replay differ from no-grad sampling by up to **0.0154–0.0195** on the [−1, 1] scale, in every process. That is 8–10 % of the 0.2 value-clip range, while log-probs differ by ≤ 4e-4. Whether this comes from the compiled training vs inference graphs or from BF16 in general was not tested.

**4. Per-rank timings, ATEN-only (component only).**

Setup:
- Isaiah split: the global config is 256 envs × 64 steps = 16,384 env steps per update.
- Update wall = 64·t_A + t_C + 16·t_B + t_D, from medians of 20 CUDA-event iterations. Every p90 is within 1.4 % of its median.
- B is the cached-teacher PPO step with a Muon step. C is `compute_teacher_distillation_targets` on the rank's rollout.
- **Executed B loss:** clipped ratio + **0.25·MSE** value − 0.01·entropy + 0.005·teacher KL + 0.005·teacher value CE. The run statement declared 0.5·MSE, but `c4_timing.py` computes a half-MSE and multiplies it by 0.5 again. A scalar coefficient on one loss term does not change the kernels or shapes, so the component timings remain usable.

| ranks (envs/rank, spm) | mid t_A / t_B / t_C / t_D | mid wall → global SPS (per rank) | dense t_A / t_B / t_C / t_D | dense wall → global SPS (per rank) |
|---|---|---|---|---|
| 2 (128, 8): rows 256 / 1,024 / 16,384 / 256 | 15.26 / 186.6 / 1,012.3 / 11.89 ms | 4.987 s → **3,286** (1,643) | 29.17 / 336.5 / 1,741.2 / 25.82 ms | 9.018 s → **1,817** (908) |
| 4 (64, 4): 128 / 512 / 8,192 / 128 | 7.33 / 97.8 / 505.7 / 5.45 | 2.545 s → **6,437** (1,609) | 14.12 / 172.9 / 868.4 / 12.30 | 4.551 s → **3,600** (900) |
| 8 (32, 2): 64 / 256 / 4,096 / 64 | 4.97 / 51.3 / 246.5 / 2.93 | 1.389 s → **11,796** (1,475) | 7.05 / 93.1 / 426.5 / 5.96 | 2.372 s → **6,906** (863) |

- **Scaling.** Efficiency is global SPS divided by the ideal: **2×** the 2-rank rate at 4 ranks and **4×** at 8 ranks. It is 0.98 / 0.90 at 4 / 8 ranks for mid and 0.99 / 0.95 for dense.
  - The extra wall is `f × wall(ranks) − wall(2 ranks)` with f = 2 or 4, split by component as `f × n·t(ranks) − n·t(2 ranks)` (n = 64 for A, 16 for B, 1 for C and D; `summary.json` `c4_scaling`).
  - At 8 ranks, mid adds 0.569 s: **+0.295 s from A** (64 rollout forwards), **+0.300 s from B** (16 train steps), −0.026 s from C. Dense adds 0.472 s: **A improves by 0.063 s**, **B adds 0.572 s**, C −0.035 s.
  - At 4 ranks the loss is B in both densities (+0.144 s mid, +0.149 s dense), while A improves (−0.038 s, −0.060 s).
  - So the smaller train-step batch (B) is the common cost. The small rollout forwards (A) add as much again only at mid at 8 ranks. B is 59–63 % of the wall, A 18–23 % and C 18–20 %.
- **Comparison with the e1458d2 ATEN A/B at the 2-rank shape.**
  - A and D are within 0.5 %.
  - B (now cached-teacher) is +0.2 % mid and +1.1 % dense.
  - C (now real teacher targets) is +0.8 % mid and +0.5 % dense.
  - The wall is 4.970 → 4.987 s (mid) and 8.944 → 9.018 s (dense).
- **Trunk chunks** matched the guard's prediction on every call: C was 2 at mid 16,384 rows, and 2 / 3 at dense 8,192 / 16,384 rows.
- **Memory.** Peak `max_memory_allocated` was 40.70 GiB (dense 2-rank B). Peak reserved was 69.21 GiB (mid 2-rank C).
- The Phase 2 timings ran alone on GPU 0.

**Limits:**
- Scope: one stack, synthetic observations and grammar tables, fresh weights. The small head gain understates a trained policy's logit noise, so the log-ratio and KL margins **do not qualify the 0.05 alarm for trained policies**.
- Check 1 used 256-row batches only.
- Check 2 ran at depth 1 only. Its control failed at compile, so it says nothing backward-specific.
- The replay-vs-sampling value gap is unattributed here; the later "Value gap diagnostic" section below attributes it (H1). Check 1's shared channel concentration (channel 229 at mid) stays unattributed and is left for a Phase 6 recheck.
- c3 and c4 executed the value term at 0.25·MSE rather than the declared 0.5·MSE (see the run statement's post-run addendum).
- Check 4 excludes the engine, copies, all-reduce, GAE and logging. It is one run per shape, with no nsys (not installed) and no timeline.
- The backend setting is process-global and applied by a wrapper, not by repo code.
- No cookbook note was written. Promoting these findings is left to the owner's workflow.

## Value gap diagnostic (2026-09-29, 10:47–11:07Z, pod `w7ia3zvxqsvs3g`, GPUs 0 and 1)

**Diagnostic at `8fde43c`; no repo code changed; no timing.**
- **Question** (from "GPU checks bundle (component)" check 3 on `kg/rebuild-gpu-checks`): grad-enabled replay values differ from no-grad sampling by 0.0154–0.0195 while log-probs differ by ≤ 4e-4. Is that inherent to BF16 + compile (H1), or does it come from something Kaggriculture-specific in the grad path (H2)?
- Run statement `run-statements/value-gap-diagnostic.md`. Evidence, identities and attempts: `value-gap-2026-09-29/README.md`. Numbers: `value-gap-2026-09-29/summary.json`.
- Setup:
  - Preset Kaggriculture model and Isaiah's `stateless_transformer_6m` (`configs/scaling_6m.yaml`), both with fresh weights, `.train()` and fp32 params.
  - Kaggriculture used `make_obs` at mid density. Isaiah's model saw real Orbit Wars states from his Rust `VectorizedEnv` after 24 steps: 2–4 players per row, 652 and 2,562 present players at 256 and 1,024 rows.
  - S = `model(obs)` under no_grad; R = `evaluate_actions(obs, S.actions)` with grad enabled. Both paths use BF16 autocast (TF32 on) unless marked fp32.
  - The trunk was compiled through the registered path (`max-autotune-no-cudagraphs`, dynamic).
- **Attempts:**
  - Attempt 1 stopped on an Inductor compile error in the **fp32** compiled training graph. Amendment 1 disabled `coalesce_tiling_analysis` for fp32 compiled stages only.
  - Attempt 2 stopped on an fp32 eager OOM at 1,024 rows. Amendment 2 limited fp32 to 256 rows.
  - Attempt 3 passed. Its compiled Isaiah stages hit Dynamo's recompile limit because of my `use_flash_attn` counting wrapper, which his attention calls inside the compiled trunk, so their 1,024-row cells ran eagerly. Amendment 3 removed the wrapper and reran those two stages on the same states.
  - Aggregate driver wall 449 s. The pod was left running and idle.

**Mechanism observed:**
- Every compiled stage logged exactly one Dynamo recompile, `GLOBAL_STATE changed: grad_mode`. No-grad and grad-enabled calls therefore run **different compiled trunk graphs**. Isaiah's rerun processes each report Dynamo `unique_graphs: 2`.
- Within one grad mode everything was bit-identical:
  - `evaluate_actions` under no_grad = S exactly.
  - `compute_value` under no_grad = S exactly; with grad it reproduces R's gap (same max).
  - Repeated sample and replay: exactly equal.
  - Trunk input (stems, eager): identical across grad modes.

**Pre-declared predictions (all five support H1):**

| # | observation (max \|Δ\| unless noted) | H1 bound | result |
|---|---|---|---|
| 0 | A, BF16 compiled: value gap V | [0.005, 0.05] | ATEN **0.0195 / 0.0195** (256 / 1,024 rows); default **0.0156 / 0.0233**. Reproduced. Mean \|Δ\| 0.0041–0.0044 |
| 1 | B, eager BF16: V | ≤ 1e-3 and ≤ 0.1 × A | **0** at both row counts; hidden states bit-identical (exact-equal fraction 1.0) → H1 |
| 2 | C, fp32 (autocast off, TF32 off; padded SDPA), 256 rows: V | compiled ≤ 1e-3, eager ≤ 1e-4 | compiled **8.3e-7** (ATEN) / **1.2e-6** (default); eager **0** → H1 |
| 3 | D, actor `.out` gain 1.0: event log-prob gap L | ratio ≥ 10, ≥ 5e-3, eager ≤ 1e-3 | L 3.6e-4 → **0.032–0.037**, ratio **86–101×** (≈ the 100× gain ratio) under both backends; eager **0**; per-row joint log-ratio max 0.16–0.27 nats → H1 |
| 4 | E, compiled BF16: trunk hidden at critic tokens | (i)–(iv) of the statement | See the list below → H1 at both backends and row counts |
| 5 | F, Isaiah compiled (Amendment 3 rerun): head-swap value gap | ≥ 0.25 × Kaggriculture's; his eager ≤ 1e-3 | default **0.0145 / 0.0191**, ATEN **0.0117 / 0.0161**, vs Kaggriculture 0.0195 (0.60–0.98×); his eager **0** → H1 |

Prediction 4, case E in detail:
- Grad-vs-no-grad mean |Δ| is 0.0048 (ATEN) and 0.0051–0.0052 (default). The no-grad path's own error against an fp32 reference is 0.0054, and the grad path's is 0.0060, 1.1× that.
- Critic-token relative mean equals the all-token value (0.0061 vs 0.0061), and the plan and own-actor tokens match it. The noise is not critic-specific.
- Head swap: the same critic head on each path's hidden states reproduces the full gap (HS = V). With an fp32 head, HS is 0.0148–0.0168.
- Against an fp32-reference value, the no-grad and grad values each err by 0.013–0.015 max and 0.0032–0.0041 mean. **Neither path is "the wrong one"**: the gap is the difference of two BF16 errors of the same size.
- BF16 head rounding alone contributes 0.006–0.009 (fp32 vs BF16 head on the same hidden states).

- **Isaiah's log-probs are not suppressed like Kaggriculture's.** His compiled per-entity log-prob gap is 0.044–0.057 nats, and his per-player joint log-ratio max is 0.053–0.075 (means −9e-4 to +1e-4). Kaggriculture's is 3.6e-4, even though both actors' `.out` gains are 0.01. H1's "actor heads suppress the noise ~100×" therefore holds for Kaggriculture's grammar heads, not for Isaiah's `discrete_targets` actor. The cause, for example his ship-size mixture density, was not investigated. This does not bear on the value-gap attribution.
- **Attempt 3's invalid Isaiah cells.** Its 256-row compiled Isaiah cells, one fresh recompile per call, gave a similar value gap (0.0151–0.0156). They also showed nonzero same-mode differences (0.005–0.012) that vanished once recompiles were fixed.

**Attribution: H1 is supported; H2 is not.**
- The value gap comes from grad-mode-specific compiled graphs of the trunk rounding differently in BF16. It vanishes in eager BF16 and in fp32, compiled or not, and is the same size under ATEN-only and default GEMM backends.
- It is carried to the value by the gain-1.0 critic head. Raising the actor gain to 1.0 raises Kaggriculture's log-prob gap by the gain ratio.
- Isaiah's model on real Orbit Wars states shows the same value gap, 0.6–1.0× Kaggriculture's.
- No Kaggriculture-specific grad-path cause was found: critic tokens are not special, the trunk input is identical, and the head and masking are bit-identical within a grad mode.

**Limits:**
- One stack and fresh weights. A trained critic's logit scale can grow or shrink the gap, so these magnitudes do not qualify value-clip or bootstrap margins for trained models.
- Kaggriculture observations are synthetic and mid density only. Isaiah's states come from 24 steps of a fresh policy.
- fp32 cases ran at 256 rows only, on padded SDPA rather than packed flash, and the fp32 compiled cases had Inductor's `coalesce_tiling_analysis` off.
- D mutated the measured model in place.
- The generated code of the two graphs was not read. Which fusions differ is unmeasured; caches are retained on the pod.
- The fp32 compiled training-graph compile error (attempt 1) is a separate Inductor issue on a non-production path. It is not reproduced as a minimal case.
- The process-group cleanup was tested locally only; no signal fired on the pod.
- No cookbook note was written. Promotion is left to the owner's workflow.
