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

Run statement: `run-statements/pod-flash-attn-setup.md`. Receipts: `flash-attn-setup-2026-09-29/` (README has identities and hashes). Wall ≈ 11 min of the 90-min budget (≈ $0.8 at $4.18/h); no new billable resource; pod left running and idle.

**Outcome: the blocker is cleared for the forward path.** A separate v3 environment, `/workspace/kg-v3-rebuild` (`kg/isaiah-gap-closure` @ `69397da`, via git bundle; `.venv` from `uv sync --frozen --group dev --extra flash-attn`), has torch 2.9.0+cu128, triton 3.5.0 and flash-attn 2.8.3, and the model's forced packed FlashAttention path runs on sm_120.
- **Install:** the extra's sdist build (`FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`, no build isolation) downloaded the prebuilt release wheel `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` in 7.5 s. No compile, so no `MAX_JOBS`/arch settings were needed. Wheel sha256 `4e2f9e39…0810` equals the GitHub release digest. The installed `flash_attn_2_cuda` `.so` (sha256 `8ca052bf…5807`) is byte-identical to the wheel's.
- **sm_120 support:** flash-attn 2.8.3's `setup.py` includes `120` in its default arch list (CUDA ≥ 12.8). The installed `.so` carries 72 sm_120 cubins, alongside sm_80/90/100 (`cuobjdump --list-elf`). Not a blocker.
- **Rust extension:** `maturin develop` (justfile `build`, dev profile) → `owl.rs` sha256 `35239d1b…93b8`. `owl` and `owl.model.kaggriculture` import from the new checkout, and `flash_attn_available()` is True.

**(a) Kernel vs SDPA.** `flash_attn_varlen_func` ran on BF16 packed q/k/v with 8 heads × head_dim 32, 256 sequences of 214–709 tokens, 173,518 tokens in total. Against per-sequence fp32 SDPA:
- max |Δ| is 0.00359 and mean |Δ| is 1.07e-4. BF16 SDPA vs fp32 SDPA gives the same max |Δ| (0.00359, mean 1.06e-4), so the flash error equals BF16 output rounding.
- The flash output differs from BF16 SDPA by at most 0.0039 (one BF16 ulp at |ref| ≈ 1.38).
- No element falls outside `0.02 + 0.02|ref|`. The profiler records `flash::flash_fwd_kernel<…bfloat16…>`.

**(b) Model trunk.** Setup: preset `configs/model/kaggriculture.yaml` (width 256, depth 8, 8 heads, `force_flash_attn: true`), fp32 params, `autocast(bfloat16)`, TF32 via `configure_torch()`. Batch: `make_obs` with 128 envs = 256 rows, 173,108 present tokens, lengths 221–709, padded length 709.

| Comparison (present tokens, 44.3M elements) | max \|Δ\| | mean \|Δ\| | outside tol |
|---|---|---|---|
| compiled flash vs eager flash | 0.0872 | 0.00542 | 0.018 % |
| eager flash vs eager padded SDPA | 0.0805 | 0.00498 | 0.021 % |
| compiled flash vs eager padded SDPA | 0.0775 | 0.00546 | 0.019 % |
| compiled vs eager, second shape (64 rows) | 0.0872 | 0.00542 | 0.017 % |

Output magnitudes reach about 4.0, where one BF16 ulp is 0.03. All three paths differ from one another by the same amount, which fits BF16 accumulation over 8 layers rather than a path-specific error. All outputs are finite, masked positions are exactly zero, and repeated compiled calls agree exactly (Δ = 0).

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
  - `/workspace/kg-v3`, its `.venv`, and `/workspace/gemm-limits-src-1ddc71d` were not modified.
