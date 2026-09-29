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
