# References

## Current rebuild

- [[live-differential-parity-checks-the-rust-kernel|Live differential parity checks the Rust kernel]] — Task 1.1b generates traces from Kaggle's hash-pinned engine; 8 committed games and a 40-game sweep agree with Rust; 303 probes find two malformed-input divergence classes (Unicode digits, unhashable items), kept as expected failures.
- [[failed-training-reports-status-before-distributed-cleanup|Failed training status and traceback ordering]] — Task 0.3 forwards W&B failure codes and flushes rank-tagged tracebacks before distributed teardown; offline TDD and 722 Python passes, with live telemetry/distributed verification outside scope and Rust parity blocked by missing fixtures.

## Reference branch

Since the [[../decisions/restart-the-port-from-isaiahs-clean-base|restart]], the following References describe the implementation on branch `kg/reference-2026-09-29` (tag `kg-reference-2026-09-29`), not the current tree. Use them as reference for the rebuild.

- [[full-turn-intentions-coordinate-batched-action-heads|Full-turn intentions coordinate batched action heads]] — Optional U/A hypothesis; Isaiah and the current model already retain shared scratch/plan context. No demonstrated need or cost-benefit for a dedicated coordinator.
- [[native-game-semantics-use-v3-owned-buffers|Native game semantics use v3-owned buffers]] — Preserve native semantics with checked compact masks and a fused transactional lifecycle, backed by CPU parity and bounded two-GPU PPO measurements. Explicit-state replay encoding also passes evolved-state feature parity for offline BC.
- [[starter-history-and-knowledge-remain-retrievable|Starter history and knowledge remain retrievable]] — The Isaiah source history/license, redirected remotes and selective cookbook lifecycle are retained with explicit verification limits.
- [[shared-ppo-adapts-game-batches-without-a-second-loop|Shared PPO adapts game batches without a second loop]] — One shared Isaiah PPO trainer carries Kaggriculture batches; its core is upstream-equivalent, the recipe is partially aligned, and teacher, cadence, CUDA-fault and GPU-qualification gaps are tracked in ops. Historical SPS measurements link to their receipts.
- [[explicit-game-tokens-and-grammar-replace-orbit-heads|Explicit game tokens and grammar replace Orbit heads]] — Restored player token and entity/player/plan batched heads pass local checks; checkpoint schema changes explicitly, while prior GPU results retain their old-model scope.
- [[reward-reuse-preserves-objective-and-critic-semantics|Reward reuse preserves objective and critic semantics]] — Current GPU econ coefficient0.2, observed nonzero reward with no claimed learning benefit, and exact v2 W/L/D, bank-margin and capped-economic formulas with v3 critic/bootstrap compatibility limits.
- [[bc-bootstrap-uses-native-replay-features-and-current-heads|Native replay BC bootstrap]] — Market-disabled controls preserve3000 bank; exact admitted replay actions and Rust features support separate current-model BC, with episode holdout, two-rank smoke, observed overfitting and incomplete quality validation; automatic PPO disabled; BC stopped early for a separately requested PPO experiment, which then failed with a CUDA illegal memory access later traced to a compiler GEMM overflow; the planned architecture alignment schedules one BC rerun.
- [[compiled-gemm-template-overflows-above-2-21-rows|Compiled GEMM template overflows above 2^21 rows]] — Torch 2.9 Inductor max-autotune Triton GEMM templates wrap 32-bit offsets once rows × inner dim exceeds 2^31, silently corrupting and then faulting; this caused the reference PPO-from-BC crash and its log-ratio mismatch. Applies to the rebuilt tree too.
