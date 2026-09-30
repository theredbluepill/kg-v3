# Independent numerical/model mutation audit, merge-teacher r2

Reviewed current `2390c8e239a4d57770bdafd621c05abbcb326739`. Diagnostic: determine whether current teacher numerical/model oracles fail under controlled violations, whether inherited GEMM admission still covers every teacher encode path, and whether the four documented integration blockers remain the actual first failures. Inputs are fresh copies of current source/tests/configs, the repository's teacher suite, and two explicit scratch-only probe files. Completion required controlled outcomes and byte-exact restoration; CPU only, one pytest process at a time, OMP/MKL/OpenBLAS/VECLIB/NUMEXPR thread counts set to 1. No production source or tracked file was edited.

## Results

- **37 attempts:** 33 numerical/model attempts (32 killed, one scoped survivor), plus four deferred integration-oracle attempts (all baseline-blocked, zero kills claimed).
- Initial baseline: **52 passed, 4 skipped** (48 repository teacher cases plus four inherited-GEMM probes).
- Direct FP64 teacher-probability reference baseline: **1 passed**.
- Final restored union: **53 passed, 4 skipped in 5.69 s** (`final-restored-union.log`).
- Every mutation file was restored in `finally`; all **199 copied inputs** remain byte-identical, including **195 tracked source/test/config inputs**. Those tracked inputs also match the working-tree originals. See `final-full-copy-restoration.json` and `summary.json`.

The only behavioral survivor is teacher-only FP32 demotion against the student finite-difference oracle. That oracle differentiates the student and therefore still holds for a fixed rounded teacher. Two-sided demotion fails the finite-difference oracle, and teacher-only demotion fails a supplemental direct FP64 probability reference (`fp64-teacher-demoted-reference.log`). The reviewed code preserves FP64 (`python/owl/model/actor/common.py:123`). This is the same scoped oracle limitation documented in r1, not a precision defect.

No new numerical/model defect was found. Admission/type/layout/replay/obs-spec guard families are handled by the separate guard audit.

## Oracle inventory

| Oracle family | Concrete attempts |
| --- | --- |
| T1 KL identity and off-support liveness | `kl-liveness-removed` |
| T2 replay density and finite cached mask, FP32/FP64 | `cached-mask-negative-infinity` |
| Ordinary forward/replay leave teacher fields unset | `plain-replay-exposes-teacher-logits` |
| T3 independent admissible-value KL, including HIRE capacity | `student-logits-unmasked` |
| T4 head and trunk chunk equivalence | `head-chunk-teacher-slice-reuses-first`, `trunk-chunk-reuses-first` |
| T5 fullgraph captured KL/gradient equivalence | `capture-kl-diverges` |
| T6 student gradients and FP64 finite differences | `student-kl-detached`, `fp64-demoted`, `fp64-both-demoted`; direct FP64 probability supplement `fp64-teacher-demoted-reference` |
| T7 target lead layout/dtype/optional fields/no-grad | `precompute-no-grad-removed` |
| T9 target index/concat inversion | `cache-indices-reversed`; invalid concat guards are in guard audit |
| T9b digest and hire-limit identity | `grammar-digest-ignores-values`, `grammar-signature-ignores-hire-limit` |
| T10/T10b exact cache byte accounting and pinned totals | `cache-nbytes-drops-winner`, `orbit-nbytes-drops-continuation`, `cache-row-estimate-drops-winner` |
| T11 chunked target precompute equivalence | `cache-precompute-batch-dependent` |
| T12 cached versus combined output equality | `cached-winner-diverges-from-combined` |
| T13/T14 copied-teacher entropy and live-seat mean CE | `ce-sums-live-seats`, `ce-includes-inactive-seats`, `ce-teacher-gradient-leak` |
| T15b stateless PPO dispatch | `ppo-combined-passes-dones`, `ppo-cached-passes-dones` |
| T16 unchanged Orbit CE | `orbit-value-ce-reduction-changed` |
| T17 seat isolation and no between-call state | `targets-mix-private-seats`, `teacher-targets-carry-between-call-state` |
| Cache capabilities and checkpoint grammar exclusion | `checkpoint-persists-grammar`, `teacher-cache-support-disabled`, `value-cache-support-disabled` |
| Last-best refresh copies weights and retains grammar | `refresh-omits-weight-copy` |
| Inherited GEMM guard reaches precompute/cached/combined student/combined teacher | `integration-gemm-guard-removed` kills all four scratch path cases |
| T18 trainer precompute/metrics | `blocked-trainer-cache-metric-zero`: baseline-blocked |
| Trainer checkpoint key-set/cache exclusion | `blocked-checkpoint-includes-teacher-cache`: baseline-blocked |
| T19b resume teacher identity | `blocked-resume-activates-student`: baseline-blocked |
| T19b fresh-launch teacher activation | `blocked-fresh-launch-disables-teacher`: baseline-blocked |

## Deferred integration tests

Removing only the four skip decorators produces four failures (`unskipped-integration-probe.log`): T18 and trainer-checkpoint fail at `python/owl/train/ppo.py:2140`, `unsupported type KaggricultureActionMask`; resume and fresh launch pass config/workload validation then fail at `python/owl/train/config.py:57`, `run_ppo cannot run Kaggriculture yet`. Each meaningful downstream mutant preserves exactly its corresponding first failure. These are **BLOCKED**, not killed or qualified. The current Reference's lines 131, 142 and 151–155 describe these dependencies consistently. No production seam was bypassed. After each attempt both source and skip decorators were restored byte-for-byte.

## Exact attempt outcomes

| Mutation | Outcome | Selected-test summary |
| --- | --- | --- |
| `kl-liveness-removed` | KILLED | 2 failed, 2 passed in 0.59s |
| `student-logits-unmasked` | KILLED | 2 failed in 0.74s |
| `cached-mask-negative-infinity` | KILLED | 8 failed in 1.00s |
| `fp64-demoted` | SURVIVED (scoped above) | 1 passed in 0.15s |
| `student-kl-detached` | KILLED | 1 failed in 0.14s |
| `capture-kl-diverges` | KILLED | 1 failed in 1.88s |
| `cache-indices-reversed` | KILLED | 6 failed, 3 passed in 0.87s |
| `grammar-digest-ignores-values` | KILLED | 1 failed in 0.13s |
| `grammar-signature-ignores-hire-limit` | KILLED | 1 failed in 0.11s |
| `cache-nbytes-drops-winner` | KILLED | 1 failed in 0.18s |
| `orbit-nbytes-drops-continuation` | KILLED | 1 failed in 0.20s |
| `cache-row-estimate-drops-winner` | KILLED | 1 failed in 0.11s |
| `head-chunk-teacher-slice-reuses-first` | KILLED | 1 failed in 0.19s |
| `cache-precompute-batch-dependent` | KILLED | 1 failed in 0.22s |
| `cached-winner-diverges-from-combined` | KILLED | 1 failed in 0.25s |
| `ce-sums-live-seats` | KILLED | 2 failed in 0.25s |
| `ce-includes-inactive-seats` | KILLED | 1 failed in 0.11s |
| `ce-teacher-gradient-leak` | KILLED | 1 failed in 0.12s |
| `precompute-no-grad-removed` | KILLED | 1 failed in 0.17s |
| `targets-mix-private-seats` | KILLED | 1 failed in 0.19s |
| `checkpoint-persists-grammar` | KILLED | 1 failed in 0.11s |
| `teacher-cache-support-disabled` | KILLED | 1 failed in 0.11s |
| `refresh-omits-weight-copy` | KILLED | 1 failed in 0.14s |
| `ppo-combined-passes-dones` | KILLED | 1 failed in 0.23s |
| `ppo-cached-passes-dones` | KILLED | 1 failed in 0.19s |
| `integration-gemm-guard-removed` | KILLED | 4 failed in 0.08s |
| `fp64-both-demoted` | KILLED | 1 failed in 0.15s |
| `fp64-teacher-demoted-reference` | KILLED | 1 failed in 0.01s |
| `plain-replay-exposes-teacher-logits` | KILLED | 1 failed in 0.11s |
| `value-cache-support-disabled` | KILLED | 1 failed in 0.10s |
| `teacher-targets-carry-between-call-state` | KILLED | 1 failed in 0.19s |
| `orbit-value-ce-reduction-changed` | KILLED | 1 failed in 0.20s |
| `trunk-chunk-reuses-first` | KILLED | 1 failed in 0.19s |
| `blocked-trainer-cache-metric-zero` | BLOCKED | 1 failed in 0.15s |
| `blocked-checkpoint-includes-teacher-cache` | BLOCKED | 1 failed in 0.14s |
| `blocked-resume-activates-student` | BLOCKED | 1 failed in 0.13s |
| `blocked-fresh-launch-disables-teacher` | BLOCKED | 1 failed in 0.13s |

Every killed attempt exited with ordinary pytest failure status 1; logs contain no collection/import/missing-test setup failures. Exact replacement strings and commands are preserved in `numerical_mutations.py`, `supplemental_mutations.py`, `trunk_mutation.py` and `blocked_oracle_mutations.py`. All receipts are newly generated under r2; r1 evidence was read as an audit recipe and was not overwritten. Scope does not include CUDA execution, BF16, distributed training, complete integrated Kaggriculture training or Phase 4.4.
