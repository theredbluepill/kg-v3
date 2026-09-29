Reviewed `kg/rebuild-env-briefs` at `4cb7305`. No architectural blocker; four localized edits remain:

1. **P2 — Specify truncate’s separate publication path.** [1.4.md:543](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.4.md:543) prescribes whole-batch `ObsStaging::publish` followed by transition copies, conflicting with selected-row-only truncation and untouched transitions. Task 1.3’s `publish` has no selection parameter. Explicitly define an infallible selected-row commit branch; test that unselected observation bytes and all six transition tensors remain unchanged.

2. **P2 — Test explicit reset/truncate rollback.** [1.4.md:701](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.4.md:701) covers step failures, including autoreset, but does not explicitly cover these separate publication paths. Add late selected-env construction/preparation failures and seed exhaustion, checking buffers, games, seeds and terminal records. Adapter spies cannot establish native atomicity.

3. **P2 — Unify reward-validation predicates.** [1.5.md:152](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.5.md:152) accepts positive raw event weights; [1.4.md:559](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.4.md:559) requires positive weighted products. With `W=1e-300` and both weights `1e-300`, Python accepts while native rejects through underflow. Use one exact predicate and paired tests. Also describe per-component inert-shaping rejection as an intentional strengthening: the pinned reference does not contain those exact checks.

4. **P3 — Scope the `.numpy()` prohibition to outputs.** [1.5.md:195](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.5.md:195) forbids per-step `.numpy()`, but fresh Torch actions/masks must reach NumPy-native arguments. Explicitly allow zero-copy input views per call while retaining output views once.

Otherwise, the briefs align with Isaiah’s lifecycle, contract v4 and the 1.2/1.3 interfaces. L3 seed partitioning, terminal timing and reward rounding are correct. R1’s fence preserves in-place truncation; the DMA mutation test is appropriately stronger than the ordering spy. L6 remains correctly distinguished from the historical GEMM overflow. R3 is implementable, pending its release tests and cast audit.

Read-only checks confirmed identical shared ABI sections and reproduced the numeric mismatch. No files changed; no builds or GPU tests ran.

VERDICT: APPROVE WITH EDITS