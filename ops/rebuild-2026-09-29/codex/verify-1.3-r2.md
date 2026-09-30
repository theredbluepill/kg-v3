Verified the three-dot diff at `8247ee2`. No encoder semantic defect found.

**P3 — stale diagnostic:** [regenerate.py:628](/Users/poonszesen/kg-v3-observe/scripts/kaggriculture_observation_oracle/regenerate.py:628) attributes a missing corpus to the resolved R1 quota blocker. Fix: report the missing manifest and regeneration command, add a regression, and reconcile the related cookbook claim.

| Check | Result |
|---|---|
| Engine Rust | 69 passed |
| Root Rust, including Orbit | 254 passed, 4 ignored |
| Kaggriculture Python | 311 passed, 3 skipped |
| Orbit Python | 787 passed, 3 skipped |
| Custody/trim Python | 124 passed |
| Trim checker and restored `rs-prepare` | Passed |

Python coverage completed in bounded batches after the combined run exceeded 1 GB.

The 512-state oracle passes. Swapping production market channels failed at offset 889; restoration was byte-for-byte and subsequent checks passed. Vendored bytes remain unchanged.

Optimized timing and CUDA paths remain unqualified. All 1,500 tracked files match their starting bytes.

[Full verification report](/Users/poonszesen/kg-v3-observe/ops/rebuild-2026-09-29/1.3/independent-8247ee2/review.md)

VERDICT: APPROVE WITH EDITS