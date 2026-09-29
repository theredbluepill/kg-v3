Two new findings remain:

- **P2 — [launch.sh:15](/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/launch.sh:15):** A failed compute-process query can pass the idle gate when utilization reads zero. Require successful query exit statuses before accepting idleness.
- **P3 — [c2_trunk_bwd.py:87](/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/gpu-checks-2026-09-29/scripts/c2_trunk_bwd.py:87):** An eager-reference NaN can pass the gradient oracle and hide another error in its chunk. Validate finiteness of both operands and add a regression spanning `ROW_CHUNK`.

Both were reproduced; targeted scratch fixes reject them. Neither appears to have affected the retained runs.

| Prior r1 finding | Status |
|---|---|
| 1. Worker exceptions | **RESOLVED** |
| 2. C3 teacher-value omissions | **RESOLVED** |
| 3. C1 zero median | **RESOLVED** |
| 4. Pending-merge provenance | **RESOLVED** |
| 5. Missing GPU evidence in notes | **RESOLVED** |

No merge-induced content or test loss found. Receipt hashes, summary regeneration and cookbook reconciliation check out.

| Requested check | Result |
|---|---|
| Engine Cargo, locked/offline | 69 passed |
| Root Cargo | 254 passed, 4 ignored |
| Engine trim | OK |
| Pytest suites | 1,690 passed, 11 skipped |
| Mypy | 63 files clean |

Mutation checks covered every new oracle/guard family. Scratch files were restored byte-for-byte; tracked files and index remain unchanged. No GPU run launched.

[Full report and evidence](/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/codex/verify-merge-gpu-receipts-r2/review.md)

VERDICT: REJECT