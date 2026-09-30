Two P2 findings require changes:

- [bc.py:400](/Users/poonszesen/kg-v3-bc/python/owl/train/bc.py:400): Nonzero `min_delta` can discard the lowest-NLL checkpoint. With `0.05`, NLL `2.0 → 1.97` retains `2.0`. Separate checkpoint selection from patience tolerance.
- [train_bc.py:118](/Users/poonszesen/kg-v3-bc/scripts/train_bc.py:118): Resume reuses the original source identity without checking the current checkout, potentially misattributing new checkpoints. Changed seed/batch settings also pass compatibility checks.

Checks: **1,691 passed, 11 skipped**; mypy: **no issues in 66 files**.

Mutations: **100 unique—27 killed, 73 survived**, all restored byte-for-byte. Significant coverage gaps: all 18 BC tests still pass when the real optimizer step is removed or the loss signs are reversed.

Shared Isaiah mechanisms, deterministic partitioning, documented critic handling and PPO checkpoint loaders otherwise check out within CPU scope. No production training launched; all **1,798 tracked-file hashes remain unchanged**.

[Full report and evidence](/Users/poonszesen/kg-v3-bc/ops/rebuild-2026-09-29/codex/verify-5.2-independent/report.md)

VERDICT: REJECT