- **P2 — Failed evaluations lose replay custody:** [native_evaluation.py:141](/Users/poonszesen/kg-v3-t73/python/owl/kaggriculture/native_evaluation.py:141) lets decoder/native-step errors escape without persisting active selected games. Two live probes produced zero error records after one committed turn. **Fix:** save all active selected tapes as error custody, then re-raise.
- **P2 — Canonical evaluation integration remains incomplete:** [run_ppo.py:1606](/Users/poonszesen/kg-v3-t73/scripts/run_ppo.py:1606) still rejects Kaggriculture evaluation. **Fix:** complete Task 1.5 integration and verify eight exports through the canonical evaluation path.

Every r1 finding is classified:

| Finding | Status |
|---|---|
| Eight evaluation exports | **PARTIAL** — native eight-of-ten acceptance passes; trainer connection remains |
| DONE tape with `complete=false` | **RESOLVED** |
| Byte-check regression coverage | **RESOLVED** |

Checks passed: **69 engine tests; 289 root tests, five ignored; 655 relevant pytest tests, no skips; mypy over 67 files; trim validation.**

**21/21 scratch mutations detected**, including removal of both repaired guards. Scratch files restored byte-for-byte; **all 3,272 tracked files unchanged**.

[Full report and receipts](/Users/poonszesen/kg-v3-t73/ops/rebuild-2026-09-29/7.3/independent-verifier-r2/review.md)

VERDICT: REJECT