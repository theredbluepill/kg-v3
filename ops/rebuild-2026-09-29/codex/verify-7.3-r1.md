- **P2 — Eight evaluation exports remain unimplemented:** [test_replay_export_integration.py:18](/Users/poonszesen/kg-v3-t73/tests/kaggriculture/test_replay_export_integration.py:18) unconditionally skips acceptance. This is a documented deferral, but the requested requirement remains unmet. **Fix:** wire the recorder into native evaluation and verify eight completed episodes.
- **P3 — Export can fail its own round trip:** [replay_export.rs:325](/Users/poonszesen/kg-v3-t73/src/kaggriculture/replay_export.rs:325) accepts `complete=false` when the tape reaches DONE; verification rejects the exported bytes. **Fix:** reject inconsistent completion claims.
- **P3 — Byte-check regression gap:** [test_replay_export_oracles.py:355](/Users/poonszesen/kg-v3-t73/tests/kaggriculture/test_replay_export_oracles.py:355) now fails semantic comparison before exercising byte comparison. **Fix:** add the same-kind mutation `3000.0` → `3.0e3`.

Checks passed: **69 engine tests; 269 root tests, four ignored; 121 pytest tests, five integration skips; mypy over 64 files; trim validation.** Initial `uv run` commands hit unavailable DNS; successful reruns used `--no-sync` after a fresh offline extension build.

**12/12 input mutations rejected.** Two scratch source mutations confirmed comparator sensitivity. Scratch sources were restored byte-for-byte; all **2,006 tracked files remain unchanged**.

[Full report and receipts](/Users/poonszesen/kg-v3-t73/ops/rebuild-2026-09-29/7.3/independent-verifier/review.md).

VERDICT: REJECT