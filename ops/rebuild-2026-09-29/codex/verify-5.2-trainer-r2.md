Both r1 P2 findings are **RESOLVED**. Loss-sign, normalization and optimizer-step test gaps are **RESOLVED**; other coverage gaps remain **UNRESOLVED**.

One new **P3**: [exact-restart claims](/Users/poonszesen/kg-v3-bc/README.md:479) need qualification. An off-interval stop adds a patience observation; the probe’s resumed run stopped at step 2 versus uninterrupted step 4.

- Pytest: **1,698 passed, 11 skipped**
- Mypy: **66 files clean**
- Mutations: **120 tried; 47 killed, 73 survived**, restored byte-for-byte
- All **1,930 tracked files unchanged**; no production training launched

[Full dispositions and evidence](/Users/poonszesen/kg-v3-bc/ops/rebuild-2026-09-29/codex/verify-5.2-r2-independent/report.md)

VERDICT: APPROVE WITH EDITS