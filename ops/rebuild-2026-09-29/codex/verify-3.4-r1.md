Task 3.4 has one blocking gap:

- **P1 — Startup assertion is disconnected.** [kaggriculture_workload.py:120](/Users/poonszesen/kg-v3-configs/python/owl/model/kaggriculture_workload.py:120) has no production caller. All three configs fail `FullConfig` loading; integration tests are skipped. **Fix:** wire validation and logging into canonical startup and add an unskipped rejection test.
- **P2 — Chunk-count claims are inaccurate.** [kaggriculture_workload.py:126](/Users/poonszesen/kg-v3-configs/python/owl/model/kaggriculture_workload.py:126) claims actual calls; packed batches can use fewer. The cookbook also reverses the headroom bound. **Fix:** describe upper-bound trunk calls and lower-bound headroom under full padded sizing.
- **P3 — Schema coverage is overstated.** [cookbook/log.md:5](/Users/poonszesen/kg-v3-configs/cookbook/log.md:5) claims every section is schema-validated. Env/reward sections receive only structural/value assertions. **Fix:** narrow that claim.

Confirmed: ranked recipe values match Isaiah after per-rank division; exact global workload is **256 environments, 16 optimizer steps, 16 segments/step, 16,384 transitions/iteration**. Helpers use model guard constants and fail loudly. No shim wrappers found.

Checks:

- Requested pytest: **1,324 passed, 10 skipped**; Orbit tests remain green.
- Requested mypy: **58 files, no issues**.
- Mutation produced the expected failure; restored byte-for-byte, then **19 passed, 3 skipped**.
- No tracked modifications remain.

[Full report and evidence](/Users/poonszesen/kg-v3-configs/ops/rebuild-2026-09-29/codex/verify-3.4/report.md).

VERDICT: REJECT