Verified `f15a413` against `b8747b6...HEAD`.

1. **P2 — Required reference replay coverage is missing.** [lifecycle.rs:141](/Users/poonszesen/kg-v3-t71/opponents_rs/tests/lifecycle.rs:141) replays native-generated actions and stops at the reconstruction boundary. It never compares resumed decisions against original Python behavior, as required by [brief §6](/Users/poonszesen/kg-v3-t71/ops/rebuild-2026-09-29/briefs/7.1-opponents.md:181). **Fix:** reconstruct original-Python and native controllers from matching prefixes in both seats, compare subsequent actions against frozen reference expectations, and mutation-test the comparison.

2. **P3 — Stale test count.** The [cookbook Reference:115](/Users/poonszesen/kg-v3-t71/cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md:115) reports 19 opponent tests. **Fix:** update it to **20**.

| Check | Result |
|---|---|
| Engine Cargo tests, locked/offline | 69 passed |
| Root Cargo tests | 254 passed, 4 ignored |
| Opponent Cargo tests | 20 passed |
| Relevant pytest | 178 passed, 10 skipped, 12 subtests passed |
| Mypy | Passed, 63 files |
| Engine trim checker | Passed |
| Opponent custody, original sources re-read | Passed |
| Opponent Clippy | Passed |

All eight oracles and their manifest regenerated **byte-for-byte**. Native parity matched **11,504 actions / 5,752 transitions**.

Scratch checks detected **8 oracle mutations and 4 controller mutations**. All scratch inputs were restored byte-for-byte; the restored suite passed. All **1,736 tracked files remain unchanged**, and Git status is clean.

No production behavior defect was found; rejection concerns the explicit qualification gap. [Full report and evidence](/Users/poonszesen/kg-v3-t71/.codex-tmp/verify-7-1/report.md).

**VERDICT: REJECT**