No functional merge regression or lost parent tests found. One documentation finding remains:

- **P3 — [teacher Reference:150](/Users/poonszesen/kg-v3-m-teacher/cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:150)**, also lines 130/141: obsolete statements still say configs/value guards await merging. **Fix:** reconcile them in place with the remaining trainer/native-env seams and unfinished Task 4.4.

| Check | Result |
|---|---|
| Engine Rust | 69 passed |
| Root Rust | 254 passed, 4 ignored |
| Engine trim | Passed |
| Requested pytest suite | 1,673 passed, 11 skipped |
| Mypy | Passed, 63 files |

Ran **68 scratch mutation attempts**: 66 failed as expected; two scoped survivors are explained in the [full report](/Users/poonszesen/kg-v3-m-teacher/ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1/report.md). All mutations were restored byte-for-byte. All **1,668 tracked files remain unchanged**.

The four deferred teacher integration tests independently reproduce the documented blockers. This approves the staging merge’s implemented scope, not completed Phase 4 training integration.

VERDICT: APPROVE WITH EDITS