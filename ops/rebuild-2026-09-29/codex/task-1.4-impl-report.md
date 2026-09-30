Implemented Task 1.4, including the **world-size-8 L3 case**. Changes remain unstaged. Qualification is **pending pod checks**.

The [seven-part report](/Users/poonszesen/kg-v3-env/ops/rebuild-2026-09-29/1.4/results.md) contains exact commands, A–G red/green receipts, mutations, deviations and open items. The [file inventory](/Users/poonszesen/kg-v3-env/ops/rebuild-2026-09-29/1.4/changed-files.tsv) lists every changed file and its purpose.

| Final check | Actual result |
|---|---|
| Root Rust | 274 passed, 5 ignored |
| Vendored engine | 69 passed |
| Requested five Python suites | 497 passed, 2 failed |
| Trim checker, maturin, `rs-prepare` | Passed |
| Documentation checks, `git diff --check` | Passed |
| `py-prepare`, repaired `prepare` | Memory watchdog stopped full pytest |

Both Python failures explicitly report the missing complete reference fixture.

The old seed formula failed **both world-size-2 and world-size-8 tests**. Whole-batch truncate publication and premature live writes also failed their intended tests. All performed mutations were restored.

The cast audit identified **two reachable-unbounded casts**, now guarded by executed-HIRE admission checks. Vendored bytes and Task 1.5 implementation paths remain unchanged.

Pending items:

- Full oracle: **0 games/0 transitions compared**; recording exceeded memory during compilation. Fixture hashes, coverage and trajectory mutation remain pending.
- Release overflow proof and optimized phase costs: builds exceeded memory; [timing receipt](/Users/poonszesen/kg-v3-env/ops/rebuild-2026-09-29/1.4/timing.json) contains exact pod commands.
- Broad Python preparation on the pod.
- Claude’s approved constructor wording in the contract, plus reconciliation of the sibling seed-band documentation.