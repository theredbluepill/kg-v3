Both r1 P3 findings are **RESOLVED**: native-environment blocker prose is current, and reward-schema/startup documentation matches implementation.

[Full verification report, including evidence and restoration hashes](/Users/poonszesen/kg-v3-m-env-adapter-r2/ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r2/report.md).

- All 1,730 intervening BASE paths accounted for: 1,728 preserved exactly; only intended index/log merges differ.
- All 21 References targets retained exactly once; phase grouping and log ordering preserved.
- Resolution reuse confirmed. Trim manifest and phase tracker unchanged. No conflict markers.
- 8-rank config satisfies the reward schema; startup preserves workload → compile-stack → Task 3.1 stop.

| Collected tests | BASE | Adapter | HEAD |
|---|---:|---:|---:|
| Python | 1,701 | 2,253 | 2,329 |
| Engine Rust | 69 | 69 | 69 |
| Root Rust, including ignored | 258 | 279 | 279 |

No unexplained omissions. The same two deliberate BASE test replacements documented in r1 remain.

Checks passed:

- Engine: **69 passed**.
- Root Rust: **274 passed, five ignored**.
- Python: **2,319 passed, 10 skipped**, across two disjoint shards. The initial combined run hit the unchanged oracle deadline; fresh-process sharding resolved all four failures.
- Trim checker, mypy (**69 files**) and documentation freshness passed.
- All **six scratch mutations** failed their targeted tests; baseline and restored runs passed with identical hashes.

All **5,972 tracked files** retain their initial hashes. Final `git status --porcelain` shows only the permitted untracked receipt directory. No training or GPU run occurred.

Findings: **no new P1/P2/P3 findings; no fixes required**.

VERDICT: APPROVE