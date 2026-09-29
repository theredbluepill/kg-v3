Verified `49a4835`. No production merge defect found. [Full report and evidence](/Users/poonszesen/kg-v3-m-env-adapter/ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1/report.md).

All five claimed resolutions preserve the intended contributions. The 8-rank config satisfies the required reward schema; startup retains workload → compile-stack → Task 3.1 stop before allocation. The trim manifest is identical across both parents and HEAD.

| Collected tests | BASE | Adapter | HEAD |
|---|---:|---:|---:|
| Python | 1,701 | 2,253 | 2,329 |
| Engine Rust | 69 | 69 | 69 |
| Root Rust, including ignored | 258 | 279 | 279 |

Every Rust name and adapter Python name survives. Two BASE names were intentionally replaced on the adapter branch:

- `test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native` → `test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture`
- `test_grammar_bridge_is_retired_to_root_integration` → `test_no_authored_grammar_path_include_after_root_engine_edge`

Their relevant assertions survive or are strengthened; no unexplained coverage loss was found.

Requested checks pass:

- Engine: **69 passed**
- Root: **274 passed, 5 ignored**
- Python: **2,319 passed, 10 skipped**, including a repeat against the rebuilt native extension
- Mypy: **69 files clean**
- Engine trim and doc freshness: **passed**

All six scratch mutations failed their targeted tests; restoration returned them to green with identical hashes. Full SHA-256 values are in the report. No tests cover the documentation sentence or skip-reason wording, so no text-only mutation coverage is claimed.

Final `git status --porcelain` shows no tracked changes:

```text
?? ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1/
```

Findings:

- **P3 — Stale native-env blockers:** [teacher Reference:160](/Users/poonszesen/kg-v3-m-env-adapter/cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:160), its description/index, the model Reference and config headers retain parent claims that Task 1.4 is pending. Update these to the remaining Task 3.1 mapping/storage work; preserve historical results.
- **P3 — Obsolete schema/startup description:** [configs Reference:28](/Users/poonszesen/kg-v3-m-env-adapter/cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:28) retains parent claims about defaulted rewards in `config.py`; line 30 names the former startup guard. Describe the six required fields in `rewards.py`, current admission rules, and explicit Task 3.1 stop.

VERDICT: APPROVE WITH EDITS