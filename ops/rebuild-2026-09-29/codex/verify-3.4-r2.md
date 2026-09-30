- **P1 — Prior startup finding remains partially unresolved.** [run_ppo.py:149](/Users/poonszesen/kg-v3-configs/scripts/run_ppo.py:149) now calls the checker, but all three Kaggriculture YAMLs fail the real loader first—five schema errors each. Startup tests bypass that loader. **Fix:** complete typed config integration and add unskipped real-loader rejection/logging tests.
- **P3 — Measurement wording is premature.** Both ranked configs’ line 29 ([2-rank](/Users/poonszesen/kg-v3-configs/configs/kaggriculture_2rank.yaml:29), [4-rank](/Users/poonszesen/kg-v3-configs/configs/kaggriculture_4rank.yaml:29)) describe thread counts as measured; Task 6.1 remains pending. **Fix:** label them provisional.

Prior **P2 and P3 are resolved**. Recipe values match Isaiah’s global workload and contract v4. No shim wrappers found.

Verification:

- Requested pytest: **1,331 passed, 10 skipped**; Orbit Python tests green.
- Requested mypy: **58 files, no issues**.
- Rust: **155 passed, 2 ignored**, after supplying existing sibling-worktree fixtures.
- Startup-call mutation: **4 passed → 4 failed → 4 passed**, restored byte-for-byte.
- All **766 tracked files unchanged**.

[Full report and evidence](/Users/poonszesen/kg-v3-configs/ops/rebuild-2026-09-29/codex/verify-3.4-r2/report.md).

VERDICT: REJECT