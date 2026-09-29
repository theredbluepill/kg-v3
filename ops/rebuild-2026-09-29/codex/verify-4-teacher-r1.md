Two findings:

- **P2 — [test_teacher.py:1458](/Users/poonszesen/kg-v3-trainer-model/tests/kaggriculture/test_teacher.py:1458):** T19b contains only an unconditional assertion. Implement the required resume/fresh-launch checks behind its dependency skip; identify it as unimplemented in the cookbook until then.
- **P3 — [model-architecture.md:737](/Users/poonszesen/kg-v3-trainer-model/docs/model-architecture.md:737):** Cached KL is described as supporting only `discrete_targets`. Scope that restriction to Orbit.

No functional defect found in the implemented teacher KL/value CE, guard routing, cache sizing, or dispatch. No shim wrappers found.

Validation:

- Requested pytest: **1,372 passed, 6 skipped**
- Orbit suites separately: **1,048 passed, 3 skipped**
- Requested mypy: **passed, 58 files**
- `py-prepare`: **passed**
- Three mutations caught; files restored byte-for-byte
- All **812 tracked files unchanged**

[Full report and evidence](/Users/poonszesen/kg-v3-trainer-model/ops/rebuild-2026-09-29/codex/verify-4-teacher-r1/report.md). Approval covers implemented **4.1–4.3**; full Phase 4 remains incomplete pending T18, T19b and 4.4.

**VERDICT: APPROVE WITH EDITS**