Wrote [Task 1.3 brief](/Users/poonszesen/kg-v3-observe/ops/rebuild-2026-09-29/briefs/1.3.md). Only that file changed; no implementation or commit.

Key decisions:

- Root-crate encoder; apply the test-only L4 `fixture_float` repair.
- One public `snapshot()` per env, with measured clone cost and transactional publication.
- A 512-state, both-seat oracle preserving all 8,176 reference values, with reproducible recording and SHA-256 custody.
- Depend on Task 2.1’s merged schema; test every corpus batch with its actual `check_contract()`.

Checks run:

- Offline, two-job `cargo test --locked --offline`: **155 passed, 2 ignored**.
- Project-Python trim checker: **passed**. System Python was too old.
- Brief whitespace/structure checks: **passed**; all 29 buffers match Claude’s schema.

Claude’s review questions cover imported-state reconstruction limits, configuration/transition admission, snapshot-wrapper acceptance, schema merge and header-binding ownership, downstream index validation, and oracle execution/storage budget.