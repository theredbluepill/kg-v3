Two findings:

- **P2 — [scripts/run_ppo.py:468](/Users/poonszesen/kg-v3-semantics/scripts/run_ppo.py:468):** `eval/promoted=1` is logged before promotion completes. Injected refresh and checkpoint failures both leave false success telemetry. **Fix:** log after successful promotion and add failure-ordering coverage.
- **P3 — [scripts/run_ppo.py:1442](/Users/poonszesen/kg-v3-semantics/scripts/run_ppo.py:1442):** Seed documentation overclaims nonoverlapping streams; concrete collision/overlap counterexamples exist. **Fix:** narrow the docstring, RL docs and cookbook to proven guarantees.

Raw-bank outcomes, truncation reward/bootstrap, joint clipping and value guards match the intended semantics. No shim wrappers found. Native integration remains explicitly deferred.

Checks:

- Requested pytest: **1,341 passed, 6 skipped**, including Orbit tests.
- Requested mypy: **58 files, no issues**.
- `py-prepare`: passed using cached offline tooling.
- Mutation: expected failure, then passed after byte-for-byte restoration.
- Eight outcome cases matched the reference oracle.

Tracked files remain unchanged. [Full report and evidence](/Users/poonszesen/kg-v3-semantics/ops/rebuild-2026-09-29/codex/verify-3.2-3.3-independent/review.md).

**VERDICT: APPROVE WITH EDITS**