Read-only confirmation at `ac0496e` on `kg/rebuild-bc-brief`; worktree is clean. Using the original report’s four numbered findings:

| Finding | Status | Confirmation |
|---|---|---|
| 1 — Recorded seed | **RESOLVED** | Transition headers use `info.seed`; encoding retains the placeholder and shards exclude seeds. |
| 2 — Executable check | **RESOLVED** | Offline Rust binary, invocation, per-turn results, tests and source custody are specified. |
| 3 — Hard-link safeguard | **RESOLVED** | Explicit prohibition on installed-file edits plus a fresh copy-linked venv. |
| 4 — Custody reconciliation | **RESOLVED** | Current-pod placement and `source.json` hash are consistent; cookbook note/index/log updated together. |

Two new inconsistencies need edits:

- **P2 — Normalization probe protocol.** [Q1](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:553) requests extra records, but the client requires exactly 719 results. The output also reports differences against recorded state, which cannot establish raw-versus-normalized equality when both mismatch. Specify a separate normalization probe with its own count and direct snapshot comparison, separate from baseline admission accounting.
- **P3 — Unexecuted Rust proof overstated.** The [Reference description](/Users/poonszesen/kg-v3-bcbrief/cookbook/references/rebuild-data-preparation-preserves-replay-identity.md:4) says the Rust check “proves” the pairing, although the harness is unwritten. Change this to “specifies an offline Rust transition check”; the executed evidence is Python.

`git diff --check` passes. No files changed, builds run or live pod state checked. No other new inconsistency found.

VERDICT: APPROVE WITH EDITS