Confirmed `bb0f7e4` on `kg/rebuild-env-briefs` resolves both findings.

- **P2 resolved:** [1.4.md:705](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.4.md:705) specifies ample seeds for (a)–(f), injection-hit assertions, and near-exhaustion only for (g). Independent seed-arithmetic checks pass.
- **P3 resolved:** [1.4.md:715](/Users/poonszesen/kg-v3-envbriefs/ops/rebuild-2026-09-29/briefs/1.4.md:715) correctly states nonterminal `dones` remains false and requires a terminal fixture for nonzero preservation coverage.

The commit changes only those two paragraphs and adds their confirmation note at line 781. Everything else, including `1.5.md`, is unchanged. Worktree clean; diff whitespace check passes.

Read-only specification review; no files changed or native/build tests run.

VERDICT: APPROVE