# Re-verify the BC single-player filter (r2)

Your r1 report is `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-5.1-team-filter.md` (REQUEST CHANGES; evidence in `/tmp/bc-team-review-tZFUFb/`). The fix is the latest commit on `kg/rebuild-bc-now` in `/Users/poonszesen/kg-v3-bcnow`; review `git diff 42a8b39 HEAD`.

1. **P2 abbreviations:** the parser is now built with `allow_abbrev=False`. Rerun your leak reproduction (`--tea NAME`, `--te=NAME`) and confirm it is refused, and that no accepted spelling persists the name.
2. **P2 draws:** `split_totals` counts equal terminal banks. Rerun your three-draws and decisive-mirror cases.

Mark each r1 finding RESOLVED/PARTIAL/UNRESOLVED. One round, blocking findings only. Do not modify tracked files; scratch only under /tmp. Run the BC and preparer tests plus mypy and ruff on the changed files.

End with exactly one line: `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS`, or `VERDICT: REQUEST CHANGES`.
