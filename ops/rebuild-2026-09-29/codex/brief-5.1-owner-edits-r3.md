Both r2 corrections are resolved, but the matching index still needs one edit.

- **P2 resolved:** [Q1](/Users/poonszesen/kg-v3-bcbrief/ops/rebuild-2026-09-29/briefs/5.1-bc-data.md:558) specifies a separate normalization mode, its own pair count, and direct raw-versus-normalized snapshot comparison.
- **P3 resolved in Reference:** [Description](/Users/poonszesen/kg-v3-bcbrief/cookbook/references/rebuild-data-preparation-preserves-replay-identity.md:4) now says “specifies” and identifies Python as the executed evidence. The new log entry agrees.
- **P3 remaining — index reconciliation:** [Index line](/Users/poonszesen/kg-v3-bcbrief/cookbook/references/index.md:12) is unchanged and still says “a step+1 transition check run by an offline Rust binary.” Make its specified/unexecuted status explicit, matching the Reference and the cookbook’s required note/index/log update.

`4fdd61e` changes only the brief, Reference description, and log entry; no unrelated changes. Branch and HEAD match, worktree is clean, and `git diff --check` passes. No files changed or builds run.

VERDICT: APPROVE WITH EDITS