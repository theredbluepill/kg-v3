You are Codex, INDEPENDENT VERIFIER of branch kg/rebuild-7-3 (this worktree, HEAD f23cd4fcfab3187ad008d0bfca2408d5f8f20fdf) vs base 0b8cf98ef57fc49a329dca4c8368c630586c4752 (three-dot diff). Leave no tracked modification.

Task 7.3 per ops/rebuild-2026-09-29/briefs/7.3-replay-export.md. Replay export: Kaggle episode format, round-tripped through the native engine from the seed header; byte/semantic equality of round-trip; 8 replays per evaluation.

Also mark every finding in /Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.3-r2.md as RESOLVED / PARTIAL / UNRESOLVED, with evidence.

Try at least one mutation per new oracle on a scratch copy and restore byte-for-byte (confirm with git status / checksum).

Run and report counts for:
- cargo test --manifest-path engine_rs/Cargo.toml --locked --offline
- root cargo test
- uv run python scripts/check_engine_trim.py
- relevant pytest
- uv run mypy python/owl scripts

End with VERDICT: APPROVE / APPROVE WITH EDITS / REJECT, with findings (severity, file:line, fix).
