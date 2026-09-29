You are Codex, stream A (engine). Worktree: this checkout, branch `kg/rebuild-codex`. IMPLEMENT Task 1.1 exactly per your brief `ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md` WITH Claude's required edits in `ops/rebuild-2026-09-29/briefs/1.1-review-claude.md`:
1. Python checker `scripts/check_engine_trim.py` plus pytest `tests/tools/test_check_engine_trim.py`, instead of Node.
2. Engine fmt/clippy/test wired into `just rs-prepare` as separate `--manifest-path` invocations.
Work test-first where the brief says so (the replay-parity test must be shown to fail against a deliberately broken state before it passes).

RULES: owner's Mac; `CARGO_BUILD_JOBS=3`; offline (all crates are cached; use `--offline`/`--locked`); no training. Your sandbox cannot write `.git`: do NOT try to commit. Leave all changes in the working tree. Claude commits after an independent Codex verification review.

Run at the end: `CARGO_BUILD_JOBS=3 cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`; `python scripts/check_engine_trim.py`; `uvx --offline --from rust-just just prepare` (root Rust 155 passed / 2 ignored unchanged; Python suite green). Save receipts under `ops/rebuild-2026-09-29/1.1/`.

FINAL REPORT: files changed or added, each command with pass/fail counts, the resulting `lib.rs` SHA-256 (must equal the brief's expected `c4b9bac5…`), replay-parity counts (episodes, transitions, snapshots), and any deviation from the brief with its reason.
