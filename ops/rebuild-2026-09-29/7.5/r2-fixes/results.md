# Task 7.5 r2 fixes: refresh for integration 994818b and replay negative controls

Source review: `ops/rebuild-2026-09-29/codex/claude-verify-7.5-r2.md` (main checkout, local;
independent Claude subagent standing in for Codex during its usage limit; REQUEST CHANGES at `81d0bf7`).

| Finding | Action |
| --- | --- |
| P2-1 stale at tip `994818b` | Merged `994818b` (merge commit `43bd8d9`; cookbook log/index conflicts kept both sides). The summary now names `994818b` as its tip, adds a Task 7.1 row (original-submission Python oracle, 8 games, 11,504 compared actions, 1,152 resumed actions; `opponents_rs/tests/oracle_parity.rs`, `lifecycle.rs`), narrows the 7.1 "not tested" bullet to its untested scope, and replaces "Current checks" with the checks below. |
| P3-1 1.94 GiB figure had no receipt | Replaced by the tracked `7.5/pytest.json` figure: sampled process-tree peak 989,744 KiB at 21.6 s, stopped by the 960 MiB limit. |
| P3-2 M1-M3 survived | Added `generated_trace_perturbed_rewards_are_rejected`, `..._terminal_banks_are_rejected` and `..._transition_count_fails_done` to `engine_rs/tests/replay_parity.rs`; updated its `TRIM_MANIFEST.json` authored hash. Each mutation now fails exactly its test (`replay-negative-controls.log`). |
| P3-3 note/tracker wording | Note now matches the tracker row (7.1 landed; closing pass after 7.3, 7.4 and 3.1). |
| P3-4 unscoped workflow rule | Scoped to Mac-limited review checks in this rebuild; states it does not change `just py-prepare` / `just prepare`. |

Checks: `run_checks.sh` in this directory; per-command logs with `/usr/bin/time -l` maximum RSS of the
largest single child process. Engine fmt/clippy exit 0; engine tests 72 passed; opponents 22 passed;
trim and opponent-import checks exit 0; pytest shards 178 passed / 11 skipped and 400 passed; docs-fresh exit 0.

Limit breach: the `opponents_rs` `cargo test` peaked at 3.0 GB max RSS (`opponents-tests.log`), above the
owner's 1 GB check limit. It was not rerun to attribute the peak between rustc and the test binaries
(the oracle binary ran 27 s). Future Mac checks should not run that crate's tests unguarded.

Not rerun: root `cargo test` and the full Python suite (no root Rust or Python changed after `994818b`;
latest full run `ops/rebuild-2026-09-29/merge-7.1/prepare-on-5b43062.log`).
