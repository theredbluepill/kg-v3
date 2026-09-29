# Task 7.1 verify r1 resolutions

Input: Codex verification `ops/rebuild-2026-09-29/codex/verify-7.1-r1.md`
(REJECT at `f15a413`, base `b8747b6`). Branch `kg/rebuild-7-1`.

## Findings and resolutions

1. **P2: original-reference mid-episode replay missing (resolved).**
   Test-first (`replay-python-red.log`, `replay-rust-red.log`), the generator
   gained `replay_case`, `oracle_sources`, `write_replay_fixture` and the
   `opponent-replay` preset. For each frozen oracle (custody rechecked against
   its MANIFEST) and each point 37 (day 1 hour 13), 360 (day 15 hour 0) and 695
   (last hour before the final day), fresh controllers for both seats (a fresh
   module per original submission) observe every prefix step. Each must choose
   the recorded action; recorded actions drive Kaggle's engine and the rebuilt
   public/private state must equal the trace, or generation fails. The
   controllers then act on their own for 24 steps; their actions and the final
   public/private state and statuses are frozen.

   Generation: `/Users/poonszesen/kaggriculture/.venv/bin/python` (CPython
   3.11.15, kaggle-environments 1.32.7, read-only), one invocation, 23.9 s wall,
   773 MB peak RSS (`replay-generation.log`, `generation-replay.json`). Output
   `opponents_rs/fixtures/replay/REPLAY.json.gz`, 23,323 B, SHA-256
   `0d6fee093244de7e9b89757c9a2594fc16cc8392f5f12ff523d4e96a07987391`;
   a second run reproduced it byte-for-byte (`replay-regeneration.log`).

   Native check (`opponents_rs/tests/oracle_parity.rs`): header identity,
   CPython 3.11 runtime and the oracle MANIFEST SHA-256 must match; 24 cases
   must cover every bot in both seats at every point. Fresh native controllers,
   rebuilt through the step-zero lifecycle from the same recorded prefix (each
   prefix action compared too), match **1,152 / 1,152** resumed actions (576 per
   seat) and all 24 final states (`parity-replay.json`, `replay-rust-green.log`).

   Mutation checks: a second test tampers the first resumed action of each seat
   in all eight step-37 cases, the last resumed action (step 60, seat 1) of one
   case and one final public state; all 17 fail at the exact case, step and seat.
   A scratch controller mutation (registry output altered at step 700 only)
   failed all eight step-695 cases on resume and nothing else; `registry.rs` was
   restored and its SHA-256 re-verified (`replay-controller-mutation.log`).

2. **P3: stale test count (resolved).** The Reference now reports the actual
   count after this change: 22 opponent tests (12 unit, five lifecycle/match,
   five oracle). The verifier's 20 was correct for `f15a413`; this change adds
   two oracle tests. `docs/rules-parity-coverage.md` is updated to match.

## Custody

`run2/write_opponent_manifest.py` registers the replay fixture as authored
(hash-pinned) and refreshed `OPPONENT_MANIFEST.json`; the custody checker passes
with entry pins and with `--original-sources`. The trim updater accepts the
committed `f15a413` output as input (test-first, `updater-red.log` /
`updater-green.log`, 8 tests, 12 subtests), registers the new paths and stays
byte-idempotent; `scripts/check_engine_trim.py` passes. No imported controller
or engine byte changed.

## Checks (actual)

`CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare`: exit 0
(`prepare.log`). Root Rust 254 passed, 4 ignored; engine 41 + 9 + 19 passed;
opponents 12 + 5 + 5 passed; Python 1,707 passed, 17 skipped; opponent custody,
engine trim, format, lint, mypy (64 files), docs lint and docs freshness pass.
`scripts/check_opponent_import.py --original-sources` also passes. Two earlier
prepare attempts failed and were fixed: a mypy name collision in `main`, and
the unit replay fixture named `oracle-…`, which enabled the per-process 1 GB
generation bound inside the full pytest process (renamed `replay-unit`).

## Remaining gaps

- Each prefix is the controllers' own recorded play, so every resumed action
  also equals the contiguous trace (`continuous_equal_actions` 24/24 in every
  case). This shows deterministic reconstruction on both sides; states the
  controllers did not create (foreign prefixes) are untested.
- Explicit `reset` after another episode is checked natively only.
- Earlier gaps stand: no Python-side rejection or confirmed-shortage coverage,
  Starter never hires, default config only, EcoBot/E776 license notices,
  learned-seat checks await Task 1.4, no strength or panel claim.
