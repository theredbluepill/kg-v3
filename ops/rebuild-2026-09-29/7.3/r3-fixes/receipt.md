# Task 7.3 verification r3 — fix receipt

Source report: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.3-r3.md`
(REJECT, two P2s); the verifier's full evidence is kept in
`../independent-verifier-r3/`. Branch `kg/rebuild-7-3`, worktree
`/Users/poonszesen/kg-v3-t73`, fixes applied on top of `f23cd4f`.

## Finding 1 (P2, false successful custody) — fixed test-first

`ReplayRecorder._write` wrote a `status: complete` sidecar before the episode.
An episode open/write failure then left successful custody for a missing or
corrupt episode, and the abort handler could not replace it.

Change (`python/owl/kaggriculture/replay_export.py`):

- `_publish_new_file` writes each file to a dot-prefixed temporary file in the
  output directory, flushes and fsyncs it, then hard-links it into place.
  Linking is the atomic publication step; unlike `rename` it refuses to replace a
  path created meanwhile. The temporary file is always removed.
- `_publish` publishes the episode first, then its custody, then fsyncs the
  directory. On any failure it removes the files this attempt published.
- On a failed successful/truncated publication, `_publish_failure_custody`
  publishes an error sidecar (`replay publication failed: <type>: <message>`,
  `complete: false`, no `episode_sha256`, no `verification`) and retires the
  game; the original exception is re-raised. If the error sidecar also fails,
  the game stays active (the abort handler retries it) and the failure is a note
  on the original exception.

Red first: `publication-red.log` (8 failed, 1 existing passed). Green:
`publication-green.log` (9 passed). Tests:

- `test_publication_failure_never_leaves_successful_custody` (4 cases: episode
  or custody x partial write or close failure).
- `test_unpublishable_error_custody_leaves_no_files_and_the_game_active`.
- `test_episode_is_durable_before_custody_claims_it` (fsync/link order).
- `test_publication_never_replaces_a_path_created_meanwhile`.
- `test_partial_episode_write_during_live_evaluation_leaves_error_custody`: the
  verifier's partial-write probe (20 bytes then `OSError`) over two live native
  games; both games end with error custody and no episode.

The verifier's probe scripts patch the final episode path in `xb` mode, which
the fixed code no longer opens, so they are not rerun unchanged; the live test
above injects the same failure into the staged episode write.

Mutations (`run_mutations.py`, `mutations.json`, rerun after the r4 P3
edit below): 10 of 10 detected, source restored byte-for-byte (SHA-256
matches): custody published before episode (2 failures), rename replacing an
existing path (3), no fsync before link (2), published files not removed on
failure (4), rollback failure replacing the original error (1), rollback
continuing past unremovable custody (1), no error custody after a failed
publication (6), staging file left behind (8), error custody keeping the episode
hash (5), unpublished error custody retiring the game (2).

## Verification r4 P3 edit

Codex r4 (`/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.3-r4.md`,
APPROVE WITH EDITS; evidence in `../independent-verifier-r4/`) found that a
double fault (directory fsync fails, then removing the published custody fails)
let the rollback `PermissionError` replace the original exception. `_publish`
now removes files in reverse order, catches a removal failure, notes it on the
original exception and stops, so custody that cannot be removed keeps the
episode it claims (hash still matches); the original exception is re-raised.
`test_rollback_failure_keeps_the_original_error_and_a_matching_pair` was red
first (`rollback-red.log`), then green with the replay suites
(`rollback-green.log`: 68 passed). Two added mutations detect the guard.

## Finding 2 (P2, canonical evaluation incomplete) — deferred dependency

Not implemented here, by instruction. `run_ppo._create_eval_env` still raises
`NotImplementedError` for Kaggriculture and
`test_kaggriculture_canonical_evaluation_exports_eight_replays` stays skipped.
Dependency: plan Task 3.1 (game seam in the trainer: rollout storage and
observation/action mapping for Kaggriculture batches, itself after Task 1.5),
being done on lane A. Reopening condition: once Task 3.1 lands, drive
`native_evaluation.evaluate_native_games` from `_evaluate_games` and unskip the
acceptance test so one canonical call produces eight complete, byte-verified
episodes with seed, seat and checkpoint custody. Until then Task 7.3 is not
landable as complete canonical evaluation export.

The `_create_eval_env` error text and the skip reason in
`tests/scripts/test_run_ppo.py` still name the Task 1.5 adapter and model token
policy; they are left unchanged here so this branch does not edit `run_ppo.py`
or its tests while lane A works on Task 3.1.

## Checks

Only Python, tests, docs, cookbook and ops receipts changed (no Rust), so the
check is `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just
py-prepare` (`prepare.log`, rerun after the r4 P3 edit): format, ruff,
Python 3.11 syntax, mypy (68 source files, no issues), **2,208 Python passed /
12 skipped** in 187.71 s, docs freshness OK. The r3 commit's run passed 2,207 /
12 skipped; its first attempt failed ruff ARG001 on two unused fixture
arguments (fixed with `usefixtures`).
