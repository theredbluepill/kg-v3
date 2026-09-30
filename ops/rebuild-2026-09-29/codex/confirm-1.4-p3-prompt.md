You are an independent verifier for one fix on branch `kg/rebuild-env` in the worktree
`/Users/poonszesen/kg-v3-env`, commit `b6b722f` (parent `1e63597`). Scope: only this fix
and the single open r3 finding. Do not re-review the rest of Task 1.4.

## Prior finding to re-verify

`ops/rebuild-2026-09-29/codex/verify-env-r3/review.md` (lines ~13-40), P3 "roll back a failed
fixture-pair publication": `scripts/record_kaggriculture_env_reference.py` replaced the
destination NPZ and then its JSON manifest with no rollback. An `OSError` on the second
replacement left an NPZ without a manifest (fresh output) or a new NPZ beside the old
manifest (existing output). The requested fix: keep the prior pair until publication
succeeds, restore it if the second replacement raises, remove a newly created partial pair,
and add separate fresh-output and existing-output regression tests that fail the second
replacement and assert exact destination bytes afterwards.

## What changed in b6b722f (claims to check, not to trust)

- `publish_pair` / `restore_target` in the recorder: prior bytes are read before any
  replacement; on any exception each already-replaced target is restored in reverse order
  (or unlinked when it did not previously exist), then the error re-raises. Abrupt process
  termination between the two replacements is declared out of scope.
- Two tests in `tests/tools/test_record_kaggriculture_env_reference.py`:
  `test_failed_second_replacement_removes_fresh_partial_pair` and
  `test_failed_second_replacement_restores_existing_pair`.
- Because `recorder_sha256` is part of fixture source custody, the fixture was re-recorded on
  the Mac under the recorder's own 115 s / 960 MiB watchdog with `CARGO_BUILD_JOBS=1`.
  Claimed: the NPZ is byte-identical to the previous commit (sha256 `494bbf2c…c976`) and the
  only manifest change is `sources.recorder_sha256`. Receipts: `ops/rebuild-2026-09-29/1.4/p3-rerecord/`.
- Cookbook: `cookbook/references/native-game-semantics-use-v3-owned-buffers.md`,
  `cookbook/references/index.md`, `cookbook/log.md`.

## Required work

1. Read the diff `git diff 1e63597 b6b722f` for the recorder, test, fixture JSON and cookbook.
2. Confirm the fixture claim: the NPZ blob is unchanged between `1e63597` and `b6b722f`; the
   JSON differs only in `sources.recorder_sha256`, and that value equals the sha256 of the
   committed recorder at `b6b722f`.
3. Mutation testing on a scratch copy (for example `cp` the recorder into `/tmp`, edit the
   tracked file, run the two new tests, then restore it byte-exactly and prove the restore by
   sha256). Try at least one mutation per new guard, for example: (a) remove the rollback
   loop; (b) skip the unlink for a target that did not exist before; (c) restore the prior
   bytes of only the first target, or restore without regard to what was replaced. Report
   which tests fail for each mutation. Note that editing the recorder changes its hash, so the
   frozen-fixture custody test will fail under any mutation; that failure is expected and is not
   evidence that the new tests are effective.
4. Run and report counts for: `uv run pytest tests/tools/test_record_kaggriculture_env_reference.py tests/kaggriculture/test_env_reference.py -q`
   and `uv run mypy scripts/record_kaggriculture_env_reference.py`. Keep runs small: this is a
   Mac with no GPU. Do not run the recorder itself, training, or anything over about 1 GB RSS or
   2 minutes.
5. Check that the cookbook text agrees with the code and receipts (numbers, hashes, the
   exception-only scope).

## Rules

- You are an independent verifier. Leave no tracked modification: `git status --short` must
  show no tracked change when you finish (untracked scratch files under `/tmp` are fine).
  Do not commit.
- Mark the prior r3 P3 finding RESOLVED, PARTIAL or UNRESOLVED with evidence.
- Report every finding with severity (P1/P2/P3), file:line and a concrete fix.
- End the report with exactly one final line of the form
  `VERDICT: APPROVE`, `VERDICT: APPROVE WITH EDITS` or `VERDICT: REQUEST CHANGES`.
