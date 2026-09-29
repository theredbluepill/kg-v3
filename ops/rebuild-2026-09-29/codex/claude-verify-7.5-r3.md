Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify Task 7.5 r3: Kaggriculture parity coverage summary (re-review at 78b78fb)

Date: 2026-09-30. Target: branch `kg/rebuild-7-5` at `78b78fbf7151fae2d3e094897bbd549315170e7b`
(worktree `/Users/poonszesen/kg-v3-t75`, not modified). Scratch: a detached worktree at `/tmp/cv-7.5-r2`
on the same HEAD. I copied the fixtures `generation/` and `orbit_wars_replays/` into it from the main checkout.
I built `owl.rs` there with `CARGO_BUILD_JOBS=2 uv run --offline --no-sync maturin develop --release`:
exit 0, 33 s, max RSS 902 MB.

Report path note: the task named `claude-verify-7.5-r2.md`, but that file already holds the previous
round, which reviewed `81d0bf7`. Commit `78b78fb`, the cookbook note, the log and the tracker row all cite
it, and it is untracked. So that I would not overwrite that evidence, I wrote this round to
`claude-verify-7.5-r3.md`.

## Scope

- `git diff faed717...HEAD` covers the Tasks 1.4/1.5 merge up to `bde3374` (already Codex-approved in
  `verify-merge-env-adapter-r2`), the Task 7.1 landing brought in through `994818b` (Codex-approved), and the
  Task 7.5 commits.
- I focused on what this branch adds on top of the integration tip (`git diff 994818b HEAD`, 46 files).
  That is:
  - the summary section in `docs/rules-parity-coverage.md`;
  - three new negative controls in `engine_rs/tests/replay_parity.rs` and their `TRIM_MANIFEST.json` hash;
  - the cookbook note, `log.md` and `references/index.md`;
  - the `7.5/` and `7.5/r2-fixes/` receipts;
  - the tracker row.
- `kg/isaiah-gap-closure` is still `994818b`, and HEAD contains it. Merge `43bd8d9` is a regular two-parent
  merge (`81d0bf7`, `994818b`). Relative to `994818b`, `cookbook/log.md` and `references/index.md` only
  gain lines, so nothing from the integration side was lost. The only non-docs/ops/cookbook paths changed
  are `engine_rs/tests/replay_parity.rs` and `engine_rs/TRIM_MANIFEST.json`, as the page states.
- Spec (plan Task 7.5): every claim must trace to a test or receipt at the integration tip, and gaps must
  be stated explicitly.

## Checks (scratch, HEAD 78b78fb, Mac CPU, CARGO_BUILD_JOBS=2, OMP_NUM_THREADS=2)

| Command | Result |
| --- | --- |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | 72 passed (41 + 9 + 22), 0 failed; 18 s; max RSS 283 MB |
| `pytest -m "not slow"` on the parity, trim, opponent-import and opponent tests, `-rs` | 178 passed, 11 skipped; max RSS 524 MB. Skips: 1 Mac-bound regeneration, 1 sibling-source reread, and 8 + 1 learned-seat opponent tests (`no opponents_rs hook`). This matches the page. |
| `pytest -m "not slow"` on `test_env_reference.py`, `test_native_env.py`, `test_env.py` | 400 passed; max RSS 379 MB. `test_native_matches_training_batch_16_complete_games` runs and is not skipped (2.09 s). |
| `scripts/check_engine_trim.py` | "engine trim manifest: OK" (the new `replay_parity.rs` hash matches) |
| `scripts/check_doc_freshness.py` | exit 0 |
| `.claude/hooks/cookbook-lint.mjs` on the note, `log.md` and `references/index.md` | exit 0, no issues |
| Cited paths: every `repository:` source in the note, every `R/…` and `R2/…` receipt, every test file and function in the "What is tested" table (including `episode_*`, `env_directory_traces`, `test_project_environment_satisfies_the_engine_pin`, `load_pinned_kaggle`) | All exist at HEAD |
| 7.1 row figures against `7.1/review/results.md` and `7.1/verify-r1/parity-replay.json` | Seeds 20260929–20260936, 8 games, 11,504/11,504 actions, 5,752 transitions, 4×2×1,438 = 11,504; replay `cases: 24`, `compared_actions: [576, 576]` = 1,152. All match. |
| Full-suite counts cited for `994818b` against `merge-7.1/prepare-on-5b43062.log` | Root 274 passed / 5 ignored; Python 2,400 passed / 21 skipped. Both match. |
| `opponents_rs` cargo test | **Not rerun.** Its recorded peak is 3.0 GB, above the 1 GB check limit. I checked `R2/opponents-tests.log` instead: 12 + 5 + 5 = 22 passed, `3000795136 maximum resident set size`, exit 0. |

## Mutations (scratch only, each restored)

| # | Mutation | Result |
| --- | --- | --- |
| M1 | `replay_parity.rs` rewards check disabled (`if false && actual.rewards != rewards`) | Caught: `generated_trace_perturbed_rewards_are_rejected` FAILED (21/22) |
| M2 | Rewards compared for seat 1 only (`actual.rewards[1] != rewards[1]`) | Caught: same test FAILED |
| M3 | Terminal-bank check disabled | Caught: `generated_trace_perturbed_terminal_banks_are_rejected` FAILED |
| M4 | Terminal banks compared for seat 1 only (`banks.get(1) != header.terminal_banks.get(1)`) | Caught: same test FAILED |
| M5 | Transition `done` check disabled | Caught: `generated_trace_perturbed_transition_count_fails_done` FAILED |
| M6 | `done` checked only in one direction (`last && !actual.done`) | Caught: same test FAILED. The count check then reports kind `format`, and the test asserts kind `done`. |
| M7 | `generate_traces.py::oracle_python_runtime` accepts any CPython 3.x (`version[:1] != ORACLE_PYTHON[:1]`); this guards the 7.1 oracle runtime cited in the new row | Caught: 3 failed, including `test_opponent_preset_refuses_other_runtimes_before_playing` and `test_replay_preset_refuses_other_runtimes_before_playing` |

No mutation survived. I also checked the r2-fixes receipt `replay-negative-controls.log` against these
results. It records M1–M3 of r2 failing exactly their new tests and a 22-pass restore at hash `6520938a…`,
which is the manifest value. That is consistent with M1, M3 and M5 above.

## Prior findings (claude-verify-7.5-r2 on 81d0bf7)

| Finding | Status | Evidence |
| --- | --- | --- |
| P2-1: summary stale at `994818b` | **RESOLVED** | See the list below this table. |
| P3-1: 1.94 GiB figure without a receipt | **RESOLVED** | The page now cites `7.5/pytest.json` (989,744 KiB, 21.6 s, 960 MiB stop). The receipt content matches. See new P3-1 for wording. |
| P3-2: rewards, done and terminal-bank checks unguarded | **RESOLVED** | Three tests added at `replay_parity.rs:763-798`. M1–M6 above are all caught. |
| P3-3: note and tracker wording | **RESOLVED** | Tracker: "7.1 is covered at `994818b`; needs a closing pass after 7.3, 7.4 and 3.1". Note lines 137-138 say the same in substance. |
| P3-4: unscoped workflow rule | **RESOLVED** | `docs/rules-parity-coverage.md:293-295` scopes it to Mac-limited review checks and says it does not change `just py-prepare` / `just prepare`. |

Evidence for P2-1:

- HEAD contains `994818b`, and the header (`docs/rules-parity-coverage.md:141-147`) names it.
- The Task 7.1 row at `:189` gives the oracle, the denominator and 1,152 resumed actions, the tests
  `oracle_parity.rs`, `lifecycle.rs` and `test_kaggriculture_parity.py`, and four receipts. All of these
  exist and the figures match. The anchor resolves to `### Task 7.1 Opponents…` at `:299`.
- The 7.1 gap bullet (`:246-250`) is narrowed to the untested scope. It agrees with the 7.1 section (`:299+`),
  which reports the same 22 tests, 11,504 actions and CPython 3.11 requirement.
- "Current checks" (`:259-285`) matches the r2-fixes logs and my reruns.
- The statement about what changed since `994818b` is true, as confirmed by `git diff 994818b HEAD --name-only`.
- The cookbook conflicts are resolved with no lost lines.

## Findings

### P3-1: "the same pytest selection" misstates the guarded run the memory inference rests on

- Where: `docs/rules-parity-coverage.md:290-293`.
- Problem: the page says "Codex's guarded run of the same pytest selection" reached 989,744 KiB, and so
  "the unguarded run probably exceeded the 1 GB check limit".
- According to `7.5/pytest.json` and `7.5/pytest.log`, the guarded run was
  `pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py tests/kaggriculture -q`.
  That is a three-path subset with no `-m "not slow"`. Its figure was a sampled **process-tree** aggregate.
- `just py-prepare` runs `pytest tests/ -m "not slow"` (`justfile:4,29`).
- The two selections differ, so the sentence is inaccurate. The "probably" also rests on a comparison of a
  different selection under a different memory measure.
- The hedge keeps this from being a false conclusion, but a reader will take "same" as fact.
- Fix: say "a guarded run of a subset (parity, trim and `tests/kaggriculture`, slow tests included)". Either
  keep "probably" with that qualifier, or soften it to "may have exceeded".

### P3-2: superseded first-round prose in the cookbook note still reads as current

- Where: `cookbook/references/kaggriculture-parity-summary-maps-tested-and-untested-layers.md:82-94`.
- Problem: the "Verification" list opens with the `bde3374` round and gives no marker that it is historical:
  - "Engine `cargo test` passed 69 tests";
  - "No Rust source, Cargo manifest or lockfile changed after merge commit `7f797a3`. The page therefore
    cites the merge's `just prepare` result (274 passed, 5 ignored) as the latest full run."
- At this tip, `opponents_rs` Rust did change after `7f797a3`, through `994818b`. The page now cites
  `994818b`'s `prepare-on-5b43062.log`, not the `7f797a3` merge.
- Lines 124-130 give the current statement, so the note contradicts itself on a quick read.
- CLAUDE.md asks that superseded current prose be replaced, with history linked.
- Fix: prefix lines 82-94 with "First round, at `bde3374`:" or fold them into a one-line history pointer to
  `7.5/results.md`.

### Note (not a defect of the deliverable): the `opponents_rs` check breached the 1 GB limit

`R2/opponents-tests.log` records 3.0 GB max RSS. That figure is the largest single reaped descendant, so it
is probably rustc or a test binary, but that is unattributed. The page, the note, the log and
`r2-fixes/results.md` all disclose it, which is honest. I did not rerun it. Before that crate's tests run
on the Mac again, they need a memory guard. Alternatively, attribute the peak by timing the already-built
test binaries separately from compilation.

## Owner-rule check

This change is docs, receipts, cookbook and test-only Rust (negative controls). It adds no trainer, model
or v2 code, and no policy input or between-turn state. The page keeps `scripts/run_ppo.py` as the one
canonical trainer, stopping at the Task 3.1 seam. It mentions no deadlines or submission timing. The
commits carry the trailer, and the merge is a regular merge. Nothing was pushed.

## Cleanup

Before removal, `git status --short --untracked-files=no` in the scratch copy was empty.
`git -C /Users/poonszesen/kg-v3-t75 worktree remove --force /tmp/cv-7.5-r2` exited 0, and the path is gone.
`/Users/poonszesen/kg-v3-t75` is still at `78b78fb` with 0 tracked modifications. I did not touch
`kg-v3-pod6000`, the pod branches or `~/kaggriculture-v2`.

VERDICT: APPROVE WITH EDITS
