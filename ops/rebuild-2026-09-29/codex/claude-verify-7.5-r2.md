Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify Task 7.5 r2: Kaggriculture parity coverage summary (re-review at 81d0bf7)

Date: 2026-09-30. Target: branch `kg/rebuild-7-5` at `81d0bf7b7b453c37372cb116e7a1b739294476e1`
(worktree `/Users/poonszesen/kg-v3-t75`, not modified). Scratch: detached worktree `/tmp/cv-7.5-r1`
at the same HEAD. I copied the fixtures `generation/` and `orbit_wars_replays/` in from the main checkout.
I built `owl.rs` there with `CARGO_BUILD_JOBS=2 uv run --offline maturin develop --release`: exit 0, 80 s,
max RSS 859 MiB.

Report path note: the task named `claude-verify-7.5-r1.md`, but that file already holds the previous
subagent round on `e9aafba`. Commit `81d0bf7` and the cookbook note cite it, and it is untracked. I wrote
this round to `claude-verify-7.5-r2.md` so that evidence is not overwritten.

## Scope

- `git diff faed717...HEAD` spans 20 commits. Up to `bde3374` it is the Tasks 1.4/1.5 merge, which Codex
  approved in `codex/verify-merge-env-adapter-r2.md`. After that come the three Task 7.5 commits:
  - `a8afb33`: the summary, seven corrections plus a pointer, the `7.5/` receipts and the cookbook note/index/log;
  - `e9aafba`: the record that Codex hit its usage limit;
  - `81d0bf7`: the P3 fixes from the r1 subagent review.
- I reviewed all of `81d0bf7`. I re-checked the new section's claims and the P3 fixes against the source.
  I also checked the branch against the **current integration tip** `kg/isaiah-gap-closure` = `994818b`, not only against its base.
- Spec: plan Task 7.5 requires that every claim trace to a test or receipt at the integration tip, and
  that gaps are stated explicitly.

## Checks (scratch, HEAD 81d0bf7)

| Command | Result |
| --- | --- |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | 41 + 9 + 19 + 0 doc = 69 passed, 0 failed; max RSS 270 MiB |
| `uv run --offline --no-sync python scripts/check_engine_trim.py` | exit 0, `engine trim manifest: OK` |
| `uv run --offline --no-sync python scripts/check_doc_freshness.py` | exit 0 |
| `pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py` | 97 passed; max RSS 304 MiB |
| `pytest test_env_reference.py test_native_env.py test_env.py` | 400 passed (16-game `TrainingBatch` comparison ran, not skipped); max RSS 282 MiB |
| `pytest test_codec.py test_native_tables.py test_game.py test_rewards.py test_native_grammar_bindings.py` | 163 passed; max RSS 282 MiB |
| `pytest tests/kaggriculture/test_observe.py` | 55 passed, 2 skipped (pinned memory needs CUDA); max RSS 296 MiB |
| cookbook-lint hook mode on the note, `log.md` and `references/index.md` | `{}` for each |
| cookbook-lint `--staged-sources --require-log` over `bde3374..81d0bf7` (soft reset in scratch) | exit 0 |
| every cited path in the new section (42 paths) | all exist at the tip; `pod-oracle/` and the fixture `.npz` are tracked |
| `git merge-tree --write-tree 994818b 81d0bf7` | textual conflicts only in `cookbook/log.md` and `cookbook/references/index.md`; `docs/rules-parity-coverage.md` auto-merges (see P2-1) |

The RSS figures come from `/usr/bin/time -l`, which reports the maximum RSS of a single child process,
not the sum across the process tree. I did not run the root `cargo test` or the whole `tests/kaggriculture`
directory, because both are known to exceed the 1 GB Mac check limit.

Claims I re-checked against the source:
- The 35-buffer count: `test_env.py:158,489,638`.
- 964 table bits: `grammar_tests.rs:630` and `test_native_grammar_bindings.py:54`.
- The one-ULP reward tolerance and the bitwise rewards/dones/banks/econ comparison:
  `test_env_reference.py:141-190`.
- `Game::new_with_seed_decimal` in the replay path: `replay_parity.rs:484`. `Game::new` at line 335 is only the synthetic API test.
- The Codex rows' offline and thread environment: `7.5/results.md:22-26`.
- uv rebuilt the package during `py-prepare`: `7.5/py-prepare.log:5-8`.
- The 10 py-prepare skips: `7.5/py-prepare.log`.
- The 14 "not tested" bullets.

## Mutations (scratch only; each reverted with `git checkout`, tree clean before removal)

| # | Mutation | Claim probed | Result |
| --- | --- | --- | --- |
| M1 | `engine_rs/tests/replay_parity.rs:574`: terminal-bank check disabled (`if false && banks != …`) | "The replay comparator checks … terminal banks" (docs:178-179) | **SURVIVED**: 19/19 replay tests pass |
| M2 | `replay_parity.rs:553`: transition `done` check disabled | "… step/done" | **SURVIVED**: 19/19 pass |
| M3 | `replay_parity.rs:548`: transition `rewards` check disabled | "… typed rewards" | **SURVIVED**: 19/19 pass |
| M1-3 liveness | Three scratch probe tests: perturb `rewards` at step 100, header `terminal_banks[0]` +1, header `transitions` +1 on `gen-edge-vs-edge` | Are the checks live at all? | All 3 probes pass on the unmutated code (kinds `rewards`, `terminal_banks`, `done`). With M1+M2+M3 applied, all 3 fail. The checks are live today; no committed test guards them. |
| M4 | `src/kaggriculture/env.rs:628`: `transition_econ_before` filled from `after_econ` (rebuilt `owl.rs`) | Task 1.4: "compares … economic counters bitwise" | Caught: `test_env_reference.py::test_native_matches_training_batch_16_complete_games` |
| M5 | `env.rs:36`: `stride < 1` changed to `stride < 0` (rebuilt) | Task 1.4 seed admission | Caught: `test_native_env.py::test_strict_seed_admission[0-0-ValueError]` |
| M6 | `python/owl/kaggriculture/codec.py` `encode_actions`: `order_limits[env, 1 - seat]` | Task 1.5 codec | Caught: `test_codec_batch_uses_each_seats_public_grammar_parameters`, `test_native_codec_batch_preserves_json_order_and_input_rows` |
| M7 | `codec.py` `decode_actions`: `actor_mask[env, 1 - seat]` | Task 1.5 codec | Caught by the same 2 tests |
| M8 | `engine_rs/src/lib.rs` `fib`: seed `b = 2` (hire cost) | Official episodes, generated traces, trim custody | Caught: 8 replay tests fail (all 4 `episode_*`, `generated_fixtures_replay`, 3 perturbation tests); lib test `econ_v4_labor_and_malformed_commands` fails; `check_engine_trim.py` exit 1 |

M4 and M5 were rebuilt one at a time. I rebuilt clean afterwards and confirmed 400 passed. Task 7.5 adds no
code guard or oracle of its own, so these mutations probe the oracles the page cites. Three survived (P3-2).

## Findings

### P2-1: the summary is stale at the current integration tip and would merge with a silent contradiction

- Where: `docs/rules-parity-coverage.md:234-235`, `:141-143`, `:246-265` and `:163-176`. Also
  `cookbook/references/kaggriculture-parity-summary-maps-tested-and-untested-layers.md:56-59,109-111`.
- Problem: the integration branch `kg/isaiah-gap-closure` moved to `994818b` at 23:36 on 2026-09-29.
  That is before `a8afb33` (23:54) and `81d0bf7` (00:08). `994818b` contains the Task 7.1 merge
  (`b6cd4f2`), and its phase tracker marks 7.1 as "merged". It also adds `### Task 7.1 Opponents: Snapshot
  View and Original-Submission Parity` to this same doc, plus `opponents_rs/tests/oracle_parity.rs` and
  279 lines in the cited `tests/scripts/test_kaggriculture_parity.py`. `git merge-tree` shows the doc
  auto-merges with no conflict. In the merged tree, line 234 still says "Task 7.1 opponents and their oracle
  in this integration. The recorded approval is on unmerged `kg/rebuild-7-1`", while line 273 documents
  Task 7.1's parity.
- Further stale points in the merged tree:
  - the "What is tested" table has no 7.1 row;
  - "Current checks" presents 2,319 passed / 10 skipped as current, while the merged page (lines ~735-739)
    reports Python at 2,400 passed / 21 skipped after 7.1;
  - "no Rust source … changed between `7f797a3` and this HEAD" is false once `opponents_rs` is in.
- The page's own reopening condition ("Update it when Task 7.1 … merges") had already fired before the summary
  was committed. The spec requires every claim to trace to the integration tip. The per-branch scoping to
  `bde3374` is honest, but the branch cannot land as-is.
- Fix: merge `994818b` into `kg/rebuild-7-5` (or rebase onto it) and refresh the summary.
  - Add a Task 7.1 row: oracle = original-submission Python actions on Kaggle 1.32.7; denominator =
    8 games, 4 bots × 2 seats × 1,438 = 11,504 compared actions; tests = `opponents_rs/tests/oracle_parity.rs`
    and `lifecycle.rs`; receipts; link to the 7.1 section.
  - Delete the 7.1 "not tested" bullet, or narrow it to what 7.1 does not cover.
  - Update the header HEAD and the "Current checks" table and prose (rerun targeted shards plus
    `opponents_rs` cargo test).
  - Resolve the cookbook log/index conflicts.
  - Re-verify.

### P3-1: the 1.94 GiB figure has no receipt at the tip

- Where: `docs/rules-parity-coverage.md:257-258` ("`tests/kaggriculture` alone later peaked at 1.94 GiB RSS
  in a separate scratch run").
- Problem: the only source is the untracked `codex/claude-verify-7.5-r1.md` in the main checkout. The page
  cites no path for it, which breaks the "trace to a receipt at the tip" rule for this sentence.
- Fix: cite the tracked receipt `7.5/pytest.json`. It records a sampled process-tree peak of 989,744 KiB
  after 21.6 s, with the stop reason "960 MiB aggregate RSS limit", before the selection finished. That
  already supports "probably exceeded 1 GB". Alternatively, name the local review file as the source.

### P3-2: the replay comparator's reward, done and terminal-bank checks have no negative control (M1-M3 survived)

- Where: `engine_rs/tests/replay_parity.rs:548,553,574`, and the page claim at `docs/rules-parity-coverage.md:178-180`.
- Problem: disabling any of the three checks leaves all 19 replay tests green. My probes show the checks are
  live today, so the parity claim itself holds. But a regression that disables them would go unnoticed.
  Existing negative controls cover state values, key order, actions, rejection labels and the rollback
  helper, but not these three. This gap predates Task 7.5 (it comes from Task 1.1/1.1b), and the page does not state it.
- Fix: add three tests beside `generated_trace_perturbed_state_is_rejected`. Each one uses `edit_line`
  on `gen-edge-vs-edge` and asserts the divergence kind:
  - `rewards` at step 100 set to `[1, 0]` asserts `rewards`;
  - header `terminal_banks[0]` +1 asserts `terminal_banks`;
  - header `transitions` +1 asserts `done`.

  All three pass on the unmutated code and catch M1-M3. Otherwise, list the gap under "What is not tested".

### P3-3: the note's gap sentence contradicts the updated tracker row

- Where: note line 111 ("The tracker's 7.5 row asks for a closing pass after 7.1, 7.3 and 7.4").
- Problem: `81d0bf7` changed the tracker row to "after 7.1, 7.3, 7.4 and 3.1".
- Fix: sync the wording. Also update note lines 56-59 when P2-1 is refreshed.

### P3-4: a workflow policy is stated in the coverage doc

- Where: `docs/rules-parity-coverage.md:258-259` ("Future checks use targeted pytest shards instead of an
  unguarded full run").
- Problem: this is a process rule inside the parity system of record. Unscoped, it reads as overriding
  CLAUDE.md's "Run `just py-prepare` after any python code edits" / `just prepare` before a PR.
- Fix: scope it ("Mac-limited review checks in this rebuild use targeted shards"), or move it to `ops/`.

## Owner-rule check

The change is docs, receipts and cookbook only. It adds no trainer, model or v2 code, and no policy
input or state. The page keeps `scripts/run_ppo.py` as the one canonical trainer, stopping at the Task 3.1 seam.
It mentions no deadline or submission timing.

## Cleanup

`git -C /Users/poonszesen/kg-v3-t75 worktree remove --force /tmp/cv-7.5-r1` exited 0, and the path is gone.
Before removal, `git status --short` in the scratch copy was empty. `/Users/poonszesen/kg-v3-t75` is still at
`81d0bf7` with 0 tracked modifications (`git status --short --untracked-files=no` is empty). There was no push,
commit or branch change. I did not touch `kg-v3-pod6000`, the pod branches or `~/kaggriculture-v2`.

VERDICT: REQUEST CHANGES
