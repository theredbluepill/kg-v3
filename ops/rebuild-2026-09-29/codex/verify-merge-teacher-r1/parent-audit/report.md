# Parent retention and documentation audit

Reviewed `a424d8cd20ad2734c5cd8159664d7c4be683d923` with parents
`b8747b6e8acece5f561d09a75bb914364a60ac05` and
`8fde43cd7408c9c4f9147eeb8f916bd08f8266ed`. Their common ancestor is
`4cac1a18f2209c54d40bef80d44755334a1a1ed2`.

## Finding

**P3 — cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:150:**
The current Limits still say T18 waits for Task 3.2, T19b waits for the
`kg/rebuild-configs` merge, and Task 4.4 waits on that same merge. Lines 130 and
141 repeat the obsolete skip dependency in present tense. Line 151 correctly
adds the post-merge status, and the description/index are current, but the old
current claims were not reconciled in place. This leaves contradictory reopen
conditions in the durable Reference. The governing cookbook contract explicitly
requires replacing superseded current prose instead of appending progress.

Fix: rewrite the old current statements to name the remaining trainer action-mask
mapping and run_ppo/native-env seams, state configs and value guards have landed,
and state 4.4 is undone with its prerequisite satisfied. Preserve historical
verification receipts as history rather than as current waiting conditions.

No functional merge-resolution issue or lost parent work was found in this audit.

## Test and path retention

The inventories count source test definitions, not parametrized pytest cases.
Python names were extracted with AST from tracked `tests/**/*.py` at each ref.
Rust names were extracted from `#[test]` attributes; the compiled-source counts
exclude historical `.rs` artifacts in `ops/`.

| Inventory | Integration | Teacher | Merge |
| --- | ---: | ---: | ---: |
| Python test definitions | 893 | 797 | 926 |
| Rust test definitions in `src/` + `engine_rs/` | 327 | 226 | 327 |
| Rust test-bearing source files | 14 | 10 | 14 |
| Cookbook log entry titles | 100 | 75 | 105 |

- No tracked path from either direct parent is absent from the merge.
- All integration Python test names are retained. All but the value-CE test have
  identical ASTs; that test preserves its assertion while calling the new
  model-owned method instead of the removed PPO free function.
- Exactly one teacher Python test name is absent:
  `tests/tools/test_check_engine_trim.py::test_task_authored_inventory_requires_exact_replay_test`.
  The integration had already renamed it to
  `test_task_authored_inventory_requires_exact_authored_set`, added the missing
  generated-manifest case and retired-bridge rejection, and expanded independent
  authored-set mutation coverage. This is preserved integration work, not loss.
- Teacher test AST changes otherwise consist of retained integration compile/
  evaluation tests and the two renamed skip-reason constants in the teacher
  launch/resume decorators. `test_teacher.py` differs from the teacher parent
  only in the skip-reason strings/constant name; test bodies are unchanged.
- All integration Rust source test files are byte-identical to the merge; the
  whole `git diff b8747b6...HEAD -- engine_rs src` is empty. No teacher Rust test
  name is lost. `src/rules_engine/generation.rs` preserves the integration's
  arbitrary-precision fixture deserializer repair and two added tests.
- All log entries from both parents survive with identical bodies after trimming
  section-boundary whitespace. The merge adds one entry; 105 titles are unique.

Machine-readable inventories and differences are in `inventory.json`,
`summary.json`, and `log-entry-content.json`.

## Resolutions and documentation

- `scripts/run_ppo.py` retains both `terminal_seat_banks` (used by Kaggriculture
  raw-bank evaluation) and `KaggricultureObsConfig`. Teacher obs-spec dispatch
  preserves Orbit's max-entities adjustment, requires Kaggriculture equality,
  and rejects cross-game pairs. The remaining env/trainer seams are explicitly
  disclosed rather than mistaken for Phase 4 completion.
- `docs/rl-api-specs.md` retains the teacher-target contract, the integration's
  longer Environment bullet (draws, bootstrap, evaluation seed-band scope), and
  both native observation and grammar sections. The removed shorter Environment
  bullet added no distinct semantics.
- The references index preserves both parents' topics, keeps the integration's
  current model-registration statement, and updates the teacher summary to the
  actual missing seams.
- The new coverage totals agree with the committed `merge-teacher/prepare.log`;
  root verification is independently re-running the required commands.
- The unskipped probe records all four intended failures at the disclosed seams.
  Phase 4.4 and integrated T18/T19b completion remain explicitly open. Brief
  section 6 permits review of 4.1–4.3 while those checks are dependency-skipped.
- `git diff --check b8747b6...HEAD` reports trailing whitespace only in retained
  failure logs under `ops/`; no code/doc whitespace finding is raised from those
  historical raw receipts.

This audit made no tracked edit. Only this untracked receipt directory was written.
