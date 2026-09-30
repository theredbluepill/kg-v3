# Independent merge verification — round 3, delta only

Verified `994818b87041426c6fc442a85fb06932937d58a7`, parents
`cb7006b49d72aa2de39ecda28818e476e3c25446` and
`5b43062fdab6c63d20d3f92f1e51b93b1b025b2c`. Merge base: `faed717`.
Round 2's approval and the approved adapter-parent work are scope boundaries.

## Findings

No new P0–P3 finding. No fix required. The literal Python-name-union expectation
has two inherited, already approved exceptions described below; these are not
merge regressions. No training or GPU work ran.

## Test preservation

Collections executed from exact scratch parent/HEAD checkouts with independently
built native modules, CPython 3.12.13, OMP_NUM_THREADS=2 and CARGO_BUILD_JOBS=2.
Rust counts include ignored tests; parent2 has no opponents crate.

| Collected inventory | cb7006b | 5b43062 | HEAD | Parent union |
|---|---:|---:|---:|---:|
| Python IDs | 1,793 | 2,329 | 2,421 | 2,423 |
| Root Rust names | 258 | 279 | 279 | 279 |
| Engine Rust names | 69 | 69 | 69 | 69 |
| Opponent Rust names | 22 | 0 | 22 | 22 |

All Rust inventories are exact parent unions. Python has no extra IDs and omits
only these two old names, both already replaced by approved parent `5b43062`:

- `tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native`
  → `test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture` (HEAD line 3010).
  The Orbit assertions survive; the obsolete no-native-environment expectation is
  replaced by an actual Kaggriculture construction assertion.
- `tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration`
  → `test_no_authored_grammar_path_include_after_root_engine_edge` (HEAD line 362).
  The absent-engine-bridge assertion survives and the root imports, no path include
  and absence of authored grammar in the trim manifest are checked more strongly.

`git diff cb7006b 5b43062` shows the substitutions predate this merge, and
`cookbook/log.md:9` records the previous verifier's acceptance. There is no
unexplained coverage loss. Raw lists and comparisons: `head-inventory.json`,
`parent*-pytest-ids.json`, `parent-rust-summary.json`, `parent-head-union.json`,
and their collection logs.

## Merge resolutions and documentation

- No parent path is lost: parent trees contain 4,133 / 6,073 entries; HEAD has
  6,250. Every one-parent-only content change is retained verbatim except the two
  deliberate hook-wording updates in the opponent Reference and skipped tests.
  The only paths changed by both parents are log, References index, coverage and
  tracker. See `tree-preservation.json`.
- Log sections: 130 / 137 parent headings become their exact 142-heading union;
  zero duplicates; dates descend and each parent's relative heading order survives.
  The first three are the revised 7.1 landing, 1.4/1.5 verifier edits, then 1.4/1.5
  landing. All 137 second-parent sections survive unchanged (outer whitespace
  ignored); only the first parent's top 7.1 entry is reconciled.
- Coverage retains all eight section headings. The changed kernel/opponent and
  grammar-count sections preserve the adapter coverage, then the opponent landing
  and re-merged counts. Other sections survive from the corresponding parent.
- Tracker retains all 13 sections, the second-parent header and Phase 6 content,
  first-parent Phase 7 with corrected opponent hook wording, and the 7.1 update
  immediately after the 1.4/1.5 update. Ancestry checks support the changed landing
  claims. Of 89 commit IDs in merged-status rows, 87 are HEAD ancestors; the two
  exceptions explicitly describe other unmerged branches and correctly contain
  `1e63597` rather than `b6b722f`. Snapshot cells untouched by the merge remain
  scoped to their dated snapshot.
- References index lists all 22 Reference notes exactly once. Its 23rd link is a
  deliberate Decision link. No duplicated or missing Reference.
- `python/owl/rs.pyi` and `src/kaggriculture/{bindings,env}.rs` expose observation,
  reset, step, truncate, metrics, snapshots and seed state, with no opponent seat
  setup. The root crate has no `opponents_rs` dependency. The new skip reason is
  true; the nine deferred tests do not become executable merely because 1.4 landed.
  Current opponent Reference, coverage and tracker agree. Old binding statements
  in cookbook/log.md are historical records, not current blockers. No current
  Task 7.1 blocker in cookbook/, docs/ or phase-status still claims the 1.4 binding
  itself is missing.
- `cookbook/log.md:5` and `docs/rules-parity-coverage.md:594–608` match the submitted
  prepare receipt. Independent scratch counts reproduce it.

Full section and ancestry evidence: `docs-audit.json`.

## Mutation and trim checks

`check_binding_skip.py` asserts that the native-semantics Reference says no test
carries the old skip, then runs `rg -n -F 'needs Task 1.4 binding'` over tests.
Baseline exits 0. In the scratch opponent test, replacing the constant's value
with that literal exits 1 and names the contradiction at line 15. Restoring the
exact original bytes returns exit 0. Restored SHA-256:
`957138ece4aa87862ff447be533811a6a8f0bd2ca9af1707e83550c161d04160`.
This is a documentation/skip contradiction check, not a claim of runtime opponent
seat coverage. Evidence: `skip-{baseline,mutated,restored}.log`.

`check_engine_trim.py` passes before and after running the 7.1 updater on scratch.
The updater preserves the manifest byte-for-byte, SHA-256
`e460abbb56e4c25239ad3e4b2b2d0f69ac1f2f0c904314c64c9d75f366323927`.
The initial system-python attempt lacked tomllib; rerunning with the project's
CPython 3.12 interpreter passed, without any repository change.
Evidence: `mutation-and-trim.json` and `trim-*.log`.

## Full preparation and custody

Command in the scratch HEAD clone:

```sh
OMP_NUM_THREADS=2 CARGO_BUILD_JOBS=2 UV_OFFLINE=1 uvx --from rust-just just prepare
```

Exit 0. This includes native build, engine/opponent custody, formatting, Clippy,
Python 3.11 syntax check, Ruff, Markdown lint, mypy, all Rust/Python tests and doc
freshness. No DOCS_CURRENT override. Scratch git status is empty afterwards.

| Check | Actual result |
|---|---|
| Engine Rust | 69 passed (41 + 9 + 19) |
| Root Rust | 274 passed, 5 ignored |
| Opponent Rust | 22 passed (12 + 5 + 5) |
| Python | 2,400 passed, 21 skipped, 75.34 seconds |
| Mypy | 71 source files, no issues |
| Engine trim / opponent custody | Pass |
| Documentation freshness | Pass |
| Skip mutation | Pass → detected failure → restored pass |
| Staging worktree | HEAD unchanged; status empty; 6,249 file-content hashes unchanged; both symlinks match Git |

The 21 skips remain explicit in `prepare.log`, including nine opponent-seat tests,
trainer seams, unavailable CUDA/quantized backends, Mac-bounded live regeneration
and unavailable sibling original sources. Existing local replay fixture bytes
were copied into scratch; no oracle regeneration or strength qualification is
claimed. Original worktrees were not modified. All review artifacts and builds
live under this /private/tmp directory. See `prepare-result.json`, `prepare.log`
and `source-final-check.json`.

VERDICT: APPROVE
