# Independent merge verification — 82091936

Target: `82091936df52125e7f4cce687887a8c259b7ee02`, checked at HEAD on `kg/merge-1.1b`.
Parents: `69397da3d1559be4fbb525636122d8a9bc614a49` and `16e56b60c4bd11da3bdacdd6a7e45e5637654ad6`.
Date: 2026-09-29. No tracked files modified. This is verification evidence, not a repository adaptation.

## Finding

P3 — `cookbook/log.md:7–10` reverses part of the checker sequence. It says the fixed Task 1.1 checks (including lockfile derivation and appendix hash) run before generated-manifest verification. Actual `verify_task` at `scripts/check_engine_trim.py:480–542` runs: parse, split generated traces, generic verification (including EDITABLE), generated-manifest verification and minimum count, then fixed retained/authored/justfile/lib/Cargo/lock/provenance checks. Correct the log's sequence. Both contracts execute, so this is documentation only and does not change acceptance/rejection.

No blocking or behavioral merge findings.

## Preservation review

Against the first parent, every checker definition/constant is retained; 31 top-level AST nodes are identical. Intentional changed functions are `verify_task_authored` (two authored paths), `verify_task` (generated split/verification), and `main` (wider caught exceptions). Against the second parent, every definition/constant is retained; 33 nodes are identical, with changes only to `verify` (EDITABLE strengthening) and `check` (delegation). Task 1.1's EDITABLE, RAYON_CLOSURE, appendix heading/hash, reference_files, thin check, verify_task, _expected_lock and _verify_provenance survive. Task 1.1b's constants, engine_pin, split_generated and verify_generated remain intact. The generated minimum count remains six; actual committed inventory is 15 traces.

All eight first-parent check-level regression functions and their supporting fixtures are unchanged. The two merged-path tests exercise trace-hash and unlisted-trace rejection through the full `check()` entry point. The eleven parametrized generated-manifest drift cases remain.

| Python test function definitions (not parametrized test cases) | First parent | Second parent | Merge |
| --- | ---: | ---: | ---: |
| tests/tools/test_check_engine_trim.py | 21 | 17 | 27 |
| tests/scripts/test_kaggriculture_parity.py | 0 | 17 | 18 |
| All files under tests/ | 682 | 601 | 706 |

The raw parent test-name union has 705 names. Four merge-only names and three absent names yield 706. Every absent name has an intentional replacement:

- `test_task_authored_inventory_accepts_replay_test` becomes `test_task_authored_inventory_accepts_replay_test_and_generated_manifest` for the two-path inventory.
- `test_retained_rust_cannot_be_edited_even_when_declared` becomes `test_retained_file_cannot_be_edited_even_when_declared` with the strengthened diagnostic.
- `test_generator_refuses_the_project_kaggle_environments` is replaced by `test_generator_refuses_before_writing_when_the_pin_differs` and `test_project_environment_satisfies_the_engine_pin`, because the merged project already pins 1.32.7. The negative check verifies no output directory is written.

The other merge-only names are the two full-check generated-trace regressions. Rust engine test-name counts are 59/69/69; none from either parent is missing. Replay-parity counts are 9/19/19.

The log retains all parent headings and entry bodies (59/50/63 headings, no duplicates); the references index retains all parent links. The coverage document retains the first-parent hardening paragraph and the complete live-parity section with the project-pin clarification. The live-parity Reference also reconciles that pin. The new log sequence wording above is the sole requested correction.

## Manifest and byte custody

`TRIM_MANIFEST.json` is byte-identical to the second parent; SHA-256:
`e507deba750e5bc64abaf3463b0d8e7f1aa13eba0db5fb017276905f2b1b6b5c`.

There is no merge-specific manifest edit requiring regeneration. For independent confirmation of the first-parent delta, ran the committed `ops/rebuild-2026-09-29/1.1b/update_trim_manifest.py` with redirected paths on a scratch copy starting with first-parent manifest bytes and merged authored files. Its output reproduced the merge byte-for-byte; a second run was idempotent. This establishes reproducibility, without claiming to observe the historical edit process. See `manifest_check.py` and `manifest-check.json`.

All 12 retained engine paths are byte-identical to the first parent, including the seven vendored source/Cargo/provenance/license paths. `replay_parity.rs` and generated `MANIFEST.json` are byte-identical to the second parent.

## Executed checks

| Command | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed: 41 unit, 9 RNG, 19 replay-parity; 0 failed, 0 ignored; 0 doc tests |
| `uv run python scripts/check_engine_trim.py` | Exit 0, `engine trim manifest: OK` |
| `uv run pytest tests/tools tests/scripts -q` | 261 passed, 0 skipped, 12.34 s |

Raw output: `cargo.log`, `checker.log`, `pytest.log`. No full prepare, new sweep, training or GPU run was performed in this verification.

## Independent scratch mutation checks

Used a `git archive` snapshot of the merge's checker, tests and engine. Git reference reads were directed to the repository object store, while GIT_WORK_TREE pointed at scratch. The original tracked checkout was never mutated. Ran the full checker test file with the project Python, without bytecode/cache reuse.

- Baseline: 71 passed.
- Task 1.1 mutation: replace `if path not in EDITABLE` with the old Rust-only guard. Result: 1 failed, 70 passed. `test_declared_license_edit_is_rejected` fails because the declared LICENSE edit is accepted.
- Task 1.1b mutation: remove the trace-byte SHA-256 equality requirement, retaining digest shape validation. Result: 2 failed, 69 passed. The same-size synthetic edited trace is accepted, failing the parametrized drift test; the full-check edited trace instead reaches its size check, failing its required trace-hash diagnostic.

See `mutation_check.py`, `mutation-summary.json` and the three mutation logs. Restored the scratch checker afterward. Scratch copies were removed after recording results; the scripts reconstruct them.

Final worktree check: HEAD remains the target; tracked index and worktree diffs are empty. Only this untracked verification evidence directory was added.

VERDICT: APPROVE WITH EDITS
