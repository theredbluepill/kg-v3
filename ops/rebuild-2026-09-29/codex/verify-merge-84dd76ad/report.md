# Independent verification of merge 84dd76ad

Verified merge `84dd76ad570bf35b0947df52f3697f80339098e6` on `kg/merge-grammar`, with parents `e1458d2a717d9d731a367cbb78b98616ee6649f4` and `7877c46e39687e4332fe3ad5fc65d3c76feb2125`. The tracked tree was clean before verification. No tracked implementation or documentation was changed. These verification receipts are untracked working artifacts.

Target/stopping condition: independently establish that the merge preserves both parents' contracts, test surfaces and vendored custody; reproduce the committed manifest twice; require the specified suites to pass and deliberate scratch mutations to fail. No training, GPU or performance claim is made.

## Finding — P3: update the remaining heads-documentation handoff

`cookbook/references/index.md:23`, `cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md:4` and `docs/model-architecture.md:752` still describe the Python stand-in as lasting “until Task 1.2.” Task 1.2's native Rust tables now exist and the merged native Reference correctly identifies the absent Task 1.4 PyO3 binding as the remaining dependency. The index explicitly describes the current tree, so these descriptions are stale after this merge.

Update the heads Reference description/current limit prose and matching index/model documentation to say that Task 1.2 supplies the Rust tables and Task 1.4 still owns the binding; distinguish historical Task 2.3 test results from the current remaining dependency. The native equality test remains correctly skipped, and no production behavior defect was found. Leave historical test totals scoped to their branch. Per the repository contract, record the documentation correction in the log when implementing it. This verification does not apply edits, as requested.

## Conflict and contract preservation

The checker differs from the integration parent only in `verify_task_authored`: its fixed set is exactly replay_parity.rs, grammar_kernel.rs and GENERATED_MANIFEST, with the Task 1.2 label. All Task 1.1 hardening and Task 1.1b validation code is unchanged: EDITABLE, exact source/Cargo changes, Rayon closure derivation, provenance appendix hash, generated splitting, schema/TRACE_KEYS, engine pin, each trace hash/size, complete inventory, 4,000,000-byte budget and at least six traces.

Checker tests preserve both parents' suites. The accept/exact-set tests were renamed and combined to cover all three paths; the exact-set parametrization has the five claimed cases. Task 1.2's omitted-file, self-declared-file and wrong-hash attacks remain; Task 1.1b's generated-manifest/check-entry-point attacks remain.

## Manifest reproduction and custody

Executed the committed `ops/rebuild-2026-09-29/merge-1.2/update_trim_manifest.py` against this merged tree twice. Before, after first execution and after second execution, SHA-256 was exactly:

`750b3473ab4fba19dcf6263a44c662c4cf06d07c3e3dc74ef89e2b6d5be992f7`

Both outputs are byte-identical to the committed manifest; no restoration was necessary. A finally guard would restore the original bytes on disagreement. The updater file itself matches its committed blob. Reproduction establishes the committed tool's exact output; it cannot establish how the historical author originally produced the file.

All retained/excluded manifest sections are byte-identical across the merge base, both parents and merge. All 12 retained files are byte-identical across those revisions and the worktree. All 113 excluded paths remain absent. The merged manifest has 3 authored entries and 28 non-engine entries; all authored hashes match current bytes, 13 non-engine paths are new versus integration, and the five combined reasons are exactly the declared shared paths. All other reasons select the changed parent. Four scratch input controls independently reject schema, reference-pin, retained or excluded drift before writing output.

Evidence: `manifest_check.py`, `manifest-check.json`, `updater_controls.py`, `updater-controls.json`.

## Test preservation and documentation

Every tracked Python/Rust source was scanned from both parent trees and the merge. Python test ASTs were compared; Rust test names and test-bearing source blobs were compared. The only missing names are the two authored-inventory tests from each parent that were deliberately merged/renamed. No test is dropped. Other existing Python tests are AST-identical to their parent, except the model-config test extension already present on integration. Integration's Rust test-bearing files are all byte-identical; Task 1.2's changed replay test is exactly integration's version.

| Test definitions (not parametrized runtime cases) | Integration | Task 1.2 | Merge |
| --- | ---: | ---: | ---: |
| Python, excluding ops | 746 | 640 | 755 |
| Rust, excluding ops | 226 | 234 | 244 |
| Rust historical ops copies | 7 | 8 | 8 |

The shared grammar's nine tests compile in both Rust packages, so definition counts are not runtime totals. Python parsing had no errors, and the Rust scan found no unfamiliar alternative test annotations. Evidence: `inventory-docs/`.

Coverage preserves Task 1.1 kernel coverage, Task 1.1b scope/live-parity/divergence sections, and Task 1.2 grammar coverage. It correctly replaces the stale “no fresh differential run” statement and reconciles 59 → 69 → 87 engine tests and 155 → 164 root passes plus two ignores. Historical Task 1.2 totals (77 engine / 164 root / 1,045 Python) remain branch-scoped; merged totals (87 / 164 / 1,337 plus four Python skips) match committed prepare logs. Root/full-Python/prepare were not rerun during this verification. Tools total 106 is 79 checker + 25 recorder + 2 replay-viewer cases.

The log contains exactly 71 unique headings, preserving every heading and relative ordering from both parents: new merge entry, two Task 1.2 entries, then all 68 integration entries. Evidence: `log-order.json`.

A fresh scratch Rust build emitted all eight native tables; `grammar_tables_from_arrays` accepted them and every one of 964 bits matched `expected_grammar_tables`. The emitted data also matches the committed cross-check receipt. Evidence: `crosscheck_tables.py`, `native-tables.json`, `native-tables-comparison.json`. This does not qualify a PyO3 binding or GPU behavior.

## Independent non-vacuity checks

All mutations were made only in a scratch copy; scratch sources were restored byte-for-byte.

| Source mutation | Baseline | Mutated result |
| --- | --- | --- |
| Revert EDITABLE restriction | 79 checker pass | 1 fail, 78 pass |
| Remove generated trace hash check | 79 checker pass | 2 fail, 77 pass |
| Remove grammar authored registration | 79 checker pass | 10 fail, 69 pass |
| Remove generated MANIFEST authored registration | 79 checker pass | 9 fail, 70 pass |
| Remove coupled-HIRE capacity guard | 18 grammar/kernel pass | 5 fail, 13 pass |

Failures include the intended LICENSE declaration attack, both synthetic/check-level edited-trace controls, authored exact-set admission/omission tests, and HIRE fixture oracle/distribution/encoding/dense-capacity checks. Evidence: `mutations/` contains scripts, exact diffs, baseline logs, mutation logs and result summaries. The four updater controls above supplement these source mutations.

## Required runs

| Command | Result |
| --- | --- |
| `CARGO_BUILD_JOBS=2 cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | Exit 0; 87 passed (41 unit + 18 grammar/kernel + 9 RNG + 19 replay); 0 failed/ignored |
| `uv run python scripts/check_engine_trim.py` | Exit 0; engine trim manifest: OK |
| `OMP_NUM_THREADS=2 uv run pytest tests/tools tests/scripts tests/kaggriculture -q` | Exit 0; 550 passed, 1 skipped |

The single skip is `test_native_tables_match_expected_tables`: no native grammar binding exists yet. Logs: `engine-cargo.log`, `checker.log`, `pytest.log`.

Final tracked cleanliness is captured separately in `final-status.txt`; the required `git status --porcelain --untracked-files=no` output is empty.

VERDICT: APPROVE WITH EDITS
