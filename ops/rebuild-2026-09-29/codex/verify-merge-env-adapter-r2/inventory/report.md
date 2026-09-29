# Independent r2 parent test inventory

Compared actual pytest collection and actual Cargo test lists from separate git-archive snapshots of BASE `faed71773fa9f6414e4379ace904780349979cd8`, adapter `8699ca9eab63d0dd3d951fa9cb58e65b1f1c650a`, and HEAD `bc953e98fd70e08c5c87d67a210f8a970d186b6b`. All credited commands exited 0. CPU limits were `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`; commands ran sequentially. No training or GPU execution occurred.

| Revision | Pytest nodes | Engine Rust tests | Root Rust tests, including ignored |
| --- | ---: | ---: | ---: |
| BASE | 1,701 | 69 | 258 |
| Adapter | 2,253 | 69 | 279 |
| HEAD | 2,329 | 69 | 279 |

Every adapter pytest node and every Rust test name from both parents survives at HEAD. The literal pytest parent union contains 2,331 nodes; exactly two BASE names are absent, both deliberately replaced in the adapter before this merge:

1. `tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native` becomes `test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture` (HEAD line 3010). The Orbit constructor argument assertions at lines 3027–3037 survive. The obsolete no-native exception expectation becomes real native environment construction, type, pinning, and factory-isolation assertions at lines 3038–3046. Additional factory-argument, independent-env, and deterministic-replay tests cover the new path. The replacement is identical between adapter and HEAD; HEAD restores the integration's 8-rank startup parametrization as an additional case.
2. `tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration` becomes `test_no_authored_grammar_path_include_after_root_engine_edge` (HEAD line 362). Engine bridge absence is retained. Reading the root test file retains its existence requirement while adding grammar/engine import, forbidden path include, and manifest assertions at lines 364–369. The replacement is identical between adapter and HEAD.

Source-level checks also found no hidden removal: Python AST test declaration counts are BASE 930, adapter 1,005, HEAD 1,045, with exactly the same two deliberate BASE replacements and zero duplicate declarations. The `#[test]` Rust declaration scan counts BASE 327, adapter 348, HEAD 348, with no parent declaration lost or duplicated. This supplementary Rust scan covers ordinary attributed test functions; the actual Cargo inventories remain the authoritative executable test lists. Every tracked path beneath `tests/` from either parent exists at HEAD.

Reviewed source diffs also preserve the relevant existing assertions: config tests adopt required reward fields and the native config round trip; the native-table test removes its old skip fallback; the adapter's model-head test helpers are moved into the integration's shared test helper module without removing test bodies. The integration adds observation-custody cases and routes the retained loss assertion through the model method. Saved diffs provide the exact changes.

Evidence and custody:

- `manifest.json` records revisions, scratch roots, exact commands, environment, durations, exit codes, and native-module provenance. `collect.py` and `recollect_fixed_native.py` reproduce the collections.
- Each Rust revision used an initially absent, distinct `CARGO_TARGET_DIR`, preventing the shared-target stale-binary issue found in r1. `PYO3_PYTHON` was explicitly bound to the existing interpreter. Final Rust logs are `*-cargo-engine-list.log` and `*-cargo-root-list.log`.
- Each Python collection sets `PYTHONPATH` to its own archived `python/`; `*-import-path-fixed-native.log` proves the loaded Python source and native module paths. The existing HEAD native module is used only for collection; no parent native behavioral execution is claimed.
- A concurrent verifier rebuilt the HEAD native extension during initial collection, changing its hash from `a4ca6a6f…` to `f165fc43da61cedb2ea4d06ee60ec2d3496dd53db498813b8e3c8eea78526efe`. Therefore the initial Python logs are retained but not credited for exact native identity. All three Python collections were repeated against a frozen scratch copy of the rebuilt module. That exact hash matches before and after the rerun; only `*-pytest-collect-fixed-native.log` is credited above.
- Archive extraction omits symlinks: `CLAUDE.md`, and (BASE/HEAD only) one historical ops receipt's absolute `scripts` link. No production source or test is omitted.
- `results.json` contains counts and every set difference; `*-names.txt` contains normalized inventories (Rust identities include the Cargo test binary). `sha256.json` hashes receipt files.

No test-retention finding. No tracked repository file was edited by this inventory task; all authored receipts are under the authorized r2 evidence directory and all snapshots/build outputs are under `/private/tmp`.
