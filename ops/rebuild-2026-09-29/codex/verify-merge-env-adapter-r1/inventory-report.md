# Test inventory verification

Actual pytest collection (`python -m pytest --collect-only -q tests`) and actual Cargo lists (`cargo test [--manifest-path engine_rs/Cargo.toml] --locked --offline -- --list`) were run from git-archive snapshots of BASE 666deec, adapter 8699ca9, and HEAD 49a4835. Commands, exit codes, elapsed times and archive locations are in `inventory-manifest.json`. All final commands exit 0. CPU limits: `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`.

| Revision | Pytest nodes | Engine Rust tests | Root Rust tests (including ignored) |
| --- | ---: | ---: | ---: |
| base | 1701 | 69 | 258 |
| adapter | 2253 | 69 | 279 |
| head | 2329 | 69 | 279 |

Every adapter-parent pytest node is in HEAD. Two BASE literal pytest names are absent, both intentionally replaced in the adapter parent, with their replacement present unchanged in HEAD:

1. `tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native` → `test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture` (HEAD line 3010, adapter commit d0d65f7). Orbit constructor-argument assertions remain; the obsolete no-native-environment assertion becomes actual native environment construction, type and pinning assertions. Additional factory/replay/independence tests cover the new path.
2. `tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration` → `test_no_authored_grammar_path_include_after_root_engine_edge` (HEAD line 362, adapter commit 8d98ea8). Engine bridge absence remains; reading the root test preserves the old file-existence requirement and adds assertions for explicit grammar/engine imports, absence of path includes, and no authored grammar manifest entry.

Thus the literal Python parent union has 2,331 nodes, with these two renamed nodes absent; no uncovered test deletion was found. A separate AST declaration scan confirms exactly the same two BASE-name changes and zero missing adapter declarations (BASE 930 declarations, adapter 1005, HEAD 1045).

All engine and root Rust names from both parents are retained at HEAD. Engine union = 69; root union = 279. Names are qualified by Cargo test binary target for set comparisons. Normalized lists and exact set differences are in `inventory-*-names.txt` and `inventory-results.json`.

Collection custody and limits:

- Each Python subprocess sets `PYTHONPATH` to its own snapshot's `python/`; `inventory-*-import-path.log` proves the imported package path. The existing HEAD native extension was symlinked into each snapshot solely for collection, with its then-current SHA-256 recorded in the manifest. No parent behavioral execution is claimed from this shared extension.
- Archives omit symlinks, which are only `CLAUDE.md` and a historical ops receipt's absolute `scripts` link. Production files and tests are retained.
- Initial root Cargo lists failed because snapshot `.cargo/config.toml` referenced absent snapshot `.venv/bin/python`. The retry sets `PYO3_PYTHON` to the existing interpreter.
- The first retry reused a shared Cargo target and reported the BASE binary unchanged for adapter and HEAD. Those shared-target logs are INVALID and not credited. The final `*-cargo-*-list-isolated.log` runs each use an initially absent, separate per-revision target directory and compile that revision afresh. Their final counts alone appear above.

No tracked file was edited by this inventory task. All receipts are beneath the authorized untracked evidence directory; archives and build outputs are beneath `/private/tmp`.
