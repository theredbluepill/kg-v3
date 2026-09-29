Independent verification of `kg/merge-env-adapter`, HEAD `49a48350cdd72d6b546320441d677fbcbae2d893`, against BASE `666deec789b2e75f56afb6dbb7b7acd40171c6b1` and adapter parent `8699ca9eab63d0dd3d951fa9cb58e65b1f1c650a`. Merge `e5db8cafa70c00dceac8f276d818e030de6f304f` has exactly those parents; their merge base is `e197528`. Reconciliation is `49a4835`. No production merge defect or unexplained coverage loss found. Two documentation findings require edits.

The bounded verification question was whether both parents' behavior and test coverage survived, the strict reward schema admits the integration's 8-rank config, and startup checks still precede the explicit Task 3.1 stop and allocation. Inputs were the three committed trees, their diffs, actual test collection, requested CPU suites and six scratch mutations. Expected discriminators were retained test names/assertions, passing current checks and failing guard-removal mutants with green restoration. Stop condition: complete these comparisons and checks, account for all missing names, and confirm no tracked changes. No training or GPU run was performed. Commands used `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`.

All five claimed reconciliation groups were checked:

1. The log has 131 headings versus BASE's 120 and adapter's 109: every parent section survives, dates are nonincreasing, headings are unique, and the new merge entry precedes integration-only then adapter-only entries. The References index retains every one of the 21 union targets; the requested integration and adapter lines are retained. The configs Reference has the exact 30-source parent union, all three ranked shapes, the Task 3.1 stop reason and native evaluation wording. Model docs retain 8-rank chunk calls and the adapter guard sentence. RL API docs retain Teacher targets, Environment and the full adapter section; content from the adapter section through EOF is byte-identical to the adapter parent. No conflict markers were found in README/configs/Python/scripts/Rust/docs/cookbook/tests.
2. The 8-rank config's only executable change from BASE is `econ_ineffective_cap: 0.1`. All four configs load through real `FullConfig`, supply all six required reward coefficients and have terminal scale .75. The 8-rank per-rank env/minibatch shape is 32/2; its global workload and startup tests pass. Removing the cap fails both those checks.
3. `NEEDS_RUN_PPO_GAME_SEAM` correctly names Task 3.1 rollout storage/action mapping, while acknowledging the existing native env. The actual skipped-test output agrees. Related current prose was not fully reconciled; see findings.
4. Both parity-coverage native-table statements now say wiring exists; the new count sentence agrees with independently rerun tests: engine 69, root 274 passed/5 ignored, Python 2,319 passed/10 skipped.
5. `engine_rs/TRIM_MANIFEST.json` is identical in both parents and HEAD: git blob `8c1b7734a4a99f8e70ceaacf17d0f86f1ae24e5a`, SHA-256 `b2d1892a15baf5653e493fdb885c5f2baac7a21095a87eb95d9730782ad9272f`. The trim checker passes.

The non-conflicted files named in the request were compared against both parents. README/configs keep the integration's ranked recipe and adapter's native boundary. `gpu_grammar.py` combines table digest identity with strict native admission. `model/kaggriculture.py` combines native default tables with teacher targets, chunk slicing/joining and `_run_trunk` backend checks. `run_ppo.py` keeps teacher dispatch and the native evaluation factory. Startup order is final runtime config → workload check → compile-stack check → explicit Task 3.1 stop → Orbit narrowing → allocation. Native policy evaluation stops before the remaining Orbit mapper. Config/startup tests preserve both contributions. Parity coverage preserves observation custody, teacher/8-rank and native lifecycle evidence. Detailed read-only audits are in `semantic-audit.md` and `docs-audit.md`.

Actual collected test names:

| Suite | BASE | Adapter | HEAD | Unaccounted omissions |
| --- | ---: | ---: | ---: | ---: |
| Pytest | 1,701 | 2,253 | 2,329 | 0 |
| Engine Rust | 69 | 69 | 69 | 0 |
| Root Rust, including ignored | 258 | 279 | 279 | 0 |

The literal Python union has 2,331 names, so it is not a literal subset of HEAD. Exactly two BASE names were replaced on the adapter branch before this merge:

- `tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native` → `test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture` (`d0d65f7`). Orbit constructor assertions remain; native construction replaces the obsolete no-env expectation, with additional factory/replay/independence coverage.
- `tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration` → `test_no_authored_grammar_path_include_after_root_engine_edge` (`8d98ea8`). Bridge absence remains; reading the root test preserves existence checking and adds explicit-import/no-path-include/manifest checks.

Every adapter pytest name and every Rust name from either parent remains. Independent AST comparison found the same two Python replacements. `inventory-report.md`, `inventory-results.json` and `inventory-*-names.txt` provide the complete inventories. Parent Python collection uses each parent's Python source with the existing native extension solely for import/collection, not behavioral qualification. Final Cargo lists use fresh independent target directories per revision. Earlier shared-target lists reused an old binary and are explicitly invalidated in the inventory receipts; only isolated lists are credited.

Requested checks all pass:

| Command | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed |
| `cargo test --locked --offline` | 274 passed, 5 ignored |
| `uv run python scripts/check_engine_trim.py` | OK |
| `uv run pytest tests -m 'not slow' -q` | 2,319 passed, 10 skipped |
| `uv run mypy python/owl scripts` | No issues in 69 source files |
| `uv run python scripts/check_doc_freshness.py` | No doc updates required |

The ten Python skips are six CUDA/backend cases and four explicitly pending Task 3.1 integration cases. No sharding was needed. The freshness script inspects changes relative to HEAD, so its clean-worktree success does not itself validate the merge's prose; the parent-to-HEAD documentation audit supplies that check. `git diff --check 666deec HEAD -- ':!ops'` also passes.

An additional source-bound native rebuild succeeded with `uv run maturin develop --locked --offline --skip-install`. Plain `develop --locked --offline` first attempted a Python dependency-group installation and failed DNS; the build-only retry required no network or dependency change. Because the rebuilt extension hash differed, the full Python suite and all six mutations were repeated against rebuilt module SHA-256 `8da073cbee3b6243ca5b7365f2c8e9662b1074998548a693238c7f0e7935f8c0`; the suite again yielded 2,319 passed/10 skipped (`pytest-rebuilt-native.log`). Initial and final module hashes are preserved in `custody.json`. Environment: macOS arm64, Python 3.12.13, Torch 2.9.0, pinned Rust nightly; see `environment.json`. These are CPU correctness checks, not throughput or GPU qualification.

Six mutations were applied only to a HEAD archive under `/private/tmp`. Each baseline and restored test passed; each mutant exited 1 through its intended assertion/error mismatch:

| Mutation | Targeted failures |
| --- | --- |
| Remove 8-rank ineffective cap | 8-rank config loader/global workload and startup cases |
| Remove explicit Task 3.1 startup stop | Exact remaining-blocker startup case |
| Remove pre-stop workload check | Unserviceable-workload and 8-rank headroom cases |
| Remove pre-stop compile-stack check | Unprobed-stack rejection |
| Remove policy-evaluation mapping stop | Named remaining-mapper blocker |
| Restore expected-table model default | Native loader called exactly once |

No test asserts the 8-rank documentation sentence or the teacher skip-reason wording. Those text-only changes were inspected, not falsely claimed as mutation-covered. Full mutation argv, outputs, mutant hashes and restoration hashes are in `mutations.json` and `mutation-*.log`.

Before and restored SHA-256 values are identical, and match the untouched tracked worktree:

| File | Before = restored = worktree SHA-256 |
| --- | --- |
| `configs/kaggriculture_8rank.yaml` | `ec1fdd1621936b6c1ba776b1881d1426215e35b72206eb4d97c411f98094bf77` |
| `scripts/run_ppo.py` | `c99f32ab96d06252a95930c7f9bc6ed48c0c6736053d9a7e6223a0c7998fc574` |
| `python/owl/model/kaggriculture.py` | `5ac201ad7940963da720367af7918ea1d1a47704260d4efdef5af7e7ac506854` |

No tracked file was modified. Final `git diff HEAD` and staged diff are empty; `git status --porcelain` contains only:

```text
?? ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1/
```

Findings, with parent origin and fix:

- **P3 — Stale native-environment blockers.** `cookbook/references/kaggriculture-teacher-distills-per-slot-kl-and-per-seat-winner-ce.md:160` still names Task 1.4 as open; lines 134/145, its description at line 4 and `cookbook/references/index.md:7` retain the old blocker/“no-env stop” description. This prose comes from BASE unchanged and now contradicts the corrected skip reason. Related stale parent prose survives at `cookbook/references/kaggriculture-model-joins-isaiahs-factory-compile-and-masked-critic.md:61`, `configs/kaggriculture.yaml:7`, `configs/kaggriculture_2rank.yaml:12`, `configs/kaggriculture_4rank.yaml:12`, `configs/kaggriculture_8rank.yaml:13`, and the future-tense adapter ownership statement at `docs/rl-api-specs.md:1262`. The model note/first three config headers predate the merge in both parents; 8-rank comes from BASE; the API sentence comes from adapter. **Fix:** describe the existing native env/adapter and remaining Task 3.1 rollout/observation/action mapping; reconcile teacher description/index, preserve dated historical outcomes, and retain the pending CUDA DMA limitation.
- **P3 — Superseded reward schema and startup caller.** `cookbook/references/kaggriculture-configs-follow-isaiahs-scaling-6m-recipe.md:28` still places `KaggricultureRewardConfig` in `config.py`, claims defaults and describes raw-positive-weight admission. This line is identical in both parents and was already stale in adapter. The class is in `rewards.py`, all six fields are required, and admission uses positive binary64 products and enabled-cap rules. Line 30 retains the old `require_orbit_env` startup-stop attribution from parent prose, despite the new explicit guard and preceding compile-stack check. **Fix:** name the required reward definition and its actual validation (or link the reward Reference), distinguish shipped YAML values from defaults, and state the actual startup ordering. Keep earlier Task 3.4 results as history.

VERDICT: APPROVE WITH EDITS
