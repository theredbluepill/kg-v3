import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=root/'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r2'
shards=json.loads((out/'shards.json').read_text());assert len(shards)==2 and all(r['exit_code']==0 for r in shards)
assert '2266 passed, 10 skipped' in (out/'pytest-rest-shard.log').read_text()
assert '53 passed' in (out/'pytest-oracle-shard.log').read_text()
mutations=json.loads((out/'mutations.json').read_text());assert len(mutations['rows'])==6
hashes={}
for row in mutations['rows']:
 assert row['baseline']['returncode']==row['restored']['returncode']==0 and row['mutant']['returncode']==1
 assert row['before_sha256']==row['restored_sha256']==row['main_worktree_sha256']
 hashes[row['path']]=row['before_sha256']
checks=json.loads((out/'checks.json').read_text());assert all(r['exit_code']==0 for r in checks if r['name']!='pytest')
pre=json.loads((out/'tracked-before.json').read_text());post={}
for p in pre:
 f=root/p;post[p]=hashlib.sha256(f.read_bytes() if not f.is_symlink() else str(f.readlink()).encode()).hexdigest()
assert pre==post
status=subprocess.check_output(['git','status','--porcelain'],text=True)
assert all(l.startswith('?? ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r2/') for l in status.splitlines())
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
assert not subprocess.check_output(['git','diff','HEAD'],text=True)
assert not subprocess.check_output(['git','diff','--cached'],text=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='bc953e98fd70e08c5c87d67a210f8a970d186b6b'
(out/'status-final.txt').write_text(status)
(out/'tracked-final-check.json').write_text(json.dumps({'tracked_files_checked':len(pre),'hash_changes':[],'tracked_status':'','diff_HEAD':'','staged_diff':'','head':'bc953e98fd70e08c5c87d67a210f8a970d186b6b'},indent=2)+'\n')
report='''Independent r2 verification of `kg/merge-env-adapter-r2`, HEAD `bc953e98fd70e08c5c87d67a210f8a970d186b6b`, against BASE `faed71773fa9f6414e4379ace904780349979cd8` and adapter `8699ca9eab63d0dd3d951fa9cb58e65b1f1c650a`. Merge `7f797a3` has exactly those parents. No merge defect or unexplained coverage loss was found. Both r1 P3 findings are resolved.

The bounded question, inputs, expected discriminators and stopping condition are in `run-statement.md`. This re-verification used actual archived parent collections, source/tree comparisons, a rebuilt HEAD native extension, CPU suites, and scratch-only guard mutations. All work used `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2`; behavior checks additionally used `RUST_TEST_THREADS=2` and offline uv. No training or GPU run occurred. No tracked repository adaptation was made.

R1 findings:

| R1 finding | Status | Evidence in HEAD |
| --- | --- | --- |
| P3: stale native-environment blockers | **RESOLVED** | Teacher Reference lines 4, 134, 145, 160 and index line 27 acknowledge the native env/adapter and limit the remaining seam to Task 3.1. Model Reference line 61, all four config headers, and `docs/rl-api-specs.md:1262` agree. Historical outcomes and pending CUDA DMA qualification remain scoped. |
| P3: superseded reward schema and startup caller | **RESOLVED** | Configs Reference line 28 identifies `owl.kaggriculture.rewards.KaggricultureRewardConfig`, its six required finite/nonnegative coefficients, enabled-cap rules and positive binary64 product admission. Line 30 gives workload → compile-stack → explicit Task 3.1 stop → Orbit narrowing, matching `scripts/run_ppo.py:177–188`. |

The full note filenames, exact line evidence, side-by-side index descriptions and comparison receipts are in `docs-audit/report.md` and its JSON/text companions. The configs Reference retains the 30-source parent union and adds the reward-module source (31 total).

Parent preservation and merge resolution:

- All **1,730 paths** in `git diff --name-only 666deec faed717` are accounted for. **1,728 retain identical git mode/blob** in HEAD; only the intended `cookbook/log.md` and `cookbook/references/index.md` differ. Value-gap evidence, custody changes, cookbook-note changes and the phase tracker therefore survive exactly.
- The five named resolution files—configs Reference, model architecture, RL API, parity coverage and 8-rank YAML—did not change from `666deec` to BASE. At `7f797a3` they equal the r1-verified `e5db8ca` resolution byte for byte. Final HEAD matches `09727f7` except the intended r2 branch/base identity sentence in parity coverage. Every runtime/build/config/test tree is identical to r1 plus its P3 fixes (`r1-runtime-equivalence.json`).
- The References index contains all **21 parent-union catalog targets exactly once** (22 links including the incidental restart Decision). All phase headings match BASE. Only reward-reuse moves, appropriately, from historical to Phase 0/1. All six changed lines agree with their notes' descriptions and limits.
- The log has **137 unique dated headings**, compared with BASE 125 and adapter 109. Every parent section body survives, both parent orders are preserved and dates are nonincreasing; the r1-fix/r2-merge entries precede integration then adapter history. There are **no conflict markers in any tracked text**, including ops receipts.
- `engine_rs/TRIM_MANIFEST.json` is identical across BASE, adapter and HEAD: blob `8c1b7734a4a99f8e70ceaacf17d0f86f1ae24e5a`, SHA-256 `b2d1892a15baf5653e493fdb885c5f2baac7a21095a87eb95d9730782ad9272f`.
- `ops/rebuild-2026-09-29/phase-status.md` is untouched from BASE: blob `b6788f48f63ec183cd19ee2e40943ef0493c2bcf`, SHA-256 `8fc7066bbee2b1ae0d974d156c838e694d89db9e62ad48b40b9d712b79e954a2`. It was intentionally not refreshed before landing.

Actual collected test names:

| Suite | BASE | Adapter | HEAD | Unaccounted omissions |
| --- | ---: | ---: | ---: | ---: |
| Pytest | 1,701 | 2,253 | 2,329 | 0 |
| Engine Rust | 69 | 69 | 69 | 0 |
| Root Rust, including ignored | 258 | 279 | 279 | 0 |

Every adapter pytest name and every Rust test from both parents survives. The literal Python union has 2,331 names: two BASE names were deliberately replaced on adapter before this merge, just as in r1:

1. `tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native` → `test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture`. Orbit constructor assertions remain; the obsolete missing-native assertion becomes real native construction, with additional factory/replay/independence coverage.
2. `tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration` → `test_no_authored_grammar_path_include_after_root_engine_edge`. Bridge absence remains; reading the root test retains its existence requirement and adds explicit-import/no-path-include/manifest assertions.

The replacements are unchanged between adapter and HEAD. AST/source scans find no other missing or duplicate declaration, and no parent `tests/` path is deleted. See `inventory/report.md`, `results.json`, normalized name lists, saved source diffs and command manifest. Each Rust revision used an initially empty independent target directory. Parent Python collection used its own archived Python source and a frozen HEAD extension solely for import/collection; no parent native behavior is credited. Initial Python collection overlapped the HEAD rebuild, so all three collections were repeated against the fixed copy and only `*-fixed-native.log` inventories are credited.

The 8-rank config loads through real `FullConfig`, supplies all six reward coefficients, keeps 32 envs and two segments/minibatch per rank, and has terminal scale .75. All four shipped Kaggriculture configs load (`config-loads.json`). `run_ppo` checks the final runtime workload, then the compile stack, then raises the explicit Task 3.1 error before run-dir/env/model allocation. The native evaluation factory exists and its remaining policy mapper is explicitly guarded. Teacher skip reasons and current docs agree with these seams.

Requested checks:

| Command | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**: 41 unit, nine RNG, 19 parity |
| `cargo test --locked --offline` | **274 passed, five ignored** |
| `uv run python scripts/check_engine_trim.py` | **OK** |
| `uv run pytest tests -m 'not slow' -q` | Initial combined run: 2,315 passed, 10 skipped, four shared-deadline failures; permitted fresh-process sharding below yields **2,319 passed, 10 skipped** |
| `uv run mypy python/owl scripts` | **No issues in 69 source files** |
| `uv run python scripts/check_doc_freshness.py` | **No doc updates required** |

The combined pytest run reached four observation-oracle custody cases after the module's import-time shared deadline (about 110 seconds) had expired. All four failed before child launch with `shared oracle execution deadline expired before launch`; the oracle and its tests are byte-identical to BASE. The recovery kept the production guard intact: `uv run pytest tests/tools/test_observation_oracle_custody.py -m 'not slow' -q` passed **53**, and `uv run pytest tests --ignore=tests/tools/test_observation_oracle_custody.py -m 'not slow' -q` passed **2,266 with 10 skipped**. These disjoint shards cover the complete requested suite. Initial failure and successful recovery logs are retained in `pytest.log`, `pytest-*-shard.log`, `checks.json`, `shards.json` and `resource-guard-recovery.json`.

The ten skips are six hardware/backend cases and four pending Task 3.1 integration cases. The freshness checker examines changes relative to HEAD; its clean-worktree pass alone does not qualify merged prose. The parent-to-HEAD documentation audit supplies that check. `git diff --check faed717 HEAD -- ':!ops'` also passes.

A source-bound `uv run maturin develop --locked --offline --skip-install` rebuild succeeded before behavioral tests and mutations. Runtime imports resolve inside this checkout. Rebuilt extension SHA-256 is `f165fc43da61cedb2ea4d06ee60ec2d3496dd53db498813b8e3c8eea78526efe`; macOS arm64, Python 3.12.13, Torch 2.9.0, pinned Rust nightly. `environment.json` and `runtime-identity.json` preserve exact versions and the prebuild module hash. These checks establish CPU correctness only.

Six mutations were applied to a HEAD archive under `/private/tmp`. Each baseline and restored run passed, and each mutant exited 1 through the intended assertion/error mismatch:

| Mutation | Targeted failures |
| --- | --- |
| Remove 8-rank `econ_ineffective_cap` | Config/global-workload and startup cases, two failures |
| Remove explicit Task 3.1 startup guard, restoring the preceding Orbit guard path | Exact remaining-blocker startup case, one failure |
| Remove pre-stop workload check | Unserviceable-workload and 8-rank headroom cases, two failures |
| Remove pre-stop compile-stack check | Unprobed-stack rejection, one failure |
| Remove policy-evaluation mapping stop | Named remaining-mapper blocker, one failure |
| Restore expected-table model default | Native loader called exactly once, one failure |

`mutations.py`, `mutations.json` and `mutation-*.log` preserve exact substitutions, argv, failures and mutant/restoration hashes. Before and restored SHA-256 values are identical and match the untouched tracked worktree:

| File | Before = restored = worktree SHA-256 |
| --- | --- |
'''
for p,h in hashes.items():report+=f'| `{p}` | `{h}` |\n'
report+='''
All **5,972 tracked files** match the initial byte hashes (symlink targets checked as link text). `git diff HEAD`, staged diff and tracked-only porcelain status are empty; HEAD remains `bc953e9`. Final `git status --porcelain` contains only:

```text
'''+status+'''```

Findings: **no new P1/P2/P3 findings; no fix required**. Both previous P3 findings are **RESOLVED** at the file/line evidence above. The shared-deadline failure was recovered by the explicitly permitted sharding, without altering any guard. Remaining trainer/GPU/DMA gaps are accurately documented and are not qualified by this CPU merge verification.

VERDICT: APPROVE
'''
(out/'report.md').write_text(report)
print('Wrote report; tracked byte hashes unchanged:',len(pre))
print(status,end='')
