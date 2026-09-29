# Independent verification of Task 7.3

Reviewed branch `kg/rebuild-7-3`, HEAD `1e7edb65766b164281f8edb9c3f64e064d5f0ecf`, against `0b8cf98ef57fc49a329dca4c8368c630586c4752...HEAD`. The base equals the merge base. The initial tracked worktree and index were clean.

Target: check the requested Kaggle episode format, seed-header reconstruction, byte/semantic round trip, independent oracle sensitivity, and eight completed exports per evaluation. Stop after requested checks and scratch mutations, with an explicit account of missing acceptance criteria. This is verification only; no implementation fix or tracked cookbook adaptation is made.

## Findings

1. **P2 — Eight completed replays per evaluation is still unimplemented.** `tests/kaggriculture/test_replay_export_integration.py:15–18` unconditionally skips the acceptance test. `docs/rl-api-specs.md:1050–1054` confirms the native environment and evaluation wiring are absent. `scripts/run_ppo.py:45` still imports the Orbit recorder; the new Kaggriculture recorder has no evaluation caller. The rank configs selecting eight ordinals and mocked recorder tests do not demonstrate eight completed evaluation exports, consumed-seed/reset custody, or terminal-before-reset capture. **Fix:** integrate the recorder with the Task 1.4 native binding, replace the five placeholders with executable lifecycle/evaluation tests, and retain eight completed source-bound round-trip receipts. This is an unfinished acceptance criterion, not an undisclosed regression: `ops/rebuild-2026-09-29/7.3/results.md:137` records that the original implementation prompt intentionally deferred wiring. That explains the scope but does not satisfy the full acceptance criteria in this verification request.

2. **P3 — Successful export can fail its own round trip.** `src/kaggriculture/replay_export.rs:325` only checks `complete=true` against the final engine status. Reproduction: supported `episodeSteps=2`, one PASS/PASS transition, `complete=false`. Export succeeds with both seats DONE and `info.v3_native_replay.complete=false`; verification of the returned bytes rejects `/steps/1/0/status`, transition 0. The otherwise identical `complete=true` input round-trips. **Fix:** reject a false completion claim when the replay has reached DONE, and assert that all successfully exported completion variants verify. The recorder catches the verification failure before publishing an episode, limiting impact. Receipt: `completion-probe.log`.

3. **P3 — The byte-oracle regression no longer isolates the byte check.** `tests/kaggriculture/test_replay_export_oracles.py:355–359` changes `3000.0` to integer `3000`. Since the reward-kind repair, this fails the semantic number-kind comparison before reaching the canonical-byte guard. **Fix:** retain that number-kind check and add a raw-text `3000.0` to `3.0e3` mutation, asserting the canonical-number-byte diagnostic. The same-kind mutation was rejected at `/steps/0/0/observation/farms/0/money` in this verification, so this is a regression-test gap, not a demonstrated production byte-comparison failure.

## Actual checks

| Check | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed: 41 + 9 + 19; zero failures/ignores |
| `cargo test` | 269 passed, four ignored, zero failures |
| `uv run python scripts/check_engine_trim.py` | Initial automatic build failed before checker execution: unavailable PyPI DNS |
| `uv run mypy python/owl scripts` | Initial automatic build failed before mypy execution: unavailable PyPI DNS |
| Relevant plain `uv run pytest -q ...` | Initial automatic build failed before pytest execution: unavailable PyPI DNS |
| `uv run --no-sync maturin develop --offline --locked --skip-install` | Fresh current-source debug extension built successfully in place |
| `uv run --no-sync python scripts/check_engine_trim.py` | Passed |
| `uv run --no-sync mypy python/owl scripts` | Passed, 64 source files |
| Relevant `uv run --no-sync pytest -q ...` | 121 passed, five integration skips, 114.47 seconds |

Pytest paths: `tests/kaggriculture/test_replay_export.py`, `test_replay_export_oracles.py`, `test_replay_export_integration.py`, `tests/tools/test_replay_trim_manifest.py`, and `test_observation_oracle_custody.py`.

The initial `maturin develop --offline` fallback also tried dependency installation and hit DNS; `--skip-install` allowed a source rebuild without changing dependencies. All failures and successful reruns remain in this directory. No test failure is hidden as an environment failure. Full preparation was not required or run by this read-only review.

Four official fixture baselines independently compare all 2,876 transitions and their initial/terminal/public/private state, order, statuses, rewards and banks. The small real pinned-framework game covers nine transitions at seed `2**80 + 19`; installed version 1.32.7 and all four pinned source hashes match. Native byte round trip, exact wide seed and captured evidence pass.

Scratch-input mutation results: **8 positive baselines and 12/12 rejected controls**, 116.03 seconds. Controls cover each of four official fixtures' raw rewards, framework seed, ACTIVE reward kind, native equivalent float bytes, native number kind, captured banks, shared/private separation and two private-map order cases. Every mutated file was restored byte-for-byte with matching SHA-256. Exact errors, pointers, transitions and hashes are in `mutations-results.json` and `mutations.log`.

Two additional source mutations ran in an isolated archive of HEAD. Disabling the canonical-byte guard allowed the otherwise-rejected equivalent-float spelling to pass. Replacing `compare_tree` with `Ok(())` allowed nine otherwise-rejected corruptions (all four fixture rewards, framework seed, captured bank, private leakage, private map order and reward number kind) to pass. Both extensions were loaded from the scratch tree, not the worktree. The scratch Rust source was restored byte-for-byte with matching SHA-256. Receipts: `source-byte-baseline.json`, `source-byte-mutant.json`, `source-comparator-mutant.json`, and `source-mutation-restoration.json`.

All **2,006 tracked files** have identical before/after SHA-256 lists. Both tracked and staged diffs are empty. The only untracked addition is this verification-receipts directory. The worktree extension retained its independently tested fresh-build SHA-256 `11d4e8deacce0b49cb7d25ed8c8ac1fcdb2224270bb1da050aaa0995dbdfdf2e` after the scratch source probes.

`git diff --check` outside `ops/` passes. The full diff check reports whitespace retained in historical operational logs; this is not a code correctness finding.

VERDICT: REJECT for full Task 7.3 acceptance, principally because the requested eight-export evaluation path remains absent. The implemented diagnostic exporter passes the reported normal-path oracles, with the two P3 corrections above.
