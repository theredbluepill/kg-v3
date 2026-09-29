Verified `kg/rebuild-parity` at **25ec814768d35f4ee1f89678485cc9099e27019b**, reviewing `0dc9bdd..HEAD` and independently checking refixes `9ee7fb8..25ec814`. All executions were offline, using the requested TMPDIR. Scratch evidence is under `.codex-tmp/verify-1.1b-r2/`, with coverage in `.codex-tmp/r2-coverage/` and deletion controls in `.codex-tmp/r2-refix-controls/`.

**Blocking findings:** None.

**Non-blocking finding:** Current test totals are stale: `docs/rules-parity-coverage.md:167` says 16 replay tests and `:199` says 66 engine tests; `cookbook/references/live-differential-parity-checks-the-rust-kernel.md:92` also says 66. HEAD has **19 replay / 69 total**. Update these or explicitly scope them to the pre-refix revision. The updated receipt at `ops/rebuild-2026-09-29/1.1b/results.md:129` is correct.

**Every round-one finding:** R1 reported no blocking findings. All its requested edits are resolved:

| R1 finding | Status | Independent evidence in the refixes |
|---|---|---|
| Input-only known-divergence classification | RESOLVED | `sweep.py:136-173,378-387` checks D2's observed exception/acceptance or runs D1 ASCII rechecks. Repeating the corrupted `public.day` experiment yields `unclassified: 1`, `new_divergences: 1` (`classification-summary.json`). |
| Null probe replaced by PASS | RESOLVED | `generate_traces.py:729-735,767-778` submits scripted values exactly. All 303 live probe labels match submitted actions; probe 296 submits null. |
| Incomplete/order-insensitive rejection rollback | RESOLVED | `replay_parity.rs:399-452,516-522` checks both trees, recursive key order, statuses, rewards and done. Three added rollback regressions pass. |
| Seven repros not minimized | RESOLVED | `generate_traces.py:473-523` and actual fixtures have no preamble, one divergent field, and first mismatch at line 1/step 0. All seven reproduce; all seven live argument-deletion controls pass Rust replay. |
| “Million-unit” wording | RESOLVED | `docs/rules-parity-coverage.md:275` now says 10^12, matching actual actions. |
| Quantity coverage says every market verb | RESOLVED | `docs/rules-parity-coverage.md:282` names the four quantity-taking verbs. Null and minimization wording are also corrected. |

Code/fixtures changed in `6217868`; `25ec814` supplies updated documentation and receipts. The stale test totals above are a remaining documentation issue introduced by the expanded test surface.

| Check | Result | Evidence |
|---|---|---|
| 1. Real Kaggle engine/hash refusal | PASS | Generator `:99-117` computes/checks SHA-256; `:120-149` resolves the installed module; `:754,778` invokes Kaggle make/step. Installed version 1.32.7 hashes to `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`. Injecting a wrong expected hash into main returns 2 before creating output (`hash-refusal.log`). Hash tests pass. |
| 2. Official fixture format | PASS | Independently inspected all 15 fixtures/4,012 records. Header, transition and state structures match official fixtures; provenance differs and rejected records are the documented extension. `tests/scripts/test_kaggriculture_parity.py:98` passes. Live regeneration reproduces all fixture and manifest bytes. |
| 3. Complete shared comparison | PASS | Official, generated and sweep callers at `replay_parity.rs:590,645,817` all use `replay_text`. `:35-91,490-575` checks entire public/private trees and object order, statuses, rewards, step/done and terminal banks. Rejection rollback is now equally comprehensive. No weaker generated path. RNG/shop schedules remain metadata, with effects checked through state, as documented. |
| 4. Actual edge coverage | PASS | Counts below come from decoded fixture actions; 3,113 seeded edge-policy decisions were independently reconstructed and matched. All 303 live probe labels match submitted actions. |
| 5. Perturbation/restoration | PASS | A scratch copy of gen-edge-vs-edge was changed at step 100, money 0.0→1.0. Replay exits 101: line 103, public.farms[0].money, expected 1.0 actual 0.0. Exact-byte restoration passes: one test, 719 transitions/10 rejections (`perturb-summary.json`). Tracked fixtures were never altered. |
| 6. Hashes/sizes | PASS | All 15 current manifest hashes/sizes and the manifest's trim pin match. All eight historical manifest entries also match (five archived files plus three unchanged current files). Sizes below are reasonable for Git. |
| 7. Documentation/evidence | PARTIAL | Coverage and recorded sweep counts agree: 8 committed games/3,960 transitions/25 rejections; refix sweep 40 games/21,824 game transitions plus 303 probes/1,515 transitions, 37 known divergences, 0 new, 46.68 phase-seconds. Only the stale current test totals above need editing. |
| 8. Dependency files untouched | PASS | `git diff 0dc9bdd..HEAD -- pyproject.toml uv.lock` is empty; neither file was changed during verification. |
| 9. Divergences/minimization | PASS | All seven live-regenerated repros diverge at line 1/step 0. D1 market: expected money 2699.0, actual 3000.0; D1 unit: cannot parse Arabic digit; five D2 cases: Python unhashable TypeError versus Rust acceptance. Removing the offending argument eliminates each divergence: 7/7 live controls agree (`divergences-report.json`, `controls-report.json`). |

**Edge-action counts:** 3,148 edge-seat submissions, including rejected pairs and fallback actions. Categories overlap; submission does not imply execution. Malformed cases use exact JSON membership in the generator's named case sets; empty markets count dictionary actions with empty/default market, not malformed non-dictionary actions. Detailed definitions and source-bound audit: `.codex-tmp/r2-coverage/audit.py` and `result.json`.

| Claimed case | Occurrences |
|---|---:|
| HIRE orders / over-limit queues | 11,060 / 947 |
| Zero / 1023 / negative quantities | 178 / 180 / 293 |
| Float / string / boolean quantities | 439 / 701 / 273 |
| Null / array / object / ≥10^12 quantities | 157 / 163 / 119 / 132 |
| Empty orders / empty markets | 510 / 309 |
| Duplicate / reversed queues | 1,510 / 287 |
| BUY_LAND orders / queues with ≥4 | 1,169 / 272 |
| Unknown uppercase / lowercase verbs | 145 / 81 |
| Malformed orders / unit commands / whole actions | 1,079 / 534 / 93 |
| Missing-hand commands / affected actions | 1,081 / 425 |
| Oversubscribed PLANT actions | 476 |
| Uncaught-int candidates / actual rejected pairs | 69 / 25 |

Execution reaches **250 actors**, includes **two hiring transitions above 241 actors**, **30 in-limit BUY_LAND orders after all quadrants unlock**, and the rich game's **99,999-unit market escape limit** (step 4: 10^12 TOMATO order plus separate four-unit order yields 100,003 seeds).

**Compressed fixture sizes**, bytes; filenames abbreviated:

| Fixture | Bytes |
|---|---:|
| random-vs-random | 105,461 |
| edge-vs-edge | 117,171 |
| starter-vs-random | 85,670 |
| random-vs-edge | 111,741 |
| edge-free-hire | 49,255 |
| edge-vs-random-rich | 21,607 |
| random-vs-starter-custom | 70,921 |
| pass-vs-edge-negative-seed | 95,654 |
| D1 unit / market | 1,043 / 1,077 |
| D2 plant / unit / shed / missing-hand / market | 1,079 / 1,071 / 1,090 / 1,092 / 1,089 |

Total **665,021 bytes**, below the 4,000,000-byte budget. Archived failing traces add **386,227 bytes**.

**Command results** (all with requested TMPDIR and UV_OFFLINE=1 where relevant):

- `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`: **69 passed, 0 failed, 0 ignored** (41 library + 9 RNG + 19 replay). The external-directory test is a no-op when unset; the sweep below exercises it.
- `uv run python scripts/check_engine_trim.py`: **engine trim manifest: OK**, including final recheck.
- `uv run pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py --basetemp .codex-tmp/verify-1.1b-r2/pytest -q`: **78 passed, 0 skipped**, including live byte-identical regeneration.
- `uv run python scripts/kaggriculture_parity/sweep.py --games 8 --traces .codex-tmp/verify-1.1b-r2/sweep-traces --out .codex-tmp/verify-1.1b-r2/sweep`: **8/8 games match**, 5,752 game transitions; with 303 probes, **311 traces/7,267 transitions, 274 matches, 37 divergences (D1 12/D2 25), 0 new**. All 12 D1 rechecks pass. Runtime **15.49 seconds across timed phases** (4.77 generation + 3.71 probes + 6.89 replay + 0.12 recheck). Sweep exits 0; initial inner replay exits 101 for the known probe divergences.
- Perturbed fixture: **0 passed/1 failed**, restored: **1 passed/0 failed**. Seven deletion controls: **1 test passed, 7/7 traces agree**.

Final `git status --porcelain` is **empty**. Tracked tree and index are clean; no commits or tracked modifications were made. Scratch artifacts are ignored.

VERDICT: APPROVE WITH EDITS
