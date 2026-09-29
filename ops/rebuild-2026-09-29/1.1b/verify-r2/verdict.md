Verified **`kg/rebuild-parity` at `25ec814768d35f4ee1f89678485cc9099e27019b`**, reviewing `0dc9bdd..HEAD` and refixes `9ee7fb8..25ec814`. All execution was offline with the requested `TMPDIR`.

[Full report and command details](/Users/poonszesen/kg-v3-parity/.codex-tmp/verify-1.1b-r2/report.md).

**Blocking findings:** None.

**Non-blocking finding:** Documentation retains stale test totals. [Coverage:167](/Users/poonszesen/kg-v3-parity/docs/rules-parity-coverage.md:167) says 16 replay tests; [coverage:199](/Users/poonszesen/kg-v3-parity/docs/rules-parity-coverage.md:199) and [cookbook:92](/Users/poonszesen/kg-v3-parity/cookbook/references/live-differential-parity-checks-the-rust-kernel.md:92) say 66 engine tests. HEAD has **19 replay / 69 total**. Update these totals or explicitly label them historical.

**Every round-one finding:** R1 had no blocking findings. All requested fixes are resolved.

| R1 finding | Status | Independent evidence |
|---|---|---|
| Overbroad divergence classification | **RESOLVED** | [sweep.py:136](/Users/poonszesen/kg-v3-parity/scripts/kaggriculture_parity/sweep.py:136) requires observed D2 rejection/acceptance or a successful D1 ASCII recheck. Repeated state corruption reports **unclassified: 1, new: 1**. |
| Null probe substituted with PASS | **RESOLVED** | [generator:729](/Users/poonszesen/kg-v3-parity/scripts/kaggriculture_parity/generate_traces.py:729) preserves scripted values. All **303/303** live probe labels match submitted actions; probe 296 submits `null`. |
| Incomplete rejection rollback comparison | **RESOLVED** | [replay:413](/Users/poonszesen/kg-v3-parity/engine_rs/tests/replay_parity.rs:413) checks public/private values and recursive key order, statuses, rewards and completion. Three added regressions pass. |
| Nonminimal divergence repros | **RESOLVED** | Seven fixtures now diverge at **line 1, step 0**, without a preamble. All reproduce; **7/7** live argument-deletion controls agree with Rust. |
| “Million-unit” wording | **RESOLVED** | [Coverage:275](/Users/poonszesen/kg-v3-parity/docs/rules-parity-coverage.md:275) correctly states **10¹²**, matching fixture actions. |
| Quantity probes described as covering every market verb | **RESOLVED** | [Coverage:282](/Users/poonszesen/kg-v3-parity/docs/rules-parity-coverage.md:282) names the four quantity-taking verbs. Null and minimization wording are also corrected. |

Implementation changes are in `6217868`; `25ec814` updates documentation and receipts.

| Required check | Result | Evidence |
|---|---|---|
| **1. Real Kaggle engine and hash guard** | **PASS** | [Generator:103](/Users/poonszesen/kg-v3-parity/scripts/kaggriculture_parity/generate_traces.py:103) computes/checks SHA-256; `:143` checks the installed engine; `:754/:778` call Kaggle `make`/`step`. Installed **1.32.7** matches pin `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`. Incorrect-pin injection makes `main` return **2**, without creating output. |
| **2. Official trace format** | **PASS** | Inspected all **15 fixtures / 4,012 records**. Header, transition and state structures match official fixtures; provenance differs and `rejected` is the documented extension. Schema tests and byte-identical live regeneration pass. |
| **3. Complete shared comparator** | **PASS** | Official, generated and external callers at [replay:590](/Users/poonszesen/kg-v3-parity/engine_rs/tests/replay_parity.rs:590), `:645`, `:817` all call `replay_text`. Complete public/private trees, key order, statuses, rewards, step/completion and terminal banks are compared. No weaker generated path. |
| **4. Actual edge coverage** | **PASS** | Counts below come from fixture actions. Independently reconstructed **3,113** seeded edge-policy decisions and matched their outputs. |
| **5. Perturbation/restoration** | **PASS** | Changed a scratch fixture copy’s step-100 money **0.0→1.0**. Replay exits **101**, reporting `line 103 … public.farms[0].money: expected 1.0 actual 0.0`. Exact-byte restoration passes **719 transitions / 10 rejections**. |
| **6. Manifest hashes and sizes** | **PASS** | **15/15** current hashes/sizes, the manifest’s trim pin, and all **eight historical** manifest entries match. Size details below. |
| **7. Documentation matches evidence** | **PARTIAL** | Fixture/sweep counts and corrected coverage claims agree with evidence. Only the stale test totals identified above need editing. |
| **8. Dependencies untouched** | **PASS** | `git diff 0dc9bdd..HEAD -- pyproject.toml uv.lock` is empty. Neither file was modified during verification. |
| **9. Divergence reproduction/minimization** | **PASS** | All seven live-regenerated repros reproduce at step 0. D1 market expects money **2699.0**, Rust produces **3000.0**; D1 unit errors; five D2 cases show Python’s unhashable `TypeError` versus Rust acceptance. Removing each offending argument eliminates its divergence. |

**Edge-action counts:** **3,148** edge-seat submissions, including rejected pairs and fallbacks. Categories overlap; submission does not imply execution. Malformed counts use exact JSON membership in the generator’s case sets.

| Case | Occurrences |
|---|---:|
| HIRE orders / over-limit queues | 11,060 / 947 |
| Zero / 1023 / negative quantities | 178 / 180 / 293 |
| Float / string / boolean quantities | 439 / 701 / 273 |
| Null / array / object / ≥10¹² quantities | 157 / 163 / 119 / 132 |
| Empty orders / empty dictionary-action markets | 510 / 309 |
| Duplicate / reversed queues | 1,510 / 287 |
| BUY_LAND orders / queues containing ≥4 | 1,169 / 272 |
| Unknown uppercase / lowercase verbs | 145 / 81 |
| Malformed orders / unit commands / whole actions | 1,079 / 534 / 93 |
| Missing-hand commands / affected actions | 1,081 / 425 |
| Oversubscribed PLANT actions | 476 |
| Uncaught-int candidates / actual rejected pairs | 69 / 25 |

Execution reaches **250 actors**, includes **two hiring transitions above 241 actors**, **30 in-limit BUY_LAND orders after all quadrants unlock**, and the rich-game market escape limit. [Detailed audit and definitions](/Users/poonszesen/kg-v3-parity/.codex-tmp/r2-coverage/result.json).

**Compressed fixture sizes**, bytes; abbreviated filenames:

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

Total **665,021 bytes**, below the 4 MB budget and reasonable for Git. Archived failing fixtures add **386,227 bytes**.

**Command results:**

- `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`: **69 passed, 0 failed, 0 ignored**—41 library, nine RNG, 19 replay.
- `UV_OFFLINE=1 uv run python scripts/check_engine_trim.py`: **`engine trim manifest: OK`**, including final recheck.
- Targeted Python suites: **78 passed, 0 skipped**, including live byte-identical regeneration.
- `UV_OFFLINE=1 uv run python scripts/kaggriculture_parity/sweep.py --games 8 --traces .codex-tmp/verify-1.1b-r2/sweep-traces --out .codex-tmp/verify-1.1b-r2/sweep`: **8/8 games match**, **5,752 game transitions**. Including probes: **311 traces / 7,267 transitions, 274 matches, 37 divergences—D1 12/D2 25, zero new**. All 12 ASCII rechecks pass. Runtime **15.49 seconds across timed phases**. Sweep exits **0**; initial inner replay exits **101** for known probe divergences. [Receipt](/Users/poonszesen/kg-v3-parity/.codex-tmp/verify-1.1b-r2/sweep/sweep-summary.json).

Final **`git status --porcelain` is empty**. No commits or tracked modifications were made; scratch artifacts are ignored.

VERDICT: APPROVE WITH EDITS