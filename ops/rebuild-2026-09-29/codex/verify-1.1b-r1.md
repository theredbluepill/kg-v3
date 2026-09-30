Verified **`kg/rebuild-parity` at `9ee7fb86c0e1118bfe59093c00af0db0b7fde0a3`**, reviewing `0dc9bdd..HEAD`. All executions were offline, with scratch under the requested `TMPDIR`.

**Blocking findings:** None invalidating the observed parity results.

**Non-blocking edits:** Correct the sweep’s overly broad known-divergence classification, the null-probe substitution, rejection-state checking, and the coverage/minimization wording below.

| Check | Result | Evidence |
|---|---|---|
| **1. Real Kaggle engine and hash guard** | **PASS** | [generate_traces.py:103](/Users/poonszesen/kg-v3-parity/scripts/kaggriculture_parity/generate_traces.py:103) computes SHA-256 and rejects mismatches; `:143` checks the installed engine; `:744/:772` call Kaggle `make`/`env.step`. Installed **1.32.7** hashes to `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`. Injecting an incorrect expected hash exercised `main`: **exit 2, no output directory**. Hash-refusal tests also passed. |
| **2. Official trace format** | **PASS** | All 15 generated fixtures match official header/transition keys, ordering and state structure. Generation provenance differs intentionally; `rejected` is the documented extension. [Schema test:98](/Users/poonszesen/kg-v3-parity/tests/scripts/test_kaggriculture_parity.py:98) passed, as did byte-identical live regeneration. |
| **3. Complete, shared comparison** | **PARTIAL** | Official, generated and swept traces all call `replay_text` at [replay_parity.rs:469](/Users/poonszesen/kg-v3-parity/engine_rs/tests/replay_parity.rs:469), `:523`, `:695`. Successful transitions compare complete public/private trees, recursive key order, statuses, rewards, completion and terminal banks. However, rejected-step rollback at `:398` uses order-insensitive JSON equality and excludes statuses/rewards/done. No actual rollback bug was observed; the kernel clones before committing. |
| **4. Claimed edge coverage** | **PARTIAL** | Claimed edge classes occur; counts below. **Probe 296 is mislabeled:** header says `null`, but its submitted action is PASS because [generator:760](/Users/poonszesen/kg-v3-parity/scripts/kaggriculture_parity/generate_traces.py:760) treats `None` as a sentinel. Exactly **1/303** probe labels disagrees with its submitted action. Full-game policies separately exercise null actions. |
| **5. Deliberate perturbation** | **PASS** | Changed committed `gen-edge-vs-edge` money from `0.0` to `1.0` at step 100. Replay exited **101**: `line 103 from_step Some(100): public state at public.farms[0].money: expected 1.0 actual 0.0`. Restored exact original bytes; replay then **1 passed, 0 failed**. [Receipt](/Users/poonszesen/kg-v3-parity/.codex-tmp/independent-1.1b-9ee7fb8/perturb-summary.json). |
| **6. Manifest hashes and sizes** | **PASS** | All **15/15** current hashes and sizes match; manifest’s own trim pin matches. Historical manifest’s eight hashes also match: five archived failing files plus three unchanged current files. Sizes below are reasonable for Git. |
| **7. Documentation versus evidence** | **PARTIAL** | Counts match: committed **8 games/3,960 transitions/25 rejections**; saved sweep **40 games/21,824 transitions/155 rejections**, plus **303 probes/1,515 transitions**, with **37 divergences**. Recorded runtime **45.91 s** supports “46 s.” Wording needs correction: injected large orders are **10¹²**, not one million; quantity probes cover four quantity-taking market verbs; null-probe and minimization claims need qualification. [Coverage:263](/Users/poonszesen/kg-v3-parity/docs/rules-parity-coverage.md:263). |
| **8. Isaiah dependency files untouched** | **PASS** | `git diff 0dc9bdd..HEAD -- pyproject.toml uv.lock` is empty. Neither file was modified during verification. |
| **9. Divergence reproduction/minimization** | **PARTIAL** | Reproduced **all seven** divergence fixtures against live Kaggle and Rust. All retain the same divergence kind/field after removing the entire preamble, unused PASS fields and trailing turns: they fail at **step 0**. They isolate the defects but are not minimal repros. [Reduced replay evidence](/Users/poonszesen/kg-v3-parity/.codex-tmp/independent-1.1b-9ee7fb8/reduced-replay.log). |

The sweep classification needs a specific fix: [sweep.py:159](/Users/poonszesen/kg-v3-parity/scripts/kaggriculture_parity/sweep.py:159) classifies solely from action contents; `:302` returns success when no divergence remains unclassified. I deliberately corrupted `public.day` in a D1-containing trace. Rust correctly failed, but the summary reported **D1, `new_divergences: 0`**. Classification should validate the observed mismatch, not merely recognize a problematic input. [Evidence](/Users/poonszesen/kg-v3-parity/.codex-tmp/independent-1.1b-9ee7fb8/classification-summary.json).

**Actual edge-action counts:** 3,148 edge-seat submissions, including rejected pairs and fallbacks. Categories overlap; submission does not imply execution.

| Case | Occurrences |
|---|---:|
| HIRE orders / over-limit queues | 11,060 / 947 |
| Zero / 1023 quantities | 178 / 180 |
| Negative / float / string quantities | 293 / 439 / 701 |
| Boolean / null / array / object quantities | 273 / 157 / 163 / 119 |
| Quantities ≥10¹² | 132 |
| Empty orders / empty markets | 510 / 342 |
| Duplicate / reversed queues | 1,510 / 287 |
| BUY_LAND orders / queues containing ≥4 | 1,169 / 272 |
| Unknown-order cases / unknown verbs / lowercase verbs | 592 / 145 / 81 |
| Malformed orders / unit commands / whole actions | 1,079 / 557 / 93 |
| Missing-hand commands / affected actions | 1,081 / 425 |
| Oversubscribed PLANT actions | 476 |
| Uncaught-int candidates / actual rejections | 69 / 25 |

Effective boundaries also occur: **250 actors**, two accepted hire transitions beyond 241 actors, **30 in-limit BUY_LAND orders after all quadrants unlock**, and a rich-game seed purchase reaching the **100,000-iteration escape**. [Detailed audit](/Users/poonszesen/kg-v3-parity/.codex-tmp/coverage-audit/result.json).

**Fixture sizes**, bytes; abbreviated filenames:

| Game fixture | Bytes |
|---|---:|
| random-vs-random | 105,461 |
| edge-vs-edge | 117,171 |
| starter-vs-random | 85,670 |
| random-vs-edge | 111,741 |
| edge-free-hire | 49,255 |
| edge-vs-random-rich | 21,607 |
| random-vs-starter-custom | 70,921 |
| pass-vs-edge-negative-seed | 95,654 |
| D1 unit / market | 1,337 / 1,345 |
| D2 plant / unit / shed / missing-hand / market | 1,378 / 1,369 / 1,388 / 1,387 / 1,375 |

Total **667,059 bytes**, below the 4 MB budget. Historical failing fixtures add **386,227 bytes**.

**Command results:**

- `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`: **66 passed, 0 failed, 0 ignored**—41 library, nine RNG, 16 replay.
- `UV_OFFLINE=1 uv run python scripts/check_engine_trim.py`: **`engine trim manifest: OK`**, including after restoration.
- Targeted Python tests: **73 passed, 0 skipped**.
- `UV_OFFLINE=1 uv run python scripts/kaggriculture_parity/sweep.py --games 8`, with fresh scratch `--traces`/`--out`: **8/8 games matched**, 5,752 game transitions. Including probes: **311 traces, 274 matches, 37 divergences—D1 12/D2 25**. Timed phases totaled **15.32 s**. Sweep exit **0**; inner Rust replay exit **101** because known divergences are included. [Sweep receipt](/Users/poonszesen/kg-v3-parity/.codex-tmp/independent-1.1b-9ee7fb8/sweep/sweep-summary.json).

Final tracked-tree status is **clean**. `git status --porcelain` reports only `?? .codex-tmp/`; no commits or tracked modifications remain.

VERDICT: APPROVE WITH EDITS