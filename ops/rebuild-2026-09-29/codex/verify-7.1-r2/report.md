# Independent verification — Task 7.1 r2

Reviewed branch `kg/rebuild-7-1`, HEAD `908c73fce0be298f81e229b08bf0ca082ad2e075`, against `b8747b6e8acece5f561d09a75bb914364a60ac05...HEAD`. The merge base equals the requested base. Scope is the Task 7.1 brief, including four native opponents, separate manifests, distinct mechanisms, original-behavior parity and seed determinism.

## Findings and prior dispositions

**No new actionable finding.** Every finding from `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-7.1-r1.md` is accounted for:

| Prior finding | Severity / original location | Status | Implemented fix and current evidence |
| --- | --- | --- | --- |
| Missing original-reference resumed replay | P2, `opponents_rs/tests/lifecycle.rs:141` | **RESOLVED** | `scripts/kaggriculture_parity/generate_traces.py:1537` constructs fresh original controllers, verifies recorded prefixes and generates resumed actions. `opponents_rs/tests/oracle_parity.rs:354` rebuilds fresh native controllers and compares prefix/resumed actions and final states. All 24 cases / 1,152 resumed actions match. Independently regenerated fixture is byte-identical and mutation comparison is sensitive. |
| Stale opponent test count | P3, old Reference line 115 | **RESOLVED** | `cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md:136` says **22**, matching this version's 12 unit, five lifecycle/match and five oracle tests. Prior requested 20 was correct for f15a413; two replay tests have since been added. |

The original r1 report is preserved; detailed new annotations are in `prior-findings.md` beside this report. No additional fix is required for approval.

## Executed checks

| Command / check | Result | Receipt |
| --- | --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**, zero failed/ignored (41 + 9 + 19) | `engine.log` |
| Root `cargo test` | **254 passed, 4 ignored**, zero failed | `root.log` |
| `cargo test --manifest-path opponents_rs/Cargo.toml --locked --offline` | **22 passed**, zero failed/ignored | `opponents.log` |
| Relevant `uv run pytest` | **187 passed, 10 skipped**, zero failed | `python.log` |
| `uv run mypy python/owl scripts` | Passed, **63 source files** | `mypy.log` |
| `uv run python scripts/check_engine_trim.py` | Passed | `trim.log` |
| `uv run --offline python scripts/check_opponent_import.py --original-sources` | Passed, original source closure re-read | `custody.log` |
| Opponent all-target locked/offline Clippy, warnings denied | Passed | `clippy.log` |

Relevant pytest covered `tests/tools/test_check_opponent_import.py`, `tests/scripts/test_kaggriculture_parity.py`, `tests/owl/kaggriculture/test_opponents.py`, `tests/tools/test_check_engine_trim.py`, and `ops/rebuild-2026-09-29/7.1/test_update_trim_manifest.py`. Nine skips explicitly await Task 1.4 bindings; one skips the broad committed-corpus regeneration on this Mac. The bounded opponent regeneration was independently executed below. Pytest ran on the project's Python 3.12.13; original opponent generation used the required sibling CPython 3.11.15.

`git diff --check` finds whitespace in frozen historical execution receipts, not production sources. Those receipts were preserved; this is not a new code defect.

## Oracle reproduction and scope

Fresh generation using `/Users/poonszesen/kaggriculture/.venv/bin/python` read the pinned original submissions. Four invocations generated two games each, seeds 20260929–20260936. All eight gzip files **and their manifest** reproduced byte-for-byte. A separate replay invocation regenerated all 24 original-Python replay cases in 23.726 seconds at 772,734,976 bytes peak RSS; `REPLAY.json.gz` also reproduced byte-for-byte. See `regeneration/byte-comparison.json` for all ten files.

Native continuous comparison matched **11,504 / 11,504 actions**, all **5,752 transitions**, public/private state, statuses, rewards and terminal banks (`parity.json`). Native reconstructed replay matched **1,152 / 1,152 resumed actions** and **24 final states** (`replay.json`). Four imports and the E776 policy data retain their pinned bytes. Styles are distinct by their implemented mechanisms: stationary crop cycle, adaptive task assignment, economic route planning and repaired production tape. Same-seed determinism has native tests plus matching independently repeated oracle comparisons.

Market coverage was checked against engine semantics: `engine_rs/src/lib.rs:3733` treats inventory below I0 as scarcity pricing; `engine_rs/src/lib.rs:3893` permits BUY_PRODUCT based on money and shed space, without a finite-stock check. Default I0 is 10,000 (`engine_rs/src/lib.rs:171`). Every trace contains 718/719 predecision states with at least one scarce product, and the corpus contains **529 BUY_PRODUCT orders** for products whose predecision inventory is below I0. See `scarcity-census.json`. The existing quantity-above-inventory proxy cannot establish physical stock shortage, because this engine has no such cap. Therefore its zero count is not a missing executable stock-rejection case.

The qualification remains default-config/CPython-3.11 scoped. Own-play replay prefixes do not qualify arbitrary foreign prefixes. Explicit reset is tested natively; learned-seat binding remains Task 1.4. Rival-private perturbation tests are bounded, though hidden engine state is inaccessible through the view. Original notice/redistribution limits remain documented. These checks do not establish playing strength or a held-out evaluation panel.

## Independent mutations and restoration

On an isolated engine/opponent scratch copy, every continuous oracle was independently changed at step 37 (alternating seats): **eight semantic comparison failures** at the exact changed actions. Each of the **24 replay cases** was changed five steps after its reconstruction point; one batch run had to report all 24 exact source/point/step/seat mismatch labels. Then each of the **four native bot return paths** was independently changed at step 700; each failed the original-Python resumed comparison for its final-day cases. These are **36 detected injected changes in 13 trials**, with no compiler failure counted as detection. See `run_mutations.py`, `mutations-results.json`, and per-trial logs.

Every scratch mutation was restored in a finally block. All **56 scratch source/data files** matched their original SHA-256 inventory afterward. The restored full opponent suite passed **22 tests** (`mutations-restored-full-suite.log`). This covers each of the eight newly added continuous oracle files and the new replay oracle, including every replay case.

## Worktree preservation

All **1,750 tracked files** match their starting SHA-256s, both staged and unstaged diffs are empty, and HEAD is unchanged. No tracked modifications were made. New verification receipts are untracked under `ops/rebuild-2026-09-29/codex/verify-7.1-r2/`; scratch copies/build artifacts are under ignored `.codex-tmp/`. Historical r1 evidence was left intact.

VERDICT: APPROVE
