Independent Task 1.3 verification, 2026-09-29.

Reviewed exactly `git diff kg/isaiah-gap-closure...HEAD` on `kg/rebuild-observe`:

- HEAD: `7b5eacd0d393ef65f6bec0998184ed3ae160d378`.
- Base/merge base: `f464c3db45449ce362fde25c5d9b37f88ad20e5e`.
- Reference: `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0` (`kg/reference-2026-09-29`).
- Governing scope: Task 1.3 brief (including Claude's R1 recipe correction), plan Task 1.3 and the accepted observation contract v4. The current contract file is v4.1; its additional grammar-bridge retirement requirement also applies.

No encoder semantic defect was identified. Review compared the structured writer, config admission, buffer boundary and test reconstruction with the pinned reference `engine_rs/src/myolie_features.rs`, engine constructors/hire rules and reference L4 fixture decoder. Field formulas, coordinates, both-seat private isolation, insertion ranks including present-zero keys, exact side tensors, hire costs, masks and transaction ordering agree with the reviewed contract. The binding checks layout and overlapping byte ranges before slices and prepares every environment before publication. This is observation qualification, not environment-lifecycle or learner-throughput qualification.

Findings:

1. **P2 — Pinned-memory capability probe crashes the required test suite.** `tests/kaggriculture/test_observe.py:285` calls `torch.empty(1, pin_memory=True).fill_(0)` in the pytest process and catches only `RuntimeError`. On this macOS/torch 2.9.0 environment it terminates with SIGSEGV (wrapper exit 139), after two unpinned tests pass. A standalone process containing only `import torch` and that expression also exits 139; it does not import or call `owl`. This does not demonstrate an encoder fault, but the new test suite is not green. Fix: avoid the unsupported pinned allocator before touching it (for example an explicit supported CUDA-host-pinning guard), or use a subprocess capability probe that can handle signal termination. Retain positive pinned-buffer checks on a supported host. Rerun the suite and reconcile the claims that these cases safely skip in `docs/rules-parity-coverage.md:485–487` and the observation cookbook Reference at lines 156–157. Evidence: `tests-kaggriculture-test_observe.log`, `pinned-repro.log` and their JSON receipts.
2. **P2 — Contract-required grammar bridge retirement remains undone.** `Cargo.toml:15` introduces the first production root-to-engine edge. `docs/kaggriculture-contract.md:296` and `ops/rebuild-2026-09-29/plan.md:194` explicitly require moving kernel acceptance/replay tests to root integration at that edge and removing the temporary engine test and its authored registration. `engine_rs/tests/grammar_kernel.rs` still exists; `docs/rules-engine.md:237–239` and the observation cookbook Reference at lines 187–190 defer the work to Task 1.4. Disclosure does not amend the requirement. Fix: migrate the acceptance/replay-state tests to the root route, remove the engine bridge, and update the checker/manifest registration and coverage docs, retaining the same test coverage and pinned kernel bytes.
3. **P3 — Current coverage points to the superseded handoff receipt.** `docs/rules-parity-coverage.md:493` links `1.3/results.md` for actual command counts. That frozen file documents the incomplete pre-integration handoff, including zero full recorded reference rows and blocked schema qualification. Current qualification is recorded in `1.3/claude-review.md`. Fix the current-document pointer and label the earlier receipt historical; preserve the frozen earlier evidence.
4. **P3 — Missing-corpus error asserts obsolete coverage evidence.** `src/kaggriculture/oracle_corpus.rs:2182` hardcodes the old `0 < 4` R1 actor-quota diagnosis. The current corpus has six qualifying states. Fix the error to describe missing qualified fixture files and regeneration, without asserting a coverage failure that was not measured by this branch.

Independent execution results (counts exclude repeat runs):

| Check | Result | Evidence |
|---|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked`, serial test scheduling | 87 passed: 41 unit + 18 grammar bridge + 9 RNG + 19 replay | `engine-serial.{json,log}` |
| `uv run python scripts/check_engine_trim.py` | OK | `trim.{json,log}` |
| `cargo test --locked`, serial | 244 passed, 4 ignored; repeated with the same result after mutation restoration | `root.{json,log}`, `restored-root.{json,log}` |
| Retained root Orbit tests | 157 passed (95 RL + 62 rules-engine), 2 additional expensive RL diagnostics ignored; included in 244 above | `restored-root.log` |
| Kaggriculture Python completed disjoint shards | 311 passed, 1 existing native-grammar-binding skip; 2 pinned variants omitted from completed result after the first segfaulted | `shards-tests-kaggriculture.json`, `head-node-groups.json`, `heads-000` through `heads-060`, `observe-all-unpinned.{json,log}` |
| Actual-schema native observation tests, excluding only the two pinned variants | 55 passed, 2 deselected; includes all 512 frozen records | `observe-all-unpinned.{json,log}` |
| Isaiah `tests/owl`, all 22 files separately | 787 passed, 3 platform skips (2 CUDA flash-attention, 1 x86 quantization) | `shards-tests-owl.json` |
| Oracle custody + trim checker tests | 124 passed: 45 custody + 79 trim | `custody.{json,log}`, `custody-collection.log` |
| All seven script test files | 188 passed (153 retained starter + 35 Kaggriculture) | `shards-tests-scripts.json` |
| Remaining grammar-recorder/replay-viewer tool tests | 27 passed | `remaining-tools.{json,log}` |

Across all completed disjoint Python shards: **1,437 passed, 4 skipped**. Only the two pinned-memory observation variants are excluded; the first crashes before the second can execute in their normal file run. This reproduces the recorded passing-test denominator, but not its claim of six safe skips. No completed full-suite command is claimed.

The exact requested combined `uv run pytest tests/kaggriculture -q` was attempted and stopped by the 1 GB resource guard at 1,013,891,072 sampled bytes. Its failure is not counted as an assertion failure or as a completed suite. Per-file sharding exposed the pinned probe crash. The heads file also exceeded the bound as one process, so all 65 head tests were completed in disjoint groups of 12 (last group 5); none was omitted. The 311 successful Kaggriculture tests are 1 typing + 23 grammar + 41 encoder + 65 heads + 55 observation + 126 schema tests. The one pre-existing skip requires the later native grammar binding.

All test/build diagnostics used the reviewed offline/thread settings with a 120-second/1,000,000,000-byte sampled process-group guard. The initial engine run using default Rust test concurrency also exceeded 1 GB; rerunning with `RUST_TEST_THREADS=1` completed at approximately 369 MB. The full root runs completed in 18.26 and 17.03 seconds. No bound was relaxed to obtain a pass. The requested `uv run` trim check rebuilt the Python extension from this source before Python tests; later shards reused that extension with `--no-sync`.

Oracle non-vacuity:

- The clean and restored root runs compare 512 states × 2 seats × 8,176 offsets = 8,372,224 f32 values bitwise against the pinned recorded reference. Custody validates the reference/source identities, record order, per-seat hashes and unchanged quotas. Recorder source calls the actual pinned `encode_invest`, not the new encoder.
- One temporary mutation swapped market inventory and price inputs only inside production `write_shops_and_market`. Custody still passed, then the oracle failed at `record=official:95324500:0 seat=0 offset=889`: actual `0.0025`, recorded `1.0`. This is a semantic failure against independently recorded data, not merely a source-hash rejection.
- `mutation.py` restores original bytes in `finally`; original/restored SHA-256 is `0d84eb9df01f699767fc674996e5eae14be31c09f70b058b5af35fce36365749`. The restored full root suite passes. Evidence: `mutation.json`, `mutant.{json,log}`, `restored-root.{json,log}`.

The three-dot diff contains no `engine_rs` or `scripts/check_engine_trim.py` changes. The manifest checker independently passes. Full-diff whitespace checking reports only whitespace in frozen raw ops logs; excluding those receipts is clean. The tracked worktree/index were clean at entry. Pre-existing untracked `codex-verify-*` artifacts were left untouched and were not used as evidence for this verification. This run's receipts are all in this separate untracked directory.

The docs/cookbook otherwise accurately describe the R1 v2 policy correction, fixed quotas, 512/1,024 corpus denominators, exact caller-buffer interface, and timing limitation. No optimized timing or reference regeneration was rerun here. The brief explicitly allows the existing optimized-build memory-stop/pod handoff, so that timing gap alone is not a new rejection reason. Pinned/GPU behavior remains unqualified locally.

The required Python suite is not green and the contract migration is incomplete. No tracked fixes were made because this is the requested independent verification.

VERDICT: REJECT
