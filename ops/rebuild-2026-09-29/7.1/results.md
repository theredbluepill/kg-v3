# Task 7.1 final report

The standalone import, view, registry, runner, custody, original-Python loader and
qualification tests are implemented. **Original-Python parity is blocked**:
the first R04 mismatch was localized and corpus expansion stopped as instructed.
No imported controller or retained engine bytes changed. Work is uncommitted
on kg/rebuild-7-1. This is not a qualified opponent panel.

## 1. Files changed/added, one line each with purpose

The inventory includes the deleted stop-only note and every run-2 receipt.
Run-1 probe receipts remain byte-exact; commit 21d0f45 preserves the prior report.
The trim updater records every path below and exactly five excluded-copy reasons,
leaving retained/authored engine entries unchanged.

- `engine_rs/TRIM_MANIFEST.json` — updater-generated five excluded-copy reasons and non-engine task inventory
- `cookbook/log.md` — record the import and correct the historical stop attribution.
- `cookbook/references/frozen-engine-api-blocks-standalone-opponent-import.md` — retire the stop-only note in favor of the snapshot-view Reference.
- `cookbook/references/index.md` — index the current opponent-import Reference.
- `cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md` — record implemented import and qualification limits; correct stop attribution.
- `docs/rules-parity-coverage.md` — document snapshot-view, lifecycle and original-Python parity coverage.
- `justfile` — check opponent byte custody and standalone fmt, Clippy and tests.
- `opponents_rs/Cargo.lock` — pin the opponent crate's offline dependency closure.
- `opponents_rs/Cargo.toml` — declare the standalone edition-2024 evaluation crate.
- `opponents_rs/OPPONENT_MANIFEST.json` — pin imported/authored and oracle custody.
- `opponents_rs/README.md` — document view, execution and parity qualification limits.
- `opponents_rs/fixtures/e776-kenjo-trace.json` — preserve E776 executable policy data.
- `opponents_rs/fixtures/oracle/MANIFEST.json` — freeze generated Python oracle metadata.
- `opponents_rs/fixtures/oracle/oracle-00-starter-vs-r04.jsonl.gz` — freeze original Starter/R04 observations and actions.
- `opponents_rs/src/lib.rs` — isolate engine ownership behind a current snapshot view.
- `opponents_rs/src/native_agents.rs` — declare only four byte-exact controller modules.
- `opponents_rs/src/native_agents/e776.rs` — preserve pinned controller bytes.
- `opponents_rs/src/native_agents/ecobot.rs` — preserve pinned controller bytes.
- `opponents_rs/src/native_agents/r04.rs` — preserve pinned controller bytes.
- `opponents_rs/src/native_agents/starter.rs` — preserve pinned controller bytes.
- `opponents_rs/src/registry.rs` — enforce per-seat, per-episode controller lifecycle.
- `opponents_rs/src/runner.rs` — execute bounded default-config evaluation matches.
- `opponents_rs/src/view_tests.rs` — check the view and private-state visibility.
- `opponents_rs/tests/lifecycle.rs` — check lifecycle, determinism and config bounds.
- `opponents_rs/tests/oracle_parity.rs` — compare original Python actions and state.
- `ops/rebuild-2026-09-29/7.1/plan.md` — separate planned checks from actual results.
- `ops/rebuild-2026-09-29/7.1/results.md` — report actual commands, scope, coverage and remaining work.
- `ops/rebuild-2026-09-29/7.1/run2/clippy-final.log` — record final standalone Clippy success.
- `ops/rebuild-2026-09-29/7.1/run2/clippy-first.log` — retain imported-code lint findings before allowances.
- `ops/rebuild-2026-09-29/7.1/run2/clippy-green.log` — record initial standalone Clippy success.
- `ops/rebuild-2026-09-29/7.1/run2/clippy-second.log` — retain authored-code lint findings before correction.
- `ops/rebuild-2026-09-29/7.1/run2/closure.log` — record final docs, byte preservation and inventory checks.
- `ops/rebuild-2026-09-29/7.1/run2/commands.json` — record final command arguments, statuses and counts.
- `ops/rebuild-2026-09-29/7.1/run2/coverage.json` — record exact trace coverage and unmeasured categories.
- `ops/rebuild-2026-09-29/7.1/run2/custody-check.log` — record final opponent custody verification.
- `ops/rebuild-2026-09-29/7.1/run2/custody-content-red.log` — retain strict trace-metadata test failures.
- `ops/rebuild-2026-09-29/7.1/run2/custody-coverage-red.log` — reject invented positive coverage counts.
- `ops/rebuild-2026-09-29/7.1/run2/custody-green.log` — record custody mutation tests and binding skips.
- `ops/rebuild-2026-09-29/7.1/run2/custody-red.log` — retain missing-checker test-first failure.
- `ops/rebuild-2026-09-29/7.1/run2/engine-test.log` — record frozen engine regression results.
- `ops/rebuild-2026-09-29/7.1/run2/engine-trim.log` — record final engine trim verification.
- `ops/rebuild-2026-09-29/7.1/run2/inventory.json` — list final changed paths and their task purposes.
- `ops/rebuild-2026-09-29/7.1/run2/lifecycle-red.log` — retain registry/lifecycle test-first compile failures.
- `ops/rebuild-2026-09-29/7.1/run2/localize_r04_python.py` — localize original R04 decision-path divergence.
- `ops/rebuild-2026-09-29/7.1/run2/opponent-import.log` — record required opponent custody checker status.
- `ops/rebuild-2026-09-29/7.1/run2/opponents-test.log` — record required opponent crate test status.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-comparator-green.log` — record comparator regression success.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-comparator-red.log` — retain comparator test-first failure.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-first-generation.log` — record bounded original-Python generation.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-first-parity.log` — retain typed-reward comparator harness failure.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-first-summary.json` — record first Python game's runtime/counts.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-green.log` — record original-submission loader test success.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-mypy.log` — record trace generator static checking.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-parity.json` — pin actual compared/matched action denominators.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-red.log` — retain original-submission loader test-first failures.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-second-parity.log` — record original R04 action mismatch.
- `ops/rebuild-2026-09-29/7.1/run2/oracle-tests.log` — record trace generator and coverage test results.
- `ops/rebuild-2026-09-29/7.1/run2/prepare.log` — record full requested repository preparation status.
- `ops/rebuild-2026-09-29/7.1/run2/py-prepare.log` — record required Python preparation status.
- `ops/rebuild-2026-09-29/7.1/run2/r04-diagnostic.rs` — reproduce native R04 debug decisions in scratch.
- `ops/rebuild-2026-09-29/7.1/run2/r04-mismatch.md` — explain first R04 mismatch and bounded source attribution.
- `ops/rebuild-2026-09-29/7.1/run2/r04-native-debug.json` — retain native decision-path localization evidence.
- `ops/rebuild-2026-09-29/7.1/run2/r04-python-debug.json` — retain original and counterfactual Python decisions.
- `ops/rebuild-2026-09-29/7.1/run2/reset-green.log` — record fresh-episode reset enforcement.
- `ops/rebuild-2026-09-29/7.1/run2/reset-red.log` — retain same-episode reset bypass failure.
- `ops/rebuild-2026-09-29/7.1/run2/rs-prepare.log` — record required Rust preparation status.
- `ops/rebuild-2026-09-29/7.1/run2/rust-first-green.log` — record initial view/controller test success.
- `ops/rebuild-2026-09-29/7.1/run2/rust-first.log` — retain initial view/controller compile findings.
- `ops/rebuild-2026-09-29/7.1/run2/rust-lifecycle-visibility.log` — record bounded lifecycle/visibility checks.
- `ops/rebuild-2026-09-29/7.1/run2/targeted-tests.log` — record required Python checks and binding skips.
- `ops/rebuild-2026-09-29/7.1/run2/updater-green-final.log` — record final updater tests and byte idempotency.
- `ops/rebuild-2026-09-29/7.1/run2/updater-green.log` — record six updater tests and 12 mutation subtests.
- `ops/rebuild-2026-09-29/7.1/run2/updater-red.log` — retain six failures before resumed-updater support.
- `ops/rebuild-2026-09-29/7.1/run2/write_opponent_manifest.py` — freeze explicit opponent custody inventory.
- `ops/rebuild-2026-09-29/7.1/test_update_trim_manifest.py` — test preservation, committed inputs, idempotency and drift refusal.
- `ops/rebuild-2026-09-29/7.1/update_trim_manifest.py` — register import reasons and task inventory from pinned inputs.
- `scripts/check_opponent_import.py` — validate strict opponent import and trace custody.
- `scripts/kaggriculture_parity/generate_traces.py` — load pinned original Python submissions with isolated per-seat lifecycle.
- `tests/owl/kaggriculture/test_opponents.py` — declare learned-seat checks skipped until the Task 1.4 binding exists.
- `tests/scripts/test_kaggriculture_parity.py` — test submission custody and independent original-Python agent instances.
- `tests/tools/test_check_opponent_import.py` — exercise opponent custody mutation attacks.

## 2. Commands, actual checks and red/green evidence

All checks used offline Cargo/uv, CARGO_BUILD_JOBS=2, RAYON_NUM_THREADS=2,
OMP_NUM_THREADS=2, MKL_NUM_THREADS=1, task-local TMPDIR and RUST_TEST_THREADS=1.
No training, GPU, panel, network or Git mutation occurred. Existing model unit
tests ran only as part of required repository preparation. Exact argv/status/
elapsed times are in run2/commands.json. All receipts below are under run2/.
Affected checks were rerun after the final Debug repair; metadata retains prior
attempt times.

| Required command | Actual result | Receipt |
| --- | --- | --- |
| cargo test --locked --offline --manifest-path opponents_rs/Cargo.toml | Exit 101: 19 passed, 1 failed, 0 ignored; unit 12 pass, lifecycle 5 pass, oracle 2 pass/1 fail | opponents-test.log |
| cargo test --locked --offline --manifest-path engine_rs/Cargo.toml | Exit 0: 69 passed, 0 failed/ignored (41 unit, 9 RNG, 19 replay), doc tests 0 | engine-test.log |
| uv run --offline python scripts/check_engine_trim.py | Exit 0 | engine-trim.log |
| uv run --offline python scripts/check_opponent_import.py | Exit 0; custody does not waive parity | opponent-import.log |
| uv run --offline pytest tests/tools/test_check_opponent_import.py tests/tools/test_check_engine_trim.py tests/scripts/test_kaggriculture_parity.py tests/owl/kaggriculture/test_opponents.py -q | Exit 0: 164 passed, 10 skipped | targeted-tests.log |
| uvx --offline --from rust-just just prepare | Exit 101 at strict original-Python opponent comparison; build/fmt/lint/docs lint/typing pass; root Rust 254 passed/4 ignored, engine 69 passed, opponents 19 passed/1 failed; Python stage not reached | prepare.log |

Additional preparations:

- uvx --offline --from rust-just just py-prepare: exit 0; **1,692 passed,
  17 skipped**; formatting/lint/typing/doc freshness pass (py-prepare.log).
- uvx --offline --from rust-just just rs-prepare: exit 101 solely at the
  preserved parity failure; root 254 passed/4 ignored, engine 69 passed,
  opponents 19 passed/1 failed (rs-prepare.log).
- Trim updater: six tests plus 12 unexpected-input mutation subtests pass;
  committed 21d0f45 accepted, byte-idempotent, retained/authored unchanged,
  exactly five excluded reasons amended (updater-green-final.log).
- Direct cookbook/source lint, inventory closure, Markdown, whitespace and
  final custody checks are in closure.log.

Targeted skips: nine literal “needs Task 1.4 binding” cases and the existing
committed live-regeneration test, explicitly skipped on Darwin because it
launches more than two games. Full-Python skips add the seven existing cases:
native grammar binding (1), pinned host CUDA memory (2), FlashAttention CUDA (2),
quantization backend (1), native Kaggriculture evaluation environment (1).

Test-first and mutation receipts:

- lifecycle-red.log: unresolved new API against the empty crate. The current
  suite checks exact registry keys, wrong seat/environment, repeated/skipped
  steps, reset, independent state, action SHA-sequence/bank determinism, a
  changed seed changing actions, and rejection of non-default matches.
- reset-red.log / reset-green.log: resetting onto the existing episode initially
  allowed another step-zero action; the new-episode guard now rejects it.
- closure.log: the Debug regression first failed on hidden seed fields.
  EngineOwner now has no Debug; Game formats only snapshot and controller config.
  The regression then passes.
- All five unchanged Starter inline tests pass. Authored view tests check
  BigInt Fibonacci costs, cost conversion, refresh and failure retention.
- Four native visibility games check both seats at steps
  0/1/23/24/250/696/718. Rival-private changes never change actions.
  Own-state positive changes per seat: Starter 7, R04 6, EcoBot 7, E776 4.
  RNG/seed/counters have no exposed getter and cannot be directly perturbed.
- Native mid-episode replay reconstructs 48 transitions for each bot, both seats,
  from fresh controllers. This is lifecycle evidence, not Python-oracle coverage.
- custody-red.log first records missing-checker collection failure (the initial
  shell status was not reliably captured). custody-content-red.log and
  custody-coverage-red.log record later red attacks. Current custody suite has
  60 passes covering schema/path/inventory/hash/reference-byte/source/trace/
  budget and independently recounted coverage drift.
- oracle-red.log: five generator tests fail before implementation.
  oracle-tests.log: 25 pass, one Mac skip. Both independent instances of actual
  original R04/EcoBot/E776 source closures also load and clean up; no extra games.
- oracle-comparator-red.log / oracle-comparator-green.log: comparator initially
  missing, then tests reject numeric representation, array-order and state-map
  order mutations.
- For the retained oracle, each seat's first farmer action is independently
  changed to TASK_7_1_MUTATION. Comparisons reject the exact seat/field at step
  zero, before the natural mismatch. A changed-seed attempt hit the same natural
  step-12 mismatch and is **not credited** as an independent oracle control.
- updater-red.log / updater-green-final.log: six tests fail against the old
  updater, then pass; 12 changes across three accepted states/four inventory
  sections are refused.

## 3. Parity per bot/seat, coverage and fixture custody

Exactly one original-Python game was generated, with default configuration and
719 transitions. Original sources are Kaggle 1.32.7 Starter and sibling commit
e8884aae82eddeb7a1aeae99ecceeca7c830d67e. The dedicated manifest records all source
hashes. Python submissions were never copied into this repository.

| Bot | Seat | Traces / seed | Actions compared / matched | First mismatch |
| --- | --- | --- | --- | --- |
| starter | 0 | 1 / 20260929 | 13 / 13 | None in prefix; full game not qualified |
| starter | 1 | 0 / none | 0 / 0 | Not evaluated |
| r04 | 0 | 0 / none | 0 / 0 | Not evaluated |
| r04 | 1 | 1 / 20260929 | 13 / 12 | Step 12, hands[2][0]: native WEST, Python NORTH |
| ecobot | 0 | 0 / none | 0 / 0 | Not evaluated |
| ecobot | 1 | 0 / none | 0 / 0 | Not evaluated |
| e776 | 0 | 0 / none | 0 / 0 | Not evaluated |
| e776 | 1 | 0 / none | 0 / 0 | Not evaluated |

Both actions match through step 11; 12 applied transitions match complete
public/private state including map order, statuses, typed rewards and completion.
The comparator reconstructs from configuration/seed, uses fresh native
controllers, applies recorded Python actions and remains failing at step 12.

The first internal difference is step-zero R04 anchors. Python 3.12.13
sum([0.35] * 25) is 8.75; native sequential reduction is 8.749999999999996.
The sector threshold puts the fifteenth point in a different group. The anchor
penalty later changes Hungarian assignment at step 12. Original Python
reproduces all 13 frozen actions; a diagnostic changing **only** its
compute_anchors sum binding to sequential addition reproduces all 13 native
anchors/actions. This localizes the native port mismatch against the actual
runtime, without implicating engine state or the view. Original competition
runtime remains unknown. Source lines, arithmetic, reproducer and counterfactual
are in run2/r04-mismatch.md and its debug JSONs.

An earlier harness compared reward JSON 0 against native f64 0.0. It was
corrected to the existing kernel comparator's exact typed-f64 contract.
Action/public/private numeric representation comparisons remain strict.
No import, trace action or equality rule was relaxed to hide the R04 mismatch.

Coverage is the **whole stored Python trace**, distinct from the 12-transition
native matched prefix. The checker independently recounts each field:

| Bot/seat | Openings | Day-reset actions | Steps with own weeds | Quantity/index proxy | Rejected joint steps | Added hands | Final-day SELL orders | Mid-episode replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| starter/0 | 1 | 29 | 527 | 0 | 0 | 0 | 0 | 0 |
| r04/1 | 1 | 29 | 201 | 0 | 0 | 285 | 12 | 0 |

The other six bot-seat combinations have zero oracle observations in every
category. Market inventory is a signed price index, not bounded stock:
quantity/index is only a submitted-input proxy. **Actual market shortages and
individual rejected orders are unmeasured.** Hires count positive hand deltas,
not attempts; final-day sales count submitted SELL orders, not proceeds.
Original-Python mid-episode reset/replay is uncovered.

Frozen oracle: opponents_rs/fixtures/oracle/oracle-00-starter-vs-r04.jsonl.gz.
**178,476 / 4,000,000 bytes**, SHA-256
**39b8bc35c223594d2d6740fc35d0b8242e130351e0d69b2e4bc34594976cf97e**.

The five byte-exact native inputs total 335,571 bytes, separately from the oracle
budget. E776's 117,954-byte tape is executable policy data, not new parity
evidence. Its SHA-256 is
da0d5d1bd326cb5bf068c2065ba1fe8f7e644107db806d7f9a1eae4dafd89692.
All five pinned source/data hashes are in OPPONENT_MANIFEST.json and the
preserved native-api-probe receipt.

## 4. Deviations from the brief/prompt, with reasons

- Used the authorized v3 snapshot view with an opaque holder and safe Debug;
  frozen engine files and imported bytes remain unchanged except allowed trim
  bookkeeping. No policy_rows, copied dispatcher or root dependency was added.
- Stopped after the first trace, following the explicit mismatch stop rule.
  Other seats/bots, second seed and later native qualification were not widened.
  The real failure is neither ignored nor marked expected.
- Engine acceptance is joint-transaction acceptance recorded per seat. Per-seat
  execution metrics are private, so the runner exposes public joint market
  metrics and makes no per-order success claim.
- from_engine trusts its caller's original Config because the frozen engine has
  no getter. Matches support default config only; Starter inline tests retain
  custom configurations.
- The pre-existing multi-game regeneration test explicitly skips on this Mac
  for the two-game bound; all non-live generator checks still run.
- One live Python game completed in 1.2s generator wall (1.78s launch total),
  but sandboxed /usr/bin/time could not read kern.clockrate. Its RSS is unmeasured,
  so no measured <1GB claim is made. Future summaries use getrusage and enforce
  bounds. Loader-only RSS was 48,136,192 bytes, not a game-memory measurement.
- Corrected the false owner attribution: Claude's original placement prompt
  supplied the STOP instruction. Commit 21d0f45 and unchanged probe receipts
  preserve that history; the renamed Reference describes current implementation.

## 5. Open questions and unresolved items

- R04 needs an explicit required-runtime contract and authorized corrected/pinned native source before parity can qualify; current imports cannot be edited.
- EcoBot/E776 original parity, Starter seat 1, R04 seat 0, later native transitions and additional seeds remain untested after the mandated stop.
- Actual market shortages, individual rejected orders and original-Python mid-episode replay remain uncovered.
- EcoBot/E776 software-license custody is unresolved; do not redistribute. R04 has no agent-level PROVENANCE.md.
- Nine learned-seat/auto-reset tests require Task 1.4's real binding; no substitute was created.
- Snapshot cloning cost is unmeasured; this evaluation view is not the training hot path.
- First-game RSS and original competition interpreter version were not established.
- just prepare remains red on the real oracle mismatch; Task 7.1 parity completion cannot be claimed.
