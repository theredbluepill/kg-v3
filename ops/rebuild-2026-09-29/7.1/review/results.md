# Task 7.1 Claude review

Branch `kg/rebuild-7-1`, base `b8747b6e8acece5f561d09a75bb914364a60ac05`.
Codex run 1 (`21d0f45`) stopped at the engine API boundary; run 2 (`7ae9bbf`)
implemented the snapshot-view design. This review checked every changed file
against brief `ops/rebuild-2026-09-29/briefs/7.1-opponents.md` and finished the
incomplete parity work.

## Findings and resolutions

1. **Parity oracle ran on the wrong interpreter (resolved).** Run 2's single
   oracle came from CPython 3.12.13 and failed at R04 step 12. Codex localized
   it to `sum()` of 25 weights of 0.35 (`run2/r04-mismatch.md`): CPython 3.12's
   compensated float `sum()` gives 8.75, sequential summation 8.749999999999996.
   Kaggle's simulation image `gcr.io/kaggle-images/python:v163` runs CPython
   3.11.13 (read directly in v2: `/Users/poonszesen/kaggriculture-v2/cookbook/references/kaggle-simulation-container.md`);
   the sibling repository's JA25 log entry (`/Users/poonszesen/kaggriculture/cookbook/log.md`)
   records the same step-12 `NORTH`/`WEST` split under CPython 3.12.3 and a pass
   under 3.11. Test-first, the generator, custody checker and Rust oracle test
   now refuse oracles not generated on CPython 3.11 (`python_runtime` in each
   oracle MANIFEST entry). Limit: Kaggle's cloud build may use a different image
   tag; the v163 reading is the evidence.
2. **Corpus widened to both seats for every bot (resolved).** Eight games,
   generated with the sibling repository's CPython 3.11.15 interpreter
   (`/Users/poonszesen/kaggriculture/.venv/bin/python`, kaggle-environments
   1.32.7, used read-only), two games per invocation. Summaries:
   `oracle-gen-{0,2,4,6}.json`, each invocation under 4.1 s wall and under
   465 MB peak RSS (`/usr/bin/time -l`).
3. **`just prepare` needed the owner's sibling repository (resolved).** The run-2
   custody checker re-read original Python blobs from
   `/Users/poonszesen/kaggriculture` on every run, so `just prepare` and
   `prepare-container` would fail on a pod or in a container. The default mode
   now pins the original entry hashes structurally; `--original-sources`
   re-reads every file. Tests: the default mode never calls the source reader;
   an altered E776 dependency hash passes the default mode and fails
   `--original-sources` (this test is skipped where the sibling repository is absent).
4. **Binding-test docstring contradicted Task 1.4 (resolved).** It said the
   learned seat submits JSON; Task 1.4 makes JSON a cold codec boundary. The
   docstring now says the learned seat steps grammar tokens.
5. **README named the wrong engine license (resolved).** It said MIT;
   `engine_rs/LICENSE` is Apache-2.0.
6. **Trim updater could not accept run 2's committed output (resolved).** It now
   accepts `7ae9bbf` as input, registers all eight traces and the review
   receipts, and stays idempotent (7 tests, 12 subtests).

No finding was made against the view, registry, runner, visibility tests or the
byte-exact imports. All five imported files still equal their pinned blobs.

## Parity result

| Trace | Seed | Seat 0 | Seat 1 | Actions matched |
| --- | ---: | --- | --- | --- |
| oracle-00 | 20260929 | starter | r04 | 719 / 719 each seat |
| oracle-01 | 20260930 | r04 | ecobot | 719 / 719 each seat |
| oracle-02 | 20260931 | ecobot | e776 | 719 / 719 each seat |
| oracle-03 | 20260932 | e776 | starter | 719 / 719 each seat |
| oracle-04 | 20260933 | starter | r04 | 719 / 719 each seat |
| oracle-05 | 20260934 | r04 | ecobot | 719 / 719 each seat |
| oracle-06 | 20260935 | ecobot | e776 | 719 / 719 each seat |
| oracle-07 | 20260936 | e776 | starter | 719 / 719 each seat |

11,504 of 11,504 actions match; all 5,752 transitions match public/private
state, statuses, rewards and terminal banks. Oracle bytes: 1,779,187 of
4,000,000. Report: `oracle-parity.json`.

## Non-vacuity

- Oracle comparison: a tampered first action fails for both seats of all eight
  traces (Rust test). The 3.12 trace at `7ae9bbf` is a real negative control:
  the comparator detects a single hand's direction change.
- Runtime guard: generator, checker and Rust tests were red before the
  `python_runtime` implementation (3.12 refusal, missing field, empty field).
- Production mutations (`mutations.log`), each restored afterwards: dropping the
  repeated/skipped-step guard fails the lifecycle test; a stale snapshot after
  `step` fails seven tests; a wrong `fib` fails the Fibonacci test.
- Custody checker: the default mode leaves the sibling source reader uncalled;
  an altered entry hash fails it.

## Checks (actual)

`CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare`: exit 0
(`prepare.log`). Root Rust 254 passed, 4 ignored; engine 41 + 9 + 19 passed;
opponents 12 + 5 + 3 passed; Python 1,699 passed, 17 skipped; engine trim,
opponent custody, format, lint, types, docs lint and docs freshness pass.
`scripts/check_opponent_import.py --original-sources` also passes on this Mac.

## Remaining gaps

- No Python-side coverage of rejected steps, market shortages or mid-episode
  replay; reset and mid-episode replay are checked natively only.
- Starter never hires in the oracle games.
- Default game configuration only; custom v4.1 configs are unsupported.
- EcoBot and E776 declare no software license; do not redistribute.
- Learned-seat tests are skipped until the Task 1.4 binding lands.
- Engine acceptance is per joint action, not per seat or order.
- No strength, panel or held-out result follows from these checks (Task 7.2).
