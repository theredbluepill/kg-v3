---
type: "Reference"
title: "Snapshot view isolates byte-exact evaluation opponents"
description: "Task 7.1 preserves four controllers behind a v3-owned snapshot view; original-Python parity stops at R04 step 12, so lifecycle and custody checks do not qualify the opponents."
tags: ["kaggriculture-v3", "adaptation", "opponents", "parity"]
status: "implemented-parity-mismatch"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "external-repository:/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/task-7.1-impl-prompt.md"
  - resource: "external-repository:/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/task-7.1-claude-view-probe-lib.rs"
  - resource: "repository:opponents_rs/src/lib.rs"
  - resource: "repository:opponents_rs/src/native_agents.rs"
  - resource: "repository:opponents_rs/src/registry.rs"
  - resource: "repository:opponents_rs/src/runner.rs"
  - resource: "repository:opponents_rs/src/view_tests.rs"
  - resource: "repository:opponents_rs/tests/lifecycle.rs"
  - resource: "repository:opponents_rs/tests/oracle_parity.rs"
  - resource: "repository:opponents_rs/OPPONENT_MANIFEST.json"
  - resource: "repository:opponents_rs/fixtures/oracle/MANIFEST.json"
  - resource: "repository:scripts/check_opponent_import.py"
  - resource: "repository:tests/tools/test_check_opponent_import.py"
  - resource: "repository:scripts/kaggriculture_parity/generate_traces.py"
  - resource: "repository:tests/scripts/test_kaggriculture_parity.py"
  - resource: "repository:tests/owl/kaggriculture/test_opponents.py"
  - resource: "repository:justfile"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/7.1-opponents.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/native-api-probe.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/native-api-probe-red.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/run2/r04-mismatch.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/run2/coverage.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/run2/oracle-parity.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/run2/updater-green-final.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/run2/opponents-test.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/run2/closure.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/plan.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/update_trim_manifest.py"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/test_update_trim_manifest.py"
  - resource: "repository:docs/rules-parity-coverage.md"
---

# Snapshot view isolates byte-exact evaluation opponents

Task 7.1 imports Starter, R04, EcoBot and E776, plus E776's executable policy
tape, byte-exact from `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0` into the
standalone edition-2024 `opponents_rs` crate. The frozen engine stays unchanged;
no `policy_rows`, all-controller dispatcher or root-crate dependency is added.
The controllers remain external scripted opponents. Their state and identity
belong only to evaluator bookkeeping, never the learned actor/critic, rewards,
normalization or checkpoint selection.

## Information and lifecycle boundary

The authored crate-root `Game` owns the native engine through an opaque sibling
module and exposes a current `StepSnapshot`, refreshed at construction and after
each successful step. The opaque module is necessary because Rust child modules
can access their parent's private fields. No controller can obtain the engine
reference, RNG, seed or hidden counters through this view. Final review caught
a derived-`Debug` leak through formatting: the holder no longer implements
`Debug`, and `Game` formats only snapshot/config. A regression fails before
that correction and passes afterward; `run2/closure.log` preserves both. The only extra
configuration is Starter's hire-cost multiplier, converted through serde with
explicit failure, plus an authored BigInt `fib` helper. `from_engine` requires
the caller to provide that engine's original configuration because the frozen
engine exposes no configuration getter.

The snapshot includes public state, both seats' private state, statuses and
rewards. It is not a redacted per-seat tensor observation. Rival-private
perturbations therefore test the actual imported controller paths; own-private
or public-state perturbations supply a positive control. Hidden engine fields
cannot be perturbed through this interface and are excluded by construction.
The full snapshot is cloned on refresh. This is an evaluation path, with no
Mac throughput measurement or training-path claim.

The registry admits exactly `starter`, `r04`, `ecobot`, `e776`. The seat wrapper
owns a controller per environment, seat and episode, requires exactly one call
per step, and rejects a wrong seat, repeated/skipped step or mismatched episode.
An explicit reset requires a fresh episode and creates fresh state. `play_match` accepts only the pinned
default game configuration, submits the official JSON without grammar
truncation or reinterpretation, and records actions, engine acceptance, controller
errors, raw banks and winner. Acceptance is the engine's joint transaction
outcome recorded for each seat; available market metrics are joint rather than
seat-attributed. It is not a claim that every requested command executes.

## Verification and limits

The final opponent test command reports 19 passes and one failing
original-Python comparison, with no ignored tests. The 69 frozen-engine tests
pass; required targeted Python checks report 164 passes and ten skips (nine
Task 1.4 binding cases, one multi-game regeneration outside the Mac bound).
Both custody checkers pass. Mutation checks independently reject tampered first actions for both
seats. The changed-seed oracle attempt reproduced the already-known step-12
mismatch and does not count as an independent control; native same/different-seed
determinism is a separate check. These successes do not override the parity failure.

The first original-Python trace is Starter in seat 0 versus R04 in seat 1,
seed `20260929`, with 719 Python transitions. The native comparator stops at
step 12, seat 1, `action.hands[2][0]`: native `"WEST"`, Python `"NORTH"`.
It compared 13 actions per seat; Starter matched 13, R04 matched 12. The first
12 applied transitions matched full state including object order. This is a
localized original-submission behavior mismatch, not a passing parity result.
Widening stops here, preserving the imported bytes and failing comparison.
The source cause is a floating reduction difference: actual Python 3.12.13
sums 25 weights of 0.35 to `8.75`, while the native sequential reduction gives
`8.749999999999996`. The resulting angular-sector threshold changes anchors
at step zero and assignment at step 12. A diagnostic that replaces only the
original `compute_anchors` function's `sum` binding with sequential accumulation
matches all 13 native anchors/actions, including `WEST`. The original module
independently reproduces all 13 frozen Python actions. This identifies a native
port mismatch to the requested original-source oracle under this runtime;
the original competition runtime remains unknown. The engine/view is not
implicated by this bounded comparison. Source lines, arithmetic and controls
are in `run2/r04-mismatch.md`; no diagnostic alteration becomes an oracle.

The trace's full-Python coverage is distinct from that 12-transition native
matched prefix. R04 seat 0, Starter seat 1, and both seats for EcoBot/E776 have
no original-Python action comparison. Additional pairs/seeds and any coverage
category absent from the first trace remain unqualified. The lifecycle tests
cannot fill those denominators. The complete Python trace records opening/day
reset/weed-step/added-hand/final-day SELL counts of `1/29/527/0/0` for Starter
and `1/29/201/285/12` for R04. It has no whole-step rejection or mid-episode
replay. The quantity-versus-inventory-index proxy is zero for both seats and
does not establish engine-confirmed market shortages or individual-order
rejection. Those categories remain gaps. The compressed oracle is 178,476 of
4,000,000 allowed bytes; its exact SHA-256 is in the manifest and coverage page.

The source-level boundary, lifecycle checks and original-Python comparison are
separate claims: compiling the imported Rust or reproducing its own actions
cannot establish original-submission parity. The original source is Kaggle
1.32.7 Starter or the pinned sibling submissions, with fresh module/agent state
for every seat and game. Action, public and private state comparisons preserve
JSON number representation; rewards use the frozen engine's typed-f64 semantics,
as the existing kernel comparator does. The initial reward `0` versus `0.0`
harness mismatch was corrected before the R04 behavior mismatch was reached.

Default-config-only support is deliberate: the controllers hardcode calendar
and rule constants. Custom configs accepted by the general v4.1 environment
contract are not thereby supported by these opponents. Task 1.4 still gates
learned-seat binding checks; explicit skipped tests describe that future API.
No playing-strength, held-out panel or generality result follows from bounded
CPU qualification.

Original-submission provenance is separate from byte equality. EcoBot and E776
explicitly declare no software license and must not be redistributed; the
reference engine's license does not resolve that gap. R04 has no agent-level
`PROVENANCE.md`. The source audit at
`ops/rebuild-2026-09-29/7.1/python-oracle-source-audit.json` preserves the original
notice text and hashes; no Python submission source is copied into this repo.

## Adaptation custody and history

The dedicated opponent manifest/checker owns imported bytes, authored support,
Python source identities and oracle trace inventory/hash/size/budget. The
engine trim updater accepts the exact `b8747b6` integration input, committed
`21d0f45` run-1 manifest, or its exact final output. It preserves retained and
authored engine entries, updates exactly five excluded reasons to identify their
new `opponents_rs` copies, and registers this task's non-engine paths. Updater
checks first failed against the old signature; six tests then passed, including
12 unexpected-input mutations and the real committed input, with red/green
receipts under `run2/`.

Run 1 stopped because Claude's original placement prompt explicitly required
stopping on private engine items: `fib`, `Game.config` and five missing
accessors. This was Claude's design instruction, not an owner directive. Commit
`21d0f45` preserves that compiler finding and its original receipts. Claude's
revised snapshot-view design resolves the placement boundary without changing
engine bytes. This current Reference replaces the stop-only note and removes
its unsupported `user-directive` attribution; the original failure remains
historical evidence, not a current implementation blocker.

The coherent adaptation inventory is `opponents_rs/`, the import checker and
its tests, original-Python trace generator/tests, binding-dependent skipped
tests, `justfile`, coverage docs, this note/index/log, the trim manifest/updater
and compact Task 7.1 receipts. Exact paths are enumerated by the updater.
Existing-concept search covered opponent import, snapshot/accessors, private
engine fields, parity, custody and negative evidence; this revises the existing
concept rather than creating a parallel claim. Independent checks are the
pinned Git blobs, original Python submissions, compiler and mutation tests.
Future integration must preserve these boundaries and re-open any uncovered
behavior before claiming broader qualification. R04 parity needs an explicit
runtime contract and an authorized change to the byte-exact-source requirement,
or a newly pinned corrected upstream controller; relaxing equality or changing
the oracle is not a resolution. No result board is warranted.
