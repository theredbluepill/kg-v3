---
type: "Reference"
title: "Snapshot view isolates byte-exact evaluation opponents"
description: "Task 7.1 runs four byte-exact controllers behind a v3-owned snapshot view; all match their original Python submissions on eight CPython 3.11 oracle games (11,504 actions, both seats) and after mid-episode reconstruction (1,152 resumed actions). CPython 3.12 changes R04 through compensated float sum()."
tags: ["kaggriculture-v3", "adaptation", "opponents", "parity"]
status: "implemented"
generated: {"by": "openai/codex; revised by anthropic/claude", "at": "2026-09-29"}
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
  - resource: "repository:ops/rebuild-2026-09-29/7.1/review/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/review/oracle-parity.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/review/mutations.log"
  - resource: "repository:opponents_rs/fixtures/replay/REPLAY.json.gz"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/verify-r1/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/verify-r1/parity-replay.json"
  - resource: "repository:ops/rebuild-2026-09-29/7.1/verify-r1/replay-controller-mutation.log"
  - resource: "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/references/kaggle-simulation-container.md"
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

**Original-submission parity (Claude review).** Eight oracle games were
generated from Kaggle 1.32.7's own Python engine with the original submissions
(Kaggle Starter; sibling R04, EcoBot and E776 at `e8884aae`, hash-verified, a
fresh module per seat and game) on CPython 3.11.15, default configuration,
seeds 20260929–20260936. The pairs rotate so every bot plays both seats twice.
Native controllers, fed the native engine state rebuilt from configuration and
seed, match every recorded Python action on both seats: 8 × 2 × 719 = 11,504
actions, with public/private state, statuses, rewards and terminal banks equal
after all 5,752 transitions. Traces total 1,779,187 of 4,000,000 bytes.
A tampered first action fails for each seat of each trace.

**Why CPython 3.11.** Codex's first oracle ran on CPython 3.12.13 and failed at
R04 step 12 (native `WEST`, Python `NORTH`); its localization
(`run2/r04-mismatch.md`) traced this to `sum()` of 25 weights of 0.35: 3.12's
compensated float `sum()` gives 8.75, sequential accumulation 8.749999999999996.
CPython 3.12 changed `sum()`; the native port reproduces sequential summation.
Kaggle's simulation image `gcr.io/kaggle-images/python:v163`, read directly in
v2, runs CPython 3.11.13, so 3.11 is the competition runtime. The generator,
custody checker and Rust oracle test now refuse oracles from any other
interpreter. The 3.12 trace remains in history at `7ae9bbf` as a negative
control: the comparator does detect a one-hand decision difference. Limit: the
simulation build may use another image tag; the v2 reading is the evidence.

**Mid-episode replay against Python (verify r1).** Codex's verification
rejected the first qualification because replay was checked only natively and
stopped at the reconstruction boundary. The generator now rebuilds fresh
original controllers in both seats from each oracle's recorded prefix at steps
37 (mid-day), 360 (day reset) and 695 (whole final day follows): each prefix
observation must yield the recorded action and the rebuilt state must equal the
trace. The controllers then resume for 24 steps. The 24 frozen cases
(`fixtures/replay/REPLAY.json.gz`, CPython 3.11.15, bound to the oracle
manifest hash, regenerated byte-identically) cover every bot in both seats at
every point. Fresh native controllers rebuilt through the step-zero lifecycle
match 1,152 / 1,152 resumed actions and all 24 final states. Tampered resumed
actions (first action per seat, a last action) and a tampered final state fail
at the exact case, step and seat; a restored step-700 controller mutation fails
all eight step-695 cases on resume. Limit: prefixes are the controllers' own
play, so resumed actions equal the contiguous traces; foreign-prefix states are
untested.

**Other checks.** 22 opponent-crate tests pass: Starter's five pinned inline
cases, view/visibility, lifecycle, determinism and the five-test oracle suite. Rival
private perturbations never change an action at seven checkpoints per seat;
own-state perturbations do (positive control). Mutations in `review/mutations.log`
(dropping the step guard, a stale snapshot, a wrong `fib`) each fail tests.

**Coverage gaps.** The oracles exercise openings, 29 day resets per seat, weeds,
hires (R04, EcoBot, E776) and final-day sells. No trace has a rejected step, a
buy above the market inventory index. Explicit fresh-controller reset is
checked natively only (lifecycle test); mid-episode replay has the Python
oracle above. Starter never hires in these games.

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
Python source identities and oracle trace inventory/hash/size/budget. Its
default mode pins original entry hashes without the sibling repository, so
`just prepare` stays portable to pods and containers; `--original-sources`
re-reads every original file on the owner's machine. The engine trim updater accepts the exact `b8747b6` integration input, committed
`21d0f45` run-1 manifest, committed `7ae9bbf` run-2 output, or its exact final
output. It preserves retained and
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
behavior before claiming broader qualification. Reopen parity if Kaggle's simulation runtime moves to CPython 3.12 or later:
R04's decisions then follow compensated float summation, which the byte-exact
native port does not reproduce. No result board is warranted.
