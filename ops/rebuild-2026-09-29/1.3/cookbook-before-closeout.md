---
type: "Reference"
title: "Structured observations preserve legal state and order"
description: "Task 1.3 implementation in progress: all 29 native observation fields and 54 tests pass; oracle, binding, schema and optimized timing qualification remain pending."
tags: ["kaggriculture-v3", "adaptation", "observation"]
status: "in-progress"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.3.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/progress.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-a.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/r1-audit.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-b.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/c8-bc-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/c6-red-rival-hire-mutation.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/d5-restored-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/d4-red-transpose-mutation.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/e10-restored-module-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/e9-red-reversed-ranks.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/f9-final-module-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/f7-pending-bonus-red.log"
  - resource: "repository:src/lib.rs"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/buffers.rs"
  - resource: "repository:src/kaggriculture/observe.rs"
  - resource: "repository:src/kaggriculture/config.rs"
  - resource: "repository:src/kaggriculture/tests.rs"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:Cargo.toml"
  - resource: "repository:Cargo.lock"
  - resource: "repository:src/rules_engine/generation.rs"
  - resource: "repository:docs/rules-engine.md"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_features.rs"
---

# Structured observations preserve legal state and order

This is an in-progress adaptation record, not encoder qualification. The reviewed
Task 1.3 brief specifies 29 named caller-owned buffers, exact side tensors, own
private inventory order, strict engine-shaped tiles, and one public snapshot per
environment. Task 2.1's shared Python schema must merge before Task I can pass;
no second schema is authorized.

## Executed adaptation

Root Cargo dependencies now include the unchanged standalone Kaggriculture kernel,
ordered arbitrary-precision JSON and exact integer hire-cost dependencies, added
with offline Cargo commands. Root feature unification requires the test-only
`fixture_float` repair in Orbit generation fixtures. Production Orbit generation
and parity tolerances are unchanged. The new decimal regression passed before
unification, failed afterward with `invalid type: map, expected f64`, and passed
with Number conversion. The root suite at this checkpoint has 157 passes and two
ignored tests; the trim checker passes. Detailed timings and commands are in the
Task A receipt. No engine bytes were edited.

Current inventory: `Cargo.toml`, `Cargo.lock`, `src/rules_engine/generation.rs`,
`docs/rules-engine.md`, `docs/rules-parity-coverage.md`, this note/index/log and
`ops/rebuild-2026-09-29/1.3/` receipts, resource monitor and coverage audit.
Task B adds `src/lib.rs` module registration, `src/kaggriculture/mod.rs`, explicit
`buffers.rs`, error types in `observe.rs`, config scaffolding and `tests.rs`.
`docs/rl-api-specs.md` lists the 29 fields. The restart Decision/index relabels
Task 1.1 isolation as historical. No state encoder was implemented by B.

Task B passes 11 boundary tests and all-target Clippy. Tests exercise every field
short/long, checked capacity, both seat rows, serial/two-worker byte agreement,
unchanged allocations and atomic unequal-E rejection. Disabling length rejection
makes the test fail; restoration passes. Independent read-only review found no
blocking boundary issue. Scope/details are in `task-b.md`; full observation and
Python admission remain pending.

Task C binds an immutable checked config to its game and prepares one snapshot.
Constructor admission preserves exact i64 values; scaled values must remain
finite but have no invented normalization ceiling. Exact BigInt Fibonacci costs
use each farm's public hire count; a zero multiplier does not iterate to a huge
count. Nine config and four hire tests pass (24 combined B/C tests), including
literal costs and native HIRE bank deltas. Replacing the rival's count with the
own count fails two tests; restoration and Clippy pass. Pinned `PyInt` has no
Display implementation: constructor-only exact integer serialization followed
by checked decimal parsing replaces that unavailable brief API. No live state
serialization or engine accessor is added. Tiles, actors and market writing are
still pending at that checkpoint.

Task D adds strict engine-constructor tile key sets, actual integer/bool/string
admission, applicable sentinels, wide date arithmetic and all seven tile tensors.
Nine tests fail against the prior partial writer, then all ten tile tests pass.
The independent shape scanner covers all 2,880 official snapshots and 576,000
tiles, including every constructor shape. A transpose mutation fails the
asymmetric cell assertion; restoration and Clippy pass. Read-only independent
review found no blocking C/D issue. Actor/storage and market/mask encoding remain
pending at that checkpoint; the shape scan does not establish new engine-rule
differential parity.

Task E fills all 241 own and rival actor slots, exact own inventory/storage and
insertion-order ranks, and public player channels. Only the requesting private
state enters the private writer. Twelve semantic failures become twelve passes;
the full B–E module has 46 passes. Tests distinguish absent keys from zero-valued
keys, changed-value replacement from remove/reinsert, both-seat privacy with
positive controls, complete seat swapping, dense-to-sparse clearing, sub-f32 bank
differences, and i128 sums of twelve i64::MAX counts. Reversing actor/shed ranks
fails two regressions; restored tests and Clippy pass. Shops/market/masks and
complete-row validation remained Task F work at that checkpoint.

Task F completes all 29 fields: ordered duplicate shops, signed exact market
integers, all public context, constant live-row status, configured order limits
and potential-frame masks. Five context tests fail before implementation. The
diagnostic `check_row` then rejects 72 deliberately corrupted rows, with a
separate fractional scaled-bonus red before its integer-consistency repair.
All 54 B–F tests and Clippy pass. Every successful test encoding is row-checked;
two-environment staging preserves all published bytes on a late market error,
serial/two-worker outputs agree, and dense-to-sparse reuse clears old fields.
This proves output publication behavior, not Task 1.4 game/seed/terminal rollback
or reuse fences. The Python boundary, frozen corpus and timing remain pending.

## Corpus limit found before generation

Streaming the 384 required official states finds no state with more than 16
actors (maximum 13). The fixed seeded policy can produce at most 6, 4, 2, 2, 9
and 4 actors in the six profiles: each successful HIRE adds one hand, the policy
submits at most two cyclic market entries, and all hands reset daily. Thus the
R1 quota of four non-synthetic states above 16 actors is unreachable under the
specified policy. Keep the quota; final corpus publication requires a reviewed
recipe correction. Other measured official counts are retained in `r1-audit.json`.
This is independent source/arithmetic evidence, not a completed oracle comparison.

## Scope and consequences

Existing-concept search covered the [[native-game-semantics-use-v3-owned-buffers|native
buffer Reference]], [[full-turn-intentions-coordinate-batched-action-heads|information
loss discussion]], and [[../decisions/restart-the-port-from-isaiahs-clean-base|restart
Decision]]. Those historical reference-branch implementations are design/oracle
sources only. Future consumers must use the checked root boundary and preserve
ranks and seat privacy. Snapshot cloning allocates; no zero-copy or performance
claim is established. Full corpus reconstruction, native buffer tests, Python
schema integration and optimized timing remain pending. No training/GPU run or
new Python-engine differential parity has been performed.
