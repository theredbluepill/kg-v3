---
type: "Reference"
title: "Structured observations preserve legal state and order"
description: "Checked native observations and caller-owned buffers are implemented; R1 corpus qualification, the Task 2.1 schema merge and optimized timing remain blocked."
tags: ["kaggriculture-v3", "adaptation", "observation"]
status: "implemented-with-qualification-blocks"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.3.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/progress.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-a.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-b.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-c.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-d.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-e.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-f.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-g.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-h.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-i.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/task-j.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/timing.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/g-input-generation.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/g-input-audit.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/g-recorder-compile.json"
  - resource: "repository:Cargo.toml"
  - resource: "repository:Cargo.lock"
  - resource: "repository:src/lib.rs"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/buffers.rs"
  - resource: "repository:src/kaggriculture/config.rs"
  - resource: "repository:src/kaggriculture/observe.rs"
  - resource: "repository:src/kaggriculture/tests.rs"
  - resource: "repository:src/kaggriculture/oracle_corpus.rs"
  - resource: "repository:src/rules_engine/generation.rs"
  - resource: "repository:python/owl/rs.pyi"
  - resource: "repository:scripts/kaggriculture_observation_oracle/record.rs"
  - resource: "repository:scripts/kaggriculture_observation_oracle/regenerate.py"
  - resource: "repository:tests/tools/test_observation_oracle_custody.py"
  - resource: "repository:tests/kaggriculture/test_observe.py"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:docs/rules-engine.md"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_features.rs"
---

# Structured observations preserve legal state and order

The root native encoder and its explicit-header Python binding implement the
reviewed contract v4/schema 3. Task 1.3 is **incomplete**: the exact R1 recipe
cannot qualify its corpus, the actual Task 2.1 Python schema has not merged, and
the optimized timing build exceeded the Mac memory limit. Implemented controls
and those missing qualifications must remain distinct for downstream consumers.

## Boundary and adaptation inventory

- `Cargo.toml`, `Cargo.lock`: offline Cargo-managed root dependencies on the
  unchanged kernel, ordered arbitrary-precision JSON and exact integer costs.
- `src/rules_engine/generation.rs`: test-only Number decoding repairs L4 feature
  unification. Production Orbit generation and parity tolerances are unchanged.
- `src/lib.rs`, `src/kaggriculture/mod.rs`, `python/owl/rs.pyi`: register the
  exact 29-argument writer in the existing extension, with typed fallible NumPy
  borrowing and matching Python signature.
- `src/kaggriculture/config.rs`, `observe.rs`: bind immutable checked config to
  the game; validate exact counts, constructor-shaped tiles and finite derived
  values before writing both legal seat views from one public snapshot. Exact
  BigInt Fibonacci costs use each farm's own public hire count. Pinned PyInt has
  no Display API, so constructor-only exact integer serialization plus checked
  parsing supplies the required conversion; no live-state serialization is used.
- `src/kaggriculture/buffers.rs`: 29 named typed buffers, checked lengths and byte
  products, reusable storage, safe serial/Rayon views and equal-capacity publish.
  The Python boundary checks complete shapes, native dtype, C layout, alignment,
  writable borrows and disjoint byte intervals before making mutable slices.
  Every environment is prepared before any output write; no replacement output
  arrays are allocated. Borrow guards stay alive across detached native work.
- `src/kaggriculture/tests.rs`: hand-derived values, private noninterference with
  positive own-seat controls, exact insertion ranks, huge counts, atomic failure,
  serial/two-worker equality, reuse, snapshot counter and optimized timing test.
- `src/kaggriculture/oracle_corpus.rs`: test-only deterministic input recipes,
  legacy-domain admission, tensor-only 8,176-offset reconstruction and streaming
  bitwise comparison. Legacy limits do not constrain the production encoder.
- `scripts/kaggriculture_observation_oracle/{record.rs,regenerate.py}`: the
  recorder calls the actual pinned reference feature function in a verified
  temporary export; custody validates source identity, record/seat hashes,
  quotas, order and sizes. Source identity is fixed before execution and checked
  before installation; the exported recorder bytes must match that identity.
- `tests/tools/test_observation_oracle_custody.py`: corruption, source drift,
  exported-recorder integrity, lossless byte-plane and resource-guard controls.
- `tests/kaggriculture/test_observe.py`: direct imports of the real schema, all
  named buffers, pointer/layout/alias/rollback assertions and streamed corpus
  acceptance. There is no substitute schema or missing-dependency skip.
- `docs/rl-api-specs.md`, `docs/rules-engine.md`, `docs/rules-parity-coverage.md`:
  current interface, L4 repair, observation coverage and qualification limits.
  The restart Decision and its index label root isolation as the historical
  Task 1.1 checkpoint. This note, references index and log record the adaptation.
- `ops/rebuild-2026-09-29/1.3/`: plans separated from actual red/green, mutations,
  source hashes, bounded command logs, independent native smoke, quota audit and
  timing receipts. The pre-closeout dirty note and checksum are preserved there.

All new implementation is in the root crate. No vendored bytes, model, grammar,
reward, policy or PPO implementation is changed. No Git write, training, GPU or
network run was performed.

## Independent evidence and its limits

The L4 decimal test passed before dependency unification, failed afterward with
`invalid type: map, expected f64`, then passed with Number conversion. The 54
B–F tests cover native fields and boundaries; the independent tile scan reads
2,880 official states / 576,000 tiles. Eight producer controls, eight legacy
reconstruction/stream controls and five added-fact tests pass. Hand expectations
cover every legacy offset and all reviewed boundaries, including the last actor
and inventory element. Sixteen temporary H mutations fail, then restoration
passes. Earlier length, coordinate, rank and rival-hire mutations also fail and
are restored. The snapshot mutation observes two acquisitions against one;
restoration passes with unchanged output allocations.

The current custody suite passes 43 tests. Nine source-custody regressions fail
before the capture/recheck repair and pass afterward. The native Python smoke
passes 55 checks and explicitly reports `schema_checked=false`; Clippy, targeted
Ruff and stub mypy pass. Real-schema collection fails with
`ModuleNotFoundError: No module named 'owl.kaggriculture'`. Complete final command
outcomes, including failures, belong to `results.md`; these local controls do
not replace the missing schema/corpus integration.

Final root Rust tests have 233 passes, one missing-corpus failure and four
ignored diagnostics; standalone engine tests pass 59/59 and the trim checker
passes. The broad fast Python suite without the undefined `reference` extra
passes 1,056 tests with three platform skips. Both Rust preparation commands
stop at the same corpus failure; Python preparation stops at the missing-schema
import after formatting/lint/mypy pass. Full preparation also passes docs lint;
the separately reached docs-fresh command passes. These failed integration gates
remain failures, not completion of Task 1.3.

Actual generation emits all 512 inputs: 384 selected official, 96 seeded and 32
dense. Independent Python recount agrees with Rust. Non-synthetic coverage has
zero >16-actor states versus required four; the fixed HIRE cycle and daily reset
bound all seeded profiles below 17 actors, and the official selection's maximum
is 13. Other R1 quotas pass; zero reordered non-synthetic sheds uses only the
reviewed exact d30/d31 dense exception. Quotas and recipe are unchanged. No final
reference fixture is installed, so 1,024-seat reference comparison, mismatch
count, final compressed sizes and fixture hashes are unavailable. The source-only
recorder compile preserves all 125 original exported reference files, but does
not run feature recording or establish parity.

The fat-LTO release build stops after 53.81 seconds at sampled aggregate RSS
1,052,393,472 bytes. No test body or phase costs are measured. `timing.json`
contains the exact pod handoff command; debug costs are not substituted. The
snapshot still clones both private inventories internally. Stable output storage
is not a zero-copy or end-to-end throughput claim.

## Consequences and remaining work

Existing-concept search covered the [[native-game-semantics-use-v3-owned-buffers|native
buffer Reference]], [[full-turn-intentions-coordinate-batched-action-heads|information
loss discussion]], [[bc-bootstrap-uses-native-replay-features-and-current-heads|BC
boundary]] and [[../decisions/restart-the-port-from-isaiahs-clean-base|restart
Decision]]. Their older implementations remain reference-branch sources, not
qualification for this rebuild. This episode supports a Reference, not a Lesson
or result-ranking board.

Consumers must preserve exact indices, zero-versus-absent insertion ranks and
seat privacy; externally corrupted indices must be rejected before gather.
After a reviewed R1 recipe correction, regenerate the unchanged full-size oracle
and run all offsets plus the actual-schema corpus test. Merge Task 2.1 before
claiming Task I or 1.3 complete. Run the optimized timing command on the pod;
only Task 1.4/6.1 representative complete-work measurements can reopen the pinned
snapshot API. Task 1.4 separately owns huge-aggregate transition admission,
game/seed/terminal rollback and buffer reuse fences. The unavailable pinned weed
JSON-schema range remains the reviewed Task 7.5 parity item; no <=1 cap is invented.
