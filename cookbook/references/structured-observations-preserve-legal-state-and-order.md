---
type: "Reference"
title: "Structured observations preserve legal state and order"
description: "Checked native observations and caller-owned buffers match a frozen 512-state pinned-reference oracle bitwise and pass the real Task 2.1 schema; optimized phase timing remains a pod handoff."
tags: ["kaggriculture-v3", "adaptation", "observation"]
status: "oracle-qualified-timing-pending"
generated: {"by": "openai/codex; revised by anthropic/claude", "at": "2026-09-29"}
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
  - resource: "repository:ops/rebuild-2026-09-29/1.3/r1-blocker.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/claude-review.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/independent-7b5eacd/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/r1-fixes/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.3/retire_grammar_bridge.py"
  - resource: "repository:tests/fixtures/kaggriculture/observation-v3/manifest.json"
  - resource: "repository:Cargo.toml"
  - resource: "repository:Cargo.lock"
  - resource: "repository:src/lib.rs"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/buffers.rs"
  - resource: "repository:src/kaggriculture/config.rs"
  - resource: "repository:src/kaggriculture/observe.rs"
  - resource: "repository:src/kaggriculture/tests.rs"
  - resource: "repository:src/kaggriculture/oracle_corpus.rs"
  - resource: "repository:src/kaggriculture/grammar_kernel_tests.rs"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:scripts/check_engine_trim.py"
  - resource: "repository:tests/tools/test_check_engine_trim.py"
  - resource: "repository:src/rules_engine/generation.rs"
  - resource: "repository:python/owl/rs.pyi"
  - resource: "repository:scripts/kaggriculture_observation_oracle/record.rs"
  - resource: "repository:scripts/kaggriculture_observation_oracle/regenerate.py"
  - resource: "repository:tests/tools/test_observation_oracle_custody.py"
  - resource: "repository:ops/rebuild-2026-09-29/merge-1.3/verify-r1-fix/results.md"
  - resource: "repository:tests/kaggriculture/test_observe.py"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:docs/rules-engine.md"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_features.rs"
---

# Structured observations preserve legal state and order

The root native encoder and its explicit-header Python binding implement the
reviewed contract v4, observation schema 3. A frozen oracle of 512 states and
1,024 seat rows, recorded by the pinned reference `encode_invest`, matches the
tensor-only reconstruction bitwise at all 8,176 offsets. The merged Task 2.1
`check_contract()` accepts every emitted test batch and all 512 records.

One qualification remains open. The optimized timing build exceeded the Mac
memory limit, so phase costs are unmeasured and handed to the pod. Seeded corpus
states come from Claude's reviewed R1 policy correction, `observation-corpus-v2`;
the quota itself is unchanged.

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
  It covers every engine build input, and an undeclared engine module fails.
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

The custody suite passed 43 tests at Codex's handoff and passes 45 after Claude's
watchdog fix. Nine source-custody regressions fail before the capture/recheck
repair and pass afterward. Codex's merge verification of `e197528` found that
source identity omitted the live engine `Cargo.toml` and `py_random.rs`; edits
to either survived every recheck and the output was installed. Claude added
those, `econ_attrib.rs` and a guard that rejects an undeclared engine module.
Seven new tests failed first and pass after the repair; the suite passes 53
(`merge-1.3/verify-r1-fix/`). The committed corpus predates the repair, so its
identity lacks those three hashes; their committed bytes are unchanged since its
root commit `469e8ec`.

Codex's handoff (`results.md`) stopped at two failures. The first was R1: the v1
seeded policy reaches at most 9 actors, because it hires on at most two of every
eight turns and end_of_day clears hands, so zero states met the quota of four.
The second was that the Task 2.1 schema had not merged.

Claude merged integration, which brought the schema and the Task 1.2 grammar,
moved the grammar declaration into `src/kaggriculture/mod.rs`, and adopted
policy v2. Policy v2 appends HIRE entries during hours 0–7, up to `min(M,4)`
entries, and stops at 16 hands. Actual generation yields 6 qualifying states
(4 without the stop). Every other quota passes; zero reordered non-synthetic
sheds relies only on the reviewed dense d30/d31 exception.

`regenerate.py` built and ran the pinned reference crate in 43 seconds. The raw
reference is 33,488,896 bytes, and both files total 781,743 bytes compressed,
within the 8 MiB budget. A second regeneration at a later commit reproduced both
compressed files byte for byte.

Three mutations show the checks discriminate, and each was restored:
- An encoder market-channel swap fails the comparison at offset 889.
- A one-byte reference corruption fails custody.
- Reversed shed ranks fail the real-schema corpus test.

Codex's independent verification of `7b5eacd` found no encoder semantic defect
but rejected on four findings, all fixed test-first (`r1-fixes/results.md`).
The in-process pinned-memory probe could kill pytest (SIGSEGV) on macOS, so the
pinned cases now run only where CUDA is available, as in the starter, and then
assert every buffer is pinned. The contract v4.1 grammar bridge is retired (see
below). Coverage docs point to current receipts, and the missing-oracle error
no longer asserts the obsolete R1 quota diagnosis.

After those fixes `just prepare` passes:
- root Rust: 254 passed, four ignored
- engine: 69 passed
- Python: 1,437 passed, six skipped (two are the CUDA-guarded pinned cases)

Claude's earlier review also fixed the watchdog, which had charged the caller's
pre-existing memory (pytest with torch, over 1 GB) to child commands. Receipts
are in `claude-review.md` and `r1-fixes/`.

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
Any change to the seeded policy, profiles or selection requires full regeneration.
Generation fails loudly if a quota drops. Run the optimized timing command on the pod;
only Task 1.4/6.1 representative complete-work measurements can reopen the pinned
snapshot API. Task 1.4 separately owns huge-aggregate transition admission,
game/seed/terminal rollback and buffer reuse fences. The unavailable pinned weed
JSON-schema range remains the reviewed Task 7.5 parity item; no <=1 cap is invented.

Contract v4.1 retires the temporary engine `grammar_kernel.rs` bridge at the first
production root-to-engine dependency, which Task 1.3 creates. The nine kernel
acceptance/replay-state tests now run in root `grammar_kernel_tests.rs`; the
engine file and its authored trim registration are gone, and a decode mutation
fails the root route. Pinned-memory and GPU paths are unqualified on this Mac.
