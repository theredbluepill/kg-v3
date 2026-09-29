---
type: "Reference"
title: "Native game semantics use v3-owned buffers"
description: "Task 1.2 rebuilds the observation-local grammar and strict native codec; historical adapter and GPU evidence remains scoped to the reference branch."
tags: ["kaggriculture-v3", "adaptation"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/briefs/1.2.md"}, {"resource": "repository:ops/rebuild-2026-09-29/1.2/results.md"}, {"resource": "repository:ops/rebuild-2026-09-29/1.2/claude_review_mutations.py"}, {"resource": "repository:ops/rebuild-2026-09-29/1.2/checks.json"}, {"resource": "repository:ops/rebuild-2026-09-29/merge-1.2/results.md"}, {"resource": "repository:python/owl/kaggriculture/gpu_grammar.py"}, {"resource": "repository:src/kaggriculture/grammar.rs"}, {"resource": "repository:src/kaggriculture/grammar_tests.rs"}, {"resource": "repository:src/kaggriculture/grammar_kernel_tests.rs"}, {"resource": "repository:engine_rs/TRIM_MANIFEST.json"}, {"resource": "repository:scripts/check_engine_trim.py"}, {"resource": "repository:tests/tools/test_check_engine_trim.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/training.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/native-lifecycle/README.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/native-lifecycle/benchmark.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_codec.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_env.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:record-every-adaptation"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/V3_IMPORT.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/VENDORED_FROM.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/Cargo.toml"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/lib.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/ffi.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_features.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_sampler.rs"}, {"resource": "repository:Cargo.toml"}, {"resource": "repository:src/lib.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/src/kaggriculture.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/types.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/native_bridge.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/env.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/actor_codec.py"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/gpu_sampling_grammar.py"}, {"resource": "repository:Cargo.lock"}, {"resource": "repository:python/owl/rs.pyi"}, {"resource": "repository:docs/rl-api-specs.md"}, {"resource": "repository:src/rules_engine/generation.rs"}, {"resource": "repository:docs/rules-engine.md"}, {"resource": "repository:docs/rules-parity-coverage.md"}]
---

# Native game semantics use v3-owned buffers

## Task 1.2 — current rebuild

The existing-concept search covered grammar, HIRE, action codec, strict padding,
reference disagreement and L4. This note owns those native semantics; no
parallel concept is added. The reviewed brief, pinned reference sampler/codec,
independently repeated native recording and direct kernel execution delimit the
claim. Hash equality establishes custody, not semantic truth.

`src/kaggriculture/grammar.rs` is the root's compiled, observation-local grammar:
typed plans and cursors, eight factored tables totaling 964 booleans, checked i64
transport, strict canonical JSON encode/decode and submitted-HIRE capacity.
It preserves every actor ordinal, omitted unit quantity, market zero and EMPTY
position. Padding must be zero and is then ignored; native length0 rejects.
The encoder validates through the same cursor and leaves output untouched on
error. No retained engine bytes changed and no second grammar was imported.

Nine shared tests pass in both root and engine packages. They compare independent
reference tables, every reachable support class, the 140-unit/98-market matrix,
321 accepted fixture/control round trips and malformed transport, and enumerate
the coupled-HIRE law through actual Rust masks. Initial error-stub failures and
restored negative controls establish the tests' discriminators. Review caught a
mistyped PLANT token in a new example and a missing final-sentinel guard case;
the receipt records their diagnosis and corrected controls without weakening
expected behavior.

The fixture contains 320 scheduled accepted programs (256 synthetic, 64 real),
64 dense programs including 22 at length252, plus 43 rejections and one accepted
zero-padding control. The two native reference decoders and Python codec agree
on all scheduled actions. Eight classified disagreements preserve the historical
FFI capacity bypass, prefix-only padding admission and inapplicable incomplete
prefix. All 5,752 trace candidates were scanned; no replay codec rejection
category occurred. A second native recording exactly reproduces the fixture
and manifest; expected actions never come from the new grammar.

Nine authored kernel tests plus the shared nine pass. Both seats execute all
256 synthetic programs and the 140/98 matrices; meaningful bank/inventory
checks avoid acceptance-only evidence. Explicit `Game::from_header` states
execute 241 actors with ten orders and a successful 240-to-241 HIRE; subsequent
HIRE support is absent and overcapacity encode rejects. Distinct farmer/hand
moves confirm actor order. Sixty-four selected real actions preserve exact
public/private snapshots, recursive object order, outcomes and counters in
actual seeded replay states. The retained full 2,876-transition parity suite
has a separate denominator. Full `just prepare` passes with 164 root Rust tests
(two existing ignores), 77 engine tests (none ignored), 1,045 Python tests
(three platform skips), both Clippy/formatter graphs, mypy, doc lint/freshness
and the trim checker. The receipt preserves the actual command logs.

Adaptation inventory (one coherent native grammar change): `Cargo.toml`
(promote existing Serde JSON through Cargo), `src/lib.rs`,
`src/kaggriculture.rs`, `src/kaggriculture/grammar.rs`,
`src/kaggriculture/grammar_tests.rs`, `engine_rs/tests/grammar_kernel.rs` (retired
to root `grammar_kernel_tests.rs` in Task 1.3),
`engine_rs/TRIM_MANIFEST.json`, `scripts/check_engine_trim.py`,
`tests/tools/test_check_engine_trim.py`,
`tests/tools/test_record_grammar_reference.py`, the fixture and manifest in
`tests/fixtures/kaggriculture/`, `docs/rl-api-specs.md`,
`docs/rules-parity-coverage.md`, `docs/kaggriculture-contract.md`,
`ops/rebuild-2026-09-29/plan.md`, this Reference/index/log, and the recorder,
harness, source/control custody, declared checks, logs and result receipt in
`ops/rebuild-2026-09-29/1.2/`. On the Task 1.2 branch the fixed authored engine
allowlist contained exactly the replay and grammar integration tests; after the
merge with Task 1.1b it is exactly those two tests plus the generated-trace
`MANIFEST.json`. Its final hashes derive from formatted bytes. Contract v4.1 records all five Claude/Codex-agreed clarifications
as clarifications, without semantic changes.

Claude's implementation review (2026-09-29, at `75169be`) read every changed
path and found no production defect. Twenty one-per-oracle source mutations,
each restored byte-exactly, made every named shared grammar, kernel, trim-checker
and recorder test fail at an assertion. A fresh `record_reference.py verify`
reproduced the fixture hashes and counts. The brief's nonexistent `reference`
Python extra was corrected. Encoder rejection categories remain checked only on
synthetic inputs, because the traces contain no replay codec rejection.
Evidence: `ops/rebuild-2026-09-29/1.2/results.md` (Claude review) and
`ops/rebuild-2026-09-29/1.2/logs/claude-review/`.

Future consequence: this codec/table source serves Tasks 1.3–1.5, 2.3 and BC/export.
Task 1.3 created the first production root → engine edge, repaired L4 and retired
the temporary engine include: the kernel acceptance/replay tests now run in root
`src/kaggriculture/grammar_kernel_tests.rs`, and the engine file and its authored
registration are removed. The engine counts below are historical. Task 2.3's reviewed
v3 table shapes and post-STOP zero rule match this interface.

Merge into the integration branch (after Task 1.1b live parity and Task 2.3
heads): engine 87/87 (19 replay-parity, 18 grammar/kernel), root Rust 164 with
two ignored, Python 1,337 passed with four skips, and the trim checker pass. A
merge-time scratch cross-check found the native `grammar_tables()` equal to the
Python heads' `expected_grammar_tables` on all 964 bits and accepted by
`grammar_tables_from_arrays`. No PyO3 binding exists yet, so
`native_grammar_tables` still raises and the heads use the Python stand-in; the
binding stays with Task 1.4. Receipts: `ops/rebuild-2026-09-29/merge-1.2/`. CPU checks do not qualify the
L6 GEMM fix, GPU sampling/replay, native batch transactions, buffer lifetime,
model behavior or learning. No training, GPU, network or performance run occurred.

## Historical reference-branch scope

Everything below describes `kg/reference-2026-09-29` at `65f0eac5`, including
files that share paths with the rebuilt tree. Its adapter, PPO and performance
results do not qualify Task 1.2 or the clean-base integration.

## Historical entire-pipeline correction

The owner rejects PyO3 interface alignment alone as sufficient Isaiah reuse. The old decoder's bounded two-GPU baseline is retained in `ops/gpu-sps-2026-09-29/results.md`; native game semantics must support an efficient complete data/model/PPO path. `BatchedSlotMasks` now derives compact local mask tables from native plans for batched heads. The implementer reports exhaustive local-condition checks plus coupled-Gumbel queue enumeration for three positions/eight kinds/HIRE budgets 0,1,2,3,10, with canonical probabilities agreeing within 1e-12, and 22 environment/codec cases passing. The proof uses final prefix HIRE count=min(raw prefix count,budget), preserving first final NONE, EMPTY order consumption and marginalized post-STOP choices. This scopes the grammar claim; it is not replacement-model or GPU qualification.

The fused native lifecycle is locally checked and executes in the measured two-GPU PPO path. `engine_rs/src/training.rs` and `owl.rs.KaggricultureTrainingEnv` own the active training game batch, BigInt seed streams, typed terminal state, reward/economic/mask computation and synchronous auto-reset. It reuses the engine/decoder/encoder while dropping unused legacy native controllers from active training. The Python adapter makes one native call per step; diagnostic terminal JSON is generated lazily only when requested. Caller-owned int64 NumPy inputs remain views when contiguous, with checked Rust narrowing into scratch; strided inputs are normalized at the Python boundary. This is not a zero-copy claim for narrowing, transactional game clones or double output staging, which remain for rollback correctness.

Fallible mutable NumPy borrows and explicit C-layout validation reject alias/Fortran misuse before mutation; staged step/reset catch panics before commit. Final local checks report 32 environment/codec and 14 PPO/incumbent cases, 155 root Rust cases with two existing ignored, 120 vendored cases, pinned build/fmt/Clippy/docs, seven-file mypy and Ruff. The paired fixed-action CPU adapter benchmark retains 108 equality checkpoints and exact source/binary identities in `ops/gpu-sps-2026-09-29/native-lifecycle/`; at eight games/two threads PASS measures 22,727→32,295 game steps/s and movement-plus-market 18,738→25,184. The benchmark excludes model/GPU/PPO and high actors. Combined Linux/GPU evidence now covers the fixed-minibatch8/32/128/2,048/4,096-environment runs in the GPU receipt; earlier adapter checks below remain historical support for the preserved low-level oracle, not substitution for new-path qualification.

## Initial adapter contract and retained oracle

Owner: “note that it must follow the current v3 data pipeline (RUST I/O, Training PPO, etc.)”. The adapter therefore reuses Kaggriculture game semantics and native grammar while preserving the starter extension and caller-owned NumPy/Torch storage. `engine_rs/V3_IMPORT.json` binds imported v2 working bytes with per-file hashes and explicitly records its dirty source checkout; its parent commit alone cannot reproduce that import. Historical v2 tests are not transferred qualification.

The initial `src/kaggriculture.rs` adapter exposes `owl.rs.KaggricultureBatch` through the existing PyO3/maturin extension; that retained low-level API now also serves as an independent lifecycle oracle for the fused path. The native kernel performs observe/reset/step work, retaining its Rayon pool. Python supplies typed buffers; Rust calls the engine's C-shaped functions internally. The active step boundary uses no Python ctypes and no per-step JSON. Setup headers and diagnostic snapshots remain distinct from the tensor hot path.

`python/owl/kaggriculture/types.py` declares two seat perspectives, 241 actor capacity, 252 command-frame capacity, twelve categorical slots, and versioned feature width. Version 2 declares 8,176 floats versus 8,165 in version 1. Each observation row is one player's legal perspective; attention must not mix private views. Actor presence and command positions are separate masks; prefix-dependent vocabulary legality comes from native grammar. The Python adapter passes contiguous `int64` action tensors through checked conversion to native `int16[env,2,252,12]` frames and `int32` lengths; observation context is `[env,4]`. Native work releases the GIL. Terminal `win_loss`/`win_only` reward conversion compares raw final banks before reset; the adapter retains terminal snapshots/metrics and synchronously resets finished native games. These claims require the adapter's own tests rather than inherited v2 qualification.

## Interpretation and consequence

Keep the game's rules authoritative in the vendored kernel and the v3 training data boundary authoritative in the adapter. Do not reintroduce the v2 collector/trainer as a parallel path. Preserve STOP/no-op, queue order, quantity and HIRE distinctions rather than collapsing them to fit an Orbit Wars action shape. Copy buffers intentionally where storage must survive a later native write. Fail clearly on mixed game specs and unsupported versions. Distributed native seeds use initial seed+rank and stride=world size throughout creation and every reset. A finite rank*n_envs offset was rejected by independent review because streams eventually collide.

## Bounded CPU hot-path refinement

The initial Python adapter reused checked int16/int32 action scratch arrays, mask-index tensors and the NumPy economic-counter view. When shaping is disabled it skips unnecessary float64 penalty arithmetic, retaining native cumulative counters and metrics. The regression suite includes narrowing-overflow rollback and replacement of subsequent actions.

A paired local benchmark uses Apple M5, eight environments, Torch 2.9 with one intra-op thread and one native thread, five pairs of 800 steps including auto-reset. PASS-only median adapter time is .411 ms optimized versus .436 ms baseline (19,050 versus 17,957 game transitions/s); EAST + BUY_SEED(1) + SELL(1) is .536 versus .562 ms (14,535 versus 13,875). Ninety paired checkpoints agree on features, rewards and dones. One transition means one two-seat game step; actions are pregenerated, excluding the model, GPU, PPO and optimizer. One farmer per seat was used; HIRE/high-actor behavior is not performance-qualified. Independent before/after runs drifted with unchanged native speed, so only this paired scope supports the local reduction claim. The manifest identifies exact evidence/script hashes; the final receipt retains artifacts.

## Retained starter rule fixtures

Adding the vendored crate unifies `serde_json/arbitrary_precision`, exposing a test-only float decode incompatibility in starter `src/rules_engine/generation.rs`. The fixture enum now reads `serde_json::Number` and checks conversion to finite f64; production game logic and oracle comparisons are unchanged. Missing official Orbit fixtures were regenerated with `kaggle-environments==1.32.7` and replays downloaded using `kaggle==2.1.0`; `docs/rules-engine.md` and `docs/rules-parity-coverage.md` record the retained coverage. This preserves starter regression discipline rather than claiming Kaggriculture parity. The retained final `rs-prepare` log reports 155 passed, zero failed, two ignored, with formatting, all-target Clippy and documentation freshness checks; intermediate missing-fixture, decode and stale-doc failures are retained in the final receipt.

## Checks and remaining gaps

The contracts above have been inspected in the source during implementation. The native implementer reports Cargo check/build passing. The vendored engine unit suite passes120/120 (`cargo +stable test --manifest-path engine_rs/Cargo.toml --lib --locked`), with its final log independently read and hashed in the setup receipt. Four external test fixtures were made self-contained after preserved missing-file failures. The proper maturin abi3 wheel on CPython3.12.13/Torch2.9 passes **20/20 native environment/codec/reward tests**, including complete719-step termination/autoreset, seat-private isolation, caller-buffer reuse, serial/Rayon parity, rollback, grammar replay, economic counters/reset and disjoint rank seed streams. Current manifest verification pins exact source hashes, commands, toolchain and retained failures. Python typing for seven game files and root Clippy pass; Linux execution was outside that initial receipt; the later fused-path GPU evidence above closes that platform-execution gap. The pinned nightly installation initially lacked its rustc-driver library; Rustup repaired the partial installation to `1.97.0-nightly (e9e32aca5 2026-04-17)`. The original pinned-toolchain `rs-prepare` then passed all 155 cases with two existing ignored, formatting, Clippy and documentation freshness; stable 1.94.1 verification is also retained. The repair and final pinned-toolchain logs were independently read and retained, so the initial local toolchain gap is closed. An unsupported direct dylib import test harness exited 139 before the proper maturin wheel environment was prepared; no such copied module is retained. The final shared receipt adds 772 passing Python cases and actual single/two-process CPU PPO updates. Source-scoped status does not extend these checks to future-engine parity or playing strength. The CPU adapter timing and later complete GPU measurements have separate explicit workload limits.

The retained batch oracle also exposes keyword-only `explicit_state=True` for offline recorded-observation encoding through the existing native explicit-state constructor. An evolved-state regression matches features, context and banks exactly;814 Python/155 Rust checks pass. Default and live PPO seed/reset semantics are unchanged. Data rejection, BC validation and handoff limits are recorded in [[bc-bootstrap-uses-native-replay-features-and-current-heads|the BC Reference]].
