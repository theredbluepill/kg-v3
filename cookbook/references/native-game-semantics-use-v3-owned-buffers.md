---
type: "Reference"
title: "Native game semantics use v3-owned buffers"
description: "Task 1.4 adds transactional native lifecycle, checked seed streams, rewards and codec/table bindings over the rebuilt grammar/encoder; CPU checks are scoped separately from pending full trajectory and release qualification and the Task 1.5 adapter."
tags: ["kaggriculture-v3", "adaptation"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.2.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.2/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.2/claude_review_mutations.py"
  - resource: "repository:ops/rebuild-2026-09-29/1.2/checks.json"
  - resource: "repository:ops/rebuild-2026-09-29/merge-1.2/results.md"
  - resource: "repository:python/owl/kaggriculture/gpu_grammar.py"
  - resource: "repository:src/kaggriculture/grammar.rs"
  - resource: "repository:src/kaggriculture/grammar_tests.rs"
  - resource: "repository:src/kaggriculture/grammar_kernel_tests.rs"
  - resource: "repository:engine_rs/TRIM_MANIFEST.json"
  - resource: "repository:scripts/check_engine_trim.py"
  - resource: "repository:tests/tools/test_check_engine_trim.py"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/training.rs"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/native-lifecycle/README.md"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/native-lifecycle/benchmark.json"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"
  - resource: "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_codec.py"
  - resource: "reference-branch:kg/reference-2026-09-29/tests/kaggriculture/test_env.py"
  - resource: "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"
  - resource: "user-directive:2026-09-28:record-every-adaptation"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/V3_IMPORT.json"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/VENDORED_FROM.md"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/Cargo.toml"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/lib.rs"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/ffi.rs"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_features.rs"
  - resource: "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_sampler.rs"
  - resource: "repository:Cargo.toml"
  - resource: "repository:src/lib.rs"
  - resource: "reference-branch:kg/reference-2026-09-29/src/kaggriculture.rs"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/types.py"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/native_bridge.py"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/env.py"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/actor_codec.py"
  - resource: "reference-branch:kg/reference-2026-09-29/python/owl/kaggriculture/gpu_sampling_grammar.py"
  - resource: "repository:Cargo.lock"
  - resource: "repository:python/owl/rs.pyi"
  - resource: "repository:docs/rl-api-specs.md"
  - resource: "repository:src/rules_engine/generation.rs"
  - resource: "repository:docs/rules-engine.md"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.4.md"
  - resource: "repository:ops/rebuild-2026-09-29/briefs/1.5.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/progress.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/final-checks.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/cast-audit.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/d-release-attempt.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/timing.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/reference-recording-attempt.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/g-reference-recording.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/e-rust-restored-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/e-seed-mutation-custody.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/f-combined-final-green.log"
  - resource: "repository:src/kaggriculture/mod.rs"
  - resource: "repository:src/kaggriculture/env.rs"
  - resource: "repository:src/kaggriculture/reward.rs"
  - resource: "repository:src/kaggriculture/admission.rs"
  - resource: "repository:src/kaggriculture/bindings.rs"
  - resource: "repository:src/kaggriculture/env_tests.rs"
  - resource: "repository:src/kaggriculture/lifecycle_timing_tests.rs"
  - resource: "repository:scripts/record_kaggriculture_env_reference.py"
  - resource: "repository:scripts/kaggriculture_env_reference_policy.py"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/reference_recorder.rs"
  - resource: "repository:tests/kaggriculture/test_native_env.py"
  - resource: "repository:tests/kaggriculture/test_native_grammar_bindings.py"
  - resource: "repository:tests/kaggriculture/test_env_reference.py"
  - resource: "repository:tests/tools/test_record_kaggriculture_env_reference.py"
---

# Native game semantics use v3-owned buffers

## Native lifecycle and bindings — current rebuild

Task 1.4 implements the reviewed native boundary over Tasks 1.2 and 1.3. This
existing concept was selected after searching native buffers, rewards, seeds,
statelessness and negative evidence, including the reward-reuse, structured
observation and evaluation/truncation References. The approved brief, unchanged
kernel, actual caller-buffer tests and independent recorded grammar corpus
support the claim. The historical adapter and GPU results below remain scoped
to `kg/reference-2026-09-29`; they do not qualify this implementation.

`owl.rs.KaggricultureEnv` constructs seeded games, publishes cached observations
without another reset, computes rewards in Rust and commits whole batches only
after all preparation and return allocation succeed. Typed borrow guards stay
alive while native work runs detached from Python. All 35 caller destinations
retain their storage, and admission checks exact layout, types and overlapping
byte regions. Truncation has a separate selected-row commit: unselected
observation bytes and all six transition outputs survive exactly. A terminal
step captures the completed game before publishing its newly seeded successor.
There is no native lifecycle state-import API.

The approved constructor takes JSON config, an exact-key reward dict and
required `hire_limit`; Python pinning moves to the Task 1.5 adapter. Its single
buffer set must fence current-stream CUDA readers before reuse when pinned.
Task 1.4 does not implement that adapter or claim the GPU fence is qualified.
Native seed admission is nonnegative i64 with checked successor reservation;
there is no training-only `2**62` cap. Rank partition is `base+rank+k*world_size`.
Any narrower training/evaluation band policy belongs to the factory. The Q1
constructor refinement is recorded here and in the API; the contract document
awaits Claude's agreed wording update.

Reward shaping uses only own starvation/drought/ineffective counters and raw
terminal banks, with the reference's float64 → float32 economic rounding before
terminal float64 addition and final float32 rounding. The explicit binary64
per-component admission predicate prevents inert shaping even through product
underflow; it intentionally strengthens the pinned reference. The cast audit
finds two reachable unbounded HIRE cash casts and checks exact executed hire
costs before commit. Root release policy enables dependency overflow checks;
there are no additional invented game-envelope caps.

Adaptation inventory and reasons:

- `src/kaggriculture/mod.rs`, `bindings.rs`, `python/owl/rs.pyi`: extend the
  existing module/extension with the class and four codec/table functions while
  retaining the header encoder; exact keyword ABI and fallible NumPy admission.
- `src/kaggriculture/env.rs`, `reward.rs`, `admission.rs`: transactional staged
  lifecycle, ordered seeds/errors, terminal capture, reference reward arithmetic
  and the two cast-audit guards. `Cargo.toml` changes only root engine-package
  release overflow policy; vendored kernel bytes are unchanged.
- `src/kaggriculture/env_tests.rs`, `tests/kaggriculture/test_native_env.py`:
  independent reward bounds, injected real-core rollback/retry, terminal timing,
  sentinel-preserving truncate, strict binding admission and sequential world-2
  and world-8 seed partition. `tests/tools/test_check_engine_trim.py` strengthens
  the already-satisfied grammar-bridge retirement regression. Existing
  `grammar_kernel_tests.rs` remains the sole root kernel route.
- `tests/kaggriculture/test_native_grammar_bindings.py`: checks all 964 table
  bits, exact metadata, copied arrays, atomic codec writes and the frozen Task
  1.2 corpus, including dense actors and all negative classes. The Python codec
  and `native_grammar_tables(device)` remain Task 1.5.
- `scripts/record_kaggriculture_env_reference.py`,
  `scripts/kaggriculture_env_reference_policy.py`,
  `ops/rebuild-2026-09-29/1.4/reference_recorder.rs`,
  `tests/tools/test_record_kaggriculture_env_reference.py`,
  `tests/kaggriculture/test_env_reference.py`: deterministic pinned
  TrainingBatch recording, fixed observation-local actions, source/fixture
  custody and strict full-trajectory comparison; incomplete recording may never
  publish a partial fixture.
- `src/kaggriculture/lifecycle_timing_tests.rs` and
  `ops/rebuild-2026-09-29/1.4/`: bounded phase harness, cast audit, planned run
  conditions, actual command receipts, mutation custody and pod handoffs.
  `docs/rl-api-specs.md`, `docs/rules-parity-coverage.md`, this Reference, its
  index and the cookbook log record the exact ABI, evidence and remaining gaps.

Actual scoped checks: the trim checker and nine root kernel tests confirm the
bridge was already retired. The native lifecycle suite passes 12 tests and the
cast helper suite passes six; Python native environment and grammar suites
pass 383 (340 + 43), including two-thread versus one-thread native-pool
equivalence. The grammar corpus contributes 321 accepted round trips
(320 scheduled, 64 dense, 22 full-length) and 43 rejection classes. World sizes
2 and 8 each consume 67 seeds per rank, running rank batches sequentially with
at most two live games. A controlled latch proves GIL release during real native
observation work. These are CPU semantics checks, not a distributed or training
run.

Restored mutations discriminate premature live seed writes in step/reset,
whole-batch publication during truncate, and the old finite rank-offset seed
formula. The two seed-partition cases both report `rank 1 collides with rank 0`
under that mutation. Review also replaced an actual-error-derived reward budget
that wrongly admitted zero rewards; the corrected independent endpoint and ULP
budget reject the mutation. Malformed initial tests and diagnostic-pattern
repairs are retained in operational receipts, not counted as behavior reds.

The release overflow attempt exceeded the Mac RSS budget before its test body
(28.873 seconds, 1,007,714,304 sampled bytes); release proof is PENDING (pod).
The optimized timing build also stopped before its test body (7.840 seconds,
1,018,937,344 sampled bytes). `ops/rebuild-2026-09-29/1.4/timing.json` preserves
the paired commands with and without dependency overflow checks; phase costs
and policy cost remain PENDING (pod). The sole full-oracle recording attempt
also stopped during exported-reference debug engine compilation: the inner
watchdog killed its worker at 11.297 seconds and 1,012,252,672 sampled bytes;
the outer command exited 1 at 11.443 seconds. Zero games/transitions were
recorded and no fixture/manifest was published. Reference-harness compilation,
the full 16-game TrainingBatch comparison, its reward/done mutation, trajectory
coverage and fixture-size/hash qualification remain PENDING (pod). The missing
fixture makes replay fail loudly. Final root Rust tests pass 274 with five
ignored and no failures; engine tests pass 69 with none ignored or failed.
`just rs-prepare`, trim verification and debug extension installation exit zero.
The requested five-file Python command passes 497 and fails only the full
16-game replay and frozen-fixture custody tests because the NPZ is absent (zero
skips). The native environment/grammar subset contributes 383 passes; the
recorder tooling contributes 35 synthetic custody passes and one of the
missing-fixture failures.

`just py-prepare` passes formatting, Ruff, syntax and mypy over 65 files. Its
pytest run collects 2,045 tests, then the watchdog kills it during the first
existing typing test at 15.693 seconds and 1,015,529,472 sampled bytes; no broad
suite completion counts are claimed. The repaired full `just prepare` passes
formatting, lint, documentation checks, mypy, build, trim and the same Rust
suites before its Python stage stops at that existing test (56.574 seconds,
1,017,036,800 sampled bytes). Broad Python completion for `just py-prepare`
and `just prepare` remains PENDING (pod). Exact commands and limits are
retained in `results.md` and
`final-checks.json`. No debug timing substitutes for optimized costs. Future
consumers may rely on the checked native ABI; completing Task 1.5, the oracle
and pod checks is required before widening claims to the Python adapter,
trainer integration, pinned DMA safety or complete-update throughput.

## Task 1.2 — grammar checkpoint

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
`grammar_tables_from_arrays`. That checkpoint had no PyO3 binding; Task 1.4
above now provides it. The Python device-table bridge and model wiring remain
Task 1.5. Receipts: `ops/rebuild-2026-09-29/merge-1.2/`. Those CPU checks do not qualify the
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
