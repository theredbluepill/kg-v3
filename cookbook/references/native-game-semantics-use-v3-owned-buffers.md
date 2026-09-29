---
type: "Reference"
title: "Native game semantics use v3-owned buffers"
description: "Task 1.4 adds the transactional native lifecycle, checked seed streams, rewards and codec/table bindings (16-game TrainingBatch oracle bit-exact, release overflow proof on the pod); Task 1.5 Stage 1 adds the typed Python adapter, one-buffer entry fence, config/factory and codec seam with CPU contracts, while Stage 2 native wiring, Task 3.1 integration and the pod DMA fence remain pending."
tags: ["kaggriculture-v3", "adaptation"]
status: "verified-scoped"
generated: {"by": "openai/codex; revised by anthropic/claude-opus-5-5", "at": "2026-09-29"}
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
  - resource: "repository:tests/fixtures/kaggriculture_env_reference_v1.npz"
  - resource: "repository:tests/fixtures/kaggriculture_env_reference_v1.json"
  - resource: "repository:docs/kaggriculture-contract.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/run-statement-pod-oracle.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/record.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/replay-green.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/mutation-done.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/release-overflow.log"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/timing-enabled.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/pod-oracle/timing-disabled.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/claude-review/just-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-env-independent/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-env-independent/oracle/semantic-guard-probes.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/verify-r1-fixes/guard-removal-mutations.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/verify-r1-fixes/guard_removal_mutations.py"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-env-r2/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/codex/verify-env-r3/review.md"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/p3-rerecord/reference-recording-attempt.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/verify-r2-fixes/size-hash-guard-mutations.json"
  - resource: "repository:ops/rebuild-2026-09-29/1.4/verify-r2-fixes/size_hash_guard_mutations.py"
  - resource: "repository:python/owl/kaggriculture/env.py"
  - resource: "repository:python/owl/kaggriculture/codec.py"
  - resource: "repository:python/owl/kaggriculture/types.py"
  - resource: "repository:python/owl/kaggriculture/config.py"
  - resource: "repository:python/owl/game.py"
  - resource: "repository:tests/kaggriculture/fake_env.py"
  - resource: "repository:tests/kaggriculture/test_env.py"
  - resource: "repository:tests/kaggriculture/test_env_cuda_fence.py"
  - resource: "repository:tests/kaggriculture/test_codec.py"
  - resource: "repository:tests/kaggriculture/test_game.py"
  - resource: "repository:ops/rebuild-2026-09-29/stage1-adapter/results.md"
---

# Native game semantics use v3-owned buffers

## Task 1.5 Stage 1 — Python boundary before the Task 1.4 merge

On base `e197528` the real extension exposes only the Task 1.3 header encoder;
Task 1.2 grammar is Rust-internal. The owner explicitly limits this episode to
Python work that can be checked before Task 1.4 merges. Existing-concept search
covered caller-owned buffers, adapter, codec, seed streams, fences, rollback,
reward and negative GPU evidence; this Reference and the reward Reference own
the adaptation. No new concept, Lesson or result board is warranted.

The adapter allocates one set of 29 structured observation and six transition
tensors, with one NumPy view per output. Exact native keyword calls fill those
buffers; construction observes without another reset. Every reset/step/truncate
fences the current CUDA stream first when storage is pinned. CPU/unpinned
methods make no CUDA call. Requested unavailable pinning fails before the Mac's
unsafe allocator path. Strict CPU int64 actions and bool masks are borrowed
through one zero-copy input view per call; no Python rewards or grammar enter
the live path. Diagnostics delegate lazily and full snapshots stay outside
policy inputs. The native contract preserves outputs on error and selected-row
truncate behavior; fake checks establish only the adapter's lack of extra writes.

The factory retains Isaiah's exact Orbit constructor and Task 3.4's single
observation-tag config union. Kaggriculture receives `base_seed+rank` and
`world_size` stride, the existing `reward_shaping` field, top-level reward mode,
and action-owned hire limit. Seed/count validation is strict; the game envelope
explicitly canonicalizes integral JSON floats, as the real header encoder
independently confirms. The cold codec delegates grammar to pending native
functions and publishes batch output only after every seat succeeds. The Task
1.4 stub is copied verbatim; a stub is not a runtime binding. The Task 1.5 merge of `kg/rebuild-env` (`b6b722f`) kept
Task 1.4's ruff-formatted stub: its declarations are AST-identical to the
Stage 1 copy, which differed only in line wrapping, function order and the
`fmt: off`/E501 exemptions that the formatted text no longer needs. The file now
has exactly one `KaggricultureEnv` class block and one declaration of each of
the four grammar/codec functions.

Changed-path inventory: `python/owl/kaggriculture/env.py`, `codec.py`, `types.py`,
`config.py`, `python/owl/game.py`, `python/owl/rs.pyi`;
`tests/kaggriculture/fake_env.py`, `test_env.py`, `test_env_cuda_fence.py`,
`test_codec.py`, `test_game.py`; `docs/rl-api-specs.md`, the appended Stage 1
reconciliation in `ops/rebuild-2026-09-29/briefs/1.5.md`, this Reference/index/log
and receipts in `ops/rebuild-2026-09-29/stage1-adapter/`. Reward/config migration
paths and their checks are inventoried in [[reward-reuse-preserves-objective-and-critic-semantics|the reward Reference]].

Independent evidence: the exact-signature fake rejects incorrect keyword calls;
all 35 shapes/dtypes, disjoint storage and stable views are asserted, input
pointers share caller tensors, and fence ordering/conditions are exercised with
CPU spies. The real Task 1.3 encoder proves seat 1 private inventory cannot
change seat 0's allocated row. Factory tests establish 64 implied seeds/rank are
disjoint and demonstrate the rejected finite-offset collision. Full Python
preparation passed with 1,705 tests and 22 skips, including unchanged Isaiah
suites; focused adapter tests passed 29 with two skips, game tests 26 with one,
and codec tests six with one. Import-red and subsequent green logs retain actual
failures, including corrected fixture/test-helper mistakes. The receipt names
commands, exits and all inherited versus new skips.

Future consequence: Task 3.1 can consume the typed boundary after Task 1.4 and
Stage 2 qualification. Stage 2 must remove the explicit binding skips and test
real admission, reward fixture/schema, codec replay, seed consumption, live
seat actions and rollback. Native table loading and the model's stand-in remain
unchanged here. The pod DMA test is written with all 35 destination tensors
allocated before the delayed copies and includes fence-removal mutation; it has
not run. Native transactions, CUDA/pinned behavior, trainer integration and
complete-work throughput remain unqualified. No Rust/engine bytes, runner,
trainer, model, dependencies or Isaiah tests changed, and no build, training,
GPU or performance run occurred.

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
constructor refinement is recorded here, in the API and as contract v4.2 (a
refinement both agents agreed in the 1.4 brief review, not a semantic change).

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
  `tests/fixtures/kaggriculture_env_reference_v1.{npz,json}` is the recorded
  16-game fixture (pod recording, Claude review).
- `src/kaggriculture/lifecycle_timing_tests.rs` and
  `ops/rebuild-2026-09-29/1.4/`: bounded phase harness, cast audit, planned run
  conditions, actual command receipts, mutation custody and pod handoffs.
  `docs/rl-api-specs.md`, `docs/rules-parity-coverage.md`, this Reference, its
  index and the cookbook log record the exact ABI, evidence and remaining gaps.
  `docs/kaggriculture-contract.md` records the agreed Q1 constructor refinement
  as v4.2. `ops/rebuild-2026-09-29/1.4/claude-review/` holds the review, the pod
  run statement and receipts, and the mutation logs.

Actual scoped checks: the trim checker and nine root kernel tests confirm the
bridge was already retired. The native lifecycle suite passes 12 tests and the
cast helper suite passes six; Python native environment and grammar suites
pass 387 (344 + 43) after Claude's review, including two-thread versus one-thread native-pool
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

Codex's Mac attempts at the release overflow proof, the optimized timing build
and the reference recording all stopped at the 960 MiB watchdog before their
test bodies ran. `results.md`, `timing.json` and
`reference-recording-attempt.json` keep those receipts. Claude then ran these
checks on the pod (`w7ia3zvxqsvs3g`, CPU only, separate clone at `9dc2d02`,
run statement `claude-review/run-statement-pod-oracle.md`):

- **Trajectory oracle.** The recorder compiled the exported reference and
  recorded all 16 games (seeds 17000–17015, 719 transitions each) in 32 s.
  Every coverage counter was positive in every game. The fixture is 310,365
  bytes compressed (npz sha256 `494bbf2c…c976`, manifest `aa6cc641…28b7` at
  that recording; the r3 fix below re-recorded the manifest).
  The native replay matches the reference TrainingBatch bit for bit on rewards,
  dones, banks, counters, seeds and terminal records over all 11,504
  transitions. The replay and custody suites pass 37 tests, on the pod and on
  the Mac (14 s, 411 MB). One deviation was recorded: the reference lockfile's
  crates were missing from the pod cache, so an explicit `cargo fetch --locked`
  (lockfile-checksum verified) preceded the offline build.
- **Oracle mutation.** Inverting native `dones` fails with
  `first divergence game=0 seed=17000 step=0 seat=0 … field=dones`. After a
  byte-exact restore, the replay passes again.
- **Release overflow proof.** `release_dependency_overflow_is_caught` passes in
  release with the root engine overflow-check override. With the override
  disabled through `--config`, the same test fails, so the override is what
  makes the release step fail safely.
- **Release phase timing** (fat LTO, one live env, one thread, component only).
  Dense 241-actor states take a 747.6 µs median for the composed clone, step,
  prepare and write. The outer candidate clone is 159.5 µs of that, the kernel
  step including its inner clone 270.3 µs and snapshot acquisition 171.7 µs.
  Early states take 54.8 µs. Engine overflow checks cost no measurable time
  (752.0 µs dense with them off). These numbers make no complete-update or
  throughput claim.

Claude's review added one oracle,
`test_calls_overwrite_every_output_byte_of_the_same_buffer_set`. It is the
native half of L6: observe, reset, step and terminal step rewrite all 35
outputs of the one caller-owned buffer set in place, including padding. Two
runs that start from different poison bytes produce identical outputs, and the
buffer addresses do not change. Dropping the `transition_econ_after` copy makes
it fail. Further restored native mutations fail their named oracles:

- advancing seeds by 1 instead of `seed_stride` fails world sizes 2 and 8;
- clearing selected terminal records before reset staging succeeds fails the
  reset/truncate rollback test.

Full `just prepare` on the Mac, without Codex's 960 MiB watchdog, exits 0:
Python 2,042 passed with 7 skips (1.95 GB peak RSS). The skips include
`test_native_tables_match_expected_tables`, which waits for Task 1.5's
`native_grammar_tables(device)`.

Codex's independent verification of `1e26760`
(`ops/rebuild-2026-09-29/codex/verify-env-independent/review.md`) found no
production defect. It showed that the recorder's nine inventory tests changed
arrays without refreshing `manifest["arrays"]`, so every case stopped at the
hash custody check: removing seven semantic guards (length range, offsets,
transition indices, seed consumption, terminal steps, done schedule, terminal
values) still passed. The hash-only test is now kept separately, and 17
coherent probes refresh the array metadata before asserting the exact semantic
error. They also cover the packed token count, terminal winner, bank/economic
continuity, nonfinite rewards and decreasing counters. Replacing each of the 14
`require` guards with a no-op fails at least one named test
(`1.4/verify-r1-fixes/guard-removal-mutations.json`); the recorder bytes were
restored (sha256 `156bee30…3499`). `docs/rl-api-specs.md` now states that
contract v4.2 incorporates the approved Q1 constructor refinement.

Codex's second verification of `ba9b59b`
(`ops/rebuild-2026-09-29/codex/verify-env-r2/review.md`) confirmed both fixes
and again found no production defect. It found that the size and archive-hash
tests still stopped early: they made the declared size disagree with the file,
so the size check fired first. The compressed-size cap and both archive hash
checks could be removed and all tests still passed. The new tests keep every
other field consistent and assert the exact error. A valid fixture fails
against a cap lowered one byte below its size, for both the compressed and
expanded budgets. A declared size one byte too large fails the size check
while staying within budget. A same-length flipped byte and a wrong
`fixture_sha256` both fail the archive hash. A wrong `expanded_sha256` fails
the expanded hash. Every one of these checks runs before `np.load`. Replacing
each of the six size/hash `require` guards with a no-op fails at least one of
these tests (`1.4/verify-r2-fixes/size-hash-guard-mutations.json`). The
recorder bytes were restored (sha256 `156bee30…3499`).

Codex's third verification of `1e63597`
(`ops/rebuild-2026-09-29/codex/verify-env-r3/review.md`) confirmed the r2 fix
and again found no production defect. It found one P3 in the recorder's final
publication: after replacing the NPZ, a failure while replacing the JSON
manifest left half a pair. A fresh output kept an NPZ with no manifest. An
existing fixture lost its NPZ but kept its old manifest. The loader rejected
both, so no false oracle could pass. The recorder now publishes the pair
through `publish_pair`. It saves each target's prior bytes before replacing,
and if a replacement raises, it restores every replaced target in reverse
order. A target that did not exist before is removed. Two tests fail only the
manifest replacement, after the NPZ replacement has happened. The fresh-output
test leaves an empty directory. The existing-output test restores the exact
prior bytes, and the pair still loads. Both tests fail against the previous
recorder. This covers raised exceptions only. If the process is killed between
the two replacements, a half pair can still remain, and the loader still
rejects it.

The recorder's own bytes are part of the fixture's source custody, so the fix
required a new recording. It ran on the Mac under the recorder's 115 s /
960 MiB watchdog with `CARGO_BUILD_JOBS=1`: 28.5 s, sampled peak 980 MB
(`1.4/p3-rerecord/`). An earlier attempt ran before `ruff format`
rewrapped one recorder line, which changed its hash, so it was redone. The NPZ is byte-identical to the pod recording
(`494bbf2c…c976`). The only manifest change is `recorder_sha256`
(`156bee30…3499` to `50766851…5d3`), giving manifest sha256
`36dffed2…7732`. The recorder and replay suites pass 61 tests.

Future consumers may rely on the checked native ABI and the recorded reference
equivalence. Task 1.5 (the adapter, its pinned CUDA entry fence and the pod DMA
test) is required before widening claims to trainer integration, pinned DMA
safety or complete-update throughput.

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
