# Native SPS implementation review

Reviewer: independent Codex subagent, 2026-10-01. Production code reviewed
read-only against base `b2276bc5b70073b58a27f9e5fbd52473dbd48569`.
The reviewer added four tests in `src/kaggriculture/env_tests.rs`; production
implementation was authored by the other agents.

## Scope and result

No production correctness defect found in the reviewed changes to
`src/kaggriculture/env.rs`, `buffers.rs`, `observe.rs`, or
`engine_rs/src/lib.rs`. The unchanged binding's two detached phases were also
read, including NumPy preflight and Python return allocation before commit.

- Raw transport errors still precede seed reservation errors, which precede
  grammar errors, which precede engine execution. Nested results preserve this
  ordering despite decoding other environments concurrently. Indexed collection
  selects errors and terminal metrics in environment order.
- The engine candidate-return method preserves early checks and the original
  transactional engine clone. Reading pre-step counters from the committed
  game does not mutate it. Hosted controller cloning still precedes acting.
- Successful and failed staging reuse is safe because `write_seat` clears all
  29 fields before writing, including unused actor slots. A dropped pending
  batch loses only scratch storage; the next prepare allocates replacement
  scratch without advancing games or seeds.
- Commit validates lengths before publication. The parallel iterator gives
  each worker disjoint caller-owned rows and one slot. Copies have fixed admitted
  dimensions; replaced slots have no fallible destructor. Truncation preserves
  unselected rows and transition values. Existing Python borrow guards survive
  both detached phases, with no new Python API use inside workers.

## Added discriminating tests

All four execute at native thread counts 1, 4 and 8:

1. `parallel_admission_preserves_error_precedence`: later raw errors beat
   earlier grammar failures and seed overflow; overflow beats grammar errors;
   multiple grammar/raw failures select the first environment and seat. An
   injected engine failure is never reached while grammar admission fails.
2. `discarded_pending_batch_preserves_state_and_can_retry`: discard two
   successful prepares for both terminal and nonterminal batches, then execute
   a different action and compare every output byte, game snapshot, seed and
   terminal record with the ordinary one-thread control.
3. `rejected_publication_preserves_state_and_can_retry`: incompatible
   observation output dimensions reject before changing either destination or
   live state; a subsequent retry matches the one-thread control.
4. `failed_parallel_step_clears_reused_actor_padding_on_retry`: first hire an
   extra actor into both seats, then fail environment 1 at StepPrepare,
   StepPanic or AutoReset while its peer prepares reset rows. Failure preserves
   all published state. Retrying after clearing the fault resets actor counts
   and matches fresh zeroed observation/transition storage byte for byte.

`fixture` now delegates to `fixture_with_threads(..., 1)`, preserving all
pre-existing test calls.

## Actual checks

- `rustfmt src/kaggriculture/env_tests.rs`: exit 0.
- `cargo test --lib kaggriculture::env_tests`: exit 0, **33 passed**, zero
  failures/ignores, 274 filtered out, 1.27 seconds in the test bodies.
  Full output: `review-native-tests.log` in this directory.

No benchmark, remote access, full prepare, mutation run, or GPU test was run
by this reviewer. The main task owns the old/new trajectory oracle, benchmark,
full repository checks and cookbook record. The review does not establish
throughput, pinned CUDA DMA, or exhaustive parity for arbitrary action streams.

Reviewed source SHA-256 values (before any later formatting):

```text
e7e4284de04b4465cb03e9fbba972a30eb5596980455ba13ebdcc5173291a87f  src/kaggriculture/env.rs
53472fe3f49bcad88643829cd8472ce7094b45197d8943aaebfc04677820f9a4  src/kaggriculture/buffers.rs
2648da5f3c0f053206d0b51ee5fcb68a2c99c0ed76e4366a05bf17bd09843c85  src/kaggriculture/observe.rs
9aece0a5589bfc3dae7027fee19994adbd29a105933dcf4c092959e705ad765a  src/kaggriculture/env_tests.rs
2669c74e20f7ce2876c983ed56fa519aac121d662ab44ddd1191b7f89f091c4b  engine_rs/src/lib.rs
5f1b0c8a40f130a9f4f8ed91986e0e2c198027d4fc9172166ffc6269ae168698  src/kaggriculture/bindings.rs
```

## Final Clippy follow-up review

Read the final `decode_actions` loop and its sole caller. Replacing
`for seat in 0..2` plus `lengths[seat]` with
`for (seat, &length) in lengths.iter().enumerate()` plus `length` preserves
both iterations and their order: the caller always passes exactly
`&lengths[i * 2..i * 2 + 2]` after batch-length admission. No issue found.

The current `src/kaggriculture/env.rs` SHA-256 is
`3e55fae6a7a02d6aa8da1b908fd65a1d7b0494388b412fa432dffaa435e79b33`.
Reversing only those two substitutions in memory reproduces the originally
reviewed hash `e7e4284de04b4465cb03e9fbba972a30eb5596980455ba13ebdcc5173291a87f`,
confirming that this is its entire intervening change. This follow-up was a
source review, not an additional test or build run.

## Final report and delivery-scope review

Read `report.md`, `parity-benchmark.md`, the current cookbook adaptation
section/index/log, and the authoritative source-verified Python logs after the
watchdog test-isolation repair. The full run records **3,016 passed, nine
skipped in 196.61 seconds**; the requested focused run records **615 passed,
three skipped, 825 deselected in 19.12 seconds**. The task's current report and
cookbook sections use 3,016; the superseded 3,015 count is removed from them.

The benchmark receipt now includes explicit pristine export, locked release
build and wheel-extraction commands. All three timing series are retained,
including the two losing alternating pairs and host-variation limits. Claims
remain scoped to the timed native CPU component and finite parity evidence.
No additional implementation or reporting defect found.

Original-worktree commit publication remains blocked by protected Git metadata.
The report explicitly distinguishes that limitation from the local
`kg/sps-rollout` delivery clone and its adjacent commit/bundle receipt. Creation
and final identity checks of those delivery artifacts remain the main agent's
final step; this review does not claim that prospective artifacts already exist.
No tests or builds were launched for this final review.

VERDICT: APPROVE (reviewed implementation and report scope; original-worktree publication remains blocked)
