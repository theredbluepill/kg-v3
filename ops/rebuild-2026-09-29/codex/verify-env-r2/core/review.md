# Independent Task 1.4 core verification

Source: `ba9b59bbf1581bda5b7e9389b74f16c8ba4c8f98`, reviewed against
`e197528...HEAD` and `ops/rebuild-2026-09-29/briefs/1.4.md`.

**No core production defect found. Core scope recommendation: APPROVE.**
The parent verifier owns the combined verdict, prior-finding statuses, full
check counts, Python bindings, world-8 execution and reference replay.

## Source review

- `src/kaggriculture/env.rs:34`: seed admission is nonnegative i64 with positive
  stride. Reservation computes every checked successor on a local stream in
  selected-env order. No native rank offset or training-only seed cap exists.
- `src/kaggriculture/env.rs:350`: reset/truncate prepare candidate games and
  observations without changing live slots, seeds, terminal records or outputs.
  Worker failures and panics join before ordered error selection. All-false
  truncation produces no selected output write and consumes no seed.
- `src/kaggriculture/env.rs:424`: action shape/raw transport admission precedes
  local seed reservation; complete grammar decoding precedes candidate steps.
  Banks, economic counters, rewards and terminal records are captured before
  automatic reset. Horizon prediction is checked against the kernel result.
- `src/kaggriculture/env.rs:647`: publication completes preflight before any
  copy. Truncation uses `commit_selected_rows`, which copies exactly selected
  observation rows with exhaustive field destructuring. Its pending transition
  is None, retaining all six transition buffers and the cached transition.
  Slot/stream installation uses swaps after admitted copies.
- `src/kaggriculture/reward.rs:18`: finite/nonnegative checks, active cap sum and
  per-component admission match the brief exactly, including separately rounded
  binary64 products `W*s` and `W*d`. Reward calculation preserves the required
  economic float32 rounding, terminal float64 addition and final float32
  rounding. Only counters 0/1/2 and raw terminal banks affect reward.
- `src/kaggriculture/admission.rs:16`: successful HIRE deltas drive exact
  Fibonacci cost admission. The strict float64 2^64 bound covers costs below
  u64::MAX that round up to 2^64. No-execution and zero-multiplier paths avoid
  inventing configuration caps. Lifecycle wiring is before candidate publication.
- The root package-specific release overflow policy leaves vendored kernel bytes
  unchanged. Catching the candidate unwind is the transactional failure path.

## Fresh mutation results

`run_core_mutations.py` ran 24 distinct source mutations in
`.codex-tmp/verify-env-r2-core`, a separate copy of tracked source. Every case
failed at its named semantic assertion: **24 killed / 24 attempted**. These
cover own reward counters, endpoint telescoping, all reward-admission classes,
seed stride/overflow, premature step/reset seed writes, early terminal clearing,
whole-batch truncate publication, transition clearing, horizon timing, HIRE
lifecycle wiring and all six admission helper oracles, and native GIL progress.
The GIL mutant removes `py.detach` from actual native observation work; the
controlled latch rejects it because the Python thread cannot progress.

`run_additional.py` replaced the dense timing fixture with an early state. The
named fixture-density assertion rejected it: **1 killed / 1 attempted**.
Thus the core source campaign is **25 killed / 25 attempted**, with no compiler
errors or resource stops counted as kills. See `core-mutations.json` and
`core-timing-fixture-density.mutation.json`; all patches and command logs remain
beside those receipts.

Release arithmetic used an additional build-policy mutation:

| Check | Result |
| --- | --- |
| Canonical fat-LTO release proof | Build stopped at sampled 960 MiB guard (16.94 s); test body did not run |
| Release, LTO=false/codegen-units=16 | 1 passed |
| Same build, engine overflow-checks=false | 1 failed at the required caught-overflow assertion |
| Restored release policy | 1 passed |
| Restored debug Kaggriculture subset | 117 passed, 3 ignored |

The release variant demonstrates optimized overflow/rollback correctness, not
canonical-LTO execution or performance. Source comparison shows **no changed
paths** between the existing canonical pod proof at `9dc2d02` and current HEAD
for `Cargo.toml`, `Cargo.lock`, `src/` and `engine_rs/`; see
`pod-release-source-identity.json`. The existing pod pass/override-off failure
therefore applies to identical Rust/kernel/build source, but is reported as
inspected historical evidence rather than a fresh execution.

## Custody and limits

Every source mutation restored the scratch bytes in a finally block and checked
the corresponding primary file was unchanged. `restoration.json` verifies all
seven recorded build/native/test source files remain byte-identical to their
pre-run hashes in both primary and scratch. `git status --short
--untracked-files=no` remained empty. No tracked source was edited; only this
new operational evidence directory and ignored scratch/build artifacts were
created. Native builds were serialized with the other verifier lanes.

Commands used offline dependencies, one test thread, no more than two Cargo
build jobs, and per-command 115-second / sampled 960 MiB guards. No training,
GPU operation, kernel modification or throughput claim occurred. Task 1.5's
pinned CUDA fence and DMA behavior remain outside Task 1.4 core qualification.
