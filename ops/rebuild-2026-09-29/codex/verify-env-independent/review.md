# Independent verification — Task 1.4 native environment

Reviewed `kg/rebuild-env` at `1e2676079856693864384bf972d64e2cdb21510e` against
`e197528820ab7cfb429e21259000957370abf1c6...HEAD` (the merge base is exactly the requested base).
The three commits are `8d98ea8`, `9dc2d02`, and `1e26760`. No production defect
was found. Two P3 edits are recommended below. No tracked file was modified.

## Findings

1. **P3 — Exercise semantic inventory guards independently of checksum custody.**
   `tests/tools/test_record_kaggriculture_env_reference.py:128–134` changes an
   array while leaving `manifest["arrays"]` unchanged, then expects a `custody`
   error. All nine cases therefore exit at the generic hash comparison before
   reaching the named semantic invariant. Removing seven semantic guards
   (length bounds, offsets, transition indices, seed consumption, terminal
   steps, terminal done schedule, terminal values) still passes all 35 recorder
   behavioral tests; only the frozen recorder-source hash test was excluded
   because any source mutation necessarily changes that hash. Independent
   coherent-invalid-input probes verify the current production guards work and
   detect their removal. **Fix:** keep separate hash-corruption coverage, refresh
   `manifest["arrays"] = recorder.metadata(arrays)` after coherent semantic
   corruption, and assert the specific error. Keep offsets/token counts coherent
   for a length-range probe; use a non-coverage counter for terminal consistency.
   Evidence: `oracle/review.md`, `oracle/full-tools-inventory-mutations.json`,
   `oracle/semantic-guard-probes.json`.
2. **P3 — Remove stale constructor approval status.**
   `docs/rl-api-specs.md:1061` says the contract wording remains for Claude
   review, but `docs/kaggriculture-contract.md:220,298` already records the
   accepted v4.2/Q1 constructor refinement. **Fix:** state that v4.2 incorporates
   the approved refinement.

## Requested checks

| Check | Actual result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | 69 passed: 41 unit + 9 PRNG + 19 replay; zero ignored |
| `cargo test --locked --offline -- --test-threads=1` | 274 passed, 5 ignored |
| `uv run python scripts/check_engine_trim.py` | PASS |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | Combined attempt hit RSS guard; complete sharded result 2,042 passed, 7 skipped |
| `uv run mypy python/owl scripts` | PASS, 64 source files |
| `git diff --check` | PASS |

Bounded local attempts used at most 115 seconds and a sampled 960 MiB process-group
RSS limit, offline dependencies, and at most two Cargo build jobs. Root Rust's
initial default parallel test run hit that RSS limit; one test thread completed.
Python file/node shards completed 2,041 tests. The unchanged generic typing probe
passed separately with `--noconftest` (1 test), avoiding unrelated Torch runtime
imports while mypy builds its fresh cache. It needs only pytest's built-in
`tmp_path` and no compile-state fixture. Model-head shards total 65 passing
cases. No test body or expected result was changed to obtain these counts.
See `python-summary.json`, `python-shards.json`, and their command receipts.
The root extension was freshly built from reviewed source with
`uv run --offline maturin develop --locked` before the accepted Python runs.

## Mechanisms and mutation evidence

- Native candidate preparation/ordered publication, terminal capture/autoreset,
  reset and selected-row truncation, checked seed successor reservation,
  per-rank stride, exact reward predicate/two-rounding formula, HIRE cast
  admission, and array borrow/preflight paths were reviewed directly.
- 24 core mutations were rejected by their named Rust assertions, plus one
  timing-fixture density mutation: **25/25**. These cover own economic counters,
  reward telescoping, each admission predicate class, wrapping/wrong-stride
  seeds, early reset/step seed writes, early terminal clearing, whole-batch
  truncation publication, transition clearing, terminal timing, all six HIRE
  helper oracles and lifecycle wiring, and GIL progress. All source bytes restored.
- 13 rebuilt binding mutations: **12 rejected**, one redundant-guard survivor.
  Rejected cases include strict seeds/reward dict, overlapping regions, table
  bits/constants, C layout/exact shape, codec tail, L6 econ-output omission,
  wrong stride (both world sizes 2 and 8 fail), and native inverted dones.
  Removing the explicit alignment predicate still rejects all malformed inputs:
  numpy 0.28 `PyArrayMethods::as_slice_mut` independently requires alignment
  (`array.rs:779`), so this survivor is not a coverage defect.
- The exact-shape mutant used an extra scratch-only probe with an incorrect
  `(2,2,7,200)` shape containing the same number of cells as `(2,2,200,7)`.
  It passes current source and fails when exact-shape admission is removed.
- The native dones mutant fails the real reference replay at game 0 / seed 17000 /
  step 0 / seat 0 / field dones, with decoded action in the error. No source-hash
  failure was counted as this kill.
- The strengthened duplicate-grammar-path guard passes on baseline/restoration
  and fails after inserting a path-included duplicate in a separate scratch copy.
- Canonical fat-LTO release compilation exceeded the Mac RSS guard. A separately
  labeled release build with LTO disabled, 16 codegen units and one build job
  passed `release_dependency_overflow_is_caught`; disabling the engine overflow
  override makes the same test fail at its expected caught-overflow assertion.
  This is optimized release correctness evidence, not a canonical-LTO rerun or
  a performance measurement. The recorded canonical pod proof was inspected.
- The recorder's 16-game/11,504-transition fixture is from the pinned reference
  `TrainingBatch`, not an alternate simulator. Full replay is bit-identical on
  rewards/dones/banks/counters/seeds/terminal records. The oracle lane passed
  36 recorder tests + 1 full replay both before and after restoration.
- Oracle mutation work includes seven killed replay/math perturbations,
  sixteen successful coherent semantic rejection probes with corresponding
  guard omissions detected, and safe recipe/RSS guard probes. The exploratory
  24-omission stock campaign and seven whole-suite confirmations are reported
  separately in `oracle/review.md`; not all stock survivors are defects.
  One early CLI omission entered a scratch recorder and timed out before the
  safer sentinel probe was used; it was not counted as an assertion kill, and
  no worker remained. No new full reference regeneration is claimed.

All mutation builds used the archive scratch copy; the binding harness extracts
fresh dev-wheel extensions only into scratch and verifies the imported extension
path. The shared Cargo cache and Python environment are build/runtime artifacts.
An initial scratch replay setup failure (Git cwd prefix) was corrected with
explicit read-only Git environment; its receipts remain in
`binding-initial-baseline/`. A receipt classifier initially mistook Cargo's
`error: test failed` footer for a compiler failure; it was corrected against the
actual logs, preserving `core-mutations-initial-classification.json`. Neither
harness correction changed reviewed source or test outcomes.

After source restoration, the root Kaggriculture Rust subset passed **117 tests**
with 3 ignored, and the four native/codec/oracle Python files passed **424 tests**.
Receipts: `restored-rust-kaggriculture.json`, `restored-python-native.json`.

## Custody and limits

`restoration-custody.json` compares SHA-256 of all **1,870 tracked files** before
and after: zero differences, no staged/unstaged tracked diff. The main five
mutated native scratch sources are byte-identical to the reviewed originals;
the oracle lane separately verifies its 13 copied tracked inputs. This directory
contains only untracked review evidence. No commit, branch switch, reference
worktree edit, kernel edit, training, GPU action, or deployment occurred.

Native L6 overwrite and lifetime checks do not qualify Task 1.5's pinned CUDA
entry fence or DMA behavior. No end-to-end throughput claim is made.

VERDICT: APPROVE WITH EDITS
