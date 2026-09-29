# Actual A–G receipts

Planned expectations are in run-statement.md and the reviewed brief. These are
observations, not acceptance predictions.

- Base: kg/rebuild-env, e197528820ab7cfb429e21259000957370abf1c6; clean initially.
- A: existing bridge retirement verified. Existing regression strengthened (not
  duplicated); no missing-behavior red claimed. a-trim-tests: 3 passed, 76
  deselected; a-source-trim: exit 0, manifest OK. a-root-kernel: 9 passed, 249
  filtered, 33.53 seconds total (21.50 build / 11.28 test). No engine edits.
- B: initial missing-helper red also exposed ambiguous test float type; corrected
  before implementation. b-red-valid contains ONLY missing env/reward imports.
  b-green: all 5 tests pass; 6.06 s total / 848,805,888 sampled peak RSS bytes.
- C: c-binding-red: 246 failures exclusively absent KaggricultureEnv, 19
  deselected, 2.20 s. Initial registration typo then fixed in mod.rs.
  c-build also discovered uv's implicit editable release build; subsequent commands
  set UV_NO_SYNC=1 (documented by `uv help run`) to use the prepared environment
  and explicit debug maturin builds. No measured release check is claimed.
- C latch: c-latch-red first failed from a malformed keyword in its fixture,
  NOT the missing GIL release. That error was corrected before the valid red.

- Resume: operator committed partial work at `8d98ea8`. Pause interrupted the
  zero-reward mutation before its restoration, so HEAD contained that mutation.
  The resumed test failed at its independent return-bound assertion (1 failed);
  `Ok(r)` was restored immediately and the test passed (1 passed). The earlier
  actual-error-derived budget had allowed the mutation; that weak test was
  replaced, not counted as valid mutation evidence.
- C final: real detached-work latch red was the expected missing GIL release;
  restored implementation passed. Native destination/constructor suite reached
  265 passes before D.
- D: missing step behavior and absent reset fault hooks produced separate valid
  reds. Restored native core: 9 passed; admission helper: 6 passed; binding:
  328 passed. Premature live seed writes in step and reset caused the named
  rollback tests to fail at complete-state equality; restored hashes match.
  These mutations target live seed state, not a destination buffer.
- D optimized overflow attempt: stopped for sampled group RSS 1,007,714,304 bytes
  after 28.873 seconds, before the test body. Root dependency overflow checks
  are installed, debug rollback passes; release proof is PENDING (pod), not a
  release green. No second Mac attempt is authorized.
- E: prerequisite C/D code already implemented terminal reset and selected commit.
  New valid Rust suite passed 12 tests; first Rust attempt had only a malformed
  mutable binding and is not behavior-red evidence. Python E first exposed a
  malformed combined unit/market frame in six tests; it was repaired before the
  valid 11-test pass. Full Python native suite now 339 passed.
- E mutations: whole-batch publish in truncate failed at `unselected observation
  bytes changed: env=0, terminal=false`; byte-exact restore followed by 12 Rust
  passes. Old rank offset with stride one failed world-size 2 AND 8 with `rank 1
  collides with rank 0`; restored full native suite 339 passed. Each rank consumes
  67 seeds; ranks are exercised sequentially, at most E2 live.
- E source review: separate read-only agent found no C/D/E production defect.
  Source review adds no runtime qualification; F/G were outside that review.
- F: 42 missing-function failures became 42 passes; a valid contiguous-view
  case raised the grammar binding suite to 43. The final native environment
  suite has 340 cases after one/two-thread pool equivalence. No Python codec or
  device table adapter was added.
- G: recorder/replay missing-code reds and export-drift helper reds are retained
  in oracle-receipt.md. Final targeted Python: 497 passed, 2 missing-fixture
  failures, no skips. The only recording attempt exceeded RSS during reference
  compilation, publishing nothing. Full16 replay/mutation and release costs
  remain PENDING (pod); timing.json records no substituted debug costs.
- Final: root 274 passed/5 ignored, engine 69 passed, rs-prepare/trim/dev build
  pass. Both broad preparation commands hit the memory limit during the first
  existing Python typing probe, after static checks (and Rust for prepare).
  The coverage-doc blank-line lint failure was repaired; separate final
  docs-lint/docs-fresh pass without an override. See results.md,
  final-checks.json and changed-files.tsv for the complete handoff.
