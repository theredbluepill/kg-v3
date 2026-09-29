# Actual A–C receipts (ongoing)

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
