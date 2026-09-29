# Independent Task 1.4 trajectory-oracle review

Scope: `e197528...HEAD`, recorder/policy, Rust example, packed oracle custody,
recorder tests and exact native trajectory replay. No tracked changes. All
mutations occurred in `/private/tmp/kg-verify-env-oracle-20260929` and were restored.

## Finding

**P3 — Exercise semantic inventory guards after refreshing array custody**
`tests/tools/test_record_kaggriculture_env_reference.py:128-134` mutates an array
while leaving `manifest["arrays"]` unchanged, then specifically expects
`"custody"`. Every case exits at the generic metadata hash comparison, before
its intended lengths/offsets/indices/seed/terminal semantic validator.

Seven independently omitted guards survive not only the nine parametrized
inventory tests, but all 35 recorder tests other than the frozen-source-hash
check: length range, program offsets, transition indices, seed consumption,
terminal steps, done schedule, and terminal-value consistency. The frozen test
is deliberately excluded from source mutations because its recorder hash would
reject any edit, which is source drift detection rather than behavioral coverage.

Fix: retain the hash-corruption test separately; add coherent invalid-array
cases, refresh `manifest["arrays"] = recorder.metadata(arrays)`, and assert each
specific semantic rejection. For length bounds, rebuild offsets and packed
lengths coherently; for terminal values use a non-coverage economic counter so
an unrelated winner/coverage check cannot mask the guard. Run omission mutations.
This is a test gap, not a demonstrated production defect. Independent scratch
probes confirm all sixteen tested production semantic validators reject invalid
inputs and detect corresponding omitted-guard mutants.

## Verification and mutations

- Baseline: 36 recorder tests plus 1 full native replay = **37 passed** in 14.74 s,
  sampled peak 337,805,312 bytes. Replay compares all **16 games / 11,504
  transitions** exactly against the committed actual-TrainingBatch fixture.
- Restored final: **37 passed** in 15.85 s, sampled peak 342,016,000 bytes.
- Seven independent replay perturbations were killed: rewards, dones, before/after
  banks, before/after economic counters, and the separate mathematical reward.
  The six exact-output probes fail at game 0 / step 0 / seat 0 with field/action.
- Sixteen coherent-invalid-array semantic probes passed on current source and
  killed corresponding guard omissions: dtype, shape, length range, offsets,
  transition index, seed progression, terminal steps/dones/values/winner,
  economic/bank continuity, finite rewards, nondecreasing counters,
  positive coverage, and coverage-counter correspondence.
- Separate fixed-recipe and supervisor-inclusive RSS probes kill their omissions
  without launching workers; the RSS mutant changes the stop reason from memory
  to wall time at 700,000 rather than 1,100,000 counted bytes.
- Exploratory stock-test campaign: 24 mutants, 5 killed, 18 survived, 1 timed out.
  Survivors are not all findings: several are redundant rejection checks or
  mismatched malformed inputs. Only the seven demonstrated semantic coverage
  gaps above are reported. Detailed per-command outcomes are in
  `custody-mutations.json`; full-suite confirmation is in
  `full-tools-inventory-mutations.json`.
- The initial CLI-guard omission let the stock test enter a scratch recording
  worker and timed out at 12 s. That attempt is not counted as a clean mutation
  kill. The recorder supervisor was active until the harness killed the test process;
  a subsequent libproc inventory found no process in the scratch cwd. No output
  from this accidental recording attempt was accepted as verification evidence. The safe fixed-recipe probe
  replaced `supervise` with a sentinel before running `main`.
- Two initial baseline harness attempts failed before useful testing (git archive
  relative cwd prefix; pytest ancestor traversal). Corrected by an external
  scratch directory, explicit git environment, and explicit pytest root/confcut.
- SHA-256 restoration: all **13 copied tracked inputs** and scratch copies match
  the baseline byte-for-byte (`restoration-custody.json`).

The actual reference recorder was source-audited, not intentionally rebuilt.
Pinned source archives/hashes, policy recipe and Rust example use the actual
reference `TrainingBatch`; the fixed fixture's recorded lineage and exact native
replay are consistent. Existing recorded pod build/oracle results remain external
historical evidence rather than a new independent regeneration.
