Independent CPU oracle/guard review, current HEAD `666deec`, scratch only.

No additional defect found in the assigned oracle/guard families. No GPU computation, compilation or remote process was started. The retained-record baselines are historical data checks, not fresh GPU validation.

All current-version checks passed:

| Check | Count | Evidence |
|---|---:|---|
| Oracle/guard assertions, including retained controls and fault mutations | 136 / 136 | `mutations.json`, `oracle-mutations.log` |
| Driver stop/deadline/backend/exit/timeout/judge/control matrix | 11 / 11 | `driver-guard-matrix.json`, `driver-matrix.log` |
| Independent malformed-backend and launch-exception probes plus baseline | 3 / 3 scenarios | `driver-mutations.json`, `driver-mutations.log` |
| Full current standalone guard suite | 8 / 8 groups | `guard-suite.log` |
| Current cleanup suite, including as-run positive control | 6 / 6 scenarios | `cleanup.log` |
| Fresh source regressions rejected by designated checks | 5 / 5 | `source-regressions.json`, `source-regressions.log` |

Mutation coverage inventory (every individual probe and its result appears in the JSON evidence):

- C1: reference/output nonfinite counters, compiled/eager packing and FlashAttention claims, omitted density results, actual tensor outlier coordinates and valid/masked NaNs, two-times-median classification and zero-median regression. Valid retained controls and all-zero classification pass.
- C2: output/gradient/parameter record corruption, wrong/absent status, packing mismatch, missing parameter gradients, ordinary relative-error and amended key-bias bounds; CPU tensor errors, nonfinite output/gradient/parameter values, masked gradients, invalid/packed mask inputs. Root's separate matrix covers the new reference-side nonfinite repair more deeply.
- C3: step/error/sample log probability, both replay paths, self and perturbed teacher KL/event bounds, values for sample/replay/critic and all four teacher paths, loss/gradient finiteness, FlashAttention and compiled-call claims. Every injected fault was rejected.
- C4: incomplete records, FlashAttention false, chunk-count mismatch, nonfinite training loss, missing derived record. Every injected fault was rejected.
- Shared source/compile guards: unavailable CUDA, missing compiled callable, incorrect compile count, force-FlashAttention disabled, wrong imported source tree/HEAD, missing FlashAttention. Every injected fault was rejected.
- Wrapper: contaminated backend environment, missing argument separator, invalid backend choice. Every injected fault exited nonzero.
- Driver: already stopped, deadline expiry, missing/wrong start/wrong end backend records, subprocess exit/timeout, judge exception/error, malformed backend JSON and launch exceptions. Faults stop subsequent work with the intended code; baseline and expected-control behavior pass.
- Template census: injected extern GEMM, template definition/launch, and all four marker families appear in the census. This is text-parser sensitivity only.
- Summary: scratch regeneration is byte-identical; bad outlier-format input is rejected; changing one input global-SPS value propagates to derived efficiency.
- Cleanup: Phase 1/2 signals, TERM-resistant child escalation, main-thread signal inside Popen and after Popen before registration, normal completion, and as-run positive control all produce the expected result.

Fresh source-regression mutations, each applied only to the second scratch bundle and restored:

| Mutation | Result |
|---|---|
| Restore `med and` bypass in C1 | KILLED: median-zero classification regression fails |
| Remove two teacher value paths from C3 | KILLED: omitted-path nonfinite/range mutations escape and the guard test fails |
| Remove `Driver.stream` exception boundary | KILLED: malformed record and spawn failure return 0 and launch Phase 2, violating the guard probe |
| Disable spawn signal deferral, signal after Popen | KILLED: exact boundary PID survives, violating cleanup check |
| Disable spawn signal deferral, signal inside Popen | KILLED: exact boundary PID survives, violating cleanup check |

No source regression survived its designated check. The cleanup harness kills its intentionally surviving positive-control/mutant processes afterward.

Cleanup measurement limitation: sandbox process inventory disallows `ps`/`pgrep`, so a wrapper replaces the inventory functions with `os.kill(pid, 0)` probes of exact PIDs recorded by the test and suppresses token-name scans. In spawn-window scenarios, a stage can be killed before writing its own PID file; the harness separately records `boundary.stage_pid` before signalling and directly probes that PID. The final independent check covered 13 exact leader/boundary/survivor PIDs from the cleanup and source-regression evidence; all were absent (`final-pid-check.json`). No process-name sweep is claimed. GNU timeout is emulated, as documented by the repository suite.

Restoration: initial SHA256 inventory covers 117 files (excluding generated Python bytecode). All 117 match byte-for-byte in the tracked source bundle, the oracle scratch copy, and the source-regression scratch copy (`scratch.json`, `final-restoration.json`, `source-regression-restoration.json`). No tracked modification was made.
