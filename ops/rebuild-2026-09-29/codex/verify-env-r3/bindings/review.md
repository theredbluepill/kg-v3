# Independent Task 1.4 binding verification, r3

Reviewed `1e63597957ed4995dc3eac47e996349a485670a9` against `e197528...HEAD`, Task 1.4 brief, current native-buffer Reference, native implementation, stubs and lifecycle/codec/table tests. No production or documentation finding in this lane. Lane verdict: **APPROVE**.

## Static audit

Constructor and all four lifecycle signatures match `python/owl/rs.pyi`; all 140 typed output extractors match their corresponding method's names, positions and NumPy dtypes (`static-abi.json`). Shapes match the 29-observation plus six-transition inventory. All output arguments are required keyword-only parameters. Native caller-owned storage is retained; output borrows are fallible. Shape/layout/alignment/address-overlap preflight completes before the first Rust mutable slice. Inputs are also included in the disjointness check. Typed guards stay alive across detached preparation and commit. The step metrics dictionary is allocated before native commit, preserving rollback on Python conversion failure.

The header encoder remains exported. The class plus four cold codec/table functions are registered in the existing extension. Tables are separately owned arrays with exact expected 964 booleans; version/names/widths match the adopted grammar. Codec writes use native grammar staging. The L6 poison oracle covers all 35 outputs, including padding, for observe/reset/ordinary step/terminal step; selected-row truncate has a separate poison-and-preservation oracle. CUDA entry fencing is explicitly Task 1.5, outside this native qualification.

The prior r1 documentation P3 remains **RESOLVED**: `docs/rl-api-specs.md:1060–1062` states that Q1 was approved and v4.2 incorporates it; `docs/kaggriculture-contract.md:3` marks v4.2 accepted. The r2 recorder finding is owned by the independent recorder lane, not reclassified here.

## Independent execution

Fresh current sources were copied into `.codex-tmp/verify-env-r3-bindings`; the build target was independently APFS-cloned from the cache. No scratch target symlink points at the main target. Each wheel was built offline, extracted into scratch, and never installed. A pytest startup plugin checks the exact imported extension path. Each build/test had a 115 s / 960 MiB sampled aggregate-process-group resource guard; all completed within bounds, with `CARGO_BUILD_JOBS=2`.

All **113 selected baseline cases passed**, including the recorded reference TrainingBatch's 16 complete games / 11,504 transitions. Fourteen one-at-a-time mutated builds succeeded. Thirteen mutants failed ordinary behavioral assertions: reward boolean coefficients, strict seed types, plain reward dictionary, exact key count, independent-base byte overlap, table bits, constants version, Fortran/C layout, exact shape despite unchanged element count, codec zero padding, L6 transition publication, world-2/world-8 stride, and native dones. The native-dones mutation fails at game 0 / seed 17000 / step 0 / seat 0. The missing `transition_econ_after` copy fails all four L6 operation variants. Exact patches, per-command receipts and failure lines are retained in this directory.

The alignment-only mutant **survived** all 35 selected unaligned-output tests: rust-numpy 0.28.0 `src/array.rs:742–784` independently requires alignment in `as_slice` and `as_slice_mut`, so invalid inputs remain rejected. This redundant-guard survivor is not counted as a kill or a production defect. The mutation harness exits 1 to report the survivor accurately.

After restoring source bytes, a fresh restored wheel passed **395 cases**: native lifecycle, grammar bindings, reference replay, and seven scratch-only exact-shape/boolean-coefficient probes. This is 388 repository tests plus seven verifier probes; the repeated repository cases must not be added to the parent's broad-suite denominator. Final restored pytest wall time was 25.781 s, sampled peak RSS 377913344 bytes. `final-custody.json` confirms all 506 copied tracked files match the current worktree and the extracted extension is restored to its original absence. No tracked modification was made.

These are bounded CPU correctness results, with no claim about training quality, GPU transfer lifetime, or complete-update throughput.
