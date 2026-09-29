# Task 1.4 independent binding review, second verification

Reviewed `ba9b59bbf1581bda5b7e9389b74f16c8ba4c8f98` against `e197528...HEAD`. No production or documentation finding in this lane. The prior report’s P3 at `docs/rl-api-specs.md:1061` is **RESOLVED**: lines 1060–1062 now state that Q1 was approved and contract v4.2 incorporates it; the accepted contract records that refinement explicitly.

## Inspection

- Native constructor and all four lifecycle signatures match `python/owl/rs.pyi`; all 140 typed output extractors match their stub names, positions and NumPy dtypes (`static-abi.json`). Buffer shapes match the 29-observation/six-transition inventory. Constructor arguments and `hire_limit` are required, all lifecycle output buffers are keyword-only.
- The boundary acquires fallible typed borrows, validates layout, shape, alignment and disjoint byte regions, then obtains Rust slices. Guards remain live across detached native preparation/publication. Step allocates its Python metrics result before native commit.
- The extension retains the header encoder and registers `KaggricultureEnv` plus encode, decode, tables and constants. Table arrays are separate copies with 964 exact booleans; constants are the accepted version/names/widths. Codec publication is staged by the existing native grammar.
- Native L6 is the one-buffer contract: poison controls distinguish incomplete writes in all 35 outputs including padding; the test covers observe/reset/ordinary and terminal steps. Selected-row truncate is covered separately. CUDA lifetime fencing remains Task 1.5, as the brief and current docs state.

## New independent execution

A scratch copy of all 2,350 tracked source files was used; the main installed extension was never replaced. Each freshly built dev wheel was extracted only into scratch. A pytest startup plugin asserts the actual imported extension path. Build/tests ran offline with one command at a time, two Cargo build jobs, 115-second and 960 MiB sampled process-group limits. All builds/tests completed without a resource stop.

All 113 selected baseline cases passed before mutations, including the 16-game / 11,504-transition TrainingBatch oracle. Fourteen one-at-a-time mutation builds succeeded. Thirteen mutants fail their intended behavioral assertions: boolean reward coefficients, seed type admission, plain reward dictionary, exact key count, overlapping byte regions, table bits, constants version, C layout, exact shape with preserved element count, codec zero tail, L6 transition publication, per-rank stride at worlds 2 and 8, and native dones vs the complete-game oracle. Error excerpts are in `failure-audit.json`, exact patches and per-command receipts in this directory. Source-hash drift is not used as a mutation kill.

The remaining mutation removes only the explicit alignment predicate and **survives** all 35 unaligned-output probes. This does not admit invalid buffers: numpy 0.28.0 `src/array.rs` lines 742–784 independently checks alignment in `as_slice` and `as_slice_mut` before creating slices. This is redundant defense, not a new defect or a claimed kill. The harness exits 1 because it accurately records the survivor.

After byte restoration, a fresh restored wheel passed **395 tests** (native lifecycle, native grammar bindings, complete-game oracle, and seven scratch-only exact-shape/boolean-coefficient probes) in 12.30 seconds. This includes 388 repository tests and seven independent scratch probes; do not add these repeated cases to the parent’s full-suite denominator.

`final-custody.json` confirms all 2,350 copied tracked files match both the scratch originals and current worktree, and the scratch extension has been restored to its original absence. No tracked file was modified. These are CPU semantics results; they make no training, GPU lifetime or complete-update throughput claim.

**Lane verdict: APPROVE.**
