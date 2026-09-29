# Independent verification of kg/merge-1-3 at b8747b6

Reviewed `b51b0c0...b8747b6e8acece5f561d09a75bb914364a60ac05`: staging merge `e197528` of Task 1.3 (`dc6b200`) onto integration `b51b0c0`, plus the subsequent engine-source custody repair. HEAD remained at the launch revision throughout verification. No new actionable findings; the one previous P2 is resolved.

## Every prior finding

Source report: `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/codex/verify-merge-1-3-r1.md`. It contains exactly one finding. Its historical contents were preserved; its current disposition is recorded here.

| Severity and original finding | Status | Current location, fix and verification |
| --- | --- | --- |
| P2 — Oracle source custody omits live engine inputs | **RESOLVED** | `scripts/kaggriculture_observation_oracle/regenerate.py:964` now captures engine Cargo.toml, py_random.rs and econ_attrib.rs; `:990` rejects undeclared/missing Rust inputs at capture and recheck; `:1030` checks captured bytes. Tests at `tests/tools/test_observation_oracle_custody.py:611`, `:669`, `:690` cover drift, identity inclusion and undeclared modules. Three fresh real-source drift probes are rejected; adding a module or build.rs is rejected at capture and recheck. Scratch bypasses of each added input and each declaration call fail focused tests. No further code fix is requested. |

Disposition totals: **1 RESOLVED, 0 PARTIAL, 0 UNRESOLVED**. No additional finding requiring severity/location/fix was established.

The committed corpus predates the repair and lacks these three historical hashes. This remains explicitly disclosed at `docs/rules-parity-coverage.md:501` and `cookbook/references/structured-observations-preserve-legal-state-and-order.md:141`. Their committed bytes equal corpus root commit `469e8ec`; that comparison cannot establish unrecorded dirty state during the original generation. No corpus regeneration or retrospective complete-custody claim is made here. Regeneration is the condition for a future fully captured fixture identity.

## Fresh required checks

| Command | Result |
| --- | --- |
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**, 0 failed: 41 unit, 9 RNG, 19 replay; zero doc tests |
| `cargo test` | **254 passed**, 0 failed, **4 ignored** |
| `uv run python scripts/check_engine_trim.py` | **PASS** |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | **1,625 passed**, **7 skipped**, 0 failed/errors, equivalent selection completed through shards |
| `uv run mypy python/owl scripts` | **PASS**, 62 source files |

The current native extension was rebuilt successfully with `uv run maturin develop --locked --offline` before Python checks. Offline environment settings were retained; Cargo and Rayon used two workers, OMP/MKL/Rust test threads one. No test-source exclusions or parity-disable variables were introduced.

The combined Python command hit the sampled 1,000,000,000-byte process-group guard after 7.47 seconds (sampled peak 1,005,846,528). All 45 selected test files subsequently completed in **55 successful invocations**. `test_model_heads.py` and `test_kaggriculture_parity.py` also hit the file-level guard, then completed in groups of up to eight selected test cases. Aborted attempts are excluded from totals. JUnit aggregation verifies **1,632 unique cases with no duplicate counting**. The seven skips are two pinned-memory CUDA cases, two FlashAttention/CUDA cases, the x86 quantized backend, the future native grammar-table binding, and the future native Kaggriculture evaluation environment. These are platform/task limits, not executed-path passes. Exact reasons and unique case inventory: [python-summary.json](python-summary.json).

Each command has a log and JSON receipt in this directory. Additional read-only checks passed: diff whitespace check over non-ops changes and documentation freshness (the latter checks its own working-tree mapping, not semantic truth).

## Both parents preserved

The Git-blob test-name inventory finds zero lost declarations after accounting for the kernel-file relocation and the authored-inventory test rename: integration 1,099 declarations, Task 1.3 1,117, merge 1,217, current HEAD 1,220. These declaration counts differ intentionally from parameterized runtime test counts.

All **nine** kernel tests moved from `engine_rs/tests/grammar_kernel.rs` to `src/kaggriculture/grammar_kernel_tests.rs`; assertion coverage survives. Nine ordinary grammar tests previously duplicated through the engine bridge still run in the root module. Thus engine 87 → 69 is fully accounted for, while the root grows from 164 passes/two ignored to 254 passes/four ignored.

All **137** protected integration model/trainer/Kaggriculture/config/test blobs are byte-identical to `b51b0c0`. This includes cuBLAS-only stack/backend claims, direct compile entry checks, the concurrency lock, runtime trunk guard, startup validation/telemetry, and Tasks 3.1–3.4 registration, evaluation, truncation, config and workload guards. Existing GPU qualification limits remain unchanged. The observation declarations in `python/owl/rs.pyi` are additive.

The new root engine dependency and ordered/arbitrary-precision JSON feature unification survive. The retained Orbit production generator is untouched; its test-only Number decoding fixes the unified serde representation and is covered by a discriminating mutation.

Running the committed `retire_grammar_bridge.py` against a scratch copy of the integration manifest reproduces HEAD and `dc6b200` exactly: SHA-256 `b2d1892a15baf5653e493fdb885c5f2baac7a21095a87eb95d9730782ad9272f`. A second run is identical; scratch bytes were restored. The earlier checked-in receipt also records generator execution. The live trim checker passes.

Both parents' cookbook log sections survive with byte-identical bodies: all 89 integration sections and all 81 Task 1.3 sections. Current docs distinguish historical test counts, bridge retirement, the L4 repair, corpus/schema correctness, the custody repair and its historical limit, and still-open optimized timing/pinned/GPU work. Detailed inventories and exact comparison scripts: [parent audit](parent-docs/report.md).

## Fresh mutation evidence

All mutations used scratch copies and were restored byte-for-byte. Baseline and restored checks pass.

- **41 native source mutants** compiled successfully and failed their intended test assertions. They cover buffer size/overflow/publication/reuse, config/range/finiteness, exact counts/coordinates/privacy/ranks, constructor keys, clock and cardinality, row checks, snapshot count, relocated grammar execution, input policy/domain/order, reconstruction anchors/coverage/redundancy/bitwise comparisons, stream formats/EOF, full corpus comparison, and L4 feature unification. A coherent exact-plus-scaled market corruption passes row/custody checks and then fails the independent reference comparison at record `official:95324500:0`, seat 0, offset **889** (1.0001 versus 1.0). [Native matrix and logs](encoder-scratch/report.md).
- **22 Python mutation executions** fail focused tests and pass after restoration. They cover fixture bytes/header order/schema/identity/quotas, byteplanes, source/export/recorder custody, resource budgets/deadline, safe pinned admission, bridge/authored-set retirement, all three repaired engine inputs, and declaration checks at capture and recheck. A supplemental header-order probe also rehashes coverage to isolate header-order custody from downstream coverage checks. [Python matrix and limits](python-mutations/README.md).
- **8 boundary-input mutations** are rejected before writes: shape, dtype, endianness, strides, readonly storage, alignment, aliasing, and invalid late-environment state. All output bytes and pointers are unchanged; the restored valid batch matches baseline bitwise. [Boundary receipts](encoder-scratch/boundary-probes.json).
- **5 real-source custody probes** independently exercise the unmocked snapshot/check implementation against copied repository files: all three repaired inputs reject drift, and a new source module/build.rs is rejected at both capture and recheck. All inputs restore exactly. [Custody receipts](source-snapshot-probes.json).

Native scratch baseline and restored subset each pass 97 tests with two ignored and 159 filtered; broad worktree counts are reported separately above. Python regeneration phase mutations use mocked executable runs and do not claim actual corpus regeneration. CUDA allocations, optimized timings, throughput, training and competitive strength are outside this verification.

## Workspace custody

All **1,633 tracked files** match the initial SHA-256 inventory. `git status --short --untracked-files=no` is empty, and HEAD remains `b8747b6e8acece5f561d09a75bb914364a60ac05`. Existing untracked verification directories were preserved. This review's evidence is untracked under `ops/rebuild-2026-09-29/verify-b8747b6-independent/`; native build scratch is under `.codex-tmp/`. No tracked implementation, test, cookbook, fixture, manifest or historical review file was modified. [Final tracked-byte check](tracked-after.json).

VERDICT: APPROVE
