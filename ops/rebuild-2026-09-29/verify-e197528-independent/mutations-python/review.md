# Independent Python oracle/guard review of e197528

Scope: Task 1.3 Python oracle custody, resource watchdog, retired engine bridge, and CUDA-only pinned-test admission. Target: each selected guard must admit its baseline, make its focused test fail when removed/corrupted on a scratch copy, and pass after byte-exact restoration. No training, GPU allocation, real corpus regeneration, or tracked edit was performed.

## Finding

**P2 — Live engine dependencies escape oracle source custody.** `scripts/kaggriculture_observation_oracle/regenerate.py:975-978` records only engine `TRIM_MANIFEST.json`, `src/lib.rs`, and `Cargo.lock`; `source_snapshot()` at lines 985–990 never snapshots `engine_rs/src/py_random.rs` or `engine_rs/Cargo.toml`. Hashing the trim manifest does not verify the current bytes of its retained files. Both are live producer/build inputs, so an edit at unchanged HEAD can change generated states or build behavior while the recorded source identity remains unchanged, and every later `check_source_snapshot()` accepts it.

Two independent reproductions:

1. `real_source_snapshot_probe.py` copies real repository bytes to scratch and calls the actual source capture/check functions with real bounded git commands (no mocks). Appending a comment separately to each omitted input survives the check. Both are restored byte-for-byte and restored checks pass. `real_source_snapshot_receipts.json` records original/mutant/restored hashes and actual HEAD e197528820ab7cfb429e21259000957370abf1c6.
2. `source-gap-test.py` reuses the repository's regeneration harness, changing either omitted input during mocked producer execution. Both expected-rejection tests fail, and the final fixture gets installed. A narrow scratch-only fix adding both dependencies to SOURCE_PATHS gives **2 failed → 2 passed → 2 failed after restoration**. See `source-gap-receipts.json` and phase logs. Source bytes and mutated fixture input bytes are restored; the fix is not retained. This harness proves custody/installation behavior, not actual corpus regeneration.

Fix: include every live retained engine/build input in source capture and rechecks (at minimum these two files; preferably enumerate the retained inputs), or verify retained manifest bytes at capture and every relevant recheck. Add both drift regressions and make the source-identity metadata describe the captured inputs.

## Discriminating mutation checks

Sixteen scratch-only guard mutants were all killed. Selected baseline and restored invocations each account for 26 passing tests in total; every mutant invocation exited 1. These are focused mutation counts, not the broad repository suite counts owned by the parent verifier. Details, exact commands, hashes and per-phase logs are in `results.json`; `restoration.json` proves exact restoration of copied source and unchanged repository bytes.

| Guard mutant | Baseline → mutant → restored exit | Exact restoration |
|---|---|---|
| seat_bytes | 0 → 1 → 0 | yes |
| header_order | 0 → 1 → 0 | yes |
| schema_version | 0 → 1 → 0 | yes |
| duplicate_record | 0 → 1 → 0 | yes |
| source_header_step | 0 → 1 → 0 | yes |
| quota | 0 → 1 → 0 | yes |
| byteplanes | 0 → 1 → 0 | yes |
| source_snapshot | 0 → 1 → 0 | yes |
| archive_custody | 0 → 1 → 0 | yes |
| recorder_copy | 0 → 1 → 0 | yes |
| caller_baseline | 0 → 1 → 0 | yes |
| caller_growth | 0 → 1 → 0 | yes |
| shared_deadline | 0 → 1 → 0 | yes |
| retired_authored_set | 0 → 1 → 0 | yes |
| bridge_retirement | 0 → 1 → 0 | yes |
| pinned_availability | 0 → 1 → 0 | yes |

Pinned admission was evaluated from the actual `_CUDA_PINNED` AST expression with mocked CUDA availability false and true. No torch allocator was imported or called in that probe. Replacing the skip condition with false fails the false-CUDA case; restoring the original expression passes both cases. GPU execution and actual pinned allocation remain unqualified on this Mac.

Caller-growth and shared-deadline mutants allowed the existing 30-second child to finish and then failed their expected-error tests; normal and restored guards stop the child promptly. All child commands finished or were reaped by the actual watchdog.

No further Python-specific defect was established. `git diff --stat` was empty after these checks.
