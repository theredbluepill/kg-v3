# Task 1.4: Claude review (2026-09-29)

**Scope.** Codex's two commits on `kg/rebuild-env`: `8d98ea8` (partial work, committed as-is at the operator pause) and `9dc2d02` (the resumed session `01a0ec7f`, which wrote `results.md`). Claude read every changed source, test, doc and cookbook file against `ops/rebuild-2026-09-29/briefs/1.4.md`.

**Verdict.** No production defect was found. Every open item in `results.md` §7 that belongs to Task 1.4 is closed. Task 1.5 keeps its own items: the adapter, `rewards.py`, the codec, the device tables, and the CUDA fence with its pod DMA test.

## Source review against the brief

- **ABI.** The class and the four functions match the shared ABI: keyword-only 35 outputs, `seed_state` as nested tuples, and error classes `ValueError`/`OverflowError`/`RuntimeError`. The stub in `python/owl/rs.pyi` matches.
- **Transaction order** (`env.rs`). The order is:
  1. preflight, with validation of every output shape, layout, dtype and byte-range overlap in `bindings.rs`;
  2. raw token and padding admission before seed reservation;
  3. horizon prediction, widened to i128, then reservation of only the terminal envs;
  4. decode of every seat before any engine step;
  5. candidate clones on the owned pool, with `catch_unwind` in each worker;
  6. an indexed join, returning the first error in env order;
  7. a return dict prebuilt under the GIL;
  8. commit.

  In commit, `ObsStaging::publish`'s only failure precedes its first copy, and the transition copies and `mem::swap`s are infallible. Truncate uses `commit_selected_rows`, which destructures `ObsRowMut` exhaustively without `..`.
- **Terminal records.** They are replaced on every successful step, cleared only for selected envs on truncate and reset, and preserved on failure.
- **Rewards** (`reward.rs`). The formula, `terminal_scale`, the two f32 roundings and the binary64 admission predicate match the brief. Disabled components short-circuit, and a positive W times infinity saturates at the cap.
- **Cast audit** (`admission.rs`). The two HIRE casts are checked after the candidate step, from the executed-hire delta. The check uses the observed `hires_today` index. If the day counter resets inside a step, the check overestimates the Fibonacci index; that is conservative, because it can only reject extreme multipliers more often, never admit an unsafe cast.
- **Accepted deviations** (recorded in `results.md` §6): the existing `grammar_kernel_tests.rs` stands in for `kernel_integration_tests.rs`, which 1.3 already made unnecessary; `mod.rs` is the module root; `UV_NO_SYNC=1` is used for Codex's checks.

## Changes made in this review

1. **New L6 oracle, native half.** `tests/kaggriculture/test_native_env.py::test_calls_overwrite_every_output_byte_of_the_same_buffer_set` covers `observe`, `reset`, `step` and a terminal step.
   - Two identical runs start from different poison bytes; afterwards all 35 outputs, including padding, must be equal and the buffer addresses unchanged.
   - This is the native side of plan Task 1.4's L6 item under the R1 design: one buffer set, rewritten in place and completely. The guarantee that step-t bytes survive until their device copies finish is Task 1.5's entry fence and pod DMA test, not this test.
2. **Recorded fixture.** `tests/fixtures/kaggriculture_env_reference_v1.{npz,json}` was recorded on the pod (see below). The missing-fixture failures are gone.
3. **Contract v4.2.** It records the Q1 constructor refinement approved in the brief review. `docs/rl-api-specs.md` and `docs/rules-parity-coverage.md` now replace the PENDING text with results.
4. **Cookbook.** The native buffer Reference, the evaluation Reference's seed-band sentence, the index and a prepended log entry were updated.

## Non-vacuity: one mutation per oracle, each restored byte-exact

| Oracle | Mutation | Result | Receipt |
|---|---|---|---|
| 16-game trajectory vs reference TrainingBatch | `fill(row.done)` → `fill(!row.done)` | fails: `first divergence game=0 seed=17000 step=0 seat=0 … field=dones(0,) actual=True expected=False`; restored sha256 `671b53a3…`, replay passes | `pod-oracle/mutation-done*.log` |
| L3 seed partition (world sizes 2 and 8) | `SeedStream::reserve` advances by 1, not `seed_stride` | both world sizes fail (`[17000, 17001] != [17000, 17002]` / `… != [17000, 17008]`) | `mutation-l3-stride.log` |
| L6 buffer overwrite (new test) | drop the `transition_econ_after` copy from transition publication | all 4 cases fail: `transition_econ_after kept poison bytes` | `mutation-l6-overwrite.log` |
| reset/truncate rollback | clear selected live terminal records before reset staging succeeds | `reset_and_truncate_failures_preserve_every_published_byte` fails on `terminals` (ResetConstruct, mask `[true, true]`) | `mutation-rollback-terminal-clear.log` |
| release overflow proof | disable the engine overflow-check override via `--config` | `release_dependency_overflow_is_caught` fails | `pod-oracle/release-overflow-override-off.log` |

The source hash after each restore is `671b53a396567825ff21a74b9a33ae42a0496edaec5b83e6d165301a36e55db8`, the committed hash. After the restores, the extension was rebuilt and the targeted Rust test plus the native and grammar suites passed (387): `restored-green.log`.

Codex's mutations stay valid: the old rank offset in the test helper, whole-batch truncate publication, and premature live seed writes in step and reset (`results.md` §2).

## Pod run (`run-statement-pod-oracle.md`, pod `w7ia3zvxqsvs3g`, CPU only)

- **Setup.** A separate clone `/workspace/kg-v3-env-oracle` at `9dc2d02`, with its own `.venv`. `/workspace/kg-v3-rebuild` and `/workspace/kg-v3` were not modified. The GPUs were at 0 MiB before and after.
- **Deviation.** The reference `engine_rs/Cargo.lock` crates were missing from the pod cache, so the first recording failed offline (`autocfg`). An explicit `cargo fetch --locked` (lockfile-checksum verified) was then run and recorded in `cargo-fetch-reference.log`. Nothing was fetched silently.
- **Recording.** 16 games in 32.3 s, with a sampled peak RSS of 1.21 GB. Every game had positive starvation, drought, ineffective, hires, animal-placement and sales coverage. The npz is 310,365 bytes (sha256 `494bbf2c80af9adba66cbcfbccdfd7c638c7cc3bb74e438517dad3d39bb5c976`); the manifest sha256 is `aa6cc641484b3b348b24753785fc9082dc88dcadfb5fcf32b0901f3b26e728b7`.
- **Replay and custody.** 37 passed (pod, 20.4 s). On the Mac: 37 passed in 14.1 s at 411 MB, within the tiny-check budget (`mac-replay-green.log`).
- **Release overflow proof.** Passes with the override, fails without it.
- **Release timing.** Fat LTO, 20 warmups, 200 samples, one env, one thread; component timing only. `timing-{enabled,disabled}.json` hold the measurements.

  | Phase (median) | Dense 241 actors | Early state |
  |---|---|---|
  | outer clone | 159.5 µs | 9.5 µs |
  | kernel step, including its inner clone | 270.3 µs | 27.1 µs |
  | prepare snapshot | 171.7 µs | 3.8 µs |
  | validate | 31.6 µs | 1.0 µs |
  | write | 28.7 µs | 8.1 µs |
  | composed | 747.6 µs | 54.8 µs |

  With overflow checks off, the dense composed median is 752.0 µs, so the policy costs nothing measurable. The outer clone is 21% of the dense composed work. That does not dominate, so the brief's reopen condition for cheaper staging is not met.

## Checks (Mac, this worktree)

- `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` exits 0. Rust (274 passed, 5 ignored; engine 69), build, trim, docs and mypy all pass, and Python reports **2,042 passed, 7 skipped**. The peak RSS of 1.95 GB was measured, not capped (`just-prepare.log`). After the doc edits, `just docs-lint docs-fresh` and the cookbook lint were rerun (see the commit).
- Skips that remain for Task 1.5: `test_native_tables_match_expected_tables` needs `native_grammar_tables(device)`, and `test_run_ppo.py:3044` needs the adapter.

## Remaining gaps (not Task 1.4)

- **Task 1.5:** the adapter, `rewards.py` and its oracle, the Python codec, the device tables, and the pinned CUDA entry fence with its pod DMA test.
- **Later tasks:** complete-update throughput at 128 envs per rank, the choice of `native_threads`, and trainer integration (3.1/3.2), which are measured elsewhere.
