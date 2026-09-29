# Task 1.4 handoff

Native implementation is present; qualification is **not all green**. Root Rust,
retained engine, native bindings, grammar, rollback and world-size-8 seed checks
pass. The full reference fixture/replay/mutation, release overflow proof,
optimized costs and broad Python preparation remain **PENDING (pod)** under the
authorized Mac resource limits. No partial fixture or debug timing substitutes
for acceptance evidence.

Base: `e197528820ab7cfb429e21259000957370abf1c6`, branch `kg/rebuild-env`.
The operator committed partial work as `8d98ea8` before resumption. Codex did
not stage, commit, branch, merge, push or write another worktree. The inventory
covers the entire change from the base, including that operator commit.
Planned expectations are separate in `run-statement.md` and the approved brief.

## 1. Changed files

`changed-files.tsv` gives **one line per changed/added file and its purpose**,
including every operational receipt. Principal implementation inventory:

- `Cargo.toml`: root engine-package release overflow policy.
- `src/kaggriculture/mod.rs`: class/four-function registration, retaining the header encoder.
- `src/kaggriculture/env.rs`: checked seed streams, staged lifecycle, owned pool, ordered errors, terminal records and selected-row publication.
- `src/kaggriculture/reward.rs`: exact seven-field admission and reference reward rounding.
- `src/kaggriculture/admission.rs`: two audit-driven executed-HIRE cast guards.
- `src/kaggriculture/bindings.rs`: exact PyO3 ABI, fallible NumPy admission, GIL detachment and grammar bindings.
- `src/kaggriculture/env_tests.rs`: rewards, seeds, real-core rollback/retry, terminal/truncate and overflow regressions.
- `src/kaggriculture/lifecycle_timing_tests.rs`: early/dense fixtures and ignored release phase measurements.
- `python/owl/rs.pyi`: matching native signatures while retaining existing APIs.
- `tests/kaggriculture/test_native_env.py`: 340 binding/lifecycle cases, world sizes 2/8 and one/two-thread equivalence.
- `tests/kaggriculture/test_native_grammar_bindings.py`: 43 codec/table cases using the existing frozen grammar corpus.
- `tests/kaggriculture/test_env_reference.py`: strict full16 replay, first-divergence diagnostics and independent math oracle.
- `tests/tools/test_check_engine_trim.py`: strengthens the existing grammar-retirement regression.
- `scripts/record_kaggriculture_env_reference.py`: supervised actual TrainingBatch recording and deterministic source/fixture custody.
- `scripts/kaggriculture_env_reference_policy.py`: exact fixed observation-local policy recipe.
- `tests/tools/test_record_kaggriculture_env_reference.py`: 36 custody/coverage/size/publication tests, including the required full fixture.
- `docs/rl-api-specs.md`: exact ABI/lifetimes/seeds/rewards/truncate and Task 1.5 handoff.
- `docs/rules-parity-coverage.md`: actual native coverage and gaps.
- `cookbook/references/native-game-semantics-use-v3-owned-buffers.md`: existing concept revised with current inventory/evidence/limits.
- `cookbook/references/index.md`: current concept retrieval description.
- `cookbook/log.md`: prepended adaptation entry.
- `ops/rebuild-2026-09-29/1.4/reference_recorder.rs`: actual reference example added only to the temporary archive.
- `ops/rebuild-2026-09-29/1.4/bounded.py`: process-group watchdog and receipts; Linux path added for the pod but not runtime-qualified here.
- `ops/rebuild-2026-09-29/1.4/mutate_rollback.py`: premature live seed-write controls with restoration.
- `ops/rebuild-2026-09-29/1.4/mutate_selected_rows.py`: whole-batch truncate publication control with restoration.
- `ops/rebuild-2026-09-29/1.4/mutate_reference_done.py`: pending full-reference native done-bit mutation/rebuild/restore harness.

No vendored engine bytes, lockfiles, contract, Python adapter, rewards.py, Python
codec, device grammar bridge, model, PPO or trainer implementation changed.

## 2. Final commands and test-first evidence

Shell environment for checks (the OMP exception below was repeated correctly):

```sh
export CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true UV_NO_SYNC=1 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 TMPDIR=/Users/poonszesen/kg-v3-env/.codex-tmp RUST_TEST_THREADS=1
```

The paired `.json`/`.log` receipts contain exact argv, exit, elapsed time and
sampled group RSS; `final-checks.json` collects final commands and actual result
counts. The watchdog samples every 0.1 seconds at 115 seconds/960 MiB, so the
triggering sample can exceed the cap. No `DOCS_CURRENT=1` override was used.

| Command | Actual result | Receipt stem |
|---|---|---|
| `cargo test --offline --locked --lib -- --test-threads=1` | 274 passed, 0 failed, 5 ignored; exit 0 | final-root |
| `cargo test --offline --locked --manifest-path engine_rs/Cargo.toml` | 69 passed, 0 failed, 0 ignored (41+9+19; zero doctests); exit 0 | final-engine |
| `uv run --offline python scripts/check_engine_trim.py` | manifest OK; exit 0 | final-trim |
| `uv run --offline maturin develop --locked` | dev-profile extension installed; exit 0 | final-maturin |
| `uv run --offline pytest tests/kaggriculture/test_native_env.py tests/kaggriculture/test_native_grammar_bindings.py tests/kaggriculture/test_env_reference.py tests/tools/test_record_kaggriculture_env_reference.py tests/tools/test_check_engine_trim.py -q` | 497 passed, 2 failed, 0 skipped; exit 1 | final-targeted-python |
| `uvx --offline --from rust-just just rs-prepare` | formatting, Clippy, trim/docs-fresh pass; root274/5ignored, engine69; exit 0 | final-rs-prepare |
| `uvx --offline --from rust-just just py-prepare` | format/Ruff/syntax/mypy65 files pass; pytest memory stop, child -9/runner137 | final-py-prepare |
| `uvx --offline --from rust-just just prepare` (first) | exit1: coverage-doc double blank MD012/MD022; repaired | final-prepare |
| `uvx --offline --from rust-just just prepare` (repaired) | build/trim/format/lint/docs/mypy and root274/5ignored+engine69 pass; pytest memory stop, child -9/runner137 | final-prepare-restored |
| `uvx --offline --from rust-just just docs-lint docs-fresh` | exit 0 after final mapped-document edits | final-docs |
| `git diff --check` | exit 0, no whitespace errors | final-diff |

Targeted failures are exactly
`test_native_matches_training_batch_16_complete_games` and
`test_frozen_fixture_custody_and_complete_coverage`, both with:

```text
ValueError: Task 1.4 complete 16-game reference fixture is missing: /Users/poonszesen/kg-v3-env/tests/fixtures/kaggriculture_env_reference_v1.npz; run the approved recorder on the pod if the Mac watchdog cannot fit it
```

Both broad pytest stages collected 2,045 tests and stopped during the first
pre-existing `test_base_generics_typing.py` in-process mypy probe, before any
completion summary. `py-prepare`: 15.692802s/1,015,529,472 bytes; repaired
`prepare`: 56.573671s/1,017,036,800 bytes. Exact cause: watchdog
`stop_reason: "memory"`, not an assertion result. No broad Python counts are
invented and the existing test was not weakened. Both commands are PENDING (pod).

Final `uvx --offline --from rust-just just docs-lint docs-fresh` passes (exit 0,
1.049s); direct cookbook shape/source lint passes, all 47 repository sources
exist, producer hashes match, and protected paths have no diff from the base.
Receipts: `final-docs.*` and `source-custody-final.json`. The final
`git diff --check` result is recorded in `final-diff.*` and the final-check
summary after document edits.

| Task | Red / satisfied prerequisite | Green / limitation |
|---|---|---|
| A | Already retired by1.3; no missing-behavior red invented. Absent old engine file/manifest exception and actual root imports verified. | 3 trim tests/76 deselected; 9 root kernel tests/249 filtered; checker exit0. |
| B | `b-red-valid`: missing reward/seed helpers, after correcting first malformed ambiguous float type. | `b-green`: 5 passed. Independent telescoping/rounding-budget correction rejects zero-output mutation. |
| C | Missing class:246 failed/19 deselected; valid controlled-latch red for missing detachment; wrong reward-mode exception red. | 265 binding cases; latch passes after detachment. |
| D | Missing step:2 Python failures; real-core batch/reset-hook reds; missing admission helper red. | Core9 passed, admission6 passed, bindings328 passed; release proof pending. |
| E | C/D prerequisites already implemented terminal/reset paths; valid new Rust12 initially green. Malformed initial Rust binding/Python frames are not behavior reds. Negative controls below discriminate. | Restored Rust12; 11 new Python pass; native339, then340 after pool equivalence. |
| F | 42 absent-function failures. | 42 then43 passed; all964 table bits,321 accepted programs and43 rejection cases. |
| G | Absent recorder collection error and replay assertion; later two missing export-guard helper failures. Fixture-dependent tests remain loud failures. | 35 synthetic recorder tests pass; fixture custody/replay pending. Export guard2 pass; debug timing-fixture1 pass. |

`progress.md`, `oracle-receipt.md` and per-command receipts preserve detailed
red/green timings and distinguish malformed test/style failures from behavior.

Mutation results (all performed mutations restored):

- Full16 reference reward/done perturbation: **UNPERFORMED/PENDING (pod)**,
  because no fixture exists. The supplied `mutate_reference_done.py` requires
  the full fixture, mutates native done publication, rebuilds, requires a
  first-divergence failure, restores source/extension, then runs restored replay.
- Old L3 formula: native-constructor test helper changed to
  `(base+rank*n_envs, 1)`; **world2 and world8 both failed** with
  `rank 1 collides with rank 0`. Byte-exact restore, then native339 passed.
  Each rank consumes67 seeds across construction/full/partial/terminal resets;
  ranks run sequentially at E2. Exhaustion cases also cover both worlds.
- Replacing truncate step8b with `ObsStaging::publish` made
  `truncate_commits_only_selected_rows` **fail** with
  `unselected observation bytes changed: env=0, terminal=false`. Byte-exact
  restore followed by Rust12 passes.
- A live seed-counter write before `prepare_step` commit made
  `batch_failure_preserves_every_published_byte` **fail** at captured seed-state
  inequality (cargo101); matching restored SHA-256, then test passes.
- A live seed-counter write before `prepare_reset` commit made
  `reset_and_truncate_failures_preserve_every_published_byte` **fail** at captured
  seed-state inequality (cargo101); matching restored hash, then test passes.
  These mutate live seed state, not destination bytes. Tests compare all35
  destination bytes, snapshots/fresh observations, seeds and terminal records,
  then retry against untouched controls.
- Extra zero-reward mutation: initial actual-error-derived budget wrongly
  passed. Replaced it with independent endpoint/output-ULP budget; the corrected
  test **failed** at
  `(actual_sum[s] - endpoint).abs() <= output_budget[s] + budget64`, then passed
  after restoring `Ok(r)`. The pause left this temporary mutation in the partial
  commit; resumption restored it immediately. No production mutation remains.

## 3. Oracle results

**0 games / 0 transitions compared**. First divergence unavailable. Positive
per-game starvation/drought/ineffective counters, executed hires, GOOSE placement
and sales are implemented mandatory assertions, not observed claims. No fixture
was published: compressed/expanded sizes versus8MiB/256MiB and fixture/manifest
SHA-256s are unavailable.

The sole recording attempt stopped during exported-reference debug-engine
compilation before the example compiled or any game ran: inner watchdog
11.296769s/1,012,252,672 bytes, worker -9; outer exit1 at11.442614s. The inner
supervisor includes itself and worker/compiler descendants in its memory
accounting; the outer receipt alone excludes that separately supervised group.
Failed-attempt producer hashes are frozen in
`reference-attempt-source-custody.json`; later export-drift guard hashes are
separate in `reference-pod-source-custody.json` and unattempted.

PENDING (pod), exact commands with the task exports and a run statement:

```sh
uv run --offline python scripts/record_kaggriculture_env_reference.py --reference 65f0eac5bb00b18a9d3acce319c2a231cbd5dff0 --games 16 --first-seed 17000 --max-live-envs 1 --max-seconds 115 --max-rss-mib 960 --output tests/fixtures/kaggriculture_env_reference_v1.npz
uv run --offline pytest tests/kaggriculture/test_env_reference.py tests/tools/test_record_kaggriculture_env_reference.py -q
python3 ops/rebuild-2026-09-29/1.4/mutate_reference_done.py
```

Linux permits larger explicit positive budgets; Mac caps stay fixed. Coverage
and game count cannot be reduced. No partial fixture, fake expected data or
missing-fixture skip. Reference branch/worktree unchanged; only temporary
archive export receives the example.

## 4. Cast audit

`cast-audit.md`:99 numeric casts (78kernel including8test,10attribution,11RNG),
seven saturating subtractions and seven MT wrapping operations enumerated.
**Two reachable-unbounded casts**: HIRE cash at kernel `lib.rs:4018` and
`econ_attrib.rs:179`. Exact BigInt Fibonacci cost admission guards actually
executed hires, requiring float cost below `2^64`. No arbitrary config caps.
Bank finiteness and u64-counter-to-i64 outputs stay checked before publication.
Root dependency overflow policy covers integer arithmetic; release proof pending.

## 5. Timing

`timing.json`: release, fat LTO, one codegen unit, CARGO_BUILD_JOBS=2. Timing
build stopped at7.839793s/1,018,937,344 bytes before its test body; phase costs
are null. Separate release-overflow build stopped at28.872657s/1,007,714,304
bytes before its test body. Neither was retried on Mac. Debug fixture validation
does not supply release costs. PENDING (pod):

```sh
cargo test --release --offline --locked --lib release_dependency_overflow_is_caught
KG_OVERFLOW_CHECKS_LABEL=enabled cargo test --release --offline --locked --lib measure_lifecycle -- --ignored --nocapture --test-threads=1
KG_OVERFLOW_CHECKS_LABEL=disabled cargo --config 'profile.release.package.kaggriculture-engine.overflow-checks=false' test --release --offline --locked --lib measure_lifecycle -- --ignored --nocapture --test-threads=1
```

Same early/241-actor headers, source/header hashes,20 warmups/200samples; costs
separate outer clone, kernel step including inner clone, snapshot, validation,
write and composed work. No speed assertion or kernel redesign is justified.

## 6. Adaptations and deviations

- All three base refinements are true: existing `mod.rs` root/registration,
  already-retired bridge, and actual merged staging interfaces. Existing
  `grammar_kernel_tests.rs` satisfies the brief's integration-test role.
- Used real `ObsStaging::{new,buffers_mut,publish}`, validated serial/parallel
  per-env iterators, `ObservationGame` and `write_env`; no1.3 API changes.
- C/D necessarily installed reset/terminal staging before E; E's valid new tests
  initially passed. Mutation evidence replaces any fictitious initial red.
- `UV_NO_SYNC=1` prevents uv implicit editable release builds discovered in C;
  explicit dev maturin builds are used. One file-only export-guard green had
  OMP_NUM_THREADS=1 and was repeated with prescribed2.
- Required unchanged just recipes internally omit `--locked` on some root
  cargo/maturin calls; direct build/test commands use it. All run offline and
  both lockfiles remain unchanged from the base.
- Final coverage-doc double blank was repaired; complete prepare then hit the
  existing typing test memory demand. No suite weakening or approval bypass.
- Export-drift guard added after sole failed recording attempt; failed-attempt
  and subsequent source custody separated, no second recording attempt.
- Budgeted oracle/release/preparation handoffs are explicit gaps. No dependency
  addition, kernel edit, online fetch, training or GPU run was substituted.

## 7. Open items for Claude

- Record all16 games, compile/run actual reference example, verify coverage/sizes/hashes, replay and perform/restore native done-bit mutation on pod.
- Run release overflow proof and paired phase costs on pod; replace pending timing with actuals.
- Complete `uvx --offline --from rust-just just py-prepare` and `uvx --offline --from rust-just just prepare` where the full-suite typing probe fits.
- Add approved Q1 constructor refinement to contract Review and changes; contract was deliberately not edited here.
- Reconcile sibling evaluation/truncation Reference's native `<2**62` wording with approved checked-i64 ABI; narrower bands belong to factory policy.
- Task1.5 owns Python adapter/reward config-oracle/codec/device tables and pinned CUDA entry fence; none was implemented here.

Proposed Q1 wording for Claude:

> Native construction is `KaggricultureEnv(n_envs, seed, seed_stride, config,
> reward_config, native_threads, *, hire_limit)`, with validated JSON config and
> an exact-key reward dictionary. Python owns `pin_memory`, persistent pinned
> buffers and their reuse fence. `observe` consumes no reset seed. `step`,
> `reset` and `observe` accept all six transition outputs; `truncate_envs`
> preserves them and writes only selected observation rows.
