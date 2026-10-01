# Rules Parity Coverage

This document is the system of record for Orbit Wars and Kaggriculture parity
coverage. Keep it updated whenever the Python reference, fixture generators,
or Rust rules engines change. Start with the
[Kaggriculture summary](#kaggriculture-coverage-summary-task-75) for its tested
layers and open gaps. The Orbit Wars coverage below remains unchanged.

## Covered By Replay Fixtures

Replay fixtures are generated with `scripts/download_replays.py` and loaded from
`tests/fixtures/orbit_wars_replays/replay-*.jsonl`.

The parity tests fail by default when required fixtures are missing. Set
`REQUIRE_PARITY_FIXTURES=0` to skip replay parity, and to skip generation parity
when generation fixtures are missing.

Replay coverage is required for:

- `75930761`: 2 players, 103 recorded transitions.
- `75926553`: 4 players, 222 recorded transitions.

The replay parity test checks each transition against the Python reference for:

- step counter
- per-player Kaggle status/reward, mapped to active, won, or lost. The replay
  check requires exact status parity once Kaggle marks every player `DONE`.
  Before that global terminal row, Kaggle can keep eliminated players `ACTIVE`
  while Rust intentionally reports them as `Lost`, so the test only verifies
  that Rust has not ended the game early.
- angular velocity
- planets: id, owner, position, radius, ships, production
- initial planets
- fleets: id, owner, position, angle, source planet, ships
- next fleet id
- comet planet ids
- comet groups: planet ids, full paths, path index

Auxiliary Rust `StepResult` counters for fleets and ships lost in the sun, out
of bounds, or combat resolution, and for planets and comet planets captured,
plus neutral capture/undershot counters, are not fields in the Kaggle rows.
They are covered by focused Rust unit tests rather than replay parity
assertions.

The current local replay set covers both 2-player and 4-player games, launches,
production, fleet movement, collisions, captures, and comet movement and expiry.
Step-limit termination is covered by focused Rust unit tests.

## Covered By Generation Fixtures

Generation fixtures are produced by `scripts/generate_reference_fixtures.py`
from the installed `kaggle-environments` Orbit Wars implementation and written
to `tests/fixtures/generation/reference_generation.json`. Rust consumes the
recorded random call stream and compares generated output.

The generated fixture currently covers:

- planet generation from seed `42`
- current Python-reference random static and fill phases, including the
  reference fourfold symmetry ordering
- full reset for 2-player and 4-player games, including angular velocity,
  planet generation, initial planets, and current random-group home assignment
- comet paths at spawn steps `50`, `150`, `250`, `350`, and `450`
- comet path generation with existing comet ids excluded
- comet path generation where failed attempts occur before success
- comet ship sampling
- step-limit no-op terminal tie status/reward mapping, where all tied players
  win

## Covered By Unit Tests

Rust unit tests cover focused rules behavior that is hard to isolate from full
replays:

- geometry helpers
- Python-compatible inclusive `RandomSource::uniform` behavior, including equal
  bounds
- RL discrete-target launch-angle selection for static target cones, dynamic
  target-hit windows, sun avoidance, static and dynamic blocker arc
  subtraction, strict static arc caching, full-mask masking for fully blocked
  static targets, and no-op fallback accounting
- fleet speed curve
- launch validation and side effects
- production order
- fleet movement and removal
- fleet collision priority: swept planet/comet collisions before out-of-bounds
  or sun removal
- sun, planet, out-of-bounds, and simultaneous swept-pair collisions
- combat resolution, ties, and reinforcement
- neutral non-comet planet and comet undershot counters
- comet spawning before same-step movement
- comet movement and expiry
- comet planet/path alignment when only part of a comet group expires
- terminal score ties where all tied players win
- invalid owner and planet-ID invariant rejection
- immediate nonterminal elimination and result cardinality matching the actual
  player count
- replay accepted-action filtering with exact planet-ID matching and no numeric
  ID truncation
- replay skip-spawn injection that keeps step parity isolated from RNG-backed
  comet generation

## Known Boundaries

Replay parity intentionally injects comet paths and comet ships from the
expected observation. It also explicitly injects "skip spawn" on comet spawn
steps where the replay has no new comet, so replay parity never falls through to
RNG-backed comet generation. This isolates step parity from random generation.
Comet generation parity is covered separately by generation fixtures.

The Rust simulator receives typed actions and fails fast on invalid actions.
Kaggle/Python action parsing is outside the inner simulator API. The replay
harness mirrors Python's accepted-action filtering only to turn historical
Kaggle replay actions into typed Rust actions.

Floating-point parity uses close comparisons rather than bit-for-bit equality.
Discrete ids, owners, ship counts, production, and removals match exactly. Final
terminal statuses match exactly; on nonterminal rows the harness permits Rust's
intentional early `Lost` status while requiring the game remain nonterminal.

The Rust state stores planets and initial planets in ID-indexed slots. Parity
comparison iterates live slots in ID order, which matches generated and fixture
planet ordering because fixture IDs are unique and contiguous for live planets.
Manual duplicate planet IDs and IDs at/above `MAX_PLANET_ID` are intentionally
rejected before parity comparison.

Replay and generation fixtures are ignored by Git because full episodes and
recorded reference streams can be large. A fresh checkout must run
`scripts/regenerate_test_fixtures.sh` or restore fixtures from cache before
running required parity. Use `REQUIRE_PARITY_FIXTURES=0 just rs-test` only when
intentionally skipping replay parity or missing generation fixtures.

RL-only terminal metrics such as `ships_lost_in_combat_per_game`,
`fleets_lost_in_combat_per_game`, `neutral_planet_undershot_rate`, and
`neutral_comet_undershot_rate` are derived from simulator step results and are
covered by focused Rust/Python metric tests, not by replay fixture parity
assertions.

## Kaggriculture Coverage Summary (Task 7.5)

This map describes integration tip `994818b87041426c6fc442a85fb06932937d58a7`
(Tasks 1.4/1.5 and 7.1 merged) as merged into `kg/rebuild-7-5` on 2026-09-30.
It was first written from `bde337465a9fa7c07bedded88d5d696d7cefb7ef`, before
Task 7.1 landed. The sections below retain their historical check counts.
The current checks for this documentation change are listed separately here.
Claim-by-claim sources are in `ops/rebuild-2026-09-29/7.5/claims.md`; the
Task 7.1 refresh is in `ops/rebuild-2026-09-29/7.5/r2-fixes/results.md`.
Task 7.5 landed on the integration through staging branch `kg/merge-7-5-c`,
onto integration tip `bd1c9279ec09a27a5c05069197fef404263cb55b` (Task 4.4, the
Task 3.1 remainder with Task 3.5, and the W&B wiring). Those three landings
change no Rust, no `python/owl/kaggriculture` module and none of the test files
the rows below cite, so every tested-layer row still holds; the Task 3.1 gap is restated for the merged trainer, and
the landing's full `just prepare` is listed under current checks.

### Compatibility target

The target is `kaggle-environments==1.32.7`, pinned in `pyproject.toml` and
`uv.lock`. `engine_rs/Cargo.toml` metadata pins the interpreter file SHA-256 to
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.
The vendored reference commit is
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`.

`tests/scripts/test_kaggriculture_parity.py::test_project_environment_satisfies_the_engine_pin`
checks the installed package version and engine hash through
`scripts/kaggriculture_parity/generate_traces.py::load_pinned_kaggle`.
The generator refuses a different version or hash before writing traces.
The same test file checks version/hash rejection and live fixture regeneration.
`scripts/check_engine_trim.py` checks vendored source custody and generated
trace inventory, hashes and Cargo-pin consistency; it does not inspect the
installed Python engine. `tests/tools/test_check_engine_trim.py` tests that checker.

### What is tested

In the receipt column, `R/` means `ops/rebuild-2026-09-29/`.
Each oracle qualifies its own layer; their game and case counts are not additive.

| Layer | Oracle | Scope / denominator | Test entry point(s) | Receipt path | Detail |
| --- | --- | --- | --- | --- | --- |
| Rules kernel, official episodes | Four recorded official Python-engine traces; Rust resets from configuration and seed | Episodes 95324500, 95901360, 95921764, 95990191; 719 transitions each, 2,876 transitions and 2,880 snapshots | `engine_rs/tests/replay_parity.rs::episode_*` | `R/1.1/results.md`; `R/7.5/engine-tests.log` | [Kernel](#kaggriculture-test-surface) |
| Live differential parity, Task 1.1b | Kaggle's own hash-pinned 1.32.7 Python engine | Committed: 8 games / 3,960 transitions, plus 7 one-step divergence repros. Recorded local sweep: 40 games / 21,824 transitions agree; 303 probes / 1,515 transitions give 266 agreements and 37 D1/D2 divergences | `engine_rs/tests/replay_parity.rs::generated_fixtures_replay`, `env_directory_traces`; `tests/scripts/test_kaggriculture_parity.py` | `R/1.1b/results.md`; `R/1.1b/verify-r1/sweep-summary.json` | [Differential](#kaggriculture-live-differential-parity) |
| Retained unit and RNG tests | Hand-derived synthetic expectations and embedded CPython RNG vectors | 41 library tests + 9 RNG integration tests; no episode reads | `engine_rs/src/lib.rs` test module; `engine_rs/tests/py_random.rs` | `R/7.5/engine-tests.log` | [Test surface](#kaggriculture-test-surface) |
| Native grammar, Task 1.2 | Frozen outputs of three reference decoders, literal command supports, and direct kernel execution | 320 scheduled programs (256 synthetic + 64 replay); 44 controls (43 rejections + 1 padding acceptance); 964 table bits; 64 selected replay seat actions | `src/kaggriculture/grammar_tests.rs`; root `src/kaggriculture/grammar_kernel_tests.rs` | `R/1.2/results.md`; `R/1.2/oracle/README.md`; `R/1.3/r1-fixes/results.md` | [Grammar](#kaggriculture-native-grammar-task-12) |
| Observation encoder, Task 1.3 | Recorded `encode_invest` rows from the pinned reference Rust crate; tensor-only reconstruction | 512 states / 1,024 seat rows / 8,176 offsets per row, compared bitwise; actual Python observation-schema checks | `src/kaggriculture/oracle_corpus.rs::compare_observation_oracle`; `tests/kaggriculture/test_observe.py::test_every_frozen_oracle_record_passes_the_actual_schema` | `R/1.3/claude-review.md`; `R/1.3/r1-fixes/results.md` | [Observation](#kaggriculture-observation-coverage-task-13) |
| Native environment, Task 1.4 | Synthetic lifecycle expectations and untouched controls; recorded reference Rust `TrainingBatch`; independent Python reward formula | 35 destination buffers; reset/step/truncate rollback and seed admission; 16 games, seeds 17000–17015, 719 transitions each = 11,504 | `src/kaggriculture/env_tests.rs`, `admission.rs`; `tests/kaggriculture/test_native_env.py`; `test_env_reference.py::test_native_matches_training_batch_16_complete_games` | `R/1.4/claude-review/pod-oracle/`; `R/1.4/p3-rerecord/reference-recording-attempt.json` | [Lifecycle](#kaggriculture-native-lifecycle-coverage-task-14) |
| Python adapter / codec / table bridge, Task 1.5 | Real native binding plus lifecycle expectations, frozen grammar corpus, independent expected tables and reward arithmetic | Stable 35-buffer lifecycle; 321 accepted / 43 rejected codec records; 964 table bits; recorded reward trajectories plus 3 extreme-coefficient games of 96 transitions each | `tests/kaggriculture/test_env.py`, `test_codec.py`, `test_native_tables.py`, `test_game.py`, `test_rewards.py` | `R/stage2-adapter/native/results.md`; `R/stage2-adapter/tables/results.md` | [Native boundary](#kaggriculture-native-lifecycle-coverage-task-14) |
| Evaluation opponents, Task 7.1 | Original Starter, R04, EcoBot and E776 Python submissions on Kaggle 1.32.7's engine under CPython 3.11.15 | 8 default-config games, seeds 20260929–20260936, each bot in both seats twice: 4 bots × 2 seats × 1,438 = 11,504 compared actions and 5,752 transitions of state; 24 mid-episode reconstructions with 1,152 resumed actions | `opponents_rs/tests/oracle_parity.rs`; `opponents_rs/tests/lifecycle.rs`; `tests/scripts/test_kaggriculture_parity.py` | `R/7.1/review/results.md`; `R/7.1/verify-r1/parity-replay.json`; `R/codex/verify-7.1-r2.md`; `R/merge-7.1/prepare-on-5b43062.log` | [Opponents](#task-71-opponents-snapshot-view-and-original-submission-parity) |
| Cha22 opponent import (2026-09-30) | Original Cha22 submission (SHA-256 `127ed3e6…`, `ig_agent`) on Kaggle 1.32.7's engine under CPython 3.11.15 | 3 default-config games vs Starter, seeds 20260937–20260939, Cha22 in seat 0 twice and seat 1 once: 4,314 compared actions (2,157 Cha22) and 2,157 transitions of state | `opponents_rs/tests/oracle_parity.rs`; `opponents_rs/tests/lifecycle.rs`; `tests/tools/test_check_opponent_import.py`; `tests/scripts/test_kaggriculture_parity.py` | `ops/cha22-opponent-import-2026-09-30/` | [Cha22](#cha22-opponent-import-light-original-submission-parity) |

The replay comparator checks public/private values and recursive object key
order, statuses, typed rewards, step/done and terminal banks. Rejected actions
must leave the checked state unchanged. Negative controls in
`engine_rs/tests/replay_parity.rs` perturb a committed trace's state value,
private key order, action, rejection label, per-transition rewards, header
transition count (caught by the done check) and header terminal banks; each must
be rejected with its divergence kind. The last three were added after a scratch
review disabled those checks without any test failing
(`R/7.5/r2-fixes/replay-negative-controls.log`). The sweep receipt is historical;
Task 7.5 does not launch another sweep.

Observation reconstruction is observation-information coverage, not rules parity.
For Task 1.4, `scripts/record_kaggriculture_env_reference.py` exports the reference
commit named above and runs `training::TrainingBatch` through its Rust recorder.
The fixture pair is `tests/fixtures/kaggriculture_env_reference_v1.json` and
`tests/fixtures/kaggriculture_env_reference_v1.npz`.
Its replay compares rewards, dones, before/after banks and economic counters
bitwise, then checks terminal records, metrics, seed progression and autoreset.
The separate Python reward formula allows one f32 ULP. This is a native-wrapper
comparison to the historical Rust training environment, not a second Python
rules-engine differential check.

### Local evidence that is not parity

The BC pairing diagnostic exists on `kg/rebuild-bc-now` at
`933d661097bed36859fd9a0d7505ac66d8659b7a`, outside this integration.
Its local evidence paths are
`ops/rebuild-2026-09-29/bc-a100-2026-09-29/pairing.json` and `receipts.md`
on that branch. The implementation is
`scripts/kaggriculture_prepare_bc.py::pairing_check` and `PAIRING_PUBLIC_KEYS`
at that commit. The receipt says the pairing result was copied, not rerun there.
The receipt at this tip is `ops/rebuild-2026-09-29/7.5/bc-audit.txt`. It records
the `pairing.json` SHA-256, the aggregate sums and the read-only branch check, so
the figures below stay checkable if that branch is rebased or deleted.

Kaggle's own 1.32.7 Python interpreter re-steps each archived `steps[t]`
observation with the recorded `steps[t+1]` actions and episode configuration.
It compares values of `day`, `hour`, `farms`, `market`, `town` and both seats'
`private`. It excludes `step`, key order, statuses and rewards.
Across 8 sampled episodes, 5,743/5,752 transitions match. All 9 mismatches are
private-only, at turns 335, 383, 431, 455, 527 and 623; their cause is unattributed.
The receipt calls them day ends, but the sampled episode configurations are not
in the tracked evidence, so this page does not confirm that classification.
No Rust engine participates. This is not engine parity or proof of label correctness.

### What is not tested

- D1/D2 agreement: Unicode-decimal quantities and unhashable fields remain known
  divergences. Their repros assert failures; they do not establish repaired parity.
- Exhaustive Python rules-path or malformed-input agreement. Model grammar actions
  cannot exercise D1/D2, and the generated policies/probes are a bounded sample.
- Kaggle framework behavior outside the interpreter: timeouts, agent errors and
  `INVALID` statuses.
- Strong-play worlds beyond the four official episodes, or a larger pod parity sweep.
- Direct equality of recorded RNG/shop schedule headers; replay tests check their
  effects through state instead.
- Full-season codec parity on the official action streams, or every actor/order/HIRE combination;
  selected replay actions and local support classes are the tested scope.
- Whole-observation or whole-snapshot Python parity from the Task 1.4 fixture;
  its reference is Rust and its comparisons target the native transition boundary.
- Complete historical observation-corpus source custody: three engine input hashes
  were omitted; generation-time dirty bytes and the full producer module inventory are absent.
- Task 7.1 beyond its tested scope: custom configurations, CPython 3.12 or
  later (R04 diverges there), states the controllers did not create, individual
  order rejection and engine-confirmed shortages, and playing strength. Since
  the fixed-opponent collection (`env.opponent_mix`, 2026-09-30), the native
  environment hosts an `opponents_rs` seat (`HostedSeat`) against the learned
  policy. Its nine learned-seat tests now run, and native replays against an
  independent kernel + controller reference check every bot's actions
  (`src/kaggriculture/opponent_env_tests.rs`, `opponents_rs/tests/hosted.rs`),
  Cha22 included since the cha22 anchor merge.
  Custom configurations remain unqualified there too.
- Cha22 beyond its three-game light corpus: mid-episode Python replay, Python
  hash seeds whose equal-price ADV ordering differs from tape order, the
  inactive PIPE opening alternatives, custom configurations (including the
  short-episode test configurations the native hosted tests use) and playing
  strength (v2's fuller checks are cited, not rerun).
- Task 7.3 replay export / Kaggle-episode round trip in this integration;
  it is unmerged, with no approving verdict in this tip's phase tracker.
- Task 7.4 Kaggriculture packaging; this tip records a brief under review, not implementation.
- Kaggriculture trainer qualification beyond CPU functional checks. Since
  `821b446` canonical `scripts/run_ppo.py` carries Kaggriculture through the
  shared rollout storage, mask/action mapping, evaluation and W&B logger, and
  Task 3.5's two-update tiny CPU run exercises it; that run carries no learning
  claim, and GPU, multi-rank and learning qualification wait on Phase 6. None of
  this is rules parity: the trainer consumes the native environment rows above.
- CUDA/BF16 native-adapter parity, hardware table upload and pinned-memory DMA
  reuse-fence qualification. Separate GPU model diagnostics do not qualify these paths.
- Complete-update throughput. Task 1.4 measured observation/lifecycle components;
  Task 1.3's dedicated timing diagnostic remains incomplete.

### Current checks

Checks for this refresh ran on 2026-09-30 on `kg/rebuild-7-5` after merging
integration tip `994818b` and adding the three replay negative controls. Logs,
the runner and resource figures are under `ops/rebuild-2026-09-29/7.5/r2-fixes/`
(`R2/` below). They ran on the owner's Mac, CPU only, with `CARGO_BUILD_JOBS=2`
and `OMP_NUM_THREADS=2`; each figure is `/usr/bin/time -l` maximum RSS of the
largest single child process, not a process-tree sum. The `opponents_rs` run
peaked at 3.0 GB, above the owner's 1 GB check limit; this run did not
attribute that peak to compilation or to a test binary. Between `994818b` and
this branch, only documentation, `ops/` receipts, the cookbook,
`engine_rs/tests/replay_parity.rs` and its `engine_rs/TRIM_MANIFEST.json` hash
changed. The root Rust suite and the full Python suite were therefore not rerun;
their latest full runs are the Task 7.1 merge receipt for `994818b`
(`merge-7.1/prepare-on-5b43062.log`).

| Command | Actual result | Receipt |
| --- | --- | --- |
| `cargo fmt --check` and `cargo clippy --all-targets --locked` (`engine_rs`, with the `justfile` allowances) | Exit 0 each | `R2/engine-fmt.log`; `R2/engine-clippy.log` |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | 72 passed (41 unit, 9 RNG, 22 replay parity), 0 failed; 171 MB | `R2/engine-tests.log` |
| Replay negative controls with each comparator check disabled in turn | Each mutation fails exactly its new test; the restored file passes 22 | `R2/replay-negative-controls.log` |
| `cargo test --offline --manifest-path opponents_rs/Cargo.toml --locked` | 22 passed (12 unit, 5 lifecycle, 5 oracle), 0 failed; 3.0 GB | `R2/opponents-tests.log` |
| `scripts/check_engine_trim.py`; `scripts/check_opponent_import.py` | Exit 0 each | `R2/engine-trim.log`; `R2/opponent-import.log` |
| `pytest -m "not slow"` on the parity, trim, opponent-import and opponent tests | 178 passed, 11 skipped (nine learned-seat opponent tests, one broad regeneration, one original-source reread); 509 MB | `R2/pytest-parity-custody.log` |
| `pytest -m "not slow"` on `test_env_reference.py`, `test_native_env.py`, `test_env.py` | 400 passed (includes the 16-game `TrainingBatch` comparison); 374 MB | `R2/pytest-env.log` |
| `scripts/check_doc_freshness.py` | Exit 0 | `R2/docs-fresh.log` |
| Root `cargo test` and the full Python suite | Not rerun; at `994818b` root 274 passed, 5 ignored; Python 2,400 passed, 21 skipped | `merge-7.1/prepare-on-5b43062.log` |
| Landing: full `just prepare` on the `kg/merge-7-5-c` merge onto `bd1c927` | Exit 0: engine 72 (41 unit, 9 RNG, 22 replay parity), root 274 passed, 5 ignored; opponents 22; Python 2,587 passed, 17 skipped; mypy and docs-fresh clean. `/usr/bin/time -l` peak 3.3 GB (largest single process, unattributed; above the 1 GB check limit) | `merge-7-5-c/prepare.log` |

The first version of this summary ran its checks at `bde3374` (receipts in
`ops/rebuild-2026-09-29/7.5/`): engine 69 passed, trim OK, and an unguarded
`just py-prepare` with 2,319 passed and 10 skipped. That run recorded neither its
environment nor its peak memory. Codex's guarded run covered a different
selection: a subset of the tests (the parity and trim test files and
`tests/kaggriculture`) with slow tests included, whereas `just py-prepare` runs
`pytest tests/ -m "not slow"`. That guarded run had already reached a sampled
process-tree peak of 989,744 KiB after 21.6 s and was stopped by its 960 MiB
limit before finishing (`7.5/pytest.json`), so the unguarded run may have
exceeded the 1 GB check limit. Mac-limited review
checks in this rebuild therefore use targeted shards, as above; this does not
change the `just py-prepare` / `just prepare` workflow.

## Kaggriculture Rules Kernel

### Task 7.1 Opponents: Snapshot View and Original-Submission Parity

`opponents_rs` is a standalone edition-2024 crate holding byte-exact Starter,
R04, EcoBot and E776 controller sources plus E776 policy data from reference
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. Its authored `Game` owns the
frozen engine behind an opaque sibling module, exposes current snapshot
accessors, and derives only Starter's hire-cost multiplier through serde. An
authored BigInt `fib` port supplies the imported helper. Full snapshots are
cloned after construction and successful steps; this evaluation path has no
Mac performance claim and is not the training hot path. `from_engine` requires
the caller to supply that engine's original configuration.

The view includes both private states, public state, statuses and rewards.
Rival-private perturbation checks therefore remain necessary. Engine RNG,
seed and hidden counters cannot be reached or perturbed through the view.
A test-first correction removed derived `Debug` from the opaque engine holder:
`Game` now formats only snapshot/config, so formatting cannot reveal engine
seed/counter fields. The regression failed before that correction and passes
in the final 12-test unit suite (`run2/closure.log`).
Tests clone controllers/views at seven steps for each bot and both seats,
change rival-private state and require unchanged actions; own money/seeds/shed
perturbations provide a positive control. The seat wrapper owns independent
controller state, resets explicitly and rejects repeated/skipped steps,
wrong-seat calls and use with another episode. Same-seed replay preserves
complete action hashes and raw banks; a different seed changes the sequence.

The match runner supports only default configuration and sends official
`farmer`/`hands`/`market` JSON through the engine without grammar truncation.
It records the applied action, joint engine acceptance per seat, controller
errors, raw banks and winner. Acceptance is not proof that every individual
order executed; engine market metrics are available only as joint aggregates.
The native crate's 22 tests pass: 12 unit tests (including Starter's five
pinned inline cases), five lifecycle/match integration tests and five oracle
tests (comparator regression, original-Python parity, per-seat tampering,
original-Python mid-episode replay and its tampering checks).

**Original-submission parity (Claude review).** The oracle corpus holds eight
default-config games generated from Kaggle 1.32.7's Python engine on CPython
3.11.15, seeds `20260929`–`20260936`, pairs rotating Starter–R04, R04–EcoBot,
EcoBot–E776 and E776–Starter so each bot plays both seats twice. Seat policies
are Kaggle's Starter and the hash-pinned original R04, EcoBot and E776
submissions at sibling commit `e8884aae82eddeb7a1aeae99ecceeca7c830d67e`, each
a fresh module per seat and game. Native controllers, fed native state rebuilt
from configuration and seed, match every recorded Python action:

| Bot | Seat 0 compared / matched | Seat 1 compared / matched |
| --- | --- | --- |
| Starter | 1,438 / 1,438 | 1,438 / 1,438 |
| R04 | 1,438 / 1,438 | 1,438 / 1,438 |
| EcoBot | 1,438 / 1,438 | 1,438 / 1,438 |
| E776 | 1,438 / 1,438 | 1,438 / 1,438 |

All 5,752 transitions also match public/private state (including key order),
statuses, rewards and terminal banks. Tampering either seat's first recorded
action fails each trace. Actions and state compare JSON numbers strictly;
rewards are typed f64, as in the existing kernel comparator.

**Runtime.** Codex's first oracle ran on CPython 3.12.13 and failed at R04 step
12 (`hands[2][0]`: native `WEST`, Python `NORTH`). `run2/r04-mismatch.md`
localized it to `sum()` over 25 weights of 0.35: CPython 3.12's compensated
float `sum()` gives `8.75`, sequential accumulation `8.749999999999996`,
moving an anchor threshold. The native port reproduces sequential summation,
which is CPython 3.11 behavior. Kaggle's simulation image
(`gcr.io/kaggle-images/python:v163`, read directly in v2's container Reference)
runs CPython 3.11.13, so the generator, custody checker and Rust test now
require a 3.11 oracle. The 3.12 trace, preserved at commit `7ae9bbf`, is a
negative control. If Kaggle's runtime moves to 3.12 or later, R04 parity must
be reopened.

Coverage of the Python traces, per seat-game (eight games, 16 seat-games):
openings 1 and day resets 29 in every seat-game; own-farm weeds in all;
hires in every R04, EcoBot and E776 seat-game and none for Starter; day-29
SELL orders in every R04, EcoBot and E776 seat-game and none for Starter.
No trace has a whole-step rejection or a BUY_PRODUCT above the inventory index.
`buy_quantity_above_inventory_index` compares a quantity to a price index, not
stock, so engine-confirmed shortages and individual-order rejection are
uncovered. The contiguous traces' `mid_episode_replay` count stays zero; replay
has its own oracle below. Explicit reset is checked natively (lifecycle test).

**Original-submission mid-episode replay (verify r1).** The generator's
`opponent-replay` preset rebuilds fresh controllers in both seats (a fresh
module per original submission) from each frozen oracle's recorded prefix:
every prefix observation is presented, the controller must choose the recorded
action, recorded actions drive Kaggle's engine and the rebuilt public/private
state must equal the trace. The controllers then act on their own for 24 steps.
Reconstruction points are step 37 (day 1 hour 13), step 360 (a day reset) and
step 695, whose window covers the whole final day. All eight oracles at all
three points give 24 cases, so every bot resumes in both seats at every point.
`opponents_rs/fixtures/replay/REPLAY.json.gz` (23,323 bytes, CPython 3.11.15,
bound to the oracle MANIFEST SHA-256) freezes the resumed actions and final
public/private state and statuses; regeneration was byte-identical. Native
controllers, rebuilt through the same step-zero lifecycle from the same prefix,
match **1,152 / 1,152** resumed actions (576 per seat) and all 24 final states.
Tampering the first resumed action of either seat in each step-37 case, the
last resumed action of one case and one final state each fails with the exact
case, step and seat. A restored scratch controller mutation at step 700 failed
all eight step-695 cases on resume (`verify-r1/replay-controller-mutation.log`).
Limit: each prefix is the controllers' own recorded play, so every resumed
action also equals the contiguous trace; states the controllers did not
create (foreign prefixes) are untested.

The eight compressed oracles use **1,779,187 / 4,000,000 bytes**; their
SHA-256s are in `opponents_rs/fixtures/oracle/MANIFEST.json` and
`opponents_rs/OPPONENT_MANIFEST.json`. E776's separately pinned executable
tape is policy data, not part of this oracle budget.

The dedicated opponent manifest/checker owns imported/authored file inventory,
source hashes, Python source provenance and oracle trace hashes/size/budget.
Its default mode, run by `just prepare`, needs no sibling repository: it pins
the original entry hashes structurally. `--original-sources` re-reads every
original file (owner's machine only); Claude's review split the modes because
the first version made `just prepare` require the owner's sibling tree.
EcoBot/E776 explicitly declare no software license and must not be redistributed;
engine licensing does not resolve that notice gap. No original Python submission
source is copied. Learned-seat integration tests were skipped until the
fixed-opponent collection (2026-09-30) gave `KaggricultureEnv` its
`opponents_rs` seat hook (`HostedSeat`, constructor keywords
`opponent_bot`/`opponent_envs`); they now run in
`tests/owl/kaggriculture/test_opponents.py`. Default-config
CPU qualification establishes neither custom-config support nor playing strength.

Run 1 at `21d0f45` stopped on private `fib`, private `Game.config` and missing
accessors under Claude's original placement prompt. Its compiler/source receipts
remain unchanged. The revised view resolves that placement boundary without
editing the frozen engine, restoring `policy_rows`, copying the reference
all-controller dispatcher or adding a root-crate dependency. The trim updater
accepts the committed run-1 manifest and preserves retained/authored entries.

### Cha22 Opponent Import: Light Original-Submission Parity

`opponents_rs` registers `cha22`, the full `ig_agent` port with its dependency
closure (V43, V47, V48, Farm2945, Metav4, Pipe16), taken from the same pinned
commit `65f0eac5…` as Task 7.1. 27 Rust files and the three embedded fixtures
are byte-exact; 8 files differ only in three Game-view accessor lines, which
`scripts/check_opponent_import.py` re-derives from the pinned blobs (its
`adapted` section). `Game::configuration()` supplies the serialized
configuration those controllers read. Upstream Apache-2.0 notices are under
`opponents_rs/notices/cha22/`.

Parity is deliberately light at the owner's request to accelerate the setup.
Three default-config games against Starter were generated from the original
submission on Kaggle 1.32.7 under CPython 3.11.15 (`--preset cha22`), with Cha22
in seat 0 twice and seat 1 once. Native controllers match **4,314 / 4,314**
recorded actions (2,157 Cha22, 2,157 Starter); all 2,157 transitions agree on
public/private state, statuses, rewards and terminal banks. Tampering Cha22's
step-399 market in each trace fails at that step and seat. Regenerating under
`PYTHONHASHSEED` 0, 12345 and random reproduced the committed bytes, so these
games never reached an equal-price ADV tie whose order depends on the hash seed.
The three traces use 564,323 bytes; with the 7.1 corpus the budget use is
2,343,510 / 4,000,000. A full native Cha22–Starter match (seed 17, both seats)
is deterministic and Cha22 wins it. v2's fuller import checks (5,752 actions,
237 direct cases, 64 clones) are cited from kaggriculture-v2
`ops/cha22-opponent-import-2026-09-24/`, not rerun. Details and limits are in
`ops/cha22-opponent-import-2026-09-30/results.md`.

### Kernel Inventory

Task 1.1 retains a standalone `engine_rs` package pinned to reference commit
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. Its compatibility target is
`kaggle-environments==1.32.7`, Python engine SHA-256
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.
`engine_rs/TRIM_MANIFEST.json` accounts for all 125 reference files: 12 retained,
113 excluded, plus exactly two authored files (the replay-parity test and the
generated-trace `MANIFEST.json`; Task 1.3 retired the grammar/kernel bridge to
root integration) and the non-engine change inventory.
`python scripts/check_engine_trim.py` checks hashes, exhaustive inventory,
declared original-line edits, exact Cargo removals and append-only provenance.
Independently of manifest declarations, only `lib.rs`, `Cargo.toml`,
`Cargo.lock` and `VENDORED_FROM.md` may differ from the reference; the lockfile
must equal the reference minus the six-package Rayon closure, and the Task 1.1
provenance appendix must follow the historical bytes with a pinned SHA-256.

The Task 1.1 trim removes reference `lib.rs` lines 19, 21–25 and 27: declarations for
`ffi`, `joint_matching`, `myolie_features`, `myolie_sampler`, `native_agents`,
`policy_rows` and `training`. That historical trimmed file was 185,626 bytes,
SHA-256 `c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd`.
The 2026-10-01 native SPS refinement adds a narrowly pinned transactional
`stepped_with_market_metrics(&self)` wrapper and delegates the existing mutating
method to it. The rules body is unchanged; exact current bytes and edits are in
`TRIM_MANIFEST.json`, with an independent expected replacement in the checker.
`ops/sps-2026-10-01/` records baseline trajectory parity and Mac step timings.
`py_random.rs`, `econ_attrib.rs` and the RNG integration tests retain exact
reference bytes, as do the Apache-2.0 license and four compressed fixtures.
Cargo removes the unused binary, cdylib target and Rayon dependency; its
generated lockfile prunes the unused closure. Historical provenance is retained
with an explicit trim note.

### Kaggriculture Test Surface

The retained 41 library unit tests exercise configuration/numeric behavior,
market ordering, native reset/randomness, terminal behavior, transactional
failure rollback, and economic/attribution counters including state neutrality.
Nine RNG integration tests use embedded CPython vectors, including large and
negative seeds, rollover and long streams. These 50 tests do not read episodes.

Task 1.1 added nine `tests/replay_parity.rs` tests: four replay tests, a public
API test, and four comparator regressions. Task 1.1b extends the file to 19
(see [live differential parity](#kaggriculture-live-differential-parity)); the
Task 7.5 review adds three negative controls, for 22. The API test covers decoded PASS
commands, public state, terminal banks, and economic/attribution counters.
Comparator tests reject private inventory and private field insertion-order
drift, public market-map insertion-order drift, and public integer-versus-float
drift. Replay tests compare complete public and private state, the key order of
every object at every depth of both (market inventory/prices, shed, seeds,
inventories), statuses, typed `Vec<f64>` rewards, step/done and terminal banks. Typed rewards accommodate recorded `[0,0]` versus
serialized `[0.0,0.0]` without relaxing public market-number comparisons.

Each replay constructs `Game::new_with_seed_decimal(config, seed, 2)` directly. No expected initial
state, RNG/shop schedule or final bank value is injected into the constructor.
System `gzip -dc` reads each pinned trace; decompression or fixture failures are
hard failures. Each episode has 719 transitions and 720 checked snapshots
(initial plus successors): **2,876 transitions and 2,880 snapshots** overall.

| Episode | Seed | Final bank seat 0 | Final bank seat 1 |
| --- | --- | --- | --- |
| 95324500 | 181681617 | 97,126 | 32,640 |
| 95901360 | 804786120 | 143,344 | 151,788 |
| 95921764 | 2089097928 | 7,843 | 94,230 |
| 95990191 | 1447832391 | 87,792 | 99,703 |

The 79 excluded library unit tests belong to excluded bots/controllers (47),
FFI (16), matching (3), flat features (5), grammar (4), and policy rows (4).
The excluded ShopRouter integration test compares controller actions and executes
zero engine transitions. Bot/controller/matching fixtures, examples and binaries
are excluded with per-file reasons. Grammar coverage returned with Task 1.2;
selected opponent coverage returns when those opponents are imported.

### Kaggriculture Verification and Limits

At completion of Task 1.1, offline engine tests passed **59/59 with none ignored**
(69/69 after Task 1.1b, 87/87 after merging Task 1.2); the root suite passed
**155 with two ignored** (164 with two ignored after Task 1.2). Receipts in
`ops/rebuild-2026-09-29/1.1/` show the
private-order regression fail with plain JSON equality, then pass with explicit
key-order checks. Claude's review added the public-order and nested private-order
regressions, which fail against the earlier field-specific check and pass with
the recursive check. A separate deliberately
corrupted initial snapshot makes episode 95324500 fail; restoring the snapshot
passes the episode and full suite. Pinned fixture bytes are never corrupted.

`just rs-prepare` and `just prepare` run engine formatting in check mode, Clippy
and tests with separate `--manifest-path engine_rs/Cargo.toml` invocations.
Preparation also runs the provenance checker. Two byte-frozen sources (`lib.rs`
and `econ_attrib.rs`) have inherited formatting drift and are excluded by exact
path in root `rustfmt.toml`; retained RNG and authored replay files remain checked.
Clippy reports six inherited style findings: one `too_many_arguments`, four
`collapsible_if`, and one `needless_range_loop`. Only the engine command allows
those three named lints after `-D warnings`; the authored replay test explicitly
re-denies them. These exceptions preserve the required source hashes, and both
raw failing checks are retained. They are tooling deviations from the reviewed
brief, not changes to rules or weaker parity comparisons.
The package keeps its own lockfile and remains outside the root workspace.
Task 1.2 added the root grammar and promoted the existing Serde JSON dependency.
Task 1.3 now adds a root path dependency, deliberately unifying arbitrary-precision and
ordered JSON features. The test-only `RandomCall::Uniform` Number decoder repairs
L4 without changing production rules or tolerances. The literal regression passes
before dependencies, fails after feature unification, and passes after repair;
root tests at the Task 1.3 A checkpoint are 157 passed and two ignored.
Observation encoding does not expand rules parity.

Task 1.1 parity was scoped to these four recorded worlds plus synthetic unit/RNG
coverage. The trace headers' recorded RNG and shop schedules are not compared
directly; their effects are checked only through the resulting public and
private state. Task 1.1b adds a live differential check against Kaggle's own
Python engine (below). Neither establishes adapter/model integration, learning
quality or GPU throughput. Historical full-engine and performance claims in
provenance do not qualify this trim. No training or network access is required
by these checks.

### Kaggriculture Live Differential Parity

Task 1.1b answers the owner request "add a parity check after your rust engine,
with kaggle environments". `scripts/kaggriculture_parity/generate_traces.py`
runs `kaggle_environments.make("kaggriculture", configuration=...)` with an
explicit seed and records traces in the official
`kaggriculture-re-parity-v1` format: complete public and private state, statuses
and rewards after every step. It refuses to run unless the installed package is
1.32.7 and `envs/kaggriculture/kaggriculture.py` has exactly the SHA-256 pinned
in `engine_rs/Cargo.toml`; each header records both. Task 1.1b was built when
the project lock still pinned 1.29.0 (no Kaggriculture environment), so its
receipts and the sweep use an isolated environment. The owner's 1.32.7 project
pin, merged alongside, also satisfies the guard; a Python test asserts that the
project environment loads the pinned engine. The isolated command stays valid:

```sh
UV_OFFLINE=1 uv run --isolated --no-project --with kaggle-environments==1.32.7 \
  python scripts/kaggriculture_parity/generate_traces.py --preset committed \
  --out engine_rs/fixtures/generated --manifest
```

The format adds one record type. When Python's interpreter raises on an input,
Kaggle's `env.step` fails and keeps its state; the generator records a
`rejected` record (actions and Python exception) and then steps the policy's
fallback actions. Rust must return an error for the same actions without
changing any state: public and private trees (values and map key order),
statuses, rewards and completion are compared before and after. Actions are recorded as submitted (after a
JSON round trip); Kaggle's action-schema defaulting only adds missing top-level
`farmer`/`hands`/`market` keys, which the interpreter defaults identically.

Policies are seeded and selectable per seat: `random` (legal-looking unit
commands for every actor, usually meaningful on the actor's tile, plus 0–4
market orders and occasional over-limit queues); `edge` (the random policy plus
HIRE until and past the 241-actor model capacity, 0/1023/negative/float/string/
boolean/null/array/huge quantities, empty orders and markets, duplicate and
reversed queues, BUY_LAND past three quadrants, unknown and lower-case verbs,
malformed orders and unit commands, commands for non-existent hands,
oversubscribed PLANT, malformed whole actions, and uncaught-`int()` probes);
Kaggle's built-in `pass`, `random` (seeded through the policy RNG) and `starter`
agents; and mixed seats. Configuration variants cover the defaults, free hires
with 12 orders per turn (up to 249 hands, 250 actors), 10,000,000 starting money
for the edge policy's 10^12-unit seed orders, which reach Python's
100,000-iteration market-loop escape,
and a custom set (7 turns per day, capacity 30, weed chance 0.2, shop and
town-centre intervals, hire multiplier 3, 4 orders and sparse `marketParams`
overrides, including `hinge` and `log10` curves). Seeds include negative and
above-`u64` integers. The 303 one-command probes step a two-turn preamble (buy
wheat, a goose and carrots, hire, pick up), then one probe command: every
malformed unit and order, 33 quantity spellings for PICKUP, PLACE and the four
quantity-taking market verbs (BUY_SEED, SELL, BUY_PRODUCT, BUY_ANIMAL),
unhashable verbs and items, missing-hand commands and malformed whole actions,
including `null`. Scripted actions are submitted exactly; before the Task 1.1b
verification fix the `null` whole-action probe (probe 296) was replaced by PASS,
so the sweep recorded at `8ca378b` did not probe it (full-game policies did).

`engine_rs/tests/replay_parity.rs` uses one comparator for three sources and
reports the first divergence as line, step, kind and field path with expected
and actual values. Rust starts from configuration and seed only.

| Source | Traces | Transitions | Result |
| --- | --- | --- | --- |
| Official recorded episodes | 4 | 2,876 | agree |
| Committed generated games | 8 | 3,960 (25 Python rejections) | agree |
| Committed divergence repros | 7 | 7 (5 Python rejections) | expected failures, asserted exactly |
| Local sweep, 40 games | 40 | 21,824 (155 Python rejections) | agree |
| Local sweep, 303 probes | 303 | 1,515 | 266 agree, 37 diverge (D1 12, D2 25) |

The committed games are `random` vs `random`, `edge` vs `edge`, built-in
`starter` vs built-in `random`, `random` vs `edge`, `edge` free-hire, `edge` vs
`random` rich, `random` vs built-in `starter` custom, and built-in `pass` vs
`edge` with seed −123,456; seeds are 11, 22, 33, 44, 55, 66, 2^64+7 and −123,456.
`engine_rs/fixtures/generated/MANIFEST.json` pins each trace's SHA-256, size,
policies, policy seed, configuration, counts and any expected divergence. The
15 files total 665,021 bytes (budget 4,000,000). `scripts/check_engine_trim.py` validates
that manifest, its exact file inventory, the Cargo engine pin and the budget;
`TRIM_MANIFEST.json` pins the manifest bytes. Rust tests also fail when a
committed state value, private key order, action or rejection claim is
perturbed; since Task 7.5, also when per-transition rewards, the header
transition count or terminal banks are. A Python test regenerates the committed set from the live engine and
requires identical bytes; it skips with a message when an isolated 1.32.7
environment cannot be built offline.

The current recorded sweep ran `uv run python
scripts/kaggriculture_parity/sweep.py --games 40` at commit `6217868` (after the
verification fixes) on the owner's Mac (single process, 47 s): base seed
20,260,929, 12 rotating policy/configuration pairs, 343 traces and 23,339
transitions, 204 Python rejections (155 in games, 49 in probes). All 40 games
agree; 37 probe divergences are confirmed D1 12 and D2 25 (every D1 ASCII
recheck passes), with 0 new. Its summary, including per-policy counts and every
first divergence, is `ops/rebuild-2026-09-29/1.1b/verify-r1/sweep-summary.json`;
the first sweep at `8ca378b` (same counts, input-only classification, null probe
submitted as PASS) is kept at `ops/rebuild-2026-09-29/1.1b/sweep-summary.json`.
With `--include-known-divergences` (12 games, base seed 777), 8 diverge, all
confirmed (D1 2, D2 6). Set
`KAGG_PARITY_TRACES=<absolute dir>` (and optionally `KAGG_PARITY_REPORT`) to
replay a larger directory with the same comparator.

#### Known Divergences

Both classes concern malformed input only. The vendored kernel bytes are pinned,
so they are recorded, not repaired. Seven minimal repros are expected-failure
fixtures with these reasons: one-step games (`episodeSteps` 2) whose first
seat-0 action carries only the divergent field, so each diverges at line 1,
step 0; the test fails if any stops diverging exactly there. Full-game policies
exclude these inputs by default so a game can test the rest of the episode;
`--include-known-divergences` restores them.

The sweep classifies a divergence from the observed mismatch, not from the
input alone. D2 requires a `rejected` record whose Python error is
`TypeError: unhashable type`, an unhashable input on that line and a Rust
`rust_accepted` result. D1 requires a Python-accepted transition with Unicode
digit input on the divergent line, and a Rust recheck of the same trace with
only that line's digits spelled in ASCII (what Python's `int()` read) must pass
that line. Anything else is reported as a new divergence and the sweep exits 1;
a trace corrupted at `public.day` on a D1 line stays unclassified.

- **D1, Unicode decimal digits.** Python `int()` accepts non-ASCII decimal digit
  strings such as `"\u0663"` (Arabic-Indic three) and `"\uff13"` (full-width
  three). In PICKUP/PLACE counts Rust returns an error where Python acts; in
  BUY_SEED, SELL, BUY_PRODUCT and BUY_ANIMAL quantities Rust drops the order
  where Python executes three units. First found at step 13 of the free-hire
  edge game (`BUY_PRODUCT FERTILIZER "\u0663"`, money 2,159 versus 2,460).
- **D2, unhashable arrays/objects.** Python raises `TypeError`, so Kaggle's step
  fails, when an array or object reaches a dict lookup: a unit verb
  (`[["NORTH"]]`), a PLANT crop (also for a non-existent hand, via the atomic
  seed check), a PICKUP or PLACE item, or a BUY_SEED or BUY_ANIMAL item. Rust
  treats these as no-ops and accepts the step.

All ASCII quantity spellings agree, including `" 4 "`, `"+2"`, `"0002"`,
`"1_0"`, floats, booleans, `null`, arrays, 10^30 and uncaught-`int()` errors
(`"abc"`, `null`, `[1]` counts), which both engines reject. Our policy's
grammar emits only ASCII verbs, item names and integers, so neither class is
reachable from model actions; an adapter accepting external actions would need
explicit handling. The original failing full-game traces, their first
divergences and the replay receipt are in
`ops/rebuild-2026-09-29/1.1b/evidence/first-failing-games/`.

Limits: generated games come from seeded random, edge and built-in policies,
not strong play, and most random seats go bankrupt; the four official episodes
remain the only recorded competitive worlds. Framework behavior outside the
interpreter (timeouts, agent errors, `INVALID` statuses) is not modeled. The
sweep is a bounded sample: 40 games and 303 probes, not exhaustive input
coverage. A larger pod sweep remains open.

## Kaggriculture Native Grammar (Task 1.2)

The root compiles the v4.1 typed grammar, strict JSON encoder and checked i64
decoder in `src/kaggriculture/grammar.rs`. Task 1.3 retired the temporary
engine-side include and moved the kernel acceptance tests to root
`src/kaggriculture/grammar_kernel_tests.rs` when adding the production engine
dependency and reopening L4. No retained kernel bytes changed.

The independent fixture pins reference `65f0eac5` and all recorder inputs. It
contains 320 scheduled accepted programs: 256 seeded synthetic programs and
64 unmodified real replay seat actions. There are 64 synthetic 241-actor
programs, of which 22 have ten orders and length 252 (16 forced by the schedule,
six incidental). Full-market counts are 180 synthetic and eight real. All three
reference decoders agree on accepted actions. The four traces contain 5,752
candidate seat actions; every candidate passes complete-layout/codec admission,
so the manifest explicitly records absent replay-rejection categories.

The additional 44 controls contain 43 v4 rejections and one zero-padding
acceptance. Eight records have classified oracle differences: four capacity
cases accepted by the old training decoder, three nonzero-padding cases
accepted by prefix-only decoders, and one incomplete-prefix case for which the
training/Python decoders are inapplicable. Errors and each original verdict are
preserved separately. The new grammar never supplies expected oracle actions.
Two native reference recordings reproduce the fixture and manifest exactly.

The shared grammar tests cover all 964 independent recorded table bits,
reachable local support/transition classes, actor ordinals 1 through 241,
shape affinity, 140 unit and 98 market cases, strict encoder round trips and
nonmutation, signed i64 transport and padding failures, and terminal traversal
of every accepted fixture. The coupled-HIRE test enumerates all 14³ races for
each budget 0/1/2/3/10 against the actual Rust cursor law, within 1e-12. This
enumerates support equivalence classes, not every A/O/H combination: A affects
readiness, ordinal and capacity; O affects market queue availability.

Kernel acceptance tests require exact JSON and compare direct execution with
decoded execution. They cover both seats, command matrices, meaningful transfer
effects, EMPTY/zero quantities, insufficient funds, explicit dense states via
`Game::from_header`, actor order and the 240-to-241 HIRE boundary. Selected
replay comparisons check 64 seat actions in their actual seeded replay states,
including recursive public/private object order, outcomes and counters. These
64 comparisons supplement the retained 2,876-transition replay suite; they are
not a new full-season codec qualification.

On the Task 1.2 branch (before Task 1.1b merged), full `just prepare` passed:
root Rust **164 passed, two ignored**; engine **77 passed, none ignored** (41
retained unit, nine RNG, nine replay, 18 shared grammar/kernel tests); Python
**1,045 passed, three platform skips**; tooling pytest alone 87. Actual command
results and fault-injection evidence are recorded in
`ops/rebuild-2026-09-29/1.2/results.md`. After merging Task 1.2 with Task 1.1b's
live parity and Task 2.3's heads, the engine suite passes **87, none ignored**
(41 retained unit, nine RNG, 19 replay-parity, 18 shared grammar/kernel), the
root suite **164 passed, two ignored**, Python **1,337 passed, four skipped**
and tooling pytest alone 106; receipts are in
`ops/rebuild-2026-09-29/merge-1.2/`. After the later Tasks 3.1–3.4 and
1.4/1.5-brief merges (`kg/merge-trainer-lanes`), which change no Rust, full
`just prepare` passes with the same engine **87** and root **164 passed, two
ignored**, and Python **1,458 passed, five skipped** (the fifth skip waits for
the native Kaggriculture evaluation env); receipt
`ops/rebuild-2026-09-29/merge-trainer-lanes/prepare.log`. Those engine counts
are historical: Task 1.3 retired the grammar bridge (below), so the engine suite
is now 69 tests (41 retained unit, nine RNG, 19 replay-parity) and the nine
kernel acceptance tests run in the root crate. After merging
Task 1.3 onto that integration (`kg/merge-1-3`), full `just prepare` passes:
engine **69, none ignored**, root **254 passed, four ignored** (two
`rl::action_spec` audits plus Task 1.3's explicit oracle generation and
optimized cost diagnostic), Python **1,618 passed, seven skipped** (Task 1.3's
two CUDA-only pinned-memory cases added); receipt
`ops/rebuild-2026-09-29/merge-1.3/prepare.log`. After merging Phase 4's
teacher distillation (`kg/merge-teacher`), which changes no Rust, full
`just prepare` passes with the same engine **69** and root **254 passed, four
ignored**, and Python **1,673 passed, 11 skipped** (Phase 4's two trainer-seam
and two run_ppo-seam tests added); receipt
`ops/rebuild-2026-09-29/merge-teacher/prepare.log`. After merging the 8-rank
config and Phase 6.3b plan (`kg/merge-8rank`), which changes only configs,
tests, the plan and the cookbook, full `just prepare` passes with the same
engine **69** and root **254 passed, four ignored**, and Python **1,690 passed,
11 skipped** (the 8-rank config and startup-workload tests added); receipt
`ops/rebuild-2026-09-29/merge-8rank/prepare.log`. After merging Tasks 1.4 and
1.5 (`kg/merge-env-adapter-r2`, onto the custody-sweep integration `faed717`), full
`just prepare` passes with engine **69** (41 unit, nine RNG, 19 replay-parity),
root **274 passed, five ignored** (Task 1.4's native lifecycle, admission and
recorder tests added) and Python **2,319 passed, 10 skipped** (both sides'
suites; Task 1.5 un-skips the native-table and evaluation-env tests, while
the four teacher trainer/run_ppo seam tests, CUDA, pinned-memory, flash-attn and
x86 quantization cases stay skipped); receipt
`ops/rebuild-2026-09-29/merge-env-adapter/prepare.log`. After merging Task 7.1's
opponents (`kg/merge-7-1`), which adds the standalone `opponents_rs` crate and
changes no engine or root-crate Rust, full `just prepare` passes with engine
**69**, root **274 passed, five ignored**, opponents **22** (12 unit, five
lifecycle, five oracle), and Python **2,400 passed, 21 skipped** (Task 7.1 adds
81 tests and 11 skips: nine learned-seat checks awaiting an `opponents_rs` seat
in the native env, one broad regeneration and one original-source reread);
receipt `ops/rebuild-2026-09-29/merge-7.1/prepare-on-5b43062.log`. After
merging the Task 3.1 remainder and Task 3.5 (`kg/merge-3-1-3-5-c`, onto the
Task 4.4 integration `f02ed02`), which change no Rust, full `just prepare`
passes with engine **69**, root **274 passed, five ignored**, opponents **22**
and Python **2,494 passed, 17 skipped** (the four teacher trainer/run_ppo
seam tests now run); receipt `ops/rebuild-2026-09-29/merge-3-1-3-5/prepare.log`.
After merging the W&B wiring (`kg/merge-wandb-c`, `bd1c927`), which changes no
Rust, full `just prepare` passes with engine **69**, root **274 passed, five
ignored**, opponents **22** and Python **2,587 passed, 17 skipped**; receipt
`ops/rebuild-2026-09-29/merge-wandb-c/prepare.log`. After merging Task 7.5
(`kg/merge-7-5-c`), which adds three replay negative controls, full `just prepare` passes with engine **72** (22 replay-parity), root **274 passed,
five ignored**, opponents **22** and Python **2,587 passed, 17 skipped**;
receipt `ops/rebuild-2026-09-29/merge-7-5-c/prepare.log`.
The trim
checker's fixed authored set is now exactly the replay-parity test plus the
generated-trace manifest. The native `grammar_tables()` matches all 964 bits of the Python
heads' `expected_grammar_tables` in that merge-time cross-check. Task 1.4 now
exports the native table/codec bindings and tests all 964 bits again; Task 1.5
wires `native_grammar_tables(device)` into the Python model as its default
tables (see the [Task 1.5 summary row](#what-is-tested)).
CPU grammar admission does not qualify the model sampler/replay, CUDA/BF16
behavior or L6's distinct Inductor GEMM overflow fix. Native batch transactions
and CPU buffer admission have separate Task 1.4 evidence below.

## Kaggriculture Observation Coverage (Task 1.3)

The root encoder adds hand-derived field, strict tile, exact integer/rank,
privacy, role-order, finite-cost, buffer-reuse and transactional-output tests.
The tile-shape scan checks 576,000 tiles across all 2,880 pinned official states.
The tensor-only reconstructor covers all 8,176 legacy offsets with independent
hand expectations and 16 failing/restored mutations; eight reconstruction
controls and five added-fact tests pass. This is observation information
coverage, not new Python-engine rules differential parity.

The deterministic generator produces 512 input records. The original v1 seeded
policy could not reach R1's quota of four non-synthetic states with more than 16
actors: it produced zero, because end_of_day clears hands and v1 hires on at most
two of every eight turns. Claude's reviewed `observation-corpus-v2` correction
appends HIRE entries during hours 0–7, up to `min(M,4)` entries per turn, and
stops at 16 hands. It changes no quota, and actual generation now yields six
such states. Every other quota passes; only the reviewed dense d=31 case covers
shed reordering.

The pinned reference crate recorded all 1,024 seat rows (33,488,896 raw bytes,
781,743 bytes compressed across both fixture files). The tensor-only
reconstruction matches every row bitwise at all 8,176 offsets, with no
tolerance. Regenerating at a later commit reproduced both compressed fixtures
byte for byte.

These checks discriminate:

- Swapping the market inventory and price channels in the encoder fails the
  comparison at offset 889.
- A one-byte reference corruption fails custody validation.
- Reversing shed ranks fails the real-schema Python corpus test.

All three mutations were restored. Fifty-three custody tests cover corruption,
ordering, strict metadata, source drift and resource guards. Source custody
captures and rechecks every engine build input: `engine_rs/Cargo.toml`,
`Cargo.lock`, `TRIM_MANIFEST.json` and each `engine_rs/src` module, and an
undeclared engine module fails regeneration. The committed corpus predates that
repair, so its identity omits the engine `Cargo.toml`, `py_random.rs` and
`econ_attrib.rs` hashes; their committed bytes are unchanged since its root
commit `469e8ec`. The actual Task
2.1 `check_contract()` runs on every binding batch and on all 512 records. The
pinned-memory variants run only where CUDA is available, matching the starter's
CUDA-only pinning, and then also assert that every buffer is pinned. Elsewhere
they skip without touching the allocator: on macOS torch 2.9 routes pinned
allocation to MPS, where a fill raises or kills the process (SIGSEGV observed in
independent verification). Pinned and GPU behavior are therefore unqualified.

The exactly-one-snapshot test passes after its two-acquisition mutation fails.
The Task 1.3 optimized timing build stopped at the Mac memory limit before any phase ran,
and no debug timing is substituted. The pod command is in
`ops/rebuild-2026-09-29/1.3/timing.json`; that diagnostic remains incomplete.
Task 1.4 later measured snapshot acquisition, validation and observation writing
as components (see [native lifecycle coverage](#kaggriculture-native-lifecycle-coverage-task-14)). Current
qualification counts and receipts are in `ops/rebuild-2026-09-29/1.3/claude-review.md`
and, for the verification round 1 fixes, `ops/rebuild-2026-09-29/1.3/r1-fixes/`.
`ops/rebuild-2026-09-29/1.3/results.md` is the frozen, historical pre-integration
handoff receipt (incomplete qualification).

Task 1.3 also retires the contract v4.1 grammar bridge: the nine kernel
acceptance/replay-state tests moved from `engine_rs/tests/grammar_kernel.rs` to
root `src/kaggriculture/grammar_kernel_tests.rs`, and the engine file and its
authored trim registration are removed. A restored BuyLand-as-HIRE decode
mutation fails the root `decoded_programs_feed_kernel` test.

## Kaggriculture Native Lifecycle Coverage (Task 1.4)

The native class and four cold grammar functions use the root crate's merged
Task 1.2 grammar and Task 1.3 encoder. The root remains
`src/kaggriculture/mod.rs`; there is no second module root or duplicate kernel
test file. Task A confirms Task 1.3 already retired
`engine_rs/tests/grammar_kernel.rs`: root `grammar_kernel_tests.rs` imports the
real grammar and retained engine, the manifest contains no authored grammar
exception, and the trim checker passes. The strengthened
`test_no_authored_grammar_path_include_after_root_engine_edge` pins that state.
The targeted trim tests pass 3 cases, and the root kernel route passes all 9.
No vendored kernel bytes changed.

Native reward tests isolate own starvation/drought/ineffective counters from
unweighted counters and raw-bank terminal outcomes. They pin the ten-case binary64
admission predicate plus Task 1.5's strengthening case, disabled-component
behavior, finite outputs,
reference two-rounding schedule and an independent telescoping/ULP budget.
Review found the first budget depended on actual errors and therefore admitted
a zero-reward mutation; the corrected independent endpoint/budget test rejects
that mutation and passes after restoration. That correction does not replace
the full recorded TrainingBatch trajectory comparison.

The lifecycle tests drive real `ValidatedObsBuffersMut` destinations. Batch
failure cases include malformed peer actions, an engine Result error, a worker
panic, failed auto-reset and failed late observation preparation. Reset and
truncate separately cover late selected-environment construction/preparation
errors, worker panic and seed exhaustion. Injection-hit assertions prove the
ample-seed cases reached their intended failure points. Every case compares all
35 destination bytes, game snapshots, a fresh observation, seeds/counter and
terminal records, then retries against an untouched control. Moving a live seed
counter write before commit makes each named rollback test fail; both source
mutations were restored byte-exactly. These negative controls perturb live seed
state, not output-buffer bytes.

The full default-horizon PASS test ends exactly at transition 719, captures final
banks/counters/winner and consumes one auto-reset seed while publishing reset
clock/live observations. Truncation tests preserve a nonzero economic transition
and distinguish the pre-reset bootstrap state. Sentinel tests replace unselected
observation rows with 0x5A bytes (true for bool), then check both single-selection
masks and an all-false mask in nonterminal and terminal fixtures. Only selected
observation rows may change; all six transition tensors remain identical.
Replacing the selected-row commit with whole-batch `ObsStaging::publish` fails
at the unselected-row comparison; restoration passes the 12-test native
lifecycle suite.

Python tests exercise all 35 destinations' dtype, shape, layout, alignment,
writeability and byte-overlap admission, plus token/length/mask admission,
terminal-record copies and the real schema. A controlled native latch verifies
a second Python thread progresses while observation work is detached. Seed
partition tests exercise world sizes 2 and 8, all ranks sequentially, with 67
consumed seeds per rank across constructor, full/partial resets and simultaneous
terminal resets. They also check failed-step nonconsumption, ordering and
exhaustion rollback. Mutating the factory arithmetic to
`(base + rank*n_envs) + k` makes both cases fail with `rank 1 collides with rank
0`; the restored native binding suite passes 339 cases. No distributed process,
model or training run is implied.

Task F adds 43 Python codec/table cases. Direct missing-function calls first
failed 42 cases; implementation passed those 42, then a contiguous caller-view
case was added. All 964 table bits match the independent expected tables and
returned arrays are independent. The frozen Task 1.2 corpus passes 321 accepted
round trips (320 scheduled, including 64 with 241 actors and 22 at length 252)
and 43 rejection classes. Every failed encode preserves all 3,024 cells.
The final native environment/grammar suites pass **383 cases** (340 + 43),
including the two-thread versus one-thread native-pool equivalence case.
Receipts, including corrected malformed test fixtures and diagnostic-message
matches, are in `ops/rebuild-2026-09-29/1.4/`; malformed-test failures are not
counted as missing-behavior evidence.

The cast audit examines 99 numeric casts plus saturating/wrapping expressions
and identifies two reachable unbounded HIRE cash casts. Root `admission.rs`
checks the exact executed hire costs; six helper tests and a transactional
native rejection test pass. Root Cargo release policy enables engine overflow
checks.

On the Mac, Codex's release overflow proof and optimized timing builds stopped at
the 960 MiB watchdog before their test bodies ran (`timing.json`,
`d-release-attempt.json`). Claude ran both on the pod (`claude-review/pod-oracle/`):

- `release_dependency_overflow_is_caught` passes in release, and fails when the
  engine overflow-check override is disabled with `--config`.
- The fat-LTO phase timings are component measurements only: dense 241-actor
  composed work has a 747.6 µs median, and overflow checks show no measurable
  cost.

Task G implements the deterministic recorder, policy, custody and replay checks
for exactly 16 games, seeds 17000–17015, 719 transitions each, with one live
reference game at a time. The Mac recording attempt stopped during
exported-reference compilation and published nothing
(`reference-recording-attempt.json`). The pod recording completed in 32 s. Every
game reached positive starvation, drought and ineffective counters, executed
hires, animal placement and sales. The fixture is 310,365 bytes compressed,
under the 8 MiB / 256 MiB budget; its hashes are in the manifest.

The native replay matches the reference bit for bit on rewards, dones,
transition banks, economic counters, seeds and terminal records over all 11,504
transitions. It also agrees with the independent Python reward formula within
one f32 ULP. Inverting native `dones` fails at
`first divergence game=0 seed=17000 step=0 seat=0 … field=dones`. The
restored source passes again.

Root Rust passes **274 tests, zero failures, five ignored**; the engine passes
**69**. Claude's review adds a native L6 oracle: every call rewrites all 35
outputs of the one caller-owned set in place, including padding. With it the
native environment and grammar suites pass **387**. Full `just prepare` on the
Mac passes Rust, build, trim, documentation, mypy and **2,042 Python tests with
7 skips** (`claude-review/just-prepare.log`). Among those skips,
`test_native_tables_match_expected_tables` then waited for Task 1.5's
`native_grammar_tables(device)`; Task 1.5 un-skips it.

Task 1.5 has merged the Python adapter, `rewards.py`, Python codec and device
table bridge with CPU tests. Its pinned CUDA reuse-fence qualification remains
open. CPU checks establish no GPU,
training or complete-update throughput claim.
