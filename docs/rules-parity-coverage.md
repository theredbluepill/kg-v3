# Rules Parity Coverage

This document is the system of record for Orbit Wars and Kaggriculture parity
coverage. Keep it updated whenever the Python reference, fixture generators,
or Rust rules engines change. The Orbit Wars coverage below remains unchanged.

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

## Kaggriculture Rules Kernel

Task 1.1 retains a standalone `engine_rs` package pinned to reference commit
`65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. Its compatibility target is
`kaggle-environments==1.32.7`, Python engine SHA-256
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.
`engine_rs/TRIM_MANIFEST.json` accounts for all 125 reference files: 12 retained,
113 excluded, plus the authored replay test and non-engine change inventory.
`python scripts/check_engine_trim.py` checks hashes, exhaustive inventory,
declared original-line edits, exact Cargo removals and append-only provenance.
Independently of manifest declarations, only `lib.rs`, `Cargo.toml`,
`Cargo.lock` and `VENDORED_FROM.md` may differ from the reference; the lockfile
must equal the reference minus the six-package Rayon closure, and the Task 1.1
provenance appendix must follow the historical bytes with a pinned SHA-256.

Only reference `lib.rs` lines 19, 21–25 and 27 are removed: declarations for
`ffi`, `joint_matching`, `myolie_features`, `myolie_sampler`, `native_agents`,
`policy_rows` and `training`. The resulting file is 185,626 bytes, SHA-256
`c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd`.
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

The nine new `tests/replay_parity.rs` tests comprise four replay tests, a public
API test, and four comparator regressions. The API test covers decoded PASS
commands, public state, terminal banks, and economic/attribution counters.
Comparator tests reject private inventory and private field insertion-order
drift, public market-map insertion-order drift, and public integer-versus-float
drift. Replay tests compare complete public and private state, the key order of
every object at every depth of both (market inventory/prices, shed, seeds,
inventories), statuses, typed `Vec<f64>` rewards, step/done and terminal banks. Typed rewards accommodate recorded `[0,0]` versus
serialized `[0.0,0.0]` without relaxing public market-number comparisons.

Each replay constructs `Game::new(config, seed, 2)` directly. No expected initial
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
are excluded with per-file reasons. Grammar coverage returns with Task 1.2;
selected opponent coverage returns when those opponents are imported.

### Kaggriculture Verification and Limits

At the Task 1.1 checkpoint, offline engine tests pass **59/59 with none ignored**;
the retained root suite passes **155 with two ignored**. Receipts in
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
The package keeps its own lockfile and remains outside the root workspace. Task
1.3 now adds a root path dependency, deliberately unifying arbitrary-precision and
ordered JSON features. The test-only `RandomCall::Uniform` Number decoder repairs
L4 without changing production rules or tolerances. The literal regression passes
before dependencies, fails after feature unification, and passes after repair;
root tests at the Task 1.3 A checkpoint are 157 passed and two ignored.
Observation encoding does not expand rules parity.

Parity is scoped to these four recorded worlds plus synthetic unit/RNG coverage.
The trace headers' recorded RNG and shop schedules are not compared directly;
their effects are checked only through the resulting public and private state.
It does not establish exhaustive malformed-input parity, a fresh differential
run against Python, adapter/model integration, learning quality or GPU throughput.
Historical full-engine and performance claims in provenance do not qualify this
trim. No training or network access is required by these checks.

## Kaggriculture Observation Coverage (Task 1.3)

The root encoder adds hand-derived field, strict tile, exact integer/rank,
privacy, role-order, finite-cost, buffer-reuse and transactional-output tests.
The tile-shape scan checks 576,000 tiles across all 2,880 pinned official states.
The tensor-only reconstructor covers all 8,176 legacy offsets with independent
hand expectations and 16 failing/restored mutations; eight reconstruction
controls and five added-fact tests pass. This is observation information
coverage, not new Python-engine rules differential parity.

The deterministic generator produces 512 input records, then rejects them:
the unchanged R1 recipe has zero non-synthetic states with more than 16 actors
against quota four. No qualified reference fixture is published, and the full
comparison test remains a visible failure. Other quotas pass, using only the
reviewed dense shed-order exception. Source-only recorder compilation verifies
the exported pinned crate, but does not execute feature recording. Forty-three
custody tests cover corruption, ordering, strict metadata, source drift and
resource guards; 55 NumPy boundary checks pass separately from the unavailable
Task 2.1 schema. The real-schema test has an unconditional import and fails
collection until that merge.

The exactly-one-snapshot test passes after its two-acquisition mutation fails.
The optimized timing build stops at the Mac memory limit before any phase runs;
there is no debug timing substitute. Actual final command counts, fixture and
schema errors, resource receipts and the pod timing command are recorded in
`ops/rebuild-2026-09-29/1.3/results.md` and `timing.json`. Task 1.3 is incomplete;
no test skip or quota reduction makes these gaps green.
