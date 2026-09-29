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
The native crate's 20 tests pass: 12 unit tests (including Starter's five
pinned inline cases), five lifecycle/match integration tests and three oracle
tests (comparator regression, original-Python parity, per-seat tampering).

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
No trace has a whole-step rejection, a BUY_PRODUCT above the inventory index or
a mid-episode replay. `buy_quantity_above_inventory_index` compares a quantity
to a price index, not stock, so engine-confirmed shortages and individual-order
rejection are uncovered. Mid-episode replay and reset are checked natively
(lifecycle test), not against Python.

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
source is copied. Learned-seat integration tests are explicitly skipped with
`needs Task 1.4 binding`; no substitute binding is introduced. Default-config
CPU qualification establishes neither custom-config support nor playing strength.

Run 1 at `21d0f45` stopped on private `fib`, private `Game.config` and missing
accessors under Claude's original placement prompt. Its compiler/source receipts
remain unchanged. The revised view resolves that placement boundary without
editing the frozen engine, restoring `policy_rows`, copying the reference
all-controller dispatcher or adding a root-crate dependency. The trim updater
accepts the committed run-1 manifest and preserves retained/authored entries.

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

Task 1.1 added nine `tests/replay_parity.rs` tests: four replay tests, a public
API test, and four comparator regressions. Task 1.1b extends the file to 19
(see [live differential parity](#kaggriculture-live-differential-parity)). The API test covers decoded PASS
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
perturbed. A Python test regenerates the committed set from the live engine and
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
decoder in `src/kaggriculture/grammar.rs`. The authored engine integration test
includes that same source and its shared tests under the engine's separate
edition/feature graph. No retained kernel bytes change. The include is temporary:
Tasks 1.3/1.4 retire it and move acceptance tests to root integration when adding
the first production engine dependency, reopening L4 then.

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
`ops/rebuild-2026-09-29/merge-1.3/prepare.log`. The trim
checker's fixed authored set is now exactly the replay-parity test plus the
generated-trace manifest. The native `grammar_tables()` matches all 964 bits of the Python
heads' `expected_grammar_tables` in a merge-time cross-check, but no Python
binding exists yet, so the heads still use the Python stand-in.
CPU grammar admission does not qualify
native batch transactions, PyO3 buffer ownership, the model sampler/replay,
CUDA/BF16 behavior or L6's distinct Inductor GEMM overflow fix.

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
The optimized timing build stops at the Mac memory limit before any phase runs,
and no debug timing is substituted. The pod command is in
`ops/rebuild-2026-09-29/1.3/timing.json`; phase costs remain unmeasured. Current
qualification counts and receipts are in `ops/rebuild-2026-09-29/1.3/claude-review.md`
and, for the verification round 1 fixes, `ops/rebuild-2026-09-29/1.3/r1-fixes/`.
`ops/rebuild-2026-09-29/1.3/results.md` is the frozen, historical pre-integration
handoff receipt (incomplete qualification).

Task 1.3 also retires the contract v4.1 grammar bridge: the nine kernel
acceptance/replay-state tests moved from `engine_rs/tests/grammar_kernel.rs` to
root `src/kaggriculture/grammar_kernel_tests.rs`, and the engine file and its
authored trim registration are removed. A restored BuyLand-as-HIRE decode
mutation fails the root `decoded_programs_feed_kernel` test.
