# Kaggriculture Rust parity engine (RE)

`engine_rs` is a standalone deterministic transition kernel for the public
Kaggriculture engine. RE0 established complete-world transition parity from an exported
initial state and exogenous schedule. RE1 added native reset, CPython-compatible daily
randomness, shop generation, and permissive custom market configuration. **RE2 repairs
the independently found mixed-number square-denominator counterexample and hardens the
native gate against exported-oracle reuse. RE3 adds an exact bounded kernel for the
RQ60/RQ61 crop-modified complete-matching candidate enumeration.** It is derived
from Kaggle's Apache-2.0 `kaggle-environments` implementation; the applicable license is
included as [`LICENSE`](LICENSE).

## Official compatibility target

The transition kernel targets `kaggle-environments==1.32.7`, with
`kaggriculture.py` SHA-256
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.
These identifiers are recorded in `Cargo.toml` under
`package.metadata.kaggriculture`; the crate's own `0.1.0` version is independent.
Changing that metadata alone does not certify compatibility: rebuild the release
library and binary and retain an exact differential receipt against the installed
official source before using a different target.

The 2026-09-20 synchronization confirmed that the official source is unchanged,
checked the release library/binary, and passed Cargo tests, all-target Clippy and
the 17-case differential gate. Its
[receipt](../data/intel/engine-sync-20260920-055943Z/result.json) preserves the
before/after source and binary hashes; historical `results/` receipts are unchanged.

## Current RE2 gate

From the repository root, run the broad differential suite:

```bash
.venv/bin/python scripts/re1_parity.py
```

The gate builds the release binary and compares the official Python engine with Rust at
every observed snapshot: canonical public state, both ordered private states,
`ACTIVE`/`DONE`, rewards, and terminal banks where a case reaches `DONE`. The checked
suite is exact for **17 cases and 2,914 attempted transition boundaries**: **2,911**
transitions succeed, three fail at the intended interpreter boundary, and **2,928**
snapshots are compared.

| mode | cases | attempts | successful | snapshots | boundary |
| --- | ---: | ---: | ---: | ---: | --- |
| recorded, native reset | 4 | 2,876 | 2,876 | 2,880 | all `DONE`, banks exact |
| generated, native reset | 6 | 30 | 30 | 36 | all `DONE`, including `episodeSteps=1` and RE1V's 4-vs-5 price case |
| generated native prefix | 1 | 1 | 1 | 2 | still `ACTIVE`; huge finite horizon |
| generated explicit state | 3 | 3 | 3 | 6 | dense lifecycle/action regressions |
| generated expected error | 3 | 4 | 1 | 4 | matching failure phase; no failed-call snapshot |
| **total** | **17** | **2,914** | **2,911** | **2,928** | **13 `DONE`, one prefix, three errors** |

The generated cases cover a negative arbitrary-width seed and dense RNG rollovers;
integral-float and arbitrary-width top-level configuration; sparse/float/non-dict/
extra-key and exact wide-integer market patches; mixed float-numerator/integer-square
denominator evaluation across the 2^53 boundary; wide plant timestamps; Python-style
ASCII count coercion; all action and lifecycle families; capacity behavior; minimum
termination; and three expected-error/transaction regressions. `PLACE` retains carried
excess, while `DROP` and end-of-day deposits discard it.

The native gate deliberately corrupts every header's exported initial public/private
state, terminal banks, `shop_schedule`, and `rng_schedule` before invoking Rust, while
retaining the uncorrupted official oracle for comparison. Exactness therefore cannot
come from any exported state or schedule field. Expected failures must also match a
case-specific diagnostic category, and explicit-state regressions compare initial
lifecycle fields as well as state.

The recorded set is compact rather than a bulk replay corpus:

| episode | seed | terminal banks | role in the suite |
| ---: | ---: | ---: | --- |
| 95324500 | 181681617 | 97,126 / 32,640 | tomato lifecycle at seat 0; goose/coop at seat 1; overflow |
| 95901360 | 804786120 | 143,344 / 151,788 | tomato at seat 1; product placement at seat 0; overflow |
| 95921764 | 2089097928 | 7,843 / 94,230 | full land both seats; goose/coop seat 0; direct DROP overflow |
| 95990191 | 1447832391 | 87,792 / 99,703 | original RE0 complete-world oracle |

A one-off exhaustive audit of all 988 local replays found no one- or two-world solution
under the strict seat-symmetric coverage definition; the first three form a three-world
cover. Their union covers all five crops, all animals and buildings, all eight shop
types, full land, inventory operations, market coupling, 87 day rollovers, and terminal
reward. The audit selected fixtures; it did **not** parity-run all 988 worlds, and its
exploratory implementation was not retained as a repository tool. The generated cases
cover configuration and seed domains absent from the recorded corpus.

## Native reset and randomness

`Game::new(config, seed, 2)` constructs the official initial farm, private inventory,
market, and town state without loading a fixture. `Game::new_with_seed_decimal` accepts
signed arbitrary-width Python integers. At every end of day Rust reproduces the
official engine's fresh `random.Random((seed * 1_000_003) ^ day)` stream, including:

* CPython integer seeding and MT19937 state/tempering;
* `random()`'s 53-bit fraction construction;
* `getrandbits`, rejection-sampled ranges, `choice`, and `shuffle` behavior;
* conditional weed draws in farm/row/column traversal order; and
* the following shop draw from the same daily stream.

The RNG regression vectors cover negative and hundreds-of-bit seeds, word boundaries,
state twists, and 10,000-word stream hashes. Their CPython interpreter and `random.py`
SHA-256 provenance is recorded in `tests/py_random.rs`.

## Configuration and numeric semantics

The official framework normalizes advertised integral configuration values even when
provided as JSON floats; Rust accepts the same finite integral form and retains unknown
top-level keys for forward-compatible trace round trips. Nested `marketParams` remain
permissive dictionaries: patches are sparse, unknown metadata is retained, unknown
products and non-dict patches are ignored exactly where Python ignores them, and JSON
integer/float representation is preserved in public market state.

`serde_json` arbitrary precision plus `num-bigint`/`num-rational` preserve the measured
unbounded configuration comparisons, plant timestamps, market intermediate products,
integer shape order, and true-division ratios before values enter floating arithmetic.
Runtime inventory counts remain deliberately resource-bounded. The comparator treats
`20` versus `20.0` inside market state as a meaningful divergence.

## Replay and benchmark commands

The compact deterministic fixture format remains useful as an immutable action/state
oracle. Export a recorded world through the live official engine with:

```bash
.venv/bin/python scripts/export_re_trace.py \
  data/replays/climbers/kobe-bryant-55668682/episode-95990191-replay.json \
  --out engine_rs/fixtures/episode-95990191.jsonl.gz
```

`re_engine` accepts an uncompressed replay payload on stdin:

* `replay` — normal RE2 mode, reset from configuration+seed and ignore exported state
  and schedules except as comparison oracles;
* `replay-state` — explicit-state first-divergence regression mode for synthetic dense
  mechanics cases; and
* `branch-batch` — native-reset a common prefix once, clone the exact `Game` for each
  named candidate action/suffix, and emit only the common state, immediate successor,
  and final state for every branch; and
* `bench --iterations N` — quiet native reset plus complete action-tape rollout, without
  per-step JSON IPC.

RQ2 exercises `branch-batch` on **55 complete-matching branches at seven decision
states**. Official Python and Rust agree on all 55 immediate successors and terminal
states. The mode currently accepts complete action suffixes; it is batched transition
plumbing, not an in-Rust policy server or tensor runtime.

## In-process batch ABI

RQ3 adds a dependency-free C ABI from the same library crate (`rlib` + `cdylib`) and a
Python owner in `scripts/re_batch.py`. Persistent native `Game` vectors support
transactional reset and step, common-prefix fork by clone, snapshots, and explicit
buffer/lifetime management. A failed action in any member rolls back the entire batch.

`tests/test_re_batch.py` compares two seeds (including `2^53+1`) against official
Python, tests a 2→3 fork, reset, and whole-batch rollback. RQ3 additionally replayed five
complete 719-transition branches through official Python; immediate successors and
terminal states are exact. A 16-game × 120-step all-PASS smoke returning full snapshots
measured 24,616 transitions/s in-process versus 2,008/s through sequential official
`env.step` (12.3×); this is a one-off JSON-boundary measurement, not a universal speed
claim.

The complete-world parity and calibrated throughput companion remains:

```bash
.venv/bin/python scripts/re_parity.py
```

It times the median of three trials after warmup/calibration, with each engine running
for at least one second per timed trial. The named paths are official Python
`make`/`env.step` including framework schema/history overhead versus Rust native
configuration+seed `Game::new`/`Game::step` without agent calls or per-step JSON IPC.
Parity is the gate; the ratio is operational evidence, not a universal language
benchmark. Historical RE1 outputs live in
[`results/re1-parity.json`](results/re1-parity.json) and
[`results/re1-throughput.json`](results/re1-throughput.json). Repaired current outputs
are [`results/re2-parity.json`](results/re2-parity.json) and
[`results/re2-throughput.json`](results/re2-throughput.json).

On the repaired transactional RE2 path, the stored median-three result is **650.1
steps/s** for official Python and **11,784.1 steps/s** for Rust, an **18.1×** speedup.
The prior RE1 run measured 18.0×; throughput is machine-load-sensitive and parity is the
gate.
RE0's 149.3× remains a historical measurement of the older fixture-state/non-
transactional path, not the RE1 native-reset result.

## Direct quality gates

```bash
cargo test --manifest-path engine_rs/Cargo.toml --locked
cargo clippy --manifest-path engine_rs/Cargo.toml --all-targets --locked -- -D warnings
.venv/bin/python -m pytest -q tests
```

The crate deliberately keeps its dependency surface to ordered maps, Serde JSON,
`num-bigint`, `num-rational`, and `num-traits`; no training framework belongs here.

## RE3 joint-matching kernel

RQ61 localized its only failed deployment gate to Python candidate compilation: the
unchanged 30 daily boundaries measured 412.6 ms mean, 1,566.1 ms p99, and 1,641.8 ms
max. RE3 ports only the rectangular Hungarian solve and RQ60/RQ61 complete-matching
enumeration. R04 still generates units, jobs, and scored edges in Python, and Python
still materializes low-level actions.

Run the exact fixture and latency gate with:

```bash
.venv/bin/python scripts/re3_joint_matching_parity.py
```

The stored result covers nine recorded crop-modified graphs across early/mid/late
season and all four viable crop families, plus two synthetic tie/missing-edge cases.
Rust matches the Python graph reference in **11/11** cases and the full hybrid
materialized rows in **9/9** recorded cases. The graph wrapper improves from 423.6 ms
mean in Python to 46.2 ms through JSON+ctypes+Rust (**9.17x**). Substituting only this
kernel into the unchanged RQ61 daily identity yields **719/719** exact fallback actions,
exact banks **90,058/86,986**, and compiler latency **103.6 ms mean / 340.1 ms p99 /
382.9 ms max**, clearing the frozen 800/1,000 ms gate.

The semantic boundary is deliberate. Rust returns the complete ranked novel-matching
list rather than truncating at four. Distinct graph matchings can materialize to the
same low-level action, so Python performs action-level deduplication in rank order and
then stops at four. RQ62 owns durable actor integration and the complete unchanged
joint-family gate; RE3 alone does not claim a trained policy or submission strength.

## Scope boundary and next handoff

RE2 establishes exactness over the recorded worlds and normalized/error-phase cases
measured above, including the RE1V counterexample; it is not a universal proof over
malformed inputs or resources too
large to execute. All advertised integer configuration fields except allocation-sized
`boardSize` use arbitrary-width `PyInt`; the suite tests a finite prefix when a horizon
cannot realistically be exhausted. Actual board allocation, step/day indices, unit
positions, inventory materialization, and counts remain host-bounded. The permissive
string-count adapter intentionally covers ASCII decimal syntax, not CPython's Unicode
decimal alphabet or unhashable list-valued action names, both outside the documented
action schema. Framework `TIMEOUT`/`ERROR`/`INACTIVE` agent handling is also outside the
transition-kernel boundary. Extreme custom prices that drive money/rewards to IEEE
infinity are outside the JSON replay protocol: standard `serde_json` has no `Infinity`
number token and serializes those diagnostic fields as `null`; the fixed competition
configuration and every measured RE1 case remain finite.

Errors are part of parity: the suite verifies the same failure phase, official rollback,
and absence of a committed Rust snapshot. Rust clone/commit rollback is additionally
checked directly by a unit regression. A non-integer seed is retained at reset and
fails at the first daily integer mix, and an integer too large for weed probability's
float conversion fails at first end of day, matching the official phases.

Passing RE2 removes the known parity blocker; it does **not** add PPO, behavior cloning,
GPU plumbing, or raw `MOVE`/`PLANT` learning. RQ1 separately proves that a changed
complete matching and its materialized low-level action reach an exact Rust successor.
RQ2 adds common-prefix branch batching and exports padded edge/matching/successor/return
tensors; it demonstrates non-degenerate value labels on one sampled native-seed world,
not learned strength. RQ3 completes the in-process reset/step/fork handoff and confirms
the value opportunity across 53 worlds, but its flattened one-step-successor ranker
fails episode and family holdouts. The next RE handoff is a packed, Markov-complete
encoder that includes R04 continuation state for a permutation-equivariant value model.
The learned surface remains centralized complete-matching/job value behind the
deterministic executor, not low-level farm control.
