# Task C — checked configuration and public hire context

Completed local implementation slice on 2026-09-29. This is not a completed
observation encoder or Task 1.3 qualification. Parent owns durable note/index/log,
combined preparation and the later task handoffs.

## Mechanism and stopping condition

Question: does a wrapper bind checked public rules to its engine, reject
nonrepresentable context before publication, and compute each role's next hire
cost from that farm's public hire count? Inputs: pinned `Config`/`TraceHeader`,
root `src/kaggriculture/{config,observe,tests}.rs`, typed B buffers. Distinguishing
observations: exact integer context, hand-computed floats, the six literal hire
multipliers, engine bank deltas, zero-multiplier huge counts, and unchanged
poisoned destination bytes on rejection. Stop after config/hire filters, semantic
mutation red/restored green, B/C regression and all-target Clippy; no training,
GPU, network, model, private engine accessor or engine-byte edit.

## Implementation inventory and limits

- `config.rs`: strict board/orders/product/scalar/extra/default-market admission;
  checked exact i64 values; nonnegative numeric finite weed without a unit cap;
  checks f64 and narrowed f32 finiteness without clipping large finite channels.
  Positive hire multipliers precompute exact BigInt Fibonacci products until the
  first scaled f32 overflow, keeping the last supported index. Zero multiplier
  selects a constant-zero branch without count-dependent iteration.
- `observe.rs`: minimal `ObservationGame` with private Game and immutable shared
  `Arc<ObservationConfig>`, public stepping forwarder, raw-header clock checks,
  two farms/private states, one snapshot per `prepare()`, context validation and
  write after preparation. Config and Game are constructed from the same input;
  no `game_mut`, reset, reward or seed-stream allocation is added.
- C currently fills exact `banks`, `player_features` channels 0, 1, 9, 10,
  `global_features`, `globals_int` and `order_limits`. All rows are first cleared
  by B's named-buffer helper. Tile/actor/storage/shop/market semantics and masks
  remain for D–F. These partial rows must not be described as complete valid
  schema batches. The private validation seam is ready for those additions.
- `mod.rs`: exports the checked configuration, wrapper, prepared snapshot and
  writer interfaces. `tests.rs`: nine config test families and four hire test
  families, with shared encode/rejection helpers for later tasks.

## Correction to the reviewed brief

Pinned `engine_rs/src/lib.rs` has no `Display` implementation for `PyInt`; its
numeric methods at lines 364–411 are private. The public `Serialize` implementation
at lines 426–433 builds a JSON Number from the exact BigInt decimal. The brief's
Display-based admission therefore cannot compile. The parent accepted the narrow
correction: at constructor admission only, `serde_json::to_string(&PyInt)` then
checked `parse::<i64>()`. There is no float intermediate, Debug parsing, new
accessor or live state serialization. Starting money and every other exact config
integer at 2^63 are rejected; i64::MAX remains exact in side tensors. This is an
implementation correction from inspected source, not an invented owner decision.

Claude Q4 remains scoped as reviewed: no locally available pinned official JSON
schema was introduced; numeric nonnegative finite weed follows the kernel and
contract, with schema-range parity still a later Task 7.5 item. Huge imported
aggregate transition safety remains Task 1.4, separate from encoding.

## Actual checks

Every build/check used these shell exports:

```sh
export TMPDIR=/Users/poonszesen/kg-v3-observe/.codex-tmp
export CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RUST_TEST_THREADS=1
```

Each command below was wrapped by
`python3 ops/rebuild-2026-09-29/1.3/bounded.py --name NAME -- COMMAND`.
The matching `.json`/`.log` files preserve full argv, status, wall time, sampled
aggregate process-group RSS and stop reason. Limits were 120 seconds/1 GB; every
receipt has `stop_reason: null`. Peak sampled RSS across C checks was 758,677,504
bytes; sampling is 0.1 seconds and does not establish unsampled peak memory.

| Receipt | Command/result | Wall seconds | Sampled RSS bytes |
| --- | --- | ---: | ---: |
| `c1-red-missing-api` | `cargo test --locked --offline kaggriculture::tests::hire_cost`: expected unresolved new APIs, exit 101 | 1.254 | 354762752 |
| `c2-hire-green` | same filter: 4 passed | 4.584 | 654311424 |
| `c3-config-green` | `cargo test --locked --offline kaggriculture::tests::config_`: 9 passed | 0.529 | 60669952 |
| `c4-format` | explicit four-file Rust formatting | 0.107 | 688128 |
| `c5-clippy` | all-target Clippy: two style findings, corrected (checked-product boolean; test float literal precision) | 1.682 | 758677504 |
| `c6-red-rival-hire-mutation` | hire filter after replacing the per-role count with own count: 2 failed, 2 passed | 2.504 | 456982528 |
| `c7-format` | explicit four-file Rust formatting after restoration | 0.106 | 704512 |
| `c8-bc-green` | `cargo test --locked --offline kaggriculture::tests::`: all 24 B/C tests passed | 2.093 | 417300480 |
| `c9-clippy-green` | `cargo clippy --all-targets --locked --offline -- -D warnings`: passed | 1.475 | 724828160 |

The semantic mutation was exactly
`config.prepared_hire_cost(farm.hires_today)` →
`config.prepared_hire_cost(own.hires_today)` in `write_seat`, then restored.
`hire_cost_uses_each_public_count` detected 0.000175 versus expected 0.00028;
the independent engine-delta sequence also failed. Thus the red record is not
limited to a missing-import failure. Final source SHA-256 values are recorded in
`task-c-source.sha256` before the D handoff.

Config cases cover all six prescribed profiles, episode lengths 1/2 and shorter
than a day, framework extras plus daily RNG boundary, board 9/11, orders 0/11,
D*M at 240/241 and checked multiplication overflow, positive-only episode/day/
capacity/interval scalars at 0/-1, orders at 0/11, negative starting money/multiplier,
unknown extras, both config and explicit
market overrides, all exact integers beyond i64 and accepted i64 maxima, weed
numeric/type/range errors, finite-f64-to-infinite-f32 overflow, large finite
scaled values, inconsistent header clocks, wrong farm counts, nonfinite banks
and both seat-role orderings. Rejections compare all poisoned named-buffer bytes.

Hire evidence includes literals `[1,1,2,3,5,8]` at multiplier 7 against six actual
kernel HIRE bank deltas, independent public counts 4/5 in both seat perspectives,
i64::MAX hires with multiplier zero, and the positive-multiplier finite-prefix
boundary. Independent integer arithmetic pins multiplier 7: index 206 costs
57023591856623591583060292457594532272891566 (scaled finite); index 207 costs
92266109784618691863425340395105880651619197 (scaled f32 overflow). Error identifies
`player_features[1,10]`, count 207, and last supported index 206.

No full-root suite, `rs-prepare`, Python/schema check, frozen oracle, optimized
timing, game/seed rollback or full-buffer publication transaction is claimed by
this slice. Parent runs the combined required checks after D and remaining work.

Independent read-only C review found no blocking logic issue. It identified two
nonblocking test gaps: no isolated private-state-count rejection case, and no
prepared-snapshot immutability assertion after stepping the bound game. Ownership
enforces the latter; the count guard exists separately. These are recorded as
coverage limits, not claimed test results.
