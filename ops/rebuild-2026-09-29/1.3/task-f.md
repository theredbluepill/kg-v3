# Task F — complete context, diagnostic rows and staged publication

Completed local implementation slice on 2026-09-29. All 29 named native fields now
have writers. This does not complete Task 1.3: frozen reference comparison, actual
merged Python schema/binding and bounded optimized timing remain later tasks.

## Mechanism, inputs and stopping condition

Question: do shops, exact markets and potential-frame masks preserve their public
meaning, and can a failed second candidate avoid publishing any output? Inputs:
reviewed Task F, contract v4/schema 3, pinned public kernel process_market, B–E
named buffers/writers and synthetic hand-checked TraceHeaders. Discriminating
observations: literal shop type/slot order, exact signed market integers, order
limits independent of cash/hires, raw-kernel three-of-four order execution,
positive-zero padding, rejected diagnostic corruptions, and unchanged published
bytes for both environments after env 1 fails. Stop after semantic red→green,
whole Kaggriculture module tests and all-target Clippy. No lifecycle, model,
training, GPU, network, engine-byte, dependency or schema-copy change.

## Changes

- `src/kaggriculture/observe.rs`: recognizes the eight pinned shop names in native
  order, permits duplicate types, preserves shop slots/masks and clears padding.
  Requires every pinned market product, rejects extra keys, and admits only actual
  JSON integer representations fitting i64. Both exact market channels remain
  signed, matching the contract and shared schema; no artificial stock, price or
  normalization bound is added. Scaled market fields use f64 division then f32.
- `still_playing` is true for all returned explicit observations, including a
  terminal snapshot. `order_limits` remains the configured M. `can_act` follows
  checked own actors + M + STOP, with 5 and 252 frame boundaries exercised.
- Added diagnostic `check_row`, separate from release preparation/write. It checks
  every categorical range, even under false masks; contiguous actor/shop masks;
  slot/cell/role correspondence; exact count/rank relationships and unique gap-free
  positive ranks; applicable tile sentinels and flags; integer and float padding;
  positive-zero private rival/reserved channels; finite floats/banks; and exact
  side-tensor agreement with derived scales, global context and potential frames.
  Animal pending-care scaled values must be nonnegative and lie on the integer/8
  grid. Hire checking stops at the first overflow and zero multipliers do not
  iterate to a large count.
- `src/kaggriculture/mod.rs`: exports `check_row` and updates the scope comment.
- `src/kaggriculture/tests.rs`: five context families, one 72-case corruption
  family and two transaction/reuse families. The local encoding helper runs
  `check_row` on every successful seat, so all existing semantic fixtures also
  exercise complete row checking. Successful serial, two-worker and reused
  published buffers are checked again after copying.
- Existing `buffers.rs` required no modification. Test-only batch composition
  checks staging/header E equality before candidate work, uses admitted serial
  chunks and Rayon `zip_eq`, gathers results in environment order, and calls one
  `publish` only after every candidate succeeds. It neither changes nor claims
  game/seed/terminal rollback; that belongs to Task 1.4.

The prepared-market writer uses explicit match on `get(...).and_then(as_i64)`;
its unreachable branch identifies the private preparation invariant. It does
not `expect` caller JSON. Rank validation takes arrays of exactly twelve counts
and ranks, so its zip cannot truncate.

## Checks and receipts

All runs exported:

```sh
export TMPDIR=/Users/poonszesen/kg-v3-observe/.codex-tmp
export CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RUST_TEST_THREADS=1
```

Each command was run as
`python3 ops/rebuild-2026-09-29/1.3/bounded.py --name NAME -- COMMAND`.
Matching logs/JSON preserve argv, status, timing and sampled process-group RSS.
All stop reasons are null under the 120-second/1-GB bounds. Highest sampled RSS
was 642,236,416 bytes. Sampling is every 0.1 seconds; this is not an unsampled
peak-memory or end-to-end throughput claim. Tests keep at most two live games.

| Receipt | Result | Wall seconds | Sampled RSS bytes |
| --- | --- | ---: | ---: |
| `f1-context-red` | Five context tests fail against the prior writer: shops, markets/admission, masks/order context, terminal rows | 2.833 | 520699904 |
| `f2-context-green` | All five context tests pass after implementation | 2.703 | 518701056 |
| `f3-check-row-red` | No-op diagnostic accepts all 66 initial corruption variants; test fails | 4.276 | 642236416 |
| `f4-complete-green` | Full B–F module: 54 passed after diagnostic implementation | 9.089 | 566755328 |
| `f5-format` | Explicit mod/observe/tests Rust formatting passes | 0.212 | 26132480 |
| `f6-clippy` | All-target Clippy passes | 1.360 | 633077760 |
| `f7-pending-bonus-red` | Added integer-grid corruption tf14=0.1 is accepted; sole surviving case 66 fails | 4.278 | 637648896 |
| `f8-format` | Explicit mod/observe/tests formatting after repair passes | 0.107 | 688128 |
| `f9-final-module-green` | Full B–F module: 54 passed, including 72 checker corruptions | 7.952 | 500318208 |
| `f10-final-clippy` | `cargo clippy --all-targets --locked --offline -- -D warnings` passes | 1.050 | 587857920 |

Test invocations use `cargo test --locked --offline` with the named
`kaggriculture::tests::context_`, `context_check_row` or full
`kaggriculture::tests::` filter. Formatting uses `rustfmt --edition 2021` on the
three changed Rust files. Final handoff hashes are in `task-f-source.sha256`.

Context positives include `[PIZZA_SHOP,BAKERY,PIZZA_SHOP]` → `[5,0,5]` plus all
eight types, raw market -12345/7777, both i64 extrema and a negative price,
custom globals inherited from C's explicit all-channel assertions, low money and
100/101 prior hires with different seat actor counts, and a raw kernel queue
where only three of four HIRE entries execute (cost 1+1+2, bank 2996). Episode
lengths 1 and 2 each reach a terminal snapshot after one transition while encoded
`still_playing` remains true. Fractional/float-valued 1.0, wrong type, out-of-i64,
missing product, extra product and unknown shops reject without publication.

The transaction tests stage two environments with a fractional WHEAT price in
env 1. Both serial and two-worker execution report env 1 and preserve every
published byte in both rows of both environments. Correcting the price yields
identical serial/two-worker bytes. E=1 staging → E=2 publish and mismatched
header/staging counts reject without changing output. Dense→sparse reuse matches
fresh sparse encoding over every field, including shop/actor/mask padding.

## Remaining boundaries

`check_row` is an invariant diagnostic for the represented row, not proof that an
arbitrary externally assembled row came from the engine or that discarded source
precision can be reconstructed. Source admission validates the exact integer
bonus before scaling; diagnostic grid checking does not infer a source integer
upper limit from a rounded float32. Frozen-oracle losslessness, actual Python
`check_contract`, NumPy alias/shape admission and optimized phase timing are not
claimed here. Full-root preparation and cookbook note/index/log reconciliation
remain parent-owned combined checks. No game/seed rollback, submission strength
or learner-throughput claim follows from the two-environment buffer transaction.
