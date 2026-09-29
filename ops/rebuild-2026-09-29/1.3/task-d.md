# Task D — strict public tiles

Target: implement and independently check all seven tile tensors, strict
engine-constructor admission and unchanged destination bytes on rejection.
Stopping condition: hand assertions and the complete official key-set scan pass,
the coordinate mutation fails, restoration passes, and Clippy is clean. This
slice does not claim a complete observation encoder.

## Adaptation

- `src/kaggriculture/observe.rs`: strict tile parser; exact constructor key sets;
  board shape, enum, compatibility, boolean, integer, sentinel and finite-value
  admission; both-farm preflight; all seven tile writes in own/rival order.
- `src/kaggriculture/tests.rs`: ten D tests, constructor-shaped fixtures, complete
  output-byte rollback checks, and a streamed `gzip -dc` scan with child cleanup,
  sequential steps, source identity, exact EOF/count and successful-exit checks.
- This receipt and `d1`–`d6` command/log receipts record the actual check episode.
  The parent owns the governing cookbook/docs reconciliation.

`PreparedObservation` retains only the existing immutable snapshot/config.
`TileFields` is one tile's stack scratch, parsed during preflight and writing;
there is no second stored output representation or per-write output allocation.
The writer's explicit unreachable branch asserts the private constructor's
admission invariant. It never unwraps caller JSON or supplies coercive defaults.
No state serialization was added to the live path. Date subtraction uses i128;
scaled values use f64 and one checked f32 cast. Exact channels preserve i64::MAX.
Pending care bonus is a nonnegative engine integer, including large positive
values without saturation, as required by brief Q1 and clarified by the parent.

## Checks

Every command used `bounded.py` (120 seconds, sampled 1 GB process-group RSS
limit) with TMPDIR at `.codex-tmp`, offline Cargo/uv, two Cargo/Rayon threads,
single OpenMP/MKL/test threads. No guard was triggered.

| Receipt | Command / discriminating observation | Result |
| --- | --- | --- |
| `d1-tile-red` | `cargo test --locked --offline kaggriculture::tests::tile_ -- --nocapture` before implementation | 9 semantic failures, independent key-set scan passes; 5.98 s, 529,072,128-byte sampled peak |
| `d2-tile-green` | Same command after implementation | 10 pass; 7.51 s, 677,969,920-byte sampled peak |
| `d3-format` | `cargo fmt --all` | pass |
| `d4-red-transpose-mutation` | Change `y*10+x` to `x*10+y`; run asymmetric test | fails: kind 4 where kind 3 expected; 2.50 s |
| `d5-restored-green` | Restore coordinates; rerun all ten D tests | 10 pass; 5.45 s, 423,755,776-byte sampled peak |
| `d6-clippy` | `cargo clippy --locked --offline --all-targets -- -D warnings` | pass; 1.87 s, 730,333,184-byte sampled peak |

Hand assertions cover the asymmetric plant at `[2][7]` and animal at `[7][2]`,
all 15 floats and seven exact channels, all crops/animals, both perspectives,
empty structures, sentinels, fertilizer flags, signed dates, i64::MAX values,
and clearing previously applicable channels. Wrong key sets, null animal,
missing fields, incompatible structures, invalid shapes/vocabularies/booleans,
fractional or float-represented integers, wrong scalar types, out-of-range
integers, negative counts and invalid sentinels fail without changing any of
the 29 destination fields in either seat.

The durable official scan reads 720 snapshots for each episode 95324500,
95901360, 95921764 and 95990191: 2,880 states / 576,000 tiles. Independent literal
constructor sets match every tile. Counts ordered as EMPTY, LOCKED, WEED, PLANT,
empty COOP, empty PASTURE, GOOSE, COW, SHEEP are:
`[73459, 210575, 10257, 212147, 970, 4593, 3320, 38060, 22619]`.
This checks constructor shape compatibility, not full replay/observation parity.

Post-restoration source SHA-256 at this handoff (E/F will extend these files):

- `observe.rs`: `68cbff4bbd4eae647ccaa9f695cf5f57a4708b224613daeef8ab0d76b1003d6a`
- `tests.rs`: `be2b6b4b0d8ad83ed5ba07d2cb642cda86c6934df159cce5cc97e4c3d48ea298`

## Limits

Actors, private storage/ranks, shops/market/masks and complete row validation are
still E/F work. Full reference corpus qualification and real Python schema
admission remain pending, including the existing R1 corpus quota blocker. No
kernel edit, training, GPU, performance qualification, commit or network action
occurred. Full `rs-prepare` belongs to the parent after the remaining coherent
source slice; D ran focused tests, formatting and all-target Clippy only.
