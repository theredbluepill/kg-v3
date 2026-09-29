# Task E — actors, storage and private insertion order

Target: fill all seven actor tensors, exact own storage/ranks and remaining
public/player features; reject malformed state before either seat is published.
Completion: rank/count assertions, both-seat privacy with positive controls,
full seat-swap symmetry and dense-to-sparse reuse pass; rank mutation fails and
restoration passes. This is an observation-domain check, not Task 1.3 completion.

## Adaptation

`src/kaggriculture/observe.rs` now validates both farms' actor coordinate pairs,
matching inventory counts, known item/crop keys, nonnegative exact counts and
unique known quadrants. Public context already enforces the 241-actor capacity.
The actor/storage writer receives public farm references and **only** the
requesting seat's `PrivateState`. It preserves engine actor order, IndexMap ranks
(including zero-valued keys), exact i64 counts and f64 banks, and writes own
private features only. Padding/reserved/rival-private fields remain positive
zero from the shared row clear. Sum(shed) and room use i128, with room alone
floored at zero; no count, bank or scaled-feature ceiling is added.

`src/kaggriculture/tests.rs` adds 12 tests. The independent literal Item list is
cross-checked against the kernel's public PRODUCTS/ANIMAL_NAMES. Actor fixtures
exercise all 241 positions on both farms, asymmetric cells/roles, >2^24 and
i64::MAX counts, empty present farmers and absent keys. Actor and shed tests
separately check different-value replacement preserves the complete rank vector,
while remove/reinsert preserves counts and moves/compacts ranks. The native
initial all-zero shed must have ranks 1..12.

Player assertions cover all 44 channels, noncanonical quadrant order, exact
sub-f32 bank differences, empty/partial/full/overfull shed room and the literal
12*i64::MAX shed sum. Invalid coordinates, inventory counts, actor 242, negative
counts, unknown keys/crops and duplicate/unknown quadrants preserve all 29
fields for both destination seats. Privacy mutates either seat's shed, seeds and
inventory values and key order; the other entire row stays byte-identical while
its owner's row changes. Public coordinate controls change both rows. Complete
farm/private swaps exchange rows exactly. Dense-to-sparse writes reuse storage
and clear all absent actor/private channels.

## Actual bounded checks

All commands used `bounded.py` with the 120-second/1-GB sampled process-group
limits, TMPDIR at `.codex-tmp`, offline Cargo/uv, two Cargo/Rayon threads and
single OpenMP/MKL/test threads. No guard fired; at most one live Game existed in
these diagnostics. No engine bytes, model, training or GPU code changed.

| Receipt | Check | Result |
| --- | --- | --- |
| e1-actors-red | actors_ against D | 5 semantic failures |
| e2-storage-red | storage_ against D | 5 semantic failures |
| e3-privacy-red | privacy_ against D | 2 failures on missing changed-row controls |
| e4-actors-green | actors_ after implementation | 5 pass; 2.42 s, 529,858,560-byte sampled peak |
| e5-storage-green | storage_ plus stronger replacement tests | 5 pass; 2.81 s, 497,418,240-byte sampled peak |
| e6-privacy-green | both-seat privacy/key order/public controls/swap | 2 pass |
| e7-format | cargo fmt --all | pass |
| e8-clippy | all-target Clippy with -D warnings | pass; 1.55 s, 593,346,560-byte sampled peak |
| e9-red-reversed-ranks | change both exact ranks from rank+1 to 12-rank | 2 named rank tests fail; source restored in finally |
| e10-restored-module-green | complete Kaggriculture test module | 46 pass; 6.78 s, 444,416,000-byte sampled peak |

The privacy red's huge debug byte-vector lines are omitted from its compact log;
that log records the SHA-256, byte count and `.codex-tmp` path of the preserved
complete output. Assertions now avoid printing entire vectors on that failure.

Source SHA-256 at E handoff:

- observe.rs: `c4948c84397b6a85e15acffc093bb2bd8e08c7fb76f9b27b411c7b015ad5c15d`
- tests.rs: `0a1e5f5d09379796590a9c2547e9ccf1f152f81202cdb616e084f5fcac1757cf`

## Remaining boundaries

Task F still owns shops/market/masks and comprehensive output checking. Corpus,
real Python schema, full root/engine preparation and optimized timings remain
pending. i64-exact observation encoding of very large aggregate inventory does
not qualify unchanged engine transition arithmetic; Task 1.4 owns that admission.
No commit or network action ran. Parent owns combined cookbook/docs and the final
preparation checks.
