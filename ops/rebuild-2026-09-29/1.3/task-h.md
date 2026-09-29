# Task H — reconstruction controls implemented; full oracle blocked

Target: reconstruct all 8,176 legacy offsets using only structured tensors,
anchor every retained range and required boundary with independent hand values,
prove discrimination with actual fail/restore runs, and verify facts absent from
the legacy vector. Full qualification requires G's valid 512-record fixture.

## Adaptation and scope

- `src/kaggriculture/oracle_corpus.rs` gains the test-only tensor reconstructor,
  exactly-once offset coverage, bitwise comparison with record/seat/group/offset
  errors, independent recorded constant/duplicate checks, ordinary and byte-plane
  feature readers, and the nonignored `compare_observation_oracle` test.
- `src/kaggriculture/tests.rs` gains five `added_facts_` tests (authored by the
  parent, checked and corrected in this H slice).
- This receipt and `h1`–`h18` command evidence, including 16 `h13-m*` mutation
  receipts, record this check episode. The parent owns cookbook/docs updates.

The reconstructor takes only `&ObsRowMut`; it never reads Game, headers or
reference features. Every output offset is written exactly once, including the
positive-zero range and constants. Duplicate writes and unwritten zero holes
have explicit failing controls. Its arithmetic follows the reviewed table and
the pinned reference read via `git show`; no reference encoder was copied.

The future full comparison first runs the source-bound Python custody validator
with `uv run --offline --no-sync`, then streams one state and each seat's feature
block. It validates the oracle-only legacy domain, native row contract, recorded
constants/scaled duplicate positions/inventories/counts/clocks, every f32 bit,
and exact EOF. Byte-plane decoding uses bounded chunks and owned temporary files;
tests cover both formats, early EOF handling, signed zero, exact bits and an
8,195-value decode crossing the 8,192-value block boundary with a short tail.
No additional legacy restriction enters the production encoder.

## Actual checks

All Rust build/test commands used `bounded.py`, the standard offline/resource
exports and worktree `.codex-tmp`. Across 33 H command receipts, the longest
command was 3.44 seconds and the largest sampled process-group RSS was
640,663,552 bytes. No 120-second/1-GB guard fired. There was no training/GPU run.

| Receipt | Observation |
| --- | --- |
| `h1-semantic-red` | Against an all-zero reconstructor stub, fixed hand-vector and legacy day/sentinel assertions fail; comparator control passes (2 fail, 1 pass). |
| `h2-missing-qualified-corpus` | Full comparison explicitly fails because the qualified fixture is absent. This is a dependency failure, not semantic red. |
| `h3-reconstruction-green` | First compile caught a shared-reference borrow error; fixed before the next check. |
| `h4-reconstruction-green` | Three initial reconstruction controls pass. |
| `h5-stream-controls-red` | No-op recorded checks and byte-plane decoder fail their independent assertions (2 fail, 5 pass). |
| `h6-stream-controls-green`, `h7-reader-formats` | Seven controls pass; ordinary/shuffled reader smoke also passes. |
| `h9-added-facts-first-check` | Four added-fact tests pass; rank-pair whole-row comparison fails because the test normalized exact ranks but omitted their two scaled duplicates. This was a test expectation defect, not an encoder defect. |
| `h10-added-facts-corrected` | Explicit scaled-rank assertions and normalization repair the test; all five pass. Literal MILK rank anchors were included in the final check. |
| `h12-clippy` | One style finding (`is_multiple_of`) corrected; no lint allowance added. |
| `h13-m*` | All 16 temporary input/formula mutations produce the intended fixed-hand assertion failure. The mutation manifest confirms byte-exact source restoration. |
| `h15-restored-reconstruction-green` | Final eight reconstruction/stream/coverage controls pass. |
| `h16-final-added-facts-green` | Final five added-fact tests pass. |
| `h17-final-clippy` | All-target Clippy with `-D warnings` passes. |
| `h18-full-oracle-still-blocked` | Final full comparison still explicitly fails on the missing qualified corpus; test is neither ignored nor skipped. |

Formatting passes are `h8`, `h11`, and `h14`. The hand vector is a separately
written literal expectation, not a generated reference fixture. It checks all
8,176 values plus named boundary assertions at 615/616/679/680/871/872/906/907/
959/960/1024/1027/3827/5273/8165/8175. Additional anchors distinguish the legacy
`max(1, Ep/D)` day denominator, sentinel plant age, saturation in legacy packing,
and ever-fertilized from current-fertilized state.

Actual temporary mutations and first mismatching offsets:

| Mutation | Offset |
| --- | --- |
| summary bank | 3 |
| tile category | 16 |
| first actor position | 616 |
| first inventory prefix | 871 |
| storage scaling | 872 |
| market price | 906 |
| swapped shop slots | 960 |
| full actor count | 1025 |
| maintenance integer | 1031 |
| last rival actor position | 5271 |
| last own actor inventory | 8164 |
| rule suffix | 8172 |
| ever/current fertilizer confusion | 18 |
| dropped range signed zero | 907 |
| availability constant | 1024 |
| investment constant | 8165 |

`h13-mutations.json` preserves exact removed/inserted snippets, individual receipt
names, mutated hashes, expected offsets, exit status 101 and observed semantic
failure for every case. Both its original/restored hashes match. Restoration
was followed by final formatting and the green checks above.

The added-fact tests show d30/d31 have equal exact counts and reconstructed legacy
vectors while farmer/shed MILK ranks move 1→11/12 (and scaled duplicates change);
normalizing only those rank representations restores every row byte. They pin
new config/rival-hire fields absent from the seat-zero legacy vector; compare
27 hire cases against an independent Python exact-integer loop for both roles
and seats; admit i64::MAX counts/config, wide overfull shed totals, noncanonical
legal quadrants and exact values above 2^24 outside oracle admission; reject the
next positive hire-cost overflow; and verify remaining transitions against
actual Ep=1/2/5 engine termination.

## Qualification boundary and handoff

G's actual 512-input generation has 0 non-synthetic states above 16 actors,
against the unchanged requirement of at least 4. No qualified manifest or
recorded reference-feature stream exists. Therefore no 512-state/1,024-seat
reference comparison, full H acceptance, or complete Task 1.3 qualification is
claimed. Hand controls and negative mutations support the implemented mapping,
not equivalence to an unrecorded full oracle. The future full stream path is
compiled and its readers are tested, but that complete path cannot run yet.

Final source SHA-256 at H source/build release:

- `oracle_corpus.rs`: `09ea83031649662c198f047fedbd06e9d9e5bbe8135a7b6b566a5b3abb168a76`
- `tests.rs`: `56dcbb176d927147a2d6d05a42050ab862aeee8e310bc879263840a2a0de58ef`

No kernel, production policy/model/reward/grammar, Git state or other worktree
was modified. Source/build ownership is released to the parent for I/J and
aggregate preparation; this H slice ran focused tests/formatting/Clippy only.
