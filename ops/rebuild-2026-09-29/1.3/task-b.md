# Task B — typed observation buffers

Target: admit all 29 named flat fields before mutation, expose both seats through
safe serial/Rayon typed views, and publish reusable staging only to equal-E
validated destinations. Stop after boundary correctness; game/config admission
and observation writing belong to C onward.

Implemented `src/kaggriculture/{mod,buffers,observe,config,tests}.rs` and the root
`src/lib.rs` module registration. No engine bytes changed. All row, flat and owned
fields are explicit Rust members. The sole Rust shape table is visible only
inside the Kaggriculture module for later NumPy checks. Every shape's row,
element and byte counts (including aggregate bytes) are checked before the first
staging allocation; allocation reservation errors are returned. Immutable E and
private vectors preserve admission. Safe `as_chunks_mut` conversions assert
empty remainders; serial zips assert equal lengths, Rayon uses `zip_eq`.
Publication checks E equality in release before all-field copies. Row `clear()`
writes positive zeros/false without allocating output storage.

Actual bounded checks (each `.json` records argv, elapsed time and sampled
process-group RSS; every build/test inherited the brief's offline/thread bounds,
`TMPDIR=.codex-tmp`, and `RUST_TEST_THREADS=1`):

- `b1-red-missing-api`: minimal overflow test fails compilation on missing API.
- `b2-red-full-boundary-tests`: complete boundary tests fail on missing APIs.
- `b3-first-green-compile`: initial 10 tests pass; 5.96 seconds, 697,548,800-byte
  sampled peak RSS.
- `b5-red-length-check-disabled`: temporarily disabling the length check makes
  `shape_every_field_rejects_short_and_long_without_writes` fail with
  `accepted tile_kind long=false`; original source restored in `finally`.
- `b6-green-boundaries`: 11 tests pass after row-clear test addition.
- `b7-clippy` / `b8-clippy-cleanup`: test-only no-op casts/arithmetic caught and
  removed; `b9-clippy-green`: root all-target Clippy with `-D warnings` passes.
- `b10-format`: root formatting passes; `b11-green-final`: all 11 boundary tests
  pass on the final B source.

Tests cover every field one short/long (58 cases), zero E, wrong E, usize and
byte-capacity overflow, 251-frame can_act, both directions of unequal-E publish
with all destination bytes unchanged, every element's environment/seat/field
order through both serial and two-thread Rayon views, stable pointer/capacity
for every staging vector across eight writes/publications, all-field clearing,
exact contextual error display, and the two-private-state header helper.

API handoff: `fresh_header`, `field_bytes`, `all_bytes` and `row_bytes` are test
helpers; `encode_header` waits for a real encoder. `config.rs` is documentation
scaffolding only. There is no false successful encoder, public Python binding,
config validator, state preflight or observation semantics claim in B. Parent
owns combined documentation/cookbook and final `rs-prepare` / `prepare`.
