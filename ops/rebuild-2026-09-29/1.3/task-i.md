# Task I: typed native bridge implemented; integration blocked

The one-shot `owl.rs.encode_kaggriculture_headers_into` writer is implemented
with the exact 29 typed keyword-only NumPy arguments and matching `rs.pyi`.
Task I and Task 1.3 are **not complete**: the real Task 2.1 schema is absent on
this branch, and the qualified 512-state corpus remains blocked by R1 quotas.

## Implementation and scope

Changed `src/kaggriculture/mod.rs`, `src/kaggriculture/buffers.rs`, `src/lib.rs`,
`python/owl/rs.pyi`, `tests/kaggriculture/test_observe.py`, and the independent
`ops/rebuild-2026-09-29/1.3/native_boundary_smoke.py` check script. No schema was
copied, substituted, generated or conditionally imported. The actual schema was
reviewed read-only with `git show kg/rebuild-model:python/owl/kaggriculture/types.py`.

Pinned numpy 0.28.0 default read/write argument extraction calls a panicking
borrow method. Per-argument `from_py_with` helpers retain the required
`PyReadwriteArrayDyn<T>` signature, use fallible typed casting and borrowing, and
map supplied-data errors to `ValueError`. Standard missing/unknown Python
arguments still use the normal signature `TypeError`. Arrays are admitted using
the existing shared Rust shape table: exact dtype, full shape, checked sizes,
C layout, alignment and pairwise byte-range disjointness. The address check runs
before constructing mutable Rust slices; it covers overlapping allocations
exposed through distinct Python/Torch base objects, which the numpy borrow
registry alone cannot identify.

The JSON top-level value must be a nonempty array. Each full TraceHeader is
deserialized separately so late parse/state failures identify their environment.
Typed borrow guards remain alive across the detached operation. Every header is
prepared before any output mutation, followed by infallible direct writes to
caller rows. Preparation may allocate header/config/snapshot scratch; it does
not allocate another observation output array or return new NumPy arrays.

## Executed evidence

Every command below used the common TMPDIR/CPU/offline exports and the 120-second,
1,000,000,000-byte sampled process-group guard. Receipts contain exact argv,
exit status, elapsed time and sampled peak RSS; these are correctness/build
checks, not throughput measurements.

| Receipt | Actual result |
|---|---|
| `i1-native-missing-binding-red` | NumPy-only smoke fails with `AttributeError`: `owl.rs` has no `encode_kaggriculture_headers_into`, before implementation. This is the native API red. |
| `i2-real-schema-collection-block` | Direct real-schema pytest import fails with `ModuleNotFoundError: No module named 'owl.kaggriculture'`. This is a dependency block, **not** a semantic TDD red. |
| `i3-rust-check` | `cargo check --locked --offline` passes. |
| `i4-maturin-build` | `uv run --offline --no-sync maturin develop --locked --offline` passes, 20.66 s, sampled peak 620,363,776 bytes; debug extension installed. |
| `i5-native-boundary` | Initial 54 native boundary cases pass. |
| `i6-native-dense` | Expanded 55 cases pass, adding full 241-actor capacity and dense-to-sparse reuse. |
| `i7-rust-format`, `i9-python-format` | Rust/Python formatting passes. |
| `i8-python-lint`, `i11-python-lint` | Initial style failures preserved; formatting resolves the first group, explicit exception assertion resolves PT017. |
| `i10-clippy` | `cargo clippy --locked --offline --all-targets -- -D warnings` passes, 2.61 s, sampled peak 744,472,576 bytes. |
| `i14-python-lint-green` | Ruff passes for stub, real-schema test and native smoke. |
| `i15-native-retry-green` | Final 55 native cases pass; corrected same-buffer call after a late failure proves native borrow release. Output explicitly reports `schema_checked: false`. |
| `i16-stub-typecheck` | `uv run --offline --no-sync mypy python/owl/rs.pyi` passes. |
| `i13-schema-final` | Real-schema pytest still fails collection for the missing `owl.kaggriculture`; no tests silently skipped or replaced. |

The 55 native cases cover E=1/E=2 and hand-derived values, pointer preservation,
241 actors with exact counts above 2^24, complete dense-to-sparse byte equality,
all 29 right-numel/wrong-shape buffers, four wrong-dtype groups, opposite-endian
i64/f32/f64, Fortran/strided/read-only/unaligned storage, same/mixed dtype aliases
with shared and distinct base objects, accepted adjacent nonoverlapping views,
wrong E, malformed/non-array/empty JSON, a malformed second header, a second
environment state failure, and retry. Every rejected native call compares every
destination's bytes, plus complete backing storage where applicable.

The real-schema module directly imports the actual Task 2.1 types and uses its
declared schema only for allocation. It has explicit native kwargs, all-pointer
checks, independent actor/storage/rank/privacy/mask equations, E=1/E=2,
conditional host pinned-allocation cases, dense reuse, boundary rejection and
rollback cases. Its corpus test calls the real `validate_corpus` custody checker
before streaming all 512 records through the actual schema. These schema/corpus
tests are authored but **unexecuted** because collection is blocked; pinned
memory/GPU behavior and full Rust-to-schema qualification are not claimed.

No full root suite was run in this subtask: H's nonignored missing-corpus test
is expected to fail and is retained. Parent owns J, aggregate preparation,
documentation and cookbook closeout. Source/build ownership was released after
the final native/lint/type checks; only this receipt followed the handoff.
