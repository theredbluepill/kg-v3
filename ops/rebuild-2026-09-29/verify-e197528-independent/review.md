# Independent verification — kg/merge-1-3

Reviewed `e197528820ab7cfb429e21259000957370abf1c6` against `b51b0c0...HEAD`. Actual parents: integration `b51b0c0382a13a72854860ae17d7798f0875e4ee` and Task 1.3 `dc6b2005516327dbe7b5bfd05ef8da4c3b0c6f05`.

## Finding

**P2 — Capture all live engine inputs in oracle source custody.** [scripts/kaggriculture_observation_oracle/regenerate.py:975](/Users/poonszesen/kg-v3-m-1-3/scripts/kaggriculture_observation_oracle/regenerate.py:975)

`ENGINE_PATHS` includes the trim manifest, engine `src/lib.rs`, and engine lockfile, but omits `engine_rs/src/py_random.rs` and `engine_rs/Cargo.toml`. `source_snapshot()` only hashes the listed paths; hashing the manifest does not verify the live files whose hashes it contains. The producer compiles and uses `py_random.rs` (engine `lib.rs:20–22`). A dirty change to either omitted input, before or during regeneration at unchanged HEAD, is neither recorded nor rejected. Generated states/build behavior can therefore differ without a corresponding source-identity change.

Reproduction:

- On scratch copies of actual repository bytes, actual `source_snapshot()` and `check_source_snapshot()` with real git commands accept separate comment mutations to both omitted files. No mocks were used in this capture/recheck probe. Original, mutated, and restored SHA-256 values are in [real_source_snapshot_receipts.json](mutations-python/real_source_snapshot_receipts.json).
- The existing regeneration test harness, with producer execution mocked, installs the final output after either dependency changes. Two expected-rejection probes fail. Adding the two paths on scratch yields **2 failed → 2 passed → 2 failed after exact restoration**. [Source-custody review and receipts](mutations-python/review.md).

**Fix:** capture and recheck all live retained engine/build inputs, at least these two paths, or verify all retained manifest file hashes at capture and relevant rechecks. Add drift regressions for both inputs and ensure identity metadata records the complete set. Keep the cookbook adaptation record current when implementing the correction.

This defect is in code introduced by the three-dot diff; no merge-specific loss caused it. The checked-in corpus passed its current validation and bitwise comparison. The evidence does not establish that the existing frozen data is wrong, and the mocked-install probe is not a real regeneration run.

## Requested checks

| Command | Fresh result |
|---|---|
| `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **69 passed**, 0 failed (41 unit + 9 RNG + 19 replay; 0 doc tests) |
| `cargo test` | **254 passed**, 0 failed, **4 ignored** |
| `uv run python scripts/check_engine_trim.py` | **PASS** |
| `uv run pytest tests/kaggriculture tests/owl tests/scripts tests/tools -m 'not slow' -q` | **1,618 passed**, **7 skipped**, 0 failed/errors, via 45 file shards |
| `uv run mypy python/owl scripts` | **PASS**, 62 source files |

The initial combined pytest attempt hit the bounded runner's 1,000,000,000-byte process-group RSS limit at sampled 1,007,435,776 bytes, after 9.45 seconds. All 45 file shards then completed below the guard, with 1,625 unique JUnit cases and no duplicate counting. [Python totals and exact skip reasons](python-summary.json). The seven skips are the native grammar table binding, native Kaggriculture evaluation environment, two pinned-memory CUDA cases, two FlashAttention/CUDA cases, and one x86 quantized backend case.

The native Python extension was freshly rebuilt with `uv run maturin develop --locked --offline` before sharded testing; build passed. Checks used offline mode, 2 Cargo/Rayon workers and 1 OMP/MKL/Rust test thread. Rust: `1.97.0-nightly (e9e32aca5 2026-04-17)`; uv: `0.10.9`. Per-command logs and JSON receipts are in this directory. The requested commands were run independently of pre-existing `verify-merge-1.3` artifacts.

## Parent preservation and documentation

- All Task 3.x model registration, training, evaluation, configuration, startup guard and cuBLAS-only implementation/config/test files retain integration-parent bytes. The only change to a pre-existing Python interface is additive observation declarations in `python/owl/rs.pyi`.
- All Task 1.3 implementation files retain `dc6b200` bytes. No Task 1.3 test name is missing.
- All **nine** kernel acceptance/replay test names formerly in `engine_rs/tests/grammar_kernel.rs` survive in root `src/kaggriculture/grammar_kernel_tests.rs`. The other nine tests removed from the engine execution count were duplicate inclusions of root grammar tests; their root source and registration remain intact. Engine **87 → 69** and root **164 passed/2 ignored → 254 passed/4 ignored** reconcile.
- Running the committed retirement generator against a scratch copy of integration's manifest reproduces HEAD and Task 1.3 byte-for-byte: SHA-256 `b2d1892a15baf5653e493fdb885c5f2baac7a21095a87eb95d9730782ad9272f`.
- Both parents' cookbook log headings survive. Current docs explain the root engine dependency, L4 feature-unification fix, bridge retirement, historical and merged test counts, and remaining GPU/timing gaps. No additional documentation finding was established. The custody finding above limits the source-identity claim.

Complete inventories, relocation diff, generator replay and doc review: [parent-review/review.md](parent-review/review.md).

## Scratch mutation coverage

- **31 native source mutants** compiled and failed their intended oracle/guard tests. Families: lengths/overflow/publication/clearing; config bounds/finiteness/hire cost; coordinates/privacy/order; row checks/snapshot count; relocated grammar; reconstruction coverage/bitwise comparison/stream lengths; corpus input policy and domain; action frame masks; L4 decimal deserialization. [Full matrix](mutations-native/summary.md).
- A coherent exact-plus-scaled market corruption passed custody and row checks, then failed the independent 512-state reference comparison at `official:95324500:0`, seat 0, legacy offset **889**. Thus the corpus check discriminates beyond internal consistency.
- **16 Python source guard mutants** failed their focused tests, with **26 baseline and 26 restored passes**. Families: bytes/header order/schema/identity/quotas/byteplanes; source/archive/recorder custody; memory baseline/growth/deadline; retired bridge and authored set; safe CUDA skip admission. [Full matrix](mutations-python/review.md).
- **8 additional native Python boundary input mutations** were rejected: wrong shape/dtype/endianness/stride/writability/alignment, aliasing through Torch storage, and invalid late-environment state. Every output byte and pointer stayed unchanged, and the restored valid write matched baseline. [Boundary receipts](boundary-probes.json).
- The two omitted-dependency drift mutations survived and produced the P2 above.

All scratch source mutations and the temporary candidate fix were restored byte-for-byte. The restored complete scratch Rust suite also passed **254 tests with 4 ignored**, after copying the required local ignored Orbit fixture files with matching hashes. No GPU/pinned allocation, training, performance claim, or corpus regeneration was performed in this verification.

## Workspace custody

All **1,629 tracked files** match the initial SHA-256 inventory; `git status --short --untracked-files=no` is empty. Evidence is untracked under this directory; scratch copies are ignored under `.codex-tmp`. The pre-existing untracked `ops/rebuild-2026-09-29/verify-merge-1.3/` directory was preserved.

VERDICT: REJECT
