Implemented Task 7.3’s binding-independent scope. Changes remain uncommitted.

[Full handoff report](/Users/poonszesen/kg-v3-t73/ops/rebuild-2026-09-29/7.3/results.md) contains the red/green evidence, every mutation’s pointer/transition, exact allowlist and deviations. [File inventory](/Users/poonszesen/kg-v3-t73/ops/rebuild-2026-09-29/7.3/changed-files.md) lists every changed file with its purpose.

All required commands passed:

| Command | Actual result |
|---|---|
| `cargo test --offline --lib kaggriculture::replay_export` | 12 passed |
| `cargo test --offline` | 266 passed, 4 ignored |
| `cargo test --locked --offline --manifest-path engine_rs/Cargo.toml` | 69 passed |
| `uv run --offline python scripts/check_engine_trim.py` | Exit 0 |
| `uv run --offline maturin develop` | Exit 0, debug build |
| Requested focused pytest command | 123 passed, 5 skipped |
| `uvx --offline --from rust-just just prepare` | Exit 0; Python 1,737 passed, 16 skipped |

Round trips passed for **four fixtures × 719 transitions**, the real pinned framework’s nine-transition game at seed `2**80 + 19`, canonical byte equality, and seeds exceeding 100 bits. Fixture comparison times were 11.37, 14.08, 15.64 and 16.69 seconds. All four framework source hashes matched.

The foreign-only allowlist covers host `info` except seed, per-seat `info`, measured `remainingOverageTime`, and framework `actTimeout`/`runTimeout`; each justification is recorded in the report.

The necessary scope deviation was a test-only deadline-isolation repair: longer replay tests exposed four existing custody tests sharing a deadline initialized during collection. Production timeout behavior is unchanged.

Remaining limits: five explicit Task 1.4 integration skips; evaluation-loop wiring and pod-scale eight-game evaluation await that binding. Recorder overhead and exhaustive framework-configuration coverage are unmeasured. Vendored engine bytes remain frozen; only its updater-generated manifest changed.