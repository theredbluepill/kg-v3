Implemented Task 1.1 on `kg/rebuild-codex`; all changes remain **uncommitted**.

| Required command | Result |
|---|---|
| `CARGO_BUILD_JOBS=3 cargo test --manifest-path engine_rs/Cargo.toml --locked --offline` | **57 passed**, 0 failed/ignored |
| `python scripts/check_engine_trim.py` | **PASS** |
| `uvx --offline --from rust-just just prepare` | **PASS:** root Rust 155 passed/2 ignored; engine 57 passed; Python 769 passed/3 skipped |

Replay parity covers **4 episodes, 2,876 transitions, 2,880 snapshots**. Both the deliberately broken comparator and corrupted snapshot failed before restoration.

Resulting `lib.rs` SHA-256 matches exactly:

```text
c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd
```

Added the 14-file engine package, Python checker, pytest coverage, and receipts. Updated `justfile`, `rustfmt.toml`, parity documentation, and the cookbook Decision/index/log. Root Cargo files, lockfiles, and Rust sources remain unchanged.

**Necessary deviation:** pinned sources contain formatting differences and six Clippy style findings. Two exact formatter exclusions and three engine-only lint allowances preserve their required bytes; authored replay code re-denies those lints. Independent review found no remaining blocking issues.

The [complete report](/Users/poonszesen/kg-v3-codex/ops/rebuild-2026-09-29/1.1/results.md) lists every changed file, command outcome, failure receipt, and deviation. No training, network access, or Git writes.