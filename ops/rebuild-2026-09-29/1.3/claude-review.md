# Task 1.3: Claude review and completion receipt (2026-09-29)

This review covers Codex's handoff at `cac6455`, checked against the brief (including R1–R5 and
Q1–Q6), contract v4 and the plan. Codex's own report is in `results.md`. This file records only
what Claude ran after that handoff. Host: owner's Mac, `CARGO_BUILD_JOBS=2`, offline, with no
training or GPU work.

## Commits

1. `78ab94f`: merged `kg/isaiah-gap-closure` (Task 2.1 `owl.kaggriculture`, Task 1.2 grammar).
   - `Cargo.toml` keeps Task 1.3's engine path dependency and the unified serde_json features.
   - `src/kaggriculture.rs` is removed, because a module file cannot exist at both paths. The
     `pub mod grammar;` line moves into `src/kaggriculture/mod.rs`.
   - Both sides are kept in `docs/rl-api-specs.md` and `cookbook/log.md`.
   - `docs/rules-parity-coverage.md` is merged by hand.
2. `469e8ec`: R1 correction, seeded policy `observation-corpus-v2` (decision text appended to
   `briefs/1.3.md`).
   - Red: the new test `producer_policy_v2_appends_hire_burst_within_order_limit` fails to
     compile against the two-argument v1 `policy_actions`.
   - Green: 9 of 9 `producer_` tests pass.
   - A first variant, with no 16-hand stop, generated exactly 4 qualifying states. It was run
     only in scratch and never committed. The committed variant yields 6.
3. `f15b1ac`: installed the fixtures and fixed two defects found by `just prepare`.
   - The pinned-memory probe must write. On macOS, `torch.empty(pin_memory=True)` succeeds but
     `fill_` raises "missing kernel for mps". Before the fix, 2 tests failed; after it, they skip
     with that reason.
   - The watchdog charges child-group RSS plus caller growth after launch. Under `just prepare`,
     pytest (with torch loaded) already held more than 1 GB, so `run()` killed trivial children.
     This failed 2 custody tests.
     - Red: new test `test_preexisting_caller_memory_is_not_charged_to_the_child` fails.
     - Green: 45 of 45 custody tests pass.
     - Companion test `test_caller_growth_after_launch_still_counts` passes, so growth after
       launch still trips the limit.
4. `rustfmt`-only follow-up commit, then the docs and cookbook commit.

## Oracle generation and qualification (actual)

- `regenerate.py --reference 65f0eac… --output tests/fixtures/kaggriculture/observation-v3` at
  `469e8ec`: exit 0, 43.3 s wall, max RSS 56.9 MB (`/usr/bin/time`). Log:
  `claude-g1-regenerate.log`.
- Sources: 512 records (384 official, 96 seeded, 32 dense). Non-synthetic `actor_gt16_states`
  is 6, against quota 4. All other quotas pass. The shed-order exception comes from dense
  d30/d31, as reviewed.
- Fixture sizes:
  - `states.jsonl.gz`: 202,797 bytes, sha256 `5eaaf797…16d8`. Expanded: 8,410,885 bytes.
  - `reference.f32le.gz`: 578,946 bytes, sha256 `8821cf8b…1ebff`. Expanded: 33,488,896 bytes.
  - Together they total 781,743 bytes compressed, well under the 8 MiB budget, so the
    byte-plane shuffle was not needed.
- `regenerate.py --check`: OK (512 states, 1,024 seats).
- `cargo test … compare_observation_oracle -- --exact`: passes. It compares 1,024 seat rows ×
  8,176 offsets bitwise, with tolerance 0.
- A re-run of the regeneration at `fdada13`, written to scratch, reproduced both `.gz` files
  byte for byte (log: `claude-g2-reproduce.log`).

## Non-vacuity mutations (each restored; restored sha256 verified)

| Oracle | Mutation | Observed failure |
|---|---|---|
| Rust 512-state comparison | encoder writes `[prices, inventory]` instead of `[inventory, prices]` | `record=official:95324500:0 seat=0 offset=889 field=market blocks: actual=0.0025 recorded=1.0` |
| Custody `--check` | one byte flipped in a copy's decompressed reference, then recompressed | `ValueError: reference.f32le.gz: compressed size/hash mismatch` |
| Real-schema Python corpus test | shed ranks reversed, with consistent float rank channels so `check_row` passes | `assert 12 == 1` in `_assert_semantics` (storage_rank) |

## Final checks (actual)

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path engine_rs/Cargo.toml --locked`: exit 0. It
  runs 41 + 18 + 9 + 19 = 87 tests, all passing, none ignored.
- `uv run python scripts/check_engine_trim.py`: `engine trim manifest: OK`.
- `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare`: exit 0.
  - Root Rust: 244 passed, 4 ignored (2 inherited Orbit tests, corpus generation, optimized
    timing).
  - Engine: 87 passed.
  - Python: 1,437 passed, 6 skipped (2 Mac pinned-memory, the rest CUDA/x86 platform cases).
  - Docs lint and docs-fresh passed.
- `uv run pytest tests/kaggriculture/test_observe.py tests/tools/test_observation_oracle_custody.py`:
  100 passed, 2 skipped.
- `git diff --check`: clean.

## Residual risks and open items

- **R4 timing**: not run. The optimized fat-LTO build exceeded 1 GB on the Mac. The pod command
  is in `timing.json`. Snapshot-path reopening stays gated on measured phase costs.
- **Contract v4.1 grammar bridge**: `engine_rs/tests/grammar_kernel.rs` should retire at the
  first production root-to-engine dependency, which Task 1.3 creates. It still exists. The
  move is deferred to Task 1.4.
- **R1 margin**: 6 of 4 states qualify. Any change to the policy, profiles or selection needs
  a regeneration, which fails loudly if a quota drops.
- **Pinned memory and GPU**: pinned-host and GPU binding behavior are unqualified on the Mac.
- **GIL release**: the binding writes caller buffers with the GIL released. Python code that
  touches those tensors concurrently from another thread is outside the contract.
- **Q4 / Task 7.5**: the weed JSON-schema range is still unavailable locally.
- **Brief Task J command**: the brief's `--extra reference` pytest extra does not exist. The
  full Python suite runs under `just prepare` without it.
