# Independent lane B re-verification

Reviewed detached HEAD `94f778dc867e5402c5cd37fc3b117f7928870ab7`, range `4fdb526..94f778d`, against the three P2 findings in `ops/rebuild-2026-09-29/codex/verify2-lane-B-r1b.md`.

Target and stopping condition: independently inspect all executable changes in this range, confirm the three fixes, complete the requested test/type-check commands, prove that constant-critic and permissive tile-bound mutations trigger the relevant regression tests, and restore every tracked file byte-for-byte. No implementation change or cookbook adaptation was made.

## Findings

**No actionable findings remain. All three prior P2 findings are resolved; no further fix is required.**

1. **P2, resolved — tile count bounds.** `python/owl/kaggriculture/types.py:198` applies a separate nonnegative check to channels 0, 5 and 6 after schema validation. Channels 1–4 retain the existing −1 lower bound. `tests/kaggriculture/test_types.py:289` covers each forbidden count sentinel; the existing date/deadline sentinel and signed market test still passes.
2. **P2, resolved — contract-valid fixture.** `tests/kaggriculture/conftest.py:135` implements the Fibonacci sequence starting 1,1; `:165` encodes `100 × fib(hires_today)` with the documented scale. This matches the pinned engine at `kg/reference-2026-09-29:engine_rs/src/lib.rs:3792–3797` and `:3821`. The independent expected-cost table is at `tests/kaggriculture/test_types.py:297`. `conftest.py:216` rejects zero for the fixture's configured maximum; `test_types.py:308` verifies that rejection. This leaves the production per-turn `order_limits` lower bound of zero unchanged, as required.
3. **P2, resolved — constant-critic regression protection.** `tests/kaggriculture/test_model_encoder.py:726` sets controlled tokens and head weights and checks independently computed nonuniform winner probabilities and values. `:722` also checks nonconstant values, and `:806` requires the changed seat to respond while the unchanged seat remains equal. Repeating the exact previous constant-critic mutation now fails all three response tests.

A separate read-only subagent audit confirmed the tile and fixture fixes against stable commit blobs and the pinned engine. The parent independently inspected the changed code and executed every check below.

## Checks

| Command | Result |
|---|---|
| `uv run pytest tests/kaggriculture -q` | **161 passed**, 9.27s, exit 0 |
| `uv run pytest tests/owl tests/scripts tests/tools -m "not slow" -q` | **723 passed, 3 skipped**, 5.74s, exit 0 |
| `uv run mypy python/owl scripts` | **50 source files, no issues**, exit 0 |
| `uvx --offline --from rust-just just py-prepare` after restoration | **884 passed, 3 skipped**, 11.71s test phase, exit 0; formatting, lint, static typing and docs freshness pass |

The initial online `uvx --from rust-just just py-prepare` launcher failed to resolve pypi.org before running the recipe. Using the existing cached tool offline succeeded; both logs are retained. Baseline pytest suites ran concurrently; these timings are not throughput measurements. Skips: two unavailable CUDA flash-attn cases and one unavailable x86 quantized backend. Runtime: Python 3.12.13, Torch 2.9.0, macOS arm64, no CUDA.

## Mutation evidence

Mutations were sequential and run in fresh pytest processes. Each target was restored in `finally`, with original/restored SHA-256 and byte equality checked.

- **Constant critic:** replace `return logits.log_softmax(dim=-1)` with `return torch.zeros_like(logits).log_softmax(dim=-1)` at `python/owl/model/kaggriculture.py:455`. Command: `uv run pytest tests/kaggriculture/test_model_encoder.py -q -k "critic or values_are"`. Result: **3 failed, 3 passed, 28 deselected**, exit 1. Failures are the nonconstant-value, controlled-probability and changed-seat-response tests.
- **Tile bound:** replace `int(tile_counts.min()) < 0` with `< -1` at `python/owl/kaggriculture/types.py:199`. Command: `uv run pytest tests/kaggriculture/test_types.py -q`. Result: **3 failed, 123 passed**, exit 1. Failures are precisely count channels 0, 5 and 6 accepting −1.
- Restored model SHA-256: `92f4df615656f761262ecc62a1219336e5b7904bcde3a4876ccd2cf4bc109dee`.
- Restored types SHA-256: `114b746f8bdde0e1cf3e03e7046e39cc1774d4a3e7491fb2ee449f76316ccc2d`.

The full post-restoration `py-prepare` run passes. All **278 tracked files** match their initial byte hashes; `git diff --exit-code HEAD`, staged diff and tracked porcelain status are clean. The pre-existing untracked `verify-b-4fdb526-local/` evidence is preserved; this verification adds only this untracked receipt directory.

## Limits and incidental check

This verifies the reviewed CPU contract/fixture/critic changes. It does not qualify real CUDA compilation, FlashAttention, native writer semantic parity, combined-lane integration, training throughput or playing strength.

The range whitespace check passes for source/tests/cookbook. Its unrestricted form flags trailing whitespace only in three newly committed raw transcript logs (626 locations); this is retained transcript formatting, not an executable regression. See `range-whitespace.log`.

Raw checks, mutation script/results, initial tracked hashes, environment and final restoration checks are adjacent to this report. `SHA256SUMS` binds the receipt files.

VERDICT: APPROVE
