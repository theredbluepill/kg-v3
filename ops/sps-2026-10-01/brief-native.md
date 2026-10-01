# Codex brief: cut the serial native env step (Rust), behaviour-identical

Owner (verbatim, 2026-10-01): "please accelerate SPS boost with CODEX", "please accelerate on SPS diagnosis".

## Context
- Live run: 8x H200, 20 envs/rank, `env.native_threads=4`, horizon 720, self-play. Rollout is 27 s of a 32 s iteration.
- Measured from the live run log: `KaggricultureVectorizedEnv.step` costs ~16 s per rank per iteration (720 calls, ~22 ms/call, ~1.1 ms per env-step). That is ~60% of rollout. The per-rank CPU budget is ~12 cores, but rayon workers run at only 22-29% each and the main thread at ~63%, so the step is dominated by SERIAL work on the calling thread.
- Full code map with file:line references: `ops/sps-2026-10-01/code-map.md` (read it first; step 4 is yours).

## Target (what to change)
In `src/kaggriculture/env.rs` (`prepare_step` ~512-788, `commit` ~791-827), `src/kaggriculture/buffers.rs`, `src/kaggriculture/bindings.rs` (722-885), and only if behaviour-identical `engine_rs/src/lib.rs` (~1446):
1. Move the serial per-env work into the existing rayon `pool.install(into_par_iter)`: raw token admission, terminal prediction, `grammar::plan` + `grammar::decode`, the Game clone, and whatever else is per-env and independent.
2. Remove the double Game clone (prepare_step clones every Game, then `step_with_market_metrics` clones again internally). Keep semantics exactly.
3. Stop allocating and zero-filling a fresh `ObsStaging::new(n)` (~5.6 MB) every step: reuse per-env staging (double-buffer if needed), and have each env write its own slice in parallel.
4. Make `commit` cheap: parallel per-env copies into the pinned numpy buffers, or a swap instead of `copy_from_slice` of all 29 fields; drop old games/buffers off the main thread or reuse them.
5. Keep GIL release (`native_work`/`py.detach`) as is or wider.
Optional if clearly safe: avoid `serde_json::Value` round-trips in decode if a typed path exists already.

## Hard requirements
- **Bitwise-identical outputs**: observations (all 29 fields), rewards, dones, truncations, metrics dict, auto-reset behaviour and RNG streams must be identical to the current code for any seed/action sequence, for any `native_threads` (1, 4, 8). Env order must not affect results.
- Add a parity test that proves this: run the old path and the new path (or a golden recording produced from the current code at this commit, stored small) for >= 2 full 720-step games x several envs with fixed seeds and random LEGAL actions (use the existing action sampling/grammar helpers or recorded actions), compare everything bitwise. Also test native_threads 1 vs 4 vs 8 give identical results.
- Add a micro-benchmark (Rust bench or a Python script under `scripts/` or `ops/sps-2026-10-01/`) that times `step` at n_envs=20, native_threads=4, 720 steps, and report before/after numbers on this Mac (Apple CPU; we will re-measure on an H200 host). Put the numbers in the report.
- If a switch is natural, gate the new path behind a config knob defaulting to the NEW path only if parity is proven; otherwise keep old path selectable for A/B. Say which you did.
- Follow repo CLAUDE.md: run `cargo fmt`, `cargo clippy`, `cargo test` (the rs-prepare steps; `just` may be missing — run the commands directly), rebuild the extension (`uv run maturin develop --release`), run the Python env tests (`uv run pytest tests/kaggriculture -q -k "env or native or buffer or step"` and anything touching the bindings), `uv run python scripts/check_doc_freshness.py`. Update `docs/` mapped docs if the architecture description changes.
- Record the adaptation in the cookbook (note + folder index.md + prepended cookbook/log.md entry) per CLAUDE.md, with the parity and benchmark evidence and limits.
- Commit on branch `kg/sps-rollout` with message ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do NOT push. Do NOT touch any remote machine.

## Report (the -o file)
Changes (files), parity evidence (what was compared, how many steps/envs/seeds), benchmark before/after with numbers, test results, limits, and a final line `VERDICT: DONE` or `VERDICT: BLOCKED <reason>`.
