# Codex task: write the Task 1.2 brief (grammar kernel, plan C3)

You are working in the git worktree `/Users/poonszesen/kg-v3-grammar` on branch `kg/rebuild-grammar` (based on `kg/isaiah-gap-closure`, which already includes Task 1.1, the vendored rules kernel in `engine_rs/`). Your deliverable is ONE file: `ops/rebuild-2026-09-29/briefs/1.2.md`. Do not write any code, tests, fixtures or other files in this task; this is the design brief that a later implementation task will execute. Do not commit (your sandbox cannot write `.git`; Claude commits for you).

## Read first (required)

- `ops/rebuild-2026-09-29/plan.md`: Global Constraints, "How we use the reference branch" (dispositions V/P/R/X, component table C1–C17), lessons L1–L15 (especially L4 and L6), Isaiah principles I0–I12, "Working with Codex", Tasks 1.1–1.5 and 2.3.
- `docs/kaggriculture-contract.md` (v4, accepted): the whole "Action: `KaggricultureActions`" section, and "Masks and context" (`order_limits`, `action_mask.can_act`). The contract is binding; if you find a contradiction between the contract and the reference, report it as an open question instead of silently choosing.
- `ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md` and `engine_rs/VENDORED_FROM.md`, `engine_rs/TRIM_MANIFEST.json`, `scripts/check_engine_trim.py`: how the kernel is byte-pinned, what "authored" entries are, and the L4 (`serde_json/arbitrary_precision`) analysis and its reopening condition.
- `/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/briefs/2.3-action-heads.md` (Task 2.3 brief v3, lives in another worktree; read it by absolute path), especially §2 "Grammar tables and masks" and §3: these are the mask tables and runtime context the Python heads will consume.
- `AGENTS.md` / `CLAUDE.md` in the worktree (repo rules: fail fast, no `getattr`, `cargo add`/`uv add`, docs-fresh, cookbook contract).
- Isaiah's own native style: `src/rl.rs`, `src/rl/action_spec.rs`, `src/rl/obs_spec.rs`, `src/rl/vec_env.rs`, `docs/rl-api-specs.md`.

## Reference sources (read them yourself with `git show kg/reference-2026-09-29:<path>`)

Note: in zsh, write `git show "kg/reference-2026-09-29:engine_rs/..."` with quotes (an unquoted `:e` is a zsh modifier).

- `engine_rs/src/myolie_sampler.rs` (C3, disposition P renamed to a v3 grammar module): `Shape`, `State::allows`/`advance`, `plan` (binary DFA format), `decode`, and its unit tests.
- `engine_rs/src/ffi.rs`: the sampler FFI wrappers (`re_myolie_sampler_plan`/`decode`), the constants near `MYOLIE_WIDTHS`, `myolie_action` (the decoder the training path actually used) and the test modules `myolie_sampler_ffi_tests` and `myolie_array_tests` (C5, disposition R: nothing from `ffi.rs` is ported as code; read it for semantics and tests).
- `engine_rs/src/training.rs` (how frames/lengths reached the decoder and the kernel step), `src/kaggriculture.rs` (the PyO3 wrappers `kaggriculture_grammar_plan` / `kaggriculture_decode`).
- `python/owl/kaggriculture/actor_codec.py` (C6, R; oracle only), `python/owl/kaggriculture/gpu_sampling_grammar.py` (C7, P in Task 1.5; see how it extracted the mask tables from the plan and the "market quantity digits differ" check), `python/owl/kaggriculture/native_bridge.py`.
- `tests/kaggriculture/test_codec.py` (the coupled-Gumbel HIRE enumeration `test_parallel_hire_correction_exact_distribution_with_stop_and_empty`, the table/plan agreement test, and the replay test `test_native_device_grammar_replays_oracle_program_without_python_masks`), and the grammar-related tests in `tests/kaggriculture/test_model.py` and `tests/kaggriculture/test_env.py`.
- The vendored kernel `engine_rs/src/lib.rs` in this worktree: how `Game::step` parses `farmer` / `hands` / `market` JSON commands (`parse_order` etc.), so that you can state exactly which command strings canonical decoding must produce and how the kernel treats `[]` (EMPTY), `null`, and quantities.
- Real programs available as data: `engine_rs/fixtures/episode-*.jsonl.gz` (four official episodes kept by Task 1.1).

## Hard constraints

- This is the owner's Mac: CPU-light only, no training, no GPU. Any build you run uses `CARGO_BUILD_JOBS=2`. You may run read-only inspection commands and light `cargo`/`pytest` commands if they help you verify a claim, but you do not need to.
- Never edit vendored kernel bytes. `engine_rs/src/lib.rs` is byte-pinned by `TRIM_MANIFEST.json` (its only permitted edits are the recorded `mod`-line deletions), and so are `engine_rs/Cargo.toml`/`Cargo.lock`. A file placed at `engine_rs/src/grammar.rs` is not compiled unless `lib.rs` declares it. The brief must therefore choose and justify where the grammar module lives (for example: a new authored crate, an authored file reached without touching pinned bytes, or the root crate under `src/kaggriculture/`), how the manifest registers any new authored files under `engine_rs/`, how the "decoded programs feed the kernel step" test gets access to both the grammar and the kernel, and what this choice means for L4 (does it add a root → engine dependency now, or defer it to 1.3/1.4, and what exactly then happens to `src/rules_engine/generation.rs`?).
- The reference branch is a design and test oracle only: port with understanding, never copy blindly. For every reference item you use, give its disposition and what is deliberately NOT carried over (and why).
- Isaiah alignment: follow his `src/rl` native conventions (typed constants, caller-owned buffers, fail-fast `Result` errors) and the repo rules (I10–I12).

## Required scope of Task 1.2 (the brief must cover each item)

1. The v3 grammar module from `myolie_sampler.rs`: rename (no "myolie"), type (typed enums for `UnitKind`, `MarketKind`, `ActionItem`, slot names/widths as named constants matching the contract's `SLOT_NAMES` order), drop what is not needed, keep masks, finite-state transitions (`State::allows` / `advance`) and canonical decoding.
2. Contract v4 "Action" semantics exactly: 252 frames × 12 slots; widths (241,20,128,16,2,32,32,8,16,32,32,2); `unit_target` reserved `{0}`; unit kind 19 and items 13–15 always masked; HIRE capacity `plan(actors, order_limit, hire_limit)` with v3 default `hire_limit = 241`, counting submitted HIRE orders even if they later fail; STOP = the first final NONE; the forced NONE sentinel at market position `p == order_limits`; EMPTY consumes an order; quantity rules (unit quantity omitted or 1–1023, explicit unit zero rejected; market quantity 0–1023 with explicit zero preserved). State the exact HIRE inequality and check that 241 actors never become 242 through grammar-admitted HIREs.
3. Tokens: the contract ships `tokens int64 [E,2,252,12]` and `lengths int64 [E,2]`. Define the decode entry point that Task 1.4 calls per (env, seat) with the padded frame block and its length, including validation of negative/out-of-width tokens before any indexing (L6: fail fast, never clamp) and what happens to padding frames after `lengths`.
4. Exported mask tables for the Python heads (Task 2.3 brief v3 §2): `unit_kind[20]`, `unit_item[20,16]`, `unit_quantity_present[20,2]`, `unit_quantity_high[2,32]`, `unit_quantity_low[2,2,32]`, `market_kind[8]`, `market_item[8,16]`, `market_quantity[8,32]`, in a form a later PyO3 binding (Task 1.4) can expose without re-deriving them, plus a Rust-side assertion that the market high- and low-digit supports are equal. Say whether the binary `plan` DFA is still needed by any v3 consumer (2.3 uses the factored tables plus runtime context); if it is dropped, say what replaces its role and its tests.
5. Tests with oracles, as a TDD list (failing test first): the coupled-Gumbel HIRE enumeration (3 positions, 8 kinds, budgets 0/1/2/3/10) ported so that its sequential law is driven by the Rust grammar itself; table/transition agreement over all reachable local states; the reference unit tests (exact quantities, EMPTY, HIRE tracking, malformed/noncanonical rejection, bounds); the replay tests; and a test that feeds decoded programs into the vendored kernel's step and checks acceptance.
6. Oracle fixture: decode at least 200 recorded reference programs identically to the reference decoder, including dense 241-actor programs and full-market programs, plus rejected (malformed) programs with the reference's accept/reject verdict. Specify how the programs are generated/recorded on the reference branch (which worktree, which recorder, which seeds, which shapes, real replay actions vs. sampled programs), the fixture format and size budget, and its provenance and SHA-256 hashes. The fixture is small and committed.

## Brief format

Write `ops/rebuild-2026-09-29/briefs/1.2.md` in the superpowers:writing-plans style used by the existing briefs in `ops/rebuild-2026-09-29/briefs/` (look at `1.1-rules-kernel.md` and `2.1-encoder.md`): header with goal/architecture/spec links, then these sections:

1. **Files**: every file created or modified, with owner and whether it is authored, vendored or generated.
2. **Interfaces**: exact Rust signatures (types, errors) of everything later tasks consume, and the Python-facing shapes/dtypes that Task 1.4 (PyO3 env) and Task 2.3 (heads) will read. Show how the 2.3 brief's `GrammarTables` maps onto them.
3. **Reference files and dispositions**: a table (reference path, lines/functions, disposition, what is kept, what is NOT copied and why).
4. **Semantics**: the state machine per slot (a table of `allows`/`advance`), the HIRE budget, STOP/NONE/EMPTY, quantities, and the canonical JSON command form accepted by the kernel.
5. **Tests with oracles**: numbered TDD steps (test code sketch, exact command, expected failing then passing), including the fixture recorder.
6. **Commands and acceptance**: exact commands (with `CARGO_BUILD_JOBS=2`), `just rs-prepare`, the trim-manifest check, and docs to update (`docs/kaggriculture-contract.md` if a contradiction needs a recorded change, `docs/rl-api-specs.md`, cookbook adaptation record per the repo's cookbook contract).
7. **Risks**: including L4, L6, divergence between the reference's two decoders, fixture custody.
8. **Open questions**: numbered, each with your recommended answer.

Ground every claim in a file and line (reference `path:line` or worktree `path:line`). Do not invent results; mark anything you could not verify as unverified.

## Final report

In your final message, report: the brief path, the files you read, any commands you ran with their results, and your open questions.
