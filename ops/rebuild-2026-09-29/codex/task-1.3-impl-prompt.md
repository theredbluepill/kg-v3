You are Codex, stream A (observation encoder). Worktree: this checkout, branch `kg/rebuild-observe`. IMPLEMENT Task 1.3 exactly as specified in the reviewed brief `ops/rebuild-2026-09-29/briefs/1.3.md`, including every Claude review edit (R1-R5) and the answers to Q1-Q6 in its "Claude review (2026-09-29)" section. Where the brief and the review differ, the review wins. Do not redesign, widen scope, or add policy/model/PPO/reward/grammar code.

Read first: the whole brief; `docs/kaggriculture-contract.md` (v4, observation schema 3); `docs/rl-api-specs.md`; `ops/rebuild-2026-09-29/plan.md` Task 1.3 / C4 / L4 / L6 / L13 / I0-I12; `CLAUDE.md`; `cookbook/index.md` and the newest `cookbook/log.md` entries. The reference branch `kg/reference-2026-09-29` is a design and test oracle only: read it with `git show kg/reference-2026-09-29:<path>`, port with understanding, never copy blindly, never write to it.

WORK TEST-FIRST, task by task (A through J), in the brief's order. For each step, write the test, run it and observe the stated red (the red must be attributable to the missing behavior, not to a malformed test; R5 applies to the L4 regression), then implement, then run green. Perform the brief's mutation checks (temporarily break, observe failure, restore, rerun) and do not leave any mutation in production code. Record concise red/green receipts and timing under `ops/rebuild-2026-09-29/1.3/`, keeping planned expectations separate from actual results.

TASK I DEPENDENCY: `python/owl/kaggriculture/types.py` (Task 2.1's shared schema) is not on this branch. Do NOT copy, extract or re-author a second schema. Implement Tasks A-H and J and everything in Task I that does not need the merged schema (the Rust binding and `python/owl/rs.pyi` signature). Write the Task I Python tests as specified, and report plainly that Task I, and therefore Task 1.3 completion, is blocked on the 2.1 merge. Do not report the task complete.

MAC RULES (owner's Mac, non-negotiable):
- CPU-light only. No training, no GPU, no model runs. At most two live games per diagnostic, under 1 GB RAM and two minutes of execution per bounded diagnostic. Stream corpus records; never materialize a large observation batch.
- Export in every shell before building or testing: `CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Use `--locked --offline` for cargo and `--offline` for uv/uvx. Invoke just as `uvx --offline --from rust-just just`.
- R4: the optimized-build timing test. If the fat-LTO release build exceeds 10 minutes or 1 GB at `CARGO_BUILD_JOBS=2`, stop that step and put the exact command in your report for Claude to run on the pod. Never substitute a debug-profile number.
- Any other generation or build that would exceed these bounds: stop and hand the exact command back in your report. Do not shrink coverage or quotas (R1) to fit.

VENDORED KERNEL: never edit vendored kernel bytes under `engine_rs/` (including `lib.rs` `mod` lines, `Cargo.toml` and `Cargo.lock`). The brief places all new code in the root crate (`src/kaggriculture/`, `src/rules_engine/generation.rs`, `scripts/kaggriculture_observation_oracle/`), so no new engine files should be needed. If you nonetheless must add a new authored file under `engine_rs/`, register it as authored in `engine_rs/TRIM_MANIFEST.json` using the existing trim tooling (`scripts/check_engine_trim.py` and its manifest format; do not hand-invent a new format or a new generator) and report why. `uv run --offline python scripts/check_engine_trim.py` must pass at the end.

DEPENDENCIES: add root crate dependencies only with `cargo add` (offline; all crates should be cached). Add Python dependencies only with `uv add`. Never edit `.toml` or `.lock` files by hand. If a crate is not cached offline, stop and report it; do not go online.

GIT: your sandbox cannot write `.git`. Do NOT attempt to commit, stage, branch, push or merge. Leave all changes in the working tree. Do not write into `.codex-tmp/` except as a temp dir. Do not touch other worktrees or branches.

DOCS AND COOKBOOK (Task J): update `docs/rl-api-specs.md`, `docs/rules-engine.md` and `docs/rules-parity-coverage.md` to the implemented state, and add the cookbook adaptation record, its folder `index.md` entry and a prepended `cookbook/log.md` entry together, following the CLAUDE.md cookbook contract (provenance fields, first tag `kaggriculture-v3`, `repository:` sources, actual checks, remaining gaps including the Task I block).

RUN AT THE END and report actual results (pass/fail/ignored counts; never predicted counts):
```
cargo test --locked --offline
cargo test --locked --offline --manifest-path engine_rs/Cargo.toml
uv run --offline python scripts/check_engine_trim.py
uv run --offline pytest tests/kaggriculture/test_observe.py tests/tools/test_observation_oracle_custody.py -q
uv run --offline --extra reference pytest tests/owl tests/scripts tests/tools -m "not slow" -q
uvx --offline --from rust-just just rs-prepare
uvx --offline --from rust-just just py-prepare
uvx --offline --from rust-just just prepare
```
Expected failures caused only by the missing Task 2.1 schema must be reported as such, with the exact error. Any other failure must be fixed or reported with its cause.

FINAL REPORT (this is what Claude reads; be precise and brief):
1. Files changed or added, one line each, with purpose.
2. Every command run at the end, with its actual pass/fail/ignored counts or exit status; plus the red/green evidence per task (A-J) and the mutation checks performed.
3. Oracle results: corpus record count, seat rows, offsets compared, mismatches; R1 coverage quota results; compressed fixture sizes against the R3 8 MiB budget; fixture SHA-256s.
4. Timing: profile, build time, phase costs, or the handed-off command if R4 stopped it.
5. Any deviation from the brief, with its reason.
6. Open questions and unresolved items (including the Task I block), each as one line.
