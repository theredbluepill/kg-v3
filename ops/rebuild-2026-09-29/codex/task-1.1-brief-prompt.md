You are Codex, working with Claude on the Kaggriculture v3 rebuild, in your worktree (branch `kg/rebuild-codex`, now at the integration branch head). You may commit on `kg/rebuild-codex` (Git metadata is writable this time). Do not push or touch other branches.

Read: `AGENTS.md`; `ops/rebuild-2026-09-29/plan.md` (Global Constraints, "How we use the reference branch" with component dispositions C1–C2, lessons L4, Task 1.1); `cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md`.

YOUR TASK NOW: write the BRIEF for Task 1.1 (vendor the trimmed rules kernel). Don't implement it yet: Claude reviews the brief first. Save it as `ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md`, in the superpowers writing-plans format (bite-sized TDD steps, exact commands, expected results).

Investigate the reference with `git show kg/reference-2026-09-29:<path>` and `git ls-tree -r kg/reference-2026-09-29 engine_rs`. The brief must decide, with evidence:
1. **The minimal retained file set** for a rules kernel that can construct a `Game` from config and seed, step it with decoded commands, and expose state and economic counters. Start from `engine_rs/src/lib.rs`, `py_random.rs`, `econ_attrib.rs`. List every `mod`/`use` in the retained files that points at an excluded module (`ffi`, `native_agents`, `myolie_features`, `myolie_sampler`, `policy_rows`, `joint_matching`, `training`, bins/examples), and state the exact minimal edit for each. The goal is retained files byte-identical except for listed, justified line removals.
2. **Which engine tests and fixtures** prove rules parity after trimming (lib.rs unit tests, `tests/py_random.rs`, replay-verification fixtures such as `fixtures/episode-*.jsonl.gz`). Which tests depend on excluded modules, and are therefore excluded with a reason?
3. **Workspace integration**: Cargo workspace membership versus a path dependency; and how to avoid L4 (`serde_json/arbitrary_precision` unification breaking Isaiah's `src/rules_engine/generation.rs` fixture decode). Check whether the root crate needs the engine at all before Task 1.4. If not, keep the engine a separate package that the root crate doesn't depend on yet, and say whether Cargo feature unification still happens in a shared workspace (and if it does, the minimal fix).
4. `engine_rs/TRIM_MANIFEST.json` format: retained files with reference SHA-256, edited files with line-level justifications, excluded files with reasons. Also a checker script that verifies it.
5. Commands and expected results: `cargo test --manifest-path engine_rs/Cargo.toml --locked`, root `cargo test` (Isaiah's 155 passed / 2 ignored must be unchanged), `just rs-prepare`.

RULES: owner's Mac, so tests and builds only; no network needed beyond what `cargo fetch` already cached (report if the engine needs crates that aren't cached). Commit the brief on `kg/rebuild-codex`.

FINAL REPORT: the brief path, the commit hash, and any decision you want Claude to confirm.
