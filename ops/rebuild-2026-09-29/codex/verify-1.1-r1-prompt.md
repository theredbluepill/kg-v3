# Independent verification: Task 1.1 (vendor the trimmed rules kernel)

You are an independent verifier. Repository: /Users/poonszesen/kg-v3-codex, branch `kg/rebuild-codex`, HEAD `0dc9bddb5369d988be230b022eff3eb2f363e624` ("Vendor the trimmed Kaggriculture rules kernel (Task 1.1)"). Base: `kg/isaiah-gap-closure`.

Read first (all in this repo):
- `ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md` (the brief)
- `ops/rebuild-2026-09-29/briefs/1.1-review-claude.md` (the prior review; its required edits must be reflected)
- `ops/rebuild-2026-09-29/plan.md` rows C1, C2 and L4, and the "Task 1.1" section

Then independently verify `git diff kg/isaiah-gap-closure..HEAD` against those documents. Do not trust the commit message or VENDORED_FROM.md claims; check them.

## What to verify
1. **Byte identity of retained files.** The retained files (per the brief: `lib.rs` apart from its recorded removals, `py_random.rs`, `econ_attrib.rs`, and any other files the brief says are kept byte-identical, plus fixtures) must match the reference source named in the brief / `engine_rs/VENDORED_FROM.md` (locate the reference ref/commit via git; e.g. `git show <ref>:<path> | shasum -a 256`). Compare hashes yourself against the manifest.
2. **Exactly seven `lib.rs` removals.** Diff the vendored `engine_rs/src/lib.rs` against the reference `lib.rs`. The only differences must be exactly seven removed lines (the `mod`/`pub mod` lines for excluded modules), each recorded in `VENDORED_FROM.md`. Report any other difference.
3. **No excluded modules.** No C2 modules (`native_agents/`, `joint_matching.rs`, `policy_rows.rs`, `src/bin/*`, `examples/*`, bot fixtures), no `myolie_features`, no `ffi` present in `engine_rs/` and nothing references them.
4. **Replay-parity test strength.** Confirm the replay-parity test actually compares engine output against fixture data and would fail on a rules change. Prove it is not vacuous: apply a small temporary mutation (e.g. perturb one rule constant or step computation in a retained file, or perturb one fixture value), run the parity test, observe the failure, then restore the file exactly with `git checkout -- <path>` and confirm `git status --porcelain` is clean.
5. **Checker correctness.** Review `scripts/check_engine_trim.py` and `tests/tools/test_check_engine_trim.py`: does the checker actually enforce hashes, the removal count, excluded modules and provenance, with fail-closed behavior? Are the tests meaningful (would they catch a regression)?
6. **justfile / rustfmt scope.** Check any `justfile`, `rustfmt.toml`, or formatting configuration changes: they must be limited to what's needed (e.g. excluding vendored bytes from reformatting) and must not weaken checks for the rest of the repo.
7. **No root Cargo changes.** Confirm the root `Cargo.toml` / `Cargo.lock` are unchanged (or, if changed, whether that contradicts the brief/plan L4 decision) and report how L4 (`serde_json/arbitrary_precision`) was resolved and whether that resolution is sound.

## Commands to run (report pass/fail counts for each)
- `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`
- `uv run python scripts/check_engine_trim.py`
- `uv run pytest tests/tools -q`
- `cargo test` (repository root)

If a command cannot run in the sandbox (network, permissions), say so precisely and report what you could establish instead.

## Rules
- Do NOT leave any modification in tracked files. Any mutation for step 4 must be restored; finish with `git status --porcelain` clean (untracked build output under ignored paths is fine). Do not commit, do not create branches.
- Do not edit briefs, plan, or code to "fix" things; report instead.

## Output
Write a concise report: HEAD verified, each numbered check with evidence (hashes, diff lines, command output counts), the mutation you tried and its result, command results with counts, and a findings list separated into blocking and non-blocking. End with exactly one line:
`VERDICT: APPROVE` or `VERDICT: APPROVE WITH EDITS` or `VERDICT: REJECT`
