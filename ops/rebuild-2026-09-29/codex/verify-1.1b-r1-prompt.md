# Independent verification: Task 1.1b (live Kaggle-vs-Rust differential parity)

You are an independent verifier. Repository: /Users/poonszesen/kg-v3-parity,
branch kg/rebuild-parity, HEAD 9ee7fb86c0e1118bfe59093c00af0db0b7fde0a3.
Review `git diff 0dc9bdd..HEAD` (commits 33e1428, 8ca378b, 9ee7fb8). Read the
code and docs yourself; do not trust commit messages or docs claims without evidence.

Use TMPDIR=/Users/poonszesen/kg-v3-parity/.codex-tmp for scratch. Work offline
(UV_OFFLINE=1, cargo --offline). Do NOT leave tracked modifications: any
perturbation or experiment must be reverted (`git status --porcelain` must be
clean of tracked changes when you finish; do not commit anything). Do not modify
pyproject.toml or uv.lock.

## Checks (report evidence for each)

1. The generator really drives kaggle-environments 1.32.7's kaggriculture engine
   (not a reimplementation or the Rust engine), and refuses to run when the
   installed engine hash does not match the pinned one. Show where the hash is
   computed/checked and whether the refusal path is tested or at least exercised.
2. The generated trace format matches the official fixtures' format (same
   schema/keys/structure that the Rust harness consumes).
3. The Rust harness compares the complete state for generated traces exactly as
   it does for the official ones (no fields skipped, no looser comparison, no
   separate weaker code path).
4. The edge-case policy actually exercises the cases it claims (inspect the
   generated actions in the fixtures, count occurrences of each claimed case).
5. Perturbation: make one small deliberate change (e.g. mutate one state value
   in one generated fixture, or perturb one Rust rule) and confirm the replay
   test fails with an informative error; then restore and confirm it passes.
6. Fixture sizes and manifest hashes: verify each manifest hash against the file
   on disk; report fixture sizes and whether they are reasonable for git.
7. The documented coverage (docs/rules-parity-coverage.md and any cookbook/ops
   records in the diff) matches the actual evidence.
8. Isaiah's pyproject.toml and uv.lock are untouched by the diff
   (`git diff 0dc9bdd..HEAD -- pyproject.toml uv.lock` must be empty).
9. Any reported divergences are accurately minimized (reproduce at least one if
   reported; check the minimization claim is true).

## Commands to run and report counts for

- `cargo test --manifest-path engine_rs/Cargo.toml --locked --offline`
  (report passed/failed/ignored counts)
- `uv run python scripts/check_engine_trim.py` (with UV_OFFLINE=1)
- A small sweep with UV_OFFLINE=1 and N=8 using the sweep entry point added in
  the diff (find the right invocation from the diff/docs). Report episodes run,
  matches, divergences, and runtime.

## Output

Write a concise report: HEAD verified, each check with PASS/FAIL/PARTIAL and
evidence (file:line, command output counts), findings split into blocking and
non-blocking, command results, and final tracked-tree status. End with exactly
one line:

VERDICT: APPROVE | APPROVE WITH EDITS | REJECT
