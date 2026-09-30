You are Codex, implementing Task 7.5 (parity docs). Worktree: this checkout, `/Users/poonszesen/kg-v3-t75`, branch `kg/rebuild-7-5`, created from the integration tip `kg/isaiah-gap-closure` at `bde3374` (Tasks 1.4 and 1.5 merged; Task 7.1 opponents, Task 7.3 replay export, Task 7.4 packaging and the BC lanes are NOT merged). Plan line (`ops/rebuild-2026-09-29/plan.md`): "7.5 (Codex) Parity docs: `docs/rules-parity-coverage.md` gains a Kaggriculture section stating what is tested and what isn't." Plan invariant I9: "Parity-first engine; `docs/rules-parity-coverage.md` is the source of truth; Kaggriculture parity documented the same way."

## What to write

Add ONE new section to `docs/rules-parity-coverage.md`, titled `## Kaggriculture Coverage Summary (Task 7.5)`, placed immediately before `## Kaggriculture Rules Kernel` so a reader meets it first. It is a map over the detailed sections that follow, not a copy of them. Target roughly 80-140 lines of prose plus tables. It must contain:

1. **Compatibility target.** `kaggle-environments==1.32.7` as pinned in `pyproject.toml`/`uv.lock`, the interpreter SHA-256 pinned in `engine_rs/Cargo.toml` metadata, the vendored reference commit, and which test(s) assert the installed pin/hash (find them; e.g. `tests/scripts/test_kaggriculture_parity.py`, `scripts/kaggriculture_parity/generate_traces.py` guard, `scripts/check_engine_trim.py`).
2. **A "What is tested" table**, one row per layer, columns: layer; oracle (what independent truth it is compared against); scope/denominator (games, transitions, cases); test entry point(s) (file and, where useful, test name); receipt path; link to the detailed section below. Layers, each only if tested at this tip:
   - Rules kernel vs the four official recorded episodes (engine replay parity, `engine_rs/tests/replay_parity.rs`).
   - Live differential parity vs Kaggle's own 1.32.7 Python engine (Task 1.1b: committed generated traces, divergence repros, the recorded local sweep).
   - Retained engine unit and RNG integration tests (synthetic, read no episodes).
   - Task 1.2 native grammar (fixture of 320 accepted programs + controls, kernel acceptance tests now in root `src/kaggriculture/grammar_kernel_tests.rs`).
   - Task 1.3 observation encoder (tensor-only reconstruction against the reference crate's recorded rows; state explicitly this is observation-information coverage, not rules parity).
   - Task 1.4 native environment lifecycle, admission and the native-vs-reference oracle (16 recorded games, seeds 17000-17015, 11,504 transitions; `tests/fixtures/kaggriculture_env_reference_v1.*`, its recorder/test). State precisely what the "reference" is (read the recorder and its manifest: which engine produced it) and what fields are compared.
   - Task 1.5 Python adapter/codec/table bridge, only the parts with real tests at this tip.
3. **Local evidence that is not parity** (a short subsection): the BC pairing diagnostic. It lives only on branch `kg/rebuild-bc-now` (commit `933d661`), NOT in this integration; read it with `git show kg/rebuild-bc-now:ops/rebuild-2026-09-29/bc-a100-2026-09-29/pairing.json`, `...:ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md` and `git show kg/rebuild-bc-now:scripts/kaggriculture_prepare_bc.py` (`pairing_check`, `PAIRING_PUBLIC_KEYS`). Never check out, write to or run anything in that branch or `/Users/poonszesen/kg-v3-bcnow`. What it is: Kaggle's own 1.32.7 Python interpreter re-stepped from each archived Kaggle episode's `steps[t]` observations with the recorded `steps[t+1]` actions, compared by value on `day`/`hour`/`farms`/`market`/`town` and both seats' `private` (not `step`, not key order, not statuses/rewards); 8 sampled archive episodes, 5,743/5,752 transitions match; all 9 mismatches are private-only. Verify from the JSON and the episode configuration yourself whether the mismatch turns (335, 383, 431, 455, 527, 623) are day ends before saying so. It does not involve the Rust engine, so it is not engine parity; the private-only mismatch cause is unattributed. Cite it as local evidence on that branch with its path and commit.
4. **"What is not tested" list**, explicit and specific, each item one or two lines, e.g. (verify each; drop any that is actually tested, add any you find): D1/D2 malformed-input divergences (recorded, not repaired); framework behavior outside the interpreter (timeouts, agent errors, `INVALID`); strong-play worlds beyond the four official episodes; no larger pod sweep; Task 7.1 opponents and their oracle (approved on `kg/rebuild-7-1`, not in this integration); Task 7.3 replay export/Kaggle-episode round trip (not merged, not approved); Task 7.4 packaging (brief only); the Task 3.1 Kaggriculture rollout/mask mapping and any trainer-level or learning check; CUDA/BF16, pinned-memory and GPU behavior; complete-update throughput; the Python rules path differences not exercised by model grammar actions; anything else the detailed sections list as a limit.
5. **Current check counts at this tip**, from commands you actually run now (below), with the date and HEAD they were run at, and the receipt path. Do not copy historical counts as current; the detailed sections keep their historical counts.

## Rules for every claim

- Derive every claim only from what is tested or recorded at this tip (test code, fixtures, manifests, receipts under `ops/`), plus the one explicitly-labelled BC branch item. Open the test or receipt behind each number before writing it. No claim from memory, from the plan's intentions, or from unmerged branches (other than naming them as not-merged gaps).
- Write a claim ledger `ops/rebuild-2026-09-29/7.5/claims.md`: one row per factual claim in the new section (numbers, file names, test names, "tested"/"not tested" statements) with the exact file:line or command output that supports it. Claude reviews every row.
- If you find an existing statement elsewhere in the Kaggriculture sections that is now contradicted at this tip (e.g. "no Python binding exists yet" after Task 1.4/1.5), fix it minimally in place and list it in the ledger. Do not rewrite or reflow the detailed sections otherwise. Do not touch the Orbit Wars sections.
- Also update the first paragraph of the doc if it needs a pointer to the new summary.
- Plain, direct English; short sentences; no marketing adjectives; no deadlines or submission timing.
- Owner rules to respect (do not violate in prose): the policy is stateless and observation-only (`cookbook/decisions/the-policy-is-stateless-and-observation-only.md`); `scripts/run_ppo.py` is the one canonical trainer; no v2 model code.

## Scope limits

- Documentation only: `docs/rules-parity-coverage.md` and `ops/rebuild-2026-09-29/7.5/`. No code, test, fixture, config, lockfile or `engine_rs/` change. No cookbook edits (Claude writes the cookbook record).
- If `just docs-fresh` (`uv run python scripts/check_doc_freshness.py`) maps this doc to anything that needs an update, report it rather than editing code.

## Mac rules (owner's Mac, non-negotiable)

- CPU-light only: no training, no GPU, no model or panel runs, no live framework sweeps. Bounded checks only (<1 GB RAM, <2 min each where possible).
- Export in every shell: `CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 TMPDIR=/Users/poonszesen/kg-v3-t75/.codex-tmp` (create that dir; it is gitignored or leave it untracked). Use `--offline` for cargo/uv/uvx. Add no dependencies. Never go online.
- GIT: your sandbox cannot write `.git`. Do NOT commit, stage, branch, push or merge. Leave changes in the working tree. Do not touch other worktrees or `/Users/poonszesen/kg-v3` except reading.

## Run and report actual results (save logs under `ops/rebuild-2026-09-29/7.5/`)

```
cargo test --locked --offline --manifest-path engine_rs/Cargo.toml
cargo test --offline
uv run --offline python scripts/check_engine_trim.py
uv run --offline pytest tests/scripts/test_kaggriculture_parity.py tests/tools/test_check_engine_trim.py tests/kaggriculture -q
uv run --offline python scripts/check_doc_freshness.py
```

(If the root `cargo test` or the Kaggriculture pytest needs `uv run --offline maturin develop` first, run it. If any step exceeds the Mac limits, stop it and report instead of forcing it.)

FINAL REPORT (Claude reads this; precise and brief):
1. Files changed/added, one line each.
2. Each command with actual pass/fail/ignored/skipped counts or exit status.
3. The "not tested" list as written, and any existing statements you corrected (file:line, before/after in one line each).
4. Anything you could not verify and therefore left out.
5. End with a line `VERDICT: DONE` or `VERDICT: BLOCKED <reason>`.
