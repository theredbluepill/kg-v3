Reviewer: independent Claude subagent (substitute for Codex during its usage limit; owner-approved). Not a Codex verdict.

# Verify merge: Task 7.5 parity summary onto the integration (kg/merge-7-5-c)

Scope note: the workflow told this agent to spawn no subagents, so the agent that
made the merge also ran this verification, as a separate pass over the merge
diff. It is independent of the Task 7.5 author and reviewers, not of the merge
author.

- BASE (integration tip): `bd1c9279ec09a27a5c05069197fef404263cb55b` (W&B landing)
- Merged branch: `kg/rebuild-7-5` `31c19efe0e17157d9d73fef8b4fc842c508d269c`
- Merge base: `994818b87041426c6fc442a85fb06932937d58a7`
- Merge commit: `458505c801aba19202e31e39c5295a16fd01f408` on `kg/merge-7-5-c`
  (worktree `/Users/poonszesen/kg-v3-m-7-5`), `--no-ff`, regular merge.

## What each side changed since 994818b

- Integration (`bd1c927`): Task 4.4 teacher configs, the Task 3.1 remainder with
  Task 3.5, and the W&B wiring. `git diff --stat 994818b bd1c927` over
  `engine_rs opponents_rs src *.rs Cargo.* tests/fixtures scripts/kaggriculture_parity`
  is empty; over `python/owl/kaggriculture` and the test files the parity rows
  cite it is empty too (only `test_configs.py`, `test_teacher.py`,
  `test_training_smoke.py` changed, none of which a parity row cites).
- `kg/rebuild-7-5`: docs, cookbook, `ops/rebuild-2026-09-29/7.5/` receipts, three
  new negative controls in `engine_rs/tests/replay_parity.rs` and its
  `TRIM_MANIFEST.json` hash.

## Conflicts and resolutions

- `cookbook/log.md` (two hunks). Resolved by keeping both sides: integration's
  2026-09-30 entries, then 7.5's 2026-09-30 entries, then integration's
  2026-09-29 Task 3.1 entry, then 7.5's 2026-09-29 entry; the second hunk was
  integration-only (7.5 side empty) and is kept. Check: every `## ` heading from
  each parent occurs exactly once in the merge; no duplicate headings; dates are
  non-increasing top to bottom. A new landing entry is prepended.
- Auto-merged: `cookbook/references/index.md`, `docs/rules-parity-coverage.md`,
  `ops/rebuild-2026-09-29/phase-status.md`. Every line either parent added since
  `994818b` is present in the merge, except five lines replaced on purpose (the
  7.5 index line, the doc's stale Task 3.1 bullet, the tracker's "Merged"
  definition and 7.5 row).

## Nothing lost (test names)

Three-way count check over every `def test_*` and Rust `fn` name in `*.py`/`*.rs`
at `994818b`, `bd1c927`, `31c19ef` and the merge: for every name,
merge count = integration + 7.5 - base, with zero mismatches. The three 7.5
tests (`generated_trace_perturbed_rewards_are_rejected`,
`generated_trace_perturbed_terminal_banks_are_rejected`,
`generated_trace_perturbed_transition_count_fails_done`) are present once.
Per-file: every file only 7.5 changed equals `31c19ef` except the parity
Reference (intentional landing edits); every file only the integration changed
equals `bd1c927` except `plan.md` (7.5 checkbox). `replay_parity.rs` is
byte-identical to `31c19ef`; its SHA-256 `6520938a…52e1` equals the
`TRIM_MANIFEST.json` authored entry, so no regeneration was needed (the file
did not change on the integration side).

## Refreshes made for counts and facts changed since the base

- Parity summary intro: records the landing onto `bd1c927` and why the rows
  still hold.
- "What is not tested": the Task 3.1 bullet said `run_ppo` "still stops at that
  seam", false since `821b446`. It now says `run_ppo` carries Kaggriculture and
  only CPU functional checks (Task 3.5) exercise it, with no learning, GPU or
  multi-rank qualification. The count of untested areas stays 14.
- Merge-history paragraph: adds the W&B landing counts (engine 69, root 274/5
  ignored, opponents 22, Python 2,587/17 skipped, from
  `merge-wandb-c/prepare.log`) and this landing's counts.
- "Current checks" table: adds this landing's `just prepare` row.
- Reference note, its index line, the tracker (Phase 7 now 2/5; 7.5 row merged,
  naming the Claude substitute reviewers) and plan checkbox updated. The three
  Task 7.5 review reports and prompts are now tracked under `codex/`.

Spot-checked counts: the landing log's skip summary still lists nine
learned-seat opponent skips (8 + 1 in `test_opponents.py`), matching the
unchanged 7.1 gap text.

## Checks

- `CARGO_BUILD_JOBS=2 OMP_NUM_THREADS=2 uvx --from rust-just just prepare` on the
  merge: exit 0. Engine 72 (41 unit, 9 RNG, 22 replay parity), root 274 passed
  5 ignored, opponents 22 (12/5/5), Python 2,587 passed 17 skipped, mypy clean
  on 72 files, docs-fresh "No doc updates required"
  (`ops/rebuild-2026-09-29/merge-7-5-c/prepare.log`). `/usr/bin/time -l` max RSS
  3.3 GB for the largest single process, unattributed, above the 1 GB check
  limit; earlier landings ran the same full prepare.
- Run 1 (`merge-7-5-c/prepare-1-missing-orbit-fixtures.log`) failed seven root
  Orbit Wars fixture tests because the fresh worktree lacked the git-ignored
  `tests/fixtures/generation/` and `tests/fixtures/orbit_wars_replays/`. Not
  merge-induced; they were copied from `kg-v3-m-wandb` (SHA-256
  `reference_generation.json` 88527e74…c3ad, `replay-75926553.jsonl`
  8a50c788…a96a9, `replay-75930761.jsonl` bd0c3acd…04d88) and run 2 passed.
- After the final text edits: `just docs-lint` clean, `check_doc_freshness.py`
  clean, `cookbook-lint.mjs --staged-sources --require-log` exit 0.

## Owner rules

Docs, cookbook, receipts and test-only Rust. No trainer, model, v2 code, policy
inputs or between-turn state. `scripts/run_ppo.py` stays the one trainer. No
deadlines or submission timing. Regular merge commit with the trailer; nothing
pushed. Pod worktree and branches and `~/kaggriculture-v2` untouched.

## Findings

- P3 (disclosed, not blocking): the landing `just prepare` peak (3.3 GB) exceeds
  the 1 GB Mac check limit; the page and log state it as unattributed.
- No blocking findings.

VERDICT: APPROVE
