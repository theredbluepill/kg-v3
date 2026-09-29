---
type: "Reference"
title: "Kaggriculture parity summary maps tested and untested layers"
description: "Task 7.5 adds a summary at the top of the Kaggriculture part of docs/rules-parity-coverage.md. For integration tip 994818b it lists each tested layer: official episodes, live Kaggle 1.32.7 differential, retained unit/RNG, grammar, encoder, native env against its reference, the Python adapter and the Task 7.1 evaluation opponents. Each row names its oracle, denominator, tests and receipts. The BC pairing diagnostic appears as local, non-parity evidence. The page lists 14 untested areas and the current check counts; three replay negative controls now guard the rewards, done and terminal-bank checks."
tags: ["kaggriculture-v3", "adaptation", "parity", "documentation"]
status: "verified-scoped"
generated: {"by": "openai/codex; reviewed and revised by anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources:
  - resource: "repository:ops/rebuild-2026-09-29/plan.md"
  - resource: "repository:docs/rules-parity-coverage.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/claims.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/py-prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/engine-tests.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/bc-audit.txt"
  - resource: "repository:ops/rebuild-2026-09-29/merge-env-adapter/prepare.log"
  - resource: "repository:ops/rebuild-2026-09-29/merge-7.1/prepare-on-5b43062.log"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/r2-fixes/results.md"
  - resource: "repository:ops/rebuild-2026-09-29/7.5/r2-fixes/replay-negative-controls.log"
  - resource: "repository:engine_rs/tests/replay_parity.rs"
  - resource: "bc-branch:kg/rebuild-bc-now@933d661/ops/rebuild-2026-09-29/bc-a100-2026-09-29/pairing.json"
---

# Kaggriculture parity summary maps tested and untested layers

Plan Task 7.5 reads: "`docs/rules-parity-coverage.md` gains a Kaggriculture
section stating what is tested and what isn't". Invariant I9 makes that page the
source of truth for parity. The Kaggriculture detail had grown across
Tasks 1.1–1.5 into long per-task sections, each with its own historical counts.
No one place said which layers are tested, against which oracle, and what stays
open. Before writing, I searched `cookbook/` for coverage summary, parity map,
what is tested and untested. I found only the
[[live-differential-parity-checks-the-rust-kernel|live differential Reference]],
the [[native-game-semantics-use-v3-owned-buffers|native semantics Reference]] and
the [[structured-observations-preserve-legal-state-and-order|observation Reference]].
Each of them covers one layer, so this record is new rather than a revision.

## Changed paths

- `docs/rules-parity-coverage.md` gains the new section
  `## Kaggriculture Coverage Summary (Task 7.5)`, placed before
  `## Kaggriculture Rules Kernel` and linked from the page's opening paragraph.
  The section has four parts:
  - the compatibility target and the tests that assert it;
  - a "What is tested" table with one row per layer, giving the oracle,
    denominator, test entry points, receipts and a link to the detail;
  - "Local evidence that is not parity";
  - "What is not tested" and "Current checks".
- The same page also gets seven minimal corrections of statements that are
  stale at this tip, plus a pointer to the summary in its opening paragraph. For example, replay parity now names
  `Game::new_with_seed_decimal`, the Task 1.3 retirement of the grammar include
  is recorded as done, and Task 1.5 is marked as merged on CPU with its CUDA
  fence still open. The Orbit Wars sections are byte-identical.
- `ops/rebuild-2026-09-29/7.5/` adds `claims.md`, a ledger with one row per
  claim that maps it to file:line or command output. It also adds
  `results.md`, check logs, resource receipts and `py-prepare.log`.

## Scope decisions

- **Base.** The branch started from integration `bde3374`, the first tip that
  contains Tasks 1.4 and 1.5. The integration had already moved to `994818b`
  (Task 7.1 merged) before the summary was committed, which the r2 subagent
  review found. The branch now merges `994818b`; the page has a Task 7.1 row
  and lists only 7.1's untested scope (custom configurations, CPython 3.12+,
  foreign prefixes, individual-order rejection, strength, no learned-seat hook).
  Tasks 7.3 and 7.4 remain listed as untested.
- **BC pairing diagnostic.** This is local evidence on `kg/rebuild-bc-now`
  (`933d661`) and was only read with `git show`. It re-steps Kaggle's own 1.32.7
  interpreter from archived observations and compares values on
  `day`/`hour`/`farms`/`market`/`town` and both privates. Over 8 episodes,
  5,743/5,752 transitions match, and all 9 mismatches are private-only. No Rust
  engine is involved, so the page says it is neither engine parity nor proof of
  label correctness. The receipt calls the mismatch turns day ends. The episode
  configurations needed to confirm that are not tracked, so the page leaves the
  claim unconfirmed and the cause unattributed.
- **Task 1.4 oracle.** The page states that the Task 1.4 reference is the pinned
  reference crate's Rust `TrainingBatch`. That makes it a native-wrapper
  comparison, not a second Python rules differential.

## Verification

- I checked that every cited test function and each of the 27 file or receipt
  paths exists at `bde3374`. I spot-checked the numbers against their receipts:
  probe arithmetic from the 1.1b sweep summary, codec 321/43 and three
  96-transition extreme reward games from the Stage 2 native results, and the BC
  pairing JSON. The review notes are in `ops/rebuild-2026-09-29/7.5/results.md`.
- Engine `cargo test` passed 69 tests with 0 failed and 0 ignored.
  `check_engine_trim.py` passed.
- `uvx --from rust-just just py-prepare` exited 0 with 2,319 passed and
  10 skipped, and docs-fresh reported "No doc updates required".
- The root Rust suite was not rerun. Codex's guarded run stopped at its 960 MiB
  guard. No Rust source, Cargo manifest or lockfile changed after merge commit
  `7f797a3`. The page therefore cites the merge's `just prepare` result
  (274 passed, 5 ignored) as the latest full run.
- Codex verify round 1
  (`ops/rebuild-2026-09-29/codex/verify-7.5-r1-prompt.md`) and its resume both
  stopped on Codex's usage limit before doing any work, so there is no Codex
  verdict. The owner then asked: "please ask you subagents to review instead for
  now". An independent Claude subagent reviewed `e9aafba`
  (`ops/rebuild-2026-09-29/codex/claude-verify-7.5-r1.md`, a local working file
  in the main checkout). Its verdict was APPROVE WITH EDITS with five P3s and no
  P1 or P2. It reran the engine suite (69 passed), the trim check, docs-fresh and
  targeted pytest shards, and six scratch mutations of cited oracles were all
  caught. The P3 fixes followed on the branch:
  - the "Current checks" environment sentence now covers only Codex's rows;
  - the `just py-prepare` row now states that its environment and peak memory
    were not recorded, and that it probably exceeded 1 GB;
  - the BC paragraph now cites `7.5/bc-audit.txt` as its receipt at this tip;
  - stale receipt lines are marked as superseded;
  - this note's correction count is fixed;
  - the tracker row is updated.
  The subagent review is not a Codex verdict.
- A second independent Claude subagent review of `81d0bf7`
  (`ops/rebuild-2026-09-29/codex/claude-verify-7.5-r2.md`, local) returned
  REQUEST CHANGES: one P2 (the summary was stale at `994818b`) and four P3s.
  Its scratch mutations M1–M3 disabled the replay comparator's rewards, done
  and terminal-bank checks, and all 19 replay tests stayed green. The fixes,
  recorded in `ops/rebuild-2026-09-29/7.5/r2-fixes/results.md`, are:
  - the `994818b` merge and the refreshed summary;
  - three negative controls in `engine_rs/tests/replay_parity.rs`, each of
    which fails when its check is disabled;
  - the 1 GB figure now cites the tracked `7.5/pytest.json`;
  - the targeted-shard rule is scoped to Mac-limited review checks.
  Checks after the fixes: engine 72 passed, opponents 22 passed, trim and
  opponent-import checks passed, pytest shards 178 passed with 11 skipped and
  400 passed, and docs-fresh passed. The `opponents_rs` test run peaked at
  3.0 GB, above the 1 GB check limit, and that peak is unattributed. The root
  Rust suite and the full Python suite were not rerun. No Rust or Python outside
  that test file changed after `994818b`, so the page cites that tip's
  `just prepare` (root 274 with 5 ignored, Python 2,400 with 21 skipped).

## Gaps and reopening conditions

- The summary is a snapshot of integration `994818b`. Update it when
  Task 7.3 (replay export), 7.4 (packaging) or 3.1 (rollout/mask mapping)
  merges, or when the integration tip otherwise changes a cited layer. The
  tracker's 7.5 row asks for a closing pass after 7.1, 7.3, 7.4 and 3.1; 7.1
  is now covered.
- No new sweep or parity run was made. Apart from the three replay negative
  controls, the page only maps existing evidence. D1/D2 malformed-input divergences, framework behavior outside the
  interpreter, strong-play worlds, a pod-scale sweep, CUDA/BF16 and pinned-memory
  paths, and complete-update throughput all remain untested, as the page lists.
