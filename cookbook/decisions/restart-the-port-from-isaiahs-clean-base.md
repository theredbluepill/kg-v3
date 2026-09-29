---
type: "Decision"
title: "Restart the port from Isaiah's clean base"
description: "Restart from Isaiah's 32b3ec9 with the prior port as a reviewed reference; preserve layer topology, rerun BC and verify RTX PRO 6000. Task 1.1 rebuilds the isolated, hash-pinned rules kernel with 59 passing engine tests and four native-reset replays; explicit tooling exceptions preserve vendored bytes."
tags: ["kaggriculture-v3", "decisions"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
decider: "Owner: restore kg/isaiah-gap-closure and main to 32b3ec900ad406eedd965f53a1a0f4490d31c589 and work again from a clean state with Codex, keeping a reference branch and carrying the cookbook with the .claude/.codex setup."
sources: [{"resource": "repository:ops/rebuild-2026-09-29/plan.md"}, {"resource": "user-directive:2026-09-29:restart-from-isaiah-clean-base"}, {"resource": "user-directive:2026-09-29:same-layer-topology"}, {"resource": "user-directive:2026-09-29:extra-bc-rtx6000-codex"}, {"resource": "reference-branch:kg/reference-2026-09-29"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:ops/pre-commit"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md"}, {"resource": "repository:ops/rebuild-2026-09-29/1.1-brief-prepare.log"}, {"resource": "repository:ops/rebuild-2026-09-29/briefs/1.1-review-claude.md"}, {"resource": "repository:ops/rebuild-2026-09-29/1.1/results.md"}, {"resource": "repository:engine_rs/TRIM_MANIFEST.json"}, {"resource": "repository:engine_rs/tests/replay_parity.rs"}, {"resource": "repository:scripts/check_engine_trim.py"}, {"resource": "repository:tests/tools/test_check_engine_trim.py"}, {"resource": "repository:justfile"}, {"resource": "repository:rustfmt.toml"}, {"resource": "repository:docs/rules-parity-coverage.md"}]
---

# Restart the port from Isaiah's clean base

The owner directs, on 2026-09-29:

> also for kg/isaiah gap closeure and main branch can you restore to starting points 32b3ec900ad406eedd965f53a1a0f4490d31c589 and work again? For grammar/tokens/game rust engine, you can have a reference branch from this current branch snapshot to refer on, I want you & codex started from a clean state. Carry the cookbooks with you with .claude/.codex setup.

and, just before it:

> Yes, we have different games, so we had different action heads, etc., these kind of things, but the model layers/ topologies should remain. We are just playign different games.

> Also an extra BC is for sure required. Yo ucan use 2/4-rank RTX 6000 to verify your work. Work with codex as well.

## What was done

- Local `main` and `kg/isaiah-gap-closure` were reset to Isaiah's `32b3ec9`. The remote was not changed: `origin/main` is still `0d01234`, and nothing was pushed.
- Branch `kg/reference-2026-09-29` and annotated tag `kg-reference-2026-09-29` (both at `65f0eac`) hold the whole prior port: the initial adaptation `0d01234`, `a2bf23c`, and the cookbook cleanup `65f0eac`. Consult it for the grammar and action codec, observation tokens, the vendored Rust engine (`engine_rs/`), native bindings (`src/kaggriculture.rs`), rewards, BC data preparation, and run evidence under `ops/`.
- Carried onto the working branch: `cookbook/`, `.claude/`, `.codex/`, `AGENTS.md` (with the `CLAUDE.md` symlink), `kaggriculture-v3.base`, `ops/pre-commit` (Git uses `core.hooksPath=ops`), `ops/cookbook-setup-checks.md`, and `.gitignore` (the starter's rules plus bulk and credential exclusions).
- Local run artifacts under `ops/` and `runs/`, which Git ignores, stay on disk; nothing was deleted.

## Consequences

- **References describe the reference branch, not the current tree.** Sources whose files no longer exist are re-pointed as `reference-branch:kg/reference-2026-09-29/<path>`. Sources for files that still exist (Isaiah's own files) stay `repository:` so the first-edit gate still surfaces them. Until the rebuild, those notes describe the reference branch's adapted versions of those files.
- Owner Decisions remain in force.
- **Same layer topology:** the rebuilt model uses Isaiah's `StatelessTransformerV1` layer classes, arrangement and roles: observation stems, learned per-role tokens, trunk blocks, critic head and actor input projection. Only the game I/O differs: input channel widths, categorical encodings fed to the stems, and the action heads. See the [[the-policy-is-stateless-and-observation-only|stateless Decision]].
- **One BC rerun is required** on the rebuilt model, using the reference branch's BC data pipeline.
- **Verification hardware:** 2- and 4-rank RTX PRO 6000 runs are authorized for verification. Read the live price first and write a run statement before each run; see the [[start-multi-gpu-qualification-with-two-ranks|multi-GPU Decision]].
- **Codex collaboration:** Claude and Codex both start from this clean state and work in separate worktrees, cross-reviewing each other's changes.
- The previous gap-closure plan (`reference-branch:kg/reference-2026-09-29/ops/gap-closure-2026-09-29/plan.md`) feeds a rebuild plan. Its principles table and task designs stay valid; its file-level steps assumed the old code.

## Using the reference branch — implementation interpretation

The owner asks to use the reference branch "properly, without blindly copying". The rebuild plan (`ops/rebuild-2026-09-29/plan.md`) gives every reference component exactly one disposition:
- **Vendor, trimmed and hash-pinned:** only the rules kernel (`lib.rs`, `py_random.rs`, `econ_attrib.rs`), because it must match Kaggle's Python engine exactly.
- **Port after review:** the native grammar, rewards, device mask tables and benchmark harness.
- **Rebuild, with the reference as test oracle:** the observation encoding (named per-entity tensors instead of the flat v2 vector sliced at fixed offsets), the environment bindings, the codec, the model, the trainer seams, the configs and BC training.
- **Reference only:** scripted bots until evaluation needs a few of them, the v2 experiments, and the run receipts.

Lessons from the reference (raw-bank winners, truncation reward, seed streams, the CUDA fault (since traced to a compiler GEMM overflow), cadence, evaluation seed, lost observation facts) are requirements mapped to tasks. This is the implementer's interpretation of the directive, not an owner adoption of the specific table.

## Task 1.1 — rebuilt rules kernel

Claude's brief review approved the standalone package and required the Python
checker/pytest and separate engine preparation invocations; the owner requested
implementation. This is scoped to Task 1.1. The reviewed brief remains at
`ops/rebuild-2026-09-29/briefs/1.1-rules-kernel.md`; the review and actual command
receipts are linked in the sources and `ops/rebuild-2026-09-29/1.1/results.md`.

The source audit pins `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. The manifest
accounts for all 125 reference paths, retaining 12 including licensing,
append-only provenance and four traces, and excluding 113 with per-file reasons.
Exactly seven module declarations are removed from `lib.rs`, producing SHA-256
`c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd`.
`py_random.rs`, `econ_attrib.rs` and the RNG tests retain their original bytes.
Cargo alone removes Rayon and its unused lockfile closure; target cleanup removes
the excluded binary and cdylib. No model, bot, collector or trainer is imported.

The retained 41 unit and nine RNG tests pass. Nine new integration tests pass:
one API test, four comparator regressions and four native-reset replays totaling
2,876 transitions and 2,880 snapshots. State, numeric representation, object key
insertion order at every depth of public and private state, statuses, typed
rewards and terminal banks match the pinned 1.32.7 official-engine traces. The
plain-equality private comparator fails its regression before repair; Claude's
implementation review extended the order check from three private fields to all
objects (including public market maps), test-first. A deliberately corrupted
initial state also fails the first episode before restoration. All 59 engine
tests pass with none ignored. Final offline `just prepare` also
passes 769 Python tests (three platform skips), including 47 checker regressions;
build, formatting, lint, typing and documentation checks pass.

Verification round 1 (copied to
`ops/rebuild-2026-09-29/1.1/verify-r1/verify-1.1-r1.md`) approved with edits and no blocking findings. Its probes showed
that updated manifest declarations could authorize a LICENSE change or remove
the trim provenance appendix. The checker now fixes the editable set
(`lib.rs`, `Cargo.toml`, `Cargo.lock`, `VENDORED_FROM.md`), derives the exact
trimmed lockfile, and pins the appendix hash. Eight `check()`-level regressions
(55 checker tests) drive a copy of the package; three failed before the repair
(`ops/rebuild-2026-09-29/1.1/verify-r1/`). Four gitignored `replay-*.json`
receipts were force-added, and the receipt opening now matches the 59-test commit.

The package stays standalone, with its own lockfile and no root dependency or
shared workspace. This is Claude's accepted refinement of the parent plan's
workspace wording. Root Cargo files and Rust sources stay byte-identical and
the root suite remains 155 passed/two ignored. Keep the engine's numeric features
and reopen L4's test-only `fixture_float` repair at the first compiled root
consumer (potentially Task 1.3, certainly needed by Task 1.4).

Adaptation inventory: the 12 retained engine paths plus manifest/replay test;
`scripts/check_engine_trim.py`, `tests/tools/test_check_engine_trim.py`, `justfile`,
`rustfmt.toml`, `docs/rules-parity-coverage.md`, this Decision/index/log, and receipts
under `ops/rebuild-2026-09-29/1.1/`. The manifest records non-engine change reasons.
Both preparation entry points run the checker and separate engine fmt/Clippy/test
commands. Raw checks exposed inherited formatting drift and six style-only Clippy
findings. Preserve the byte contract with two exact formatter ignores (`lib.rs`,
`econ_attrib.rs`) and three engine-only Clippy allowances (`too_many_arguments`,
`collapsible_if`, `needless_range_loop`), re-denied for authored replay code.
These are explicit implementer deviations supported by failed-check receipts,
not an owner policy to weaken linting. Independent Codex review checked this
boundary; Claude's implementation review confirmed it and committed the task.
Engine Cargo invocations in `justfile` use `--locked` without `--offline`, so a
fresh machine can fetch crates; offline hosts set `CARGO_NET_OFFLINE=true`.

Existing-concept search covered this restart Decision, native-buffer and shared-PPO
records, including negative evidence from the earlier `arbitrary_precision`
failure. Independent sources/checks are the pinned Git tree, official trace
headers, test-first failures, source/hash verification and the actual test suites.
Future changes must preserve that custody, exercise new rules against real oracles,
and revisit feature unification at integration. These checks qualify only this
kernel on four recorded worlds plus unit/RNG cases. They establish neither
exhaustive malformed-input parity, fresh Python differential testing, game adapter
completion, playing strength nor GPU throughput. No training or network operation
ran. Pre-existing dirty cookbook bytes and checksums are preserved in the receipt.
