---
type: "Decision"
title: "Restart the port from Isaiah's clean base"
description: "Main and the working branch restart from Isaiah's 32b3ec9; the prior port is a reference branch for grammar, tokens and the Rust engine; the cookbook and Claude/Codex setup carry over; the model keeps Isaiah's layer topology; one BC rerun and RTX PRO 6000 verification follow."
tags: ["kaggriculture-v3", "decisions"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
decider: "Owner: restore kg/isaiah-gap-closure and main to 32b3ec900ad406eedd965f53a1a0f4490d31c589 and work again from a clean state with Codex, keeping a reference branch and carrying the cookbook with the .claude/.codex setup."
sources: [{"resource": "user-directive:2026-09-29:update-kaggle-environments-1-32-7"}, {"resource": "repository:pyproject.toml"}, {"resource": "repository:uv.lock"}, {"resource": "repository:ops/rebuild-2026-09-29/plan.md"}, {"resource": "user-directive:2026-09-29:restart-from-isaiah-clean-base"}, {"resource": "user-directive:2026-09-29:same-layer-topology"}, {"resource": "user-directive:2026-09-29:extra-bc-rtx6000-codex"}, {"resource": "reference-branch:kg/reference-2026-09-29"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:ops/pre-commit"}]
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

## kaggle-environments 1.32.7 — owner, 2026-09-29

> Found that Isaiah's lockfile pins kaggle-environments 1.29.0, which lacks Kaggriculture, while 1.32.7 (already cached locally) has it. this we will have to update it.

Source: `user-directive:2026-09-29:update-kaggle-environments-1-32-7`. Isaiah's project took `kaggle-environments` from git HEAD, locked at 1.29.0, which has no Kaggriculture environment. It is now pinned to `==1.32.7` via `uv remove` + `uv add`, which also removes the git source override. 1.32.7's `envs/kaggriculture/kaggriculture.py` has SHA-256 `bc8a5487…`, the engine the vendored Rust kernel targets.

Isaiah's critical pins are unchanged: torch 2.9.0, triton 3.5.0, flash-attn 2.8.3, numpy, pydantic, wandb, mypy. The other 55 changed dependency names (plus the `owl` requirement record) are kaggle-environments' own dependency tree (e.g. jax 0.10 → 0.11.2; `click` is shared with W&B, whose live telemetry was not exercised). Checks with 1.32.7: `just prepare` passes, with Rust 155 passed / 2 ignored (expensive action-angle audits; Orbit replay and generation parity pass) and Python 722 passed / 3 skipped. Codex independently reproduced the lock byte-for-byte from the recorded `uv remove` + `uv add` (`ops/deps-independent-verification-fb65e1f/`).
