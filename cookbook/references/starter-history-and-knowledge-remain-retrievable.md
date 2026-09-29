---
type: "Reference"
title: "Starter history and knowledge remain retrievable"
description: "The Isaiah source history/license, redirected remotes and selective cookbook lifecycle are retained with explicit verification limits."
tags: ["kaggriculture-v3", "adaptation"]
status: "verified-scoped"
generated: {"by": "openai/codex", "at": "2026-09-28"}
sources: [{"resource": "reference-branch:kg/reference-2026-09-29/ops/v3-port-checks.md"}, {"resource": "user-directive:2026-09-28:record-every-adaptation"}, {"resource": "repository:LICENSE"}, {"resource": "repository:AGENTS.md"}, {"resource": "repository:CLAUDE.md"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:ops/pre-commit"}, {"resource": "repository:.claude/settings.json"}, {"resource": "repository:.claude/hooks/cookbook-gate.mjs"}, {"resource": "repository:.claude/hooks/cookbook-lint.mjs"}, {"resource": "repository:.claude/hooks/cookbook-correction-check.mjs"}, {"resource": "repository:.codex/hooks.json"}, {"resource": "repository:.codex/hooks/cookbook-preflight.mjs"}, {"resource": "repository:.codex/hooks/cookbook-lint.mjs"}, {"resource": "repository:.codex/hooks/cookbook-correction-check.mjs"}, {"resource": "repository:kaggriculture-v3.base"}, {"resource": "repository:README.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/docs/upstream-orbit-wars.md"}]
---

# Starter history and knowledge remain retrievable

## What the sources record

The root Git history starts from Isaiah's `32b3ec900ad406eedd965f53a1a0f4490d31c589` (`add slideshow`). `origin` is `git@github.com:theredbluepill/kg-v3.git`; `upstream` is `https://github.com/IsaiahPressman/kaggle-orbit-wars.git`. These were read directly with `git log -1` and `git remote -v`. Root `LICENSE` retains IsaiahPressman's MIT notice and matches the imported commit. Root README now explains v3 setup and adaptation boundaries; the upstream Orbit Wars README is retained at `docs/upstream-orbit-wars.md`. This task makes no commit/push claim.

The cookbook adds selective v2 Decisions, a CUDA profiling Workflow, typed indexes, a newest-first log and a sibling Base. `AGENTS.md` retains starter engineering practices and merges owner autonomy, v3 pipeline constraints, and the cookbook contract. `CLAUDE.md` remains its existing symlink. Source-v2 quotations and SHA-256 identities are recorded in `ops/cookbook-setup-checks.md`; historical rankings and lane-specific mandates are not copied.

Current skill assets supply `.claude/hooks/`, `.claude/settings.json`, `.codex/hooks/`, `.codex/hooks.json` and executable `ops/pre-commit`; Git uses `core.hooksPath=ops`. There was no previous configured hook directory or non-sample Git hook to replace. Context restore, first governed-edit reminder, write/source lint, correction check and staged-note/log enforcement each retain their scoped function. Direct fixtures and hermetic suites pass as detailed in the setup receipt; live harness discovery/trust remains unverified.

The vault symlinks `cookbook/kaggriculture-v3` and `cookbook/kaggriculture-v3.base` to the repo. The Base scopes the subject folder and excludes reserved index/log notes; its browse and review views were opened and queried in Obsidian. Final row counts live in the receipt rather than becoming a second count here.

## Interpretation and consequence

Keep repository history/license and external source custody when adapting code. Read the scope Decision before changing the trainer, then the relevant port contract; later readers should be able to distinguish intended behavior, source inspection and executed checks. Record every material adaptation, as the owner explicitly requests, with reason, paths, checks and gaps. No v3 empirical Lesson or board exists until results justify it. The setup changes documentation/tooling behavior, not playing strength, CUDA performance or runtime hook trust.
