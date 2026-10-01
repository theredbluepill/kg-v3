---
type: "Decision"
title: "Keep the sprint in pure self-play without PFSP or lambda scheduling"
description: "Owner decision for the final sprint (2026-09-30/10-01): train only in mirror self-play. The owner declined a PFSP league (\"maybe it's ok let's just focus in pure self-play, no need league\") and a lambda schedule (\"i meant let it be\"). All five sprint runs kept gae_lambda 1.0 with the 720-step horizon and plain last_best self-play. A PFSP implementation was started on branch kg/pfsp and stopped with no commits. Both remain untested future options; there is no evidence for or against them."
tags: ["kaggriculture-v3", "decisions", "training", "self-play"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
decider: "Owner, 2026-09-30/10-01 (Claude Code session): \"maybe it's ok let's just focus in pure self-play, no need league\" (PFSP league) and \"i meant let it be\" (lambda scheduling)."
sources:
  - resource: "user-directive:2026-09-30:pure-self-play-no-need-league"
  - resource: "user-directive:2026-09-30:let-it-be-no-lambda-schedule"
  - resource: "user-directive:2026-09-30:switch-back-to-self-play-and-earn-money-for-real"
  - resource: "repository:ops/sprint-2026-09-30/sprint-facts.md"
  - resource: "repository:ops/rebuild-2026-09-29/sprint-8gpu/launch.sh"
  - resource: "repository:configs/kaggriculture_4rank_margin.yaml"
  - resource: "repository:cookbook/decisions/the-policy-is-stateless-and-observation-only.md"
  - resource: "wandb-run:spoon/kg-v3/r4zqqs49"
---

# Keep the sprint in pure self-play without PFSP or lambda scheduling

## Decision

During the final sprint the agent offered two training changes. The owner declined both, verbatim (Claude Code session, 2026-09-30/10-01):

- **PFSP league** (prioritized fictitious self-play against a pool of past checkpoints): "maybe it's ok let's just focus in pure self-play, no need league".
- **Lambda scheduling** (changing GAE λ during training): "i meant let it be".

These follow the owner's earlier "let's switch back to self play no matter what" on the [[the-kaggriculture-v3-board|board]].

All five sprint runs, from c50 to 210M, used:

- mirror self-play;
- `configs/kaggriculture_4rank_margin.yaml` with `rl.horizon=720`, `rl.segments_per_minibatch=1` and a fixed `rl.gae_lambda=1.0`, as pinned by `ops/rebuild-2026-09-29/sprint-8gpu/launch.sh`;
- the last_best teacher and promotion.

## What happened to the declined options

- **PFSP.** An implementation was started on branch `kg/pfsp` (worktree `/Users/poonszesen/kg-v3-pfsp`) and stopped with no commits. No PFSP or league code is in this tree.
- **λ schedule.** Never implemented. λ was 1.0 throughout.

## Future options, not results

Neither option was run, so the sprint gives no evidence for or against either one.

- **PFSP.** It would be consistent with the [[the-policy-is-stateless-and-observation-only|stateless policy constraints]] only if the opponent pool is a collection mix that never conditions the actor, critic, losses, rewards, normalization or checkpoint selection on opponent identity.
- **Reopening.** Either option comes back only by an owner decision. One example would be when self-play and the anchor panel stop improving together. The sprint's panel gains through 200M give no such signal ([[promote-and-relaunch-on-anchor-panel-evidence|promotion Decision]]).
