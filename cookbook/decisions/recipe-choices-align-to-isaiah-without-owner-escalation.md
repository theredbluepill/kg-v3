---
type: "Decision"
title: "Recipe choices align to Isaiah without owner escalation"
description: "Optimizer-step cadence and critic/value-distillation semantics are implementer choices resolved toward Isaiah scaling_6m and its own multi-GPU rule, not owner decisions; both are built into the rebuild from the start."
tags: ["kaggriculture-v3", "training", "decisions"]
status: "adopted"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
decider: "Owner: these are NOT my decisions please patch the cookbook or whatever, and align to Isaiah later on in the list."
sources: [{"resource": "user-directive:2026-09-29:recipe-choices-align-to-isaiah"}, {"resource": "user-directive:2026-09-29:diagnose-bc-and-preserve-training-config"}, {"resource": "user-directive:2026-09-29:scale-environments-keep-ppo-minibatch"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/upstream-scaling_6m.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/semantic-audit.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/isaiah-alignment-2026-09-29/results.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_4rank.yaml"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/model/kaggriculture.yaml"}, {"resource": "repository:python/owl/train/config.py"}, {"resource": "external-repository:https://github.com/IsaiahPressman/kaggle-orbit-wars/blob/32b3ec900ad406eedd965f53a1a0f4490d31c589/configs/winner_ce_6m_4x5090.yaml"}, {"resource": "external-repository:https://github.com/IsaiahPressman/kaggle-orbit-wars/blob/32b3ec900ad406eedd965f53a1a0f4490d31c589/python/owl/model/stateless_transformer_v1.py"}]
---

# Recipe choices align to Isaiah without owner escalation

Earlier the same day the owner directed: “你可以對齊全部參數吧 基本建設都在了isaiah, 我們盡量對齊”. An alignment review then listed two items as the owner's to decide: the optimizer-step cadence (batch options A/B plus `target_kl`) and the critic/value-distillation semantics. The owner replied:

> these are NOT my decisions please patch the cookbook or whatever, and align to Isaiah later on in the list.

Source: `user-directive:2026-09-29:recipe-choices-align-to-isaiah`, the owner's direct reply in the Claude Code alignment review.

## Rule

Differences between the v3 training recipe and pinned upstream `scaling_6m` are implementation choices, not owner decisions. Resolve each toward Isaiah and record any residual difference. Where literal alignment would contradict an adopted owner constraint (stateless observation-only policy, no opponent-identity conditioning, v3 Rust I/O with one trainer, no v2 model), use the nearest upstream-supported semantics and name the gap. Do not present alignment alternatives to the owner as a menu.

## Superseded framing

The owner directives “4096吧，default.” and “can you keep the original PPO minibatch, and scale the environments only? is it better?” governed the environment-scaling and capacity experiments. Those receipts remain valid history. Records that described 4,096 environments/rank, one segment per minibatch and accumulation 2 as owner-retained constraints on the Isaiah-aligned recipe are withdrawn. Implementation interpretation: the new directive governs future aligned runs; historical configs, snapshots and receipts stay unchanged.

## Alignment targets — queued, not yet applied

Implementation interpretation of “align to Isaiah” for the two items:

1. **Optimizer-step cadence.** Apply Isaiah's own multi-GPU rule from `winner_ce_6m_4x5090.yaml`: divide `n_envs` and `segments_per_minibatch` by world size, keeping the global batch, 16 optimizer steps/iteration and env steps/iteration of the single-GPU run. For `scaling_6m` (256 envs, spm 16, accumulation 1) this gives 128 envs/rank, spm 8, accumulation 1 on two ranks, and 64/4/1 on four ranks. `target_kl` stays null as upstream. Each iteration then collects 16,384 game transitions and takes 16 optimizer steps of 1,024 global transitions, matching upstream warm-up/cosine data exposure. Residual differences: two-player games only, and dense-game memory is unmeasured, so a memory smoke precedes any long run. Earlier 4,096-env SPS figures do not qualify this cadence.
2. **Critic and value distillation.** Upstream `scaling_6m` uses the default softmax winner-distribution critic (value 2p−1, MSE value loss, gamma 1) with a last-best teacher (policy KL 0.005, winner-probability value distillation 0.005). Upstream itself rejects `critic_mode='independent'` with value distillation, so alignment means implementing winner-distribution semantics rather than relaxing that guard. V3 runs one forward per seat-private observation; a critic that combines both seats' private inputs is neither Isaiah's architecture nor required. The nearest analog follows Isaiah's stateless critic: per-player critic value tokens (self and opponent) inside each seat's own observation feed a shared head, giving one logit per player, a softmax and value 2p − 1 (plan Task 3.1). That admits upstream's value distillation without reading the rival's private state. The range check is settled by source: `terminal_scale = 1 - caps` keeps shaped returns in [-1, 1].

After the restart, both targets are built in from the start of the rebuild rather than applied as later fixes; the rebuild plan orders the work.

## Limits

This record changes no training config and starts no run. The historical 4,096/spm 1/accumulation 2 cadence with teachers disabled lives only on the reference branch. After the [[restart-the-port-from-isaiahs-clean-base|restart]], the rebuilt configs must adopt these targets from the start, and alignment is complete only after GPU qualification.
