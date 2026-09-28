---
type: "Decision"
title: "V3 reuses Isaiah infrastructure with Kaggriculture semantics"
description: "Retain the starter training path and useful model infrastructure while adapting the engine, observations and actions to Kaggriculture."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:AGENTS.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/one-trainer-extend-selfplay-never-fork.md"}]
---

# V3 reuses Isaiah infrastructure with Kaggriculture semantics

The owner requests: “Reuse everything you can in here, the training infra etc., adapt the model as well, but replace the rust data pipeline, action heads, etc., for ~/kaggriculture-v2.” The owner also requests “Copy useful cookbook decisions and discipline from v2 cookbook as well,” confirms the Isaiah starter, and directs “just clone the repo in and start adapting”.

This repository is Kaggriculture v3. Preserve the imported starter's history and license. Adapt its training infrastructure and model to Kaggriculture's actual observation, action and engine semantics using the qualified v2 Rust implementation as source material. The owner subsequently specifies: “note that it must follow the current v3 data pipeline (RUST I/O, Training PPO, etc.)”. **The v3 starter's Rust/PyO3 caller-owned buffer I/O and existing PPO loop are canonical.** Reuse v2 engine semantics behind the v3 adapter; do not transplant its collector or training loop. Keep one canonical trainer, `scripts/run_ppo.py`; extend its shared training path instead of copying a second PPO loop. Diagnostics may remain separate when they do not train a deployment policy.

The one-trainer discipline comes from v2's owner quote: “禁止另起訓練器 沿用現在訓練（沒有feature就加進去) 器保證訓練質素/效率.” Mapping that discipline to the starter trainer is the implementation interpretation of today's reuse instruction; v2's old `selfplay.py` path is not imported as a constraint.

## Model and reward boundary — owner clarification

The owner subsequently directs:

> dont take model implementation v2, but you can take away the reward shaping from v2, such as econ shaping, our-opp bank, win/less/draw

Reuse the **starter transformer infrastructure with newly written Kaggriculture stems and action heads**. Do not import v2's model implementation, encoder, attention stack or trainer. V2 game-engine semantics, feature coordinate meanings, action grammar/codec contracts and selected reward formulas are permitted source material. The [[../references/explicit-game-tokens-and-grammar-replace-orbit-heads|model record]] names the implementation boundary; the [[../references/reward-reuse-preserves-objective-and-critic-semantics|reward record]] distinguishes exact v2 formulas from a newly selected v3 recipe. No v2 reward coefficient becomes an optimal v3 default merely because the formula can be reused. The owner reiterates that most of the model architecture should come from Isaiah and that training performance problems must be addressed. Accordingly retain the starter transformer/attention/training mechanisms, justify game-specific replacements, and measure actual bottlenecks rather than claiming performance from ancestry. This does not import the v2 neural implementation.

## Knowledge boundaries

Subject: **kaggriculture-v3**. A bounded piece of repository work is a **change**. Keep durable knowledge in tracked `cookbook/`; use `ops/` for change/run artifacts and git for implementation history. The vault exposes the bundle by symlink and a sibling Base. These are inherited working conventions chosen to carry out the instruction, not fabricated interview answers. Comparable results later need dated outcomes, denominators, seeds/seats/opponents, configuration and status; there are no v3 result entries, rankings or empirical lessons at setup.

Prior-project results remain scoped to their source version, inputs and host. Do not transfer v2 lane names, Myolie's fixed experiment ID, Dream-RSI mandates, pod allocation, retired checkpoints, rankings or numerical gains. A replaced game interface needs its own meaningful parity checks; passing software checks alone establishes neither GPU throughput nor competitive strength.
