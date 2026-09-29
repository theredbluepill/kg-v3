---
type: "Decision"
title: "V3 reuses Isaiah infrastructure with Kaggriculture semantics"
description: "Deliver one complete final Kaggriculture adaptation of the efficient Isaiah pipeline that respects Isaiah's own principles in their game form; partial resource mechanisms are not finished deliverables."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-29"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "user-directive:2026-09-29:respect-isaiah-principles"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gap-closure-2026-09-29/plan.md"}, {"resource": "user-directive:2026-09-29:scale-environments-keep-ppo-minibatch"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/kaggriculture_2rank_32env_fixedppo.yaml"}, {"resource": "user-directive:2026-09-29:complete-final-delivery"}, {"resource": "user-directive:2026-09-29:reject-python-autoregressive-pipeline"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "repository:scripts/run_ppo.py"}, {"resource": "repository:AGENTS.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/one-trainer-extend-selfplay-never-fork.md"}]
---

# V3 reuses Isaiah infrastructure with Kaggriculture semantics

The owner requests: “Reuse everything you can in here, the training infra etc., adapt the model as well, but replace the rust data pipeline, action heads, etc., for ~/kaggriculture-v2.” The owner also requests “Copy useful cookbook decisions and discipline from v2 cookbook as well,” confirms the Isaiah starter, and directs “just clone the repo in and start adapting”.

This repository is Kaggriculture v3. Preserve the imported starter's history and license. Adapt its training infrastructure and model to Kaggriculture's actual observation, action and engine semantics using the qualified v2 Rust implementation as source material. The owner subsequently specifies: “note that it must follow the current v3 data pipeline (RUST I/O, Training PPO, etc.)”. **The v3 starter's Rust/PyO3 caller-owned buffer I/O and existing PPO loop are canonical.** Reuse v2 engine semantics behind the v3 adapter; do not transplant its collector or training loop. Keep one canonical trainer, `scripts/run_ppo.py`; extend its shared training path instead of copying a second PPO loop. Diagnostics may remain separate when they do not train a deployment policy.

The one-trainer discipline comes from v2's owner quote: “禁止另起訓練器 沿用現在訓練（沒有feature就加進去) 器保證訓練質素/效率.” Mapping that discipline to the starter trainer is the implementation interpretation of today's reuse instruction; v2's old `selfplay.py` path is not imported as a constraint.

## Model and reward boundary — owner clarification

The owner subsequently directs:

> dont take model implementation v2, but you can take away the reward shaping from v2, such as econ shaping, our-opp bank, win/less/draw

Reuse the **starter transformer infrastructure with newly written Kaggriculture stems and action heads**. Do not import v2's model implementation, encoder, attention stack or trainer. V2 game-engine semantics, feature coordinate meanings, action grammar/codec contracts and selected reward formulas are permitted source material. The [[../references/explicit-game-tokens-and-grammar-replace-orbit-heads|model record]] names the implementation boundary; the [[../references/reward-reuse-preserves-objective-and-critic-semantics|reward record]] distinguishes exact v2 formulas from a newly selected v3 recipe. No v2 reward coefficient becomes an optimal v3 default merely because the formula can be reused. The owner reiterates that most of the model architecture should come from Isaiah and that training performance problems must be addressed. Accordingly retain the starter transformer/attention/training mechanisms, justify game-specific replacements, and measure actual bottlenecks rather than claiming performance from ancestry. This does not import the v2 neural implementation.

## Entire-pipeline correction — 2026-09-29

The owner rejects the Python-driven autoregressive decoder and mere PyO3 interface alignment as fulfillment of Isaiah reuse. Preserve Isaiah's efficient complete data/model/PPO path. The implementation lead acknowledges that the initial decoder design missed this requirement; CPU correctness and transformer ancestry do not override that correction. The replacement work and rejected-source custody are recorded in the [[../references/explicit-game-tokens-and-grammar-replace-orbit-heads|model contract]] and `ops/gpu-sps-2026-09-29/results.md`. Batched heads and native masks are now the implemented path; their qualification scope lives in the model and [[../references/native-game-semantics-use-v3-owned-buffers|native]] References.

## Complete final delivery — owner correction, 2026-09-29

After the assistant proposed a seed-only allocator as a first version with other coordination problems left for later, the owner states:

> 這裏沒有第一版，第二版，只有最終版

Source: `user-directive:2026-09-29:complete-final-delivery`, the owner's direct reply in the architecture discussion. Deliver one complete final adaptation within the agreed task scope. A seed-budget mechanism alone cannot stand in for the full game policy and training integration. Withdraw the proposed partial-delivery framing; this directive does not adopt the discussed seed allocator or any other unimplemented architecture.

Implementation interpretation: internal bounded checks, profiling and revisions remain ways to finish the work, not separate incomplete deliverables. The complete design must account for labor/resource allocation, tile and within-turn action interactions, investment payback, sale timing and market adaptation through legal observation, representation, choice and executable action. Its sampling/replay probabilities, native execution, canonical PPO and complete-work performance must be checked together. Name unresolved attribution and implementation gaps rather than deferring required behavior to a later version or declaring completion from component tests. This instruction supplies no numerical performance floor, win-rate target or proof of optimality; reported qualification must still match actual evidence.

## Knowledge boundaries

Subject: **kaggriculture-v3**. A bounded piece of repository work is a **change**. Keep durable knowledge in tracked `cookbook/`; use `ops/` for change/run artifacts and git for implementation history. The vault exposes the bundle by symlink and a sibling Base. These are inherited working conventions chosen to carry out the instruction, not fabricated interview answers. Comparable results later need dated outcomes, denominators, seeds/seats/opponents, configuration and status; there are no v3 result entries, rankings or empirical lessons at setup.

Prior-project results remain scoped to their source version, inputs and host. Do not transfer v2 lane names, Myolie's fixed experiment ID, Dream-RSI mandates, pod allocation, retired checkpoints, rankings or numerical gains. A replaced game interface needs its own meaningful parity checks; passing software checks alone establishes neither GPU throughput nor competitive strength.


## Isaiah's principles on a different game — owner directive, 2026-09-29

> The plan must respect Isaiah principals while on different game (Kaggriculture)

Source: `user-directive:2026-09-29:respect-isaiah-principles`, the owner's instruction during gap-closure planning. Isaiah's principles come from his own sources at `32b3ec9` (`AGENTS.md`, `README.md`, config headers, `docs/model-architecture.md`), not from our paraphrase. They cover the stateless transformer contract; 16 optimizer steps/iteration with the split set by a memory smoke; the same global config on any hardware; an LR schedule defined over env steps; his model-size ladder; the winner-distribution critic; last-best teacher stabilization; docs as contracts; refactoring over compatibility shims; parity-first engine testing. Where Kaggriculture differs (two players, seat-private observations, grammar heads, owner econ shaping, native engine), the principle stays and only its game form changes. The principle-by-principle mapping and the tasks that satisfy it are in `ops/gap-closure-2026-09-29/plan.md`; recipe choices resolve toward Isaiah under the [[recipe-choices-align-to-isaiah-without-owner-escalation|alignment Decision]].

## Historical: fixed PPO minibatch during environment scaling

On 2026-09-29 the owner asked: “can you keep the original PPO minibatch, and scale the environments only? is it better?” That governed the environment-scaling experiment (one segment/minibatch, accumulation 2); its results are in `ops/gpu-sps-2026-09-29/results.md`. The later alignment Decision withdraws that fixed minibatch as a constraint on the Isaiah-aligned recipe.
