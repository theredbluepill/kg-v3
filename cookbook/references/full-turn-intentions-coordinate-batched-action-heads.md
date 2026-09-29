---
type: "Reference"
title: "Full-turn intentions coordinate batched action heads"
description: "Optional sampled-intention hypothesis, not a required coordinator: Isaiah and the current model already share scratch/plan context; necessity and cost-benefit remain unproved."
tags: ["kaggriculture-v3", "adaptation", "model", "coordination"]
status: "proposed"
generated: {"by": "openai/codex", "at": "2026-09-29"}
sources: [{"resource": "user-question:2026-09-29:coordination-necessity-and-retained-plan-tokens"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/isaiah-comparison.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/pre-necessity-review/manifest.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/configs/model/kaggriculture.yaml"}, {"resource": "external:https://github.com/IsaiahPressman/kaggle-orbit-wars/blob/32b3ec900ad406eedd965f53a1a0f4490d31c589/python/owl/model/stateless_transformer_v1.py"}, {"resource": "user-directive:2026-09-29:prepare-complete-efficient-coordination-design"}, {"resource": "reference-branch:kg/reference-2026-09-29/docs/coordination-system-design.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/plan.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/engine-audit.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/probability-audit.mjs"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/probability-audit.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/parameter-envelope.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/source-manifest.json"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/coordination-design-2026-09-29/source-excerpts.md"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/lib.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/engine_rs/src/myolie_features.rs"}, {"resource": "reference-branch:kg/reference-2026-09-29/python/owl/model/kaggriculture.py"}, {"resource": "repository:python/owl/train/ppo.py"}, {"resource": "external:https://arxiv.org/pdf/1707.06347"}, {"resource": "external:https://diffusion-ppo.github.io/"}, {"resource": "external:https://proceedings.neurips.cc/paper/2019/hash/f816dc0acface7498e10496222e9db10-Abstract.html"}]
---

# Full-turn intentions coordinate batched action heads

## Optional hypothesis; a dedicated coordinator is not established as necessary

The owner asks whether coordination needs a special module when Isaiah does not
use the proposed U/A stage, and points out that our scratch/plan tokens remain.
Pinned source review confirms both architectures already provide shared
observation attention and plan context. Current Kaggriculture uses four scratch
tokens plus independent player, plan and critic queries per separately encoded
legal seat view, with shared initial token weights. The player readout was
restored under the owner's later directive; actor inputs now contain entity,
player and plan. The shared observation/plan path was already present before
that repair. This does not adopt the U/A candidate.

Independent sampling limits stochastic joint distributions, not all coordinated
behavior. Distinguishable actors can choose complementary actions from the same
observation without sampled U. The 50/50 toy is not evidence that a trained
baseline cannot allocate work. Shared resources motivate diagnostics, but do
not establish the need or cost-benefit of another neural stage.

The assistant therefore withdraws the earlier recommendation to adopt U/A as
the implementation direction. Retain the existing shared trunk and batched
heads; keep this design as an optional hypothesis to revisit for an attributable
failure. This is an assistant correction supported by source review, not an
owner decision to prohibit or adopt an architecture. The source comparison and
reopening conditions are in `ops/coordination-design-2026-09-29/isaiah-comparison.md`.

## Retained candidate design

The owner asks the assistant to seriously prepare the complete efficient
coordination system after rejecting a seed-only staged-delivery proposal. The
assistant prepared the candidate in `docs/coordination-system-design.md`:
encode the legal current observation once with the Isaiah trunk; sample a full
intention program U in batches; communicate U through one starter attention
block; sample final full program A with batched heads; execute only A through the
existing Rust lifecycle. This records a design deliverable, not implementation,
new policy strength, numerical performance acceptance or owner adoption.

The explicit intention includes every actor and ordered market slot, quantities,
EMPTY and STOP. This lets final heads condition on actual shared sampled choices.
A deterministic observation token alone does not create that stochastic
dependence. No U survives the current observation, and no opponent identity or
private view enters the actor or critic. Final market heads receive distinct
`[seat,11,width]` communicated states; pooling them back to one market vector
would lose the intended order-specific coordination.

## Engine boundary changes what can be promised

Independent source review found a whole-program PLANT admission test preceding
farmer/hands execution. Tile and transfer effects then follow actor ordinal,
before all markets. Useful same-tile chains and sale-funded later orders must
remain representable; one-worker-per-tile and current-cash-only purchase masks
would remove them. The current float encoding loses inventory insertion order
and some public configuration facts, so the specification includes an explicit
observation-schema correction instead of claiming a lossless shadow simulator.

Rival simultaneous orders affect market prices/fills, and future maintenance has
randomness. The proposed network supplies learned coordination over the full
program, not guaranteed profitable execution or privileged future settlement.
It preserves current grammar support and adds no hidden post-sampling repair.

## Probability and efficiency contract

PPO records both U and A and uses `log q(U|o) + log r(A|o,U)`, with one joint
seat-turn ratio. This is an augmented policy action, not the marginalized native
action probability. U and A have independent STOP masks; unscored U suffixes are
zeroed before communication. Critic remains V(o). No separate U entropy bonus is
added; final conditional entropy retains its nominal coefficient as an explicitly
named exploration surrogate, not exact marginal entropy. Shared parameters still
receive the final regularizer's gradients.

The design retains one observation trunk and adds one batched head pass and one
small intention attention block. U stays on device and only final A crosses the
native execution boundary. There is no Python worker/frame scan. Extra latent
variance, stricter joint KL, final conflicts, inactive-U collapse, rollout memory
and actual GPU cost remain important qualification questions, not hidden gaps to
be called solved by adding tokens. The full design names all code integration,
game-mechanism, probability, privacy, DDP, complete-work and held-out checks.

## Actual checks and future consequence

Before creating this concept, the scope/stateless/debugging/throughput/evaluation
Decisions and native/model/PPO References were searched for allocation, seed,
PLANT, Gumbel and coordination. Existing concepts cover engine/PPO contracts but
not this proposed full-turn intention distribution. Separate source-semantics,
probability and execution-architecture reviews supported the written design.

`node ops/coordination-design-2026-09-29/probability-audit.mjs` passes an exhaustive
16-state toy probability audit: normalization, replay logp, finite-difference
return gradients, STOP marginalization, and joint/native entropy/KL distinctions.
Omitting U incorrectly zeros nonzero gradients. A hand-specified fair two-worker
example reaches .9802 exactly-one probability versus .5 for independent equal
marginals; it still has nonzero conflict probability and is not a learned game
result. Another example holds native action probabilities unchanged while joint
KL rises, demonstrating a real objective tradeoff rather than a free improvement.

A pre-player-repair CPU constructor/shape count gives a 9,267,647-parameter subtotal
for the specified additions, excluding new observation stems. This is a capacity
envelope, not an instantiated coordinated model or performance check. The current
base model has 8,353,727 parameters after the independent player repair; any U/A
implementation must recount parameters within the existing budget. Source hashes
and inspected excerpts are retained with the design; ongoing dirty repository
changes are not attributed to this task.

The follow-up source check pins upstream main at
`32b3ec900ad406eedd965f53a1a0f4490d31c589`, inspects its shared actor-plan inputs
and within-source sampler, and traces the current model's retained special-token
and plan-broadcast path. Independent source review supports this comparison.
Pre-correction dirty documents are preserved byte-for-byte with hashes under
`pre-necessity-review`; original check receipts keep their original source scope.

Use U/A only as a candidate for a diagnosed limitation. If selected, implement
and review its full path together; do not cite its toy audit as necessity, native
correctness, GPU speed, learning strength, full action feasibility or final
delivery. No learner or pod action was performed to prepare or review the design.
