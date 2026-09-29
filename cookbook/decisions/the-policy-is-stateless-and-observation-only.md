---
type: "Decision"
title: "The policy is stateless and observation-only"
description: "Carry forward explicit current-observation tokens and the prohibition on opponent identity conditioning; Isaiah's StatelessTransformerV1, never his recurrent model, is the reference contract, with the same layer topology and only game I/O differing; old checkpoint strength does not transfer."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "user-directive:2026-09-29:model-size-6-10m"}, {"resource": "user-directive:2026-09-29:same-layer-topology"}, {"resource": "user-directive:2026-09-29:refer-to-isaiah-stateless-approach"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gap-closure-2026-09-29/plan.md"}, {"resource": "user-directive:2026-09-29:reject-python-autoregressive-pipeline"}, {"resource": "reference-branch:kg/reference-2026-09-29/ops/gpu-sps-2026-09-29/results.md"}, {"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/myolie-is-a-stateless-agent.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/references/myolie-opponent-context-needs-explicit-requalification.md"}]
---

# The policy is stateless and observation-only

Today's instruction to reuse useful v2 decisions is applied to two recent, explicit owner constraints. Original v2 quotes:

> 移除recurrent memory是勢在必行，留顯式tokens/現有tokens便可以

> 除掉actor身分設定吧，並明言禁止這類東西在我們的線再次出現

The v3 model acts from explicit current-observation tokens. Do not introduce between-turn recurrent hidden state, carried memory, temporal memory-window training or new history tokens. The initial port allowed a reset-per-observation current-action prefix decoder as an interpretation of statelessness. The owner subsequently rejected that Python-driven autoregressive/GRU design for failing the efficient entire-pipeline requirement; that earlier permission no longer governs the current architecture. The implemented architecture uses batched heads with no GRU or Python frame loop; its qualification scope is in the [[../references/explicit-game-tokens-and-grammar-replace-orbit-heads|model Reference]]. Existing historical counters exposed by the legal observation may be represented. Adapting current-observation tokens to this game's fields is part of the requested port.

## Isaiah's stateless transformer is the reference

On 2026-09-29 the owner directs:

> Make sure we are refering to his stateless approach, thanks a lot.

Source: `user-directive:2026-09-29:refer-to-isaiah-stateless-approach`, the owner's reply during gap-closure planning. The reference model is Isaiah's `stateless_transformer_v1` (`StatelessTransformerV1`), whose contract is upstream `docs/model-architecture.md` at `32b3ec9`: one encode per current observation, no hidden state, and a named token sequence with player, board-scratch, actor-plan and per-player critic value tokens. `evaluate_actions` returns log-probs, entropy and values from one encode, and teachers are stateless. `recurrent_transformer_v1` is never the reference. Kaggriculture changes only game stems and action heads, and each deviation must be justified by game semantics. The owner then adds:

> Yes, we have different games, so we had different action heads, etc., these kind of things, but the model layers/ topologies should remain. We are just playign different games.

Source: `user-directive:2026-09-29:same-layer-topology`. The Kaggriculture model uses Isaiah's layer classes, arrangement and roles: `ObservationInputStem` stems (hidden width `embed_dim·mlp_ratio`), separate learned per-role token parameters (player, board scratch, actor plan, per-player critic value), `TransformerBlock` trunk with final LayerNorm, an `OutputProjectionMLP` critic head per critic value token, and a `3D → D` actor input projection over [entity, player, plan]. Only the game I/O differs: input channel widths, categorical fields fed to the stems as one-hot channels (as Isaiah feeds Orbit's categorical features), and the action heads. The port restarted from Isaiah's clean base to build it this way ([[restart-the-port-from-isaiahs-clean-base|restart Decision]]).

No input, embedding, head, loss, reward, normalization or checkpoint selection may condition on opponent slug, league class, ID, per-opponent baseline or per-class return normalization. Actor and critic are both covered. Opponent collection mixes and evaluation panels are allowed, without telling the model who the rival is. Diagnostic labels identify evidence, not policy input. Checkpoint schemas must reject prohibited identity-bearing state.

This is a scoped adoption of owner-directed design constraints, not an empirical finding that all memory or identity-conditioned methods fail. The v2 ban originally removed value-head conditioning, even though the owner called it actor identity. V2's corrected record shows the recurrent gradient horizon was not merely 32 turns. Its reset collapse demonstrated that checkpoint's memory dependence; no historical checkpoint score transfers to a stateless derivative or v3. Do not import the obsolete m21 conversion or distillation prescription.

## Model size — owner, 2026-09-29

> 6-10M ok as long as topologies aligned with isaiah.

Source: `user-directive:2026-09-29:model-size-6-10m`. The parameter budget is 6–10M. The trunk stays on Isaiah's 6m ladder settings (width 256, 8 heads so head_dim 32, `mlp_ratio` 2.0) and reaches the budget through depth. Depth 8 is the planned value, ≈6.2M with the game stems and heads; the exact count is confirmed once the model is built.
