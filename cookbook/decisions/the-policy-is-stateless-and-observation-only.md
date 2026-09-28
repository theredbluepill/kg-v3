---
type: "Decision"
title: "The policy is stateless and observation-only"
description: "Carry forward explicit current-observation tokens and the prohibition on opponent identity conditioning; old checkpoint strength does not transfer."
tags: ["kaggriculture-v3", "decisions"]
status: "stable"
generated: {"by": "openai/codex", "at": "2026-09-28"}
decider: "Owner requests reuse of useful v2 decisions and discipline, then directs: just clone the repo in and start adapting; scoped implementation interpretation below."
sources: [{"resource": "user-directive:2026-09-28:kaggriculture-v3-reuse-and-adapt"}, {"resource": "repository:ops/cookbook-setup-checks.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/decisions/myolie-is-a-stateless-agent.md"}, {"resource": "external-repository:/Users/poonszesen/kaggriculture-v2/cookbook/references/myolie-opponent-context-needs-explicit-requalification.md"}]
---

# The policy is stateless and observation-only

Today's instruction to reuse useful v2 decisions is applied to two recent, explicit owner constraints. Original v2 quotes:

> 移除recurrent memory是勢在必行，留顯式tokens/現有tokens便可以

> 除掉actor身分設定吧，並明言禁止這類東西在我們的線再次出現

The v3 model acts from explicit current-observation tokens. Do not introduce between-turn recurrent hidden state, carried memory, temporal memory-window training or new history tokens. A decoder may summarize the current action prefix within a turn and reset it on every observation; this is an explicit adaptation interpretation, not a claim that the owner separately adopted every old Myolie module prohibition for v3. Existing historical counters exposed by the legal observation may be represented. Adapting current-observation tokens to this game's fields is part of the requested port.

No input, embedding, head, loss, reward, normalization or checkpoint selection may condition on opponent slug, league class, ID, per-opponent baseline or per-class return normalization. Actor and critic are both covered. Opponent collection mixes and evaluation panels are allowed, without telling the model who the rival is. Diagnostic labels identify evidence, not policy input. Checkpoint schemas must reject prohibited identity-bearing state.

This is a scoped adoption of owner-directed design constraints, not an empirical finding that all memory or identity-conditioned methods fail. The v2 ban originally removed value-head conditioning, even though the owner called it actor identity. V2's corrected record shows the recurrent gradient horizon was not merely 32 turns. Its reset collapse demonstrated that checkpoint's memory dependence; no historical checkpoint score transfers to a stateless derivative or v3. Do not import the obsolete m21 conversion or distillation prescription.
