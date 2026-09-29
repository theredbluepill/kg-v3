You are Codex. READ-ONLY review of Claude's brief `ops/rebuild-2026-09-29/briefs/2.3-action-heads.md` (on branch kg/rebuild-model).
Check it against:
- the contract `docs/kaggriculture-contract.md` (v4 action section)
- the reference design and oracle: `git show kg/reference-2026-09-29:python/owl/model/kaggriculture.py` (`_batched_policy`, `_field`, `_field_density`, `_decode`) and `python/owl/kaggriculture/gpu_sampling_grammar.py`
- the grammar `engine_rs/src/myolie_sampler.rs`
- Isaiah's actor blocks on this branch: `python/owl/model/stateless_transformer_v1.py` (`source_actor_input_proj`, `DiscreteTargetsActor`, `get_input_layers` / `get_output_layers`, `_ACTOR_HEAD_INIT_GAIN`) and `python/owl/model/actor/`
Focus:
1. Are the sampling and replay semantics exactly equivalent to the reference? Cover HIRE coupled-Gumbel, STOP marginalization, quantity decoding and the mask indexing per stage.
2. Does the `GrammarTables` interface cover every mask the grammar needs, including actor-count and order-limit dependence?
3. Muon and initialization parity with Isaiah's actor. Does Isaiah exclude actor embedding tables? What gain do his head outputs use?
4. Is removing the per-forward validity sync safe, given the native env enforces admission?
5. Are the tests sufficient, above all the replay invariant (the L6 lesson)?
FINAL REPORT: numbered findings (blocker / should-fix / note) with concrete edits; a verdict: APPROVE, APPROVE WITH EDITS, or REVISE.
