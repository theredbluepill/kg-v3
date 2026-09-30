# Kaggriculture model — conformance to Isaiah's stateless transformer

Reference: `StatelessTransformerV1` and upstream `docs/model-architecture.md` at `32b3ec9`. Each row reads **same**, **game form** (with the game reason) or **deviation → task**.

| Contract point | Kaggriculture | Status |
|---|---|---|
| Stems `raw → int(D·mlp_ratio) → D` | `ObservationInputStem` per entity group | same |
| Categorical inputs | One-hot channels concatenated to stem inputs; no `nn.Embedding` | same (Orbit also feeds one-hot channels) |
| Per-role learned tokens | Separate `player_tokens[2,D]`, `board_tokens[n,D]`, `actor_plan_tokens[1,D]`, `critic_value_tokens[2,D]` | game form: only the observed seat acts, so there's 1 plan token; the players are self and opponent |
| Player summary added to player tokens | `player_tokens + player_feature_proj(player_features)` before the trunk | same |
| Global feature token | `global_proj(global_features)` | same |
| Token order `[action entities][other entities][players][global][board][plan][critic]` | own actors first, then rival actors, tiles, shops, market | same |
| Masked tokens excluded as keys and zeroed | `token_mask`; output re-masked | same |
| Trunk `TransformerBlock × depth` + `final_norm` | Isaiah's classes via a typed `StatelessTransformerV1Config` | same |
| Flash packing and compile hook | `pack_sequence` / `unpack_sequence`, `torch.compile(..., dynamic=True)` | same, plus the compiled-GEMM overflow guard and cuBLAS-only compiled GEMMs (`max_autotune_gemm_backends = "ATEN"`, probed-stack check) |
| Named encoded fields | `KaggricultureEncoded` offsets from named counts | same |
| No hidden state | Every API rejects `hidden_state`; tested | same |
| Initialization | `_init_module`, input gain 1, residual `1/sqrt(2·depth)`, token std `D^-0.5` | same |
| Ladder | Width 256, 8 heads (head_dim 32), GELU, `mlp_ratio` 2.0; depth 8 for the owner's 6–10M budget | same ladder, deeper point |
| Critic `OutputProjectionMLP` per critic-value token, masked winner softmax (`masked_softmax(logits, still_playing)`), value `2p − 1` | `critic_head` over (self, opponent) tokens from the seat's own view, masked by the critic tokens' mask | same (game form: 2 players, both active). Deviation: an all-masked row returns Isaiah's uniform masked result instead of raising like his `_critic_logits` |
| (no Isaiah counterpart) | Optional `critic_offset_head` (`model.critic_offset`, default off): an `OutputProjectionMLP` on the self critic-value token, zero-initialized output, value `2p − 1 + o` | Deviation, owner-directed (2026-09-30, "per-player critic might be the way out?"): a new game head for the non-zero-sum own-bank reward. Off builds nothing and is byte-identical |
| Actor `3D → D` input projection over [entity, player, plan] | `actor_input_proj = nn.Linear(3D, D)`; unit frames use the own-actor hidden as the entity, market frames the plan hidden | same (game form: the market queue has no entity token) |
| Actor normalizes its input before the heads | `actor.source_norm` over the projected input (plus `market_position` for queue positions) | same (game form: queue position plays the role of Isaiah's source role token) |
| Action heads | `KaggricultureGrammarActor`: one `OutputProjectionMLP` per sampled grammar slot (9), within-frame prefix embeddings, `hidden_stage = base + prefix / sqrt(stage + 1)`; masks from `GrammarTables` plus unit-liveness, queue, HIRE-capacity and STOP overlays; exact coupled Gumbel-max sampling | game form: the Kaggriculture action is a grammar program, not Orbit launches |
| `evaluate_actions` replays through the sampling path | Same core, teacher-forced; support, length and canonical-equality flags with safe indices and one host check outside the core | same (game form: grammar admission is part of replay) |
| Log-prob / entropy layout | `event [E,2,252,12]`, `per_player_entity = event.sum(-1)`, `launch` zeros, entropy `components` per slot | game form of Isaiah's per-player-entity layout |
| Actor output init gain 0.01; token std `D^-0.5` | Every head `.out` via `_init_linear(gain=0.01)`; position and slot embedding weights via `get_input_layers` | same |
| Muon membership | `actor_input_proj` and head `.up` in Muon; head `.out`, embeddings, norms and biases in AdamW | same |
| Heads outside the compiled region | Heads eager; `compile_transformer_trunk` compiles the trunk only; heads chunk rows below the GEMM-extent limit anyway | same, plus the head-extent guard |
| Teacher: cached targets, teacher-forced `KL(teacher ‖ student)` per head at the replayed prefix, masked target logits, value CE from the teacher winner distribution; cached = combined bit-for-bit | `KaggricultureTeacherTargets` (per-slot `finfo.min`-masked logits, winner probabilities, grammar signature) under `TeacherTargets`; per-slot KL weighted by the log-prob liveness in the `[E,2,252,12]` layout; value CE per seat, averaged over live seats | same / game form: 9 grammar slots instead of launch/target/size; one winner distribution per seat, so the mean keeps one CE per state; the grammar signature guards supports that replay admission cannot compare |

Parameters at the preset: 6,252,223. That is the encoder (5,312,768, trunk 4,216,832), the critic (66,049), and the actor input projection and heads (873,406). It sits inside the owner's 6–10M budget at depth 8, so no depth change was needed. The optional critic offset head adds 66,049 (6,318,272).
