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
| Flash packing and compile hook | `pack_sequence` / `unpack_sequence`, `torch.compile(..., dynamic=True)` | same, plus the compiled-GEMM overflow guard |
| Named encoded fields | `KaggricultureEncoded` offsets from named counts | same |
| No hidden state | Every API rejects `hidden_state`; tested | same |
| Initialization | `_init_module`, input gain 1, residual `1/sqrt(2·depth)`, token std `D^-0.5` | same |
| Ladder | Width 256, 8 heads (head_dim 32), GELU, `mlp_ratio` 2.0; depth 8 for the owner's 6–10M budget | same ladder, deeper point |
| Critic `OutputProjectionMLP` per critic-value token, winner softmax, value `2p − 1` | `critic_head` over (self, opponent) tokens from the seat's own view | same (game form: 2 players, both active) |
| Actor `3D → D` input projection over [entity, player, plan]; action heads | — | → Task 2.3 (grammar heads are game-specific) |

Encoder parameters: 5,312,768 (trunk 4,216,832). The full-model count, including heads, is checked against 6–10M in Task 2.3.
