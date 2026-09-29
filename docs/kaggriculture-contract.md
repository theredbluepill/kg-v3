# Kaggriculture observation, action and environment contract

Status: **v4.1, accepted** (Codex re-review: accept with edits, all applied; rebuild plan Task 0.1). Both streams build against this file. Changing it needs both agents to agree, recorded under "Review and changes".

Scope: the tensors that cross the native ↔ Python ↔ model boundary. The design follows Isaiah's `ObsBatch` pattern (`docs/rl-api-specs.md`): named per-entity tensors, written by Rust into caller-owned buffers, with explicit masks. The reference branch's flat 8,176-float vector (`kg/reference-2026-09-29:engine_rs/src/myolie_features.rs`) is the semantic source and test oracle, not the layout.

## Principles

1. **One observation per seat.** Leading dims are `[E, 2]` (env, seat). Each row is that seat's legal view. Rival-private state (shed, seeds, actor inventories) never appears in the other seat's row, so the model encodes each seat row independently, with no cross-seat attention.
2. **Roles, not seats.** Entities are marked own or rival. No seat id or opponent identity is encoded.
3. **Categories are integer indices** (pinned enums below). The model one-hot encodes them into Isaiah's `ObservationInputStem` inputs; categories are never packed into scalar codes.
4. **One field per channel.** No summed flags.
5. **Exact values travel as exact types.** Integer state is `int64`, money is `float64` in native game units, and sentinel values (e.g. `-1`) are preserved. Float model channels are **scaled** (the scale is not a bound): signed where the quantity can be negative, never clamped. A state is never rejected just because a scaled value is large.
6. **Legal information only.** Public state, own-private state, and the public configuration or pinned public rules. No RNG state, no unrevealed shop results, no rival-private state, and no hidden `Game` state.
7. **Supported-configuration envelope.** The environment validates its configuration at construction and rejects anything outside the envelope. It never truncates entities or substitutes defaults.

## Supported configuration envelope

- `boardSize = 10`
- `1 ≤ maxMarketOrdersPerTurn ≤ 10`
- `turnsPerDay × maxMarketOrdersPerTurn ≤ 240`
- observed actors ≤ 241 per farm; a state above this is an error, never truncated
- `marketParams` must be `{}`, the raw default. Custom market parameters are rejected; a future version may encode them as public rule context.
- **Framework vs game configuration:** `extra` may contain only the Kaggle framework keys `actTimeout`, `runTimeout` and `seed` (seen in the reference fixture `shop-router-0909-parity.json`). They don't affect game rules and are ignored for game state; the environment's own seed stream is authoritative. Any other `extra` key is rejected.
- **Representability:** every value written to an exact `int64` tensor must fit in `int64` (engine configuration integers are arbitrary-width; e.g. `startingMoney = 2^63` is rejected), and every float channel must be finite. A violation fails the constructor or the step transactionally.
- `episodeSteps`, `startingMoney`, `turnsPerDay`, `shedCapacity`, `weedSpawnChance`, `townShopUnlockInterval`, `townShopSellInterval`, `townCenterSellInterval` and `farmHandCostMult` may vary within their JSON-schema ranges, subject to the representability limits; they are encoded as rule context in `global_features` / `globals_int`.

## Pinned enums (index 0 is valid for padding)

| Enum | Values in order |
|---|---|
| `TileKind` | 0 EMPTY, 1 LOCKED, 2 WEED, 3 PLANT, 4 COOP, 5 PASTURE |
| `Crop` | 0 NONE, 1 WHEAT, 2 CARROT, 3 TOMATO, 4 STRAWBERRY, 5 MELON |
| `Animal` | 0 NONE, 1 GOOSE, 2 COW, 3 SHEEP |
| `Item` (I = 12) | 0 WHEAT, 1 CARROT, 2 TOMATO, 3 STRAWBERRY, 4 MELON, 5 EGG, 6 MILK, 7 WOOL, 8 FERTILIZER, 9 GOOSE, 10 COW, 11 SHEEP |
| `Product` (P = 9) | 0 WHEAT, 1 CARROT, 2 TOMATO, 3 STRAWBERRY, 4 MELON, 5 EGG, 6 MILK, 7 WOOL, 8 FERTILIZER (engine `PRODUCTS`) |
| `ShopType` (8) | 0 BAKERY, 1 BRUNCH_SPOT, 2 FARMERS_MARKET, 3 ICE_CREAM_SHOP, 4 PET_CAFE, 5 PIZZA_SHOP, 6 SMOOTHIE_SHOP, 7 YARN_STORE (engine `SORTED_SHOPS`) |
| `TileRole` | 0 OWN_FARM, 1 RIVAL_FARM |
| `ActorRole` | 0 OWN_FARMER, 1 OWN_HAND, 2 RIVAL_FARMER, 3 RIVAL_HAND |

## Observation: `KaggricultureObsBatch` (schema version 3)

`T = 200` tiles (own farm cells 0–99, then rival cells 0–99). `A = 482` actor slots (own 0–240, then rival 0–240). `S = 8` shop slots. Float tensors are `float32` unless marked otherwise.

### Tiles (always present, T fixed)

- Categorical `int64 [E,2,T]`: `tile_kind`, `tile_crop`, `tile_animal`, `tile_cell` (0–99, row-major), `tile_role`.
- `tiles_int int64 [E,2,T,7]`, exact, 0 when not applicable, in this order:
  - `yield_units`
  - `planted_day` (−1 = none)
  - `max_lifespan_step` (−1 = no deadline assigned)
  - `fertilized_until_day` (−1 = never)
  - `placed_day` (−1 = none)
  - `consecutive_unwatered`
  - `consecutive_unfed`
- **Applicability:** `yield_units` applies to plants and animals. The plant fields (`planted_day`, `max_lifespan_step`, `fertilized_until_day`, `consecutive_unwatered`; float channels 1–8) are filled only when `tile_kind == PLANT`. The animal fields (`placed_day`, `consecutive_unfed`; float channels 9–14) are filled only when `tile_animal != NONE`. Inapplicable fields are 0 in both `tiles_int` and `tiles_float`, and every derived flag (has deadline, ever and currently fertilized) is 0. The `−1` sentinels appear only on applicable tiles.
- `tiles_float [E,2,T,Ct=15]`:

| # | Channel | Scale / sign |
|---|---|---|
| 0 | `yield_units` | / 8 |
| 1 | `watered_today` | 0/1 |
| 2 | `consecutive_unwatered` | / 8 |
| 3 | plant age = `day − planted_day` (0 if none) | / 30 |
| 4 | has deadline = `max_lifespan_step ≥ 0` | 0/1 |
| 5 | deadline delta = `max_lifespan_step − step` (0 if none) | / `episode_steps`, **signed** (overdue < 0) |
| 6 | ever fertilized = `fertilized_until_day ≥ 0` | 0/1 |
| 7 | currently fertilized = `fertilized_until_day ≥ day` | 0/1 |
| 8 | fertilizer days left = `max(0, fertilized_until_day − day)` | / 30 |
| 9 | animal age = `day − placed_day` (0 if none) | / 30 |
| 10 | `consecutive_unfed` | / 8 |
| 11 | `fed_today` | 0/1 |
| 12 | `cared_today` | 0/1 |
| 13 | `fertilizer_available` | 0/1 |
| 14 | `pending_care_bonus` | / 8 |

### Actors (A slots; absent rows are zero-filled, categories 0, mask false)

- Categorical `int64 [E,2,A]`: `actor_slot` (0–240 within its farm), `actor_cell` (0–99), `actor_role`.
- `actor_mask bool [E,2,A]` (present).
- `actor_inventory int64 [E,2,241,I]`: exact own counts, including keys present with count 0.
- `actor_inventory_rank int64 [E,2,241,I]`: the current `IndexMap` key order, 1 = first key and 0 = absent key. A present key with count 0 still has a rank. Removing and reinserting a key changes the order.
- `actors_float [E,2,A,Ca=26]`:
  - 0–1: x, y / 9
  - 2–13: own inventory counts / 32 (rival rows are 0)
  - 14–25: own inventory rank / 12 (rival rows are 0)

### Players (one feature row per player token: index 0 = self, 1 = opponent)

Isaiah adds each player's summary to that player's learned token through `player_feature_proj`, and Kaggriculture keeps that topology. The opponent column holds public facts only; own-private fields are 0 there.

- `player_features [E,2,2,Cp=44]`:

| Channels | Content | Scale |
|---|---|---|
| 0 | money | / 200,000 (signed) |
| 1 | actor count | / 241 |
| 2 | unlocked quadrant count | / 4 |
| 3–6 | quadrant bits NW, NE, SW, SE | bits |
| 7–8 | empty tiles, locked tiles | / 100 |
| 9 | `hires_today` (public) | / 240 |
| 10 | next-hire cost from public rules and config | / 200,000 (can exceed 4) |
| 11–22 | shed counts (self only) | / 100 |
| 23–27 | seed counts (self only) | / 32 (can exceed 4) |
| 28–39 | shed key rank (self only) | / 12 |
| 40–41 | shed used, shed room (self only) | / `shed_capacity` |
| 42–43 | reserved | 0 |

- Exact own storage: `storage_counts int64 [E,2,17]` (12 shed items, then 5 seeds) and `storage_rank int64 [E,2,12]`.
- Observation banks: `banks float64 [E,2,2]` (self, opponent), in native game-money units. They belong to the returned observation and match `player_features[...,0]`, so after an auto-reset they are the new game's banks. The completed transition's banks are separate outputs (see Environment).

### Shops (ordered entity tokens)

- `shop_type int64 [E,2,S]`, `shop_slot int64 [E,2,S]` (0–7), `shop_mask bool [E,2,S]`.
- Present shops: `shop_slot` is the unique zero-based index in `town.unlocked_shops`. Masked slots: `shop_type = 0`, `shop_slot = 0`, `shop_mask = false`.
- The model feeds one-hot(type) ‖ one-hot(slot) into the shop stem. Attention has no implicit position, so order is carried by `shop_slot`. Duplicate **types** are allowed (the engine samples with replacement); positions are unique.

### Market (entity tokens)

- `market_product int64 [E,2,P]` (always present).
- `market_float [E,2,P,2]`: market inventory **index** / 10,000 (signed, not bounded stock) and price / 250 (no general ceiling).
- `market_int int64 [E,2,P,2]`: exact inventory index and price. The engine stores them as JSON values; within the supported profile (`marketParams = {}`) they are integers. The writer converts with a checked integer → `int64` conversion, with no tolerance or rounding. Fractional or out-of-range values (possible only with custom market parameters or imported state) are rejected.

### Globals (one global-feature token)

- `global_features [E,2,Cg=15]`:

| Channels | Content | Scale |
|---|---|---|
| 0 | step | / `episode_steps` |
| 1 | hour (within day) | / `turns_per_day` |
| 2 | day | / (`episode_steps` / `turns_per_day`) |
| 3 | remaining transitions = `max(0, max(1, episode_steps − 1) − step)` | / `episode_steps` |
| 4 | `turns_per_day` | / 24 |
| 5 | `max_market_orders_per_turn` | / 10 |
| 6 | `shed_capacity` | / 1,000 |
| 7 | `farm_hand_cost_mult` | / 100 |
| 8 | `episode_steps` | / 1,000 |
| 9 | `weed_spawn_chance` | as is |
| 10 | `town_shop_unlock_interval` | / `turns_per_day` |
| 11 | `town_shop_sell_interval` | / `turns_per_day` |
| 12 | `town_center_sell_interval` | / `turns_per_day` |
| 13 | `starting_money` | / 200,000 |
| 14 | unlocked shop count | / 8 |

- `globals_int int64 [E,2,Gi=16]`:
  - `step`
  - `day`
  - `hour`
  - `episode_steps`
  - `turns_per_day`
  - `max_market_orders_per_turn`
  - `shed_capacity`
  - `farm_hand_cost_mult`
  - `town_shop_unlock_interval`
  - `town_shop_sell_interval`
  - `town_center_sell_interval`
  - `starting_money`
  - own `hires_today`
  - rival `hires_today`
  - own actor count
  - rival actor count

### Masks and context

- `still_playing bool [E,2]`: always true. After auto-reset the returned observation belongs to the new game; see "Terminal step timing".
- `order_limits int64 [E,2]`: market orders allowed this turn.
- `action_mask.can_act bool [E,2,F]`: `frame < own_actor_count + order_limits + 1`.

### Token order (model)

Wiring: player tokens are `player_tokens + player_feature_proj(player_features)`, the global token is `global_proj(global_features)`, and the two seat rows are encoded independently. The sequence follows Isaiah's `[action entities][other entities][players][global][board scratch][actor plan][critic value]`:

- own actors (the acting entities, first)
- rival actors
- tiles
- shops
- market products
- 2 player tokens
- 1 global token
- board scratch tokens
- 1 actor-plan token (only this seat acts in its row)
- 2 critic-value tokens (self, opponent)

## Action: `KaggricultureActions`

- `tokens int64 [E,2,F=252,K=12]`, `lengths int64 [E,2]`.
- Frame layout:
  - **one unit frame per observed own actor**, in actor order; every actor gets a command, possibly `PASS`
  - then up to `order_limits` market frames
  - then **one STOP frame**
  - `lengths` counts every frame including the final STOP; native admission requires every frame after `lengths` to be zero, then ignores those frames for transitions and rendering
- Slots and widths, in `SLOT_NAMES` order. The Rust grammar (plan C3) is the single source of truth:

| Slot | `unit_actor` | `unit_kind` | `unit_target` | `unit_item` | `unit_quantity_present` | `unit_quantity_high` | `unit_quantity` | `market_kind` | `market_item` | `market_quantity_high` | `market_quantity` | `stop` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Width | 241 | 20 | 128 | 16 | 2 | 32 | 32 | 8 | 16 | 32 | 32 | 2 |

- **Action enums** (distinct from the observation enums; pinned from the grammar source `myolie_sampler.rs`):

| Enum | Values | Always masked |
|---|---|---|
| `UnitKind` (width 20) | 0 NONE, 1 PASS, 2 NORTH, 3 SOUTH, 4 EAST, 5 WEST, 6 PICKUP, 7 PLACE, 8 PLANT, 9 WATER, 10 HARVEST, 11 DROP, 12 BUILD_COOP, 13 BUILD_PASTURE, 14 FEED, 15 FERTILIZE, 16 COLLECT_FERTILIZER, 17 CARE, 18 DIG | 19 |
| `MarketKind` (width 8) | 0 NONE, 1 HIRE, 2 BUY_LAND, 3 BUY_SEED, 4 BUY_PRODUCT, 5 BUY_ANIMAL, 6 SELL, 7 EMPTY | — |
| `ActionItem` (width 16, used by `unit_item` and `market_item`) | 0 NONE, then observation `Item` + 1: 1 WHEAT … 12 SHEEP | 13–15 |

- `unit_target` is a **reserved legacy width**: its canonical support is `{0}`, and values 1–127 are always masked. Movement uses `unit_kind`, and work acts at the actor's current location.
- Quantities decode as `32·high + low`:
  - unit transfer quantity: omitted (`present = 0`) or 1–1023; an explicit unit zero is rejected
  - quantity-bearing market orders: 0–1023, and an explicit zero survives
- `NONE` is structural in market and STOP frames. `EMPTY` market orders consume a queue position.
- The HIRE budget is **actor capacity, not cash**: `plan(actors, order_limit, hire_limit)` with the v3 default `hire_limit = 241`. Prior submitted HIRE orders count against it even if they later fail to execute.
- Grammar masks are **syntax and support masks** from typed native plans and observation-local cursors (C3), with eight directly exported boolean tables (C7). Binary DFA nodes are reference-oracle data only. Affordability and successful execution are engine outcomes, not masks.
- Native per-seat decoding accepts exactly 3,024 `int64` tokens and a length in `[actors + 1, actors + order_limit + 1]`; length 0 is rejected. All-zero, length-0 synthetic inactive rows belong only to model/trainer masking.
- Strict native `encode` requires exactly `farmer`, `hands`, `market`, every observed actor command, and canonical command syntax; it never inserts PASS or normalizes a replay. It validates using the same cursor as `decode` and leaves its output unchanged on error.
- Actor order is shared across observation, grammar and engine: frame `i` = `unit_actor i` = own observation `actor_slot i` = engine unit `i` (farmer, then hands in stored order).

## Environment

- **Constructor:** `KaggricultureEnv(n_envs, seed, seed_stride, config, reward_config, native_threads, pin_memory)`, created by `owl.game.create_env`. It validates the configuration envelope; `seed ≥ 0` (CPython discards the sign of integer seeds).
- **Seeds:** the factory passes `seed = base_seed + rank` and `seed_stride = world_size`; the native side doesn't add the rank again. Rank `r` starts with `next_seed = base_seed + r` and stride `world_size`. Every game construction and every reset consumes one seed and advances the counter. Simultaneous resets consume seeds in ascending environment index, and construction consumes seeds before any explicit `reset()`.
- **API:** `reset() -> obs`; `step(actions) -> (obs, rewards f32 [E,2], dones bool [E,2], metrics)`, with synchronous auto-reset.
- **Terminal step timing:** on a step that ends a game, rewards, dones, transition banks, economic counters and terminal metrics describe the **completed** transition, while the returned observation, `obs.banks` and masks describe the **newly reset** game. The default game ends after 719 transitions.
- **Transition banks:** `transition_banks_before` and `transition_banks_after`, both `float64 [E,2]` in seat order (seat 0, seat 1), give the banks before and after the completed transition. They are valid until the next `step`/`reset`.
- **Terminal metrics** (built lazily, valid until the next `step`/`reset`):
  - `bank_0`, `bank_1` (float64)
  - `margin_0 = bank_0 − bank_1`
  - `episode_steps` (`int64`): completed transitions, 719 by default. This is not the configured `episodeSteps = 720`, and the integer dtype is a deliberate ABI choice (the reference used `f64`)
  - extensions: `winner` (0, 1 or −1 for a draw, from raw banks) and `econ_0`/`econ_1` (the 32 cumulative economic counters, int64)
- **Rewards:** for seat `s` against rival `o`, `r_s = ΔP_o − ΔP_s + done · terminal_scale · sign(bank_s − bank_o)`.
  - `P` is the sum of the separately capped **cumulative** penalties (C8): `P = min(death_cap, W·(starvation_weight·S + drought_weight·D)) + min(ineffective_cap, ineffective_weight·I)`.
  - `terminal_scale = 1 − (death_cap if W > 0 else 0) − (ineffective_cap if ineffective_weight > 0 else 0)`. "Enabled" depends on the coefficients, not on whether an event occurred. The configuration validation requiring the active caps to sum below one is kept.
  - This requires `reward_mode = win_loss` and gamma 1.
  - The complete-episode return is bounded by 1; bootstrapped PPO targets carry no such guarantee.
- **Truncation:** `truncate_envs(mask)` keeps the transition's economic reward, bootstraps from the pre-reset observation and fabricates no terminal winner (L2). The reset keeps the transition buffers (`clear_transition = False` semantics).
- **Buffer lifetime and transactions:**
  - Before the env overwrites any published buffer generation (observations, rewards, dones, banks, counters, metrics), every reader of it must have finished. A synchronous-copy baseline satisfies this; double buffering needs a per-buffer reuse fence.
  - A failed step or reset leaves game state, seed allocation, terminal records and published buffers unchanged.
  - Task 0.2 found that the historical CUDA fault was a compiler GEMM overflow, not buffer reuse; the fence is still required for correctness.

## Information audit (reference flat vector → this contract)

| Reference range | Content | New location |
|---|---|---|
| 0–15 | step/hour/day fractions, money ×2, actor counts ×2 (capped at 16), quadrants ×2, empty/locked ×2, shed/seed totals, shop count | `global_features` 0–2 and 14; `player_features` 0–2, 7–8; totals derivable from `storage_counts` (no longer capped at 16) |
| 16–615 | legacy tile code, age, packed state ×200 | Replaced by the tile enums plus `tiles_int` / `tiles_float`; the packed sums are dropped as redundant. The legacy "fertilized" flag meant *ever* fertilized, which is now channel 6 |
| 616–679 | first-16-actor positions / 10 | Duplicate of 3827–5272; dropped |
| 680–871 | first-16 own inventories / 32 | Duplicate of 5273–8164; dropped |
| 872–888 | own seeds (5) / 32, own shed (12) / 100 | `player_features` self 11–27, `storage_counts` |
| 889–906 | market inventory index (9) / 10,000, prices (9) / 250 | `market_float`, `market_int` |
| 907–959 | always zero | Dropped |
| 960–1023 | ordered unlocked shops (8 × one-hot 8) | `shop_type`, `shop_slot`, `shop_mask` |
| 1024 | constant 1 (availability) | Dropped: constant |
| 1025–1026 | own and rival actor counts (raw) | `globals_int`, `actor_mask` |
| 1027–3826 | per-tile maintenance (is_plant, is_animal, raw fields) | `tile_kind`, `tiles_int` (exact, with sentinels), `tiles_float` |
| 3827–5272 | actors: present, x, y ×482 | `actor_mask`, `actors_float` 0–1, `actor_cell` |
| 5273–8164 | own inventories ×241×12 | `actor_inventory`, `actors_float` 2–13 |
| 8165 | constant 1 (suffix present) | Dropped: constant |
| 8166–8175 | step, day, hour, episode length, turns/day, own hires, cost multiplier, quadrant bitmask, shed capacity, orders | `globals_int`, `global_features` 4–8, `player_features` 3–6 and 9 |
| (absent from reference) | inventory insertion order; explicit shed room; remaining transitions; next-hire cost; weed chance; shop and town intervals; starting money; rival `hires_today` | Added: ranks, `player_features` 10 and 40–41, `global_features` 3 and 9–13, `globals_int` |

The oracle test (plan Task 1.3) reconstructs every retained reference value from these tensors on ≥ 500 recorded states and checks each added fact against engine state.

## Answers to the v1 open points (from Codex's review)

1. **Normalization:** the scales are not bounds. Channels can exceed 4 (seeds, next-hire cost, custom configs), and some are signed. No clamping, no range rejection.
2. **`unit_target`:** reserved, always 0.
3. **Rival public state:** money, all tile and maintenance state, farmer and hands, quadrants, `hires_today`, plus the shared market and town. Previous market-order submissions are not public.
4. **Buffers:** completion fencing before reuse (see Environment).

## Review and changes

- v1 (Claude, 2026-09-29): initial draft.
- v2 (Claude): reorganized to Isaiah's token topology (`player_features`, `global_features`, token order).
- v3 (Claude): addresses all 15 findings of Codex's v1 review (`ops/rebuild-2026-09-29/codex/task-0.1-review.md`):
  - blockers: exact fertilizer and lifespan fields with sentinels (1–2), the configuration envelope and rule context (3), the seed stream (4)
  - should-fix: float64 banks (5); scale-not-bound normalization (6); reserved `unit_target` (7); explicit action rules (8); terminal timing, remaining transitions and metrics (9); the reward formula and truncation (10); buffer fencing and transactional failure (11); audit constants and suffix (12); pinned enums, dtypes, padding and rank semantics (13); `shop_slot` (14)
  - note (15): the `docs/rl-api-specs.md` companion section follows once v3 is accepted
- v4 (Claude): applies Codex's re-review edits (`ops/rebuild-2026-09-29/codex/task-0.1-rereview.md`, verdict accept with edits):
  - the framework vs game configuration boundary, `marketParams = {}`, and representability limits
  - pinned action enums
  - shop padding and slot rules
  - exact market conversion
  - separate observation and transition banks
  - tile applicability gating
  - explicit token wiring
  - the factory seed rule
  - the reward expansion
  - `docs/rl-api-specs.md` companion section added

- v4.1 (Codex implementation clarification; Claude agreed in Task 1.2 brief review, Codex agrees): no semantic change. Open questions 1–5 in `ops/rebuild-2026-09-29/briefs/1.2.md` resolve as follows:
  - Capacity admission follows the sampler rule `actors + prior submitted HIREs < hire_limit`, default 241; the historical training decoder bypass is recorded as disagreement.
  - Padding is validated as zero before being ignored.
  - Native decoding rejects length 0; synthetic inactive rows stay in model/trainer masking.
  - Typed `GrammarPlan`/`State` plus eight direct boolean tables replace the binary plan runtime interface; table version 1 pins names, widths and shapes in `docs/rl-api-specs.md`.
  - Production grammar compiles at `src/kaggriculture/grammar.rs`; the standalone engine test includes that same source temporarily. At the first production root → engine dependency (1.3/1.4), move kernel acceptance tests into root integration, delete the engine test and its authored registration, and reopen L4 feature unification.
