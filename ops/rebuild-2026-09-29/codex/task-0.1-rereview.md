**v3 resolves most findings, but needs several edits before the contract is frozen.** I reviewed against `kg/reference-2026-09-29` at `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. No files were modified; no runtime tests were run. Reference-source citations below refer to that commit.

1. **resolved — Fertilizer information.**  
   `tiles_int` preserves `fertilized_until_day`, including `-1`; the legacy flag is correctly defined as **ever fertilized**. This fixes the information loss.  
   Evidence: `engine_rs/src/myolie_features.rs:76–83,179–181,193–215`; `engine_rs/src/lib.rs:2983,3363–3365`. A separate applicability ambiguity is noted below.

2. **resolved — Lifespan sentinel and signed delta.**  
   The exact deadline, explicit presence flag and signed delta distinguish an unassigned deadline from an overdue crop.  
   Evidence: `engine_rs/src/lib.rs:2967–2984,4178–4198,4255–4260`.

3. **not resolved — Supported configuration envelope.**  
   The board/order/actor limits and added rule context are correct. Rejecting custom market parameters is a valid, explicitly narrower supported profile; the raw default is **`marketParams: {}`**.

   However, requiring `extra` to be empty is incompatible with an unprojected Kaggle default configuration. The reference fixture `engine_rs/fixtures/shop-router-0909-parity.json`, at `template.configuration`, contains `actTimeout: 1`, `runTimeout: 1200`, and `seed: null`. These enter flattened `Config.extra`; engine validation permits them. See `lib.rs:700–705,722–763,2453–2472`.

   Also, “within JSON-schema ranges” admits integers beyond the new `int64` ABI. For example, `startingMoney = 2^63` satisfies the schema and engine validation but cannot fit `globals_int`. Configuration integers use `BigInt` (`lib.rs:443–451,2476–2485`).

   **Edit:** Define the boundary between framework configuration and game configuration, allowing or explicitly projecting known framework fields. Specify `{}` as the accepted raw market override profile, and add explicit representability limits for exact tensors and finite model-channel conversions.

4. **resolved — Seed allocation.**  
   One advancing stream per rank, ascending environment-index allocation, construction consumption and nonnegative seeds match the reference.  
   Evidence: `engine_rs/src/training.rs:227–235,439–441,477–490`; `engine_rs/src/py_random.rs:8–11`. The factory should pass `base_seed + rank`; native construction must not add rank again.

5. **resolved — Exact banks and hire multiplier.**  
   `banks` now uses native-unit `float64`, and `globals_int` explicitly includes `farm_hand_cost_mult` in its enumerated 16 channels.  
   Evidence: `engine_rs/src/lib.rs:772,3820–3839`; `engine_rs/src/training.rs:124–125`. Bank-generation naming still needs the edit below.

6. **resolved — Scales versus bounds.**  
   The contract removes the false `[0,4]` restriction, preserves signed market inventory indices and allows large scaled values. This addresses the counterexamples from v1. Numeric representability is a separate envelope issue under finding 3.

7. **resolved — `unit_target`.**  
   Canonical support `{0}`, with indices 1–127 always masked, matches the grammar and decoder.  
   Evidence: `engine_rs/src/myolie_sampler.rs:161,349–380`.

8. **not resolved — Action semantics and categorical ABI.**  
   Quantities, PASS/NONE/EMPTY, STOP-inclusive lengths and HIRE-capacity accounting are corrected. The requested **action enum mappings and reserved indices remain missing**.

   Observation `Item` has WHEAT at 0; action `ITEMS` has NONE at 0 and WHEAT at 1. They must not share an enum accidentally.  
   Evidence: `engine_rs/src/myolie_sampler.rs:19–64,153–190`.

   **Edit:** Pin `UnitKind` to `UNIT_NAMES` indices 0–18, with 19 always masked; pin `MarketKind` to `MARKET_NAMES` indices 0–7; define `ActionItem` as NONE=0 followed by observation items at 1–12, with 13–15 always masked.

9. **resolved — Terminal timing and remaining transitions.**  
   Reset observations, completed-transition rewards/dones, `still_playing=true`, and the remaining-transition formula match the native lifecycle. Default termination is after **719 transitions**.  
   Evidence: `engine_rs/src/training.rs:145–165,403–449,517–529`; `engine_rs/src/lib.rs:1487–1496`.

   The terminal metric `episode_steps` means actual completed transitions—719 by default—not configured `episodeSteps=720`. Its integer dtype is a deliberate new ABI choice; the reference metric map uses `f64`.

10. **resolved — Reward and truncation semantics.**  
    The reward sign, cumulative separately capped penalties, enabled-cap scaling, gamma 1 and truncation treatment are correct. The source expansion is:
    ```
    P = min(death_cap, W × (starvation_weight × S + drought_weight × D))
        + min(ineffective_cap, ineffective_weight × I)

    terminal_scale = 1
        − (death_cap if W > 0 else 0)
        − (ineffective_cap if ineffective_weight > 0 else 0)
    ```
    “Enabled” depends on coefficients, not whether an event occurred. Preserve the source validation requiring active caps to sum below one.  
    Evidence: `python/owl/kaggriculture/rewards.py:32–43,74–92`; `engine_rs/src/training.rs:72–110,392–420,493–503`.

11. **resolved — Buffer lifetime and transactions.**  
    The contract now requires reader completion before every published generation is overwritten and rollback of state, seeds, terminal records and outputs on failure. This correctly separates publication safety from internal staging.  
    Evidence: `engine_rs/src/training.rs:296–307,387–503`; `src/kaggriculture.rs:549–607`.

12. **resolved — Information-audit constants and suffix.**  
    Both availability constants are explicitly discarded as constants, and all ten investment-suffix fields are identified. The audit no longer calls already-present clock/configuration facts absent.  
    Evidence: `engine_rs/src/myolie_features.rs:26–54,347–389`.

13. **not resolved — Observation ABI completeness.**  
    The pinned observation enum orders are correct: Products/Crops/Animals match `lib.rs:32–44`, Shops match `lib.rs:54–63`, and Items match `myolie_features.rs:62–75`. Actor padding, exact ordering and zero-count insertion ranks are also corrected.

    **Remaining edit:** Extend the padding rule to shops. Specify masked `shop_type=0`, `shop_slot=0`, `shop_mask=false`; for present shops, `shop_slot` is the unique zero-based index in `town.unlocked_shops`. Clarify that duplicate **types**, not duplicate positions, are allowed. Currently the explicit zero-fill rule appears only under Actors.

14. **resolved — Shop position through attention.**  
    One-hot `shop_slot` supplied alongside one-hot type preserves the ordered shop representation through a shared stem. Repeated shop types are legitimate: the engine samples with replacement and appends each instance.  
    Evidence: `engine_rs/src/myolie_features.rs:380–385,462–482`; `engine_rs/src/lib.rs:4517–4528`.

15. **not resolved — Task 0.1 documentation completion.**  
    The topology remains compatible, but the companion Kaggriculture section in `docs/rl-api-specs.md` is still absent. Deferring it does not complete Task 0.1.  
    **Edit:** Add that section and cross-reference I0b’s separate role parameters, shared trunk/final normalization, critic projection and `[entity, player, plan]` actor projection.  
    Evidence: [plan Task 0.1](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/plan.md:147), [topology requirements](/Users/poonszesen/kg-v3/ops/rebuild-2026-09-29/plan.md:206).

The new findings are:

- **Should-fix — “Market”: `market_int` is not fully defined.**  
  At [contract line 118](/Users/poonszesen/kg-v3/docs/kaggriculture-contract.md:118), conversion tolerance is deferred to Task 1.3. The engine stores inventory and prices as **`IndexMap<String, serde_json::Value>`**, not a fixed integer or floating type. Default initialization uses integers; inventory arithmetic preserves integers, including arbitrary-width ones; refreshed prices are rounded to integers. Custom initial `I0`/`base` values can be fractional.  
  Evidence: `engine_rs/src/lib.rs:578–647,788–794,3730–3768`.  
  **Concrete edit:** For the default-only profile, require exact checked integer-to-`int64` conversion, with no tolerance or rounding. Explicitly reject fractional or out-of-range imported state. Remove the deferred tolerance decision and connect these checks to the supported envelope.

- **Should-fix — “Players” / “Terminal step timing”: observation and transition banks conflict.**  
  [Players](/Users/poonszesen/kg-v3/docs/kaggriculture-contract.md:107) declares `banks` inside the observation, while [Environment](/Users/poonszesen/kg-v3/docs/kaggriculture-contract.md:207) says banks describe the completed transition even after observation reset. The reference uses separate observation features and `previous_banks`/`current_banks`.  
  Evidence: `engine_rs/src/training.rs:117–127,405–410,439–449`.  
  **Concrete edit:** State that `obs.banks` matches the returned observation and `player_features[...,0]`, including the new game after reset. Separately name transition-bank buffers and specify their shapes, seat ordering and completed-transition lifetime.

- **Should-fix — “Tiles”: inapplicable fields and derived flags are ambiguous.**  
  [Tiles](/Users/poonszesen/kg-v3/docs/kaggriculture-contract.md:46) says inapplicable integer fields are zero, but the derived tests use `deadline >= 0` and `fertilized_until_day >= 0`. Applied literally, an empty tile acquires a deadline and an “ever fertilized” flag; at day zero it is also “currently fertilized.”  
  Evidence: the reference gates maintenance by plant/animal presence in `myolie_features.rs:193–215`.  
  **Concrete edit:** Explicitly zero plant-derived channels for non-plants and animal-derived channels when no animal is present. Distinguish inapplicable zero-fill from meaningful `-1` sentinels.

- **Note — “Players”, “Globals” and token order: no new topology defect found.**  
  The reorganization matches Isaiah’s projected player summaries, global stem and role-token sequence. Independent seat encoding and zeroed opponent-private fields preserve privacy. One actor-plan token per private seat row is explicitly planned.  
  Evidence: current [Isaiah model](/Users/poonszesen/kg-v3/python/owl/model/stateless_transformer_v1.py:310), lines 517–518, 557–567 and 631–640; plan line 206.  
  **Concrete edit:** Pin the wiring explicitly: `player_tokens + player_feature_proj(player_features)` and `global_proj(global_features)`, with the two seat views encoded independently.

**ACCEPT WITH EDITS:** complete the configuration/representability envelope; pin action enums; define shop padding; finish the companion API/topology documentation; specify exact market conversion; separate observation and transition banks; and gate inapplicable tile channels.