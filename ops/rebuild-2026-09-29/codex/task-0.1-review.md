**Draft v1 needs revision before becoming the shared implementation contract.** I reviewed reference commit `65f0eac5bb00b18a9d3acce319c2a231cbd5dff0`. No files were modified and no runtime tests were run. Reference citations below refer to that commit unless marked “current”.

1. **Blocker — “Tiles”, channels 5–6; “Information audit”, rows 16–615 and 1027–3826.**  
   The fertilizer representation loses information. At day 4, `fertilized_until_day=-1` and `2` both become `false, 0`. The reference retains the raw date, and its legacy flag means **ever fertilized** (`>=0`), not currently fertilized. Consequently, the promised reconstruction cannot pass.

   **Evidence:** `engine_rs/src/myolie_features.rs:76–83,179–181,193–215`; `engine_rs/src/lib.rs:2983,3363–3365`.  
   **Proposed edit:** Retain exact `fertilized_until_day`, including sentinel `-1`, in a typed tile tensor. Keep current-fertilization and days-left channels as derived conveniences.

2. **Blocker — “Tiles”, channel 4 and the `[0,4]` assertion.**  
   `max_lifespan_step=-1` means no deadline has yet been assigned to an ongoing crop. Subtracting `step` produces a negative value. Finite-deadline crops also remain alive after their deadline while yield decays, so negative deltas are meaningful.

   **Evidence:** `engine_rs/src/lib.rs:2967–2984,4178–4198,4255–4260`.  
   **Proposed edit:** Preserve exact `max_lifespan_step`, distinguish its sentinel from finite deadlines, and define a **signed** normalized deadline delta. Remove this channel from any nonnegative-range assertion; do not clamp overdue values.

3. **Blocker — “Principles” item 6; “Market”; “Globals”; “Environment” constructor.**  
   “The entire legal state is available” is not yet true. Missing variable rule context includes weed probability, shop-unlock interval, shop-consumption interval, town-center consumption interval, and public custom market parameters. Meanwhile, the fixed tensor sizes lack an explicit supported-configuration envelope.

   **Evidence:** `engine_rs/src/lib.rs:676–705,728–763,788–809`; `engine_rs/src/myolie_features.rs:265–297`; `docs/coordination-system-design.md:87–94`.  
   **Proposed edit:** Add a validated configuration argument or declare a source-bound fixed profile. Encode relevant publicly visible overrides, including effective public `market.params`, or explicitly reject unsupported overrides. Require `board_size=10`, orders `1..10`, `turns_per_day × orders ≤240`, and observed actors `≤241`. Never silently truncate entities or substitute defaults. Only legal configuration or pinned public rules may supply these fields—not arbitrary hidden `Game` state.

4. **Blocker — “Environment”, Seeds.**  
   `seed + r + k·world_size`, with `k` defined per environment, gives every environment in a rank the same seed at the same reset number. The reference uses one allocation stream per batch/rank.

   **Evidence:** `engine_rs/src/training.rs:227–235,439–441,477–490`.  
   **Proposed edit:** “Rank `r` starts with `next_seed=base_seed+r`, stride `world_size`. Every game construction/reset consumes one seed and advances the counter. Simultaneous resets consume seeds in ascending environment index.” Document that construction consumes seeds before a subsequent explicit `reset()`. Use nonnegative seeds if claiming distinct RNG streams: integer seed signs are discarded (`engine_rs/src/py_random.rs:8–11`).

5. **Should-fix — “Globals”, `globals_int`; “Principles” item 5.**  
   Money is not natively an integer-cents field: `Farm.money` is `f64`. An invented cents conversion needs an explicit supported domain and cannot represent arbitrary imported fractional balances exactly. Also, `farm_hand_cost_mult` is omitted from the exact integer fields despite driving hire accounting.

   **Evidence:** `engine_rs/src/lib.rs:772,3820–3839,3881–3905`; `engine_rs/src/training.rs:124–125`; `engine_rs/src/myolie_features.rs:49–53`.  
   **Proposed edit:** Preserve raw banks as `float64` in native game-money units, with `/200_000` float32 model channels. Add exact `farm_hand_cost_mult` to the integer schema. Recalculate and enumerate `Gi`.

6. **Should-fix — All normalization tables; “Open points”, item 1.**  
   The divisors are scales, not general bounds. Beyond finding 2, several channels can exceed `[0,4]`:

   - **Seeds `/32`:** buying 129 WHEAT seeds costs 1,290 under defaults and produces `4.03125`.
   - **Actor inventories `/32`:** there is no 128-unit carrying cap.
   - **Animal age `/30`:** longer supported episodes permit ages above 120 days.
   - **Next-hire cost `/200_000`:** at `hires_today=29`, multiplier 1, cost is 832,040 → `4.1602`.
   - **Custom configurations:** shed counts/capacity, money, turns/day and hire multiplier can exceed their implied ceilings.
   - **Market inventory:** this is a signed market index, not bounded available stock; prices have no general 1,000 ceiling.

   **Evidence:** `engine_rs/src/lib.rs:77–78,713,730–763,2949–2951,3190–3214,3792–3797,3820–3821,3915–3921,4268–4319,4549–4577`; `docs/coordination-system-design.md:109`.  
   **Proposed edit:** Add a per-channel signedness/range table tied to the admitted configuration. Distinguish source-derived bounds from observed fixture ranges. Rename “market inventory” to “market inventory index” and preserve negative values. Do not reject otherwise supported states merely because a scaled quantity exceeds four.

7. **Should-fix — “Action”, slot table; “Open points”, item 2.**  
   All twelve widths match, but **`unit_target` indexes nothing**. Only token zero is legal; decoding ignores the slot.

   **Evidence:** `engine_rs/src/myolie_sampler.rs:12–15,161,349–380`; `python/owl/kaggriculture/actor_codec.py:263–264`.  
   **Proposed edit:** “Reserved legacy width 128; canonical support `{0}`. Values 1–127 are always masked. Movement uses `unit_kind`; work acts at the actor’s current location.”

8. **Should-fix — “Action”, semantics and grammar API.**  
   The wording leaves important support boundaries ambiguous:

   - Transfer quantity is omitted or **1–1023**; explicit unit zero is rejected.
   - Quantity-bearing market orders allow **0–1023**.
   - Quantity decoding is `32·high + low`.
   - Every observed actor gets a command, including `PASS`; `NONE` is structural in market/STOP frames.
   - `lengths` includes the distinct final STOP.
   - HIRE’s budget is actor capacity, **not cash**: prior submitted HIRE requests count even if execution fails.

   **Evidence:** `engine_rs/src/myolie_sampler.rs:91–99,153–193,250–257,327–380`; `python/owl/kaggriculture/actor_codec.py:12–16,275–290`; `python/owl/kaggriculture/types.py:33–36`.  
   **Proposed edit:** State these rules explicitly, pin enum values and masked reserved indices, and define `plan(actors, order_limit, hire_limit)` with v3 default `hire_limit=241`. Call these syntax/support masks; affordability and successful execution remain engine outcomes.

9. **Should-fix — “Masks and context”, `still_playing`; “Environment”, auto-reset and terminal metrics; “Globals”, remaining transitions.**  
   A terminal step returns data belonging to two episodes: observation/masks describe the newly reset game, while rewards, dones, banks and economic counters describe the completed transition. Returned training `still_playing` is therefore true even when `dones` is true. Default termination occurs after **719 transitions**.

   **Evidence:** `engine_rs/src/training.rs:145–165,403–449,517–529`; `engine_rs/src/lib.rs:1487–1496`.  
   **Proposed edit:** State this timing explicitly. For the native lifecycle, define remaining transitions as `max(0, max(1, episode_steps−1)−step)`. Pin terminal metric names, types and lifetime. The reference compact metrics contain `bank_0`, `bank_1`, `margin_0`, `episode_steps`; winner and economic-counter fields need explicit definitions as extensions.

10. **Should-fix — “Environment”, rewards and truncation.**  
    “Antisymmetric” and “1−caps” underspecify the reward. Caps apply to **cumulative episode penalties**, and only enabled components reduce terminal scale. The complete-return bound does not establish a bound on arbitrary bootstrapped PPO targets.

    **Evidence:** `python/owl/kaggriculture/rewards.py:32–43,74–92`; `engine_rs/src/training.rs:72–110,392–420`; current plan `ops/rebuild-2026-09-29/plan.md:69,213`.  
    **Proposed edit:** Specify:
    `r_self = ΔP_rival − ΔP_self + done·terminal_scale·sign(bank_self−bank_rival)`,
    where `P` is the sum of separately capped cumulative death and ineffective-command penalties. Require the planned `win_loss`, gamma-1 setup. Define `truncate_envs(mask)`: retain the transition reward and bootstrap from the pre-reset observation; do not fabricate a terminal winner. Reference reset preserves transition buffers when `clear_transition=False` (`training.rs:493–503`).

11. **Should-fix — “Environment”, caller-owned buffers.**  
    The contract needs both publication lifetime and transactional failure guarantees. Internal Rust staging does not protect Python’s reused pinned arrays from an outstanding asynchronous copy.

    **Evidence:** `engine_rs/src/training.rs:296–307,387–458,460–470`; `python/owl/kaggriculture/env.py:3–5,90–117`; `src/kaggriculture.rs:597–607`; current plan C5/L6.  
    **Proposed edit:** Require completion of all readers before overwriting any published buffer generation. Apply this to observations, rewards, dones, banks and counters. Double buffering still needs a reuse fence. Also state that failed step/reset leaves game state, seed allocation, terminal records and published buffers unchanged.

12. **Should-fix — “Information audit”, availability flags and investment suffix.**  
    **All interval endpoints are correct**, and dropping the duplicate first-16 actor blocks and zero range is justified. However, availability constants lack explicit dispositions, and some “absent” facts already exist in the investment suffix.

    **Evidence:** `engine_rs/src/myolie_features.rs:26–54,347–389`.  
    **Proposed edit:** Record `1024→constant 1` and `8165→constant 1`. Enumerate `8166–8175` as step, day, hour, episode length, turns/day, hires, multiplier, quadrant bitmask, shed capacity and orders. Describe clock/configuration as partly present in the suffix; genuinely added facts include insertion order, explicit room/remaining transitions/next-hire cost and the missing configuration context. Fix findings 1–2 before claiming reconstruction of tile blocks.

13. **Should-fix — “Observation”, categorical ABI, padding and insertion ranks.**  
    Numerical enum mappings, float dtypes, exact global ordering and masked-entry values are incomplete. Insertion ranks must distinguish absent keys from present keys whose count is zero.

    **Evidence:** `engine_rs/src/lib.rs:32–44,54–63,1275–1285,2953–2963,3206–3213`; `engine_rs/src/myolie_features.rs:62–75`; current `docs/rl-api-specs.md:60–76`.  
    **Proposed edit:** Add a versioned schema table with enum values, dtypes and channel ordering. Zero-fill absent rows with valid categorical index zero and false masks—never an invalid one-hot index. Define rank as current `IndexMap` key order, including zero-valued keys; removal/reinsertion changes actor-inventory order. Explicitly mark fixed groups as always present.

14. **Should-fix — “Shops”; information-audit row 960–1023.**  
    Array order alone does not preserve shop position through a shared per-shop stem and Isaiah’s attention, which has no implicit positional encoding.

    **Evidence:** current `python/owl/model/stateless_transformer_v1.py:2750–2769,3053–3106`; reference `engine_rs/src/myolie_features.rs:380–385,462–482`.  
    **Proposed edit:** Add `shop_slot` values 0–7, one-hot encoded into the shop stem, or explicitly use one ordered flattened shop block. This preserves the declared information; it does not claim shop order changes current game outcomes.

15. **Note — “Principles”, topology; Task 0.1 completion.**  
    Named tensors and integer categories converted to one-hot stem channels are compatible with I0b. No intrinsic topology blocker was found. Exact accounting tensors need not introduce new neural layers. The companion Kaggriculture section in `docs/rl-api-specs.md` is still absent.

    **Evidence:** current `python/owl/model/stateless_transformer_v1.py:2750–2769`; current plan `ops/rebuild-2026-09-29/plan.md:89,147,206–219`.  
    **Proposed edit:** Cross-reference I0b’s separate role tokens, shared transformer classes and `[entity, player, plan]` projection; retain independent seat encoding. Add the companion API section once this contract is corrected.

For the four **Open points for review**:

1. **Normalization:** Not confirmed as bounded by `[0,4]`. Findings 2 and 6 give counterexamples. Ordinary surviving drought/starvation counters, yield, fertilizer duration and care bonus fit the proposed scales; recorded-state verification remains unperformed.
2. **`unit_target`:** Reserved, always zero; it indexes no cells, moves or items.
3. **Rival public state:** More than positions and quadrant purchases is public: money, all tile/maintenance state, farmer/hands, quadrants and `hires_today`, plus shared market/town state. Rival hires/quadrants may be derived only under documented invariants. Previous market-order submissions are not in `PublicState`. No listed tensor directly requires rival-private data, RNG or unrevealed shop results.
4. **Buffer safety:** Require completion fencing before buffer reuse now. A synchronous-copy baseline is valid; double buffering requires per-buffer reuse protection. Task 0.2 must still determine whether buffer reuse caused the historical CUDA fault.