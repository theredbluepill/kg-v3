# Evidence: how money is earned in Kaggriculture, and what separates 60k from 150k

Angle: game economics. Tree `/Users/poonszesen/kg-v3-int` at `3e89425`. The engine is `engine_rs/src/lib.rs`, cited as `lib.rs:N`. Each claim carries a status tag: **[S]** supported, **[R]** refuted, **[O]** open.

## 0. Local measurements used here (CPU, niced, 360 s total)

- Runner: the scratchpad scripts `bc_games.py` (the pod4 `bc_vs_cha22_eval.py` loop, extended with per-day `env.state_snapshot` and `env.terminal_metrics` econ counters) and `an.py`. Both sit under `/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-.../scratchpad/`. The scratchpad is session-scoped, so the scripts are not preserved in the repo.
- Config: `configs/kaggriculture_4rank_vs_cha22.yaml` (float32, CPU), evaluation seeds `_evaluation_seed(env_steps=0)`. All three sets play the same world seeds. Actions are sampled, as in the trainer's evaluation.
- Checkpoints: BC best `bc-best/checkpoint_bc_best.pt` (sha256 `fd8545872aca59c7…`) and M at 20M `M-margin-J2-4rank-20260930/checkpoints/20260930-033319/checkpoint_00_020_004_864.pt` (sha256 `f834f80e82b6fc2a…`).
- Sets:
  - **BC vs cha22:** 4 games, BC in seats 1, 0, 1, 0.
  - **BC self-play:** 4 games, 8 seats.
  - **M20 self-play:** 4 games, 8 seats.
- Limits: the samples are tiny (n = 4 to 8 seats), so read them as direction and magnitude only. Seat means are not independent in self-play.
- cha22 oracle traces: `opponents_rs/fixtures/oracle-cha22/*.jsonl.gz`, 3 games against `starter`, terminal banks 174,272 / 179,279 / 164,412.

## 1. How money enters the bank

1. **[S] The bank is the only score.** Terminal reward is `farm.money` (`lib.rs:1489`). Unsold goods, seeds, animals and land are worth 0 at the end (`ECON_UNSOLD_END` comment, `lib.rs:1490`). The one cash inflow is `SELL` of shed stock (`commit_unit`, `lib.rs:3879`). Everything else is an outflow.
2. **[S] The chain is: buy input → unit acts on a tile → harvest into the unit's inventory → DROP at a shed-adjacent tile (or the automatic end-of-day drop, capped by the 100-unit shed, `lib.rs:4424`) → SELL from the shed.**
   - Market orders resolve **after** unit actions in the same step (`lib.rs:1457-1464`). A seed bought at step t can be planted at t+1 at the earliest.
3. **[S] Inputs and their prices.**
   - Seeds are bought at fixed prices: wheat 10, carrot 20, tomato 50, strawberry 100, melon 80 (`lib.rs:68-112`).
   - Animals are bought at fixed prices: goose 300, cow 400, sheep 500 (`lib.rs:124-158`).
   - COOP and PASTURE are built free on an empty tile (unit verbs `BUILD_*`).
   - Land costs NE 1,000, SW 2,000, SE 4,000 (`lib.rs:38-39`). The start is 1 quadrant (25 tiles) and 3,000 cash (config default, `lib.rs:704-714`).
   - `BUY_PRODUCT` accepts only WHEAT (animal feed) and FERTILIZER.
4. **[S] Labor is nearly free and daily.**
   - A HIRE costs `mult·fib(hires_today)`, that is 1, 1, 2, 3, 5, 8, … (`lib.rs:3814`).
   - All hands are cleared at the end of every day (`lib.rs:4505`).
   - 10 hands a day cost 143, 13 cost 609, 15 cost about 1.6k, 20 cost about 17.7k. The marginal hand passes roughly 200 to 1,000 cash at hands 13 to 16. That is the natural labor ceiling.
5. **[S] Upkeep kills.**
   - A plant starts with `consecutive_unwatered = 1` (`lib.rs:2973`) and becomes a weed at 2 (`lib.rs:4218`). A plant not watered on its planting day dies that night, and any plant dies after 1 dry day.
   - An animal dies after 2 consecutive unfed days (`lib.rs:4286`). Each feed consumes 1 WHEAT carried by the unit.
6. **[S] Yields.**
   - Non-perennial crops (wheat, carrot, melon) gain +1 unit per watering day inside the window `[⌈max_day/2⌉, max_day]`, or +2 when fertilized (`lib.rs:3287`). They start at 1 and are capped at `max_yield`, and each decays 1 unit per 2 steps past their deadline.
   - Tomato and strawberry are perennials: 4 productions of 1 (2 if fertilized and watered), at intervals of 1 and 2 days, after days 8 and 10.
   - An animal produces 1 unit every `interval` days, plus a care bonus. Each fed-and-CAREd day adds +1 to the next production (`lib.rs:4303`). Every animal also produces **1 FERTILIZER per day** (`lib.rs:4309`).
7. **[S] Per-tile economics at base prices (derived from the tables above; cycle = replant cadence):**

| Asset | Cost | Output / cycle | Gross / tile-day (unfert → fert) | Upkeep actions | Market cap (below) |
|---|---|---|---|---|---|
| Wheat | 10 | 4 units (6 fert) / 4 d | ~22 → 35 | water daily | none (log glut) |
| Carrot | 20 | 3 (4 fert) / 3 d | ~28 → 40 | water daily | capped |
| Tomato | 50 | 4 (8) / ~12 d | ~16 → 36 | water daily | capped |
| Strawberry | 100 | 4 (8) / ~16 d | ~24 → 54 | water daily | hard cap (linear glut) |
| Melon | 80 | 6 / 11 d | **~129** | water daily | ~112 units to half price; drained only 1 a day |
| Cow | 400 | milk 160 / 2 d, up to 3 with care, plus 1 fert/day | ~150–250 plus fert | feed + care + collect + harvest | milk capped |
| Sheep | 500 | wool 200 / 3 d, up to 4 with care | ~150–270 | same | wool: only YARN_STORE drains it |
| Goose | 300 | egg 50 / d, 2 with care | ~50–100 plus fert | same | none (log glut) |

   Payback, counted from purchase to the first sale that covers the cost: wheat and carrot about 3–4 days, melon 10–11, strawberry 10–16, cow about 9–10, sheep about 7–9, goose about 5–6. **[S]** (arithmetic on `lib.rs:68-158`.)

## 2. Market dynamics and player interaction

1. **[S] Price formula.** Price = `base ± amp·shape(|inv − 10,000|)` (`lib.rs:180-202`, `3488`, `3723-3762`).
   - Selling raises the market inventory and lowers the price, unit by unit: each committed unit is re-quoted (`lib.rs:4035-4100`), so one big order walks down the curve.
   - A sale at price 1 does not add inventory (`lib.rs:3880`).
   - Verified against the trace: step 1 of `oracle-cha22-00` gives WHEAT at inventory 9,994 = 27, STRAWBERRY at 9,999 = 128 and MELON = 256, matching `25+√6`, `120+8.4·√1` and `250+8.76·ln2`.
2. **[S] Demand is the town, and it is shared.**
   - Every 4 steps each unlocked shop removes 1 unit of each of its products, or 2 if the shop sells a single product. Every 24 steps the town center removes 1 of each product except FERTILIZER (`lib.rs:4527-4570`).
   - One shop unlocks every 3 days, up to 8 shops. The type is random with replacement, so future shops are hidden RNG.
   - **No shop buys MELON.** WOOL is bought only by YARN_STORE. **Nothing buys FERTILIZER.**
   - Both players sell into one inventory. Each unit one seat sells lowers the next quote for both seats.
3. **[S] Glut severity, as units sellable above the anchor before the price halves** (computed from the defaults):

   | Product | Units to half price | Revenue |
   |---|---|---|
   | Strawberry | 32 | 2.9k |
   | Milk | 39 | 4.7k |
   | Wool | 42 | 7.0k |
   | Melon | 112 | 23.4k |
   | Tomato | 139 | 5.6k |
   | Carrot | 230 | 5.4k |
   | Fertilizer | 250 | 18.8k |
   | Wheat, egg | never, within 100k units (log glut) | — |

   Wheat and eggs are therefore **the only demand-uncapped goods**.
4. **[S] The town's total drain value.** Over the whole game, the drain at base prices is about **195–211k for both players combined**, for the 3 oracle shop schedules. That is about 11.6k a day once all 8 shops are open (script in this session; formula from `lib.rs:4527-4570`).
   - The capped goods therefore give a combined two-player "near-base-price" pie of roughly 200k drain plus about 70k of glut buffers, plus the scarcity premium when inventory sits below 10k.
5. **[S] The market couples the seats, and the effect is large.** BC's own sale revenue (econ `SELL_CASH`) is 92.8k per seat in self-play but **60.5k against cha22**. Its cash per unit sold falls from 108 to 77, because cha22 floods the shared market.
   - Consequence **[S]**: under a relative (margin) reward, depressing the rival's prices is rewarded as much as earning. Dumping milk, wool, strawberry or fertilizer to the floor is a zero-sum weapon.
   - cha22 does this: 30 floor-price sales per game, a 33.5k sale shortfall against base, and a WOOL price of 1 at the end of 2 of 3 oracle games.
6. **[O] The self-play ceiling may be well below cha22's 150k.** cha22's 143–179k comes against opponents that sell little (starter) or less (BC).
   - With two competent sellers, the capped pie of about 270k is split. Earning more in self-play needs the uncapped goods (wheat, eggs), the scarcity premium and fertilizer.
   - cha22 vs cha22 has not been measured; that game is the discriminating one.

## 3. BC vs cha22, concretely

Per-seat means. The BC vs cha22 set has n = 4 seats per side; self-play has n = 8. Values are econ counters from `terminal_metrics`.

| Metric | cha22 (vs BC) | BC (vs cha22) | BC self-play | M20 self-play |
|---|---|---|---|---|
| Final bank | **129.4k** | 49.7k | 75.6k | **30.6k** |
| Sale revenue | 160.8k | 60.5k | 92.8k | 37.2k |
| Units sold | 1,590 | 790 | 859 | 324 |
| Harvest / water / feed actions | 494 / 1,095 / 363 | 252 / 707 / 124 | 248 / 661 / 185 | 115 / 300 / 99 |
| Ineffective unit commands | **65** | 844 | 565 | **996** |
| Drought / starvation deaths | 1 / 0 | 8 / 4 | 14 / 13 | 11 / 2 |
| Unsold at end (units) | 0 | 18 | 16 | 29 |
| Quadrants at end | 3.5 | 2.0 | 2.4 (1–3) | **1.1** (7 of 8 seats never bought land) |
| Tile-days: strawberry / melon / wheat | **544** / 120 / 446 | 147 / 74 / 518 | 200 / 105 / 420 | **5** / 77 / 229 |
| Animal-days (cow + sheep + goose) | **422** | 148 | 247 | 131 (sheep 23) |
| Empty tile-days | 201 | 276 | 356 | 282 |
| Hands at hour 23 (mean over days) | ~8.9 | ~6.6 | ~7.2 | ~8.5 |

What cha22 does, from the oracle traces (`cha22.py`, 3 games) and the bot set above:

1. **[S] Front-loaded investment.**
   - Day 0–1: 12 melons, 8 wheat, 2 cows, 2 sheep. The bank goes from 3,000 to 15 by the end of day 1.
   - Day ~6: buys NE. Day ~12: buys SW, sometimes SE.
   - 33 strawberry tiles by day 13, and 6–17 cows and sheep plus geese.
   - The bank stays under about 3.5k until day 10, then rises about 5–9k a day: 24k on day 13, 63k on day 19, 120k on day 25.
2. **[S] Zero-loss upkeep.** About 1 drought death and 0 starvation deaths per game. It does 400 CARE, 370 COLLECT_FERTILIZER and 95–123 FERTILIZE actions per game (trace verb counts).
3. **[S] Fertilizer as a product.** cha22 sold about 265–273 FERTILIZER per oracle game for about 19k, roughly 12–17% of its sales. This is estimated from shed deltas, so read it as approximate; the per-item attribution is **[O]**.
4. **[S] Volume selling and an end-of-game clear-out.** It sells about 1,600 units per game, leaves 0 units unsold and never lets animals overflow their holding caps.
5. **[S] Labor is used, not maximized.** cha22 hires about 9–10 hands a day (263–284 HIRE per game), close to BC's count. The gap is **effectiveness**: 65 ineffective commands against BC's 565–844, and harvests are about 2x.

What separates 60k from 150k **[S for this sample]**:

- (a) Capital goes early into land plus perennial and animal assets: strawberry tile-days are 3–4x and animal-days 2–3x.
- (b) Upkeep has no losses.
- (c) Unit commands have effects: ineffective commands are 10x lower.
- (d) Byproducts are sold: fertilizer, care bonus.
- (e) Market timing and flooding, which depresses the rival.

BC imitates (a) partially and (b)–(c) poorly.

## 4. What PPO did to the economy (M at 20M vs BC, same seeds)

1. **[S] PPO abandoned the long-payback, capital-heavy levers and kept the cheap ones.** Comparing M20 with BC:
   - Strawberry tile-days fall from 200 to 5.
   - Sheep-days fall from 133 to 23.
   - Land purchases stop: 7 of 8 seats stay on NW, where BC reached 2.4 quadrants.
   - Wheat tile-days halve.
   - Cows stay (105 → 108 cow-days).
   - Hands stay or rise (7.2 → 8.5 at hour 23).
   - Unit commands rise (5,704 → 6,594), but ineffective commands go from 565 to 996 and harvests halve (248 → 115).
   - The result is a bank of 30.6k against 75.6k. That matches the reported training slide in M, W&B `r350xr3w`, 63.9k to about 40k.
2. **[O] The cause is not attributed.** The pattern fits two mechanisms:
   - (i) **Investment bias.** Under a bank or margin potential every purchase is an immediate negative reward, and the payback arrives 100–300 steps later. That is beyond the ~10-step effective credit window of `gae_lambda 0.9` (`configs/kaggriculture_4rank_margin.yaml:155-156`), so it must be carried by a critic whose output is a win-probability `2p−1`.
     - In the vs-cha22 run, the margin saturates between about day 20 and day 25 (the margin passes −50k; the bot-set day-20 and day-25 banks give −34k and −66k). Early spending is still penalized there, while late income pays about 0. This **amplifies** bias (i) rather than refuting it.
   - (ii) **Undirected drift** from normalized, clipped Muon steps on a weak signal.
   - Discriminating check, no training needed: per-transition advantages on BUY_SEED STRAWBERRY, BUY_ANIMAL and BUY_LAND frames in a stored rollout. A consistent negative mean supports (i); zero mean with high variance supports (ii).
3. **[S] The training telemetry cannot see any of this.** M's W&B summary has no economic-behavior keys, and its `[nt-probe]` iteration records have 48 keys, none of them behavioral. The engine already computes these counters (`terminal_metrics` `econ_0` and `econ_1`, 32 fields, `lib.rs:1120-1175`). Logging them per update would localize the slide.

## 5. Lever map: observation → representation → choice → execution

| Lever | Observation (current tokens) | Choice (head) | Execution | Stateless? |
|---|---|---|---|---|
| Investment / payback (seed type, animals, land) | money, seed/shed counts, quadrant bits, prices, step and day (`docs/kaggriculture-contract.md` player/global tables) **[S]** | `market_kind`, `market_item`, quantity; unit `PLACE`/`PLANT` | expressible **[S]**; seeds reach the farm only next step (`lib.rs:1464`) | yes. The state is fully observed; payback must be learned by the critic **[O]** |
| Upkeep (water, feed, care, collect) | `watered_today`, `consecutive_unwatered`, `fed_today`, `cared_today`, `fertilizer_available`, age and deadline per tile **[S]** | `unit_kind` + `unit_target` per unit | expressible; masks are **not effect-aware** (no watered, seed or money checks in `src/kaggriculture/grammar.rs`, a grep negative) **[S]**, so no-effect commands stay legal | yes |
| Labor allocation across units | own actor positions and inventories **[S]** | one frame per unit, **decided in parallel with no cross-unit conditioning** (`python/owl/model/kaggriculture_actor.py:6-8`) **[S]** | two units can pick the same tile (the second is ineffective). If the total PLANT demand for a crop exceeds the seeds held, **every** PLANT of that crop becomes PASS (`lib.rs:1550`) **[S]** | yes, but coordination within a turn is an **architecture limit** candidate **[O]**: it plausibly feeds BC's and M20's 565–996 ineffective commands |
| Hiring | `hires_today`, next-hire cost, money **[S]** | market HIRE with the capacity overlay | expressible, up to 10 orders a step | yes |
| Sale timing and quantity | market inventory index and price per product, shop tokens, shop-sell and town-center intervals, step **[S]** (drain is forecastable) | SELL + item + quantity (0–1023) | expressible; each unit walks down the curve | yes. Rival shed stock (future supply) is private and future shop types are RNG; these are **legal limits** for any policy **[S]** |
| Market adaptation / rival | rival farm tiles (crops, yields, animals) are public tokens **[S]**; rival shed is not | same | same | yes. A stateless policy can read the rival's standing crops |
| End-of-game liquidation | step and remaining transitions, shed counts **[S]** | SELL | expressible | yes |

**[S]** No economic lever needs between-turn memory. The own state is fully observed, and the hidden parts (RNG, rival shed) are hidden from every policy. The limits a stateless policy faces are intra-turn coordination (parallel frames) and credit assignment over 100–300-step paybacks. Neither is an observation gap.

## 6. What "earning for real" would take (economics view; mechanisms, not recipes)

- Measure what matters, per update, from the existing econ counters. The same counters give a stable discriminating panel against BC:
  - sale revenue, units sold;
  - ineffective commands, deaths;
  - strawberry/animal tile-days, quadrants;
  - unsold at end.
- A reward in which spending is not an immediate loss. For example, value productive assets at book in the potential: `seat_wealth` already marks plants, animals and land at cost (`lib.rs:2824-2860`). That removes the purchase-time penalty that the λ = 0.9 window cannot repay. **[O]** Untested; the wealth potential would need its own critic-range check.
- In self-play, the relative reward rewards flooding. An absolute-income component keeps "both poor" from scoring 0. **[S]** (logic: the margin reward is zero-sum, as the owner already noted.)
- cha22 vs cha22 (CPU, a few games) would establish the two-competent-seller ceiling before any bank target is set **[O]**.

## Open items

- Per-item sale attribution for cha22 and BC. No per-item econ counter exists; the shed-delta estimate is approximate.
- PLANT-block frequency and same-tile collisions under parallel frames. Not directly counted.
- The payback-credit mechanism in section 4.2(i) vs drift (ii). This needs the advantage-by-action-type check.
- The self-play ceiling (cha22 vs cha22).
