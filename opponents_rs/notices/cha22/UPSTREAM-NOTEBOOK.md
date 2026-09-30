# Retained upstream notebook text

These are author claims, not results reproduced in this port.

# Kaggriculture cha22 - route-replay agent

A strong general agent for the Kaggriculture simulation competition. It plays a precomputed
**719-step action tape** per shop sequence (a "route", selected at runtime by a router) and
wraps it in reactive safety and market layers.

## Local full-pool regression

Paired A/B, 12 seeds per seat, fresh random worlds, local fast simulator: **57 opponents /
1,368 games - 1,337W / 0T / 31L (97.7% of decided games), mean paired score delta +6,670.**
A same-build self-play check adds 24 ties (shown last).

Rows are sorted by losses, then by mean score delta. A `0` in the loss column means that
matchup was never lost in its 24 games. Local simulator results, not ladder ratings.

| opponent | W | T | L | win% | mean delta |
|---|---|---|---|---|---|
| kaggriculture_market_rhythm_sale_policy | 20 | 0 | 4 | 83.3% | +2728 |
| kaggriculture_pipe18_six_layers | 22 | 0 | 2 | 91.7% | +1013 |
| kaggriculture_v52_lean_flock_yarn_route | 22 | 0 | 2 | 91.7% | +1120 |
| kaggriculture_harvest_ledger | 22 | 0 | 2 | 91.7% | +1301 |
| kaggriculture_v56_smarter_seeds_and_fertilizer | 22 | 0 | 2 | 91.7% | +1500 |
| shop_router_reactive_v7 | 22 | 0 | 2 | 91.7% | +3827 |
| kaggriculture_more_wheat_smarter_sales | 23 | 0 | 1 | 95.8% | +1473 |
| kaggriculture_master_engine_v4 | 23 | 0 | 1 | 95.8% | +1492 |
| kaggriculture_v51_lean_flock | 23 | 0 | 1 | 95.8% | +1494 |
| one_more_wheat | 23 | 0 | 1 | 95.8% | +1599 |
| v54_productive_idle_workers | 23 | 0 | 1 | 95.8% | +1609 |
| kaggriculture_master_engine_v4 | 23 | 0 | 1 | 95.8% | +1646 |
| kaggriculture_v53_opening_signature | 23 | 0 | 1 | 95.8% | +1696 |
| your_market_list_is_an_order_book | 23 | 0 | 1 | 95.8% | +1718 |
| farmer_john_and_the_idle_seller | 23 | 0 | 1 | 95.8% | +1770 |
| the_metav4_farm_submission_v13 | 23 | 0 | 1 | 95.8% | +1949 |
| kaggriculture_v49_funded_sale_timing_and_worker | 23 | 0 | 1 | 95.8% | +2210 |
| kaggriculture_v50_early_yarn_commit | 23 | 0 | 1 | 95.8% | +2461 |
| v44_market_layers_1234 | 23 | 0 | 1 | 95.8% | +3598 |
| market_smart_farming_kaggriculture | 23 | 0 | 1 | 95.8% | +3716 |
| first_in_line_stock_into_income | 23 | 0 | 1 | 95.8% | +3873 |
| v46_first_turn_microstructure | 23 | 0 | 1 | 95.8% | +3914 |
| farming_score_v5_timing_optimized | 23 | 0 | 1 | 95.8% | +4413 |
| kaggriculture_pipe16_idle_workers | 24 | 0 | 0 | 100.0% | +1342 |
| kaggriculture_v57_funding_order_invariant | 24 | 0 | 0 | 100.0% | +1538 |
| god_s_mode_hacked_stores | 24 | 0 | 0 | 100.0% | +1595 |
| v54_productive_wheat_and_patient | 24 | 0 | 0 | 100.0% | +1649 |
| kaggriculture_v55_one_turn_market_race_edge | 24 | 0 | 0 | 100.0% | +1808 |
| kaggriculture_master_engine_v3 | 24 | 0 | 0 | 100.0% | +2125 |
| the_2945_farm_96_vs_the_top_10_public_bots | 24 | 0 | 0 | 100.0% | +2245 |
| v44_winning_the_same_turn_sale_race | 24 | 0 | 0 | 100.0% | +3732 |
| ready_stock_earlier_sales | 24 | 0 | 0 | 100.0% | +4008 |
| shop_router_reactive_v5 | 24 | 0 | 0 | 100.0% | +4028 |
| v45_first_turn_wheat_round_trip | 24 | 0 | 0 | 100.0% | +4063 |
| v44_market_layers_124 | 24 | 0 | 0 | 100.0% | +4134 |
| v47_reactive_market_coordination | 24 | 0 | 0 | 100.0% | +4189 |
| demand_preserving_turn_sale_timing | 24 | 0 | 0 | 100.0% | +4383 |
| pipe_2_agent | 24 | 0 | 0 | 100.0% | +4543 |
| kaggriculture_pipe_7_wheat_microstructure | 24 | 0 | 0 | 100.0% | +4830 |
| v48_clear_the_queue | 24 | 0 | 0 | 100.0% | +5222 |
| v43_recovering_lost_harvests | 24 | 0 | 0 | 100.0% | +5298 |
| v41_review_candidate | 24 | 0 | 0 | 100.0% | +6052 |
| fully_dynamic_autonomous_agent | 24 | 0 | 0 | 100.0% | +6114 |
| beyond_48_0_128_128_worlds_with_95_cis | 24 | 0 | 0 | 100.0% | +6385 |
| more_yield_smarter_labor | 24 | 0 | 0 | 100.0% | +6642 |
| master_engine_v3 | 24 | 0 | 0 | 100.0% | +6762 |
| master_engine_v2 | 24 | 0 | 0 | 100.0% | +6991 |
| observed_timing_r37 | 24 | 0 | 0 | 100.0% | +7033 |
| v36_guarded_4turn_sales | 24 | 0 | 0 | 100.0% | +7088 |
| herd_safe_sale_window | 24 | 0 | 0 | 100.0% | +7456 |
| better_shop_v4 | 24 | 0 | 0 | 100.0% | +7510 |
| market_smart_base | 24 | 0 | 0 | 100.0% | +8220 |
| score_v35 | 24 | 0 | 0 | 100.0% | +8225 |
| smaller_shock | 24 | 0 | 0 | 100.0% | +8427 |
| v31_prod_sale_priority | 24 | 0 | 0 | 100.0% | +8938 |
| most_powerful_route | 24 | 0 | 0 | 100.0% | +9110 |
| pass_bot | 24 | 0 | 0 | 100.0% | +163036 |
| cha22 (same build, self-play) | 0 | 24 | 0 | - | +0 |

## What the ladder losses taught us

We lost games on the live ladder to some of the strongest opponents. Re-running this agent on
those worlds beats the recorded opponent's score in 14 of them. Analysing those games with our
simulator produced the design targets for this lineage:

- **The deficit is late-game conversion.** Our money path leads through day 17, then loses days
  18-29: the opponents turn the same farm into revenue better in the final third.
- **It is not a shop-draw lottery.** When we control for demand - replaying the recorded games'
  own shop schedules into our simulator - the gap widens instead of closing: the top players win
  by execution, not by lucky shops.
- **The edge is the same-turn sell race.** Across 11k public replays the most robust
  winner-vs-loser difference is realised price per unit (+8-10%): the top bots win the market-hour
  pairing race.
- **Capacity is already matched.** On the same worlds our land, crops, herd and money paths track
  theirs within a few thousand points all game - the gap is conversion, not scale.

---

Run all cells to write the SHA-verified `main.py` and build `submission.tar.gz`.
