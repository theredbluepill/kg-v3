# Final-turn liquidation A/B on c50 (fixed-shop engine, 48 paired games)

Date 2026-10-01. Owner, verbatim: "the current kaggle submission didnt sell stuff at the last day?"
Nothing was uploaded or submitted.

## Setup

- **Question.** The 08bc agent reaches the end with sellable goods in the shed. Unsold goods score 0. Does a stateless rule that sells the shed on the last resolved turn raise the margin, and does it change anything else?
- **Mechanism.** SELL on observation step 718 (`episodeSteps - 2`) still counts: worker actions resolve, then the market, then the game is scored.
- **Rule.** `python/owl/kaggriculture/final_turn.py` on `kg/submit-08bc` at `4c99768a`, default off. It only acts on observation step 718:
  - any own actor standing on a shed access tile and carrying products is switched to DROP;
  - the market becomes one SELL per product, for shed stock plus the dropped units (at most 1023), highest price first, within the order limit;
  - BUY and HIRE orders are dropped;
  - the result is re-validated through the native round trip.
- **Switch.** `KAGGRICULTURE_FINAL_TURN_LIQUIDATION=1`, read by `main.py`. When it is unset or `0` the behaviour is unchanged.
- **ON arm.**
  - Package `pkg-c50-ft/`, staged with `pkg_local_mac.py` from the clean head `4c99768a`. Its manifest sha256 is `9786d766…2a7c`.
  - Checkpoint: c50 `0cc80065…a7c2`, slimmed to `f5fd1578…063e`, the same bytes as `pkg-c50`. Config `62e0b5c1…75f7`.
  - Native module: a macOS arm64 build of the head, `maturin build --release --no-default-features`, `81b9bf3f…8507`.
  - Runs: `run_fixedshop_ft.sh c50-ft-on 1` → `games-fixedshop/c50-ft-on/`.
- **OFF arm.** The existing `games-fixedshop/c50/` games (package `pkg-c50`, which has no rule).
- **Games.** The same 8 seeds (93001–93008) × 2 seats × {smaller_market_shock, cha22, v56} = 48 paired games.
  - Both arms used the fixed-shop kaggle-environments 1.32.7: `kaggriculture.py` sha256 `f73d27ce…5004`, identical in all 96 receipts.
  - Mac, `nice -n 10`, 5 games in parallel, 1 torch thread, strict agent (any exception fails the game).
- **Discriminating observation.**
  - If the rule is correct, the replays differ only in our step-718 action (replay index 719).
  - The shed empties and the margin rises by what was sold.
  - Any other difference would mean the new head changed the policy, and the comparison would not be paired.
- **Stopping condition.** One pass of 48 games, then stop.
- **Analysis.** `ab.py` writes `ab.jsonl` (per-game rows) and `ab_tables.md` (full tables, including per-game rows).

## Checks (ON arm, 48 games)

| check | result |
|---|---|
| qualified receipts, 719 calls per game | 48 / 48 |
| exceptions, invalid raw actions, default-PASS returns (fallbacks), ERROR/TIMEOUT/INVALID statuses | 0, 0, 0, 0 |
| replays differ only in our own final-turn action (every other step and every opponent action identical) | 48 / 48 |
| the ON final action equals the packaged rule applied to the OFF obs 718 and model action | 48 / 48 |
| the rule changed the action | 48 / 48 |
| negative margin changes | 0 / 48 |
| opponent bank unchanged | 45 / 48 (the other 3 moved by +2 to +4) |

The pre-718 play is bit-identical, so the new head (main's model code plus the rule) reproduces pkg-c50's policy exactly. Wall time per game averaged 73 s ON and 82 s OFF. That is a host-load difference, not a rule cost.

## Results

The margin change is the ON gap minus the OFF gap, paired per game. The SE is taken over the 8 per-seed means; each seed mean averages 6 games, or 2 per anchor. Mirrored seat pairs are often identical games, so the seed is the independent unit.

| anchor | n | W-L OFF | W-L ON | flips | mean gap OFF | mean gap ON | margin change mean ± SE | min / max | unsold shed/carried/tile OFF | ON |
|---|---|---|---|---|---|---|---|---|---|---|
| smaller_market_shock | 16 | 4-12 | 4-12 | 0 | −4,372 | −3,425 | **+947 ± 233** | +95 / +1,945 | 21.9 / 8.8 / 9.3 | 0.0 / 8.8 / 9.3 |
| cha22 | 16 | 2-14 | 2-14 | 0 | −9,456 | −8,777 | **+679 ± 121** | +137 / +1,103 | 17.4 / 10.9 / 12.6 | 0.0 / 10.9 / 12.6 |
| v56 | 16 | 0-16 | 0-16 | 0 | −10,352 | −9,356 | **+995 ± 228** | +110 / +2,368 | 21.4 / 14.3 / 10.9 | 0.0 / 14.3 / 10.9 |
| **all** | 48 | 6-42 | 6-42 | 0 | −8,060 | −7,186 | **+874 ± 157** | +95 / +2,368 | 20.2 / 11.3 / 10.9 | 0.0 / 11.3 / 10.9 |

- **Every game gained.** The rule improved the margin in all 48 games, and the shed ends empty in all 48. It flipped no game: the 6 wins stay wins and gain more. The closest remaining losses are v56 s93005 (−1,940 → −1,283) and s93008 (−2,134 → −1,598).
- **The rule never had to add a DROP (0 of 48).** On the final turn the model already DROPs with any carrier that stands on a shed tile.
- **Goods the rule cannot reach.**
  - A mean of 11.3 carried units per game belong to actors away from the shed. The day-29 end-of-day deposit never runs, so those goods cannot be sold on the final turn.
  - A mean of 10.9 units are still on tiles.
  - Neither is reachable by a final-turn-only rule.
- **The Kaggle screenshot game** (episode 115847855, seat 1, obs 718):
  - The model ordered SELL WHEAT 12, FERTILIZER 2, FERTILIZER 2.
  - The rule instead orders STRAWBERRY 8, MILK 6, WHEAT 22, FERTILIZER 8, which is exactly the whole shed.
  - The earlier counterfactual (`cf.py`) values selling the whole shed at +2,486, a win by 1,025. That is an offline counterfactual, not a replayed game.
  - On all 10 Kaggle ladder replays in `kaggle/`, the rule's order sells the full shed.

## Limits

- Only the fixed-shop local engine, one checkpoint (c50), three anchors and 8 seeds; no Kaggle ladder games were played with the rule on.
- The official-engine arm and other checkpoints (p4 = 08bc, f610, 60f2) were not re-run with the rule. The earlier offline estimate for them was +0.7k to +2.1k per game.
- A shipped package runs with the rule off. Kaggle sets no environment variable, so shipping the rule on needs the default flipped in `main.py` or a builder option (neither is done).
- The remaining leak is the carried goods (about 11 units per game) and the goods on tiles (about 11 units per game). Reaching them would need last-day routing to the shed, which would change the model's actions before step 718; that was not tested.
