# Rule 2 (late-investment filter) A/B on c50, on top of rule 1 (fixed-shop engine, 48 paired games)

Date 2026-10-01. Owner, verbatim (2026-09-30/10-01): "if we do it right, it would be assistance, and absolutely no harm right?"; "OK go ahead to implement it."
The owner's words reached this task through the main agent's workflow task. Nothing was uploaded or submitted, no pod was used and no defaults were flipped.

## Setup

- **Question.** Rule 2 blocks only purchases that cannot produce anything sellable before the last resolved step. Does it add margin, or change wins, on top of rule 1 (final-turn liquidation)? And is it harmless in practice, the property the owner asked about?
- **Code.** `kg/submit-08bc` at `31c99619` is a `--no-ff` merge of `kg/rule2-late-invest` (`a53ad2bf`) into `4c99768a` (rule 1).
  - Both switches coexist and are off by default. `KAGGRICULTURE_FINAL_TURN_LIQUIDATION` controls rule 1 and `KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS` controls rule 2. For each, `0` or unset is off, `1` is on, and any other value raises.
  - When both are on, rule 2 filters the validated action first and rule 1 then rewrites the final turn. Each rewrite is re-validated natively.
  - Merge checks: ruff, mypy (84 files), the 3.11 syntax check, docs-fresh and 139 agent/ship tests all pass (`ops/late-invest-2026-10-01/merge-checks.log` on the branch).
- **Rule 2 thresholds.** These are the engine's own values, cited in the `late_invest.py` docstring (Kaggle `kaggriculture.py` and `engine_rs/src/lib.rs` line citations).
  - At 720/24 (last processed step 718), BUY_SEED WHEAT/CARROT is blocked from step 671 (first_yield_day 2, `kaggriculture.py:12-13`).
  - TOMATO is blocked from 527 and STRAWBERRY/MELON from 479. Animals are blocked from 694 and land from 695.
  - HIRE is blocked at hour 23 of every day and at step 717 and later.
  - Feed/BUY_PRODUCT, SELL and unit actions are never touched.
- **Arms**, all on c50 (`0cc80065…a7c2`, slimmed `f5fd1578…063e`, config `62e0b5c1…75f7`). Every arm uses the same 8 seeds (93001–93008) × 2 seats × {smaller_market_shock, cha22, v56} = 48 games.
  - **OFF:** `games-fixedshop/c50/`, package `pkg-c50`, no rules.
  - **A (rule 1):** `games-fixedshop/c50-ft-on/`, package `pkg-c50-ft` (`4c99768a`), rule 1 on. These are the rule-1 A/B receipts (`endgame/ab.md`).
  - **B (rules 1+2):** `games-fixedshop/c50-r12-on/`, package `pkg-c50-r12`, staged with `pkg_local_mac.py` from a clean clone of `31c99619`.
    - Manifest sha256 `5168aec1…06ac`. Its files differ from `pkg-c50-ft` only in `main.py`, `owl/kaggriculture/kaggle_agent.py` and the new `owl/kaggriculture/late_invest.py`.
    - Both switches were `1`, and every game's agent load line confirms `final_turn_liquidation=1 block_late_investments=1`.
- **Native module.** Arm B reuses rule 1's macOS arm64 `rs.abi3.so` (`81b9bf3f…8507`, `maturin build --release --no-default-features` from `4c99768a`).
  - `git diff 4c99768a 31c99619` touches no `.rs`, Cargo or pyproject file, so the Rust source matches the merged head.
  - The suggested `scratchpad/wheelx/owl/rs.abi3.so` (`c17aa4d7…`) is pkg-c50's older module, built from an older source tree, so it was not used.
- **Engine and host.** Fixed-shop kaggle-environments 1.32.7 (`kaggriculture.py` `f73d27ce…5004`, the same in all 48 B receipts). Mac, `nice -n 10`, 5 games in parallel, 1 torch thread, strict agent. Mean wall time was 76 s per game.
- **Block log.** `endgame/run_game_logged.py` runs `run_game.py` unchanged with a pass-through tap on the agent's `filter_late_investments`. It records each blocked order with its step, the original and filtered market, and the reason. It also keeps the agent's captured stdout lines as a cross-check. Output goes to `games-fixedshop/c50-r12-on/blocks/<game>.json`.
- **Discriminating observation.**
  - A game with no block must replay identically to arm A.
  - A game with a block must first differ from A exactly at its first blocked step, where the model's pre-filter market must equal A's market.
  - The margin effect is then B − A, paired per game.
- **Stopping condition.** One pass of 48 games, then stop.
- **Analysis.** `endgame/ab_rule2.py` produces `ab_rule2.jsonl` (per-game rows) and `ab_rule2_tables.md` (every block, per-game table). The runner script is `endgame/run_fixedshop_r12.sh`.

## Checks (arm B, 48 games)

| check | result |
|---|---|
| qualified receipts, 719 calls per game | 48 / 48 |
| exceptions, invalid raw actions, default-PASS returns (fallbacks), ERROR/TIMEOUT/INVALID statuses | 0, 0, 0, 0 |
| one package manifest, one engine hash | yes, yes |
| tap installed; both switches on (load line); rule 1 rewrote the final turn exactly once | 48, 48, 48 |
| tapped block count equals the agent's own stdout block count | 48 / 48 |
| games with no block replay identically to arm A | 32 / 32 |
| games with a block first differ at the first blocked step, with the pre-filter market equal to A's | 16 / 16 |
| every blocked order re-derives a rule-2 reason | 48 / 48 |

Before offline, rule 2 was applied to arm A's recorded actions. It would have blocked 32 `BUY_SEED WHEAT` orders in the same 16 games, and nothing else. In B, 24 were blocked, because after the first block the game diverges and the model orders fewer.

## Results

The margin is our bank minus the opponent's. The SE is taken over the 8 per-seed means. Mirrored seat pairs are often identical games, so the seed is the independent unit.

| anchor | n | W-L OFF | W-L A (r1) | W-L B (r1+r2) | mean gap A | mean gap B | **B − A** mean ± SE | B − A min / max | B − OFF mean ± SE | games with blocks | blocked orders | unsold shed/carried/tile A | B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smaller_market_shock | 16 | 4-12 | 4-12 | 4-12 | −3,425 | −3,455 | **−30 ± 27** | −220 / +111 | +917 ± 218 | 6 | 10 | 0.0 / 8.8 / 9.3 | 0.0 / 11.9 / 8.6 |
| cha22 | 16 | 2-14 | 2-14 | 2-14 | −8,777 | −8,774 | **+3 ± 15** | −69 / +90 | +681 ± 124 | 4 | 6 | 0.0 / 10.9 / 12.6 | 0.0 / 10.5 / 12.5 |
| v56 | 16 | 0-16 | 0-16 | 0-16 | −9,356 | −9,331 | **+25 ± 33** | −114 / +404 | +1,021 ± 226 | 6 | 8 | 0.0 / 14.3 / 10.9 | 0.0 / 14.8 / 11.9 |
| **all** | 48 | 6-42 | 6-42 | 6-42 | −7,186 | −7,187 | **−1 ± 21** | −220 / +404 | +873 ± 152 | 16 | 24 | 0.0 / 11.3 / 10.9 | 0.0 / 12.4 / 11.0 |

**Blocked purchases by category.** Every block was `BUY_SEED WHEAT 1`, placed on day 28 at hours 5–9 (steps 677–681). Wheat is blocked from step 671 because its first yield (day 2) falls after the last processed step. The counts were 10 against smaller_market_shock, 6 against cha22 and 8 against v56.

No seed of any other crop, no animal, no land and no hire was ever blocked. c50 does not buy them late, and it never hired at hour 23 in these games. Every order is listed in `ab_rule2_tables.md`.

- **Wins.** No game changed result: 6-42 in all three arms, 0 flips in either direction against A.
- **Margin.** Rule 2 adds nothing measurable on top of rule 1: −1 ± 21 per game overall.
  - 32 games are identical (Δ 0), 8 gained and 8 lost. The worst was −220 (smaller_market_shock s93008, both seats) and the best was +404 (v56 s93007 seat 1).
  - B − OFF (+873 ± 152) equals rule 1's own effect (+874 ± 157).
- **Why the effect is noise, not saving.**
  - A blocked wheat seed saves its price, 10 (`kaggriculture.py:12`). That is 240 over the 24 blocks, or 5 per game.
  - The observed per-game swings (−220 to +404) are 20–40 times larger. They come from trajectory divergence: the extra cash and missing seed change the model's later observations and actions.
  - The opponent bank changed in 14 of the 16 blocked games, through the shared market and town.
  - The direction of these swings is not controlled by the rule.
- **Unsold goods at the end.** The shed is empty in every B game, because rule 1 still sells it. Carried and on-tile units are unchanged within noise (11.3 → 12.4 carried, 10.9 → 11.0 on tiles).
- **Errors and fallbacks.** 0.

## Recommendation for packaging and evaluation defaults (not applied; the main agent decides)

- **Rule 1 (`KAGGRICULTURE_FINAL_TURN_LIQUIDATION`): turn on by default.**
  - It never reduced wins or margin: +874 ± 157, 48 of 48 games better, 0 worse, W-L unchanged.
  - It changes only the step-718 action.
- **Rule 2 (`KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS`): leave off by default.**
  - It never reduced wins (0 flips), and its mean margin effect is zero within noise (−1 ± 21).
  - But in 8 of 48 games the margin was lower than with rule 1 alone, by up to 220. Its direct saving is about 5 per game on c50.
  - It does not meet the "does not reduce margin" bar in every game, and it has no measured upside to trade against.
  - It is correct and harmless in the strict physical sense: it never blocked a purchase that could still sell. Keep it available for policies that make large late purchases, and re-test it there.

## Limits

- Only c50, the local fixed-shop engine, three anchors, 8 seeds and one pass. No Kaggle ladder games were played with either rule on.
- c50 made only late wheat-seed purchases here, so rule 2's animal, land, hire and long-crop branches were exercised only by the unit tests, not in games.
- The SE uses 8 seed means. Seat-mirrored games are often identical, so the effective sample is small.
- Replays, logs and the package stay local in `kg-v3-int/ops/earn-money-2026-09-30/anchor-games/`. Compact receipts, block logs, scripts and tables are committed on `kg/submit-08bc` under `ops/late-invest-2026-10-01/ab-rule2/`.
