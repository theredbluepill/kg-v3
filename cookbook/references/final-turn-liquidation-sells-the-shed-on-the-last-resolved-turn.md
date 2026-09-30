---
type: "Reference"
title: "Final-turn liquidation sells the shed on the last resolved turn"
description: "Adaptation and paired check, 2026-10-01. kg/submit-08bc 4c99768a adds an optional, default-off stateless rule to the Kaggle agent (KAGGRICULTURE_FINAL_TURN_LIQUIDATION=1). On observation step episodeSteps-2 it DROPs product carriers on shed tiles and replaces the market with one SELL per product for the full shed (plus dropped units), dropping BUY/HIRE. On c50, fixed-shop engine, 48 paired games vs smaller_market_shock, cha22 and v56 (8 seeds x 2 seats): 0 errors or fallbacks, play identical before step 718 in all 48, shed emptied in all 48, margin +874 ± 157 per game (SE over 8 seeds), no game worse, 0 of 42 losses flipped. About 11 carried and 11 tile units per game stay unsold. Packages keep the rule off unless built with --final-turn-liquidation (kg/package-60m), which bakes it on in the packaged main.py only; rule 2 stays on its env switch."
tags: ["kaggriculture-v3", "adaptation", "kaggle-runtime", "evaluation", "finding"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "user-directive:2026-09-30:the-current-kaggle-submission-didnt-sell-stuff-at-the-last-day"
  - resource: "repository:python/owl/kaggriculture/final_turn.py"
  - resource: "repository:python/owl/kaggriculture/kaggle_agent.py"
  - resource: "repository:python/kaggriculture_main.py"
  - resource: "repository:tests/kaggriculture/test_final_turn.py"
  - resource: "repository:scripts/build_kaggriculture_submission.py"
  - resource: "repository:scripts/package_checkpoint.sh"
  - resource: "repository:tests/scripts/test_build_kaggriculture_submission.py"
  - resource: "repository:README.md"
  - resource: "user-directive:2026-10-01:you-can-package-but-not-need-to-submit-60m"
  - resource: "repository:ops/submit-08bc-2026-09-30/final-turn-ab/ab.md"
  - resource: "repository:ops/submit-08bc-2026-09-30/final-turn-ab/ab.py"
  - resource: "repository:ops/submit-08bc-2026-09-30/final-turn-ab/ab.jsonl"
  - resource: "repository:ops/submit-08bc-2026-09-30/final-turn-ab/ab_tables.md"
  - resource: "repository:ops/submit-08bc-2026-09-30/final-turn-ab/run_fixedshop_ft.sh"
  - resource: "repository:ops/submit-08bc-2026-09-30/final-turn-ab/pkg-c50-ft-manifest.json"
  - resource: "local-untracked:~/kg-v3-int/ops/earn-money-2026-09-30/anchor-games/games-fixedshop/c50-ft-on"
  - resource: "local-untracked:~/kg-v3-int/ops/earn-money-2026-09-30/anchor-games/games-fixedshop/c50"
  - resource: "local-untracked:~/kg-v3-int/engine_rs/src/lib.rs"
---

# Final-turn liquidation sells the shed on the last resolved turn

## Claim

Only money counts at the end; unsold goods are worth 0. The rules make the last turn a selling opportunity:

- **Step 718 still resolves.** The last observation whose actions resolve has `step == episodeSteps - 2`, which is 718 of 720. The engine and the Kaggle interpreter both mark the game done after that step's market.
- **Order within the turn.** Worker actions resolve before the market in the same turn. A DROP on a shed access tile therefore reaches the shed in time to be sold.
- **What can be sold.** Only shed goods can be sold. On the last day there is no end-of-day deposit.

The 08bc and c50 policies see step, day and hour, yet still end with about 20 shed units per game.

`python/owl/kaggriculture/final_turn.py` is an optional rule that uses only the current observation. It leaves every other observation's action unchanged. On the final resolved observation it rewrites the action:

- every own actor that stands on a shed access tile and carries a product is switched to DROP;
- the market becomes one SELL per product for an upper bound on its units: shed stock plus what the dropping actors carry, capped at 1023. Orders go highest price first, within the order limit. A SELL that runs out of stock simply stops filling;
- BUY and HIRE orders are removed, because anything bought is worthless at scoring.

`KaggricultureAgent(final_turn_liquidation=...)` applies the rule after the model's action has been validated, then validates the rewritten action again through the native round trip. `main.py` reads `KAGGRICULTURE_FINAL_TURN_LIQUIDATION`:

- unset or `0` means off, which is the default;
- `1` means on;
- any other value raises.

Kaggle sets no such variable, so a package keeps the rule off unless it is baked in.

## Shipping rule 1 on

Owner decisions: rule 1 is to be applied, and "let's not apply rule2."; for the 60M checkpoint, "you can package, but not need to submit 60m".

- `scripts/build_kaggriculture_submission.py --final-turn-liquidation` (and `scripts/package_checkpoint.sh --final-turn-liquidation`) rewrites the staged `main.py` only: the agent is constructed with `final_turn_liquidation=True`, and the docstring says so. Each replaced text must occur exactly once in `python/kaggriculture_main.py`, or the build fails.
- The repository's `main.py` still reads `KAGGRICULTURE_FINAL_TURN_LIQUIDATION` (default off), so the agent tests keep their meaning. Rule 2 is never baked; it stays on `KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS`, which Kaggle never sets.
- The inner manifest's `entrypoint` records `final_turn_liquidation` (`baked on` or `environment (off)`) and `block_late_investments` (`environment (off)`). The packager's verify stage fails if the packaged `main.py` wiring or the manifest disagrees with the flag, and `PACKAGE.md` prints both.
- Checks on the `kg/package-60m` working tree: 13 builder tests, including one that bakes, compares the packaged `main.py` with the repository file and compiles it, and one that refuses 0 or 2 matches. The 141 tests of the merge-check list passed, and ruff, mypy and docs-freshness were clean.
- Limit: the 40-turn Kaggle-image episode never reaches step 718. Only a `--full-episode` run exercises the baked rule in the image.

## Evidence

- **Unit tests** (`tests/kaggriculture/test_final_turn.py`, 11 tests) cover:
  - the switch values;
  - the final-step arithmetic;
  - that the rule returns the policy's action unchanged on steps 0, 700, 717 and 719;
  - DROP only for product carriers on shed tiles;
  - SELL composition, which excludes animals, zero stock, the carriers away from the shed, and BUY/HIRE;
  - the order limit, the 1023 bound and price-tie order;
  - that a hand-count mismatch raises;
  - an agent that is on only when switched on, over a real 3-step Kaggle game.

  On the committed tree, the non-slow suite passed (3,058 passed, 9 skipped), and ruff, mypy and docs-freshness were clean. `just` is not installed on this Mac, so these checks were run as the recipe commands directly.
- **Paired A/B** (`ops/submit-08bc-2026-09-30/final-turn-ab/ab.md`).
  - Both arms played the c50 checkpoint on the fixed-shop kaggle-environments 1.32.7 engine: 8 seeds × 2 seats × {smaller_market_shock, cha22, v56}, strict agent.
  - The ON arm used `pkg-c50-ft`, built from `4c99768a` with a macOS arm64 `--no-default-features` module and the same slim checkpoint bytes as the OFF arm. The OFF arm is the existing `games-fixedshop/c50` games.
  - Health: 48/48 games qualified, with 0 exceptions, 0 invalid actions, 0 default-PASS returns and 0 bad statuses.
  - In 48/48 games, the only replay difference is our own step-718 action. In 48/48 games, that action equals the packaged rule applied to the OFF observation.
  - The margin change was +874 ± 157 per game (SE over the 8 seed means): +947 ± 233 against smaller_market_shock, +679 ± 121 against cha22 and +995 ± 228 against v56. The range was +95 to +2,368, with no negative game.
  - Win/loss stayed 6-42 (0 flips). Shed units at the end went from 20.2 to 0.
- **Kaggle screenshot game** (episode 115847855): applied to the recorded observation 718, the rule sells the whole shed (STRAWBERRY 8, MILK 6, WHEAT 22, FERTILIZER 8). An earlier offline counterfactual values that at +2,486, which would have turned the 1,459 loss into a 1,025 win. That figure is a counterfactual, not a replayed game.

## Consequence

- The rule recovers the shed value on every tested game at no measured cost. The owner chose to apply it; the builder option above ships it on without flipping the repository default.
- **What it cannot reach.** The rule never had to add a DROP (0 of 48): the model already drops with carriers standing on shed tiles. What stays unsold is:
  - carried goods on actors away from the shed, 11.3 units per game;
  - yields still on tiles, 10.9 units per game.

  Recovering those needs last-day routing, which would change actions before step 718 and has not been tested.

## Limits

- One checkpoint, one local engine variant (fixed shop), three anchors and eight seeds; mirrored seat pairs are often identical games.
- No Kaggle ladder or official-engine games were played with the rule.
- p4 (08bc), f610 and 60f2 were not re-run.
- Mac host, not Kaggle hardware.
- No W&B run: this local harness has no telemetry.
