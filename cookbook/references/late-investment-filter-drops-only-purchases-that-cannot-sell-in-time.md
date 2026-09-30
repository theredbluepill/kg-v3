---
type: "Reference"
title: "Late-investment filter drops only purchases that cannot sell in time"
description: "Rule 2 on kg/rule2-late-invest: an off-by-default post-decode filter in the packaged Kaggle agent replaces each BUY_SEED, BUY_ANIMAL, BUY_LAND or HIRE order with the empty market command only when no play could turn it into a sale by the last processed action (step episodeSteps - 2). Thresholds come from the Kaggle engine and engine_rs with line citations; at the default 720/24 configuration the first dropped steps are WHEAT/CARROT 671, TOMATO 527, STRAWBERRY/MELON 479, animals 694, land 695, hires 717 plus hour 23 of every day. Scripted plays in the Kaggle engine sell each last allowed purchase before the end. SELL, BUY_PRODUCT and unit actions are never touched. Local tests only; no episode-level money effect is measured."
tags: ["kaggriculture-v3", "adaptation", "kaggle-runtime", "action-filter"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-10-01"}
sources:
  - resource: "repository:python/owl/kaggriculture/late_invest.py"
  - resource: "repository:python/owl/kaggriculture/kaggle_agent.py"
  - resource: "repository:python/kaggriculture_main.py"
  - resource: "repository:tests/kaggriculture/test_late_invest.py"
  - resource: "repository:ops/late-invest-2026-10-01/checks.log"
  - resource: "repository:engine_rs/src/lib.rs"
  - resource: "repository:pyproject.toml"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/envs/kaggriculture/kaggriculture.py"
  - resource: "uv-cache:kaggle_environments-1.32.7/kaggle_environments/envs/kaggriculture/kaggriculture.json"
---

# Late-investment filter drops only purchases that cannot sell in time

## Claim

Owner, verbatim (2026-09-30/10-01): "Is there something liek this we can do in package/evaluation, not much right?", "State a list for worth considering option?", "if we do it right, it would be assistance, and absolutely no harm right?", "OK go ahead to implement it." The agreed definition of rule 2 is strict: block only purchases that physically cannot produce anything sellable before the game ends, never feed, upkeep, inputs or labour for existing assets, with thresholds taken from the engine. The definition came from the main agent's exchange with the owner. It reached this change through a workflow task, not directly from the owner. Rule 1 (last-day sell-out) is a separate change on `kg/submit-08bc`.

`KaggricultureAgent(block_late_investments=True)` passes the natively validated action through `late_invest.filter_late_investments`. The flag defaults to false; `python/kaggriculture_main.py` sets it only when `KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS=1`. A submission that wants it must set that constant before packaging. The filter reads only the observation's `step` (checked against `day` and `hour`) and the configuration's `turnsPerDay` and `episodeSteps`. It keeps no state and never touches the model. Each dropped order becomes `[]`, the canonical empty market command the engine skips, so the other orders keep their lockstep slots. The filtered action is validated again through the same native round trip.

## Derivation

Cash is the final reward and rises only through `SELL` from the shed. The last processed action is step `S = episodeSteps - 2` (Kaggle `kaggriculture.py:960`, `engine_rs/src/lib.rs:1487`). Unit actions run before the market in a step. Items reach the shed only by a later unit action or the end-of-day drop, so a sale comes at least one step after its harvest. The module docstring tabulates every cited line. Each rule is a lower bound on the earliest sale step, so any configuration can only under-block:

- **Seeds:** blocked when `(day(t+1) + first_yield_day) * T + 1 > S`. Planting is a later unit action; HARVEST needs age >= `first_yield_day`; ongoing crops first yield at the start of that day.
- **Animals:** blocked when `(day(t+2) + 1) * T + 1 > S`. PICKUP and PLACE by one unit take two later steps. The first sellable product is FERTILIZER, set at every end of day. Egg, milk and wool come later.
- **Land:** blocked when `(day(t+1) + 1) * T + 1 > S`. The fastest use is a coop built and stocked on step `t+1` by two units, with its fertilizer ready after that day's end. Crops need two days.
- **Hires:** blocked at hour `T-1` of any day, or when `t + 2 > S`. A day-end hire is removed before it acts; the engine's own wasted-hire counter uses the same condition (`lib.rs:4015-4018`). A hand whose first action is `S` cannot put anything in the shed before the last market.

At 720/24 (`S = 718`) the first dropped steps are WHEAT/CARROT 671, TOMATO 527, STRAWBERRY/MELON 479, animals 694, land 695, and hires 717 plus hour 23 of every day. The day-end hire block applies on every day, not only late ones. It follows from the same physical criterion.

## Verification

`tests/kaggriculture/test_late_invest.py` has 49 cases and reads the Kaggle engine directly:

- **Constants and step structure.** The crop and animal tables equal the Kaggle module's constants. Played episodes at 720/24 and 30/4 show that the agent acts at steps `0..E-2`.
- **Exact thresholds.** At the default configuration, every blocked step set equals the thresholds above, checked over all 719 steps. SELL, BUY_PRODUCT, empty and unknown-item orders, and unit actions are never touched.
- **Scripted plays.** In the Kaggle engine, the last allowed seed (each crop), animal (each kind), land and hire purchases, plus an hour-22 hire, are each sold before the end. Seed and animal plays also run at `episodeSteps` 699, where the sale lands on `S` itself. A day-end hire never appears in the next observation.
- **Agent seam.** With the flag off, actions equal the decoded program byte for byte. With it on, only the expected slots become `[]`, the result passes `validate_action`, and the tiny model's real outputs equal the filter applied to its decode. `main.py` reads the switch as off by default and on only for `1`.

Five threshold mutations were each killed. The last one survived until the 699-step plays were added. On `7184b729` plus this change, ruff, format, `mypy python/ scripts/`, the 3.11 syntax check, docs-fresh and 126 agent/ship/codec tests passed. The receipt is `ops/late-invest-2026-10-01/checks.log`.

## Limits and consequence

- **Direction of error.** For other configurations the land and hire rules are only lower bounds: they may keep a purchase that cannot pay back, but never drop one that can. The land play needs two steps after the fertilizer day starts, because the farmer respawns on the NW tile.
- **Unmeasured value.** No full episode compared money with the filter on and off, so its value is unmeasured. It removes spending only; it cannot add income.
- **Not in any package.** `package_checkpoint.sh` was not run, and no Kaggle submission includes the filter. Submission remains the owner's decision.
- **Legality only.** The filter checks legality and timing, not affordability or holdings. A purchase that could still pay back in principle is kept, even when the agent will never use it.
