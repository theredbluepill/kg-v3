"""Optional final-turn liquidation for the Kaggle agent (off by default).

Money is the whole score and unsold goods are worth nothing at the end. The
last turn the game still resolves is the observation with
``step == episodeSteps - 2`` (the engine and the Kaggle interpreter mark the
game done after that step's market). Worker actions resolve before the market
in the same turn, and there is no end-of-day shed deposit on the last day.

On that observation only, :func:`liquidate_final_turn` rewrites the policy's
action:

* every own actor that stands on a shed access tile and carries a product is
  switched to ``DROP``, so the goods reach the shed before the market;
* the market queue is replaced by one ``SELL`` per product that is or will be
  in the shed, highest current price first, for an upper bound of its units
  (shed stock plus what the dropping actors carry). A ``SELL`` that runs out of
  stock stops filling at no cost. Animals and seeds are not sellable.

The policy's other orders are dropped: anything bought on the last turn is
worth nothing at scoring. On every other observation the action is returned
unchanged. The rule is a pure function of the current observation, the
configuration and the policy's action; nothing carries between calls.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from owl.kaggriculture.kaggle_view import game_configuration
from owl.kaggriculture.types import PRODUCTS, JsonValue, KaggricultureGameConfig

ENV_VAR = "KAGGRICULTURE_FINAL_TURN_LIQUIDATION"
# Native grammar bound for a market quantity (two 32-way quantity slots).
MAX_ORDER_QUANTITY = 1023


def enabled_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """Read the switch: unset or ``"0"`` is off, ``"1"`` is on, anything else raises."""
    value = (os.environ if environ is None else environ).get(ENV_VAR, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_VAR} must be '0' or '1', got {value!r}")
    return value == "1"


def final_resolved_step(configuration: Mapping[str, Any]) -> int:
    """The observation step of the last turn whose actions still resolve."""
    game, _ = game_configuration(configuration)
    return KaggricultureGameConfig.model_validate(game).episode_steps - 2


def shed_access_tiles(board_size: int) -> frozenset[tuple[int, int]]:
    """The four inner-corner tiles from which DROP reaches the shed."""
    half = board_size // 2
    return frozenset(
        {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
    )


def _carries_product(inventory: Mapping[str, int]) -> bool:
    return any(item in PRODUCTS and count > 0 for item, count in inventory.items())


def liquidate_final_turn(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    action: dict[str, JsonValue],
    *,
    order_limit: int,
) -> dict[str, JsonValue]:
    """Return the final-turn liquidation of ``action``, or ``action`` itself."""
    if observation["step"] != final_resolved_step(configuration):
        return action
    game, _ = game_configuration(configuration)
    board_size = KaggricultureGameConfig.model_validate(game).board_size
    farm = observation["farms"][observation["player"]]
    private = observation["private"]
    inventories = private["inventories"]
    positions = [farm["farmer"], *farm["hands"]]
    farmer = action["farmer"]
    hands = action["hands"]
    if not isinstance(hands, list) or len(hands) != len(farm["hands"]):
        raise ValueError(
            f"action has {hands!r} hand commands for {len(farm['hands'])} hands"
        )
    commands: list[JsonValue] = [farmer, *hands]
    access = shed_access_tiles(board_size)
    upper = {item: max(0, int(private["shed"].get(item, 0))) for item in PRODUCTS}
    for index, position in enumerate(positions):
        if index >= len(inventories) or tuple(position) not in access:
            continue
        inventory = inventories[index]
        if not _carries_product(inventory):
            continue
        commands[index] = ["DROP"]
        for item, count in inventory.items():
            if item in upper and count > 0:
                upper[item] += int(count)
    prices = observation["market"]["prices"]
    items = sorted(
        (item for item in PRODUCTS if upper[item] > 0),
        key=lambda item: (-float(prices[item]), PRODUCTS.index(item)),
    )
    market: list[JsonValue] = []
    for item in items[:order_limit]:
        order: list[JsonValue] = ["SELL", item, min(upper[item], MAX_ORDER_QUANTITY)]
        market.append(order)
    return {"farmer": commands[0], "hands": commands[1:], "market": market}
