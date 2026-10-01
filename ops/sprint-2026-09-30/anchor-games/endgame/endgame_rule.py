"""Stateless last-turn liquidation for the Kaggle wrapper (scratch design, not wired in).

Pure function of (current observation, configuration, model action). On the last
turn that the engine still resolves (obs step == episodeSteps - 2, i.e. day 29
hour 22 with the defaults), it returns the model's action with:
  * every actor of ours that stands on a shed access tile and carries products
    switched to DROP (unit actions resolve before the market in the same step);
  * the market queue replaced by one SELL per product, highest current price
    first, for an upper bound of the units that can be in the shed once the
    step's unit actions resolve (shed + every carried unit of that item). A SELL
    that runs out of shed stock simply stops filling; the unused count costs
    nothing. Animals (GOOSE/COW/SHEEP) and seeds are not sellable.
Anything bought on the final turn is worthless at scoring, so the model's
non-SELL orders are dropped. No state carries between calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
MAX_QUANTITY = 1023  # native grammar bound for market quantities


def last_resolved_step(configuration: Mapping[str, Any]) -> int:
    # engine: the game is done after the step whose previous_step >= episodeSteps - 2
    return int(configuration["episodeSteps"]) - 2


def shed_access_tiles(board_size: int) -> set[tuple[int, int]]:
    half = board_size // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def liquidate_final_turn(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    if int(observation["step"]) != last_resolved_step(configuration):
        return action
    seat = observation["player"]
    farm = observation["farms"][seat]
    private = observation["private"]
    access = shed_access_tiles(int(configuration["boardSize"]))
    positions = [farm["farmer"], *farm["hands"]]
    inventories = private["inventories"]
    commands = [action["farmer"], *action["hands"]]
    for index, position in enumerate(positions):
        carries = index < len(inventories) and any(
            item in PRODUCTS and count > 0 for item, count in inventories[index].items()
        )
        if carries and tuple(position) in access:
            commands[index] = ["DROP"]
    upper = {item: max(0, int(private["shed"].get(item, 0))) for item in PRODUCTS}
    for inventory in inventories:
        for item, count in inventory.items():
            if item in upper and count > 0:
                upper[item] += int(count)
    prices = observation["market"]["prices"]
    items = sorted((item for item in PRODUCTS if upper[item] > 0), key=lambda item: -prices[item])
    limit = int(configuration["maxMarketOrdersPerTurn"])
    market = [["SELL", item, min(upper[item], MAX_QUANTITY)] for item in items][:limit]
    return {"farmer": commands[0], "hands": commands[1:], "market": market}
