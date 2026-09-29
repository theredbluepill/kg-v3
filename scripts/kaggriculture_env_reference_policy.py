"""The reviewed Task 1.4 observation-local, non-learning oracle recipe."""

from __future__ import annotations

from typing import Any

POLICY = "native-lifecycle-reference-v1"


def reward_config(game: int) -> dict[str, Any]:
    if not 0 <= game < 16:
        raise ValueError("reference game index must be in 0..15")
    return {
        "reward_mode": "win_loss",
        "econ_shaping": 0.02 if game < 8 else 0.2,
        "econ_starvation_weight": 4.0,
        "econ_drought_weight": 1.0,
        "econ_cap": 0.25,
        "econ_ineffective_weight": 0.001 if game < 8 else 0.0,
        "econ_ineffective_cap": 0.10,
    }


def action(public: dict[str, Any], seat: int) -> dict[str, Any]:
    """Read the current clock and current hands; keep no cross-turn state."""
    if seat not in (0, 1):
        raise ValueError("reference policy seat must be 0 or 1")
    step = public["step"]
    farmer: list[str] = ["PASS"]
    market: list[list[str | int]] = []
    if seat == 0:
        if step == 0:
            market = [
                ["BUY_SEED", "WHEAT", 2],
                ["BUY_ANIMAL", "GOOSE", 1],
                ["BUY_PRODUCT", "WHEAT", 3],
                ["HIRE"],
                ["BUY_LAND"],
            ]
        farmers = {
            1: ["PICKUP", "GOOSE"],
            2: ["BUILD_COOP"],
            3: ["PLACE", "GOOSE"],
            4: ["NORTH"],
            5: ["PLANT", "WHEAT"],
            6: ["WATER"],
            8: ["HARVEST"],
        }
        if step in farmers:
            farmer = farmers[step]
        if step == 10:
            market = [[], ["BUY_PRODUCT", "WHEAT", 0]]
    elif step == 0:
        market = [["BUY_PRODUCT", "WHEAT", 2]]
    elif step == 2:
        market = [["SELL", "WHEAT", 2]]
    return {
        "farmer": farmer,
        "hands": [["PASS"] for _ in public["farms"][seat]["hands"]],
        "market": market,
    }
