"""The optional final-turn liquidation rule and its agent switch."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kaggle_environments import make
from owl.kaggriculture.final_turn import (
    ENV_VAR,
    MAX_ORDER_QUANTITY,
    enabled_from_env,
    final_resolved_step,
    liquidate_final_turn,
    shed_access_tiles,
)
from owl.kaggriculture.kaggle_agent import KaggricultureAgent, validate_action
from owl.kaggriculture.types import ACTION_SLOTS, MAX_FRAMES, PRODUCTS

from tests.kaggriculture.kaggle_fixtures import write_model_root

CONFIGURATION = {
    "episodeSteps": 720,
    "boardSize": 10,
    "maxMarketOrdersPerTurn": 10,
    "__raw_path__": "/kaggle_simulations/agent/main.py",
}
PRICES = {
    "WHEAT": 21,
    "CARROT": 60,
    "TOMATO": 83,
    "STRAWBERRY": 183,
    "MELON": 106,
    "EGG": 54,
    "MILK": 135,
    "WOOL": 93,
    "FERTILIZER": 14,
}
MODEL_ACTION: dict[str, Any] = {
    "farmer": ["NORTH"],
    "hands": [["HARVEST"], ["WATER"], ["PASS"]],
    "market": [["SELL", "WHEAT", 12], ["BUY_SEED", "WHEAT", 2], ["HIRE"]],
}


def _observation(step: int = 718) -> dict[str, Any]:
    """Seat 1 of the screenshot game's shape: shed goods, two carrying actors."""
    own = {
        "farmer": [4, 4],  # shed tile, carries wheat -> DROP
        "hands": [
            [5, 5],  # shed tile, carries only animals -> keep
            [0, 0],  # off the shed, carries wheat -> cannot sell
            [4, 5],  # shed tile, carries nothing -> keep
        ],
    }
    return {
        "step": step,
        "player": 1,
        "farms": [{"farmer": [4, 4], "hands": []}, own],
        "private": {
            "shed": {
                "STRAWBERRY": 8,
                "MILK": 6,
                "WHEAT": 22,
                "FERTILIZER": 8,
                "SHEEP": 1,
                "EGG": 0,
            },
            "inventories": [
                {"WHEAT": 3, "FERTILIZER": 1},
                {"SHEEP": 2},
                {"WHEAT": 5},
                {},
            ],
        },
        "market": {"prices": dict(PRICES)},
    }


def _validate(action: object, *, actors: int = 4, order_limit: int = 10) -> None:
    validate_action(
        action,
        actors=actors,
        order_limit=order_limit,
        hire_limit=241,
        scratch=np.empty((MAX_FRAMES, ACTION_SLOTS), dtype=np.int64),
    )


def test_switch_defaults_off_and_rejects_unknown_values() -> None:
    assert enabled_from_env({}) is False
    assert enabled_from_env({ENV_VAR: "0"}) is False
    assert enabled_from_env({ENV_VAR: "1"}) is True
    for bad in ("", "true", "yes", "2"):
        with pytest.raises(ValueError, match=ENV_VAR):
            enabled_from_env({ENV_VAR: bad})


def test_final_resolved_step_follows_episode_steps() -> None:
    assert final_resolved_step(CONFIGURATION) == 718
    assert final_resolved_step(CONFIGURATION | {"episodeSteps": 3}) == 1
    assert shed_access_tiles(10) == {(4, 4), (5, 4), (4, 5), (5, 5)}


@pytest.mark.parametrize("step", [0, 700, 717, 719])
def test_rule_is_the_identity_off_the_final_turn(step: int) -> None:
    action = copy.deepcopy(MODEL_ACTION)
    assert (
        liquidate_final_turn(_observation(step), CONFIGURATION, action, order_limit=10)
        is action
    )
    assert action == MODEL_ACTION


def test_final_turn_drops_carried_goods_and_sells_every_product() -> None:
    action = copy.deepcopy(MODEL_ACTION)
    observation = _observation()
    before = copy.deepcopy(observation)
    result = liquidate_final_turn(observation, CONFIGURATION, action, order_limit=10)
    assert observation == before
    assert action == MODEL_ACTION
    # Only the carrier of products on a shed tile drops; the animal carrier,
    # the carrier away from the shed and the empty hand keep their commands.
    assert result["farmer"] == ["DROP"]
    assert result["hands"] == [["HARVEST"], ["WATER"], ["PASS"]]
    # Shed plus the dropping farmer's goods, highest price first; no SHEEP,
    # no zero-stock EGG, no BUY or HIRE, and not the off-shed hand's wheat.
    assert result["market"] == [
        ["SELL", "STRAWBERRY", 8],
        ["SELL", "MILK", 6],
        ["SELL", "WHEAT", 25],
        ["SELL", "FERTILIZER", 9],
    ]
    _validate(result)


def test_market_respects_the_order_limit_and_the_quantity_bound() -> None:
    observation = _observation()
    observation["private"]["shed"] = {item: 2 for item in PRODUCTS} | {"WHEAT": 5000}
    observation["private"]["inventories"] = [{}, {}, {}, {}]
    action = copy.deepcopy(MODEL_ACTION)
    result = liquidate_final_turn(observation, CONFIGURATION, action, order_limit=3)
    assert result["market"] == [
        ["SELL", "STRAWBERRY", 2],
        ["SELL", "MILK", 2],
        ["SELL", "MELON", 2],
    ]
    _validate(result, order_limit=3)
    full = liquidate_final_turn(observation, CONFIGURATION, action, order_limit=10)
    assert len(full["market"]) == len(PRODUCTS)
    assert ["SELL", "WHEAT", MAX_ORDER_QUANTITY] in full["market"]
    _validate(full)


def test_equal_prices_keep_the_product_order() -> None:
    observation = _observation()
    observation["market"]["prices"] = {item: 50 for item in PRODUCTS}
    result = liquidate_final_turn(
        observation, CONFIGURATION, copy.deepcopy(MODEL_ACTION), order_limit=10
    )
    assert [order[1] for order in result["market"]] == [
        "WHEAT",
        "STRAWBERRY",
        "MILK",
        "FERTILIZER",
    ]


def test_hand_count_mismatch_fails_loudly() -> None:
    action = copy.deepcopy(MODEL_ACTION)
    action["hands"] = action["hands"][:2]
    with pytest.raises(ValueError, match="hand commands"):
        liquidate_final_turn(_observation(), CONFIGURATION, action, order_limit=10)


def _short_game_observations() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Both resolvable observations of a 3-step game (steps 0 and 1)."""
    env = make(
        "kaggriculture", configuration={"seed": 5, "episodeSteps": 3}, debug=True
    )
    env.reset()
    result = []
    for _ in range(2):
        observation = json.loads(json.dumps(env.state[0].observation))
        result.append((observation, dict(env.configuration)))
        env.step([{"farmer": ["PASS"], "hands": [], "market": []}] * 2)
    return result


def _agent(root: Path, *, liquidation: bool) -> KaggricultureAgent:
    return KaggricultureAgent(
        root,
        deterministic=True,
        strict=True,
        min_overage_time=0.0,
        final_turn_liquidation=liquidation,
    )


def test_agent_applies_the_rule_only_when_switched_on(tmp_path: Path) -> None:
    root = write_model_root(tmp_path / "primary")
    off = _agent(root, liquidation=False)
    on = _agent(root, liquidation=True)
    (first, config), (last, _) = _short_game_observations()
    assert last["step"] == final_resolved_step(config) == 1
    # Give the last observation goods: shed stock and a farmer (spawned on a
    # shed tile) carrying milk.
    seat = last["player"]
    assert tuple(last["farms"][seat]["farmer"]) in shed_access_tiles(10)
    last["private"]["shed"]["WHEAT"] = 4
    last["private"]["inventories"][0] = {"MILK": 2}

    assert on.act(first, config) == off.act(first, config)
    model = off.act(last, config)
    liquidated = on.act(last, config)
    assert liquidated == liquidate_final_turn(last, config, model, order_limit=10)
    assert liquidated["farmer"] == ["DROP"]
    assert liquidated["market"] == [["SELL", "MILK", 2], ["SELL", "WHEAT", 4]]
    assert (off.liquidations, on.liquidations) == (0, 1)
    assert on.caught_errors == 0
