"""Rule 2: the late-investment filter drops only purchases that cannot pay back.

Thresholds are checked three ways: against the Kaggle engine's own constants,
as exact first-blocked steps at the default configuration, and by scripted
plays in the real Kaggle engine that turn the last allowed purchase of each
kind into a sale before the game ends (so the filter never over-blocks there).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as kaggle_engine
from owl.kaggriculture import kaggle_agent, late_invest
from owl.kaggriculture.codec import decode_action
from owl.kaggriculture.kaggle_agent import KaggricultureAgent, validate_action
from owl.kaggriculture.late_invest import (
    ANIMALS,
    CROP_FIRST_YIELD_DAY,
    FERTILIZER_READY_DAYS,
    GameClock,
    blocked_reason,
    filter_late_investments,
    observation_step,
)
from owl.kaggriculture.types import ACTION_SLOTS, MAX_FRAMES, JsonValue

from tests.kaggriculture.kaggle_fixtures import write_agent_dir, write_model_root

DEFAULT_CONFIGURATION = {"episodeSteps": 720, "turnsPerDay": 24}
CLOCK = GameClock.of(DEFAULT_CONFIGURATION)
LAST = CLOCK.last_action_step
PASS: dict[str, JsonValue] = {"farmer": ["PASS"], "hands": [], "market": []}

# First dropped step at the default configuration (720 steps, 24 per day).
FIRST_BLOCKED = {
    ("BUY_SEED", "WHEAT"): 671,
    ("BUY_SEED", "CARROT"): 671,
    ("BUY_SEED", "TOMATO"): 527,
    ("BUY_SEED", "STRAWBERRY"): 479,
    ("BUY_SEED", "MELON"): 479,
    ("BUY_ANIMAL", "GOOSE"): 694,
    ("BUY_ANIMAL", "COW"): 694,
    ("BUY_ANIMAL", "SHEEP"): 694,
    ("BUY_LAND", None): 695,
}


def _order(op: str, item: str | None) -> list[JsonValue]:
    return [op] if item is None else [op, item, 1]


def _blocked(order: list[JsonValue], step: int) -> bool:
    return blocked_reason(order, step, CLOCK) is not None


def _validate(action: object, *, actors: int = 1) -> None:
    validate_action(
        action,
        actors=actors,
        order_limit=10,
        hire_limit=241,
        scratch=np.empty((MAX_FRAMES, ACTION_SLOTS), dtype=np.int64),
    )


# --- engine constants and step structure -----------------------------------


def test_tables_match_the_kaggle_engine_constants() -> None:
    engine_days = {
        crop: data["first_yield_day"] for crop, data in kaggle_engine.CROPS.items()
    }
    assert engine_days == CROP_FIRST_YIELD_DAY
    assert set(kaggle_engine.ANIMALS) == ANIMALS
    assert "FERTILIZER" in kaggle_engine.PRODUCTS
    # Fertilizer (one end of day after placement) is every animal's first product.
    assert all(
        data["first_yield_day"] >= FERTILIZER_READY_DAYS
        for data in kaggle_engine.ANIMALS.values()
    )
    spec = kaggle_engine.specification["configuration"]
    assert (spec["episodeSteps"], spec["turnsPerDay"]["default"]) == (720, 24)


@pytest.mark.parametrize(("episode_steps", "turns_per_day"), [(720, 24), (30, 4)])
def test_last_processed_action_step_is_episode_steps_minus_two(
    episode_steps: int, turns_per_day: int
) -> None:
    configuration = {"episodeSteps": episode_steps, "turnsPerDay": turns_per_day}
    seen: list[int] = []

    def agent(obs: Any, _: Any) -> dict[str, JsonValue]:
        seen.append(obs["step"])
        return PASS

    env = make("kaggriculture", configuration=configuration | {"seed": 3})
    env.run([agent, "pass"])
    assert env.steps[-1][0]["status"] == "DONE"
    assert max(seen) == GameClock.of(configuration).last_action_step
    assert seen == list(range(episode_steps - 1))


def test_clock_and_step_reads_are_strict() -> None:
    with pytest.raises(KeyError):
        GameClock.of({"turnsPerDay": 24})
    with pytest.raises(ValueError, match="turnsPerDay"):
        GameClock.of({"episodeSteps": 720, "turnsPerDay": 0})
    with pytest.raises(ValueError, match="episodeSteps"):
        GameClock.of({"episodeSteps": 720.0, "turnsPerDay": 24})
    assert observation_step({"step": 697, "day": 29, "hour": 1}, CLOCK) == 697
    with pytest.raises(ValueError, match="disagree"):
        observation_step({"step": 697, "day": 29, "hour": 2}, CLOCK)
    with pytest.raises(ValueError, match="non-negative"):
        observation_step({"step": -1, "day": 0, "hour": 0}, CLOCK)


# --- thresholds at the default configuration -------------------------------


@pytest.mark.parametrize(("key", "first"), sorted(FIRST_BLOCKED.items()))
def test_each_purchase_is_blocked_exactly_from_its_first_impossible_step(
    key: tuple[str, str | None], first: int
) -> None:
    order = _order(*key)
    blocked = [step for step in range(LAST + 1) if _blocked(order, step)]
    assert blocked == list(range(first, LAST + 1))


def test_hires_are_blocked_only_at_day_end_and_in_the_last_two_steps() -> None:
    blocked = [step for step in range(LAST + 1) if _blocked(["HIRE"], step)]
    expected = sorted({s for s in range(LAST + 1) if s % 24 == 23} | {717, 718})
    assert blocked == expected
    assert not _blocked(["HIRE"], 716)
    assert not _blocked(["HIRE"], 262)  # hour 22 of day 10


@pytest.mark.parametrize(
    "order",
    [
        ["SELL", "WHEAT", 5],
        ["SELL", "FERTILIZER", 1],
        ["BUY_PRODUCT", "WHEAT", 3],  # animal feed
        ["BUY_PRODUCT", "FERTILIZER", 2],  # input for existing plants
        [],
        ["BUY_SEED", "UNKNOWN", 1],  # engine no-op, left alone
        ["BUY_ANIMAL", "DOG", 1],
    ],
)
def test_upkeep_sales_and_engine_no_ops_are_never_blocked(
    order: list[JsonValue],
) -> None:
    assert not any(_blocked(order, step) for step in range(LAST + 1))


def test_unit_actions_are_never_touched() -> None:
    action: dict[str, JsonValue] = {
        "farmer": ["FEED"],
        "hands": [["WATER"], ["PLANT", "WHEAT"], ["HARVEST"]],
        "market": [["BUY_PRODUCT", "WHEAT", 2], ["SELL", "EGG", 1]],
    }
    for step in range(LAST + 1):
        filtered, reasons = filter_late_investments(action, step, CLOCK)
        assert filtered is action
        assert reasons == ()


def test_filter_replaces_blocked_orders_in_place_and_stays_legal() -> None:
    action: dict[str, JsonValue] = {
        "farmer": ["HARVEST"],
        "hands": [["FEED"]],
        "market": [
            ["SELL", "WHEAT", 3],
            ["BUY_SEED", "MELON", 2],
            ["HIRE"],
            ["BUY_PRODUCT", "WHEAT", 4],
            ["BUY_ANIMAL", "COW", 1],
            ["BUY_LAND"],
            ["BUY_SEED", "WHEAT", 1],
        ],
    }
    snapshot = json.dumps(action)
    filtered, reasons = filter_late_investments(action, 694, CLOCK)
    assert json.dumps(action) == snapshot  # the input is not mutated
    assert filtered == {
        "farmer": ["HARVEST"],
        "hands": [["FEED"]],
        "market": [
            ["SELL", "WHEAT", 3],
            [],
            ["HIRE"],  # hour 22 of day 28: the hand can still harvest and sell
            ["BUY_PRODUCT", "WHEAT", 4],
            [],
            ["BUY_LAND"],  # land bought at 694 can still host an animal
            [],
        ],
    }
    assert len(reasons) == 3
    _validate(action, actors=2)
    _validate(filtered, actors=2)
    last, _ = filter_late_investments(action, LAST, CLOCK)
    assert last["market"] == [
        ["SELL", "WHEAT", 3],
        [],
        [],
        ["BUY_PRODUCT", "WHEAT", 4],
        [],
        [],
        [],
    ]
    _validate(last, actors=2)


def test_nothing_is_blocked_early_and_the_input_is_returned_as_is() -> None:
    action: dict[str, JsonValue] = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["BUY_SEED", "STRAWBERRY", 1],
            ["BUY_LAND"],
            ["BUY_ANIMAL", "SHEEP", 1],
        ],
    }
    for step in range(478):
        filtered, reasons = filter_late_investments(action, step, CLOCK)
        assert filtered is action
        assert reasons == ()


# --- scripted plays in the Kaggle engine: the last allowed step still sells --

Script = dict[int, dict[str, JsonValue]]


# 699 steps puts the last action at 697 = day 29 hour 1: the earliest sale the
# seed and animal plays reach, so there they meet the last action exactly.
TIGHT_EPISODE_STEPS = 699


def _play(script: Script, episode_steps: int = 720) -> Any:
    """Seat 0 follows ``script`` (PASS elsewhere); seat 1 passes; no weeds."""

    def agent(obs: Any, _: Any) -> dict[str, JsonValue]:
        return script.get(obs["step"], PASS)

    env = make(
        "kaggriculture",
        configuration={
            "seed": 11,
            "weedSpawnChance": 0,
            "episodeSteps": episode_steps,
        },
        debug=True,
    )
    env.run([agent, "pass"])
    assert len(env.steps) == episode_steps
    return env


def _held(observation: dict[str, Any], item: str) -> int:
    private = observation["private"]
    carried = sum(inventory.get(item, 0) for inventory in private["inventories"])
    return int(private["shed"][item]) + carried


def _sold(env: Any, step: int, item: str) -> float:
    """Cash seat 0 gained at ``step`` while its held ``item`` count fell.

    The scripted step's only market order is that SELL, so a gain is a sale.
    """
    before = env.steps[step][0]["observation"]
    after = env.steps[step + 1][0]["observation"]
    assert before["step"] == step
    assert _held(after, item) < _held(before, item)
    return float(after["farms"][0]["money"] - before["farms"][0]["money"])


def _act(
    farmer: list[JsonValue],
    hands: list[JsonValue] | None = None,
    market: list[JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {"farmer": farmer, "hands": hands or [], "market": market or []}


def _last_allowed(order: list[JsonValue], clock: GameClock) -> int:
    allowed = [
        s for s in range(clock.last_action_step + 1) if not _blocked_at(order, s, clock)
    ]
    assert allowed == list(range(allowed[-1] + 1))
    return allowed[-1]


def _blocked_at(order: list[JsonValue], step: int, clock: GameClock) -> bool:
    return blocked_reason(order, step, clock) is not None


@pytest.mark.parametrize("episode_steps", [720, TIGHT_EPISODE_STEPS])
@pytest.mark.parametrize("crop", sorted(CROP_FIRST_YIELD_DAY))
def test_last_allowed_seed_purchase_is_harvested_and_sold(
    crop: str, episode_steps: int
) -> None:
    clock = GameClock.of({"episodeSteps": episode_steps, "turnsPerDay": 24})
    bought = _last_allowed(["BUY_SEED", crop, 1], clock)
    assert bought == FIRST_BLOCKED[("BUY_SEED", crop)] - 1
    planted = bought + 1  # hour 23: a hand waters in the same step
    assert planted % 24 == 23
    harvest_day = planted // 24 + CROP_FIRST_YIELD_DAY[crop]
    harvest = harvest_day * 24
    script: Script = {
        bought - 2: _act(["PASS"], market=[["HIRE"]]),  # hand spawns at (5, 4)
        bought - 1: _act(["PASS"], [["WEST"]]),
        bought: _act(["PASS"], [["PASS"]], [["BUY_SEED", crop, 1]]),
        planted: _act(["PLANT", crop], [["WATER"]]),
        harvest: _act(["HARVEST"]),
        harvest + 1: _act(["DROP"], market=[["SELL", crop, 1]]),
    }
    for day in range(planted // 24 + 1, harvest_day):
        script[day * 24] = _act(["WATER"])
    assert harvest + 1 <= clock.last_action_step
    env = _play(script, episode_steps)
    assert _sold(env, harvest + 1, crop) > 0


@pytest.mark.parametrize("episode_steps", [720, TIGHT_EPISODE_STEPS])
@pytest.mark.parametrize("animal", sorted(ANIMALS))
def test_last_allowed_animal_purchase_yields_sold_fertilizer(
    animal: str, episode_steps: int
) -> None:
    clock = GameClock.of({"episodeSteps": episode_steps, "turnsPerDay": 24})
    bought = _last_allowed(["BUY_ANIMAL", animal, 1], clock)
    assert bought == FIRST_BLOCKED[("BUY_ANIMAL", animal)] - 1
    structure = kaggle_engine.ANIMALS[animal]["structure"]
    collect = (bought + 2) // 24 * 24 + 24
    script: Script = {
        0: _act([f"BUILD_{structure}"]),  # on the shed-access tile (4, 4)
        bought: _act(["PASS"], market=[["BUY_ANIMAL", animal, 1]]),
        bought + 1: _act(["PICKUP", animal, 1]),
        bought + 2: _act(["PLACE", animal]),
        collect: _act(["COLLECT_FERTILIZER"]),
        collect + 1: _act(["DROP"], market=[["SELL", "FERTILIZER", 1]]),
    }
    assert collect + 1 <= clock.last_action_step
    env = _play(script, episode_steps)
    assert _sold(env, collect + 1, "FERTILIZER") > 0


def test_last_allowed_land_purchase_hosts_an_animal_whose_fertilizer_sells() -> None:
    bought = FIRST_BLOCKED[("BUY_LAND", None)] - 1
    assert not _blocked(["BUY_LAND"], bought)
    day_start = bought // 24 * 24
    collect_day = (bought + 1) // 24 + 1
    script: Script = {
        day_start + 1: _act(["PASS"], market=[["BUY_ANIMAL", "GOOSE", 1]]),
        day_start + 2: _act(["PASS"], market=[["HIRE"]]),  # hand spawns at (5, 4)
        day_start + 3: _act(["EAST"], [["PICKUP", "GOOSE", 1]]),
        bought: _act(["PASS"], [["PASS"]], [["BUY_LAND"]]),  # unlocks NE
        bought + 1: _act(["BUILD_COOP"], [["PLACE", "GOOSE"]]),  # on (5, 4)
        collect_day * 24: _act(["EAST"]),
        collect_day * 24 + 1: _act(["COLLECT_FERTILIZER"]),
        collect_day * 24 + 2: _act(["DROP"], market=[["SELL", "FERTILIZER", 1]]),
    }
    env = _play(script)
    tiles = env.steps[bought + 2][0]["observation"]["farms"][0]["tiles"]
    assert tiles[4][5]["animal"] == "GOOSE"
    assert _sold(env, collect_day * 24 + 2, "FERTILIZER") > 0


def _wheat_ready_by(day: int) -> Script:
    """Wheat on (4, 4), planted at the start of ``day - 2`` and kept watered."""
    planted = (day - 2) * 24
    return {
        planted - 1: _act(["PASS"], market=[["BUY_SEED", "WHEAT", 1]]),
        planted: _act(["PLANT", "WHEAT"]),
        planted + 1: _act(["WATER"]),
        planted + 24: _act(["WATER"]),
    }


@pytest.mark.parametrize("hired", [716, 262])
def test_last_allowed_hire_harvests_and_sells(hired: int) -> None:
    assert not _blocked(["HIRE"], hired)
    script = _wheat_ready_by(hired // 24)
    # The farmer leaves (4, 4), so the hand spawns there, on the ripe wheat.
    script[hired] = _act(["NORTH"], market=[["HIRE"]])
    script[hired + 1] = _act(["PASS"], [["HARVEST"]])
    sell = hired + 2
    if hired % 24 == 22:  # the day-end drop moves the hand's wheat to the shed
        script[sell] = _act(["PASS"], market=[["SELL", "WHEAT", 1]])
    else:
        script[sell] = _act(["PASS"], [["DROP"]], [["SELL", "WHEAT", 1]])
    env = _play(script)
    assert _sold(env, sell, "WHEAT") > 0


def test_a_day_end_hire_never_acts() -> None:
    hired = 10 * 24 + 23
    assert _blocked(["HIRE"], hired)
    env = _play({hired: _act(["PASS"], market=[["HIRE"]])})
    before = env.steps[hired][0]["observation"]["farms"][0]
    after = env.steps[hired + 1][0]["observation"]["farms"][0]
    assert after["money"] == before["money"] - 1
    assert after["hands"] == []
    assert after["hires_today"] == 0


# --- the packaged agent seam ------------------------------------------------


def _late_observations(steps: list[int]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    env = make("kaggriculture", configuration={"seed": 5}, debug=True)
    env.reset()
    wanted = set(steps)
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while len(result) < len(steps):
        observation = json.loads(json.dumps(env.state[0].observation))
        if observation["step"] in wanted:
            result.append((observation, dict(env.configuration)))
        env.step([PASS, PASS])
    return result


def _agent(root: Path, **kwargs: Any) -> KaggricultureAgent:
    return KaggricultureAgent(
        root, deterministic=True, strict=True, min_overage_time=0.0, **kwargs
    )


LATE_PROGRAM: dict[str, JsonValue] = {
    "farmer": ["PASS"],
    "hands": [],
    "market": [
        ["BUY_SEED", "WHEAT", 2],
        ["SELL", "WHEAT", 1],
        ["HIRE"],
        ["BUY_PRODUCT", "WHEAT", 1],
        ["BUY_ANIMAL", "GOOSE", 1],
        ["BUY_LAND"],
    ],
}


class _InjectFirstDecode:
    """The agent's first decode returns ``program``; later ones stay native."""

    def __init__(self, program: dict[str, JsonValue]) -> None:
        self.program = program
        self.pending = False
        self.real: Callable[..., dict[str, JsonValue]] = decode_action

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, JsonValue]:
        if self.pending:
            self.pending = False
            return json.loads(json.dumps(self.program))
        return self.real(*args, **kwargs)


def test_agent_default_off_returns_the_decoded_program_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_model_root(tmp_path / "primary")
    inject = _InjectFirstDecode(LATE_PROGRAM)
    monkeypatch.setattr(kaggle_agent, "decode_action", inject)
    agent = _agent(root)
    assert agent.block_late_investments is False
    for observation, configuration in _late_observations([0, 671, 695, 717, 718]):
        inject.pending = True
        action = agent.act(observation, configuration)
        assert json.dumps(action) == json.dumps(LATE_PROGRAM)
    assert agent.blocked_orders == 0


def test_agent_filter_blocks_only_unproductive_purchases_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = write_model_root(tmp_path / "primary")
    inject = _InjectFirstDecode(LATE_PROGRAM)
    monkeypatch.setattr(kaggle_agent, "decode_action", inject)
    agent = _agent(root, block_late_investments=True)
    by_step: dict[int, JsonValue] = {}
    for observation, configuration in _late_observations([0, 670, 671, 695, 718]):
        inject.pending = True
        action = agent.act(observation, configuration)
        _validate(action)
        by_step[observation["step"]] = action["market"]
    keep = LATE_PROGRAM["market"]
    assert isinstance(keep, list)
    assert by_step[0] == keep
    assert by_step[670] == keep
    # 671 is hour 23 of day 27: the wheat seed and the day-end hire drop.
    assert by_step[671] == [[], keep[1], [], keep[3], keep[4], keep[5]]
    assert by_step[695] == [[], keep[1], [], keep[3], [], []]  # hour 23 hire
    assert by_step[718] == [[], keep[1], [], keep[3], [], []]
    assert agent.blocked_orders == 2 + 4 + 4
    assert capsys.readouterr().out.count("late-invest blocked") == 3


def test_agent_real_model_outputs_are_unchanged_when_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_model_root(tmp_path / "primary")
    decoded: list[dict[str, JsonValue]] = []
    real = decode_action

    def recording(*args: Any, **kwargs: Any) -> dict[str, JsonValue]:
        result = real(*args, **kwargs)
        decoded.append(json.loads(json.dumps(result)))
        return result

    monkeypatch.setattr(kaggle_agent, "decode_action", recording)
    off = _agent(root)
    on = _agent(root, block_late_investments=True)
    for observation, configuration in _late_observations([0, 300, 694, 718]):
        decoded.clear()
        action = off.act(observation, configuration)
        assert json.dumps(action) == json.dumps(decoded[0])
        decoded.clear()
        filtered = on.act(observation, configuration)
        expected, _ = filter_late_investments(
            decoded[0], observation["step"], GameClock.of(configuration)
        )
        assert filtered == expected


def test_switch_defaults_off_and_rejects_unknown_values() -> None:
    assert late_invest.enabled_from_env({}) is False
    assert late_invest.enabled_from_env({late_invest.ENV_VAR: "0"}) is False
    assert late_invest.enabled_from_env({late_invest.ENV_VAR: "1"}) is True
    for bad in ("", "true", "2", " 1"):
        with pytest.raises(ValueError, match=late_invest.ENV_VAR):
            late_invest.enabled_from_env({late_invest.ENV_VAR: bad})


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("0", False), ("1", True)]
)
def test_packaged_main_switch_is_off_unless_the_env_var_is_one(
    tmp_path: Path, value: str | None, expected: bool
) -> None:
    agent_dir = write_agent_dir(tmp_path / "agent")
    environment = {
        key: item
        for key, item in os.environ.items()
        if key != "KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS"
    } | {"KAGGRICULTURE_AGENT_ALLOW_DEBUG_BUILD": "1"}
    if value is not None:
        environment["KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS"] = value
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.path.insert(0, sys.argv[1]); "
            "g = runpy.run_path(sys.argv[1] + '/main.py'); "
            "print(g['AGENT'].block_late_investments)",
            str(agent_dir),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().splitlines()[-1] == str(expected)
