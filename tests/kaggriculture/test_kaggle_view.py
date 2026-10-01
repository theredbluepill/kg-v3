"""Task 7.4: one live seat's Kaggle observation through the training write_seat."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest
import torch
from kaggle_environments import make
from kaggle_environments.utils import structify
from owl.kaggriculture.codec import encode_action_into
from owl.kaggriculture.env import (
    allocate_observation_buffers,
    allocate_single_seat_buffers,
)
from owl.kaggriculture.kaggle_view import (
    GAME_CONFIGURATION_KEYS,
    SeatArrays,
    encode_seat_into,
    game_configuration,
    seat_view_json,
)
from owl.kaggriculture.types import _SCHEMA, KaggricultureObsBatch

from tests.kaggriculture.test_native_env import buffers, make_env

FIELDS = tuple(SeatArrays.__dataclass_fields__)
MOVES = ("NORTH", "SOUTH", "EAST", "WEST", "PASS")


def _scripted(step: int, seat: int, hands: int) -> dict[str, Any]:
    market: list[list[Any]] = []
    if (step + seat) % 6 == 0:
        market = [["HIRE"]]
    elif (step + seat) % 6 == 1:
        market = [["BUY_SEED", "WHEAT", 2]]
    return {
        "farmer": [MOVES[(step + seat) % 5]],
        "hands": [[MOVES[(step + index) % 5]] for index in range(hands)],
        "market": market,
    }


def _kaggle_calls(
    seed: int, episode_steps: int
) -> dict[int, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    """Each seat's (observation, configuration, action) from Kaggle's own calls."""
    calls: dict[int, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = {
        0: [],
        1: [],
    }

    def agent_for(seat: int) -> Any:
        def agent(obs: Any, config: Any) -> dict[str, Any]:
            observation = json.loads(json.dumps(obs))
            action = _scripted(
                observation["step"], seat, len(observation["farms"][seat]["hands"])
            )
            calls[seat].append((observation, json.loads(json.dumps(config)), action))
            return action

        return agent

    env = make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": episode_steps},
        debug=True,
    )
    env.run([agent_for(0), agent_for(1)])
    return calls


def _seat_bytes(arrays: SeatArrays) -> dict[str, bytes]:
    return {name: getattr(arrays, name)[0, 0].tobytes() for name in FIELDS}


@pytest.mark.parametrize("seed", [7, 20260930])
def test_kaggle_seat_rows_equal_native_rows_after_the_same_actions(seed: int) -> None:
    """Kaggle's per-seat observation, encoded alone, equals the native env's row.

    Both engines start from the same seed and take the same actions; every
    acted-on state is compared for both seats over all 29 fields.
    """
    steps = 61
    calls = _kaggle_calls(seed, steps)
    assert len(calls[0]) == len(calls[1]) == steps - 1
    native = make_env(seed=seed, stride=1, config=json.dumps({"episodeSteps": steps}))
    arrays = buffers(1)
    native.observe(**arrays)
    seat_arrays = SeatArrays.of(allocate_single_seat_buffers())
    most_hands = 0
    for step in range(steps - 1):
        tokens = np.zeros((1, 2, 252, 12), dtype=np.int64)
        lengths = np.zeros((1, 2), dtype=np.int64)
        for seat in (0, 1):
            observation, configuration, action = calls[seat][step]
            assert observation["step"] == step
            view, view_seat, dropped = seat_view_json(observation, configuration)
            assert (view_seat, dropped) == (seat, ())
            encode_seat_into(view, view_seat, seat_arrays)
            # Iterating the pinned field names is the sanctioned dynamic access.
            actual = _seat_bytes(seat_arrays)
            for name in FIELDS:
                assert actual[name] == arrays[name][0, seat].tobytes(), (
                    f"seed {seed} step {step} seat {seat} field {name}"
                )
            most_hands = max(most_hands, len(observation["farms"][seat]["hands"]))
            lengths[0, seat] = encode_action_into(
                action,
                actors=int(arrays["actor_mask"][0, seat, :241].sum()),
                order_limit=int(arrays["order_limits"][0, seat]),
                hire_limit=241,
                out=tokens[0, seat],
            )
        native.step(tokens, lengths, **arrays)
    assert most_hands >= 2


def test_struct_and_json_round_tripped_observations_encode_identically() -> None:
    observation, configuration, _ = _kaggle_calls(3, 30)[1][-1]
    seat_arrays = SeatArrays.of(allocate_single_seat_buffers())
    view, seat, _ = seat_view_json(observation, configuration)
    encode_seat_into(view, seat, seat_arrays)
    plain = _seat_bytes(seat_arrays)
    view, seat, _ = seat_view_json(structify(observation), structify(configuration))
    encode_seat_into(view, seat, seat_arrays)
    assert _seat_bytes(seat_arrays) == plain


def test_configuration_allowlist_drops_loader_keys_and_keeps_order() -> None:
    configuration = {
        "__raw_path__": "/kaggle_simulations/agent/main.py",
        "episodeSteps": 720,
        "actTimeout": 1,
        "seed": None,
        "maxLogLength": 10000,
        "marketParams": {},
    }
    kept, dropped = game_configuration(configuration)
    assert list(kept) == ["episodeSteps", "actTimeout", "seed", "marketParams"]
    assert dropped == ("__raw_path__", "maxLogLength")
    assert {"boardSize", "runTimeout", "weedSpawnChance"} <= GAME_CONFIGURATION_KEYS


class _RecordingMapping(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.read: list[str] = []

    def __getitem__(self, key: str) -> Any:
        self.read.append(key)
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def test_view_reads_only_the_public_state_own_private_and_seat() -> None:
    observation, configuration, _ = _kaggle_calls(3, 5)[0][-1]
    recording = _RecordingMapping(observation | {"rival_private": {"shed": {}}})
    view, seat, _ = seat_view_json(recording, configuration)
    assert sorted(set(recording.read)) == sorted(
        ["player", "step", "day", "hour", "farms", "market", "town", "private"]
    )
    parsed = json.loads(view)
    assert list(parsed) == ["configuration", "public", "private"]
    assert "remainingOverageTime" not in view
    assert "player" not in view
    # Insertion order survives (inventory and shed ranks depend on it).
    assert list(parsed["private"]["shed"]) == list(observation["private"]["shed"])
    assert seat == 0


def test_invalid_player_and_rejected_views_leave_buffers_unchanged() -> None:
    observation, configuration, _ = _kaggle_calls(3, 5)[0][-1]
    with pytest.raises(ValueError, match="player must be 0 or 1"):
        seat_view_json(observation | {"player": 2}, configuration)
    seat_arrays = SeatArrays.of(allocate_single_seat_buffers())
    for name in FIELDS:
        getattr(seat_arrays, name)[...] = 1
    before = _seat_bytes(seat_arrays)
    view, seat, _ = seat_view_json(observation, configuration)
    bad = json.loads(view)
    bad["private"]["inventories"].append({})
    with pytest.raises(ValueError, match="actor_inventory"):
        encode_seat_into(json.dumps(bad), seat, seat_arrays)
    bad = json.loads(view)
    bad["configuration"]["__raw_path__"] = "main.py"
    with pytest.raises(ValueError, match="__raw_path__"):
        encode_seat_into(json.dumps(bad), seat, seat_arrays)
    with pytest.raises(ValueError, match="seat"):
        encode_seat_into(view, 2, seat_arrays)
    assert _seat_bytes(seat_arrays) == before


def _filled(lead: tuple[int, int]) -> KaggricultureObsBatch:
    obs = (
        allocate_single_seat_buffers()
        if lead == (1, 1)
        else allocate_observation_buffers(lead[0], pin_memory=False)
    )
    for name in _SCHEMA:
        getattr(obs, name).zero_()
    obs.action_mask.can_act.zero_()
    return obs


def _reshaped(
    obs: KaggricultureObsBatch, lead: tuple[int, int]
) -> KaggricultureObsBatch:
    def reshape(tensor: torch.Tensor) -> torch.Tensor:
        trailing = tensor.shape[2:]
        return torch.zeros((*lead, *trailing), dtype=tensor.dtype)

    fields = {name: reshape(getattr(obs, name)) for name in _SCHEMA}
    return KaggricultureObsBatch(
        **fields,
        action_mask=type(obs.action_mask)(can_act=reshape(obs.action_mask.can_act)),
    )


def test_single_seat_contract_is_scoped_to_one_row() -> None:
    single = _filled((1, 1))
    single.check_single_seat_contract()
    with pytest.raises(ValueError, match=r"\[env, 2\]"):
        single.check_contract()
    for lead in ((1, 2), (2, 1), (2, 2)):
        with pytest.raises(ValueError, match=r"\(1, 1\)"):
            _reshaped(single, lead).check_single_seat_contract()
    pair = _filled((1, 2))
    pair.check_contract()
    for obs in (single, pair):
        obs.tile_kind[..., 0] = 99
    for check in (single.check_single_seat_contract, pair.check_contract):
        with pytest.raises(ValueError, match="tile_kind values must be < 6"):
            check()
