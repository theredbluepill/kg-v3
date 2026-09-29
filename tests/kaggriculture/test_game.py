"""Stage 1 game envelope and constructor seam, independent of the live binding."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from owl import rs
from owl.kaggriculture import types as kt
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.rl import EnvConfig, VectorizedEnv
from owl.train.config import GameEnvConfig
from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from owl.rs import KaggricultureRewardDict

_REWARD = {
    "econ_shaping": 0.2,
    "econ_starvation_weight": 4.0,
    "econ_drought_weight": 1.0,
    "econ_cap": 0.25,
    "econ_ineffective_weight": 0.0,
    "econ_ineffective_cap": 0.1,
}
_I64_MAX = 2**63 - 1
_DEFAULT_GAME = {
    "boardSize": 10,
    "episodeSteps": 720,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}


def _env_data() -> dict[str, Any]:
    return {
        "obs_spec": {"obs_spec": "kaggriculture"},
        "reward_shaping": _REWARD,
        "native_threads": 1,
        "pin_memory": False,
    }


def test_game_envelope() -> None:
    config = kt.KaggricultureGameConfig()
    assert json.loads(config.to_native_json()) == _DEFAULT_GAME
    assert config.board_size == 10
    assert config.episode_steps == 720
    assert kt.KaggricultureGameConfig(episode_steps=3).episode_steps == 3
    accepted: list[dict[str, Any]] = [
        {"maxMarketOrdersPerTurn": 1, "turnsPerDay": 240},
        {"maxMarketOrdersPerTurn": 10, "turnsPerDay": 24},
        {"weedSpawnChance": 2.0},
        {"episodeSteps": 1, "startingMoney": 0, "farmHandCostMult": 0},
        {"marketParams": {}},
    ]
    for patch in accepted:
        assert (
            json.loads(
                kt.KaggricultureGameConfig.model_validate(patch).to_native_json()
            )
            == _DEFAULT_GAME | patch
        )
    for field in (
        "episodeSteps",
        "startingMoney",
        "shedCapacity",
        "townShopUnlockInterval",
        "townShopSellInterval",
        "townCenterSellInterval",
        "farmHandCostMult",
    ):
        result = kt.KaggricultureGameConfig.model_validate({field: _I64_MAX})
        assert json.loads(result.to_native_json())[field] == _I64_MAX
        with pytest.raises(ValidationError):
            kt.KaggricultureGameConfig.model_validate({field: _I64_MAX + 1})
    rejected: list[dict[str, Any]] = [
        {"boardSize": 9},
        {"maxMarketOrdersPerTurn": 0},
        {"maxMarketOrdersPerTurn": 11},
        {"turnsPerDay": 25},
        {"turnsPerDay": 0},
        {"episodeSteps": 0},
        {"startingMoney": -1},
        {"shedCapacity": 0},
        {"townShopUnlockInterval": 0},
        {"townShopSellInterval": 0},
        {"townCenterSellInterval": 0},
        {"farmHandCostMult": -1},
        {"marketParams": {"WHEAT": 1}},
        {"extra": {}},
        {"unknown": 1},
        {"weedSpawnChance": -0.01},
        {"weedSpawnChance": float("inf")},
        {"weedSpawnChance": float("nan")},
        {"weedSpawnChance": True},
        {"weedSpawnChance": "0.1"},
    ]
    for patch in rejected:
        with pytest.raises(ValidationError):
            kt.KaggricultureGameConfig.model_validate(patch)
    integer_keys = tuple(
        key for key, value in _DEFAULT_GAME.items() if type(value) is int
    )
    for field in integer_keys:
        for value in (True, False, "10", 1.5, float("inf"), float("nan")):
            with pytest.raises(ValidationError):
                kt.KaggricultureGameConfig.model_validate({field: value})
    integral = {
        key: float(value) for key, value in _DEFAULT_GAME.items() if type(value) is int
    }
    canonical = json.loads(
        kt.KaggricultureGameConfig.model_validate(integral).to_native_json()
    )
    assert canonical == _DEFAULT_GAME
    assert all(type(canonical[key]) is int for key in integral)
    with pytest.raises(ValidationError):
        kt.KaggricultureGameConfig.model_validate({"episodeSteps": float(_I64_MAX)})
    framework = {"actTimeout": 3.0, "runTimeout": 1200, "seed": "framework-only"}
    enriched = kt.KaggricultureGameConfig.model_validate(framework)
    assert json.loads(enriched.to_native_json()) == _DEFAULT_GAME | framework
    assert (
        json.loads(
            kt.KaggricultureGameConfig.model_validate({"seed": None}).to_native_json()
        )
        == _DEFAULT_GAME
    )


def test_integral_float_envelope_matches_real_header_encoder() -> None:
    from .test_observe import _allocate, _arrays, _header, _write

    header = _header()
    integer_config = header["configuration"]
    float_config = {
        key: float(value) if type(value) is int else value
        for key, value in integer_config.items()
    }
    header["configuration"] = float_config
    arrays = _arrays(_allocate(1, pin_memory=False))
    _write(json.dumps([header]), arrays)
    before = {key: value.copy() for key, value in arrays.items()}
    config = kt.KaggricultureGameConfig.model_validate(float_config)
    header["configuration"] = json.loads(config.to_native_json())
    _write(json.dumps([header]), arrays)
    for key, values in before.items():
        assert (arrays[key] == values).all(), key


def test_hire_limit_has_one_owner() -> None:
    cfg = KaggricultureEnvConfig.model_validate(_env_data() | {"n_envs": 1})
    assert cfg.n_envs == 1
    assert cfg.seed == 0
    assert isinstance(cfg.config, kt.KaggricultureGameConfig)
    assert cfg.action_spec.hire_limit == 241
    assert "hire_limit" not in KaggricultureEnvConfig.model_fields
    assert "game" not in KaggricultureEnvConfig.model_fields
    with pytest.raises(ValidationError, match="hire_limit"):
        KaggricultureEnvConfig.model_validate(_env_data() | {"hire_limit": 17})
    for field in ("n_envs", "seed", "native_threads"):
        for value in (True, False, 1.0, "1"):
            with pytest.raises(ValidationError, match=field):
                KaggricultureEnvConfig.model_validate(_env_data() | {field: value})
    for patch in (
        {"n_envs": 0},
        {"native_threads": 0},
        {"seed": -1},
        {"seed": _I64_MAX + 1},
    ):
        with pytest.raises(ValidationError):
            KaggricultureEnvConfig.model_validate(_env_data() | patch)
    assert (
        KaggricultureEnvConfig.model_validate(_env_data() | {"seed": _I64_MAX}).seed
        == _I64_MAX
    )
    for field in ("reward_shaping", "native_threads"):
        data = _env_data()
        del data[field]
        with pytest.raises(ValidationError, match=field):
            KaggricultureEnvConfig.model_validate(data)


def test_config_union_preserves_obs_tag_discriminator() -> None:
    adapter: TypeAdapter[EnvConfig | KaggricultureEnvConfig] = TypeAdapter(
        GameEnvConfig
    )
    assert isinstance(adapter.validate_python({}), EnvConfig)
    assert isinstance(adapter.validate_python(_env_data()), KaggricultureEnvConfig)
    rejected = [
        {"native_threads": 1},
        {"seed": 0},
        {"config": {}},
        {"reward_shaping": _REWARD},
        {"obs_spec": {"obs_spec": "unknown"}},
        _env_data() | {"two_player_weight": 0.5},
        _env_data() | {"action_spec": {"action_spec": "pure"}},
        {"action_spec": {"action_spec": "kaggriculture"}},
        {"obs_spec": {"obs_spec": "entity_based"}, "reward_shaping": _REWARD},
    ]
    for data in rejected:
        with pytest.raises(ValidationError):
            adapter.validate_python(data)


def test_orbit_factory_is_original_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    import owl.game as game

    calls: list[dict[str, Any]] = []

    def init(_self: VectorizedEnv, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(VectorizedEnv, "__init__", init)
    cfg = EnvConfig(n_envs=2, two_player_weight=0.7, reward_mode="win_only")
    env = game.create_env(
        cfg,
        n_envs=4,
        base_seed=11,
        rank=1,
        world_size=2,
        pin_memory=False,
        transfer_device=torch.device("cpu"),
    )
    assert type(env) is VectorizedEnv
    assert calls == [
        {
            "n_envs": 4,
            "obs_spec": cfg.obs_spec,
            "action_spec": cfg.action_spec,
            "two_player_weight": 0.7,
            "reward_mode": "win_only",
            "pin_memory": False,
        }
    ]
    assert "game" not in EnvConfig.model_fields


def test_factory_rank_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    from owl.game import create_env
    from owl.kaggriculture.env import KaggricultureVectorizedEnv

    from .fake_env import FakeKaggricultureEnv

    constructions: list[FakeKaggricultureEnv] = []

    class RecordingNative(FakeKaggricultureEnv):
        def __init__(
            self,
            n_envs: int,
            seed: int,
            seed_stride: int,
            config: str,
            reward_config: KaggricultureRewardDict,
            native_threads: int,
            *,
            hire_limit: int,
        ) -> None:
            super().__init__(
                n_envs,
                seed,
                seed_stride,
                config,
                reward_config,
                native_threads,
                hire_limit=hire_limit,
            )
            constructions.append(self)

    monkeypatch.setattr(rs, "KaggricultureEnv", RecordingNative, raising=False)
    cfg = KaggricultureEnvConfig.model_validate(
        _env_data() | {"seed": 999, "action_spec": {"hire_limit": 17}}
    )
    streams = []
    for rank in range(2):
        env = create_env(
            cfg,
            n_envs=2,
            base_seed=11,
            rank=rank,
            world_size=2,
            pin_memory=False,
            transfer_device=torch.device("cpu"),
        )
        assert isinstance(env, KaggricultureVectorizedEnv)
        native = constructions[-1]
        assert (native.n_envs, native.seed, native.seed_stride) == (2, 11 + rank, 2)
        assert native.config == cfg.config.to_native_json()
        assert native.reward_config == _REWARD | {"reward_mode": "win_loss"}
        assert native.native_threads == 1
        assert native.hire_limit == 17
        assert native.calls == ["observe"]
        assert env.seed_state() == (15 + rank, (11 + rank, 13 + rank))
        assert env.action_spec.hire_limit == 17
        assert env.reward_mode == cfg.reward_mode
        seeds = set(env.seed_state()[1])
        for _ in range(31):
            env.reset()
            seeds.update(env.seed_state()[1])
        assert seeds == {11 + rank + k * 2 for k in range(64)}
        streams.append(seeds)
    assert streams[0].isdisjoint(streams[1])
    old_streams = [{11 + rank * 2 + k for k in range(64)} for rank in range(2)]
    assert old_streams[0] & old_streams[1]


@pytest.mark.parametrize(
    "patch",
    [
        {"n_envs": 0},
        {"n_envs": True},
        {"n_envs": "2"},
        {"n_envs": 2.0},
        {"rank": -1},
        {"rank": 2},
        {"rank": True},
        {"rank": 0.0},
        {"rank": "0"},
        {"world_size": 0},
        {"world_size": True},
        {"world_size": 1.0},
        {"world_size": "2"},
        {"world_size": _I64_MAX + 1},
        {"base_seed": -1},
        {"base_seed": True},
        {"base_seed": "0"},
        {"base_seed": 0.0},
        {"base_seed": _I64_MAX + 1},
        {"base_seed": _I64_MAX, "rank": 1},
    ],
)
def test_factory_invalid_rank_streams(patch: dict[str, Any]) -> None:
    from owl.game import create_env

    cfg = KaggricultureEnvConfig.model_validate(_env_data())
    kwargs = {
        "n_envs": 2,
        "base_seed": 11,
        "rank": 0,
        "world_size": 2,
        "pin_memory": False,
        "transfer_device": torch.device("cpu"),
    }
    with pytest.raises((TypeError, ValueError)):
        create_env(cfg, **(kwargs | patch))


@pytest.mark.skip(reason="needs Task 1.4 binding")
def test_real_binding_seed_streams() -> None:
    from owl.game import create_env
    from owl.kaggriculture.env import KaggricultureVectorizedEnv

    cfg = KaggricultureEnvConfig.model_validate(_env_data())
    streams = []
    for rank in range(2):
        env = create_env(
            cfg,
            n_envs=2,
            base_seed=11,
            rank=rank,
            world_size=2,
            pin_memory=False,
            transfer_device=torch.device("cpu"),
        )
        assert isinstance(env, KaggricultureVectorizedEnv)
        assert env.seed_state() == (15 + rank, (11 + rank, 13 + rank))
        seeds = set(env.seed_state()[1])
        for _ in range(31):
            env.reset()
            seeds.update(env.seed_state()[1])
        assert seeds == {11 + rank + k * 2 for k in range(64)}
        streams.append(seeds)
    assert streams[0].isdisjoint(streams[1])
