"""The native-bank telemetry shortcut changes validation, never metric math."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from owl.game import create_env
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.rewards import bank_rewards, bank_score, margin_rewards
from owl.train.config import FullConfig
from pydantic import ValidationError

from tests.kaggriculture.test_env import native_actions, output_tensors, reward_config


def _config(skip: bool | None = None) -> FullConfig:
    overrides: dict[str, object] = {
        "env.config.episode_steps": 4,
        "env.reward_shaping.econ_bank_weight": 0.25,
        "env.reward_shaping.econ_bank_cap": 0.25,
        "env.reward_shaping.econ_margin_weight": 0.25,
        "env.reward_shaping.econ_margin_cap": 0.25,
    }
    if skip is not None:
        overrides["env.skip_reward_telemetry_validation"] = skip
    return FullConfig.from_file(Path("configs/kaggriculture.yaml"), overrides)


def _env(config: FullConfig) -> KaggricultureVectorizedEnv:
    env = create_env(
        config.env,
        n_envs=2,
        base_seed=41,
        rank=0,
        world_size=1,
        pin_memory=False,
        transfer_device=torch.device("cpu"),
    )
    assert isinstance(env, KaggricultureVectorizedEnv)
    return env


def test_telemetry_shortcut_default_preserves_config_and_round_trips(
    tmp_path: Path,
) -> None:
    omitted, disabled, enabled = _config(), _config(False), _config(True)
    assert isinstance(omitted.env, KaggricultureEnvConfig)
    assert not omitted.env.skip_reward_telemetry_validation
    assert omitted.model_dump_json() == disabled.model_dump_json()
    assert "skip_reward_telemetry_validation" not in omitted.model_dump()["env"]
    assert enabled.model_dump()["env"]["skip_reward_telemetry_validation"] is True
    for config in (omitted, disabled, enabled):
        path = tmp_path / "config.yaml"
        config.to_file(path)
        assert FullConfig.from_file(path) == config


@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_telemetry_shortcut_requires_an_explicit_boolean(value: object) -> None:
    data = _config().model_dump()
    data["env"]["skip_reward_telemetry_validation"] = value
    with pytest.raises(ValidationError, match="skip_reward_telemetry_validation"):
        FullConfig.model_validate(data)


def test_telemetry_shortcut_factory_preserves_every_output_byte_and_metric() -> None:
    envs = [_env(_config(skip)) for skip in (None, False, True)]
    completed = 0
    nonzero = set()
    for step in range(9):
        metrics = []
        for env in envs:
            _, _, _, values = env.step(
                native_actions(env, market=[["BUY_PRODUCT", "WHEAT", 1]])
            )
            metrics.append(values)
        assert metrics[0] == metrics[1] == metrics[2]
        for key in ("reward_bank_mean", "reward_margin_abs_mean"):
            if metrics[0][key][0] != 0:
                nonzero.add(key)
        completed += int(envs[0].dones.all())
        reference = {
            name: tensor.numpy().tobytes()
            for name, tensor in output_tensors(envs[0]).items()
        }
        for env in envs[1:]:
            assert {
                name: tensor.numpy().tobytes()
                for name, tensor in output_tensors(env).items()
            } == reference
            assert env.seed_state() == envs[0].seed_state()
        if step == 3:
            for env in envs:
                env.truncate_envs(torch.tensor([True, False]))
    assert completed >= 1
    assert nonzero == {"reward_bank_mean", "reward_margin_abs_mean"}


@pytest.mark.parametrize(("skip", "expected_scans"), [(False, 4), (True, 0)])
def test_telemetry_shortcut_avoids_only_the_finite_scans(
    monkeypatch: pytest.MonkeyPatch, skip: bool, expected_scans: int
) -> None:
    env = _env(_config(skip))
    actions = native_actions(env, market=[["BUY_PRODUCT", "WHEAT", 1]])
    calls = 0
    original = torch.isfinite

    def record(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(tensor)

    monkeypatch.setattr(torch, "isfinite", record)
    env.step(actions)
    assert calls == expected_scans


@pytest.mark.parametrize("enabled", [False, True])
def test_reward_oracles_shortcut_preserves_float64_bits(enabled: bool) -> None:
    config = reward_config().model_copy(
        update={
            "econ_bank_weight": 0.1 if enabled else 0.0,
            "econ_bank_cap": 0.25,
            "econ_margin_weight": 0.1 if enabled else 0.0,
            "econ_margin_cap": 0.25,
        }
    )
    banks = torch.tensor(
        [
            [-0.0, 0.0],
            [-1.0, 3003.0],
            [3006.0, 12345.0],
            [70000.0, 1e200],
            [-torch.finfo(torch.float64).max, torch.finfo(torch.float64).max],
        ],
        dtype=torch.float64,
    )
    eager_score = bank_score(banks, config)
    fast_score = bank_score(banks, config, validate_finite=False)
    assert torch.equal(eager_score.view(torch.int64), fast_score.view(torch.int64))
    for oracle in (bank_rewards, margin_rewards):
        eager = oracle(banks[:-1], banks[1:], config)
        fast = oracle(banks[:-1], banks[1:], config, validate_finite=False)
        assert torch.equal(eager.view(torch.int64), fast.view(torch.int64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_public_reward_oracles_keep_finite_validation_by_default(bad: float) -> None:
    config = reward_config()
    banks = torch.tensor([[bad, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="must be finite"):
        bank_score(banks, config)
    good = torch.zeros_like(banks)
    for oracle in (bank_rewards, margin_rewards):
        for before, after in ((banks, good), (good, banks)):
            with pytest.raises(ValueError, match="must be finite"):
                oracle(before, after, config)


def test_reward_oracles_keep_dtype_and_shape_checks_when_finite_scan_is_off() -> None:
    config = reward_config()
    cases = [
        (torch.zeros(1, 2), TypeError, "dtype float64"),
        (torch.zeros(1, 3, dtype=torch.float64), ValueError, "trailing shape"),
    ]
    for banks, error, message in cases:
        with pytest.raises(error, match=message):
            bank_score(banks, config, validate_finite=False)
        for oracle in (bank_rewards, margin_rewards):
            with pytest.raises(error, match=message):
                oracle(banks, banks, config, validate_finite=False)
    for oracle in (bank_rewards, margin_rewards):
        with pytest.raises(ValueError, match="identical shapes"):
            oracle(
                torch.zeros(1, 2, dtype=torch.float64),
                torch.zeros(2, 2, dtype=torch.float64),
                config,
                validate_finite=False,
            )
