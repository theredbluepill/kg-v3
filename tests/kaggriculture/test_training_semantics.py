"""Kaggriculture game semantics in the shared PPO trainer (rebuild Task 3.2).

L2: a time-limit cut keeps the economic reward earned on the cut transition and
bootstraps from the critic. Joint per-player clipping: one seat's turn is one
autoregressive action, so PPO clips its joint ratio. Value-mode guards: the
per-seat winner critic (value ``2p(self) - 1``) needs the undiscounted
``win_loss`` return and the MSE value loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.rl import ObsBatch, PureActionMask
from owl.train import FullConfig, PPOConfig
from owl.train.advantages import compute_gae
from owl.train.config import _validate_kaggriculture_training
from owl.train.ppo import (
    _cut_truncated_envs_,
    _ppo_loss_components,
    _truncation_keeps_transition_reward,
)
from pydantic import BaseModel, ValidationError

from tests.kaggriculture.conftest import make_obs


def _orbit_obs(n_envs: int) -> ObsBatch:
    return ObsBatch(
        planets=torch.zeros((n_envs, 1, 1)),
        orbiting_planets=torch.zeros((n_envs, 1), dtype=torch.bool),
        fleets=torch.zeros((n_envs, 1, 1)),
        comets=torch.zeros((n_envs, 1, 1)),
        entity_mask=torch.zeros((n_envs, 1), dtype=torch.bool),
        still_playing=torch.ones((n_envs, 4), dtype=torch.bool),
        global_features=torch.zeros((n_envs, 1)),
        action_mask=PureActionMask(
            can_act=torch.zeros((n_envs, 4, 1), dtype=torch.bool),
            max_launch=torch.zeros((n_envs, 4, 1), dtype=torch.int64),
        ),
    )


# L2 -------------------------------------------------------------------------


def test_kaggriculture_truncation_keeps_economic_reward_and_bootstraps() -> None:
    obs = make_obs(envs=2)
    assert _truncation_keeps_transition_reward(obs)
    # Economic shaping paid on the cut transition (zero-sum across the seats).
    rewards = torch.tensor([[0.05, -0.05], [0.02, -0.02]])
    earned = rewards.clone()
    dones = torch.zeros((2, 2), dtype=torch.bool)
    cut_value = torch.tensor([[0.3, -0.3]])

    truncated, bootstrap_values = _cut_truncated_envs_(
        rewards,
        dones,
        rows=torch.tensor([1]),
        row_values=cut_value,
        keep_transition_reward=_truncation_keeps_transition_reward(obs),
    )

    assert torch.equal(rewards, earned)
    assert dones.tolist() == [[False, False], [True, True]]
    assert truncated.tolist() == [[False, False], [True, True]]
    torch.testing.assert_close(
        bootstrap_values, torch.tensor([[0.0, 0.0], [0.3, -0.3]])
    )
    values = torch.tensor([[[0.1, -0.1]], [[0.2, -0.2]]])
    _advantages, returns = compute_gae(
        rewards=rewards[:, None],
        values=values,
        dones=dones[:, None],
        last_values=torch.zeros((2, 2)),
        gamma=1.0,
        gae_lambda=0.95,
        truncated=truncated[:, None],
        bootstrap_values=bootstrap_values[:, None],
    )
    # The cut row's target is the earned economic reward plus the critic's value
    # of the cut state; no terminal winner is fabricated.
    torch.testing.assert_close(returns[1, 0], earned[1] + cut_value[0])


def test_orbit_truncation_still_drops_the_cut_transition_reward() -> None:
    obs = _orbit_obs(2)
    assert not _truncation_keeps_transition_reward(obs)
    rewards = torch.tensor([[1.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    dones = torch.zeros((2, 4), dtype=torch.bool)

    truncated, bootstrap_values = _cut_truncated_envs_(
        rewards,
        dones,
        rows=torch.tensor([1]),
        row_values=torch.tensor([[0.5, 0.25, -0.25, -0.5]]),
        keep_transition_reward=False,
    )

    assert rewards.tolist() == [[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert dones[1].all()
    assert not dones[0].any()
    assert truncated.shape == (2, 4)
    assert bootstrap_values[1].tolist() == [0.5, 0.25, -0.25, -0.5]


def test_truncation_reward_rule_rejects_unknown_observation_batches() -> None:
    class _Unknown(BaseModel):
        pass

    with pytest.raises(TypeError, match="no truncation reward rule for _Unknown"):
        _truncation_keeps_transition_reward(_Unknown())


# Joint per-player clipping ----------------------------------------------------


def test_kaggriculture_per_player_clip_bounds_the_joint_turn_ratio() -> None:
    frames = kt.MAX_FRAMES
    acting = 40
    old_frames = torch.zeros((1, 2, frames))
    new_frames = old_frames.clone()
    new_frames[..., :acting] = 0.01  # each frame's ratio e^0.01 is inside the clip
    frame_weight = torch.zeros((1, 2, frames))
    frame_weight[..., :acting] = 1.0
    advantages = torch.ones((1, 2))
    zeros = torch.zeros((1, 2))

    policy_loss, _, _, _, clipfrac, ratio, logratio = _ppo_loss_components(
        new_frames.sum(dim=-1),
        zeros,
        zeros,
        old_frames.sum(dim=-1),
        zeros,
        zeros,
        advantages,
        0.2,
        None,
        "per_player",
        None,
    )

    # The joint turn moved by 40 x 0.01 = 0.4 nats: ratio e^0.4 > 1.2 is clipped.
    torch.testing.assert_close(logratio, torch.full((1, 2), 0.4))
    torch.testing.assert_close(ratio, torch.full((1, 2), 0.4).exp())
    assert clipfrac.tolist() == [[1.0, 1.0]]
    torch.testing.assert_close(policy_loss, torch.full((1, 2), -1.2))

    _, _, _, _, frame_clipfrac, _, _ = _ppo_loss_components(
        new_frames,
        torch.zeros((1, 2, frames)),
        zeros,
        old_frames,
        zeros,
        zeros,
        advantages,
        0.2,
        None,
        "per_entity",
        frame_weight,
    )
    # Per-frame clipping would let the same joint move through unclipped.
    assert frame_clipfrac.tolist() == [[0.0, 0.0]]


# Value-mode guards --------------------------------------------------------------


def _isaiah_value_settings(**overrides: Any) -> PPOConfig:
    """``configs/scaling_6m.yaml``'s value and clipping settings."""
    return PPOConfig.model_validate(
        {
            "gamma": 1.0,
            "vf_clip_coef": None,
            "ppo_clip_mode": "per_player",
            "value_loss": "mse",
            **overrides,
        }
    )


def test_kaggriculture_training_accepts_isaiah_value_settings() -> None:
    _validate_kaggriculture_training(
        reward_mode="win_loss", rl=_isaiah_value_settings()
    )
    _validate_kaggriculture_training(
        reward_mode="win_loss", rl=_isaiah_value_settings(vf_clip_coef=0.2)
    )


@pytest.mark.parametrize(
    ("reward_mode", "overrides", "message"),
    [
        ("win_only", {}, "env.reward_mode='win_loss'"),
        ("ship_ratio", {}, "env.reward_mode='win_loss'"),
        ("win_loss", {"gamma": 0.99}, "rl.gamma=1.0"),
        ("win_loss", {"value_loss": "winner_ce"}, "rl.value_loss='mse'"),
    ],
)
def test_kaggriculture_training_rejects_incompatible_value_settings(
    reward_mode: Any, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_kaggriculture_training(
            reward_mode=reward_mode, rl=_isaiah_value_settings(**overrides)
        )


def test_kaggriculture_training_requires_joint_per_player_clipping() -> None:
    with pytest.raises(ValueError, match=r"rl\.ppo_clip_mode='per_player'"):
        _validate_kaggriculture_training(
            reward_mode="win_loss",
            rl=_isaiah_value_settings(ppo_clip_mode="per_entity"),
        )


def _orbit_full_config(**rl: Any) -> FullConfig:
    return FullConfig.model_validate(
        {
            "env": {"n_envs": 2},
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {"optimizer": "adamw", "learning_rate": 0.001},
            "rl": {"horizon": 4, **rl},
        }
    )


_CONFIGS = Path(__file__).parents[2] / "configs"
_KAGGRICULTURE_CONFIGS = sorted(_CONFIGS.glob("kaggriculture*.yaml"))


def _kaggriculture_config_data(path: Path) -> dict[str, Any]:
    return FullConfig.from_file(path).model_dump(mode="json", round_trip=True)


def test_full_config_applies_kaggriculture_guards_only_to_kaggriculture() -> None:
    # Orbit keeps Isaiah's discounted, per-entity settings.
    _orbit_full_config(gamma=0.99, ppo_clip_mode="per_entity")
    farm = _kaggriculture_config_data(_CONFIGS / "kaggriculture.yaml")
    farm["rl"]["gamma"] = 0.99

    with pytest.raises(ValidationError, match=r"rl\.gamma=1\.0"):
        FullConfig.model_validate(farm)


@pytest.mark.parametrize("path", _KAGGRICULTURE_CONFIGS, ids=lambda p: p.name)
@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("rl", "gamma", 0.99, r"rl\.gamma=1\.0"),
        ("rl", "value_loss", "winner_ce", "rl.value_loss='winner_ce'"),
        ("env", "reward_mode", "win_only", "reward_mode"),
        ("rl", "ppo_clip_mode", "per_entity", r"rl\.ppo_clip_mode='per_player'"),
    ],
)
def test_kaggriculture_yaml_configs_load_through_the_training_guards(
    path: Path, section: str, key: str, value: object, message: str
) -> None:
    data = _kaggriculture_config_data(path)
    assert isinstance(FullConfig.model_validate(data).env, KaggricultureEnvConfig)

    data[section][key] = value
    with pytest.raises(ValidationError, match=message):
        FullConfig.model_validate(data)
