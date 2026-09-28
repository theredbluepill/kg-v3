"""Explicit game boundaries consumed by the shared training infrastructure."""

from __future__ import annotations

from typing import TypeAlias

import torch

from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.types import (
    KaggricultureActionConfig,
    KaggricultureActionMask,
    KaggricultureActions,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)
from owl.rl import (
    ActionBundle,
    ActionMask,
    ObsBatch,
    RewardMode,
    SupportedActionConfig,
    SupportedObsConfig,
    VectorizedEnv,
)

ObservationBatch: TypeAlias = ObsBatch | KaggricultureObsBatch
ActionBatch: TypeAlias = ActionBundle | KaggricultureActions
GameActionMask: TypeAlias = ActionMask | KaggricultureActionMask
GameEnv: TypeAlias = VectorizedEnv | KaggricultureVectorizedEnv


def create_env(
    *,
    n_envs: int,
    obs_spec: SupportedObsConfig,
    action_spec: SupportedActionConfig,
    two_player_weight: float = 0.5,
    reward_mode: RewardMode = "win_loss",
    pin_memory: bool = True,
    seed: int = 0,
    seed_stride: int = 1,
    native_threads: int = 1,
    reward_shaping: KaggricultureRewardConfig | None = None,
) -> GameEnv:
    if isinstance(obs_spec, KaggricultureObsConfig):
        if not isinstance(action_spec, KaggricultureActionConfig):
            raise ValueError("Kaggriculture observations require Kaggriculture actions")
        if two_player_weight != 1.0:
            raise ValueError("Kaggriculture has exactly two players")
        if reward_mode == "ship_ratio":
            raise ValueError("ship_ratio is not a Kaggriculture reward")
        return KaggricultureVectorizedEnv(
            n_envs=n_envs,
            obs_spec=obs_spec,
            action_spec=action_spec,
            reward_mode=reward_mode,
            pin_memory=pin_memory,
            seed=seed,
            seed_stride=seed_stride,
            threads=native_threads,
            reward_shaping=reward_shaping,
        )
    if isinstance(action_spec, KaggricultureActionConfig):
        raise ValueError("Orbit observations cannot use Kaggriculture actions")
    return VectorizedEnv(
        n_envs=n_envs,
        obs_spec=obs_spec,
        action_spec=action_spec,
        two_player_weight=two_player_weight,
        reward_mode=reward_mode,
        pin_memory=pin_memory,
    )


def step_env(
    env: GameEnv, actions: ActionBatch
) -> tuple[ObservationBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    """Validate the game boundary before dispatching native action tensors."""
    if isinstance(env, KaggricultureVectorizedEnv):
        if not isinstance(actions, KaggricultureActions):
            raise TypeError("Kaggriculture requires frame actions")
        return env.step(actions)
    if isinstance(actions, KaggricultureActions):
        raise TypeError("Orbit environments cannot execute Kaggriculture actions")
    return env.step(actions)
