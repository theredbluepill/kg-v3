"""Typed environment construction; training and model construction stay separate."""

from __future__ import annotations

from typing import TypeAlias

import torch

from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.types import KaggricultureActions, KaggricultureObsBatch
from owl.rl import ActionBundle, EnvConfig, ObsBatch, VectorizedEnv

GameVectorizedEnv: TypeAlias = VectorizedEnv | KaggricultureVectorizedEnv
GameObsBatch: TypeAlias = ObsBatch | KaggricultureObsBatch
GameActions: TypeAlias = ActionBundle | KaggricultureActions
_I64_MAX = 2**63 - 1


def create_env(
    env_config: EnvConfig | KaggricultureEnvConfig,
    *,
    n_envs: int,
    base_seed: int,
    rank: int,
    world_size: int,
    pin_memory: bool,
    transfer_device: torch.device,
) -> GameVectorizedEnv:
    """Construct one game's adapter, with disjoint Kaggriculture rank streams.

    Kaggriculture consumes ``base_seed + rank + k * world_size``. With
    ``env.opponent_mix`` the first ``fraction * n_envs`` envs of each rank host
    the fixed opponent. Orbit keeps Isaiah's original constructor and its
    existing native seed behavior.
    """
    for name, value in (
        ("n_envs", n_envs),
        ("base_seed", base_seed),
        ("rank", rank),
        ("world_size", world_size),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if not 1 <= world_size <= _I64_MAX:
        raise ValueError("world_size must be in 1..2**63-1")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if not 0 <= base_seed <= _I64_MAX:
        raise ValueError("base_seed must be in 0..2**63-1")
    if base_seed + rank > _I64_MAX:
        raise ValueError("base_seed + rank must fit int64")
    if isinstance(env_config, KaggricultureEnvConfig):
        mix = env_config.opponent_mix
        return KaggricultureVectorizedEnv(
            n_envs=n_envs,
            seed=base_seed + rank,
            seed_stride=world_size,
            config=env_config.config,
            reward_config=env_config.reward_shaping,
            reward_mode=env_config.reward_mode,
            native_threads=env_config.native_threads,
            pin_memory=pin_memory,
            transfer_device=transfer_device,
            obs_spec=env_config.obs_spec,
            action_spec=env_config.action_spec,
            opponent_bot=None if mix is None else mix.bot,
            opponent_envs=0 if mix is None else mix.bot_envs(n_envs),
            skip_reward_telemetry_validation=(
                env_config.skip_reward_telemetry_validation
            ),
        )
    if isinstance(env_config, EnvConfig):
        return VectorizedEnv(
            n_envs=n_envs,
            obs_spec=env_config.obs_spec,
            action_spec=env_config.action_spec,
            two_player_weight=env_config.two_player_weight,
            reward_mode=env_config.reward_mode,
            pin_memory=pin_memory,
        )
    raise TypeError("env_config must be EnvConfig or KaggricultureEnvConfig")
