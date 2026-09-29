"""Kaggriculture environment config selected by the existing observation tag.

The explicit reward coefficients live in ``rewards.py``; ``reward_mode`` remains
owned by this environment config, matching the shared training config seam.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from owl.config import BaseConfig
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.types import (
    KaggricultureActionConfig,
    KaggricultureGameConfig,
    KaggricultureObsConfig,
)


class KaggricultureEnvConfig(BaseConfig):
    """The two-seat game config; ``action_spec`` alone owns ``hire_limit``."""

    n_envs: int = Field(default=2, ge=1, strict=True)
    seed: int = Field(default=0, ge=0, le=2**63 - 1, strict=True)
    config: KaggricultureGameConfig = Field(default_factory=KaggricultureGameConfig)
    obs_spec: KaggricultureObsConfig = Field(default_factory=KaggricultureObsConfig)
    action_spec: KaggricultureActionConfig = Field(
        default_factory=KaggricultureActionConfig
    )
    reward_mode: Literal["win_loss"] = "win_loss"
    reward_shaping: KaggricultureRewardConfig
    pin_memory: bool = True
    native_threads: int = Field(ge=1, strict=True)
