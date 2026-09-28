"""Audited v2 reward definitions, independent of any v2 trainer or model.

Source: v2 ops/myolie-dagger-2026-09-22/selfplay.py terminal_reward,
econ_caps, econ_penalty_parts and econ_shaping_rewards. These formulas do not
imply the v2 critic's projected-bootstrap bounds for the starter's PPO targets.
"""

from typing import Literal, Self

import torch
from pydantic import Field, model_validator

from owl.config import BaseConfig

RewardMode = Literal["win_loss", "win_only", "win_share", "margin"]
ECON_FIELDS = 32


class KaggricultureRewardConfig(BaseConfig):
    share_weight: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    econ_shaping: float = Field(default=0, ge=0, allow_inf_nan=False)
    econ_starvation_weight: float = Field(default=4, ge=0, allow_inf_nan=False)
    econ_drought_weight: float = Field(default=1, ge=0, allow_inf_nan=False)
    econ_cap: float = Field(default=0.25, gt=0, lt=1, allow_inf_nan=False)
    econ_ineffective_weight: float = Field(default=0, ge=0, allow_inf_nan=False)
    econ_ineffective_cap: float = Field(default=0.10, gt=0, lt=1, allow_inf_nan=False)

    @property
    def enabled(self) -> bool:
        return self.econ_shaping > 0 or self.econ_ineffective_weight > 0

    @property
    def terminal_scale(self) -> float:
        death_cap = self.econ_cap if self.econ_shaping > 0 else 0.0
        ineffective_cap = (
            self.econ_ineffective_cap if self.econ_ineffective_weight > 0 else 0.0
        )
        return 1.0 - death_cap - ineffective_cap

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.terminal_scale <= 0:
            raise ValueError("active economic penalty caps must sum below one")
        if (
            self.enabled
            and self.econ_shaping * self.econ_starvation_weight == 0
            and self.econ_shaping * self.econ_drought_weight == 0
            and self.econ_ineffective_weight == 0
        ):
            raise ValueError("economic shaping requires a positive event weight")
        return self


def terminal_rewards(
    banks: torch.Tensor, mode: RewardMode, config: KaggricultureRewardConfig
) -> torch.Tensor:
    margin = banks - banks.flip(-1)
    if mode == "margin":
        return margin / 3000.0
    win_loss = margin.sign()
    if mode == "win_loss":
        return win_loss
    if mode == "win_only":
        return (win_loss + 1) / 2
    if mode == "win_share":
        total = banks.sum(-1, keepdim=True)
        share = torch.where(total > 0, banks / total.clamp_min(1e-30), 0.5)
        return (1 - config.share_weight) * win_loss + config.share_weight * (
            2 * share - 1
        )
    raise ValueError(f"unsupported reward mode: {mode}")


def economic_penalty(
    counts: torch.Tensor, config: KaggricultureRewardConfig
) -> torch.Tensor:
    counts = counts.to(dtype=torch.float64)
    deaths = config.econ_shaping * (
        config.econ_starvation_weight * counts[..., 0]
        + config.econ_drought_weight * counts[..., 1]
    )
    ineffective = config.econ_ineffective_weight * counts[..., 2]
    return deaths.clamp(max=config.econ_cap) + ineffective.clamp(
        max=config.econ_ineffective_cap
    )


def economic_rewards(
    before: torch.Tensor, after: torch.Tensor, config: KaggricultureRewardConfig
) -> torch.Tensor:
    delta = economic_penalty(after, config) - economic_penalty(before, config)
    return delta.flip(-1) - delta
