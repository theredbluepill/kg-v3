"""Explicit reward coefficients and independent test oracles, never live rewards.

The native environment owns production reward calculation. These float64
functions expose the contract's capped cumulative penalties for diagnostics and
tests; only ``transition_rewards`` reproduces the native float32 rounding points.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Self

import torch
from pydantic import Field, model_validator
from torch import Tensor

from owl.config import BaseConfig

if TYPE_CHECKING:
    from owl.rs import KaggricultureRewardDict


class KaggricultureRewardConfig(BaseConfig):
    """Six explicit coefficients for separately capped death/ineffective penalties.

    ``econ_shaping`` is the contract's W and ``econ_cap`` its death cap. Reward
    mode is owned once by ``KaggricultureEnvConfig`` and supplied at serialization.
    """

    econ_shaping: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_starvation_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_drought_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_cap: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_ineffective_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_ineffective_cap: float = Field(ge=0, allow_inf_nan=False, strict=True)

    @property
    def terminal_scale(self) -> float:
        """Return one minus the enabled caps, regardless of observed events."""
        death_cap = self.econ_cap if self.econ_shaping > 0 else 0.0
        ineffective_cap = (
            self.econ_ineffective_cap if self.econ_ineffective_weight > 0 else 0.0
        )
        return 1.0 - death_cap - ineffective_cap

    @model_validator(mode="after")
    def _validate_budget(self) -> Self:
        death_cap = self.econ_cap if self.econ_shaping > 0 else 0.0
        ineffective_cap = (
            self.econ_ineffective_cap if self.econ_ineffective_weight > 0 else 0.0
        )
        if death_cap + ineffective_cap >= 1:
            raise ValueError("active economic penalty caps must sum below one")
        if self.econ_shaping > 0:
            if self.econ_cap <= 0:
                raise ValueError("positive econ_shaping requires positive econ_cap")
            # Shared ABI: each product is one binary64 multiplication. Do not
            # replace this with checks of the raw weights (underflow matters).
            if not (
                self.econ_shaping * self.econ_starvation_weight > 0
                or self.econ_shaping * self.econ_drought_weight > 0
            ):
                raise ValueError(
                    "economic shaping requires a positive event weight product"
                )
        if self.econ_ineffective_weight > 0 and self.econ_ineffective_cap <= 0:
            raise ValueError(
                "positive econ_ineffective_weight requires positive "
                "econ_ineffective_cap"
            )
        return self

    def to_native_dict(
        self, reward_mode: Literal["win_loss"]
    ) -> KaggricultureRewardDict:
        """Serialize the exact native ABI keys with the envelope's reward mode."""
        if reward_mode != "win_loss":
            raise ValueError("reward_mode must be 'win_loss'")
        return {
            "reward_mode": reward_mode,
            "econ_shaping": self.econ_shaping,
            "econ_starvation_weight": self.econ_starvation_weight,
            "econ_drought_weight": self.econ_drought_weight,
            "econ_cap": self.econ_cap,
            "econ_ineffective_weight": self.econ_ineffective_weight,
            "econ_ineffective_cap": self.econ_ineffective_cap,
        }


def _validate_counts(counts: Tensor, name: str) -> None:
    if counts.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype int64")
    if counts.ndim < 2 or counts.shape[-2:] != (2, 32):
        raise ValueError(f"{name} must have trailing shape [2, 32]")
    if bool((counts < 0).any()):
        raise ValueError(f"{name} must contain nonnegative cumulative counts")


def _capped_product(counts: Tensor, coefficient: float, cap: float) -> Tensor:
    if coefficient == 0:
        return torch.zeros_like(counts)
    if math.isinf(coefficient):
        # Finite config factors can overflow when multiplied. Avoid 0 * inf.
        return torch.where(
            counts > 0, torch.full_like(counts, cap), torch.zeros_like(counts)
        )
    return (counts * coefficient).clamp(max=cap)


def economic_penalty(counts: Tensor, config: KaggricultureRewardConfig) -> Tensor:
    """Return P per seat in float64, using only S0, D1 and I2."""
    _validate_counts(counts, "counts")
    values = counts.to(torch.float64)
    penalty = torch.zeros_like(values[..., 0])
    if config.econ_shaping > 0:
        weighted = (
            config.econ_starvation_weight * values[..., 0]
            + config.econ_drought_weight * values[..., 1]
        )
        raw = config.econ_shaping * weighted
        deaths = raw.clamp(max=config.econ_cap)
        if bool((~torch.isfinite(weighted)).any()):
            # Preserve the formula's operation order normally. Only overflowing
            # inner sums need rescaling: tiny W may bring their result below cap.
            scaled = _capped_product(
                values[..., 0],
                config.econ_shaping * config.econ_starvation_weight,
                config.econ_cap,
            ) + _capped_product(
                values[..., 1],
                config.econ_shaping * config.econ_drought_weight,
                config.econ_cap,
            )
            deaths = torch.where(
                torch.isfinite(weighted), deaths, scaled.clamp(max=config.econ_cap)
            )
        penalty += deaths
    if config.econ_ineffective_weight > 0:
        penalty += _capped_product(
            values[..., 2], config.econ_ineffective_weight, config.econ_ineffective_cap
        )
    if not bool(torch.isfinite(penalty).all()):
        raise ValueError("economic penalty must be finite")
    return penalty


def economic_rewards(
    before: Tensor, after: Tensor, config: KaggricultureRewardConfig
) -> Tensor:
    """Return the rival penalty increment minus the own increment in float64."""
    _validate_counts(before, "before")
    _validate_counts(after, "after")
    if before.shape != after.shape:
        raise ValueError("before and after must have identical shapes")
    if before.device != after.device:
        raise ValueError("before and after must use the same device")
    if bool((after < before).any()):
        raise ValueError("cumulative economic counters must be monotonic")
    delta = economic_penalty(after, config) - economic_penalty(before, config)
    return delta.flip(-1) - delta


def terminal_rewards(banks: Tensor, config: KaggricultureRewardConfig) -> Tensor:
    """Return the scaled terminal win/loss/draw term in float64."""
    if banks.dtype != torch.float64:
        raise TypeError("banks must have dtype float64")
    if banks.ndim < 1 or banks.shape[-1] != 2:
        raise ValueError("banks must have trailing shape [2]")
    if not bool(torch.isfinite(banks).all()):
        raise ValueError("banks must be finite")
    rival = banks.flip(-1)
    # Compare directly: subtracting opposite finite f64 extremes can overflow.
    sign = (banks > rival).double() - (banks < rival).double()
    return sign * config.terminal_scale


def transition_rewards(
    before: Tensor,
    after: Tensor,
    banks_after: Tensor,
    dones: Tensor,
    config: KaggricultureRewardConfig,
) -> Tensor:
    """Model native economic f32 rounding, then terminal f64 addition and f32."""
    economic = economic_rewards(before, after, config)
    terminal = terminal_rewards(banks_after, config)
    if banks_after.shape != economic.shape or dones.shape != economic.shape:
        raise ValueError("banks_after and dones must match the economic seat shape")
    if dones.dtype != torch.bool:
        raise TypeError("dones must have dtype bool")
    if banks_after.device != economic.device or dones.device != economic.device:
        raise ValueError("transition tensors must use the same device")
    return (economic.float().double() + terminal * dones).float()
