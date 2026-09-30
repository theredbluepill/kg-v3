"""Explicit reward coefficients and independent test oracles, never live rewards.

The native environment owns production reward calculation. These float64
functions expose the contract's capped cumulative penalties and the owner's
absolute own-bank shaping (term A, 2026-09-30) for diagnostics, telemetry and
tests; only ``transition_rewards`` reproduces the native float32 rounding points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

import torch
from pydantic import Field, model_validator
from torch import Tensor

from owl.config import BaseConfig

if TYPE_CHECKING:
    from owl.rs import KaggricultureRewardDict


class KaggricultureRewardConfig(BaseConfig):
    """Nine explicit coefficients: capped death/ineffective penalties and own bank.

    ``econ_shaping`` is the contract's W and ``econ_cap`` its death cap.
    ``econ_bank_weight`` (w_b), ``econ_bank_scale`` (S) and ``econ_bank_cap``
    (cap_b) define the own-seat bank score ``min(cap_b, w_b * max(0, bank) / S)``
    whose per-step difference is paid to that seat alone (not zero-sum). Reward
    mode is owned once by ``KaggricultureEnvConfig`` and supplied at serialization.
    """

    econ_shaping: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_starvation_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_drought_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_cap: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_ineffective_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_ineffective_cap: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_bank_weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_bank_scale: float = Field(ge=0, allow_inf_nan=False, strict=True)
    econ_bank_cap: float = Field(ge=0, allow_inf_nan=False, strict=True)

    def _active_caps(self) -> tuple[float, float, float]:
        return (
            self.econ_cap if self.econ_shaping > 0 else 0.0,
            self.econ_ineffective_cap if self.econ_ineffective_weight > 0 else 0.0,
            self.econ_bank_cap if self.econ_bank_weight > 0 else 0.0,
        )

    @property
    def terminal_scale(self) -> float:
        """Return one minus the enabled caps, regardless of observed events."""
        death_cap, ineffective_cap, bank_cap = self._active_caps()
        # Native order: 1 - death - ineffective - bank, left to right.
        return 1.0 - death_cap - ineffective_cap - bank_cap

    @model_validator(mode="after")
    def _validate_budget(self) -> Self:
        death_cap, ineffective_cap, bank_cap = self._active_caps()
        # Native order: (death + ineffective) + bank.
        if death_cap + ineffective_cap + bank_cap >= 1:
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
        if self.econ_bank_weight > 0 and (
            self.econ_bank_scale <= 0 or self.econ_bank_cap <= 0
        ):
            raise ValueError(
                "positive econ_bank_weight requires positive econ_bank_scale "
                "and econ_bank_cap"
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
            "econ_bank_weight": self.econ_bank_weight,
            "econ_bank_scale": self.econ_bank_scale,
            "econ_bank_cap": self.econ_bank_cap,
        }


def _validate_counts(counts: Tensor, name: str) -> None:
    if counts.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype int64")
    if counts.ndim < 2 or counts.shape[-2:] != (2, 32):
        raise ValueError(f"{name} must have trailing shape [2, 32]")
    if bool((counts < 0).any()):
        raise ValueError(f"{name} must contain nonnegative cumulative counts")


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
        # Native binary64 operation order is authoritative: an overflowing
        # inner sum stays infinite even for tiny W, then saturates at the cap.
        # Rescaling W into the event weights would change native rewards.
        penalty += deaths
    if config.econ_ineffective_weight > 0:
        penalty += (config.econ_ineffective_weight * values[..., 2]).clamp(
            max=config.econ_ineffective_cap
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


def _validate_banks(banks: Tensor, name: str) -> None:
    if banks.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype float64")
    if banks.ndim < 1 or banks.shape[-1] != 2:
        raise ValueError(f"{name} must have trailing shape [2]")
    if not bool(torch.isfinite(banks).all()):
        raise ValueError(f"{name} must be finite")


def bank_score(banks: Tensor, config: KaggricultureRewardConfig) -> Tensor:
    """Return each seat's own-bank score ``min(cap_b, w_b * max(0, bank) / S)``.

    Float64 in native operation order (product, then quotient, then cap); an
    overflowing product saturates at the cap. Zero when ``econ_bank_weight`` is 0.
    """
    _validate_banks(banks, "banks")
    if config.econ_bank_weight <= 0:
        return torch.zeros_like(banks)
    raw = config.econ_bank_weight * banks.clamp(min=0.0) / config.econ_bank_scale
    return raw.clamp(max=config.econ_bank_cap)


def bank_rewards(
    banks_before: Tensor, banks_after: Tensor, config: KaggricultureRewardConfig
) -> Tensor:
    """Return each seat's own bank-score increment in float64 (not zero-sum)."""
    if banks_before.shape != banks_after.shape:
        raise ValueError("banks_before and banks_after must have identical shapes")
    return bank_score(banks_after, config) - bank_score(banks_before, config)


def terminal_rewards(banks: Tensor, config: KaggricultureRewardConfig) -> Tensor:
    """Return the scaled terminal win/loss/draw term in float64."""
    _validate_banks(banks, "banks")
    rival = banks.flip(-1)
    # Compare directly: subtracting opposite finite f64 extremes can overflow.
    sign = (banks > rival).double() - (banks < rival).double()
    return sign * config.terminal_scale


def transition_rewards(
    before: Tensor,
    after: Tensor,
    banks_before: Tensor,
    banks_after: Tensor,
    dones: Tensor,
    config: KaggricultureRewardConfig,
) -> Tensor:
    """Model native economic f32 rounding, then terminal f64 addition and f32.

    The economic term is the relative penalty difference plus, only when
    ``econ_bank_weight > 0``, the own bank-score increment, summed in float64
    before the first f32 rounding. With the bank term off nothing is added, so
    the result is bit-identical to the relative-only reward.
    """
    economic = economic_rewards(before, after, config)
    terminal = terminal_rewards(banks_after, config)
    if (
        banks_before.shape != economic.shape
        or banks_after.shape != economic.shape
        or dones.shape != economic.shape
    ):
        raise ValueError(
            "banks_before, banks_after and dones must match the economic seat shape"
        )
    if dones.dtype != torch.bool:
        raise TypeError("dones must have dtype bool")
    if any(
        tensor.device != economic.device
        for tensor in (banks_before, banks_after, dones)
    ):
        raise ValueError("transition tensors must use the same device")
    if config.econ_bank_weight > 0:
        economic = economic + bank_rewards(banks_before, banks_after, config)
    else:
        _validate_banks(banks_before, "banks_before")
    return (economic.float().double() + terminal * dones).float()
