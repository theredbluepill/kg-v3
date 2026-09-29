"""Kaggriculture environment and reward configuration (contract v4, Environment).

``KaggricultureEnvConfig`` is the ``env`` section of a Kaggriculture
``FullConfig``; ``owl.train.config.GameEnvConfig`` selects it over Isaiah's Orbit
``EnvConfig`` by the observation tag. Its fields are the constructor arguments
the contract names for ``KaggricultureEnv`` that a config owns (the seeds come
from the launch). ``KaggricultureRewardConfig`` holds the capped economic
penalty coefficients of the contract's reward formula and keeps the reference
branch's audited field names and defaults.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from owl.config import BaseConfig
from owl.kaggriculture.types import KaggricultureActionConfig, KaggricultureObsConfig


class KaggricultureRewardConfig(BaseConfig):
    """Coefficients of the contract's capped cumulative economic penalty ``P``.

    ``P = min(econ_cap, econ_shaping·(starvation·S + drought·D))
    + min(econ_ineffective_cap, econ_ineffective_weight·I)``, where ``S``, ``D``
    and ``I`` are the cumulative starvation, drought and ineffective counters.
    ``econ_shaping`` is the contract's ``W`` and ``econ_cap`` its ``death_cap``.
    A cap is active when its coefficient is positive, whether or not an event
    occurs; the active caps must sum below one so the terminal win/loss term
    keeps a positive scale.
    """

    econ_shaping: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    econ_starvation_weight: float = Field(default=4.0, ge=0.0, allow_inf_nan=False)
    econ_drought_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    econ_cap: float = Field(default=0.25, gt=0.0, lt=1.0, allow_inf_nan=False)
    econ_ineffective_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    econ_ineffective_cap: float = Field(
        default=0.10, gt=0.0, lt=1.0, allow_inf_nan=False
    )

    @property
    def terminal_scale(self) -> float:
        """``1 - active caps``: the weight of ``sign(bank_s - bank_o)`` at the end."""
        death_cap = self.econ_cap if self.econ_shaping > 0.0 else 0.0
        ineffective_cap = (
            self.econ_ineffective_cap if self.econ_ineffective_weight > 0.0 else 0.0
        )
        return 1.0 - death_cap - ineffective_cap

    @model_validator(mode="after")
    def _validate_budget(self) -> Self:
        if self.terminal_scale <= 0.0:
            raise ValueError(
                "active economic penalty caps must sum below one; got "
                f"terminal_scale={self.terminal_scale}"
            )
        if (
            self.econ_shaping > 0.0
            and self.econ_starvation_weight == 0.0
            and self.econ_drought_weight == 0.0
        ):
            raise ValueError(
                "economic shaping requires a positive event weight: econ_shaping > 0 "
                "with zero starvation and drought weights penalizes nothing"
            )
        return self


class KaggricultureEnvConfig(BaseConfig):
    """The ``env`` section of a Kaggriculture training config.

    Kaggriculture always seats two players, so Isaiah's ``two_player_weight`` has
    no meaning here and is rejected as an unknown key. The contract's reward
    requires ``reward_mode='win_loss'`` (and ``rl.gamma=1``, checked by
    ``FullConfig``).
    """

    n_envs: int = Field(default=2, ge=1)
    obs_spec: KaggricultureObsConfig = Field(default_factory=KaggricultureObsConfig)
    action_spec: KaggricultureActionConfig = Field(
        default_factory=KaggricultureActionConfig
    )
    reward_mode: Literal["win_loss"] = "win_loss"
    reward_shaping: KaggricultureRewardConfig = Field(
        default_factory=KaggricultureRewardConfig
    )
    pin_memory: bool = True
    native_threads: int = Field(default=1, ge=1)
