"""Kaggriculture environment config selected by the existing observation tag.

The explicit reward coefficients live in ``rewards.py``; ``reward_mode`` remains
owned by this environment config, matching the shared training config seam.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from owl import rs
from owl.config import BaseConfig
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.types import (
    KaggricultureActionConfig,
    KaggricultureGameConfig,
    KaggricultureObsConfig,
)


class KaggricultureOpponentMixConfig(BaseConfig):
    """Fixed-opponent collection: ``fraction`` of each env batch hosts ``bot``.

    In those environments one seat is played natively by the scripted ``bot``
    (a key of the ``opponents_rs`` registry) and the other by the learner; the
    learned seat alternates by env index and episode. The remaining envs stay
    mirror self-play. The bot key is collection bookkeeping and a W&B label
    only: it never reaches observations, rewards, losses, normalization or
    checkpoint selection. Bot behaviour is qualified against the original
    Python submissions at the default game configuration only.
    """

    bot: str = Field(min_length=1, strict=True)
    fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("bot")
    @classmethod
    def _registered_bot(cls, bot: str) -> str:
        known = rs.kaggriculture_opponent_bots()
        if bot not in known:
            raise ValueError(f"unknown opponent bot {bot!r}; expected one of {known}")
        return bot

    def bot_envs(self, n_envs: int) -> int:
        """The number of fixed-opponent envs (the first ones) in an env batch.

        ``fraction * n_envs`` must be a whole number of at least one: the mix
        never rounds silently.
        """
        if type(n_envs) is not int or n_envs < 1:
            raise ValueError("n_envs must be a positive integer")
        exact = self.fraction * n_envs
        count = round(exact)
        if count < 1 or not math.isclose(exact, count, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"env.opponent_mix.fraction={self.fraction} x n_envs={n_envs} must "
                "be a whole number of at least one fixed-opponent env"
            )
        return count


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
    # None (the default) is pure mirror self-play, byte-identical to the
    # pre-mix trainer.
    opponent_mix: KaggricultureOpponentMixConfig | None = None

    @model_validator(mode="after")
    def _validate_opponent_mix(self) -> Self:
        if self.opponent_mix is not None:
            self.opponent_mix.bot_envs(self.n_envs)
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_opponent_mix(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Self-play configs dump exactly as before the mix existed, so their
        # config.yaml and v3/config_sha256 identity are unchanged.
        data: dict[str, Any] = handler(self)
        if self.opponent_mix is None:
            del data["opponent_mix"]
        return data
