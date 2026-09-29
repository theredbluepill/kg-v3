from typing import Self

from pydantic import Field, model_validator

from owl.config import BaseConfig
from owl.kaggriculture.types import KaggricultureObsConfig
from owl.model import ActorDiscreteTargetBinsConfig, ModelConfig
from owl.rl import ActionDiscreteTargetBinsConfig, EnvConfig, RewardMode

from .optimizer import OptimizerConfig
from .ppo import PPOConfig


def _validate_kaggriculture_training(*, reward_mode: RewardMode, rl: PPOConfig) -> None:
    """Reject settings that change the Kaggriculture objective or critic meaning.

    The per-seat winner critic reads ``value = 2 p(self) - 1`` in (-1, 1), the
    undiscounted ``win_loss`` return. Economic shaping keeps complete-episode
    returns in [-1, 1] only through ``terminal_scale`` at gamma 1
    (``docs/kaggriculture-contract.md``, "Rewards"), and truncation bootstraps
    from the same critic. One seat's turn is one autoregressive action over its
    frames, so PPO clips the joint turn ratio.
    """
    if reward_mode != "win_loss":
        raise ValueError(
            "Kaggriculture requires env.reward_mode='win_loss': its critic value "
            f"is 2p(self) - 1, got reward_mode={reward_mode!r}"
        )
    if rl.gamma != 1.0:
        raise ValueError(
            "Kaggriculture requires rl.gamma=1.0: terminal_scale bounds the shaped "
            f"return only undiscounted, got gamma={rl.gamma}"
        )
    if rl.value_loss != "mse":
        raise ValueError(
            "Kaggriculture requires rl.value_loss='mse': winner_ce needs the "
            "win_only reward, and the economic reward is not a winner distribution"
        )
    if rl.ppo_clip_mode != "per_player":
        raise ValueError(
            "Kaggriculture requires rl.ppo_clip_mode='per_player': a seat's turn "
            "is one joint autoregressive action, so PPO clips its joint ratio"
        )


class RuntimeConfig(BaseConfig):
    n_runtime_gpus: int = Field(default=1, ge=1)


class FullConfig(BaseConfig):
    env: EnvConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    rl: PPOConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _validate_cross_config_constraints(self) -> Self:
        if isinstance(self.env.obs_spec, KaggricultureObsConfig):
            _validate_kaggriculture_training(
                reward_mode=self.env.reward_mode, rl=self.rl
            )
        if self.model.actor.action_spec != self.env.action_spec.action_spec:
            raise ValueError("model actor action_spec must match env action_spec")
        if (
            isinstance(self.model.actor, ActorDiscreteTargetBinsConfig)
            and isinstance(self.env.action_spec, ActionDiscreteTargetBinsConfig)
            and self.model.actor.n_bins != self.env.action_spec.n_bins
        ):
            raise ValueError("model actor n_bins must match env action_spec n_bins")
        if self.env.reward_mode == "win_only" and self.model.value_mode != "win_only":
            raise ValueError(
                "env.reward_mode='win_only' requires model.value_mode='win_only'"
            )
        if self.env.reward_mode != "win_only" and self.model.value_mode == "win_only":
            raise ValueError(
                "model.value_mode='win_only' requires env.reward_mode='win_only'"
            )
        if self.rl.value_loss == "winner_ce":
            # The cross-entropy value loss trains the winner-probability softmax
            # directly against a distributional winner target. That target is a
            # valid probability distribution only with the undiscounted, sum-to-one
            # win_only reward (value_mode='win_only' follows from it above), and the
            # softmax critic; value clipping has no cross-entropy analogue.
            if self.env.reward_mode != "win_only":
                raise ValueError(
                    "rl.value_loss='winner_ce' requires env.reward_mode='win_only'"
                )
            if self.model.critic_mode != "softmax":
                raise ValueError(
                    "rl.value_loss='winner_ce' requires model.critic_mode='softmax'"
                )
            if self.rl.gamma != 1.0:
                raise ValueError("rl.value_loss='winner_ce' requires rl.gamma=1.0")
            if self.rl.vf_clip_coef is not None:
                raise ValueError(
                    "rl.value_loss='winner_ce' requires rl.vf_clip_coef=null; value "
                    "clipping has no cross-entropy analogue"
                )
        divisor = self.rl.segments_per_minibatch * self.rl.gradient_accumulation_steps
        if self.env.n_envs % divisor != 0:
            raise ValueError(
                "env.n_envs must be divisible by rl.segments_per_minibatch * "
                "rl.gradient_accumulation_steps"
            )
        if self.rl.eval_replay_games > self.env.n_envs:
            raise ValueError("rl.eval_replay_games must be <= env.n_envs")
        if self.model.critic_mode == "independent" and self.rl.teacher_value_coef > 0.0:
            raise ValueError(
                "model.critic_mode='independent' is incompatible with the "
                "winner-probability value distillation; set rl.teacher_value_coef=0"
            )
        return self

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"env", "model", "optimizer", "rl"}
