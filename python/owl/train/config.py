from typing import Annotated, Any, Self, TypeAlias

from pydantic import Discriminator, Field, Tag, model_validator

from owl.config import BaseConfig
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.types import KaggricultureObsConfig
from owl.model import ActorDiscreteTargetBinsConfig, ModelConfig
from owl.model.config import OrbitModelConfig
from owl.model.kaggriculture import KaggricultureTransformerConfig
from owl.rl import ActionDiscreteTargetBinsConfig, EnvConfig, RewardMode

from .optimizer import OptimizerConfig
from .ppo import PPOConfig

_KAGGRICULTURE_OBS_TAG = KaggricultureObsConfig.model_fields["obs_spec"].default


def _env_game(value: Any) -> str:
    """Select the env schema by its observation tag; Orbit when it is absent.

    Isaiah's ``EnvConfig`` has no game tag and defaults its observation spec,
    so only an explicit Kaggriculture observation selects the Kaggriculture env.
    """
    if isinstance(value, KaggricultureEnvConfig):
        return "kaggriculture"
    if isinstance(value, EnvConfig):
        return "orbit"
    if isinstance(value, dict):
        obs_spec = value.get("obs_spec")
        if isinstance(obs_spec, KaggricultureObsConfig) or (
            isinstance(obs_spec, dict)
            and obs_spec.get("obs_spec") == _KAGGRICULTURE_OBS_TAG
        ):
            return "kaggriculture"
    return "orbit"


GameEnvConfig: TypeAlias = Annotated[
    Annotated[EnvConfig, Tag("orbit")]
    | Annotated[KaggricultureEnvConfig, Tag("kaggriculture")],
    Discriminator(_env_game),
]


def require_orbit_env(
    env: EnvConfig | KaggricultureEnvConfig, *, context: str
) -> EnvConfig:
    """Narrow ``env`` to Isaiah's Orbit ``EnvConfig``; fail fast for Kaggriculture.

    Kaggriculture configs load, validate and pass run_ppo's workload check, but
    ``context`` builds Orbit environments only: there is no Kaggriculture
    environment until the native env binding (plan Task 1.4) and the trainer
    game seam (Task 3.1) land.
    """
    if isinstance(env, KaggricultureEnvConfig):
        raise RuntimeError(
            f"{context} cannot run Kaggriculture yet: its config loads, but there "
            "is no Kaggriculture environment until the native env binding (plan "
            "Task 1.4) and the trainer game seam (Task 3.1) land"
        )
    return env


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
    env: GameEnvConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    rl: PPOConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _validate_cross_config_constraints(self) -> Self:
        env, model = self.env, self.model
        if isinstance(env, KaggricultureEnvConfig):
            if not isinstance(model, KaggricultureTransformerConfig):
                raise ValueError(
                    "Kaggriculture env requires "
                    "model.model_arch='kaggriculture_transformer', got "
                    f"{model.model_arch!r}"
                )
            _validate_kaggriculture_rl(self.rl)
            _validate_kaggriculture_training(reward_mode=env.reward_mode, rl=self.rl)
        elif isinstance(model, KaggricultureTransformerConfig):
            raise ValueError(
                "model.model_arch='kaggriculture_transformer' requires a "
                "Kaggriculture env config (env.obs_spec.obs_spec='kaggriculture'), got "
                f"env.obs_spec.obs_spec={env.obs_spec.obs_spec!r}"
            )
        else:
            _validate_orbit_constraints(env, model, self.rl)
        divisor = self.rl.segments_per_minibatch * self.rl.gradient_accumulation_steps
        if self.env.n_envs % divisor != 0:
            raise ValueError(
                "env.n_envs must be divisible by rl.segments_per_minibatch * "
                "rl.gradient_accumulation_steps"
            )
        if self.rl.eval_replay_games > self.env.n_envs:
            raise ValueError("rl.eval_replay_games must be <= env.n_envs")
        return self

    @classmethod
    def subconfig_dirs(cls) -> set[str]:
        return {"env", "model", "optimizer", "rl"}


def _validate_kaggriculture_rl(rl: PPOConfig) -> None:
    # Contract v4 (Environment, Rewards): the capped economic reward plus the
    # terminal win/loss term bounds the complete-episode return by 1 only
    # undiscounted, and it is not the sum-to-one win_only target winner_ce needs.
    if rl.gamma != 1.0:
        raise ValueError(
            f"the Kaggriculture reward requires rl.gamma=1.0, got {rl.gamma}"
        )
    if rl.value_loss == "winner_ce":
        raise ValueError(
            "rl.value_loss='winner_ce' requires env.reward_mode='win_only', which "
            "the Kaggriculture reward does not provide"
        )


def _validate_orbit_constraints(
    env: EnvConfig, model: OrbitModelConfig, rl: PPOConfig
) -> None:
    if model.actor.action_spec != env.action_spec.action_spec:
        raise ValueError("model actor action_spec must match env action_spec")
    if (
        isinstance(model.actor, ActorDiscreteTargetBinsConfig)
        and isinstance(env.action_spec, ActionDiscreteTargetBinsConfig)
        and model.actor.n_bins != env.action_spec.n_bins
    ):
        raise ValueError("model actor n_bins must match env action_spec n_bins")
    if env.reward_mode == "win_only" and model.value_mode != "win_only":
        raise ValueError(
            "env.reward_mode='win_only' requires model.value_mode='win_only'"
        )
    if env.reward_mode != "win_only" and model.value_mode == "win_only":
        raise ValueError(
            "model.value_mode='win_only' requires env.reward_mode='win_only'"
        )
    if rl.value_loss == "winner_ce":
        # The cross-entropy value loss trains the winner-probability softmax
        # directly against a distributional winner target. That target is a
        # valid probability distribution only with the undiscounted, sum-to-one
        # win_only reward (value_mode='win_only' follows from it above), and the
        # softmax critic; value clipping has no cross-entropy analogue.
        if env.reward_mode != "win_only":
            raise ValueError(
                "rl.value_loss='winner_ce' requires env.reward_mode='win_only'"
            )
        if model.critic_mode != "softmax":
            raise ValueError(
                "rl.value_loss='winner_ce' requires model.critic_mode='softmax'"
            )
        if rl.gamma != 1.0:
            raise ValueError("rl.value_loss='winner_ce' requires rl.gamma=1.0")
        if rl.vf_clip_coef is not None:
            raise ValueError(
                "rl.value_loss='winner_ce' requires rl.vf_clip_coef=null; value "
                "clipping has no cross-entropy analogue"
            )
    if model.critic_mode == "independent" and rl.teacher_value_coef > 0.0:
        raise ValueError(
            "model.critic_mode='independent' is incompatible with the "
            "winner-probability value distillation; set rl.teacher_value_coef=0"
        )
