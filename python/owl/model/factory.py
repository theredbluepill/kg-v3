from __future__ import annotations

from typing import Any, assert_never, overload

from owl.kaggriculture.types import KaggricultureActionConfig, KaggricultureObsConfig
from owl.model.base import BaseModelAPI
from owl.model.config import ModelConfig, OrbitModelConfig
from owl.model.kaggriculture import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
)
from owl.model.recurrent_transformer_v1 import (
    RecurrentTransformerV1,
    RecurrentTransformerV1Config,
)
from owl.model.stateless_transformer_v1 import (
    StatelessTransformerV1,
    StatelessTransformerV1Config,
)
from owl.rl import ActionConfig, ObsConfig


@overload
def create_model(
    config: OrbitModelConfig,
    *,
    obs_spec: ObsConfig,
    action_spec: ActionConfig,
) -> BaseModelAPI: ...


@overload
def create_model(
    config: KaggricultureTransformerConfig,
    *,
    obs_spec: KaggricultureObsConfig,
    action_spec: KaggricultureActionConfig,
) -> KaggricultureTransformer: ...


@overload
def create_model(
    config: ModelConfig,
    *,
    obs_spec: ObsConfig | KaggricultureObsConfig,
    action_spec: ActionConfig | KaggricultureActionConfig,
) -> BaseModelAPI[Any, Any, Any]: ...


def create_model(
    config: ModelConfig,
    *,
    obs_spec: ObsConfig | KaggricultureObsConfig,
    action_spec: ActionConfig | KaggricultureActionConfig,
) -> BaseModelAPI[Any, Any, Any]:
    """Build the model for ``config`` after checking its game's specs.

    Each architecture belongs to one game; a spec from the other game raises
    instead of building a model whose tensors cannot match the environment.
    """
    match config:
        case StatelessTransformerV1Config():
            orbit_obs, orbit_action = _require_orbit_specs(
                config, obs_spec, action_spec
            )
            return StatelessTransformerV1(
                config,
                obs_spec=orbit_obs,
                action_spec=orbit_action,
            )
        case RecurrentTransformerV1Config():
            orbit_obs, orbit_action = _require_orbit_specs(
                config, obs_spec, action_spec
            )
            return RecurrentTransformerV1(
                config,
                obs_spec=orbit_obs,
                action_spec=orbit_action,
            )
        case KaggricultureTransformerConfig():
            if not isinstance(obs_spec, KaggricultureObsConfig) or not isinstance(
                action_spec, KaggricultureActionConfig
            ):
                raise TypeError(
                    f"model_arch={config.model_arch!r} requires Kaggriculture "
                    f"obs/action specs, got {_spec_names(obs_spec, action_spec)}"
                )
            return KaggricultureTransformer(
                config,
                obs_spec=obs_spec,
                action_spec=action_spec,
            )
        case _:
            assert_never(config)


def _require_orbit_specs(
    config: OrbitModelConfig,
    obs_spec: ObsConfig | KaggricultureObsConfig,
    action_spec: ActionConfig | KaggricultureActionConfig,
) -> tuple[ObsConfig, ActionConfig]:
    if isinstance(obs_spec, KaggricultureObsConfig) or isinstance(
        action_spec, KaggricultureActionConfig
    ):
        raise TypeError(
            f"model_arch={config.model_arch!r} requires Orbit obs/action specs, "
            f"got {_spec_names(obs_spec, action_spec)}"
        )
    return obs_spec, action_spec


def _spec_names(
    obs_spec: ObsConfig | KaggricultureObsConfig,
    action_spec: ActionConfig | KaggricultureActionConfig,
) -> str:
    return f"{type(obs_spec).__name__}/{type(action_spec).__name__}"
