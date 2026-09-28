from __future__ import annotations

from typing import Any, assert_never

from owl.kaggriculture.types import KaggricultureActionConfig, KaggricultureObsConfig
from owl.model.base import BaseModelAPI
from owl.model.config import ModelConfig
from owl.model.kaggriculture import KaggricultureTransformer
from owl.model.recurrent_transformer_v1 import RecurrentTransformerV1
from owl.model.stateless_transformer_v1 import StatelessTransformerV1
from owl.rl import SupportedActionConfig, SupportedObsConfig


def create_model(
    config: ModelConfig,
    *,
    obs_spec: SupportedObsConfig,
    action_spec: SupportedActionConfig,
) -> BaseModelAPI[Any, Any]:
    if config.model_arch == "kaggriculture_transformer":
        if not isinstance(obs_spec, KaggricultureObsConfig) or not isinstance(
            action_spec, KaggricultureActionConfig
        ):
            raise ValueError(
                "Kaggriculture model requires Kaggriculture observation/action specs"
            )
        return KaggricultureTransformer(
            config, obs_spec=obs_spec, action_spec=action_spec
        )
    if isinstance(obs_spec, KaggricultureObsConfig) or isinstance(
        action_spec, KaggricultureActionConfig
    ):
        raise ValueError(
            "Orbit models cannot consume Kaggriculture observations/actions"
        )
    match config.model_arch:
        case "stateless_transformer_v1":
            return StatelessTransformerV1(
                config,
                obs_spec=obs_spec,
                action_spec=action_spec,
            )
        case "recurrent_transformer_v1":
            return RecurrentTransformerV1(
                config,
                obs_spec=obs_spec,
                action_spec=action_spec,
            )
        case _:
            assert_never(config)
