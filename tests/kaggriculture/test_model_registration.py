"""Task 3.1 (model side): Kaggriculture in the shared model config and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from owl.kaggriculture import types as kt
from owl.model import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
    ModelConfig,
    StatelessTransformerV1Config,
    create_model,
)
from owl.rl import ActionPureConfig, EntityBasedConfig
from owl.train.config import FullConfig
from pydantic import TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parents[2]
PRESET = ROOT / "configs" / "model" / "kaggriculture.yaml"
# Encoder 5,312,768 + critic 66,049 + actor projection and heads 873,406
# (docs/kaggriculture-model.md).
PRESET_PARAMETERS = 6_252_223


def _preset_data() -> dict[str, Any]:
    with PRESET.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def test_preset_round_trips_through_the_shared_model_config_union() -> None:
    adapter = TypeAdapter(ModelConfig)
    config = adapter.validate_python(_preset_data())

    assert isinstance(config, KaggricultureTransformerConfig)
    assert config == KaggricultureTransformerConfig.from_file(PRESET)
    assert config.model_arch == "kaggriculture_transformer"
    assert (config.embed_dim, config.depth, config.n_heads) == (256, 8, 8)
    assert config.force_flash_attn
    dumped = adapter.dump_python(config)
    assert dumped == config.model_dump()
    assert adapter.validate_python(dumped) == config
    assert adapter.validate_json(adapter.dump_json(config)) == config


def test_union_rejects_unknown_kaggriculture_fields() -> None:
    with pytest.raises(ValidationError, match="value_mode"):
        TypeAdapter(ModelConfig).validate_python(
            _preset_data() | {"value_mode": "win_loss"}
        )


def test_factory_builds_the_preset_kaggriculture_transformer() -> None:
    config = KaggricultureTransformerConfig.from_file(PRESET)
    action_spec = kt.KaggricultureActionConfig(hire_limit=7)
    with torch.device("meta"):
        model = create_model(
            config, obs_spec=kt.KaggricultureObsConfig(), action_spec=action_spec
        )

    assert type(model) is KaggricultureTransformer
    assert model.config == config
    assert model.action_spec == action_spec
    assert len(model.blocks) == config.depth
    assert sum(p.numel() for p in model.parameters()) == PRESET_PARAMETERS


def test_factory_dispatches_a_union_typed_config() -> None:
    config: ModelConfig = TypeAdapter(ModelConfig).validate_python(
        {"model_arch": "kaggriculture_transformer", "embed_dim": 16, "depth": 1}
        | {"n_heads": 2, "n_scratch_tokens": 1}
    )
    model = create_model(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    )
    assert type(model) is KaggricultureTransformer
    assert model.config == config


def test_factory_rejects_orbit_specs_for_kaggriculture() -> None:
    config = KaggricultureTransformerConfig(embed_dim=16, depth=1, n_heads=2)
    with pytest.raises(TypeError, match="requires Kaggriculture obs/action specs"):
        create_model(
            config,  # type: ignore[call-overload]
            obs_spec=EntityBasedConfig(),
            action_spec=kt.KaggricultureActionConfig(),
        )
    with pytest.raises(TypeError, match="requires Kaggriculture obs/action specs"):
        create_model(
            config,  # type: ignore[call-overload]
            obs_spec=kt.KaggricultureObsConfig(),
            action_spec=ActionPureConfig(),
        )


def test_factory_rejects_kaggriculture_specs_for_orbit_models() -> None:
    config = StatelessTransformerV1Config(embed_dim=16, depth=1, n_heads=2)
    with pytest.raises(TypeError, match="requires Orbit obs/action specs"):
        create_model(
            config,  # type: ignore[call-overload]
            obs_spec=kt.KaggricultureObsConfig(),
            action_spec=ActionPureConfig(),
        )


def test_orbit_full_config_rejects_the_kaggriculture_model() -> None:
    with pytest.raises(ValidationError, match="requires a Kaggriculture env config"):
        FullConfig.model_validate(
            {
                "env": {},
                "model": _preset_data(),
                "optimizer": {"optimizer": "adamw"},
                "rl": {},
            }
        )
