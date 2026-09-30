"""Tiny packaged-agent fixtures for the Task 7.4 Kaggle agent tests."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from owl import OWL_ROOT
from owl.kaggriculture import types as kt
from owl.model import kaggriculture as km

REPO_ROOT = OWL_ROOT.parents[1]
TINY_MODEL = {
    "model_arch": "kaggriculture_transformer",
    "embed_dim": 16,
    "depth": 1,
    "n_heads": 2,
    "mlp_ratio": 2.0,
    "activation": "gelu",
    "n_scratch_tokens": 1,
    # The BC checkpoint's training config sets this; the agent overrides it.
    "force_flash_attn": True,
}
TRAINING_CONFIG: dict[str, Any] = {
    "env": {
        "action_spec": {"action_spec": "kaggriculture", "hire_limit": 241},
        "n_envs": 256,
        "native_threads": 2,
        "obs_spec": {"obs_spec": "kaggriculture", "schema_version": 3},
        "pin_memory": True,
        "reward_mode": "win_loss",
        "reward_shaping": {
            "econ_cap": 0.25,
            "econ_drought_weight": 1.0,
            "econ_ineffective_cap": 0.1,
            "econ_ineffective_weight": 0.0,
            "econ_shaping": 0.2,
            "econ_starvation_weight": 4.0,
        },
    },
    "model": TINY_MODEL,
    "rl": {"horizon": 64},
}


def tiny_state(*, hire_limit: int = 241, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = km.KaggricultureTransformer(
        km.KaggricultureTransformerConfig(**TINY_MODEL | {"force_flash_attn": False}),
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(hire_limit=hire_limit),
    )
    return model.state_dict()


def write_model_root(
    root: Path,
    *,
    config: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> Path:
    """Write ``config.yaml`` and a slim ``checkpoint.pt`` like the tarball's."""
    root.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(TRAINING_CONFIG) if config is None else config
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    torch.save(
        {"model": tiny_state()} if checkpoint is None else checkpoint,
        root / "checkpoint.pt",
    )
    return root


def write_agent_dir(root: Path) -> Path:
    """The extracted tarball layout: main.py, owl/ (with the dev extension), models/."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "python" / "kaggriculture_main.py", root / "main.py")
    shutil.copytree(
        OWL_ROOT, root / "owl", ignore=shutil.ignore_patterns("__pycache__")
    )
    write_model_root(root / "models" / "primary")
    return root
