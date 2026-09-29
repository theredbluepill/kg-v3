"""Task 3.4: Kaggriculture configs follow Isaiah's scaling_6m recipe.

The ranked configs apply Isaiah's multi-GPU rule (``winner_ce_6m_4x5090.yaml``):
per-rank ``n_envs`` and ``segments_per_minibatch`` are divided by the world
size, everything else is scaling_6m's. Until Task 3.1 registers the
Kaggriculture env and model in ``FullConfig``, the observation, action, model,
optimizer and PPO sections are validated against their own schemas here; the
env keys and reward-shaping values have no schema yet and receive only exact
key and value assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from owl.kaggriculture import types as kt
from owl.model import create_model
from owl.model import kaggriculture as km
from owl.model.kaggriculture_workload import (
    ForwardWorkload,
    check_workload_headroom,
    headroom_log_lines,
    ppo_forward_workloads,
)
from owl.train import FullConfig, OptimizerConfig, PPOConfig
from owl.train.ppo import _minibatch_indices
from pydantic import TypeAdapter

ROOT = Path(__file__).parents[2]
_RANKED = {"kaggriculture_2rank.yaml": 2, "kaggriculture_4rank.yaml": 4}
_ALL = (*_RANKED, "kaggriculture.yaml")
_TASK_3_1 = (
    "Task 3.1 registers the Kaggriculture env and model in FullConfig/ModelConfig "
    "and create_model; until then these configs cannot load through FullConfig"
)
_ENV_KEYS = {
    "n_envs",
    "obs_spec",
    "action_spec",
    "reward_mode",
    "reward_shaping",
    "pin_memory",
    "native_threads",
}
_REWARD_SHAPING = {
    "econ_shaping": 0.2,
    "econ_starvation_weight": 4.0,
    "econ_drought_weight": 1.0,
    "econ_cap": 0.25,
    "econ_ineffective_weight": 0.0,
}


@dataclass(frozen=True)
class _Sections:
    env: dict[str, Any]
    model: km.KaggricultureTransformerConfig
    optimizer: OptimizerConfig
    rl: PPOConfig


@dataclass(frozen=True)
class _GlobalWorkload:
    global_envs: int
    optimizer_steps_per_iteration: int
    global_segments_per_step: int
    transitions_per_iteration: int


def _sections(name: str) -> _Sections:
    with (ROOT / "configs" / name).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert set(data) == {"env", "model", "optimizer", "rl"}
    env = data["env"]
    kt.KaggricultureObsConfig.model_validate(env["obs_spec"])
    kt.KaggricultureActionConfig.model_validate(env["action_spec"])
    return _Sections(
        env=env,
        model=km.KaggricultureTransformerConfig.from_file(
            ROOT / "configs" / "model" / f"{data['model']}.yaml"
        ),
        optimizer=TypeAdapter(OptimizerConfig).validate_python(data["optimizer"]),
        rl=PPOConfig.model_validate(data["rl"]),
    )


def _scaling_6m() -> FullConfig:
    return FullConfig.from_file(ROOT / "configs" / "scaling_6m.yaml")


def _global_workload(n_envs: int, rl: PPOConfig, world_size: int) -> _GlobalWorkload:
    per_step = rl.segments_per_minibatch * rl.gradient_accumulation_steps
    minibatches = _minibatch_indices(
        config=rl, n_segments=n_envs, device=torch.device("cpu")
    )
    assert len(minibatches) % rl.gradient_accumulation_steps == 0
    return _GlobalWorkload(
        global_envs=n_envs * world_size,
        optimizer_steps_per_iteration=len(minibatches)
        // rl.gradient_accumulation_steps,
        global_segments_per_step=per_step * world_size,
        transitions_per_iteration=rl.horizon * n_envs * world_size,
    )


@pytest.mark.parametrize(("name", "world_size"), _RANKED.items())
def test_ranked_config_global_workload_equals_scaling_6m(
    name: str, world_size: int
) -> None:
    scaling = _scaling_6m()
    ours = _sections(name)
    expected = _global_workload(scaling.env.n_envs, scaling.rl, 1)
    assert expected == _GlobalWorkload(256, 16, 16, 16_384)
    assert _global_workload(ours.env["n_envs"], ours.rl, world_size) == expected


@pytest.mark.parametrize("name", _RANKED)
def test_ranked_config_optimizer_and_ppo_equal_scaling_6m(name: str) -> None:
    scaling = _scaling_6m()
    ours = _sections(name)
    assert ours.optimizer == scaling.optimizer
    # Only the per-rank minibatch differs; target_kl, teacher, compile, dtype
    # and checkpoint cadence are scaling_6m's.
    assert ours.rl.gradient_accumulation_steps == 1
    assert ours.rl.target_kl is None
    assert (
        ours.rl.model_copy(
            update={"segments_per_minibatch": scaling.rl.segments_per_minibatch}
        )
        == scaling.rl
    )


def test_ranked_configs_differ_only_in_per_rank_shapes() -> None:
    two, four = (_sections(name) for name in _RANKED)
    assert (two.env["n_envs"], two.rl.segments_per_minibatch) == (128, 8)
    assert (four.env["n_envs"], four.rl.segments_per_minibatch) == (64, 4)
    assert {**two.env, "n_envs": 0} == {**four.env, "n_envs": 0}
    assert two.model == four.model
    assert two.model.force_flash_attn
    assert two.rl.model_copy(
        update={"segments_per_minibatch": 0}
    ) == four.rl.model_copy(update={"segments_per_minibatch": 0})


@pytest.mark.parametrize("name", _ALL)
def test_config_env_and_cross_section_rules(name: str) -> None:
    ours = _sections(name)
    assert set(ours.env) == _ENV_KEYS
    assert ours.env["reward_mode"] == "win_loss"
    assert ours.env["reward_shaping"] == _REWARD_SHAPING
    assert ours.rl.gamma == 1.0
    divisor = ours.rl.segments_per_minibatch * ours.rl.gradient_accumulation_steps
    assert ours.env["n_envs"] % divisor == 0
    assert ours.rl.eval_replay_games <= ours.env["n_envs"]


def test_local_config_is_the_recipe_on_a_tiny_cpu_model() -> None:
    scaling = _scaling_6m()
    ours = _sections("kaggriculture.yaml")
    assert ours.optimizer == scaling.optimizer
    assert not ours.model.force_flash_attn
    assert ours.rl.model_compile == "none"
    assert ours.rl.dtype == "float32"
    cpu_only = {
        "segments_per_minibatch",
        "checkpoint_freq",
        "eval_replay_games",
        "compile_mode",
        "model_compile",
        "dtype",
    }
    shared = set(PPOConfig.model_fields) - cpu_only
    assert {k: getattr(ours.rl, k) for k in shared} == {
        k: getattr(scaling.rl, k) for k in shared
    }


# --- startup workload assertion (plan Task 3.4, GEMM-limit audit) -------------


def _headroom(name: str) -> dict[str, tuple[int, int, int]]:
    ours = _sections(name)
    reports = check_workload_headroom(
        ours.model,
        ppo_forward_workloads(
            n_envs=ours.env["n_envs"],
            horizon=ours.rl.horizon,
            segments_per_minibatch=ours.rl.segments_per_minibatch,
            teacher_segments_per_minibatch=ours.rl.teacher_segments_per_minibatch,
        ),
    )
    for report in reports:
        assert report.tokens_per_row == 709
        assert report.trunk_rows_per_call == 5_915
        assert report.head_rows_per_call == 11_096
    return {r.name: (r.rows, r.max_trunk_calls, r.head_calls) for r in reports}


def test_two_rank_workloads_fit_the_model_chunking() -> None:
    # teacher_chunk = min(128, 128) x 64 x 2 rows: at most three trunk calls
    # (exactly three when every row is fully padded) and two head calls.
    assert _headroom("kaggriculture_2rank.yaml") == {
        "rollout": (256, 1, 1),
        "minibatch": (1_024, 1, 1),
        "teacher_chunk": (16_384, 3, 2),
        "evaluation": (256, 1, 1),
    }


def test_four_rank_workloads_fit_the_model_chunking() -> None:
    # teacher_chunk = min(128, 64) x 64 x 2 rows.
    assert _headroom("kaggriculture_4rank.yaml") == {
        "rollout": (128, 1, 1),
        "minibatch": (512, 1, 1),
        "teacher_chunk": (8_192, 2, 1),
        "evaluation": (128, 1, 1),
    }


def test_headroom_lines_label_the_padded_trunk_bounds() -> None:
    ours = _sections("kaggriculture_2rank.yaml")
    reports = check_workload_headroom(
        ours.model, (ForwardWorkload("rollout", 256), ForwardWorkload("bc", 16_384))
    )
    assert headroom_log_lines(reports) == (
        "GEMM workload headroom rollout: 256 rows x 709 padded tokens; trunk 5915 "
        "rows/call at full padding (>= 23.11x headroom, <= 1 call(s)); heads "
        "11096 rows/call (43.34x headroom, 1 call(s))",
        "GEMM workload headroom bc: 16384 rows x 709 padded tokens; trunk 5915 "
        "rows/call at full padding (>= 0.361x headroom, <= 3 call(s)); heads "
        "11096 rows/call (0.6772x headroom, 2 call(s))",
    )


def test_packed_trunk_calls_never_exceed_the_padded_bound() -> None:
    # Codex's counterexample: 16,384 rows of 300 packed tokens need 2 packed
    # chunks where full padding plans 3, so padded trunk calls are an upper
    # bound and the padded headroom a lower bound.
    config = km.KaggricultureTransformerConfig()
    (report,) = check_workload_headroom(config, (ForwardWorkload("teacher", 16_384),))
    width = km.trunk_gemm_width(config)
    for tokens in (1, 300, report.tokens_per_row):
        packed = km.packed_row_chunks([tokens] * report.rows, width=width)
        assert len(packed) <= report.max_trunk_calls
    assert len(km.packed_row_chunks([300] * report.rows, width=width)) == 2
    assert report.max_trunk_calls == 3


def test_no_teacher_omits_the_teacher_workload() -> None:
    names = [
        w.name
        for w in ppo_forward_workloads(
            n_envs=4,
            horizon=8,
            segments_per_minibatch=2,
            teacher_segments_per_minibatch=None,
        )
    ]
    assert names == ["rollout", "minibatch", "evaluation"]


def test_workload_check_fails_fast_when_one_row_cannot_fit() -> None:
    # Trunk width 2 * 2**21 x 709 padded tokens exceeds 2**31 for a single row.
    wide = km.KaggricultureTransformerConfig(embed_dim=2**21, n_heads=2**16)
    with pytest.raises(ValueError, match="one padded row of 709 tokens"):
        check_workload_headroom(wide, (ForwardWorkload("rollout", 2),))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_envs": 0}, "n_envs must be >= 1"),
        ({"horizon": 0}, "horizon must be >= 1"),
        ({"segments_per_minibatch": 0}, "segments_per_minibatch must be >= 1"),
        ({"teacher_segments_per_minibatch": 0}, "teacher_segments_per_minibatch"),
    ],
)
def test_workloads_reject_empty_shapes(kwargs: dict[str, int], match: str) -> None:
    shapes: dict[str, Any] = {
        "n_envs": 2,
        "horizon": 4,
        "segments_per_minibatch": 1,
        "teacher_segments_per_minibatch": 1,
    }
    with pytest.raises(ValueError, match=match):
        ppo_forward_workloads(**(shapes | kwargs))


def test_workload_check_rejects_empty_input() -> None:
    config = km.KaggricultureTransformerConfig()
    with pytest.raises(ValueError, match="no workloads"):
        check_workload_headroom(config, ())
    with pytest.raises(ValueError, match="'bc' has 0 rows"):
        check_workload_headroom(config, (ForwardWorkload("bc", 0),))


@pytest.mark.skip(reason=_TASK_3_1)
@pytest.mark.parametrize("name", _ALL)
def test_configs_load_through_full_config_and_build_the_model(name: str) -> None:
    cfg = FullConfig.from_file(ROOT / "configs" / name)
    model = create_model(
        cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
    )
    assert isinstance(model, km.KaggricultureTransformer)
