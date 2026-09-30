"""Task 3.4: Kaggriculture configs follow Isaiah's scaling_6m recipe.

The ranked configs (2, 4 and 8 ranks) apply Isaiah's 6M multi-GPU
division (``winner_ce_6m_4x5090.yaml``): per-rank ``n_envs`` and
``segments_per_minibatch`` are divided by the world size, everything else is
scaling_6m's. Every config loads through the real
``FullConfig.from_file``, so each section, the Kaggriculture env and reward
schema and the cross-section rules are validated by the trainer's own loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.model import create_model
from owl.model import kaggriculture as km
from owl.model.kaggriculture_teacher import TEACHER_TARGET_BYTES_PER_ROW
from owl.model.kaggriculture_workload import (
    ForwardWorkload,
    check_workload_headroom,
    headroom_log_lines,
    ppo_forward_workloads,
)
from owl.train import (
    FullConfig,
    NoTeacherScheduleConfig,
    OptimizerConfig,
    PPOConfig,
)
from owl.train.ppo import _minibatch_indices
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
_RANKED = {
    "kaggriculture_2rank.yaml": 2,
    "kaggriculture_4rank.yaml": 4,
    "kaggriculture_8rank.yaml": 8,
}
# Recipe J's BC fine-tune presets and the ranked config each one copies.
_FINETUNE = {
    "kaggriculture_2rank_bc_finetune.yaml": "kaggriculture_2rank.yaml",
    "kaggriculture_8rank_bc_finetune.yaml": "kaggriculture_8rank.yaml",
}
_RANKED_AND_FINETUNE = {
    **_RANKED,
    **{name: _RANKED[base] for name, base in _FINETUNE.items()},
}
_ALL = (*_RANKED, *_FINETUNE, "kaggriculture.yaml")
_REWARD_SHAPING = KaggricultureRewardConfig(
    econ_shaping=0.2,
    econ_starvation_weight=4.0,
    econ_drought_weight=1.0,
    econ_cap=0.25,
    econ_ineffective_weight=0.0,
    econ_ineffective_cap=0.1,
)


@dataclass(frozen=True)
class _Sections:
    env: KaggricultureEnvConfig
    model: km.KaggricultureTransformerConfig
    optimizer: OptimizerConfig
    rl: PPOConfig


@dataclass(frozen=True)
class _PerRankShape:
    n_envs: int
    segments_per_minibatch: int
    teacher_chunk_segments: int


@dataclass(frozen=True)
class _GlobalWorkload:
    global_envs: int
    optimizer_steps_per_iteration: int
    global_segments_per_step: int
    transitions_per_iteration: int


def _sections(name: str) -> _Sections:
    cfg = FullConfig.from_file(ROOT / "configs" / name)
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    assert isinstance(cfg.model, km.KaggricultureTransformerConfig)
    return _Sections(env=cfg.env, model=cfg.model, optimizer=cfg.optimizer, rl=cfg.rl)


def _scaling_6m() -> FullConfig:
    return FullConfig.from_file(ROOT / "configs" / "scaling_6m.yaml")


def _isaiah_per_rank_shape(scaling: FullConfig, world_size: int) -> _PerRankShape:
    """Isaiah's multi-GPU rule: divide the global shapes by the world size.

    Fails loudly when a global quantity does not divide into whole per-rank
    shapes, or when the per-rank envs do not split into whole minibatches or
    whole teacher-precompute chunks.
    """
    rl = scaling.rl
    for name, value in (
        ("n_envs", scaling.env.n_envs),
        ("segments_per_minibatch", rl.segments_per_minibatch),
    ):
        if value % world_size != 0:
            raise ValueError(
                f"scaling_6m {name}={value} is not divisible by world size {world_size}"
            )
    n_envs = scaling.env.n_envs // world_size
    spm = rl.segments_per_minibatch // world_size
    per_step = spm * rl.gradient_accumulation_steps
    if n_envs % per_step != 0:
        raise ValueError(
            f"per-rank n_envs={n_envs} is not divisible by segments_per_minibatch "
            f"* gradient_accumulation_steps={per_step}"
        )
    # The teacher chunk is a precompute slice, kept at Isaiah's value (his
    # per-rank configs keep it) and clamped to the rank's envs by the trainer.
    teacher_chunk = min(rl.teacher_segments_per_minibatch, n_envs)
    if n_envs % teacher_chunk != 0:
        raise ValueError(
            f"per-rank n_envs={n_envs} is not divisible by the teacher chunk "
            f"min(teacher_segments_per_minibatch, n_envs)={teacher_chunk}"
        )
    return _PerRankShape(n_envs, spm, teacher_chunk)


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


@pytest.mark.parametrize(("name", "world_size"), _RANKED_AND_FINETUNE.items())
def test_ranked_config_global_workload_equals_scaling_6m(
    name: str, world_size: int
) -> None:
    scaling = _scaling_6m()
    ours = _sections(name)
    expected = _global_workload(scaling.env.n_envs, scaling.rl, 1)
    assert expected == _GlobalWorkload(256, 16, 16, 16_384)
    assert _global_workload(ours.env.n_envs, ours.rl, world_size) == expected


@pytest.mark.parametrize(("name", "world_size"), _RANKED_AND_FINETUNE.items())
def test_ranked_config_per_rank_shapes_are_scaling_6m_divided(
    name: str, world_size: int
) -> None:
    ours = _sections(name)
    teacher_chunk = min(ours.rl.teacher_segments_per_minibatch, ours.env.n_envs)
    assert _PerRankShape(
        ours.env.n_envs, ours.rl.segments_per_minibatch, teacher_chunk
    ) == _isaiah_per_rank_shape(_scaling_6m(), world_size)
    assert ours.rl.teacher_segments_per_minibatch == 128


@pytest.mark.parametrize(
    ("world_size", "match"),
    [
        (3, "n_envs=256 is not divisible by world size 3"),
        (32, "segments_per_minibatch=16 is not divisible by world size 32"),
    ],
)
def test_isaiah_per_rank_shape_fails_loudly_when_not_divisible(
    world_size: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _isaiah_per_rank_shape(_scaling_6m(), world_size)


def test_isaiah_per_rank_shape_rejects_partial_minibatches_and_teacher_chunks() -> None:
    scaling = _scaling_6m()
    accum = scaling.model_copy(
        update={"rl": scaling.rl.model_copy(update={"gradient_accumulation_steps": 3})}
    )
    with pytest.raises(ValueError, match="segments_per_minibatch \\* gradient"):
        _isaiah_per_rank_shape(accum, 2)
    teacher = scaling.model_copy(
        update={
            "rl": scaling.rl.model_copy(update={"teacher_segments_per_minibatch": 24})
        }
    )
    with pytest.raises(ValueError, match="teacher chunk"):
        _isaiah_per_rank_shape(teacher, 8)


@pytest.mark.parametrize("name", _RANKED)
def test_ranked_config_optimizer_and_ppo_equal_scaling_6m(name: str) -> None:
    scaling = _scaling_6m()
    ours = _sections(name)
    assert ours.optimizer == scaling.optimizer
    # Only the per-rank minibatch and the owner's halved checkpoint cadence
    # differ; target_kl, teacher, compile and dtype are scaling_6m's. Replay
    # export stays off until Task 7.3 adds it for Kaggriculture (a positive
    # count fails at startup).
    assert ours.rl.gradient_accumulation_steps == 1
    assert ours.rl.target_kl is None
    assert ours.rl.eval_replay_games == 0
    assert scaling.rl.eval_replay_games == 8
    assert (
        ours.rl.model_copy(
            update={
                "segments_per_minibatch": scaling.rl.segments_per_minibatch,
                "eval_replay_games": scaling.rl.eval_replay_games,
                "checkpoint_freq": scaling.rl.checkpoint_freq,
            }
        )
        == scaling.rl
    )


# --- checkpoint and promotion cadence (owner decision 2026-09-30) ------------

# Every Kaggriculture GPU training config; the CPU config keeps its 1,000-step
# test cadence.
_GPU_CONFIGS = {**_RANKED_AND_FINETUNE, "kaggriculture_1gpu_eager.yaml": 1}
_ENV_STEPS_PER_ITERATION = 16_384


@pytest.mark.parametrize(("name", "world_size"), _GPU_CONFIGS.items())
def test_checkpoint_interval_is_the_owners_half_of_scaling_6m(
    name: str, world_size: int
) -> None:
    # Owner: "cut the interval into half". One checkpoint_freq drives both the
    # periodic checkpoint and the last-best evaluation (promotion at >= 0.7).
    ours = _sections(name)
    assert _scaling_6m().rl.checkpoint_freq == 20_000_000
    assert ours.rl.checkpoint_freq == 10_000_000
    assert ours.rl.horizon * ours.env.n_envs * world_size == _ENV_STEPS_PER_ITERATION
    # The first checkpoint and evaluation fire after 611 iterations; the
    # interval is 610 or 611 iterations after that.
    assert -(-ours.rl.checkpoint_freq // _ENV_STEPS_PER_ITERATION) == 611
    assert ours.rl.checkpoint_freq // _ENV_STEPS_PER_ITERATION == 610


# --- recipe J: BC fine-tune presets ------------------------------------------


@pytest.mark.parametrize(("name", "base"), _FINETUNE.items())
def test_finetune_preset_divides_both_learning_rates_by_ten(
    name: str, base: str
) -> None:
    ours = _sections(name).optimizer
    ranked = _sections(base).optimizer
    assert ranked == _scaling_6m().optimizer
    assert (ours.muon_lr, ours.adamw_lr) == (0.0002, 1.0e-05)
    assert ours.muon_lr == ranked.muon_lr / 10
    assert ours.adamw_lr == ranked.adamw_lr / 10


@pytest.mark.parametrize(("name", "base"), _FINETUNE.items())
def test_finetune_preset_equals_its_ranked_config_apart_from_the_lrs(
    name: str, base: str
) -> None:
    # A diff test over the whole loaded config: env, model, schedule, PPO,
    # teacher, checkpoint cadence and runtime all equal the ranked config's.
    ours = FullConfig.from_file(ROOT / "configs" / name)
    ranked = FullConfig.from_file(ROOT / "configs" / base)
    assert ours != ranked
    restored = ours.model_copy(
        update={
            "optimizer": ours.optimizer.model_copy(
                update={
                    "muon_lr": ranked.optimizer.muon_lr,
                    "adamw_lr": ranked.optimizer.adamw_lr,
                }
            )
        }
    )
    assert restored == ranked


# --- teacher settings (plan Task 4.4) -------------------------------------------

_TEACHER_FIELDS = (
    "teacher_mode",
    "teacher_init",
    "teacher_kl_coef",
    "teacher_value_coef",
    "teacher_schedule",
    "teacher_segments_per_minibatch",
)


@pytest.mark.parametrize("name", _ALL)
def test_teacher_settings_are_scaling_6ms_last_best_teacher(name: str) -> None:
    # Named explicitly, so a future rl exemption in the whole-section equality
    # tests cannot silently drop a teacher field. The chunk size stays 128 per
    # rank (Isaiah's per-rank configs keep it), and no checkpoint path is
    # configured: the teacher source is a launch-time input.
    ours = {field: getattr(_sections(name).rl, field) for field in _TEACHER_FIELDS}
    assert ours == {
        "teacher_mode": "last_best",
        "teacher_init": None,
        "teacher_kl_coef": 0.005,
        "teacher_value_coef": 0.005,
        "teacher_schedule": NoTeacherScheduleConfig(),
        "teacher_segments_per_minibatch": 128,
    }
    scaling = _scaling_6m().rl
    assert ours == {field: getattr(scaling, field) for field in _TEACHER_FIELDS}


def test_ranked_configs_differ_only_in_per_rank_shapes() -> None:
    two, four, eight = (_sections(name) for name in _RANKED)
    assert (two.env.n_envs, two.rl.segments_per_minibatch) == (128, 8)
    assert (four.env.n_envs, four.rl.segments_per_minibatch) == (64, 4)
    assert (eight.env.n_envs, eight.rl.segments_per_minibatch) == (32, 2)
    assert two.model.force_flash_attn
    assert (two.env.native_threads, two.env.pin_memory) == (2, True)
    for other in (four, eight):
        assert two.env.model_copy(update={"n_envs": 0}) == other.env.model_copy(
            update={"n_envs": 0}
        )
        assert two.model == other.model
        assert two.optimizer == other.optimizer
        assert two.rl.model_copy(
            update={"segments_per_minibatch": 0}
        ) == other.rl.model_copy(update={"segments_per_minibatch": 0})


@pytest.mark.parametrize("name", _ALL)
def test_config_env_and_cross_section_rules(name: str) -> None:
    ours = _sections(name)
    assert ours.env.obs_spec == kt.KaggricultureObsConfig(schema_version=3)
    assert ours.env.action_spec == kt.KaggricultureActionConfig(hire_limit=241)
    assert ours.env.reward_mode == "win_loss"
    assert ours.env.reward_shaping == _REWARD_SHAPING
    assert ours.env.reward_shaping.terminal_scale == 0.75
    assert ours.rl.gamma == 1.0
    divisor = ours.rl.segments_per_minibatch * ours.rl.gradient_accumulation_steps
    assert ours.env.n_envs % divisor == 0
    assert ours.rl.eval_replay_games <= ours.env.n_envs


def test_local_config_is_the_recipe_on_a_tiny_cpu_model() -> None:
    scaling = _scaling_6m()
    ours = _sections("kaggriculture.yaml")
    assert ours.optimizer == scaling.optimizer
    assert (ours.env.n_envs, ours.env.native_threads) == (2, 1)
    assert not ours.env.pin_memory
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
            n_envs=ours.env.n_envs,
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


def test_eight_rank_workloads_fit_the_model_chunking() -> None:
    # teacher_chunk = min(128, 32) x 64 x 2 rows: one trunk and one head call.
    assert _headroom("kaggriculture_8rank.yaml") == {
        "rollout": (64, 1, 1),
        "minibatch": (256, 1, 1),
        "teacher_chunk": (4_096, 1, 1),
        "evaluation": (64, 1, 1),
    }


@pytest.mark.parametrize(
    ("name", "chunk_rows", "cache_bytes"),
    [
        ("kaggriculture_2rank.yaml", 16_384, 1_674_575_872),
        ("kaggriculture_4rank.yaml", 8_192, 837_287_936),
        ("kaggriculture_8rank.yaml", 4_096, 418_643_968),
    ],
)
def test_teacher_chunk_reports_its_target_cache_bytes(
    name: str, chunk_rows: int, cache_bytes: int
) -> None:
    # One whole-rollout chunk per rank at every world size, so the chunk and
    # the cache hold the same n_envs x 64 x 2 seat rows of 102,208 B.
    ours = _sections(name)
    reports = check_workload_headroom(
        ours.model,
        ppo_forward_workloads(
            n_envs=ours.env.n_envs,
            horizon=ours.rl.horizon,
            segments_per_minibatch=ours.rl.segments_per_minibatch,
            teacher_segments_per_minibatch=ours.rl.teacher_segments_per_minibatch,
        ),
    )
    by_name = {report.name: report for report in reports}
    teacher = by_name["teacher_chunk"]
    assert TEACHER_TARGET_BYTES_PER_ROW == 102_208
    assert (teacher.rows, teacher.cached_teacher_rows) == (chunk_rows, chunk_rows)
    assert teacher.teacher_chunk_bytes == cache_bytes
    assert teacher.teacher_cache_bytes == cache_bytes
    assert teacher.log_line().endswith(
        f"; teacher targets {cache_bytes} B per chunk, {cache_bytes} B cached "
        f"for {chunk_rows} rollout rows"
    )
    for other in ("rollout", "minibatch", "evaluation"):
        assert by_name[other].teacher_cache_bytes is None
        assert "teacher targets" not in by_name[other].log_line()


def test_teacher_cache_covers_the_rollout_when_it_spans_several_chunks() -> None:
    # 8 envs in chunks of 2 segments: each call computes 2 x 4 x 2 = 16 rows,
    # and the cache holds all 8 x 4 x 2 = 64 rollout rows.
    workloads = ppo_forward_workloads(
        n_envs=8, horizon=4, segments_per_minibatch=2, teacher_segments_per_minibatch=2
    )
    (report,) = check_workload_headroom(
        km.KaggricultureTransformerConfig(),
        tuple(w for w in workloads if w.name == "teacher_chunk"),
    )
    assert (report.rows, report.cached_teacher_rows) == (16, 64)
    assert report.teacher_chunk_bytes == 16 * 102_208
    assert report.teacher_cache_bytes == 64 * 102_208


def test_workload_check_rejects_a_teacher_cache_smaller_than_its_chunk() -> None:
    with pytest.raises(ValueError, match="caches 8 teacher rows, fewer than its 16"):
        check_workload_headroom(
            km.KaggricultureTransformerConfig(),
            (ForwardWorkload("teacher_chunk", 16, cached_teacher_rows=8),),
        )


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


@pytest.mark.parametrize("name", _ALL)
def test_configs_load_through_full_config_and_build_the_model(name: str) -> None:
    cfg = FullConfig.from_file(ROOT / "configs" / name)
    model = create_model(
        cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
    )
    assert isinstance(model, km.KaggricultureTransformer)


# --- Kaggriculture env in FullConfig (verify-3.4-r2 P1) ------------------------


def _config_data(name: str) -> dict[str, Any]:
    cfg = FullConfig.from_file(ROOT / "configs" / name)
    return cfg.model_dump(mode="json", round_trip=True)


def test_config_round_trips_through_the_kaggriculture_env_schema() -> None:
    data = _config_data("kaggriculture_2rank.yaml")
    assert data["env"] == {
        "n_envs": 128,
        "seed": 0,
        "config": kt.KaggricultureGameConfig().model_dump(mode="json"),
        "obs_spec": {"obs_spec": "kaggriculture", "schema_version": 3},
        "action_spec": {"action_spec": "kaggriculture", "hire_limit": 241},
        "reward_mode": "win_loss",
        "reward_shaping": {
            "econ_shaping": 0.2,
            "econ_starvation_weight": 4.0,
            "econ_drought_weight": 1.0,
            "econ_cap": 0.25,
            "econ_ineffective_weight": 0.0,
            "econ_ineffective_cap": 0.1,
        },
        "pin_memory": True,
        "native_threads": 2,
    }
    assert FullConfig.model_validate(data) == FullConfig.from_file(
        ROOT / "configs" / "kaggriculture_2rank.yaml"
    )


def test_orbit_configs_still_load_isaiahs_env_config() -> None:
    cfg = _scaling_6m()
    assert not isinstance(cfg.env, KaggricultureEnvConfig)
    assert not isinstance(cfg.model, km.KaggricultureTransformerConfig)


def test_kaggriculture_model_requires_the_kaggriculture_env() -> None:
    data = _config_data("kaggriculture.yaml")
    data["env"] = _scaling_6m().model_dump(mode="json", round_trip=True)["env"]
    data["env"]["n_envs"] = 2
    with pytest.raises(ValidationError, match="requires a Kaggriculture env"):
        FullConfig.model_validate(data)


def test_kaggriculture_env_requires_the_kaggriculture_model() -> None:
    data = _config_data("kaggriculture.yaml")
    data["model"] = _scaling_6m().model_dump(mode="json", round_trip=True)["model"]
    with pytest.raises(ValidationError, match="Kaggriculture env requires"):
        FullConfig.model_validate(data)


def test_kaggriculture_env_rejects_orbit_specs() -> None:
    data = _config_data("kaggriculture.yaml")
    data["env"]["action_spec"] = {"action_spec": "pure"}
    with pytest.raises(ValidationError, match="action_spec"):
        FullConfig.model_validate(data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("two_player_weight", 1.0), ("reward_mode", "win_only"), ("native_threads", 0)],
)
def test_kaggriculture_env_rejects_fields_outside_its_schema(
    key: str, value: object
) -> None:
    data = _config_data("kaggriculture.yaml")
    data["env"][key] = value
    with pytest.raises(ValidationError, match=key):
        FullConfig.model_validate(data)


def test_kaggriculture_reward_rejects_the_winner_ce_value_loss() -> None:
    data = _config_data("kaggriculture.yaml")
    data["rl"]["value_loss"] = "winner_ce"
    with pytest.raises(ValidationError, match="the Kaggriculture reward does not"):
        FullConfig.model_validate(data)


def test_kaggriculture_reward_requires_gamma_one() -> None:
    data = _config_data("kaggriculture.yaml")
    data["rl"]["gamma"] = 0.99
    with pytest.raises(ValidationError, match=r"requires rl\.gamma=1\.0"):
        FullConfig.model_validate(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"econ_shaping": 0.2, "econ_cap": 0.6, "econ_ineffective_weight": 1.0,
             "econ_ineffective_cap": 0.4},
            "active economic penalty caps must sum below one",
        ),
        (
            {"econ_shaping": 0.2, "econ_starvation_weight": 0.0,
             "econ_drought_weight": 0.0},
            "economic shaping requires a positive event weight",
        ),
        ({"econ_cap": 0.0}, "econ_cap"),
        ({"econ_shaping": float("nan")}, "econ_shaping"),
    ],
)  # fmt: skip
def test_reward_shaping_rejects_an_invalid_budget(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        KaggricultureRewardConfig.model_validate(_REWARD_SHAPING.model_dump() | kwargs)


def test_reward_terminal_scale_counts_only_enabled_caps() -> None:
    assert (
        KaggricultureRewardConfig.model_validate(
            _REWARD_SHAPING.model_dump()
            | {"econ_shaping": 0.0, "econ_cap": 0.0, "econ_ineffective_cap": 0.0}
        ).terminal_scale
        == 1.0
    )
    assert _REWARD_SHAPING.terminal_scale == 0.75
    assert (
        KaggricultureRewardConfig.model_validate(
            _REWARD_SHAPING.model_dump() | {"econ_ineffective_weight": 1.0}
        ).terminal_scale
        == 1.0 - 0.25 - 0.1
    )
