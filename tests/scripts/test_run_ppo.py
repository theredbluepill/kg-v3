from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import owl.train.logging as train_logging
import pytest
import torch
from owl.checkpoint_quantization import (
    NF4_G128_LSQ,
    dequantize_model_state_dict,
    quantize_model_state_dict,
)
from owl.game import create_env
from owl.kaggriculture.codec import encode_actions
from owl.kaggriculture.config import (
    KaggricultureEnvConfig,
    KaggricultureOpponentMixConfig,
)
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.types import (
    MAX_ACTORS,
    KaggricultureActions,
    KaggricultureObsConfig,
)
from owl.model import LoRALinear
from owl.model.compile_gemm import (
    CompileStackReport,
    GemmBackendClaim,
    InstalledCompileStack,
)
from owl.rl import (
    ACTION_ENTITY_SLOTS,
    MAX_COMETS,
    MAX_PLANETS,
    ActionPureConfig,
    EntityBasedConfig,
    EntityBasedExtV1Config,
    EnvConfig,
    ObsBatch,
    PureActionMask,
)
from owl.train import FullConfig, PPOTrainer
from owl.train.distributed import DistributedContext
from owl.train.logging import (
    ATTEMPTS_FILE,
    LogMode,
    MetricLogger,
    MissingWandbCredentialsError,
    RunIdentity,
    TelemetryMode,
    WandbMode,
    WandbRunFacts,
)
from owl.train.optimizer import CompositeOptimizer
from owl.train.ppo import CHECKPOINT_KEYS, OPTIONAL_CHECKPOINT_KEYS

_RUN_PPO_PATH = Path(__file__).parents[2] / "scripts" / "run_ppo.py"
_RUN_PPO_SPEC = importlib.util.spec_from_file_location("run_ppo", _RUN_PPO_PATH)
assert _RUN_PPO_SPEC is not None
assert _RUN_PPO_SPEC.loader is not None
run_ppo = importlib.util.module_from_spec(_RUN_PPO_SPEC)
sys.modules["run_ppo"] = run_ppo
_RUN_PPO_SPEC.loader.exec_module(run_ppo)


def _full_config(*, checkpoint_freq: int | None = None) -> FullConfig:
    return FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 1,
                "n_heads": 4,
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "lr_schedule": {
                    "schedule": "linear_warmup_cosine_decay",
                    "warmup_steps": 1,
                    "decay_steps": 4,
                    "lr_min_ratio": 0.1,
                },
            },
            "rl": {
                "horizon": 4,
                "checkpoint_freq": checkpoint_freq,
            },
        }
    )


def _config_with_envs(n_envs: int) -> FullConfig:
    cfg = _full_config()
    return cfg.model_copy(
        update={
            "env": cfg.env.model_copy(update={"n_envs": n_envs}),
        }
    )


def _config_with_resume_shape(
    *,
    n_envs: int,
    segments_per_minibatch: int,
    gradient_accumulation_steps: int,
    runtime_gpus: int,
    eval_replay_games: int = 0,
) -> FullConfig:
    cfg = _full_config()
    return FullConfig.model_validate(
        {
            **cfg.model_dump(mode="python"),
            "env": {
                **cfg.env.model_dump(mode="python"),
                "n_envs": n_envs,
            },
            "rl": {
                **cfg.rl.model_dump(mode="python"),
                "segments_per_minibatch": segments_per_minibatch,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eval_replay_games": eval_replay_games,
            },
            "runtime": {
                **cfg.runtime.model_dump(mode="python"),
                "n_runtime_gpus": runtime_gpus,
            },
        }
    )


def _distributed_context(world_size: int) -> run_ppo.DistributedContext:
    return run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=world_size,
        initialized=False,
    )


def _is_lora_adapter_key(key: str) -> bool:
    return key.endswith((".lora_down", ".lora_up"))


def _base_lora_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in state_dict.items() if not _is_lora_adapter_key(key)
    }


def _assert_quantized_state_equal(left: object, right: object) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_quantized_state_equal(left[key], right[key])
        return
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    assert left == right


def test_apply_lora_for_config_wraps_stateless_training_model() -> None:
    cfg = FullConfig.model_validate(
        {
            "env": {
                "n_envs": 2,
            },
            "model": {
                "model_arch": "stateless_transformer_v1",
                "embed_dim": 32,
                "depth": 2,
                "n_heads": 4,
                "lora": {
                    "rank": 4,
                    "target_modules": ["q", "v"],
                    "target_block_count": 1,
                },
            },
            "optimizer": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
            },
            "rl": {
                "horizon": 4,
            },
        }
    )
    model = run_ppo._create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    )

    application = run_ppo._apply_lora_for_config(model, cfg.model)

    assert application is not None
    assert application.module_count == 2
    assert application.trainable_parameters == 512
    assert not isinstance(model.blocks[0].attn.q, LoRALinear)
    assert isinstance(model.blocks[1].attn.q, LoRALinear)
    assert isinstance(model.blocks[1].attn.v, LoRALinear)
    assert all(
        name.endswith((".lora_down", ".lora_up")) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_create_training_model_roundtrip_quantizes_lora_base_model() -> None:
    base_cfg = _full_config()
    cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {
                    "rank": 2,
                    "target_modules": ["q"],
                    "roundtrip_quantization": NF4_G128_LSQ,
                },
            },
        }
    )
    torch.manual_seed(123)
    reference_model = run_ppo._create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    )
    reference_model.reset_parameters()
    assert run_ppo._apply_lora_for_config(reference_model, cfg.model) is not None
    reference_state = reference_model.state_dict()
    reference_base_state = _base_lora_state(reference_state)
    expected_quantized = quantize_model_state_dict(
        reference_base_state,
        NF4_G128_LSQ,
    )
    expected_base_state = dequantize_model_state_dict(expected_quantized)

    torch.manual_seed(123)
    model, application = run_ppo._create_training_model_for_config(
        cfg,
        device=torch.device("cpu"),
        reset_parameters=True,
    )

    assert application is not None
    actual_state = model.state_dict()
    for key, expected_tensor in expected_base_state.items():
        assert torch.equal(actual_state[key], expected_tensor)
    for key, expected_tensor in reference_state.items():
        if _is_lora_adapter_key(key):
            assert torch.equal(actual_state[key], expected_tensor)
    actual_quantized = quantize_model_state_dict(
        _base_lora_state(actual_state),
        NF4_G128_LSQ,
    )
    _assert_quantized_state_equal(actual_quantized, expected_quantized)


def test_create_eval_model_can_skip_lora_base_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_cfg = _full_config()
    cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {
                    "rank": 2,
                    "target_modules": ["q"],
                    "roundtrip_quantization": NF4_G128_LSQ,
                },
            },
        }
    )

    def fail_roundtrip(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("LoRA base roundtrip should be skipped")

    monkeypatch.setattr(
        run_ppo,
        "_roundtrip_lora_base_quantization_for_config",
        fail_roundtrip,
    )

    model = run_ppo._create_eval_model_for_config(
        cfg,
        device=torch.device("cpu"),
        roundtrip_lora_base=False,
    )

    assert isinstance(model.blocks[0].attn.q, LoRALinear)
    assert not model.training


def test_load_model_weights_allows_base_checkpoint_for_lora_model(
    tmp_path: Path,
) -> None:
    base_cfg = _full_config()
    lora_cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {
                    "rank": 2,
                    "target_modules": ["q", "v"],
                },
            },
        }
    )
    base_model = run_ppo._create_model(
        base_cfg.model,
        obs_spec=base_cfg.env.obs_spec,
        action_spec=base_cfg.env.action_spec,
    )
    lora_model = run_ppo._create_model(
        lora_cfg.model,
        obs_spec=lora_cfg.env.obs_spec,
        action_spec=lora_cfg.env.action_spec,
    )
    assert run_ppo._apply_lora_for_config(lora_model, lora_cfg.model) is not None
    path = tmp_path / "base_checkpoint.pt"
    torch.save(
        {
            "model": base_model.state_dict(),
            "env_steps": 64,
            "player_step_total": 5,
            "total_games_played": 7,
            "total_active_entities": 11,
            "wandb_run_id": "run-abc",
        },
        path,
    )
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = lora_model
    trainer.device = torch.device("cpu")
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 0

    metadata = trainer.load_model_weights(path)

    assert metadata.env_steps == 64
    assert trainer.player_step_total == 5
    assert trainer.total_games_played == 7
    assert trainer.total_active_entities == 11
    lora_state = lora_model.state_dict()
    for key, value in base_model.state_dict().items():
        assert torch.equal(lora_state[key], value)


def test_lora_fresh_launch_rejects_loading_optimizer_state() -> None:
    base_cfg = _full_config()
    cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {"rank": 2},
            },
        }
    )
    launch = run_ppo.FreshLaunch(
        config_path=Path("config.yaml"),
        output_dir=Path("runs"),
        overrides={},
        load_model_weights_path=Path("checkpoint.pt"),
        load_model_weights_mode="model_and_optimizer",
    )

    with pytest.raises(ValueError, match="model_and_optimizer is not supported"):
        run_ppo._validate_lora_launch_config(cfg, launch)


def test_initial_last_best_model_wraps_lora_and_supports_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: teacher_mode=last_best fine-tuning seeded from a non-LoRA base.
    # last_best must share the LoRA-wrapped student architecture so that a later
    # _refresh_eval_model_from_weights can copy the student's adapter-bearing
    # state dict into it instead of crashing on unexpected LoRA keys.
    monkeypatch.setattr(run_ppo, "_compile_eval_model", lambda *_a, **_k: None)
    base_cfg = _full_config()
    base_model = run_ppo._create_model(
        base_cfg.model,
        obs_spec=base_cfg.env.obs_spec,
        action_spec=base_cfg.env.action_spec,
    )
    base_model.reset_parameters()
    base_checkpoint = tmp_path / "base_checkpoint.pt"
    torch.save({"model": base_model.state_dict()}, base_checkpoint)
    # last_best validation reads the teacher checkpoint's sibling config.yaml.
    base_cfg.to_file(tmp_path / "config.yaml")

    student_cfg = FullConfig.model_validate(
        {
            **base_cfg.model_dump(mode="python"),
            "model": {
                **base_cfg.model.model_dump(mode="python"),
                "lora": {"rank": 2, "target_modules": ["q", "v"]},
            },
            "rl": {
                **base_cfg.rl.model_dump(mode="python"),
                "teacher_mode": "last_best",
                "teacher_init": str(base_checkpoint),
            },
        }
    )

    last_best = run_ppo._initial_last_best_model(
        student_cfg, device=torch.device("cpu")
    )

    assert last_best is not None
    # last_best is LoRA-wrapped like the student, and its frozen base weights are
    # seeded from the non-LoRA base checkpoint (adapters stay at config init).
    assert isinstance(last_best.blocks[0].attn.q, LoRALinear)
    last_best_state = last_best.state_dict()
    for key, value in base_model.state_dict().items():
        assert torch.equal(last_best_state[key], value)

    # Winning triggers refreshing last_best from the adapter-bearing student; this
    # is the path that previously raised on unexpected LoRA keys.
    student = run_ppo._create_model(
        student_cfg.model,
        obs_spec=student_cfg.env.obs_spec,
        action_spec=student_cfg.env.action_spec,
    )
    student.reset_parameters()
    assert run_ppo._apply_lora_for_config(student, student_cfg.model) is not None
    run_ppo._refresh_eval_model_from_weights(last_best, student)

    refreshed_state = last_best.state_dict()
    for key, value in student.state_dict().items():
        assert torch.equal(refreshed_state[key], value)


class _FakeLogger:
    def __init__(self, *, run_id: str | None = "run-123") -> None:
        self.closed = False
        self.close_exit_codes: list[int] = []
        self.logged: list[tuple[dict[str, float], int]] = []
        self.summary: dict[str, int | float | str] = {}
        self._run_id = run_id

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def wandb_run_facts(self) -> WandbRunFacts | None:
        return None

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logged.append((metrics, step))

    def set_summary(self, key: str, value: int | float | str) -> None:
        self.summary[key] = value

    def close(self, *, exit_code: int = 0) -> None:
        self.closed = True
        self.close_exit_codes.append(exit_code)


def _identity(telemetry: TelemetryMode = TelemetryMode.DISABLED) -> RunIdentity:
    return RunIdentity(
        experiment_id="exp",
        job_type="ppo",
        attempt=0,
        source_commit="abc123",
        attempt_source_commits=("abc123",),
        config_sha256="0" * 64,
        telemetry=telemetry,
    )


class _FakeTrainer:
    def __init__(
        self,
        *,
        fail: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.fail = fail
        self.metrics = {"loss": 1.0} if metrics is None else metrics
        self.checkpoints: list[tuple[Path, int, str | None]] = []
        self.iterations = 0
        self.model = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.teacher_updates: list[tuple[torch.nn.Module | None, bool]] = []
        self.checkpoint_models: dict[Path, torch.nn.Module | None] = {}

    def train_iteration(self) -> dict[str, float]:
        self.iterations += 1
        if self.fail:
            raise RuntimeError("training failed")
        return dict(self.metrics)

    def write_checkpoint(
        self,
        path: Path,
        *,
        env_steps: int,
        wandb_run_id: str | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.checkpoints.append((path, env_steps, wandb_run_id))
        self.checkpoint_models[path] = model

    def set_teacher_model(
        self,
        teacher_model: torch.nn.Module | None,
        *,
        active: bool,
    ) -> None:
        self.teacher_updates.append((teacher_model, active))


def _patch_eval_model_from_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> list[torch.nn.Module]:
    created_models: list[torch.nn.Module] = []

    def fake_create_eval_model_from_weights(
        source_model: torch.nn.Module,
        _cfg: FullConfig,
        *,
        device: torch.device,
    ) -> torch.nn.Module:
        model = torch.nn.Linear(1, 1).to(device)
        model.load_state_dict(source_model.state_dict())
        model.eval()
        created_models.append(model)
        return model

    monkeypatch.setattr(
        run_ppo,
        "_create_eval_model_from_weights",
        fake_create_eval_model_from_weights,
    )
    return created_models


def _write_checkpoint_metadata(path: Path, *, env_steps: int) -> None:
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "lr_scheduler": None,
            "env_steps": env_steps,
            "optimizer_steps": 0,
            "player_step_total": 0,
            "total_games_played": 0,
            "total_active_entities": 0,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-123",
        },
        path,
    )


def test_next_periodic_checkpoint_step_handles_crossed_cadence() -> None:
    assert run_ppo._next_periodic_checkpoint_step(checkpoint_freq=None) is None
    assert run_ppo._next_periodic_checkpoint_step(checkpoint_freq=1000) == 1000
    assert (
        run_ppo._next_periodic_checkpoint_step(
            checkpoint_freq=1000,
            env_steps=1256,
        )
        == 2000
    )


def test_format_checkpoint_step_zero_pads_grouped_digits() -> None:
    assert run_ppo._format_checkpoint_step(1_000_000_000) == "01_000_000_000"
    assert run_ppo._format_checkpoint_step(22_000_000) == "00_022_000_000"


def test_should_stop_training_checks_step_and_runtime_limits() -> None:
    assert run_ppo._should_stop_training(
        env_steps=128,
        started_at=time.monotonic(),
        max_env_steps=128,
        max_runtime_seconds=None,
    )
    assert run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic() - 2.0,
        max_env_steps=None,
        max_runtime_seconds=1.0,
    )
    assert not run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic(),
        max_env_steps=10,
        max_runtime_seconds=10.0,
    )


def test_should_stop_training_reduces_distributed_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=1,
        local_rank=1,
        world_size=2,
        initialized=True,
    )
    calls: list[bool] = []

    def fake_all_reduce_any(
        value: bool,
        _context: run_ppo.DistributedContext,
    ) -> bool:
        assert _context is context
        calls.append(value)
        return True

    monkeypatch.setattr(run_ppo, "all_reduce_any", fake_all_reduce_any)

    assert run_ppo._should_stop_training(
        env_steps=1,
        started_at=time.monotonic(),
        max_env_steps=10,
        max_runtime_seconds=10.0,
        distributed=context,
    )
    assert calls == [False]


def test_max_runtime_hours_converts_to_seconds() -> None:
    assert run_ppo._max_runtime_seconds(None) is None
    assert run_ppo._max_runtime_seconds(1.5) == 5400.0


def test_validate_args_rejects_non_positive_runtime_hours() -> None:
    with pytest.raises(ValueError, match="--max-runtime-hours must be positive"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=0.0,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
                wandb_mode=WandbMode.ONLINE,
                experiment_id=None,
            )
        )


def test_validate_args_rejects_debug_resume() -> None:
    with pytest.raises(ValueError, match="resume launches require wandb logging"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.DEBUG,
                wandb_mode=WandbMode.ONLINE,
                experiment_id=None,
            )
        )


def test_validate_args_rejects_resume_overrides() -> None:
    with pytest.raises(ValueError, match="resume launches cannot use config overrides"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=[["rl.horizon=8"]],
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
                wandb_mode=WandbMode.ONLINE,
                experiment_id=None,
            )
        )


def test_validate_args_rejects_resume_load_model_weights() -> None:
    with pytest.raises(
        ValueError,
        match="resume launches cannot use --load-model-weights",
    ):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=None,
                load_model_weights=Path("checkpoint.pt"),
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
                wandb_mode=WandbMode.ONLINE,
                experiment_id=None,
            )
        )


def test_validate_args_rejects_load_model_weights_mode_without_checkpoint() -> None:
    with pytest.raises(
        ValueError,
        match="--load-model-weights-mode requires --load-model-weights",
    ):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_and_optimizer",
                log_mode=LogMode.WANDB,
                wandb_mode=WandbMode.ONLINE,
                experiment_id=None,
            )
        )


def test_validate_args_rejects_offline_wandb_mode_with_debug_logging() -> None:
    with pytest.raises(ValueError, match="applies only to --log-mode wandb"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.DEBUG,
                wandb_mode=WandbMode.OFFLINE,
                experiment_id=None,
            )
        )


def test_validate_args_rejects_resume_experiment_id() -> None:
    with pytest.raises(ValueError, match="keep the recorded --experiment-id"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=None,
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
                wandb_mode=WandbMode.ONLINE,
                experiment_id="other",
            )
        )


def test_validate_args_rejects_a_malformed_experiment_id_before_the_run_dir() -> None:
    with pytest.raises(ValueError, match="experiment id must match"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.WANDB,
                wandb_mode=WandbMode.ONLINE,
                experiment_id="has space",
            )
        )


def test_validate_args_rejects_wandb_mode_without_wandb_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Through the parser: the one telemetry-mode check rejects the pair.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            "config.yaml",
            "runs",
            "--log-mode",
            "debug",
            "--wandb-mode",
            "offline",
        ],
    )
    with pytest.raises(
        ValueError, match="--wandb-mode offline requires --log-mode wandb"
    ):
        run_ppo._parse_args()


@pytest.mark.parametrize(
    ("flags", "expected"), [([], "online"), (["--wandb-mode", "offline"], "offline")]
)
def test_parse_args_reads_the_wandb_mode(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_ppo.py", "config.yaml", "runs", *flags])

    assert run_ppo._parse_args().wandb_mode == expected


def test_run_training_session_opens_an_offline_wandb_run_visibly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = _FakeWandbLogger(run_id="off1")
    created: list[dict[str, object]] = []
    identity = _identity(TelemetryMode.WANDB_OFFLINE)

    def create_fake_logger(*args: object, **kwargs: object) -> _FakeLogger:
        created.append({"args": args, **kwargs})
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)

    run_ppo._run_training_session(
        trainer=_FakeTrainer(),
        run_dir=tmp_path,
        cfg=_full_config(),
        log_mode=LogMode.WANDB,
        identity=identity,
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=DistributedContext.single_process_cpu(),
    )

    (call,) = created
    assert call["args"] == (LogMode.WANDB, tmp_path, _full_config())
    assert call["identity"] is identity
    assert f"W&B offline: telemetry stays under {tmp_path / 'wandb'}" in (
        capsys.readouterr().err
    )


def test_parse_cli_overrides_flattens_repeated_flags() -> None:
    assert run_ppo._parse_cli_overrides(
        [["rl.horizon=8"], ["env.n_envs=4", "model.depth=2"]]
    ) == {
        "rl.horizon": 8,
        "env.n_envs": 4,
        "model.depth": 2,
    }


def test_resolve_fresh_launch_accepts_load_model_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    output_dir = tmp_path / "runs"

    launch = run_ppo._resolve_launch(
        Namespace(
            target=config_path,
            output_dir=output_dir,
            overrides=[["rl.horizon=8"]],
            load_model_weights=checkpoint_path,
            load_model_weights_mode="model_and_optimizer",
        )
    )

    assert launch == run_ppo.FreshLaunch(
        config_path=config_path,
        output_dir=output_dir,
        overrides={"rl.horizon": 8},
        load_model_weights_path=checkpoint_path,
        load_model_weights_mode="model_and_optimizer",
    )


def test_resolve_teacher_init_path_uses_config_directory(tmp_path: Path) -> None:
    cfg = _full_config()
    cfg = cfg.model_copy(
        update={
            "rl": cfg.rl.model_copy(
                update={"teacher_init": Path("teachers/checkpoint.pt")}
            )
        }
    )

    resolved = run_ppo._resolve_teacher_init_path(cfg, tmp_path / "config.yaml")

    assert resolved.rl.teacher_init == (tmp_path / "teachers/checkpoint.pt").resolve()


def test_teacher_obs_spec_for_student_allows_max_entities_mismatch() -> None:
    student_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    teacher_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 5)

    resolved = run_ppo._teacher_obs_spec_for_student(
        teacher_obs_spec,
        student_obs_spec=student_obs_spec,
        checkpoint_path=Path("teacher.pt"),
    )

    assert resolved == student_obs_spec
    assert teacher_obs_spec.max_entities == MAX_PLANETS + MAX_COMETS + 5


def test_teacher_obs_spec_for_student_rejects_other_obs_mismatches() -> None:
    student_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    teacher_obs_spec = EntityBasedExtV1Config(max_entities=MAX_PLANETS + MAX_COMETS + 5)

    with pytest.raises(ValueError, match="except max_entities"):
        run_ppo._teacher_obs_spec_for_student(
            teacher_obs_spec,
            student_obs_spec=student_obs_spec,
            checkpoint_path=Path("teacher.pt"),
        )


def test_load_teacher_init_model_uses_student_max_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    teacher_obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 5)
    base_cfg = _full_config()
    student_cfg = base_cfg.model_copy(
        update={"env": base_cfg.env.model_copy(update={"obs_spec": student_obs_spec})}
    )
    teacher_cfg = base_cfg.model_copy(
        update={"env": base_cfg.env.model_copy(update={"obs_spec": teacher_obs_spec})}
    )
    checkpoint_path = tmp_path / "teacher" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.touch()
    teacher_cfg.to_file(checkpoint_path.parent / "config.yaml")
    teacher_model = torch.nn.Linear(1, 1)
    created_obs_specs: list[object] = []

    def fake_create_model(
        _config: object,
        *,
        obs_spec: object,
        action_spec: object,
    ) -> torch.nn.Module:
        del action_spec
        created_obs_specs.append(obs_spec)
        return teacher_model

    monkeypatch.setattr(run_ppo, "_create_model", fake_create_model)
    monkeypatch.setattr(run_ppo, "_load_model_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_ppo, "_compile_eval_model", lambda *_args, **_kwargs: None)

    loaded = run_ppo._load_teacher_init_model(
        checkpoint_path,
        student_cfg=student_cfg,
        device=torch.device("cpu"),
    )

    assert loaded is teacher_model
    assert created_obs_specs == [student_obs_spec]


def test_fixed_teacher_fresh_launch_leaves_last_best_unseeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    cfg = cfg.model_copy(
        update={
            "rl": cfg.rl.model_copy(
                update={
                    "teacher_mode": "fixed",
                    "teacher_init": Path("teacher/checkpoint.pt"),
                }
            )
        }
    )
    config_path = tmp_path / "config.yaml"
    cfg.to_file(config_path)
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    teacher_init_path = (config_path.parent / "teacher/checkpoint.pt").resolve()
    student_model = torch.nn.Linear(1, 1)
    fixed_teacher_model = torch.nn.Linear(1, 1)
    trainer_ref: dict[str, object] = {}
    session_ref: dict[str, object] = {}
    teacher_loads: list[Path] = []

    class FakeEnv:
        def __init__(
            self,
            *,
            n_envs: int,
            obs_spec: object,
            action_spec: object,
            two_player_weight: float,
            reward_mode: object,
            pin_memory: bool,
        ) -> None:
            del two_player_weight, reward_mode, pin_memory
            self.n_envs = n_envs
            self.obs_spec = obs_spec
            self.action_spec = action_spec

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.model = kwargs["model"]
            self.teacher_model = kwargs["teacher_model"]
            self.teacher_active = kwargs["teacher_active"]
            self.teacher_updates: list[tuple[torch.nn.Module | None, bool]] = []
            trainer_ref["trainer"] = self

        def set_teacher_model(
            self,
            teacher_model: torch.nn.Module | None,
            *,
            active: bool,
        ) -> None:
            self.teacher_updates.append((teacher_model, active))

    def fake_create_run_dir(output: Path) -> Path:
        assert output == output_dir
        run_dir.mkdir(parents=True)
        return run_dir

    def fake_load_teacher_init_model(
        checkpoint_path: Path,
        *,
        student_cfg: FullConfig,
        device: torch.device,
    ) -> torch.nn.Linear:
        assert student_cfg == cfg.model_copy(
            update={"rl": cfg.rl.model_copy(update={"teacher_init": teacher_init_path})}
        )
        assert device == torch.device("cpu")
        teacher_loads.append(checkpoint_path)
        return fixed_teacher_model

    def fake_run_training_session(**kwargs: object) -> None:
        session_ref.update(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            str(config_path),
            str(output_dir),
            "--log-mode",
            "debug",
        ],
    )
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    monkeypatch.setattr(run_ppo, "_create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    monkeypatch.setattr(
        run_ppo,
        "_create_model",
        lambda *_args, **_kwargs: student_model,
    )
    monkeypatch.setattr(run_ppo, "configure_model_compile", lambda *_args: 0)
    monkeypatch.setattr(
        run_ppo,
        "create_optimizer",
        lambda trainer_model, _cfg: torch.optim.SGD(
            trainer_model.parameters(),
            lr=0.1,
        ),
    )
    monkeypatch.setattr(run_ppo, "create_lr_scheduler", lambda *_args: None)
    monkeypatch.setattr(
        run_ppo,
        "_load_teacher_init_model",
        fake_load_teacher_init_model,
    )
    monkeypatch.setattr(run_ppo, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(run_ppo, "_run_training_session", fake_run_training_session)

    run_ppo.main()

    trainer = trainer_ref["trainer"]
    assert isinstance(trainer, FakeTrainer)
    assert teacher_loads == [teacher_init_path]
    assert trainer.teacher_model is fixed_teacher_model
    assert trainer.teacher_active
    assert trainer.teacher_updates == []
    assert session_ref["last_best_model"] is None


@pytest.mark.parametrize(
    ("mode", "expect_load_optimizer", "expect_fresh_sentinel"),
    [
        ("model_only", False, False),
        ("model_and_optimizer", True, False),
        ("model_fresh_critic_head", False, True),
    ],
)
def test_fresh_launch_from_checkpoint_uses_starting_checkpoint_as_teacher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expect_load_optimizer: bool,
    expect_fresh_sentinel: bool,
) -> None:
    fresh_sentinel = frozenset({"critic_head.sentinel"})
    fresh_mode_calls: list[tuple[torch.nn.Module, str]] = []
    real_fresh_state_keys_for_mode = run_ppo._fresh_state_keys_for_mode

    def fake_fresh_state_keys_for_mode(
        model_arg: torch.nn.Module,
        mode_arg: str,
    ) -> frozenset[str]:
        fresh_mode_calls.append((model_arg, mode_arg))
        if mode_arg == "model_fresh_critic_head":
            return fresh_sentinel
        return real_fresh_state_keys_for_mode(model_arg, mode_arg)  # type: ignore[arg-type]

    monkeypatch.setattr(
        run_ppo, "_fresh_state_keys_for_mode", fake_fresh_state_keys_for_mode
    )
    cfg = _full_config()
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"teacher_mode": "last_best"})}
    )
    config_path = tmp_path / "config.yaml"
    cfg.to_file(config_path)
    # More than one 1 MiB read chunk of non-constant bytes, so the recorded digest
    # proves the whole file was hashed.
    checkpoint_content = os.urandom((1 << 20) + 17)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_content)
    # The documented launch passes a relative path; the record must resolve it.
    # The ``..`` segment makes the resolved path differ from the merely absolute
    # one, so recording ``Path.absolute()`` instead of ``Path.resolve()`` fails.
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path)
    relative_checkpoint_path = Path("sub/../checkpoint.pt")
    assert relative_checkpoint_path.absolute() != checkpoint_path.resolve(), (
        "the path oracle must distinguish absolute from resolved"
    )
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    student_model = torch.nn.Linear(1, 1)
    student_model.weight.data.fill_(1.0)
    student_model.bias.data.fill_(1.0)
    teacher_model = torch.nn.Linear(1, 1)
    trainer_ref: dict[str, object] = {}
    session_ref: dict[str, object] = {}
    models = iter((student_model, teacher_model))
    compiled_models: list[torch.nn.Module] = []

    class FakeEnv:
        def __init__(
            self,
            *,
            n_envs: int,
            obs_spec: object,
            action_spec: object,
            two_player_weight: float,
            reward_mode: object,
            pin_memory: bool,
        ) -> None:
            del two_player_weight, reward_mode, pin_memory
            self.n_envs = n_envs
            self.obs_spec = obs_spec
            self.action_spec = action_spec

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.model = kwargs["model"]
            self.teacher_updates: list[tuple[torch.nn.Module, bool]] = []
            trainer_ref["trainer"] = self

        def load_model_weights(
            self,
            path: Path,
            *,
            load_optimizer: bool = False,
            fresh_state_keys: frozenset[str] = frozenset(),
        ) -> run_ppo.PPOCheckpointMetadata:
            assert path == relative_checkpoint_path
            assert load_optimizer is expect_load_optimizer
            assert fresh_state_keys == (
                fresh_sentinel if expect_fresh_sentinel else frozenset()
            )
            loaded_model = self.model
            assert isinstance(loaded_model, torch.nn.Linear)
            loaded_model.weight.data.fill_(7.0)
            loaded_model.bias.data.fill_(7.0)
            return run_ppo.PPOCheckpointMetadata(env_steps=123)

        def set_teacher_model(
            self,
            teacher_model: torch.nn.Module | None,
            *,
            active: bool,
        ) -> None:
            assert teacher_model is not None
            self.teacher_updates.append((teacher_model, active))

    def fake_create_run_dir(output: Path) -> Path:
        assert output == output_dir
        run_dir.mkdir(parents=True)
        return run_dir

    def fake_run_training_session(**kwargs: object) -> None:
        session_ref.update(kwargs)

    def fake_create_model(*_args: object, **_kwargs: object) -> torch.nn.Linear:
        return next(models)

    def fake_configure_model_compile(
        model_arg: torch.nn.Module,
        _cfg: object,
    ) -> int:
        compiled_models.append(model_arg)
        return 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            str(config_path),
            str(output_dir),
            "--load-model-weights",
            str(relative_checkpoint_path),
            "--load-model-weights-mode",
            mode,
            "--log-mode",
            "debug",
        ],
    )
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    monkeypatch.setattr(run_ppo, "_create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    monkeypatch.setattr(run_ppo, "_create_model", fake_create_model)
    monkeypatch.setattr(
        run_ppo, "configure_model_compile", fake_configure_model_compile
    )
    monkeypatch.setattr(
        run_ppo,
        "create_optimizer",
        lambda trainer_model, _cfg: torch.optim.SGD(trainer_model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(run_ppo, "create_lr_scheduler", lambda *_args: None)
    monkeypatch.setattr(run_ppo, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(run_ppo, "_run_training_session", fake_run_training_session)

    run_ppo.main()

    trainer = trainer_ref["trainer"]
    assert isinstance(trainer, FakeTrainer)
    assert len(trainer.teacher_updates) == 1
    active_teacher_model, active = trainer.teacher_updates[0]
    assert active
    assert active_teacher_model is teacher_model
    assert active_teacher_model is not student_model
    assert active_teacher_model.weight.item() == pytest.approx(7.0)
    assert compiled_models == [student_model, teacher_model]
    assert session_ref["start_env_steps"] == 123
    last_best_model = session_ref["last_best_model"]
    assert isinstance(last_best_model, torch.nn.Linear)
    assert last_best_model is teacher_model
    assert last_best_model.weight.item() == pytest.approx(7.0)
    assert fresh_mode_calls == [(student_model, mode)]
    warm_start = json.loads((run_dir / run_ppo.WARM_START_RECORD).read_text())
    assert warm_start == {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_content).hexdigest(),
        "load_model_weights_mode": mode,
    }
    assert session_ref["warm_start"] == warm_start


def test_resolve_resume_launch_prefers_final_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    final_checkpoint = run_dir / "checkpoint_final.pt"
    _write_checkpoint_metadata(final_checkpoint, env_steps=20_000)
    _write_checkpoint_metadata(
        run_dir / "checkpoint_00_000_010_000.pt",
        env_steps=10_000,
    )
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.config_path == run_dir / "config.yaml"
    assert launch.checkpoint_path == final_checkpoint
    assert launch.last_best_checkpoint_path == run_dir / "checkpoint_last_best.pt"


def test_resolve_resume_launch_uses_numbered_when_newer_than_final(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    final_checkpoint = run_dir / "checkpoint_final.pt"
    _write_checkpoint_metadata(final_checkpoint, env_steps=10_000)
    latest_checkpoint = run_dir / "checkpoint_00_000_020_000.pt"
    _write_checkpoint_metadata(latest_checkpoint, env_steps=20_000)
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.checkpoint_path == latest_checkpoint


def test_resolve_resume_launch_uses_latest_numbered_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    (run_dir / "checkpoint_00_000_010_000.pt").touch()
    latest_checkpoint = run_dir / "checkpoint_00_000_020_000.pt"
    latest_checkpoint.touch()
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(run_dir)

    assert launch.checkpoint_path == latest_checkpoint


def test_resolve_resume_launch_rejects_missing_last_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    (run_dir / "checkpoint_final.pt").touch()

    with pytest.raises(ValueError, match="expected last-best checkpoint"):
        run_ppo._resolve_resume_launch(run_dir)


def test_resolve_resume_launch_uses_adjacent_config_for_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("env: {}\nmodel: {}\noptimizer: {}\nrl: {}\n")
    checkpoint = run_dir / "checkpoint_00_000_020_000.pt"
    checkpoint.touch()
    (run_dir / "checkpoint_last_best.pt").touch()

    launch = run_ppo._resolve_resume_launch(checkpoint)

    assert launch.config_path == run_dir / "config.yaml"
    assert launch.checkpoint_path == checkpoint


def test_resume_wandb_run_id_requires_checkpoint_run_id() -> None:
    metadata = run_ppo.PPOCheckpointMetadata(
        env_steps=1,
        wandb_run_id=None,
    )

    with pytest.raises(ValueError, match="missing wandb_run_id"):
        run_ppo._resume_wandb_run_id(metadata, LogMode.WANDB)


def test_checkpoint_metadata_rejects_positional_fields() -> None:
    with pytest.raises(TypeError):
        run_ppo.PPOCheckpointMetadata(1, "run-abc")


def test_checkpoint_metadata_accepts_current_trainer_checkpoint_schema(
    tmp_path: Path,
) -> None:
    checkpoint = {
        "model": {},
        "optimizer": {},
        "lr_scheduler": None,
        "env_steps": 123,
        "optimizer_steps": 7,
        "player_step_total": 19,
        "total_games_played": 23,
        "total_active_entities": 29,
        "target_kl_exceeded_total": 3,
        "wandb_run_id": "run-abc",
    }

    metadata = run_ppo._checkpoint_metadata(
        checkpoint,
        path=tmp_path / "checkpoint.pt",
    )

    assert metadata.env_steps == 123
    assert metadata.player_step_total == 19
    assert metadata.total_games_played == 23
    assert metadata.total_active_entities == 29
    assert metadata.wandb_run_id == "run-abc"


def test_checkpoint_metadata_defaults_missing_total_active_entities(
    tmp_path: Path,
) -> None:
    checkpoint = {
        "model": {},
        "optimizer": {},
        "lr_scheduler": None,
        "env_steps": 123,
        "optimizer_steps": 7,
        "player_step_total": 19,
        "total_games_played": 23,
        "target_kl_exceeded_total": 3,
        "wandb_run_id": "run-abc",
    }

    metadata = run_ppo._checkpoint_metadata(
        checkpoint,
        path=tmp_path / "checkpoint.pt",
    )

    assert metadata.total_active_entities == 0


def test_evaluate_against_last_best_uses_eval_mode_no_grad_and_eval_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config_with_envs(4)
    current_model = torch.nn.Linear(1, 1)
    last_best_model = torch.nn.Linear(1, 1)
    current_model.train()
    last_best_model.eval()
    seen_eval_sizes: list[tuple[object, ...]] = []
    perf_times = iter([10.0, 14.0])

    def fake_evaluate_games(
        **kwargs: object,
    ) -> tuple[object, dict[int, object], dict[str, list[float]], int]:
        assert kwargs["current_model"] is current_model
        assert kwargs["last_best_model"] is last_best_model
        assert not current_model.training
        assert not last_best_model.training
        assert not torch.is_grad_enabled()
        seen_eval_sizes.append(
            (
                kwargs["n_games"],
                kwargs["n_envs"],
                kwargs["replay_games"],
                kwargs["env_steps"],
            )
        )
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        stats.add_game_result(run_ppo.MODEL_LAST_BEST)
        stats_by_count = {
            2: run_ppo._EvalStats.empty(),
            4: run_ppo._EvalStats.empty(),
        }
        stats_by_count[2].add_game_result(run_ppo.MODEL_CURRENT)
        stats_by_count[4].add_game_result(run_ppo.MODEL_LAST_BEST)
        return (
            stats,
            stats_by_count,
            {
                "game_length_mean": [12.0],
                "_neutral_planets_captured_per_game": [1.0],
                "_neutral_comets_captured_per_game": [2.0],
                "_neutral_planet_undershots_per_game": [3.0],
                "_neutral_comet_undershots_per_game": [4.0],
            },
            6,
        )

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_games",
        fake_evaluate_games,
    )
    monkeypatch.setattr(run_ppo.time, "perf_counter", lambda: next(perf_times))

    metrics = run_ppo._evaluate_against_last_best(
        current_model=current_model,
        last_best_model=last_best_model,
        cfg=cfg,
        device=torch.device("cpu"),
        env_steps=40_000_000,
    )

    assert metrics["eval/win_rate_against_last_best"] == pytest.approx(0.5)
    assert metrics["eval/win_rate_against_last_best_2p"] == pytest.approx(1.0)
    assert metrics["eval/win_rate_against_last_best_4p"] == pytest.approx(0.0)
    assert metrics["eval/game_length_mean"] == pytest.approx(12.0)
    assert metrics["eval/neutral_planet_undershot_rate"] == pytest.approx(0.75)
    assert metrics["eval/neutral_comet_undershot_rate"] == pytest.approx(2.0 / 3.0)
    assert "eval/_neutral_planets_captured_per_game" not in metrics
    assert metrics["time/eval_seconds"] == pytest.approx(4.0)
    assert metrics["perf/eval_sps"] == pytest.approx(1.5)
    # Isaiah's default evaluation count: one game per env on the main process.
    assert seen_eval_sizes == [(4, 4, 0, 40_000_000)]
    assert metrics["eval/games"] == pytest.approx(2.0)
    assert current_model.training
    assert not last_best_model.training


def test_evaluate_against_last_best_records_weighted_eval_replay_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config_with_envs(4)
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"eval_replay_games": 2})}
    )
    seen_replays: list[tuple[int, Path | None]] = []

    def fake_evaluate_games(
        **kwargs: object,
    ) -> tuple[object, dict[int, object], dict[str, list[float]], int]:
        seen_replays.append(
            (
                kwargs["replay_games"],
                kwargs["replay_output_path"],
            )
        )
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        stats_by_count = {2: stats, 4: run_ppo._EvalStats.empty()}
        return stats, stats_by_count, {}, 1

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_games",
        fake_evaluate_games,
    )

    run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=cfg,
        device=torch.device("cpu"),
        env_steps=1000,
        replay_dir=tmp_path,
    )

    assert seen_replays == [(2, tmp_path / "eval.jsonl")]


def test_evaluate_against_last_best_omits_empty_player_count_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate_games(
        **_kwargs: object,
    ) -> tuple[object, dict[int, object], dict[str, list[float]], int]:
        stats = run_ppo._EvalStats.empty()
        stats.add_game_result(run_ppo.MODEL_CURRENT)
        stats_by_count = {
            2: run_ppo._EvalStats.empty(),
            4: run_ppo._EvalStats.empty(),
        }
        stats_by_count[2].add_game_result(run_ppo.MODEL_CURRENT)
        return stats, stats_by_count, {}, 1

    monkeypatch.setattr(run_ppo, "_evaluate_games", fake_evaluate_games)

    metrics = run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=_config_with_envs(2),
        device=torch.device("cpu"),
        env_steps=1000,
    )

    assert metrics["eval/win_rate_against_last_best_2p"] == pytest.approx(1.0)
    assert "eval/win_rate_against_last_best_4p" not in metrics


def test_record_eval_terminal_result_counts_team_ties_as_half_win() -> None:
    stats = run_ppo._EvalStats.empty()

    run_ppo._record_eval_terminal_result(
        stats,
        assignment=torch.tensor([0, 1, 1, 0]),
        start_mask=torch.tensor([True, True, True, True]),
        scores=torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )

    assert stats.model_games == [1, 1]
    assert stats.wins == [0.5, 0.5]
    assert stats.win_rate(run_ppo.MODEL_CURRENT) == pytest.approx(0.5)


def test_assign_eval_models_randomizes_active_player_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = torch.full((2, 4), -1, dtype=torch.int64)
    permutations = iter(
        [
            torch.tensor([1, 0]),
            torch.tensor([3, 0, 2, 1]),
        ]
    )

    monkeypatch.setattr(run_ppo.torch, "randperm", lambda _n: next(permutations))

    run_ppo._assign_eval_models(
        assignments,
        0,
        active_slots=torch.tensor([True, True, False, False]),
        player_count=2,
    )
    run_ppo._assign_eval_models(
        assignments,
        1,
        active_slots=torch.tensor([True, True, True, True]),
        player_count=4,
    )

    assert assignments[0].tolist() == [
        run_ppo.MODEL_LAST_BEST,
        run_ppo.MODEL_CURRENT,
        -1,
        -1,
    ]
    assert assignments[1].tolist() == [
        run_ppo.MODEL_CURRENT,
        run_ppo.MODEL_CURRENT,
        run_ppo.MODEL_LAST_BEST,
        run_ppo.MODEL_LAST_BEST,
    ]


def test_eval_actions_for_assignments_uses_stochastic_model_outputs() -> None:
    class FakeModel:
        def __init__(self, *, launch_value: bool, ship_value: int) -> None:
            self.launch_value = launch_value
            self.ship_value = ship_value

        def __call__(
            self,
            obs: ObsBatch,  # noqa: ARG002
            *,
            deterministic: bool = False,
        ) -> SimpleNamespace:
            assert not deterministic
            shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
            actions = run_ppo.PureActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                ships=torch.full(shape, self.ship_value, dtype=torch.int64),
                angle=torch.zeros(shape, dtype=torch.float32),
            )
            return SimpleNamespace(actions=actions, next_hidden_state=None)

    obs = ObsBatch(
        planets=torch.zeros((1, 1, 1)),
        orbiting_planets=torch.zeros((1, 1), dtype=torch.bool),
        fleets=torch.zeros((1, 1, 1)),
        comets=torch.zeros((1, 1, 1)),
        entity_mask=torch.zeros((1, 1), dtype=torch.bool),
        still_playing=torch.ones((1, 4), dtype=torch.bool),
        global_features=torch.zeros((1, 1)),
        action_mask=PureActionMask(
            can_act=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
            max_launch=torch.zeros((1, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        ),
    )

    actions = run_ppo._eval_actions_for_assignments(
        obs,
        torch.tensor([[0, 1, 0, 1]]),
        current_model=FakeModel(launch_value=True, ship_value=3),
        last_best_model=FakeModel(launch_value=False, ship_value=7),
        config=Namespace(dtype="float32"),
        device=torch.device("cpu"),
    )

    assert actions.launch[0, :, 0, 0].tolist() == [True, False, True, False]
    assert actions.ships[0, :, 0, 0].tolist() == [3, 7, 3, 7]


def test_evaluate_games_carries_recurrent_hidden_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def obs_batch(n_envs: int) -> ObsBatch:
        return ObsBatch(
            planets=torch.zeros((n_envs, 1, 1)),
            orbiting_planets=torch.zeros((n_envs, 1), dtype=torch.bool),
            fleets=torch.zeros((n_envs, 1, 1)),
            comets=torch.zeros((n_envs, 1, 1)),
            entity_mask=torch.zeros((n_envs, 1), dtype=torch.bool),
            still_playing=torch.tensor(
                [[True, True, False, False] for _ in range(n_envs)],
                dtype=torch.bool,
            ),
            global_features=torch.zeros((n_envs, 1)),
            action_mask=PureActionMask(
                can_act=torch.zeros((n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.bool),
                max_launch=torch.zeros(
                    (n_envs, 4, ACTION_ENTITY_SLOTS), dtype=torch.int64
                ),
            ),
        )

    class FakeEnv:
        def __init__(
            self,
            *,
            n_envs: int,
            two_player_weight: float,
            **_kwargs: object,
        ) -> None:
            assert two_player_weight == pytest.approx(0.25)
            self.n_envs = n_envs
            self.steps = 0

        def reset(self) -> ObsBatch:
            return obs_batch(self.n_envs)

        def step(
            self,
            actions: run_ppo.ActionBundle,  # noqa: ARG002
        ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
            self.steps += 1
            rewards = torch.zeros((self.n_envs, 4), dtype=torch.float32)
            dones = torch.zeros((self.n_envs, 4), dtype=torch.bool)
            if self.steps == 2:
                dones.fill_(True)
            return obs_batch(self.n_envs), rewards, dones, {}

        def terminal_metrics(self, env_index: int) -> dict[str, float]:  # noqa: ARG002
            return {"game_length_mean": 2.0}

    class RecordingRecurrentModel:
        def __init__(self, *, initial: float, launch_value: bool) -> None:
            self.initial = initial
            self.launch_value = launch_value
            self.seen_hidden: list[torch.Tensor] = []
            self.reset_dones: list[torch.Tensor] = []

        def initial_hidden_state(
            self,
            batch_size: int,
            *,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.full((batch_size,), self.initial, device=device)

        def __call__(
            self,
            obs: ObsBatch,
            *,
            deterministic: bool = False,
            hidden_state: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            assert not deterministic
            assert hidden_state is not None
            self.seen_hidden.append(hidden_state.detach().cpu().clone())
            n_envs = obs.global_features.shape[0]
            shape = (n_envs, 4, ACTION_ENTITY_SLOTS, 1)
            actions = run_ppo.PureActions(
                launch=torch.full(shape, self.launch_value, dtype=torch.bool),
                angle=torch.zeros(shape, dtype=torch.float32),
                ships=torch.ones(shape, dtype=torch.int64),
            )
            return SimpleNamespace(
                actions=actions,
                next_hidden_state=hidden_state + 1.0,
            )

        def reset_hidden_state(
            self,
            hidden_state: torch.Tensor | None,
            dones: torch.Tensor,
        ) -> torch.Tensor | None:
            assert hidden_state is not None
            self.reset_dones.append(dones.detach().cpu().clone())
            keep = ~dones.all(dim=1).to(device=hidden_state.device)
            return hidden_state * keep.to(dtype=hidden_state.dtype)

    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    current_model = RecordingRecurrentModel(initial=0.0, launch_value=True)
    last_best_model = RecordingRecurrentModel(initial=10.0, launch_value=False)
    base_cfg = _config_with_envs(2)
    cfg = base_cfg.model_copy(
        update={
            "env": base_cfg.env.model_copy(update={"two_player_weight": 0.25}),
        }
    )

    stats, stats_by_player_count, env_metrics, steps = run_ppo._evaluate_games(
        current_model=current_model,
        last_best_model=last_best_model,
        cfg=cfg,
        n_games=2,
        n_envs=2,
        device=torch.device("cpu"),
        env_steps=1000,
    )

    assert steps == 4
    assert env_metrics == {"game_length_mean": [2.0, 2.0]}
    assert stats.model_games == [2, 2]
    assert stats_by_player_count[2].model_games == [2, 2]
    assert stats_by_player_count[4].model_games == [0, 0]
    assert [hidden.tolist() for hidden in current_model.seen_hidden] == [
        [0.0, 0.0],
        [1.0, 1.0],
    ]
    assert [hidden.tolist() for hidden in last_best_model.seen_hidden] == [
        [10.0, 10.0],
        [11.0, 11.0],
    ]
    assert [dones.any().item() for dones in current_model.reset_dones] == [
        False,
        True,
    ]


def test_select_actions_handles_discrete_target_bundles() -> None:
    shape = (1, 4, ACTION_ENTITY_SLOTS, 1)
    actions_a = run_ppo.DiscreteTargetActions(
        launch=torch.full(shape, True, dtype=torch.bool),
        target=torch.full(shape, 3, dtype=torch.int64),
        ships=torch.full(shape, 5, dtype=torch.int64),
    )
    actions_b = run_ppo.DiscreteTargetActions(
        launch=torch.full(shape, False, dtype=torch.bool),
        target=torch.full(shape, 7, dtype=torch.int64),
        ships=torch.full(shape, 11, dtype=torch.int64),
    )

    selected = run_ppo._select_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert isinstance(selected, run_ppo.DiscreteTargetActions)
    assert selected.target[0, :, 0, 0].tolist() == [3, 7, 3, 7]
    assert selected.ships[0, :, 0, 0].tolist() == [5, 11, 5, 11]


def test_select_actions_handles_discrete_target_bin_bundles() -> None:
    shape = (1, 4, ACTION_ENTITY_SLOTS)
    actions_a = run_ppo.DiscreteTargetBinActions(
        target=torch.full(shape, 2, dtype=torch.int64),
        fleet_bin=torch.full(shape, 4, dtype=torch.int64),
    )
    actions_b = run_ppo.DiscreteTargetBinActions(
        target=torch.full(shape, 6, dtype=torch.int64),
        fleet_bin=torch.full(shape, 8, dtype=torch.int64),
    )

    selected = run_ppo._select_actions(
        actions_a,
        actions_b,
        torch.tensor([[True, False, True, False]]),
    )

    assert isinstance(selected, run_ppo.DiscreteTargetBinActions)
    assert selected.target[0, :, 0].tolist() == [2, 6, 2, 6]
    assert selected.fleet_bin[0, :, 0].tolist() == [4, 8, 4, 8]


def test_create_model_uses_env_owned_specs() -> None:
    obs_spec = EntityBasedConfig(max_entities=MAX_PLANETS + MAX_COMETS + 2)
    action_spec = ActionPureConfig(max_per_planet_launches=1)
    model = run_ppo._create_model(
        _full_config().model,
        obs_spec=obs_spec,
        action_spec=action_spec,
    )

    assert model.obs_spec == obs_spec
    assert model.action_spec == action_spec
    assert model.fleet_proj.in_features == obs_spec.fleet_channels
    assert model.actor.max_per_planet_launches == 1


def test_create_eval_model_from_weights_builds_fresh_compiled_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config()
    source_model = run_ppo._create_model(
        cfg.model,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
    )
    with torch.no_grad():
        for index, parameter in enumerate(source_model.parameters()):
            parameter.fill_(float(index + 1))
    compiled_models: list[torch.nn.Module] = []

    def fake_configure_model_compile(
        model: torch.nn.Module,
        compile_cfg: object,
    ) -> int:
        assert compile_cfg is cfg.rl
        compiled_models.append(model)
        return 0

    monkeypatch.setattr(
        run_ppo, "configure_model_compile", fake_configure_model_compile
    )

    teacher_model = run_ppo._create_eval_model_from_weights(
        source_model,
        cfg,
        device=torch.device("cpu"),
    )

    assert teacher_model is not source_model
    assert not teacher_model.training
    assert compiled_models == [teacher_model]
    for key, source_tensor in source_model.state_dict().items():
        assert torch.equal(teacher_model.state_dict()[key], source_tensor)
    source_param = next(source_model.parameters())
    teacher_param = next(teacher_model.parameters())
    assert teacher_param.data_ptr() != source_param.data_ptr()


def test_refresh_eval_model_from_weights_updates_existing_model() -> None:
    source_model = torch.nn.Linear(1, 1)
    target_model = torch.nn.Linear(1, 1)
    source_model.weight.data.fill_(5.0)
    source_model.bias.data.fill_(7.0)
    target_model.train()

    run_ppo._refresh_eval_model_from_weights(target_model, source_model)

    assert target_model is not source_model
    assert target_model.weight.item() == pytest.approx(5.0)
    assert target_model.bias.item() == pytest.approx(7.0)
    assert not target_model.training


def test_trainable_parameter_count_ignores_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].weight.requires_grad = False

    assert run_ppo._trainable_parameter_count(model) == 10


def test_with_runtime_gpus_records_world_size() -> None:
    cfg = _full_config()

    updated = run_ppo._with_runtime_gpus(cfg, 4)

    assert updated.runtime.n_runtime_gpus == 4
    assert cfg.runtime.n_runtime_gpus == 1


def test_adapt_resume_config_returns_unchanged_config_when_gpu_count_matches() -> None:
    cfg = run_ppo._with_runtime_gpus(_full_config(), 4)

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(4),
    )

    assert updated == cfg


def test_adapt_resume_config_halves_envs_and_accumulation_for_more_gpus() -> None:
    cfg = _config_with_resume_shape(
        n_envs=64,
        segments_per_minibatch=8,
        gradient_accumulation_steps=2,
        runtime_gpus=2,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(4),
    )

    assert updated.env.n_envs == 32
    assert updated.rl.segments_per_minibatch == 8
    assert updated.rl.gradient_accumulation_steps == 1
    assert updated.runtime.n_runtime_gpus == 4
    assert cfg.env.n_envs == 64
    assert cfg.rl.gradient_accumulation_steps == 2


def test_adapt_resume_config_reduces_minibatch_segments_for_more_gpus() -> None:
    cfg = _config_with_resume_shape(
        n_envs=64,
        segments_per_minibatch=16,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(4),
    )

    assert updated.env.n_envs == 32
    assert updated.rl.segments_per_minibatch == 8
    assert updated.rl.gradient_accumulation_steps == 1
    assert updated.runtime.n_runtime_gpus == 4


def test_adapt_resume_config_doubles_envs_and_accumulation_for_fewer_gpus() -> None:
    cfg = _config_with_resume_shape(
        n_envs=32,
        segments_per_minibatch=8,
        gradient_accumulation_steps=1,
        runtime_gpus=4,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(2),
    )

    assert updated.env.n_envs == 64
    assert updated.rl.segments_per_minibatch == 8
    assert updated.rl.gradient_accumulation_steps == 2
    assert updated.runtime.n_runtime_gpus == 2


def test_adapt_resume_config_reduces_segments_for_fewer_gpus_when_needed() -> None:
    cfg = _config_with_resume_shape(
        n_envs=48,
        segments_per_minibatch=16,
        gradient_accumulation_steps=1,
        runtime_gpus=3,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(2),
    )

    assert updated.env.n_envs == 72
    assert updated.rl.segments_per_minibatch == 12
    assert updated.rl.gradient_accumulation_steps == 2
    assert updated.runtime.n_runtime_gpus == 2


def test_adapt_resume_config_reduces_segments_when_accumulation_not_exact() -> None:
    cfg = _config_with_resume_shape(
        n_envs=12,
        segments_per_minibatch=3,
        gradient_accumulation_steps=2,
        runtime_gpus=2,
    )

    updated = run_ppo._adapt_resume_config_for_runtime_gpus(
        cfg,
        _distributed_context(3),
    )

    assert updated.env.n_envs == 8
    assert updated.rl.segments_per_minibatch == 2
    assert updated.rl.gradient_accumulation_steps == 2
    assert updated.runtime.n_runtime_gpus == 3


def test_adapt_resume_config_rejects_fractional_env_count() -> None:
    cfg = _config_with_resume_shape(
        n_envs=8,
        segments_per_minibatch=2,
        gradient_accumulation_steps=2,
        runtime_gpus=2,
    )

    with pytest.raises(ValueError, match=r"env.n_envs=.*does not scale evenly"):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(3),
        )


def test_adapt_resume_config_rejects_odd_derived_env_count() -> None:
    cfg = _config_with_resume_shape(
        n_envs=6,
        segments_per_minibatch=2,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
    )

    with pytest.raises(ValueError, match="n_envs must be even"):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(4),
        )


def test_adapt_resume_config_rejects_fractional_train_batch() -> None:
    cfg = _config_with_resume_shape(
        n_envs=12,
        segments_per_minibatch=4,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
    )

    with pytest.raises(
        ValueError,
        match=r"rl.segments_per_minibatch .*does not scale evenly",
    ):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(3),
        )


def test_adapt_resume_config_rejects_eval_replay_games_above_derived_envs() -> None:
    cfg = _config_with_resume_shape(
        n_envs=8,
        segments_per_minibatch=4,
        gradient_accumulation_steps=1,
        runtime_gpus=2,
        eval_replay_games=8,
    )

    with pytest.raises(
        ValueError,
        match=r"rl.eval_replay_games must be <= env.n_envs",
    ):
        run_ppo._adapt_resume_config_for_runtime_gpus(
            cfg,
            _distributed_context(4),
        )


def test_run_training_loop_writes_periodic_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    cfg = cfg.model_copy(
        update={
            "env": cfg.env.model_copy(
                update={
                    "obs_spec": EntityBasedConfig(
                        max_entities=MAX_PLANETS + MAX_COMETS + 3
                    ),
                },
            ),
        },
    )
    trainer = _FakeTrainer(metrics={"loss": 1.0, "train/max_entities": 17.0})
    logger = _FakeLogger()
    eval_calls = 0
    _patch_eval_model_from_weights(monkeypatch)

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        nonlocal eval_calls
        eval_calls += 1
        return {"eval/win_rate_against_last_best": 0.25}

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=800,
        max_env_steps=1600,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 1600
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_001_600.pt", 1600, None),
        (tmp_path / "checkpoint_last_best.pt", 0, None),
    ]
    assert trainer.checkpoint_models[tmp_path / "checkpoint_last_best.pt"] is not None
    assert [step for _metrics, step in logger.logged] == [800, 1600, 1600]
    assert logger.logged[0][0]["train/max_entities"] == pytest.approx(17.0)
    assert logger.logged[1][0]["train/max_entities"] == pytest.approx(17.0)
    assert logger.logged[-1][0] == {
        "eval/win_rate_against_last_best": 0.25,
        "eval/promoted": 0.0,
        "eval/promotion_threshold": 0.7,
    }
    assert eval_calls == 1
    assert "model/trainable_parameters" not in logger.logged[0][0]
    assert "trainable_parameters" not in logger.logged[0][0]


def test_run_training_loop_resumes_checkpoint_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)
    (tmp_path / "checkpoint_last_best.pt").touch()

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        return {"eval/win_rate_against_last_best": 0.25}

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=800,
        max_env_steps=2000,
        max_runtime_seconds=None,
        start_env_steps=1200,
        wandb_run_id="run-123",
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 2000
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_002_000.pt", 2000, "run-123")
    ]
    assert [step for _metrics, step in logger.logged] == [2000, 2000]


def test_run_training_loop_returns_immediately_when_resume_reached_step_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=800,
        max_env_steps=1200,
        max_runtime_seconds=None,
        start_env_steps=1200,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 1200
    assert trainer.iterations == 0
    assert trainer.checkpoints == []
    assert logger.logged == []


def test_run_training_loop_saves_last_best_when_eval_clears_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        return {
            "eval/win_rate_against_last_best": 0.7,
            "eval/game_length_mean": 12.0,
        }

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    env_steps = run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=1000,
        max_env_steps=1000,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert env_steps == 1000
    assert trainer.checkpoints == [
        (tmp_path / "checkpoint_00_000_001_000.pt", 1000, None),
        (tmp_path / "checkpoint_last_best.pt", 0, None),
        (tmp_path / "checkpoint_last_best.pt", 1000, None),
    ]
    assert logger.logged[-1][0]["eval/game_length_mean"] == 12.0


def test_run_training_loop_activates_last_best_teacher_after_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"teacher_mode": "last_best"})}
    )
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    created_models = _patch_eval_model_from_weights(monkeypatch)

    def fake_evaluate_against_last_best(**_kwargs: object) -> dict[str, float]:
        return {"eval/win_rate_against_last_best": 0.7}

    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        fake_evaluate_against_last_best,
    )

    run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=1000,
        max_env_steps=1000,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert len(trainer.teacher_updates) == 1
    teacher_model, active = trainer.teacher_updates[0]
    assert teacher_model is not None
    assert teacher_model is created_models[0]
    assert len(created_models) == 1
    assert active


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_logger_session_marks_failed_runs(error_type: type[BaseException]) -> None:
    closed: list[int] = []

    class _Logger(MetricLogger):
        def close(self, *, exit_code: int = 0) -> None:
            closed.append(exit_code)

    logger = _Logger()
    error = error_type("boom")
    with (
        pytest.raises(error_type, match="boom") as exc_info,
        run_ppo._logger_session(logger) as active_logger,
    ):
        raise error
    assert active_logger is logger
    assert exc_info.value is error
    assert closed == [1]

    with run_ppo._logger_session(logger) as active_logger:
        assert active_logger is logger
    assert closed == [1, 0]


def test_run_training_session_sets_launch_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    def create_fake_logger(*_args: object, **_kwargs: object) -> _FakeLogger:
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)

    run_ppo._run_training_session(
        trainer=trainer,
        run_dir=tmp_path,
        cfg=cfg,
        log_mode=LogMode.DEBUG,
        identity=_identity(),
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=DistributedContext.single_process_cpu(),
        start_env_steps=16,
        trainable_parameters=123,
        compiled_model_modules=4,
        warm_start={
            "checkpoint_path": "/bc/best.pt",
            "checkpoint_sha256": "ab" * 32,
            "load_model_weights_mode": "model_fresh_critic_head",
        },
    )

    assert logger.summary == {
        "compiled_model_modules": 4,
        "trainable_parameters": 123,
        "warm_start/checkpoint_path": "/bc/best.pt",
        "warm_start/checkpoint_sha256": "ab" * 32,
        "warm_start/load_model_weights_mode": "model_fresh_critic_head",
    }
    assert logger.closed
    assert logger.close_exit_codes == [0]


def test_run_training_session_records_the_compile_gemm_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = _FakeLogger()
    monkeypatch.setattr(run_ppo, "create_logger", lambda *_a, **_k: logger)
    claim = GemmBackendClaim(
        game="kaggriculture",
        gemm_backends="ATEN",
        stack=CompileStackReport(
            torch="2.9.0+cu128", triton="3.5.0", nvidia_driver="595.91.07"
        ),
    )

    run_ppo._run_training_session(
        trainer=_FakeTrainer(),
        run_dir=tmp_path,
        cfg=_full_config(),
        log_mode=LogMode.DEBUG,
        identity=_identity(),
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=DistributedContext.single_process_cpu(),
        compiled_model_modules=1,
        compile_claim=claim,
    )

    assert logger.summary == {
        "compiled_model_modules": 1,
        "compile_gemm_game": "kaggriculture",
        "compile_gemm_backends": "ATEN",
        "compile_stack_torch": "2.9.0+cu128",
        "compile_stack_triton": "3.5.0",
        "compile_stack_nvidia_driver": "595.91.07",
    }
    assert run_ppo._compile_claim_log_line(claim) == (
        "Compiled GEMM backends: compile_gemm_game=kaggriculture, "
        "compile_gemm_backends=ATEN, compile_stack_torch=2.9.0+cu128, "
        "compile_stack_triton=3.5.0, compile_stack_nvidia_driver=595.91.07"
    )


def test_run_training_session_worker_skips_logger_and_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    distributed = run_ppo.DistributedContext(
        device=torch.device("cpu"),
        rank=1,
        local_rank=1,
        world_size=2,
        initialized=False,
    )

    def create_fake_logger(*_args: object, **_kwargs: object) -> _FakeLogger:
        raise AssertionError("worker rank must not create a logger")

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)

    run_ppo._run_training_session(
        trainer=trainer,
        run_dir=tmp_path,
        cfg=cfg,
        log_mode=LogMode.DEBUG,
        identity=None,
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=distributed,
    )

    assert trainer.iterations == 1
    assert trainer.checkpoints == []


def test_run_training_session_closes_logger_and_skips_final_checkpoint_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _full_config()
    trainer = _FakeTrainer()
    logger = _FakeLogger()

    def raise_from_loop(**_kwargs: object) -> int:
        raise RuntimeError("training failed")

    def create_fake_logger(*_args: object, **_kwargs: object) -> _FakeLogger:
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", create_fake_logger)
    monkeypatch.setattr(run_ppo, "_run_training_loop", raise_from_loop)

    with pytest.raises(RuntimeError, match="training failed"):
        run_ppo._run_training_session(
            trainer=trainer,
            run_dir=tmp_path,
            cfg=cfg,
            log_mode=LogMode.DEBUG,
            identity=_identity(),
            env_steps_per_iteration=8,
            max_env_steps=8,
            max_runtime_seconds=None,
            distributed=DistributedContext.single_process_cpu(),
        )

    assert logger.closed
    assert logger.close_exit_codes == [1]
    assert trainer.checkpoints == []


def test_ppo_trainer_write_checkpoint_includes_training_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.optimizer_steps = 7
    trainer.player_step_total = 19
    trainer.total_games_played = 23
    trainer.total_active_entities = 29
    trainer.target_kl_exceeded_total = 3
    path = tmp_path / "checkpoint.pt"

    trainer.write_checkpoint(
        path,
        env_steps=512,
        wandb_run_id="run-abc",
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 512
    assert checkpoint["optimizer_steps"] == 7
    assert checkpoint["player_step_total"] == 19
    assert checkpoint["total_games_played"] == 23
    assert checkpoint["total_active_entities"] == 29
    assert checkpoint["target_kl_exceeded_total"] == 3
    assert checkpoint["wandb_run_id"] == "run-abc"
    assert checkpoint["model"].keys() == model.state_dict().keys()
    assert "state" in checkpoint["optimizer"]
    assert checkpoint["lr_scheduler"] == scheduler.state_dict()
    assert set(checkpoint) == {
        "model",
        "optimizer",
        "lr_scheduler",
        "env_steps",
        "optimizer_steps",
        "player_step_total",
        "total_games_played",
        "total_active_entities",
        "target_kl_exceeded_total",
        "wandb_run_id",
    }
    assert set(checkpoint) == CHECKPOINT_KEYS
    metadata = run_ppo._checkpoint_metadata(checkpoint, path=path)
    assert metadata.env_steps == 512
    assert metadata.total_active_entities == 29
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()


def test_run_ppo_checkpoint_keys_derive_from_the_trainer_key_set(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    checkpoint: dict[str, object] = {
        "model": {},
        "optimizer": {},
        "lr_scheduler": None,
        "env_steps": 1,
        "optimizer_steps": 1,
        "player_step_total": 1,
        "total_games_played": 1,
        "total_active_entities": 1,
        "target_kl_exceeded_total": 0,
        "wandb_run_id": None,
    }
    assert set(checkpoint) == CHECKPOINT_KEYS
    assert OPTIONAL_CHECKPOINT_KEYS < CHECKPOINT_KEYS
    run_ppo._checkpoint_metadata(checkpoint, path=path)
    for optional in OPTIONAL_CHECKPOINT_KEYS:
        run_ppo._checkpoint_metadata(
            {key: value for key, value in checkpoint.items() if key != optional},
            path=path,
        )
    for required in CHECKPOINT_KEYS - OPTIONAL_CHECKPOINT_KEYS:
        with pytest.raises(ValueError, match="checkpoint keys must include"):
            run_ppo._checkpoint_metadata(
                {key: value for key, value in checkpoint.items() if key != required},
                path=path,
            )
    with pytest.raises(ValueError, match="checkpoint keys must include"):
        run_ppo._checkpoint_metadata({**checkpoint, "hidden_state": 0}, path=path)


def test_ppo_trainer_write_checkpoint_can_save_explicit_model(
    tmp_path: Path,
) -> None:
    trainer_model = torch.nn.Linear(2, 1)
    checkpoint_model = torch.nn.Linear(2, 1)
    for param in trainer_model.parameters():
        param.data.fill_(1.0)
    for param in checkpoint_model.parameters():
        param.data.fill_(3.0)
    optimizer = torch.optim.AdamW(trainer_model.parameters(), lr=0.001)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = trainer_model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = None
    trainer.optimizer_steps = 7
    trainer.player_step_total = 19
    trainer.total_games_played = 23
    trainer.total_active_entities = 29
    trainer.target_kl_exceeded_total = 3
    path = tmp_path / "checkpoint.pt"

    trainer.write_checkpoint(
        path,
        env_steps=0,
        wandb_run_id="run-abc",
        model=checkpoint_model,
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["env_steps"] == 0
    assert torch.equal(
        checkpoint["model"]["weight"],
        checkpoint_model.state_dict()["weight"],
    )
    assert not torch.equal(
        checkpoint["model"]["weight"],
        trainer_model.state_dict()["weight"],
    )


def test_ppo_trainer_load_checkpoint_restores_training_state(tmp_path: Path) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.001)
    src_scheduler = torch.optim.lr_scheduler.LambdaLR(
        src_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    dst_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dst_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_scheduler.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = src_scheduler
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = dst_scheduler
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    metadata = dst_trainer.load_checkpoint(path)

    assert metadata.env_steps == 2048
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.total_active_entities == 43
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 11
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 5
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)
    assert dst_optimizer.state_dict()["state"]
    assert dst_scheduler.state_dict() == src_scheduler.state_dict()


def test_ppo_trainer_load_checkpoint_defaults_missing_total_active_entities(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = None
    trainer.optimizer_steps = 0
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 17
    trainer.target_kl_exceeded_total = 0
    trainer.device = torch.device("cpu")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None,
            "env_steps": 1,
            "optimizer_steps": 0,
            "player_step_total": 5,
            "total_games_played": 7,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-abc",
        },
        path,
    )

    metadata = trainer.load_checkpoint(path)

    assert metadata.total_active_entities == 0
    assert trainer.total_active_entities == 0


def test_ppo_trainer_load_model_weights_keeps_only_logging_counters(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.001)
    src_scheduler = torch.optim.lr_scheduler.LambdaLR(
        src_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    dst_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dst_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_scheduler.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = src_scheduler
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = dst_scheduler
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")
    scheduler_state_before = dst_scheduler.state_dict()

    metadata = dst_trainer.load_model_weights(path)

    assert metadata.env_steps == 2048
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.total_active_entities == 43
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 0
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 0
    assert not dst_optimizer.state_dict()["state"]
    assert dst_scheduler.state_dict() == scheduler_state_before
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)


def test_ppo_trainer_load_model_weights_can_load_optimizer_without_scheduler(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.01)
    src_scheduler = torch.optim.lr_scheduler.LambdaLR(
        src_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    dst_scheduler = torch.optim.lr_scheduler.LambdaLR(
        dst_optimizer,
        lr_lambda=lambda step: 0.5**step,
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_scheduler.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = src_scheduler
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = dst_scheduler
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")
    scheduler_state_before = dst_scheduler.state_dict()

    metadata = dst_trainer.load_model_weights(path, load_optimizer=True)

    assert metadata.env_steps == 2048
    assert metadata.player_step_total == 37
    assert metadata.total_games_played == 41
    assert metadata.total_active_entities == 43
    assert metadata.wandb_run_id == "run-abc"
    assert dst_trainer.optimizer_steps == 0
    assert dst_trainer.player_step_total == 37
    assert dst_trainer.total_games_played == 41
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 0
    assert dst_optimizer.state_dict()["state"]
    assert dst_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert dst_scheduler.state_dict() == scheduler_state_before
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)


def test_ppo_trainer_load_model_weights_rejects_optimizer_state_shape_mismatch(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Linear(2, 1)
    dst_model = torch.nn.Linear(2, 1)
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=0.001)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=0.001)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = None
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict)
    optimizer_state = checkpoint["optimizer"]
    assert isinstance(optimizer_state, dict)
    state = optimizer_state["state"]
    assert isinstance(state, dict)
    param_state = next(iter(state.values()))
    assert isinstance(param_state, dict)
    param_state["exp_avg"] = torch.ones(3, 3)
    torch.save(checkpoint, path)

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = None
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    with pytest.raises(ValueError, match="optimizer state tensor 'exp_avg' shape"):
        dst_trainer.load_model_weights(path, load_optimizer=True)


def test_ppo_trainer_load_model_weights_can_load_composite_optimizer(
    tmp_path: Path,
) -> None:
    src_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    dst_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    src_optimizer = CompositeOptimizer(
        [
            torch.optim.AdamW(src_model[0].parameters(), lr=0.001),
            torch.optim.AdamW(src_model[1].parameters(), lr=0.002),
        ]
    )
    dst_optimizer = CompositeOptimizer(
        [
            torch.optim.AdamW(dst_model[0].parameters(), lr=0.01),
            torch.optim.AdamW(dst_model[1].parameters(), lr=0.02),
        ]
    )
    for param in src_model.parameters():
        param.data.fill_(3.0)
    src_optimizer.zero_grad()
    src_model(torch.ones(1, 2)).sum().backward()
    src_optimizer.step()
    src_trainer = PPOTrainer.__new__(PPOTrainer)
    src_trainer.model = src_model
    src_trainer.optimizer = src_optimizer
    src_trainer.lr_scheduler = None
    src_trainer.optimizer_steps = 11
    src_trainer.player_step_total = 37
    src_trainer.total_games_played = 41
    src_trainer.total_active_entities = 43
    src_trainer.target_kl_exceeded_total = 5
    path = tmp_path / "checkpoint.pt"
    src_trainer.write_checkpoint(path, env_steps=2048, wandb_run_id="run-abc")

    dst_trainer = PPOTrainer.__new__(PPOTrainer)
    dst_trainer.model = dst_model
    dst_trainer.optimizer = dst_optimizer
    dst_trainer.lr_scheduler = None
    dst_trainer.optimizer_steps = 0
    dst_trainer.player_step_total = 0
    dst_trainer.total_games_played = 0
    dst_trainer.total_active_entities = 0
    dst_trainer.target_kl_exceeded_total = 0
    dst_trainer.device = torch.device("cpu")

    metadata = dst_trainer.load_model_weights(path, load_optimizer=True)

    assert metadata.env_steps == 2048
    assert metadata.total_active_entities == 43
    assert dst_trainer.optimizer_steps == 0
    assert dst_trainer.total_active_entities == 43
    assert dst_trainer.target_kl_exceeded_total == 0
    assert dst_optimizer.optimizers[0].state_dict()["state"]
    assert dst_optimizer.optimizers[1].state_dict()["state"]
    assert dst_optimizer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.01)
    assert dst_optimizer.optimizers[1].param_groups[0]["lr"] == pytest.approx(0.02)
    for src_param, dst_param in zip(
        src_model.parameters(),
        dst_model.parameters(),
        strict=True,
    ):
        assert torch.equal(src_param, dst_param)


def test_ppo_trainer_load_checkpoint_rejects_scheduler_mismatch(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.optimizer_steps = 0
    trainer.player_step_total = 0
    trainer.total_games_played = 0
    trainer.total_active_entities = 0
    trainer.target_kl_exceeded_total = 0
    trainer.device = torch.device("cpu")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None,
            "env_steps": 1,
            "optimizer_steps": 0,
            "player_step_total": 0,
            "total_games_played": 0,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-abc",
        },
        path,
    )

    with pytest.raises(ValueError, match="missing lr_scheduler state"):
        trainer.load_checkpoint(path)


# Rebuild Task 3.2 (L1): raw-bank evaluation outcome -----------------------------


def _two_seat_obs(n_envs: int) -> ObsBatch:
    return ObsBatch(
        planets=torch.zeros((n_envs, 1, 1)),
        orbiting_planets=torch.zeros((n_envs, 1), dtype=torch.bool),
        fleets=torch.zeros((n_envs, 1, 1)),
        comets=torch.zeros((n_envs, 1, 1)),
        entity_mask=torch.zeros((n_envs, 1), dtype=torch.bool),
        still_playing=torch.ones((n_envs, 2), dtype=torch.bool),
        global_features=torch.zeros((n_envs, 1)),
        action_mask=PureActionMask(
            can_act=torch.zeros((n_envs, 2, ACTION_ENTITY_SLOTS), dtype=torch.bool),
            max_launch=torch.zeros((n_envs, 2, ACTION_ENTITY_SLOTS), dtype=torch.int64),
        ),
    )


class _ShapedReturnMisranksBanksEnv:
    """Two-seat one-step fake game whose shaped return misranks the banks.

    Env 0: the candidate's seat ends with more money, but its shaped return is
    lower. Env 1: equal banks (a draw) with unequal shaped returns.
    """

    def __init__(self, n_envs: int) -> None:
        assert n_envs == 2
        self.n_envs = n_envs
        self.candidate_seat = torch.full((n_envs,), -1, dtype=torch.int64)

    def reset(self) -> ObsBatch:
        return _two_seat_obs(self.n_envs)

    def step(
        self, actions: object
    ) -> tuple[ObsBatch, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        assert isinstance(actions, run_ppo.PureActions)
        # The candidate launches and the incumbent does not, so its seat is visible.
        launches = actions.launch[:, :, 0, 0]
        assert launches.sum(dim=1).tolist() == [1, 1]
        self.candidate_seat = launches.to(torch.int64).argmax(dim=1)
        rewards = torch.zeros((self.n_envs, 2))
        for env, shaped in enumerate((0.4, 0.1)):
            rewards[env, self.candidate_seat[env]] = -shaped
            rewards[env, 1 - self.candidate_seat[env]] = shaped
        dones = torch.ones((self.n_envs, 2), dtype=torch.bool)
        return _two_seat_obs(self.n_envs), rewards, dones, {}

    def terminal_metrics(self, env_index: int) -> dict[str, float]:
        seat = int(self.candidate_seat[env_index])
        candidate_bank, incumbent_bank = ((3000.0, 2000.0), (2500.0, 2500.0))[env_index]
        banks = [0.0, 0.0]
        banks[seat], banks[1 - seat] = candidate_bank, incumbent_bank
        winner = -1.0 if banks[0] == banks[1] else float(banks[1] > banks[0])
        return {
            "bank_0": banks[0],
            "bank_1": banks[1],
            "margin_0": banks[0] - banks[1],
            "winner": winner,
            "episode_steps": 1.0,
        }


class _LaunchPolicy:
    def __init__(self, *, launch: bool) -> None:
        self.launch = launch

    def initial_hidden_state(
        self,
        batch_size: int,  # noqa: ARG002
        *,
        device: torch.device,  # noqa: ARG002
    ) -> None:
        return None

    def reset_hidden_state(
        self,
        hidden_state: None,  # noqa: ARG002
        dones: torch.Tensor,  # noqa: ARG002
    ) -> None:
        return None

    def __call__(self, obs: ObsBatch, *, deterministic: bool) -> SimpleNamespace:
        assert not deterministic
        shape = (obs.global_features.shape[0], 2, ACTION_ENTITY_SLOTS, 1)
        return SimpleNamespace(
            actions=run_ppo.PureActions(
                launch=torch.full(shape, self.launch, dtype=torch.bool),
                angle=torch.zeros(shape),
                ships=torch.ones(shape, dtype=torch.int64),
            ),
            next_hidden_state=None,
        )


def _kaggriculture_eval_config() -> FullConfig:
    # The shipped Kaggriculture config loads through the registered env schema.
    return FullConfig.from_file(_CONFIGS / "kaggriculture.yaml")


def test_kaggriculture_evaluation_decides_winners_by_raw_banks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_for: list[int] = []

    def fake_create_eval_env(
        cfg: FullConfig, *, n_envs: int, device: torch.device, env_steps: int
    ) -> _ShapedReturnMisranksBanksEnv:
        assert device.type == "cpu"
        assert isinstance(cfg.env.obs_spec, KaggricultureObsConfig)
        built_for.append(env_steps)
        return _ShapedReturnMisranksBanksEnv(n_envs)

    monkeypatch.setattr(run_ppo, "_create_eval_env", fake_create_eval_env)

    stats, stats_by_player_count, env_metrics, steps = run_ppo._evaluate_games(
        current_model=_LaunchPolicy(launch=True),
        last_best_model=_LaunchPolicy(launch=False),
        cfg=_kaggriculture_eval_config(),
        n_games=2,
        n_envs=2,
        device=torch.device("cpu"),
        env_steps=20_000_000,
    )

    # Shaped returns would give the incumbent both games (win rate 0.0); the
    # banks give the candidate one win and one draw.
    assert built_for == [20_000_000]
    assert steps == 2
    assert stats.model_games == [2, 2]
    assert stats.wins == [1.5, 0.5]
    assert stats_by_player_count[2].wins == [1.5, 0.5]
    assert env_metrics["candidate_bank"] == [3000.0, 2500.0]
    assert env_metrics["last_best_bank"] == [2000.0, 2500.0]
    assert env_metrics["candidate_bank_margin"] == [1000.0, 0.0]


def test_orbit_evaluation_scores_stay_the_training_returns() -> None:
    returns = torch.tensor([1.0, -1.0, 0.0, 0.0])

    scores, metrics = run_ppo._evaluation_scores_and_metrics(
        _config_with_envs(2),
        {"game_length_mean": 12.0},
        returns,
        torch.tensor([0, 1, -1, -1]),
    )

    assert scores is returns
    assert metrics == {}


def test_candidate_bank_metrics_require_one_candidate_per_game() -> None:
    with pytest.raises(ValueError, match="one candidate and one incumbent"):
        run_ppo._candidate_bank_metrics(
            torch.tensor([1.0, 2.0], dtype=torch.float64), torch.tensor([0, 0])
        )


# Rebuild Task 3.3 (L12): fresh reproducible seed per evaluation --------------------


def test_evaluation_seed_is_reproducible_and_changes_per_evaluation() -> None:
    first = run_ppo._evaluation_seed(base_seed=7, env_steps=20_000_000)

    assert first == run_ppo._evaluation_seed(base_seed=7, env_steps=20_000_000)
    assert first != run_ppo._evaluation_seed(base_seed=7, env_steps=40_000_000)
    assert first != run_ppo._evaluation_seed(base_seed=8, env_steps=20_000_000)
    checkpoints = [
        run_ppo._evaluation_seed(base_seed=7, env_steps=step)
        for step in range(0, 20_000_000 * 1_000, 20_000_000)
    ]
    assert len(set(checkpoints)) == len(checkpoints)
    adjacent = [
        run_ppo._evaluation_seed(base_seed=base, env_steps=steps)
        for base in range(4)
        for steps in range(1_000, 1_256)
    ]
    assert len(set(adjacent)) == len(adjacent)


def test_evaluation_seed_fits_the_native_seed_with_headroom() -> None:
    # engine_rs `Game::new` takes an i64 seed and contract v4 requires seed >= 0;
    # one evaluation env consumes one seed per construction and per auto-reset.
    native_limit = 2**63
    band_floor = 2**62
    extremes = [
        run_ppo._evaluation_seed(base_seed=base, env_steps=steps)
        for base in (0, 1, 2**61 - 1)
        for steps in (0, 1, 20_000_000, 2**61 - 1)
    ]
    for seed in extremes:
        assert band_floor <= seed < band_floor + 2**61
        assert seed + 2**61 <= native_limit


def test_evaluation_seed_distinguishes_one_input_at_a_time_not_pairs() -> None:
    # Pins the documented limit: the mix is a bijection in each input while the
    # other is fixed, not an injection over (base_seed, env_steps) pairs, and it
    # does not keep consecutive seed ranges apart. Counterexamples from Codex's
    # 3.2/3.3 verification (verify-3.2-3.3-r1).
    same = run_ppo._evaluation_seed(base_seed=0, env_steps=0)
    other_run = run_ppo._evaluation_seed(
        base_seed=1, env_steps=2_131_737_497_183_550_101
    )
    same_run_later = run_ppo._evaluation_seed(
        base_seed=0, env_steps=787_325_655_728_545_358
    )

    assert other_run == same
    assert same_run_later == same + 1


@pytest.mark.parametrize(
    ("base_seed", "env_steps"),
    [(-1, 0), (2**61, 0), (0, -1), (0, 2**61)],
)
def test_evaluation_seed_rejects_values_outside_the_seed_band(
    base_seed: int, env_steps: int
) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 2\*\*61\)"):
        run_ppo._evaluation_seed(base_seed=base_seed, env_steps=env_steps)


def test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    def fake_vectorized_env(**kwargs: object) -> str:
        built.append(kwargs)
        return "orbit-env"

    monkeypatch.setattr(run_ppo, "VectorizedEnv", fake_vectorized_env)
    cfg = _config_with_envs(2)
    assert isinstance(cfg.env, EnvConfig)

    env = run_ppo._create_eval_env(
        cfg, n_envs=4, device=torch.device("cpu"), env_steps=1000
    )

    assert env == "orbit-env"
    assert built == [
        {
            "n_envs": 4,
            "obs_spec": cfg.env.obs_spec,
            "action_spec": cfg.env.action_spec,
            "two_player_weight": cfg.env.two_player_weight,
            "reward_mode": cfg.env.reward_mode,
            "pin_memory": False,
        }
    ]
    kaggriculture = run_ppo._create_eval_env(
        _kaggriculture_eval_config(),
        n_envs=1,
        device=torch.device("cpu"),
        env_steps=1000,
    )
    assert isinstance(kaggriculture, KaggricultureVectorizedEnv)
    assert not kaggriculture.pin_memory_enabled
    assert len(built) == 1


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_kaggriculture_eval_factory_arguments(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    cfg = _kaggriculture_eval_config()
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"seed": 7})})
    calls: list[tuple[object, dict[str, object]]] = []
    sentinel = object()

    def fake_create_env(config: object, **kwargs: object) -> object:
        calls.append((config, kwargs))
        return sentinel

    monkeypatch.setattr(run_ppo, "create_env", fake_create_env)
    device = torch.device(device_type)
    assert (
        run_ppo._create_eval_env(cfg, n_envs=2, device=device, env_steps=123)
        is sentinel
    )
    assert calls == [
        (
            cfg.env,
            {
                "n_envs": 2,
                "base_seed": run_ppo._evaluation_seed(base_seed=7, env_steps=123),
                "rank": 0,
                "world_size": 1,
                "pin_memory": device_type == "cuda",
                "transfer_device": device,
            },
        )
    ]


def test_kaggriculture_native_evaluations_draw_fresh_reproducible_worlds() -> None:
    cfg = _kaggriculture_eval_config()
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    cfg = cfg.model_copy(
        update={
            "env": cfg.env.model_copy(
                update={
                    "seed": 7,
                    "native_threads": 1,
                    "config": cfg.env.config.model_copy(
                        update={
                            "episode_steps": 3,
                            "turns_per_day": 1,
                            "weed_spawn_chance": 0.5,
                        }
                    ),
                }
            )
        }
    )

    def evaluate(
        env_steps: int,
    ) -> tuple[tuple[int, ...], list[object], list[tuple[float, float]]]:
        env = run_ppo._create_eval_env(
            cfg, n_envs=2, device=torch.device("cpu"), env_steps=env_steps
        )
        assert isinstance(env, KaggricultureVectorizedEnv)
        seed = run_ppo._evaluation_seed(base_seed=7, env_steps=env_steps)
        assert env.seed_state() == (seed + 2, (seed, seed + 1))
        # Match the evaluation caller's explicit reset after factory construction.
        obs = env.reset()
        assert env.seed_state() == (seed + 4, (seed + 2, seed + 3))
        game_seeds = env.seed_state()[1]
        snapshots = [env.state_snapshot(i) for i in range(2)]
        for step in range(2):
            programs = [
                tuple(
                    {
                        "farmer": ["PASS"],
                        "hands": [["PASS"]]
                        * (int(obs.actor_mask[i, s, :MAX_ACTORS].sum()) - 1),
                        "market": [],
                    }
                    for s in range(2)
                )
                for i in range(2)
            ]
            actions = encode_actions(programs, obs, action_spec=env.action_spec)
            obs, _rewards, dones, _metrics = env.step(actions)
            obs.check_contract()
            assert bool(dones.all()) == (step == 1)
            if step == 0:
                # Initial boards are deterministic. Native daily RNG is the
                # first observable effect of the distinct game seeds.
                snapshots.extend(env.state_snapshot(i) for i in range(2))
        banks = []
        for i in range(2):
            terminal = env.terminal_metrics(i)
            assert terminal is not None
            assert terminal["episode_steps"] == 2
            banks.append((terminal["bank_0"], terminal["bank_1"]))
        return game_seeds, snapshots, banks

    first_seeds, first_worlds, first_banks = evaluate(1000)
    different_seeds, different_worlds, _different_banks = evaluate(2000)
    repeated_seeds, repeated_worlds, repeated_banks = evaluate(1000)
    assert first_seeds != different_seeds
    assert first_worlds[:2] == different_worlds[:2]
    assert first_worlds[2:] != different_worlds[2:]
    assert first_seeds == repeated_seeds
    assert first_worlds == repeated_worlds
    assert first_banks == repeated_banks


def test_kaggriculture_eval_env_is_independent() -> None:
    cfg = _kaggriculture_eval_config()
    first = run_ppo._create_eval_env(
        cfg, n_envs=1, device=torch.device("cpu"), env_steps=1000
    )
    second = run_ppo._create_eval_env(
        cfg, n_envs=1, device=torch.device("cpu"), env_steps=1000
    )
    assert isinstance(first, KaggricultureVectorizedEnv)
    assert isinstance(second, KaggricultureVectorizedEnv)
    assert first is not second
    snapshot, seeds = second.state_snapshot(0), second.seed_state()
    first.reset()
    assert second.state_snapshot(0) == snapshot
    assert second.seed_state() == seeds


def test_kaggriculture_policy_evaluation_runs_native_games() -> None:
    cfg = FullConfig.from_file(
        _CONFIGS / "kaggriculture.yaml",
        overrides={
            "env.n_envs": 2,
            "env.native_threads": 1,
            "env.pin_memory": False,
            "env.config.episodeSteps": 3,
            "model.embed_dim": 16,
            "model.depth": 1,
            "model.n_heads": 1,
            "model.mlp_ratio": 1,
            "model.n_scratch_tokens": 0,
            "rl.dtype": "float32",
            "rl.model_compile": "none",
            "rl.segments_per_minibatch": 1,
            "rl.gradient_accumulation_steps": 1,
            "rl.eval_replay_games": 0,
        },
    )
    torch.manual_seed(31)
    current, _ = run_ppo._create_training_model_for_config(
        cfg, device=torch.device("cpu"), reset_parameters=True
    )
    last_best = run_ppo._create_eval_model_from_weights(
        current, cfg, device=torch.device("cpu")
    )
    current.eval()
    with torch.no_grad():
        stats, by_count, metrics, steps = run_ppo._evaluate_games(
            current_model=current,
            last_best_model=last_best,
            cfg=cfg,
            n_games=2,
            n_envs=2,
            device=torch.device("cpu"),
            env_steps=1000,
        )
    margins = metrics["candidate_bank_margin"]
    expected_wins = sum(1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in margins)
    assert stats.model_games == [2, 2]
    assert stats.wins == [expected_wins, 2 - expected_wins]
    assert by_count[2].wins == stats.wins
    assert steps == 4
    assert metrics["episode_steps"] == [2, 2]
    for candidate, incumbent, margin in zip(
        metrics["candidate_bank"], metrics["last_best_bank"], margins, strict=True
    ):
        assert torch.isfinite(torch.tensor([candidate, incumbent, margin])).all()
        assert candidate - incumbent == margin
    logged = run_ppo._eval_env_metrics(metrics)
    assert logged["eval/candidate_bank_margin"] == sum(margins) / 2


# Rebuild Task 3.3: promotion telemetry ----------------------------------------------


@pytest.mark.parametrize(("win_rate", "promoted"), [(0.7, 1.0), (0.69, 0.0)])
def test_run_training_loop_logs_promotion_telemetry_and_evaluation_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    win_rate: float,
    promoted: float,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = _FakeTrainer()
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)
    evaluated_at: list[object] = []

    def fake_evaluate_against_last_best(**kwargs: object) -> dict[str, float]:
        evaluated_at.append(kwargs["env_steps"])
        return {"eval/win_rate_against_last_best": win_rate, "eval/games": 2.0}

    monkeypatch.setattr(
        run_ppo, "_evaluate_against_last_best", fake_evaluate_against_last_best
    )

    run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        env_steps_per_iteration=1000,
        max_env_steps=2000,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    assert evaluated_at == [1000, 2000]
    eval_logs = [
        (metrics, step) for metrics, step in logger.logged if "eval/games" in metrics
    ]
    assert [step for _metrics, step in eval_logs] == [1000, 2000]
    for metrics, _step in eval_logs:
        assert metrics["eval/promoted"] == promoted
        assert metrics["eval/promotion_threshold"] == pytest.approx(0.7)
        assert metrics["eval/win_rate_against_last_best"] == win_rate
    promotions = [
        checkpoint
        for checkpoint in trainer.checkpoints
        if checkpoint[0].name == "checkpoint_last_best.pt" and checkpoint[1] > 0
    ]
    assert len(promotions) == (2 if promoted else 0)


class _FailingLastBestWriteTrainer(_FakeTrainer):
    def write_checkpoint(
        self,
        path: Path,
        *,
        env_steps: int,
        wandb_run_id: str | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        if path.name == run_ppo.CHECKPOINT_LAST_BEST and env_steps > 0:
            raise OSError("injected promoted checkpoint write failure")
        super().write_checkpoint(
            path, env_steps=env_steps, wandb_run_id=wandb_run_id, model=model
        )


@pytest.mark.parametrize("failing_phase", ["refresh", "checkpoint"])
def test_run_training_loop_reports_promotion_only_after_it_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_phase: str,
) -> None:
    cfg = _full_config(checkpoint_freq=1000)
    trainer = (
        _FailingLastBestWriteTrainer()
        if failing_phase == "checkpoint"
        else _FakeTrainer()
    )
    logger = _FakeLogger()
    _patch_eval_model_from_weights(monkeypatch)
    monkeypatch.setattr(
        run_ppo,
        "_evaluate_against_last_best",
        lambda **_kwargs: {"eval/win_rate_against_last_best": 0.7, "eval/games": 2.0},
    )
    if failing_phase == "refresh":

        def fail_refresh(*_args: object) -> None:
            raise RuntimeError("injected incumbent refresh failure")

        monkeypatch.setattr(run_ppo, "_refresh_eval_model_from_weights", fail_refresh)

    with pytest.raises((RuntimeError, OSError), match="injected"):
        run_ppo._run_training_loop(
            trainer=trainer,
            logger=logger,
            run_dir=tmp_path,
            cfg=cfg,
            env_steps_per_iteration=1000,
            max_env_steps=1000,
            max_runtime_seconds=None,
            dist_ctx=DistributedContext.single_process_cpu(),
        )

    # The iteration's training metrics were logged, but no evaluation record
    # (and so no `eval/promoted`) exists for a promotion that never completed.
    assert [step for _metrics, step in logger.logged] == [1000]
    assert all("eval/promoted" not in metrics for metrics, _step in logger.logged)


# --- Kaggriculture startup workload check (plan Task 3.4) ---------------------
#
# These drive ``main`` from the shipped YAML through the real
# ``FullConfig.from_file``, CLI overrides and resume adaptation. Only the release
# build check, torch configuration and the distributed session are stubbed; the
# allocation steps are replaced by sentinels that record and fail if reached.

_CONFIGS = Path(__file__).parents[2] / "configs"


def _patch_kaggriculture_startup(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    calls: list[str],
    *,
    log_mode: LogMode = LogMode.DEBUG,
) -> None:
    def sentinel(name: str) -> object:
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"{name} ran before the startup workload check")

        return fail

    monkeypatch.setattr(
        sys, "argv", ["run_ppo.py", *argv, "--log-mode", log_mode.value]
    )
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    for name in (
        "_create_run_dir",
        "VectorizedEnv",
        "create_env",
        "_create_training_model_for_config",
    ):
        monkeypatch.setattr(run_ppo, name, sentinel(name))
    # Hermetic: the compile-stack check reads the probed GPU stack, not the host.
    monkeypatch.setattr(run_ppo, "installed_compile_stack", lambda: _PROBED_STACK)
    # Hermetic W&B credentials: online launches see a key, never the host's.
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")


def _teacher_source_argv(tmp_path: Path) -> list[str]:
    """A metadata-only teacher checkpoint at env step 0.

    Task 4.4 requires a teacher source at a fresh Kaggriculture last-best launch;
    Task 3.1's startup reads its ``env_steps`` before allocation. These launches
    stop before the trainer loads its (empty) weights.
    """
    weights = tmp_path / "bc_best.pt"
    _write_metadata_checkpoint(weights, env_steps=0)
    return ["--load-model-weights", str(weights)]


def _kaggriculture_teacher_checkpoint(tmp_path: Path) -> Path:
    """A real tiny-model teacher checkpoint (with its config) for rl.teacher_init."""
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture.yaml")
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    cfg.to_file(teacher_dir / "config.yaml")
    torch.manual_seed(7)
    model = run_ppo._create_eval_model_for_config(
        cfg, device=torch.device("cpu"), roundtrip_lora_base=False
    )
    path = teacher_dir / "bc_best.pt"
    torch.save({"model": model.state_dict()}, path)
    return path


_PROBED_STACK = InstalledCompileStack(
    torch="2.9.0+cu128",
    triton="3.5.0",
    cuda_available=True,
    nvidia_drivers=("595.91.07",),
)


def test_main_rejects_an_unprobed_compile_stack_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    argv = [
        str(_CONFIGS / "kaggriculture_2rank.yaml"),
        str(tmp_path / "runs"),
        *_teacher_source_argv(tmp_path),
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)
    monkeypatch.setattr(
        run_ppo,
        "installed_compile_stack",
        lambda: InstalledCompileStack(
            torch="2.10.0+cu128",
            triton="3.5.0",
            cuda_available=True,
            nvidia_drivers=("595.91.07",),
        ),
    )

    with pytest.raises(RuntimeError, match=re.escape("unprobed torch 2.10.0")):
        run_ppo.main()

    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_startup_compile_stack_check_prints_the_checked_stack(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    assert cfg.rl.model_compile == "trunk"
    monkeypatch.setattr(run_ppo, "installed_compile_stack", lambda: _PROBED_STACK)

    report = run_ppo._check_compile_stack(
        cfg.model, rl=cfg.rl, distributed=DistributedContext.single_process_cpu()
    )

    assert report == CompileStackReport(
        torch="2.9.0+cu128", triton="3.5.0", nvidia_driver="595.91.07"
    )
    assert capsys.readouterr().out == (
        "Compile stack check: torch 2.9.0+cu128; triton 3.5.0; "
        "NVIDIA driver 595.91.07\n"
    )


def test_startup_compile_stack_check_skips_orbit_and_uncompiled_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run_ppo,
        "installed_compile_stack",
        lambda: pytest.fail("the stack check must not run"),
    )
    orbit = _full_config()
    kaggriculture = FullConfig.from_file(_CONFIGS / "kaggriculture.yaml")
    assert kaggriculture.rl.model_compile == "none"
    context = DistributedContext.single_process_cpu()

    assert (
        run_ppo._check_compile_stack(orbit.model, rl=orbit.rl, distributed=context)
        is None
    )
    assert (
        run_ppo._check_compile_stack(
            kaggriculture.model, rl=kaggriculture.rl, distributed=context
        )
        is None
    )
    assert capsys.readouterr().out == ""


def _headroom_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("GEMM workload")]


@pytest.mark.parametrize("seed", [2**61, 2**62, 2**63 - 1])
def test_main_rejects_kaggriculture_seed_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch,
        [
            str(_CONFIGS / "kaggriculture.yaml"),
            str(tmp_path / "runs"),
            *_teacher_source_argv(tmp_path),
            "-o",
            f"env.seed={seed}",
            "rl.eval_replay_games=0",
        ],
        calls,
    )
    with pytest.raises(ValueError, match=r"env\.seed must be in"):
        run_ppo.main()
    assert calls == []


def test_main_rejects_kaggriculture_replay_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch,
        [
            str(_CONFIGS / "kaggriculture.yaml"),
            str(tmp_path / "runs"),
            *_teacher_source_argv(tmp_path),
            "-o",
            "rl.eval_replay_games=1",
        ],
        calls,
    )
    with pytest.raises(ValueError, match=r"Task 7\.3"):
        run_ppo.main()
    assert calls == []


def test_main_rejects_kaggriculture_seed_budget_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch,
        [
            str(_CONFIGS / "kaggriculture.yaml"),
            str(tmp_path / "runs"),
            *_teacher_source_argv(tmp_path),
            "--max-env-steps",
            str(2**61),
        ],
        calls,
    )
    with pytest.raises(ValueError, match="seed budget"):
        run_ppo.main()
    assert calls == []


def test_select_kaggriculture_actions_preserves_seats_and_native_storage() -> None:
    a = KaggricultureActions(
        tokens=torch.arange(2 * 2 * 12 * 252).view(2, 2, 12, 252).transpose(-1, -2),
        lengths=torch.tensor([[2, 3], [4, 5]]).T,
    )
    b = KaggricultureActions(tokens=a.tokens + 10000, lengths=a.lengths + 3)
    use_a = torch.tensor([[True, False], [False, True]])
    result = run_ppo._select_actions(a, b, use_a)
    assert isinstance(result, KaggricultureActions)
    for field in ("tokens", "lengths"):
        tensor, first, second = (getattr(bundle, field) for bundle in (result, a, b))
        assert tensor.dtype == torch.int64
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert torch.equal(tensor[use_a], first[use_a])
        assert torch.equal(tensor[~use_a], second[~use_a])


def test_kaggriculture_session_forwards_offline_mode_and_shared_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _kaggriculture_eval_config()
    cfg = cfg.model_copy(
        update={"rl": cfg.rl.model_copy(update={"checkpoint_freq": 2})}
    )
    training = {"loss/total_loss": 1.0, "train/terminal_bank_0": 3000.0}
    evaluation = {
        "eval/win_rate_against_last_best": 0.5,
        "eval/candidate_bank_margin": 0.0,
    }
    trainer, logger = _FakeTrainer(metrics=training), _FakeWandbLogger(run_id="off1")
    identity = _identity(TelemetryMode.WANDB_OFFLINE)

    def make_logger(
        mode: LogMode, run_dir: Path, config: FullConfig, **kwargs: object
    ) -> _FakeLogger:
        assert mode == LogMode.WANDB
        assert run_dir == tmp_path
        assert config is cfg
        assert kwargs == {"identity": identity, "resume_run_id": None}
        return logger

    monkeypatch.setattr(run_ppo, "create_logger", make_logger)
    monkeypatch.setattr(
        run_ppo, "_evaluate_against_last_best", lambda **_kwargs: evaluation
    )
    _patch_eval_model_from_weights(monkeypatch)
    run_ppo._run_training_session(
        trainer=trainer,
        run_dir=tmp_path,
        cfg=cfg,
        log_mode=LogMode.WANDB,
        identity=identity,
        env_steps_per_iteration=2,
        max_env_steps=2,
        max_runtime_seconds=None,
        distributed=DistributedContext.single_process_cpu(),
    )
    assert trainer.iterations == 1
    assert logger.logged[0][0].items() >= training.items()
    assert logger.logged[1] == (
        {**evaluation, "eval/promoted": 0.0, "eval/promotion_threshold": 0.7},
        2,
    )
    assert logger.closed


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_parse_wandb_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            "config.yaml",
            "runs",
            "--wandb-mode",
            mode,
        ],
    )
    assert run_ppo._parse_args().wandb_mode == mode


def test_parse_wandb_mode_defaults_online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_ppo.py", "config.yaml", "runs"])
    assert run_ppo._parse_args().wandb_mode == "online"


def test_resume_offline_fails_before_reading_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ppo.py",
            "nonexistent-run",
            "--wandb-mode",
            "offline",
        ],
    )
    with pytest.raises(ValueError, match=r"resume.*offline"):
        run_ppo.main()


def test_kaggriculture_seed_budget_covers_resets_truncation_and_update_overshoot() -> (
    None
):
    cfg = _kaggriculture_eval_config()
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"seed": 2**61 - 1})})
    ctx = DistributedContext(
        device=torch.device("cpu"),
        rank=2,
        local_rank=2,
        world_size=3,
        initialized=False,
    )

    limit = run_ppo._kaggriculture_step_limit(cfg, ctx, max_env_steps=None)
    width = cfg.env.n_envs * ctx.world_size
    update = width * cfg.rl.horizon
    worst_steps = limit + update - 1
    assert cfg.env.seed + 2 * width + 2 * worst_steps <= 2**62
    assert worst_steps < 2**61
    assert run_ppo._kaggriculture_step_limit(cfg, ctx, max_env_steps=limit) == limit
    with pytest.raises(ValueError, match="seed budget"):
        run_ppo._kaggriculture_step_limit(cfg, ctx, max_env_steps=limit + 1)
    # Orbit has no new limit or seed checks.
    assert (
        run_ppo._kaggriculture_step_limit(_full_config(), ctx, max_env_steps=None)
        is None
    )


# --- Task 3.1 verify r1: resume seeds, seat crediting, launch forwarding -------


def _write_metadata_checkpoint(path: Path, *, env_steps: int) -> None:
    """A checkpoint with only the metadata that startup reads before allocation."""
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "lr_scheduler": None,
            "env_steps": env_steps,
            "optimizer_steps": 0,
            "player_step_total": 0,
            "total_games_played": 0,
            "target_kl_exceeded_total": 0,
            "wandb_run_id": "run-id",
        },
        path,
    )


def test_kaggriculture_resume_budget_counts_the_resumed_start() -> None:
    cfg = _kaggriculture_eval_config()
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"seed": 5})})
    ctx = DistributedContext.single_process_cpu()
    width = cfg.env.n_envs * ctx.world_size
    update = width * cfg.rl.horizon
    fresh = run_ppo._kaggriculture_step_limit(cfg, ctx, max_env_steps=None)
    start = 2**59
    limit = run_ppo._kaggriculture_step_limit(
        cfg, ctx, max_env_steps=None, start_env_steps=start
    )
    assert limit is not None
    assert fresh is not None
    assert limit < fresh
    base = run_ppo._kaggriculture_rollout_base_seed(5, start_env_steps=start)
    worst_steps = limit + update - 1
    # A resumed launch reserves construction, reset and two seeds per step from
    # its own base; the admitted limit keeps all of them below the band.
    assert base + 2 * width + 2 * (worst_steps - start) <= 2**62
    with pytest.raises(ValueError, match="seed budget"):
        run_ppo._kaggriculture_step_limit(
            cfg, ctx, max_env_steps=limit + 1, start_env_steps=start
        )
    with pytest.raises(ValueError, match="resumed at env step"):
        run_ppo._kaggriculture_step_limit(
            cfg, ctx, max_env_steps=None, start_env_steps=fresh
        )


def test_kaggriculture_resume_bases_clear_worst_case_launches() -> None:
    # Chain of launches, each resumed from a checkpoint one update (horizon 1,
    # the smallest) after its start, with changing world sizes. Worst case per
    # launch: construction and reset of every env, then an auto-reset and a
    # truncation on every transition.
    start = 0
    for global_envs in (1, 256, 3, 64, 2):
        base = run_ppo._kaggriculture_rollout_base_seed(7, start_env_steps=start)
        end = start + global_envs  # one update at horizon 1
        worst_seed = base + 2 * global_envs + 2 * (end - start) - 1
        next_base = run_ppo._kaggriculture_rollout_base_seed(7, start_env_steps=end)
        assert worst_seed < next_base
        start = end


def test_kaggriculture_resume_seeds_follow_every_first_launch_seed() -> None:
    cfg = _kaggriculture_eval_config()
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    env_config = cfg.env.model_copy(
        update={
            "seed": 41,
            "native_threads": 1,
            "pin_memory": False,
            "config": cfg.env.config.model_copy(update={"episode_steps": 3}),
        }
    )
    n_envs, transitions = 2, 4

    def launch(base_seed: int) -> KaggricultureVectorizedEnv:
        env = create_env(
            env_config,
            n_envs=n_envs,
            base_seed=base_seed,
            rank=0,
            world_size=1,
            pin_memory=False,
            transfer_device=torch.device("cpu"),
        )
        assert isinstance(env, KaggricultureVectorizedEnv)
        return env

    # First launch: construction, the trainer's reset, then passing turns that
    # finish (and auto-reset) every game twice.
    first = launch(41)
    first_seeds = set(first.seed_state()[1])
    obs = first.reset()
    first_seeds |= set(first.seed_state()[1])
    for _ in range(transitions):
        programs = [
            tuple(
                {
                    "farmer": ["PASS"],
                    "hands": [["PASS"]]
                    * (int(obs.actor_mask[i, s, :MAX_ACTORS].sum()) - 1),
                    "market": [],
                }
                for s in range(2)
            )
            for i in range(n_envs)
        ]
        obs, _rewards, dones, _metrics = first.step(
            encode_actions(programs, obs, action_spec=first.action_spec)
        )
        first_seeds |= set(first.seed_state()[1])
    assert int(dones.all(dim=1).sum()) == n_envs
    first_next = first.seed_state()[0]
    assert first_next > 41 + 2 * n_envs  # auto-resets drew seeds
    resume_base = run_ppo._kaggriculture_rollout_base_seed(
        41, start_env_steps=n_envs * transitions
    )
    resumed = launch(resume_base)
    resumed.reset()
    resumed_next, resumed_seeds = resumed.seed_state()
    assert min(resumed_seeds) >= first_next
    assert first_seeds.isdisjoint(range(resume_base, resumed_next))


def _write_online_attempt_receipt(run_dir: Path) -> None:
    """The ``attempts.jsonl`` a resume needs: one earlier online attempt."""
    identity = train_logging.plan_attempt(
        run_dir,
        job_type="ppo",
        resume=False,
        experiment_id=None,
        source_commit="v3-src-0",
        config_sha256="0" * 64,
        telemetry=TelemetryMode.WANDB_ONLINE,
    )
    train_logging.record_attempt(
        run_dir, identity, _FakeWandbLogger(run_id="on0"), start_env_steps=0
    )


def test_main_kaggriculture_resume_starts_a_disjoint_seed_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = FullConfig.from_file(_CONFIGS / "kaggriculture.yaml")
    saved = saved.model_copy(update={"env": saved.env.model_copy(update={"seed": 17})})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved.to_file(run_dir / "config.yaml")
    _write_metadata_checkpoint(run_dir / "checkpoint_final.pt", env_steps=1_000)
    _write_metadata_checkpoint(run_dir / "checkpoint_last_best.pt", env_steps=0)
    _write_online_attempt_receipt(run_dir)
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch, [str(run_dir)], calls, log_mode=LogMode.WANDB
    )
    seen: list[dict[str, object]] = []

    def factory(_config: object, **kwargs: object) -> None:
        seen.append(kwargs)
        raise AssertionError("factory reached")

    monkeypatch.setattr(run_ppo, "create_env", factory)
    with pytest.raises(AssertionError, match="factory reached"):
        run_ppo.main()
    assert seen[0]["base_seed"] == run_ppo._kaggriculture_rollout_base_seed(
        17, start_env_steps=1_000
    )
    assert seen[0]["base_seed"] != 17
    assert calls == []


def _kaggriculture_resume_run(tmp_path: Path, *, env_steps: int) -> Path:
    saved = FullConfig.from_file(_CONFIGS / "kaggriculture.yaml")
    saved = saved.model_copy(update={"env": saved.env.model_copy(update={"seed": 17})})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved.to_file(run_dir / "config.yaml")
    _write_metadata_checkpoint(run_dir / "checkpoint_final.pt", env_steps=env_steps)
    _write_metadata_checkpoint(run_dir / "checkpoint_last_best.pt", env_steps=0)
    _write_online_attempt_receipt(run_dir)
    return run_dir


def test_main_kaggriculture_resume_past_the_seed_budget_fails_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture.yaml")
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"seed": 17})})
    fresh = run_ppo._kaggriculture_step_limit(
        cfg, DistributedContext.single_process_cpu(), max_env_steps=None
    )
    assert fresh is not None
    # Admissible for a fresh launch, but a resume there has no budget left.
    run_dir = _kaggriculture_resume_run(tmp_path, env_steps=fresh)
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch, [str(run_dir)], calls, log_mode=LogMode.WANDB
    )
    with pytest.raises(ValueError, match=f"resumed at env step {fresh}"):
        run_ppo.main()
    assert calls == []


def _make_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True)
    return output_dir


class _CheckpointRewritingTrainer:
    """Rewrites the checkpoint with new env_steps before the trainer's own load."""

    rewritten_env_steps = 2_000

    def __init__(self, **_kwargs: object) -> None:
        pass

    def _reload(self, path: Path) -> Any:
        _write_metadata_checkpoint(path, env_steps=self.rewritten_env_steps)
        return run_ppo._checkpoint_metadata(
            torch.load(path, weights_only=False), path=path
        )

    def load_checkpoint(self, path: Path) -> Any:
        return self._reload(path)

    def load_model_weights(
        self,
        path: Path,
        *,
        load_optimizer: bool,
        fresh_state_keys: frozenset[str],
    ) -> Any:
        assert not load_optimizer
        assert fresh_state_keys == frozenset()
        return self._reload(path)


@pytest.mark.parametrize("launch", ["resume", "load_model_weights"])
def test_main_kaggriculture_fails_when_the_checkpoint_changes_during_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: str
) -> None:
    run_dir = _kaggriculture_resume_run(tmp_path, env_steps=1_000)
    checkpoint = run_dir / "checkpoint_final.pt"
    argv = (
        [str(run_dir)]
        if launch == "resume"
        else [
            str(run_dir / "config.yaml"),
            str(tmp_path / "runs"),
            "--load-model-weights",
            str(checkpoint),
        ]
    )
    calls: list[str] = []
    _patch_kaggriculture_startup(monkeypatch, argv, calls, log_mode=LogMode.WANDB)
    seen: list[dict[str, object]] = []

    def factory(_config: object, **kwargs: object) -> object:
        seen.append(kwargs)
        return object()

    monkeypatch.setattr(run_ppo, "_create_run_dir", _make_run_dir)
    monkeypatch.setattr(run_ppo, "create_env", factory)
    monkeypatch.setattr(
        run_ppo,
        "_create_training_model_for_config",
        lambda *_a, **_k: (torch.nn.Linear(1, 1), None),
    )
    monkeypatch.setattr(run_ppo, "configure_model_compile", lambda *_a: 0)
    monkeypatch.setattr(
        run_ppo,
        "create_optimizer",
        lambda model, _cfg: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(run_ppo, "create_lr_scheduler", lambda *_a: None)
    monkeypatch.setattr(run_ppo, "PPOTrainer", _CheckpointRewritingTrainer)
    with pytest.raises(RuntimeError, match="changed during startup"):
        run_ppo.main()
    # The env was seeded from the step read first, before the rewrite.
    assert seen[0]["base_seed"] == run_ppo._kaggriculture_rollout_base_seed(
        17, start_env_steps=1_000
    )
    assert calls == []


def test_main_kaggriculture_load_weights_continues_the_seed_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _kaggriculture_resume_run(tmp_path, env_steps=1_000)
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch,
        [
            str(run_dir / "config.yaml"),
            str(tmp_path / "runs"),
            "--load-model-weights",
            str(run_dir / "checkpoint_final.pt"),
        ],
        calls,
        log_mode=LogMode.WANDB,
    )
    monkeypatch.setattr(run_ppo, "_create_run_dir", _make_run_dir)
    seen: list[dict[str, object]] = []

    def factory(_config: object, **kwargs: object) -> None:
        seen.append(kwargs)
        raise AssertionError("factory reached")

    monkeypatch.setattr(run_ppo, "create_env", factory)
    with pytest.raises(AssertionError, match="factory reached"):
        run_ppo.main()
    # The launch keeps the checkpoint's env_steps, so its rollout seeds start
    # past every seed the loaded weights trained on, as a resume's do.
    assert seen[0]["base_seed"] == run_ppo._kaggriculture_rollout_base_seed(
        17, start_env_steps=1_000
    )
    assert calls == []


def test_main_forwards_wandb_mode_and_the_default_step_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build the real tiny CPU launch and stop at the training session.
    real = (
        run_ppo._create_run_dir,
        run_ppo.create_env,
        run_ppo._create_training_model_for_config,
    )
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch,
        [
            str(_CONFIGS / "kaggriculture.yaml"),
            str(tmp_path / "runs"),
            "--wandb-mode",
            "offline",
            "-o",
            f"rl.teacher_init={_kaggriculture_teacher_checkpoint(tmp_path)}",
        ],
        calls,
        log_mode=LogMode.WANDB,
    )
    monkeypatch.setattr(run_ppo, "_create_run_dir", real[0])
    monkeypatch.setattr(run_ppo, "create_env", real[1])
    monkeypatch.setattr(run_ppo, "_create_training_model_for_config", real[2])
    session: list[dict[str, object]] = []
    monkeypatch.setattr(
        run_ppo, "_run_training_session", lambda **kwargs: session.append(kwargs)
    )
    run_ppo.main()
    (kwargs,) = session
    assert kwargs["log_mode"] == LogMode.WANDB
    identity = kwargs["identity"]
    assert isinstance(identity, RunIdentity)
    assert identity.telemetry is TelemetryMode.WANDB_OFFLINE
    cfg = kwargs["cfg"]
    assert isinstance(cfg, FullConfig)
    assert kwargs["max_env_steps"] == run_ppo._kaggriculture_step_limit(
        cfg, DistributedContext.single_process_cpu(), max_env_steps=None
    )
    assert kwargs["start_env_steps"] == 0
    assert calls == []


# Plan Task 3.5, the bounded local functional check: the shipped CPU config
# (tiny model, 2 envs) runs 2 real updates through run_ppo.main() on the native
# Kaggriculture env, with an evaluation after each update and both outcomes of
# the promotion branch. Only launch plumbing is patched: the release-build and
# torch setup, the single-process CPU session, the probed compile stack, a
# recording metric logger, and the evaluation cadence below Isaiah's
# checkpoint_freq floor of 1000 env steps (unreachable in 2 updates at 2 envs
# within the local memory bound). Evaluation, promotion, teacher activation,
# checkpoint writes and the trainer are the real shared path.
_FUNCTIONAL_HORIZON = 2
_FUNCTIONAL_UPDATES = 2
_FUNCTIONAL_EPISODE_STEPS = 6


def _run_functional_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold: float,
    extra_overrides: tuple[str, ...] = (),
    episode_steps: int = _FUNCTIONAL_EPISODE_STEPS,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], _FakeLogger]:
    n_envs = 2
    update_steps = _FUNCTIONAL_HORIZON * n_envs
    # Task 4.4: a fresh Kaggriculture last-best launch names its teacher.
    teacher = _kaggriculture_teacher_checkpoint(tmp_path)
    argv = [
        str(_CONFIGS / "kaggriculture.yaml"),
        str(tmp_path / "runs"),
        "--max-env-steps",
        str(_FUNCTIONAL_UPDATES * update_steps),
        "-o",
        f"rl.horizon={_FUNCTIONAL_HORIZON}",
        f"env.config.episodeSteps={episode_steps}",
        f"rl.teacher_init={teacher}",
        *extra_overrides,
    ]
    monkeypatch.setattr(
        sys, "argv", ["run_ppo.py", *argv, "--log-mode", LogMode.DEBUG.value]
    )
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    monkeypatch.setattr(run_ppo, "installed_compile_stack", lambda: _PROBED_STACK)
    monkeypatch.setattr(run_ppo, "LAST_BEST_WIN_RATE_THRESHOLD", threshold)
    real_from_file = FullConfig.from_file

    def from_file_evaluating_each_update(
        path: Path, *, overrides: dict[str, Any] | None = None
    ) -> FullConfig:
        cfg = real_from_file(path, overrides=overrides)
        assert cfg.rl.checkpoint_freq == 1_000
        return cfg.model_copy(
            update={"rl": cfg.rl.model_copy(update={"checkpoint_freq": update_steps})}
        )

    monkeypatch.setattr(
        run_ppo.FullConfig, "from_file", staticmethod(from_file_evaluating_each_update)
    )
    logger = _FakeLogger(run_id=None)
    monkeypatch.setattr(run_ppo, "create_logger", lambda *_args, **_kwargs: logger)
    real_session = run_ppo._run_training_session
    session: dict[str, Any] = {}
    initial: dict[str, torch.Tensor] = {}

    def recording_session(**kwargs: Any) -> None:
        session.update(kwargs)
        model = run_ppo.unwrap_model(kwargs["trainer"].model)
        initial.update({k: v.clone() for k, v in model.state_dict().items()})
        real_session(**kwargs)

    monkeypatch.setattr(run_ppo, "_run_training_session", recording_session)
    run_ppo.main()
    return session, initial, logger


def _checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(checkpoint, dict)
    return checkpoint, checkpoint["model"]


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(value, right[key]) for key, value in left.items()
    )


@pytest.mark.parametrize("promote", [True, False], ids=["promoted", "held"])
def test_kaggriculture_two_update_functional_check_through_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, promote: bool
) -> None:
    torch.manual_seed(353)
    # Threshold 0.0 promotes on any win rate; above 1.0 never promotes.
    threshold = 0.0 if promote else 1.5
    session, initial, logger = _run_functional_check(
        tmp_path, monkeypatch, threshold=threshold
    )
    cfg = session["cfg"]
    assert isinstance(cfg, FullConfig)
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    assert cfg.env.n_envs == 2
    assert cfg.model.model_arch == "kaggriculture_transformer"
    assert cfg.model.embed_dim == 16
    assert cfg.rl.eval_replay_games == 0
    assert cfg.rl.teacher_mode == "last_best"
    assert session["log_mode"] == LogMode.DEBUG
    trainer = session["trainer"]
    assert isinstance(trainer, PPOTrainer)
    assert trainer.device.type == "cpu"
    run_dir = session["run_dir"]
    assert isinstance(run_dir, Path)

    # Two updates, each logging finite training losses, then its evaluation.
    update_steps = _FUNCTIONAL_HORIZON * cfg.env.n_envs
    steps = [update_steps * (i + 1) for i in range(_FUNCTIONAL_UPDATES)]
    assert [step for _metrics, step in logger.logged] == [
        step for step in steps for _ in range(2)
    ]
    training_logs = [metrics for metrics, _step in logger.logged[0::2]]
    eval_logs = [metrics for metrics, _step in logger.logged[1::2]]
    for metrics in training_logs:
        losses = {k: v for k, v in metrics.items() if k.startswith("loss/")}
        assert "loss/total_loss" in losses
        assert all(math.isfinite(value) for value in metrics.values())
    assert trainer.optimizer_steps == _FUNCTIONAL_UPDATES * cfg.env.n_envs
    for metrics in eval_logs:
        assert all(math.isfinite(value) for value in metrics.values())
        # Both seats of both native games are played to the engine's end.
        assert metrics["eval/games"] == float(cfg.env.n_envs)
        assert metrics["eval/episode_steps"] == float(_FUNCTIONAL_EPISODE_STEPS - 1)
        assert 0.0 <= metrics["eval/win_rate_against_last_best"] <= 1.0
        assert metrics["eval/promoted"] == float(promote)
        assert metrics["eval/promotion_threshold"] == threshold
    assert not (run_dir / "eval_replays").exists()

    # Checkpoints: one per evaluation, the final one and last_best; all load.
    final_path = run_dir / run_ppo.CHECKPOINT_FINAL
    last_best_path = run_dir / run_ppo.CHECKPOINT_LAST_BEST
    periodic = [
        run_dir / f"checkpoint_{run_ppo._format_checkpoint_step(step)}.pt"
        for step in steps
    ]
    assert sorted(path.name for path in run_dir.glob("*.pt")) == sorted(
        path.name for path in [final_path, last_best_path, *periodic]
    )
    current = run_ppo.unwrap_model(trainer.model).state_dict()
    final_checkpoint, final_state = _checkpoint_state(final_path)
    assert _states_equal(current, final_state)
    assert final_checkpoint["env_steps"] == steps[-1]
    loaded_model = run_ppo._create_eval_model_for_config(
        cfg, device=torch.device("cpu"), roundtrip_lora_base=False
    )
    metadata = run_ppo._load_model_from_checkpoint(
        loaded_model, path=final_path, device=torch.device("cpu")
    )
    assert metadata.env_steps == steps[-1]
    assert metadata.total_games_played == trainer.total_games_played
    assert _states_equal(loaded_model.state_dict(), current)
    assert not _states_equal(initial, current)
    _periodic_checkpoint, periodic_state = _checkpoint_state(periodic[-1])
    assert _states_equal(periodic_state, final_state)

    # The launch-time teacher (rl.teacher_init, Task 4.4) is the active last-best
    # teacher from update 1. Promoted, last_best is refreshed and rewritten at the
    # latest evaluation; held, it stays the teacher checkpoint's weights.
    _teacher_checkpoint, teacher_state = _checkpoint_state(
        tmp_path / "teacher" / "bc_best.pt"
    )
    assert not _states_equal(teacher_state, initial)
    last_best_checkpoint, last_best_state = _checkpoint_state(last_best_path)
    assert trainer.teacher_model is not None
    assert trainer.teacher_active
    assert all(m["teacher/cache_bytes"] > 0.0 for m in training_logs)
    if promote:
        assert last_best_checkpoint["env_steps"] == steps[-1]
        assert _states_equal(last_best_state, final_state)
        assert _states_equal(trainer.teacher_model.state_dict(), current)
    else:
        assert last_best_checkpoint["env_steps"] == 0
        assert _states_equal(last_best_state, teacher_state)
        assert _states_equal(trainer.teacher_model.state_dict(), teacher_state)


def test_kaggriculture_fixed_opponent_two_update_run_through_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """env.opponent_mix at fraction 1.0 through the canonical trainer."""
    torch.manual_seed(359)
    session, initial, logger = _run_functional_check(
        tmp_path,
        monkeypatch,
        threshold=0.0,
        extra_overrides=(
            "env.opponent_mix.bot=starter",
            "env.opponent_mix.fraction=1.0",
        ),
    )
    cfg = session["cfg"]
    assert isinstance(cfg, FullConfig)
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    assert cfg.env.opponent_mix is not None
    assert (cfg.env.opponent_mix.bot, cfg.env.opponent_mix.fraction) == ("starter", 1.0)
    trainer = session["trainer"]
    assert isinstance(trainer, PPOTrainer)
    assert trainer.rollout.learner is not None
    # The bot's name is a run label only.
    assert logger.summary["opponent_mix/bot"] == "starter"
    assert logger.summary["opponent_mix/fraction"] == 1.0
    training_logs = [metrics for metrics, _step in logger.logged[0::2]]
    eval_logs = [metrics for metrics, _step in logger.logged[1::2]]
    assert len(training_logs) == len(eval_logs) == _FUNCTIONAL_UPDATES
    for metrics in training_logs:
        assert all(math.isfinite(value) for value in metrics.values())
        assert "train/bank_games_vs_bot" in metrics
        assert metrics["train/bank_games"] == 0.0  # no self-play games at 1.0
        # One learned seat per env: two learner player-steps per step.
        assert metrics["train/policy_active_ratio"] == pytest.approx(0.5)
    # Four transitions complete no 5-transition game: the count is still
    # logged (test_opponent_mix.py covers completed training games).
    assert [m["train/bank_games_vs_bot"] for m in training_logs] == [0.0, 0.0]
    for metrics in eval_logs:
        assert all(math.isfinite(value) for value in metrics.values())
        # Promotion stays against last_best; the bot evaluation sits beside it.
        assert metrics["eval/games"] == float(cfg.env.n_envs)
        assert 0.0 <= metrics["eval/win_rate_against_last_best"] <= 1.0
        assert metrics["eval/bank_games_vs_bot"] == float(cfg.env.n_envs)
        assert metrics["eval/bank_games_vs_bot_seat_0"] == 1.0
        assert metrics["eval/bank_games_vs_bot_seat_1"] == 1.0
        for key in (
            "eval/win_rate_vs_bot",
            "eval/own_bank_mean_vs_bot",
            "eval/margin_mean_vs_bot",
            "eval/win_rate_vs_bot_seat_0",
            "eval/win_rate_vs_bot_seat_1",
        ):
            assert key in metrics
        assert metrics["eval/promoted"] == 1.0
    assert not _states_equal(initial, run_ppo.unwrap_model(trainer.model).state_dict())
    # Checkpoints carry no opponent state or label.
    run_dir = session["run_dir"]
    assert isinstance(run_dir, Path)
    for path in run_dir.glob("*.pt"):
        assert b"starter" not in path.read_bytes(), path.name


def test_cha22_anchor_two_update_run_through_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cha22 anchor setup: Cha22 hosted at fraction 1.0 under term M.

    A tiny CPU run of two updates through the canonical trainer. Three-step
    games (two transitions) complete one training game per env per update, so
    the per-update vs-bot telemetry is measured, not only present.
    """
    torch.manual_seed(367)
    session, initial, logger = _run_functional_check(
        tmp_path,
        monkeypatch,
        threshold=0.0,
        episode_steps=3,
        extra_overrides=(
            "env.opponent_mix.bot=cha22",
            "env.opponent_mix.fraction=1.0",
            # Term M exactly as configs/kaggriculture_4rank_vs_cha22.yaml.
            "env.reward_shaping.econ_shaping=0.0",
            "env.reward_shaping.econ_bank_weight=0.0",
            "env.reward_shaping.econ_bank_cap=0.0",
            "env.reward_shaping.econ_margin_weight=0.5",
            "env.reward_shaping.econ_margin_scale=50000.0",
            "env.reward_shaping.econ_margin_cap=0.5",
        ),
    )
    cfg = session["cfg"]
    assert isinstance(cfg, FullConfig)
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    assert cfg.env.opponent_mix == KaggricultureOpponentMixConfig(
        bot="cha22", fraction=1.0
    )
    assert cfg.env.reward_shaping.terminal_scale == 0.5
    trainer = session["trainer"]
    assert isinstance(trainer, PPOTrainer)
    assert trainer.rollout.learner is not None
    assert logger.summary["opponent_mix/bot"] == "cha22"
    assert logger.summary["opponent_mix/fraction"] == 1.0
    training_logs = [metrics for metrics, _step in logger.logged[0::2]]
    eval_logs = [metrics for metrics, _step in logger.logged[1::2]]
    assert len(training_logs) == len(eval_logs) == _FUNCTIONAL_UPDATES
    for metrics in training_logs:
        assert all(math.isfinite(value) for value in metrics.values())
        assert metrics["train/policy_active_ratio"] == pytest.approx(0.5)
        # Every game is against Cha22: one per env per update, none self-play.
        assert metrics["train/bank_games_vs_bot"] == float(cfg.env.n_envs)
        assert metrics["train/bank_games"] == 0.0
        assert 0.0 <= metrics["train/win_rate_vs_bot"] <= 1.0
        assert metrics["train/opponent_bank_mean_vs_bot"] > 0.0
        # Term M pays per step against the bot.
        assert metrics["train/reward_margin_abs_mean"] > 0.0
        assert metrics["train/margin_mean_vs_bot"] == pytest.approx(
            metrics["train/own_bank_mean_vs_bot"]
            - metrics["train/opponent_bank_mean_vs_bot"]
        )
    for metrics in eval_logs:
        assert all(math.isfinite(value) for value in metrics.values())
        assert metrics["eval/games"] == float(cfg.env.n_envs)
        assert metrics["eval/bank_games_vs_bot"] == float(cfg.env.n_envs)
        assert metrics["eval/bank_games_vs_bot_seat_0"] == 1.0
        assert metrics["eval/bank_games_vs_bot_seat_1"] == 1.0
        for seat in (0, 1):
            assert 0.0 <= metrics[f"eval/win_rate_vs_bot_seat_{seat}"] <= 1.0
        assert metrics["eval/margin_mean_vs_bot"] == pytest.approx(
            metrics["eval/own_bank_mean_vs_bot"]
            - metrics["eval/opponent_bank_mean_vs_bot"]
        )
    assert not _states_equal(initial, run_ppo.unwrap_model(trainer.model).state_dict())
    run_dir = session["run_dir"]
    assert isinstance(run_dir, Path)
    checkpoints = list(run_dir.glob("*.pt"))
    assert checkpoints
    for path in checkpoints:
        assert b"cha22" not in path.read_bytes(), path.name


def test_last_best_evaluation_env_never_hosts_the_training_opponent() -> None:
    cfg = _kaggriculture_eval_config()
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    mixed = cfg.model_copy(
        update={
            "env": cfg.env.model_copy(
                update={
                    "opponent_mix": KaggricultureOpponentMixConfig(
                        bot="r04", fraction=0.5
                    )
                }
            )
        }
    )
    device = torch.device("cpu")
    plain = run_ppo._create_eval_env(mixed, n_envs=2, device=device, env_steps=1000)
    assert isinstance(plain, KaggricultureVectorizedEnv)
    assert plain.opponent_envs == 0
    hosted = run_ppo._create_eval_env(
        mixed,
        n_envs=2,
        device=device,
        env_steps=1000,
        opponent_mix=KaggricultureOpponentMixConfig(bot="r04", fraction=1.0),
    )
    assert isinstance(hosted, KaggricultureVectorizedEnv)
    assert (hosted.opponent_bot, hosted.opponent_envs) == ("r04", 2)
    with pytest.raises(ValueError, match=r"requires env\.opponent_mix"):
        run_ppo._evaluate_against_bot(
            current_model=_LaunchPolicy(launch=True),
            cfg=cfg,
            device=device,
            env_steps=1000,
        )


class _SeatPinnedNativeEvalEnv(KaggricultureVectorizedEnv):
    """One-step native-typed fake: seat 1 ends with the larger raw bank."""

    def __init__(self) -> None:  # no native state; only the eval loop's calls
        self.stepped_with: list[object] = []

    def reset(self) -> Any:
        return SimpleNamespace(still_playing=torch.ones((1, 2), dtype=torch.bool))

    def step(self, actions: object) -> Any:
        self.stepped_with.append(actions)
        return (
            self.reset(),
            torch.zeros((1, 2)),
            torch.ones((1, 2), dtype=torch.bool),
            {},
        )

    def terminal_metrics(self, i: int) -> Any:
        assert i == 0
        return {
            "bank_0": 1000.0,
            "bank_1": 4000.0,
            "margin_0": -3000.0,
            "winner": 1,
            "episode_steps": 1,
            "counters_0": [0, 0],
        }


def _patch_native_eval(
    monkeypatch: pytest.MonkeyPatch, env: _SeatPinnedNativeEvalEnv, actions: object
) -> None:
    monkeypatch.setattr(run_ppo, "_create_eval_env", lambda *_a, **_k: env)

    def pin_candidate_to_seat_1(
        assignments: torch.Tensor, env_index: int, **_kwargs: object
    ) -> None:
        assignments[env_index] = torch.tensor(
            [run_ppo.MODEL_LAST_BEST, run_ppo.MODEL_CURRENT]
        )

    monkeypatch.setattr(run_ppo, "_assign_eval_models", pin_candidate_to_seat_1)
    monkeypatch.setattr(
        run_ppo,
        "_eval_actions_for_assignments_and_hidden",
        lambda *_a, **_k: (actions, None, None),
    )


def _evaluate_one_native_game() -> tuple[Any, dict[str, list[float]]]:
    stats, _by_count, metrics, _steps = run_ppo._evaluate_games(
        current_model=_LaunchPolicy(launch=True),
        last_best_model=_LaunchPolicy(launch=False),
        cfg=_kaggriculture_eval_config(),
        n_games=1,
        n_envs=1,
        device=torch.device("cpu"),
        env_steps=1000,
    )
    return stats, metrics


def test_kaggriculture_evaluation_credits_the_candidates_seat_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _SeatPinnedNativeEvalEnv()
    actions = KaggricultureActions(
        tokens=torch.zeros((1, 2, 252, 12), dtype=torch.int64),
        lengths=torch.ones((1, 2), dtype=torch.int64),
    )
    _patch_native_eval(monkeypatch, env, actions)
    stats, metrics = _evaluate_one_native_game()
    assert env.stepped_with == [actions]
    # Candidate in seat 1 with bank_1 > bank_0 wins; its bank is bank_1.
    assert stats.model_games == [1, 1]
    assert stats.wins == [1.0, 0.0]
    assert metrics["candidate_bank"] == [4000.0]
    assert metrics["last_best_bank"] == [1000.0]
    assert metrics["candidate_bank_margin"] == [3000.0]
    # The native winner reaches the scorer, which cross-checks it with the banks.
    assert metrics["winner"] == [1.0]
    assert metrics["margin_0"] == [-3000.0]


def test_kaggriculture_evaluation_rejects_orbit_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _SeatPinnedNativeEvalEnv()
    _patch_native_eval(monkeypatch, env, SimpleNamespace(launch=None))
    with pytest.raises(TypeError, match="requires KaggricultureActions"):
        _evaluate_one_native_game()
    assert env.stepped_with == []


def test_evaluate_games_rejects_kaggriculture_replay_before_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_ppo,
        "_create_eval_env",
        lambda *_a, **_k: pytest.fail("the eval env must not be built"),
    )
    with pytest.raises(ValueError, match=r"Task 7\.3"):
        run_ppo._evaluate_games(
            current_model=_LaunchPolicy(launch=True),
            last_best_model=_LaunchPolicy(launch=False),
            cfg=_kaggriculture_eval_config(),
            n_games=1,
            n_envs=1,
            device=torch.device("cpu"),
            env_steps=1000,
            replay_games=1,
        )


def test_validate_args_rejects_offline_wandb_with_debug_logging() -> None:
    with pytest.raises(ValueError, match="--wandb-mode offline requires --log-mode"):
        run_ppo._validate_args(
            Namespace(
                max_env_steps=None,
                max_runtime_hours=None,
                output_dir=Path("runs"),
                overrides=None,
                load_model_weights=None,
                load_model_weights_mode="model_only",
                log_mode=LogMode.DEBUG,
                wandb_mode=WandbMode.OFFLINE,
                experiment_id=None,
            )
        )


def test_evaluation_mapper_keeps_cuda_nonblocking_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def mapper(
        obs: object, device: torch.device, *, non_blocking: bool = False
    ) -> object:
        assert device.type == "cuda"
        calls.append(non_blocking)
        return obs

    def stop(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("after transfer")

    monkeypatch.setattr(run_ppo, "_obs_to_device", mapper)
    monkeypatch.setattr(run_ppo, "_model_output_for_eval", stop)
    with pytest.raises(RuntimeError, match="after transfer"):
        run_ppo._eval_actions_for_assignments_and_hidden(
            _two_seat_obs(1),
            torch.zeros((1, 2), dtype=torch.int64),
            current_model=_LaunchPolicy(launch=True),
            last_best_model=_LaunchPolicy(launch=False),
            hidden_current=None,
            hidden_last_best=None,
            config=_full_config().rl,
            device=torch.device("cuda"),
        )
    assert calls == [True]


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_main_kaggriculture_rollout_factory_uses_rank_seed_and_transfer_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    calls: list[str] = []
    _patch_kaggriculture_startup(
        monkeypatch,
        [
            str(_CONFIGS / "kaggriculture.yaml"),
            str(tmp_path / "runs"),
            *_teacher_source_argv(tmp_path),
            "-o",
            "env.seed=17",
            "env.pin_memory=false",
            "rl.eval_replay_games=0",
        ],
        calls,
    )
    ctx = DistributedContext(
        device=torch.device(device_type),
        rank=1,
        local_rank=1,
        world_size=2,
        initialized=False,
    )
    monkeypatch.setattr(run_ppo, "distributed_session", lambda: nullcontext(ctx))
    monkeypatch.setattr(run_ppo, "broadcast_object", lambda _obj, _ctx: tmp_path)
    seen: list[tuple[object, dict[str, object]]] = []

    def factory(config: object, **kwargs: object) -> None:
        seen.append((config, kwargs))
        raise AssertionError("factory reached")

    monkeypatch.setattr(run_ppo, "create_env", factory)
    with pytest.raises(AssertionError, match="factory reached"):
        run_ppo.main()
    config, kwargs = seen[0]
    assert isinstance(config, KaggricultureEnvConfig)
    assert kwargs == {
        "n_envs": config.n_envs,
        "base_seed": 17,
        "rank": 1,
        "world_size": 2,
        "pin_memory": False,
        "transfer_device": ctx.device,
    }
    assert calls == []


def test_main_rejects_unserviceable_kaggriculture_workload_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Trunk width 2 * 2**21 x 709 padded tokens exceeds 2**31 for one row.
    calls: list[str] = []
    argv = [
        str(_CONFIGS / "kaggriculture_2rank.yaml"),
        str(tmp_path / "runs"),
        *_teacher_source_argv(tmp_path),
        "-o",
        f"model.embed_dim={2**21}",
        f"model.n_heads={2**16}",
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)

    with pytest.raises(ValueError, match="one padded row of 709 tokens"):
        run_ppo.main()

    assert calls == []
    assert not (tmp_path / "runs").exists()
    assert _headroom_lines(capsys.readouterr().out) == []


@pytest.mark.parametrize(
    ("name", "rollout_rows", "teacher_rows", "teacher_calls", "cache_bytes"),
    [
        ("kaggriculture_2rank.yaml", 256, 16_384, 3, 1_674_575_872),
        ("kaggriculture_4rank.yaml", 128, 8_192, 2, 837_287_936),
        ("kaggriculture_8rank.yaml", 64, 4_096, 1, 418_643_968),
    ],
)
def test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    rollout_rows: int,
    teacher_rows: int,
    teacher_calls: int,
    cache_bytes: int,
) -> None:
    calls: list[str] = []
    # The shipped presets pass startup unmodified (no replay override).
    argv = [
        str(_CONFIGS / name),
        str(tmp_path / "runs"),
        *_teacher_source_argv(tmp_path),
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)

    with pytest.raises(AssertionError, match="_create_run_dir ran"):
        run_ppo.main()

    assert calls == ["_create_run_dir"]
    assert not (tmp_path / "runs").exists()
    headroom = _headroom_lines(capsys.readouterr().out)
    assert [line.split(":")[0] for line in headroom] == [
        "GEMM workload headroom rollout",
        "GEMM workload headroom minibatch",
        "GEMM workload headroom teacher_chunk",
        "GEMM workload headroom evaluation",
    ]
    assert headroom[0].startswith(
        f"GEMM workload headroom rollout: {rollout_rows} rows x 709 padded tokens;"
    )
    assert (
        f"teacher_chunk: {teacher_rows} rows x 709 padded tokens; trunk 5915 "
        f"rows/call at full padding (>= {5_915 / teacher_rows:.4g}x headroom, "
        f"<= {teacher_calls} call(s))"
    ) in headroom[2]
    assert headroom[2].endswith(
        f"; teacher targets {cache_bytes} B per chunk, {cache_bytes} B cached for "
        f"{teacher_rows} rollout rows"
    )


# --- Kaggriculture teacher checkpoint source (plan Task 4.4) ------------------

_NO_TEACHER_SOURCE = "needs a teacher checkpoint at launch"


def test_main_rejects_a_kaggriculture_launch_without_a_teacher_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    argv = [str(_CONFIGS / "kaggriculture_2rank.yaml"), str(tmp_path / "runs")]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)

    with pytest.raises(ValueError, match=_NO_TEACHER_SOURCE) as raised:
        run_ppo.main()

    message = str(raised.value)
    for remedy in (
        "--load-model-weights CHECKPOINT",
        "-o rl.teacher_init=CHECKPOINT",
        "-o rl.teacher_mode=null",
    ):
        assert remedy in message
    assert calls == []
    assert not (tmp_path / "runs").exists()
    assert _headroom_lines(capsys.readouterr().out) == []


@pytest.mark.parametrize("name", ["kaggriculture.yaml", "kaggriculture_8rank.yaml"])
def test_main_accepts_teacher_init_as_the_kaggriculture_teacher_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    teacher = tmp_path / "bc_best.pt"
    teacher.write_bytes(b"")
    calls: list[str] = []
    argv = [
        str(_CONFIGS / name),
        str(tmp_path / "runs"),
        "-o",
        f"rl.teacher_init={teacher}",
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)

    # The teacher check passes; the launch proceeds to the run directory.
    with pytest.raises(AssertionError, match="_create_run_dir ran"):
        run_ppo.main()

    assert calls == ["_create_run_dir"]


def test_main_rejects_a_missing_kaggriculture_teacher_init_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.pt"
    calls: list[str] = []
    argv = [
        str(_CONFIGS / "kaggriculture_2rank.yaml"),
        str(tmp_path / "runs"),
        "-o",
        f"rl.teacher_init={missing}",
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)

    with pytest.raises(ValueError, match="teacher_init checkpoint does not exist"):
        run_ppo.main()

    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_main_runs_a_kaggriculture_launch_without_a_teacher_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    argv = [
        str(_CONFIGS / "kaggriculture_2rank.yaml"),
        str(tmp_path / "runs"),
        "-o",
        "rl.teacher_mode=null",
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls)

    with pytest.raises(AssertionError, match="_create_run_dir ran"):
        run_ppo.main()

    assert calls == ["_create_run_dir"]
    headroom = _headroom_lines(capsys.readouterr().out)
    assert [line.split(":")[0] for line in headroom] == [
        "GEMM workload headroom rollout",
        "GEMM workload headroom minibatch",
        "GEMM workload headroom evaluation",
    ]


def test_teacher_source_check_leaves_orbit_and_resume_launches_to_isaiahs_rules(
    tmp_path: Path,
) -> None:
    orbit = _full_config().model_copy(
        update={
            "rl": _full_config().rl.model_copy(update={"teacher_mode": "last_best"})
        }
    )
    fresh = run_ppo.FreshLaunch(
        config_path=tmp_path / "config.yaml", output_dir=tmp_path, overrides={}
    )
    # Isaiah's scratch Orbit launch keeps its teacher off until a promotion.
    run_ppo._require_kaggriculture_teacher_source(orbit, fresh)
    kaggriculture = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    resume = run_ppo.ResumeLaunch(
        config_path=tmp_path / "config.yaml",
        run_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint_final.pt",
        last_best_checkpoint_path=tmp_path / "checkpoint_last_best.pt",
    )
    # A resume restores the run's own last-best teacher.
    run_ppo._require_kaggriculture_teacher_source(kaggriculture, resume)
    with pytest.raises(ValueError, match=_NO_TEACHER_SOURCE):
        run_ppo._require_kaggriculture_teacher_source(kaggriculture, fresh)


def test_resume_startup_checks_the_runtime_adapted_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A run saved at 2 ranks resumes on 1: resume adaptation doubles the
    # per-rank envs to 256, so the checked rollout is 512 rows, not the file's 256.
    saved = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    saved = saved.model_copy(
        update={
            "runtime": saved.runtime.model_copy(update={"n_runtime_gpus": 2}),
            "rl": saved.rl.model_copy(update={"eval_replay_games": 0}),
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    saved.to_file(run_dir / "config.yaml")
    # Startup reads the resumed env step for the rollout seed budget.
    for checkpoint in ("checkpoint_final.pt", "checkpoint_last_best.pt"):
        _write_metadata_checkpoint(run_dir / checkpoint, env_steps=0)
    _write_online_attempt_receipt(run_dir)
    calls: list[str] = []
    # Resume requires W&B logging; the check runs before any logger exists.
    _patch_kaggriculture_startup(
        monkeypatch,
        [str(run_dir / "checkpoint_final.pt")],
        calls,
        log_mode=LogMode.WANDB,
    )

    with pytest.raises(AssertionError, match="create_env ran"):
        run_ppo.main()

    assert calls == ["create_env"]
    headroom = _headroom_lines(capsys.readouterr().out)
    assert headroom[0].startswith(
        "GEMM workload headroom rollout: 512 rows x 709 padded tokens;"
    )


def test_startup_workload_check_omits_the_teacher_chunk_without_a_teacher(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    reports = run_ppo._check_model_workload(
        cfg.model,
        n_envs=cfg.env.n_envs,
        rl=cfg.rl.model_copy(update={"teacher_mode": None}),
        distributed=DistributedContext.single_process_cpu(),
    )

    assert [r.name for r in reports] == ["rollout", "minibatch", "evaluation"]
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_startup_workload_check_skips_isaiahs_unchunked_orbit_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _full_config()
    reports = run_ppo._check_model_workload(
        cfg.model,
        n_envs=cfg.env.n_envs,
        rl=cfg.rl,
        distributed=DistributedContext.single_process_cpu(),
    )

    assert reports == ()
    assert capsys.readouterr().out == ""


# --- W&B telemetry gate and attempt receipts ----------------------------------


def _without_wandb_credentials(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    for name in ("WANDB_API_KEY", "NETRC", "WANDB_BASE_URL", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))


def test_main_fails_fast_without_wandb_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    argv = [str(_CONFIGS / "kaggriculture_2rank.yaml"), str(tmp_path / "runs")]
    _patch_kaggriculture_startup(monkeypatch, argv, calls, log_mode=LogMode.WANDB)
    _without_wandb_credentials(monkeypatch, tmp_path)

    def reached(name: str) -> object:
        def record(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"{name} ran before the telemetry gate")

        return record

    monkeypatch.setattr(run_ppo, "_log_cli_overrides", reached("_log_cli_overrides"))
    monkeypatch.setattr(
        run_ppo.FullConfig, "from_file", reached("FullConfig.from_file")
    )
    monkeypatch.setattr(
        run_ppo, "resolve_source_commit", reached("resolve_source_commit")
    )

    with pytest.raises(
        MissingWandbCredentialsError, match=re.escape("api.wandb.ai")
    ) as info:
        run_ppo.main()

    message = str(info.value)
    assert "WANDB_API_KEY" in message
    assert "--wandb-mode offline" in message
    assert "install-the-wandb-credential-before-any-pod-launch" in message
    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_main_offline_mode_announces_the_outage_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    argv = [
        str(_CONFIGS / "kaggriculture_2rank.yaml"),
        str(tmp_path / "runs"),
        "--wandb-mode",
        "offline",
        *_teacher_source_argv(tmp_path),
    ]
    _patch_kaggriculture_startup(monkeypatch, argv, calls, log_mode=LogMode.WANDB)
    _without_wandb_credentials(monkeypatch, tmp_path)

    # Past the telemetry gate, startup reaches the run directory's sentinel.
    with pytest.raises(AssertionError, match="_create_run_dir ran"):
        run_ppo.main()

    assert calls == ["_create_run_dir"]
    err = capsys.readouterr().err
    assert "W&B TELEMETRY OUTAGE: telemetry_mode=wandb-offline" in err


class _FakeWandbLogger(_FakeLogger):
    def wandb_run_facts(self) -> WandbRunFacts | None:
        return WandbRunFacts(project="kg-v3", entity="team", url=None)


@pytest.mark.parametrize(
    ("telemetry", "logger", "outage"),
    [
        (TelemetryMode.DISABLED, _FakeLogger(run_id=None), True),
        (TelemetryMode.WANDB_OFFLINE, _FakeWandbLogger(run_id="off1"), True),
        (TelemetryMode.WANDB_ONLINE, _FakeWandbLogger(run_id="on1"), False),
    ],
)
def test_run_training_session_records_the_attempt_and_its_telemetry_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    telemetry: TelemetryMode,
    logger: _FakeLogger,
    outage: bool,
) -> None:
    monkeypatch.setattr(run_ppo, "create_logger", lambda *_a, **_k: logger)

    run_ppo._run_training_session(
        trainer=_FakeTrainer(),
        run_dir=tmp_path,
        cfg=_full_config(),
        log_mode=LogMode.WANDB,
        identity=_identity(telemetry),
        env_steps_per_iteration=8,
        max_env_steps=8,
        max_runtime_seconds=None,
        distributed=DistributedContext.single_process_cpu(),
        start_env_steps=16,
    )

    records = [
        json.loads(line) for line in (tmp_path / ATTEMPTS_FILE).read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["telemetry_mode"] == str(telemetry)
    assert record["experiment_id"] == "exp"
    assert record["attempt"] == 0
    assert record["source_commit"] == "abc123"
    assert record["start_env_steps"] == 16
    assert record["wandb_run_id"] == logger.run_id
    assert record["wandb_project"] == (
        None if telemetry is TelemetryMode.DISABLED else "kg-v3"
    )
    err = capsys.readouterr().err
    assert ("W&B TELEMETRY OUTAGE recorded" in err) is outage


def test_run_training_session_requires_the_main_rank_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_ppo, "create_logger", lambda *_a, **_k: _FakeLogger())

    with pytest.raises(RuntimeError, match="needs its run identity"):
        run_ppo._run_training_session(
            trainer=_FakeTrainer(),
            run_dir=tmp_path,
            cfg=_full_config(),
            log_mode=LogMode.DEBUG,
            identity=None,
            env_steps_per_iteration=8,
            max_env_steps=8,
            max_runtime_seconds=None,
            distributed=DistributedContext.single_process_cpu(),
        )


# --- main-level receipt wiring (claude-verify-wandb-r1 F2) ---------------------


class _StartupTrainer:
    def __init__(self, **kwargs: object) -> None:
        self.model = kwargs["model"]

    def load_checkpoint(self, path: Path) -> run_ppo.PPOCheckpointMetadata:
        assert path.name == run_ppo.CHECKPOINT_FINAL
        return run_ppo.PPOCheckpointMetadata(env_steps=64, wandb_run_id="off1")

    def set_teacher_model(self, *_args: object, **_kwargs: object) -> None:
        return None


def _patch_orbit_startup(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    envs_built: list[int],
    session: dict[str, object],
) -> None:
    """Stub everything past plan_attempt so main's receipt wiring is observable."""

    class FakeEnv:
        def __init__(self, *, n_envs: int, **_kwargs: object) -> None:
            envs_built.append(n_envs)
            self.n_envs = n_envs

    monkeypatch.setattr(sys, "argv", ["run_ppo.py", *argv])
    monkeypatch.setattr(run_ppo, "assert_release_build", lambda: None)
    monkeypatch.setattr(run_ppo, "configure_torch", lambda: None)
    monkeypatch.setattr(
        run_ppo,
        "distributed_session",
        lambda: nullcontext(DistributedContext.single_process_cpu()),
    )
    monkeypatch.setattr(run_ppo, "VectorizedEnv", FakeEnv)
    monkeypatch.setattr(
        run_ppo, "_create_model", lambda *_a, **_k: torch.nn.Linear(1, 1)
    )
    monkeypatch.setattr(run_ppo, "configure_model_compile", lambda *_args: 0)
    monkeypatch.setattr(
        run_ppo,
        "create_optimizer",
        lambda model, _cfg: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(run_ppo, "create_lr_scheduler", lambda *_args: None)
    monkeypatch.setattr(run_ppo, "PPOTrainer", _StartupTrainer)
    monkeypatch.setattr(
        run_ppo, "_create_eval_model_for_config", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        run_ppo,
        "_load_model_from_checkpoint",
        lambda *_a, **_k: run_ppo.PPOCheckpointMetadata(
            env_steps=64, wandb_run_id="off1"
        ),
    )
    monkeypatch.setattr(run_ppo, "_compile_eval_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        run_ppo, "_run_training_session", lambda **kwargs: session.update(kwargs)
    )
    for name in ("WANDB_API_KEY", "NETRC", "WANDB_BASE_URL", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)


def _without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout without git metadata, where --source-commit is the identity."""
    monkeypatch.setattr(train_logging, "git_source_commit", lambda _cwd: None)


def test_main_fresh_launch_plans_attempt_zero_with_the_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _full_config().to_file(config_path)
    envs_built: list[int] = []
    session: dict[str, object] = {}
    _patch_orbit_startup(
        monkeypatch,
        [
            str(config_path),
            str(tmp_path / "runs"),
            "--log-mode",
            "debug",
            "--experiment-id",
            "exp-fresh",
            "--source-commit",
            "v3-src-1",
        ],
        envs_built=envs_built,
        session=session,
    )
    _without_git(monkeypatch)

    run_ppo.main()

    identity = session["identity"]
    assert isinstance(identity, RunIdentity)
    assert identity.experiment_id == "exp-fresh"
    assert identity.job_type == "ppo"
    assert identity.attempt == 0
    assert identity.source_commit == "v3-src-1"
    assert identity.attempt_source_commits == ("v3-src-1",)
    assert identity.telemetry is TelemetryMode.DISABLED
    cfg = session["cfg"]
    assert isinstance(cfg, FullConfig)
    assert identity.config_sha256 == train_logging.config_sha256(cfg)
    run_dir = session["run_dir"]
    assert isinstance(run_dir, Path)
    assert identity.config_sha256 == train_logging.config_sha256(
        FullConfig.from_file(run_dir / "config.yaml")
    )
    assert envs_built == [2]


def test_main_rejects_a_source_commit_that_disagrees_with_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _full_config().to_file(config_path)
    envs_built: list[int] = []
    _patch_orbit_startup(
        monkeypatch,
        [
            str(config_path),
            str(tmp_path / "runs"),
            "--log-mode",
            "debug",
            "--source-commit",
            "not-the-checkout",
        ],
        envs_built=envs_built,
        session={},
    )
    monkeypatch.setattr(train_logging, "git_source_commit", lambda _cwd: "abc123")

    with pytest.raises(ValueError, match="disagrees with git"):
        run_ppo.main()

    assert envs_built == []
    assert not (tmp_path / "runs").exists()


def _resumable_run_dir(tmp_path: Path, *, with_receipt: bool) -> Path:
    run_dir = tmp_path / "runs" / "20260930-000000"
    run_dir.mkdir(parents=True)
    cfg = run_ppo._with_runtime_gpus(_full_config(), 1)
    cfg.to_file(run_dir / "config.yaml")
    (run_dir / run_ppo.CHECKPOINT_FINAL).write_bytes(b"stub")
    (run_dir / run_ppo.CHECKPOINT_LAST_BEST).write_bytes(b"stub")
    if with_receipt:
        identity = train_logging.plan_attempt(
            run_dir,
            job_type="ppo",
            resume=False,
            experiment_id="exp-resume",
            source_commit="v3-src-0",
            config_sha256=train_logging.config_sha256(cfg),
            telemetry=TelemetryMode.WANDB_OFFLINE,
        )
        train_logging.record_attempt(
            run_dir, identity, _FakeWandbLogger(run_id="off1"), start_env_steps=0
        )
    return run_dir


def test_main_resume_with_a_receipt_plans_attempt_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _resumable_run_dir(tmp_path, with_receipt=True)
    envs_built: list[int] = []
    session: dict[str, object] = {}
    _patch_orbit_startup(
        monkeypatch,
        [str(run_dir), "--source-commit", "v3-src-1"],
        envs_built=envs_built,
        session=session,
    )
    _without_git(monkeypatch)
    # Attempt 0 ran offline; after `wandb sync` the resume continues it online.
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")

    run_ppo.main()

    identity = session["identity"]
    assert isinstance(identity, RunIdentity)
    assert identity.attempt == 1
    assert identity.experiment_id == "exp-resume"
    assert identity.attempt_source_commits == ("v3-src-0", "v3-src-1")
    assert identity.telemetry is TelemetryMode.WANDB_ONLINE
    assert session["resume_run_id"] == "off1"
    assert session["start_env_steps"] == 64
    assert envs_built == [2]


def test_main_resume_without_receipts_fails_before_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _resumable_run_dir(tmp_path, with_receipt=False)
    envs_built: list[int] = []
    _patch_orbit_startup(
        monkeypatch,
        [str(run_dir), "--source-commit", "v3-src-1"],
        envs_built=envs_built,
        session={},
    )
    _without_git(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")

    with pytest.raises(FileNotFoundError, match="resume needs the run's attempt"):
        run_ppo.main()

    assert envs_built == []


# Learner-perspective bank telemetry --------------------------------------------------

_TRAIN_BANK_KEYS = {
    "train/bank_games",
    "train/own_bank_mean",
    "train/own_bank_p10",
    "train/own_bank_p50",
    "train/own_bank_p90",
    "train/margin_abs_mean",
    "train/margin_abs_p50",
    "train/draw_rate",
}
_EVAL_BANK_KEYS = {
    "eval/bank_games",
    *(
        f"eval/{name}_{stat}"
        for name in ("own_bank", "opponent_bank", "margin")
        for stat in ("mean", "p10", "p50", "p90")
    ),
}


def test_kaggriculture_bank_telemetry_reaches_the_logger(tmp_path: Path) -> None:
    """Real native trainer and evaluation; the fake logger stands in for W&B."""
    cfg = FullConfig.from_file(
        _CONFIGS / "kaggriculture.yaml",
        overrides={
            "env.config.episodeSteps": 3,
            "model.n_heads": 1,
            "model.mlp_ratio": 1,
            "model.n_scratch_tokens": 0,
            "rl.horizon": 2,
        },
    )
    assert isinstance(cfg.env, KaggricultureEnvConfig)
    assert cfg.env.n_envs == 2
    assert cfg.rl.checkpoint_freq == 1_000
    device = torch.device("cpu")
    torch.manual_seed(41)
    model, _ = run_ppo._create_training_model_for_config(
        cfg, device=device, reset_parameters=True
    )
    trainer = PPOTrainer(
        config=cfg.rl,
        env=create_env(
            cfg.env,
            n_envs=cfg.env.n_envs,
            base_seed=cfg.env.seed,
            rank=0,
            world_size=1,
            pin_memory=False,
            transfer_device=device,
        ),
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=device,
    )
    logger = _FakeLogger()

    run_ppo._run_training_loop(
        trainer=trainer,
        logger=logger,
        run_dir=tmp_path,
        cfg=cfg,
        # One real update (2 envs x horizon 2) counted as a checkpoint interval.
        env_steps_per_iteration=1_000,
        max_env_steps=1_000,
        max_runtime_seconds=None,
        dist_ctx=DistributedContext.single_process_cpu(),
    )

    (train, train_step), (evaluation, eval_step) = logger.logged
    assert train_step == eval_step == 1_000
    # Two envs each finish one 2-transition game in the 2-step horizon.
    assert train["train/bank_games"] == 2.0
    assert set(train) >= _TRAIN_BANK_KEYS
    assert set(evaluation) >= _EVAL_BANK_KEYS
    assert not {key for key in evaluation if key.startswith("train/")}
    assert all(
        torch.isfinite(torch.tensor(value))
        for key, value in {**train, **evaluation}.items()
        if "bank" in key or "margin" in key or "draw" in key
    )
    # Raw-bank units: the pooled own bank is the seat mean of the native records.
    assert train["train/own_bank_mean"] == pytest.approx(
        (train["train/terminal_bank_0"] + train["train/terminal_bank_1"]) / 2
    )
    # The evaluation keys aggregate the candidate's existing seat metrics, which
    # stay under their old names.
    assert evaluation["eval/bank_games"] == evaluation["eval/games"] == 2.0
    for new, old in (
        ("eval/own_bank_mean", "eval/candidate_bank"),
        ("eval/opponent_bank_mean", "eval/last_best_bank"),
        ("eval/margin_mean", "eval/candidate_bank_margin"),
    ):
        assert evaluation[new] == pytest.approx(evaluation[old])


def test_orbit_evaluation_logs_no_bank_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = run_ppo._EvalStats.empty()
    stats.add_game_result(run_ppo.MODEL_CURRENT)
    monkeypatch.setattr(
        run_ppo,
        "_evaluate_games",
        lambda **_kwargs: (stats, {}, {"game_length_mean": [12.0]}, 2),
    )
    metrics = run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=_config_with_envs(2),
        device=torch.device("cpu"),
        env_steps=1_000,
    )
    assert metrics["eval/games"] == 1.0
    assert not {key for key in metrics if "bank" in key or "margin" in key}


def test_kaggriculture_eval_bank_telemetry_follows_the_candidate_across_seats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed candidate seats, pinned rather than left to the evaluation RNG.

    Each game runs the real per-game path (terminal scalars plus
    ``_evaluation_scores_and_metrics``) and then the real
    ``_evaluate_against_last_best`` reduction, so seat-ordered banks cannot
    stand in for the candidate's banks.
    """
    cfg = _kaggriculture_eval_config()
    current, last_best = run_ppo.MODEL_CURRENT, run_ppo.MODEL_LAST_BEST
    # (bank_0, bank_1, seat assignment): candidate in seat 0 winning, candidate
    # in seat 1 losing, candidate in seat 1 winning.
    games = (
        (3000.0, 1000.0, (current, last_best)),
        (3000.0, 1000.0, (last_best, current)),
        (500.0, 4500.0, (last_best, current)),
    )
    stats = run_ppo._EvalStats.empty()
    env_metrics: dict[str, list[float]] = {}
    for bank_0, bank_1, seats in games:
        terminal = {
            "bank_0": bank_0,
            "bank_1": bank_1,
            "margin_0": bank_0 - bank_1,
            "episode_steps": 3.0,
            "winner": 0.0 if bank_0 > bank_1 else 1.0,
        }
        run_ppo._extend_single_env_metrics(env_metrics, terminal)
        _, outcome = run_ppo._evaluation_scores_and_metrics(
            cfg, terminal, torch.zeros(2), torch.tensor(seats)
        )
        run_ppo._extend_single_env_metrics(env_metrics, outcome)
        stats.add_game_result(None)
    monkeypatch.setattr(
        run_ppo,
        "_evaluate_games",
        lambda **_kwargs: (stats, {}, env_metrics, 9),
    )

    metrics = run_ppo._evaluate_against_last_best(
        current_model=torch.nn.Linear(1, 1),
        last_best_model=torch.nn.Linear(1, 1),
        cfg=cfg,
        device=torch.device("cpu"),
        env_steps=1_000,
    )

    # Candidate banks [3000, 1000, 4500]; last-best [1000, 3000, 500];
    # signed margins [2000, -2000, 4000] (linear quantiles).
    assert metrics["eval/bank_games"] == 3.0
    assert metrics["eval/own_bank_mean"] == pytest.approx(8500.0 / 3)
    assert metrics["eval/opponent_bank_mean"] == pytest.approx(1500.0)
    assert metrics["eval/margin_mean"] == pytest.approx(4000.0 / 3)
    assert metrics["eval/margin_p10"] == pytest.approx(-1200.0)
    assert metrics["eval/margin_p50"] == pytest.approx(2000.0)
    assert metrics["eval/margin_p90"] == pytest.approx(3600.0)
    # Seat-ordered keys keep their meaning and differ from the candidate view.
    assert metrics["eval/bank_0"] == pytest.approx(6500.0 / 3)
    assert metrics["eval/margin_0"] == pytest.approx(0.0)
