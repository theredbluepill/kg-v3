from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, assert_never

from owl.model.kaggriculture import KaggricultureTransformerConfig
from owl.train import FullConfig

WandbMode = Literal["online", "offline"]
WANDB_MODES: tuple[WandbMode, ...] = ("online", "offline")
# Kaggriculture runs publish under the v3 project; Isaiah's retained Orbit runs
# keep his project.
KAGGRICULTURE_WANDB_PROJECT = "kg-v3"
ORBIT_WANDB_PROJECT = "orbit-wars"


def wandb_init_identity(cfg: FullConfig) -> dict[str, Any]:
    """The W&B project and labels for a PPO run of this config's game."""
    if isinstance(cfg.model, KaggricultureTransformerConfig):
        return {
            "project": KAGGRICULTURE_WANDB_PROJECT,
            "job_type": "ppo",
            "group": "ppo",
            "tags": ["kaggriculture-v3", "ppo"],
        }
    return {"project": ORBIT_WANDB_PROJECT}


class LogMode(StrEnum):
    DEBUG = auto()
    WANDB = auto()


class MetricLogger:
    @property
    def run_id(self) -> str | None:
        raise NotImplementedError

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        raise NotImplementedError

    def set_summary(self, key: str, value: int | float | str) -> None:
        raise NotImplementedError

    def close(self, *, exit_code: int = 0) -> None:
        raise NotImplementedError


class DebugLogger(MetricLogger):
    @property
    def run_id(self) -> str | None:
        return None

    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        print(metrics)

    def set_summary(
        self,
        key: str,  # noqa: ARG002
        value: int | float | str,  # noqa: ARG002
    ) -> None:
        return None

    def close(self, *, exit_code: int = 0) -> None:  # noqa: ARG002
        return None


class WandbLogger(MetricLogger):
    def __init__(
        self,
        run_dir: Path,
        cfg: FullConfig,
        *,
        mode: WandbMode = "online",
        resume_run_id: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        init_kwargs: dict[str, Any] = {}
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        self._run = wandb.init(
            **wandb_init_identity(cfg),
            dir=run_dir,
            name=run_dir.name,
            config=cfg.model_dump(mode="json"),
            mode=mode,
            **init_kwargs,
        )

    @property
    def run_id(self) -> str | None:
        return self._run.id

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def set_summary(self, key: str, value: int | float | str) -> None:
        run = self._wandb.run
        if run is None:
            raise RuntimeError("wandb run is not initialized")
        run.summary[key] = value

    def close(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def create_logger(
    log_mode: LogMode,
    run_dir: Path,
    cfg: FullConfig,
    *,
    wandb_mode: WandbMode = "online",
    resume_run_id: str | None = None,
) -> MetricLogger:
    match log_mode:
        case LogMode.DEBUG:
            return DebugLogger()
        case LogMode.WANDB:
            return WandbLogger(
                run_dir, cfg, mode=wandb_mode, resume_run_id=resume_run_id
            )
        case _:
            assert_never(log_mode)
