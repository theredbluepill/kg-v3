from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, TypeAlias, assert_never

from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.train import FullConfig

WandbMode: TypeAlias = Literal["online", "offline"]


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
        resume_run_id: str | None = None,
        wandb_mode: WandbMode = "online",
    ) -> None:
        if wandb_mode == "offline" and resume_run_id is not None:
            raise ValueError(
                "W&B offline mode cannot resume an existing run; use online "
                "mode to preserve its telemetry history"
            )

        import wandb

        self._wandb = wandb
        init_kwargs: dict[str, Any] = {}
        project = "orbit-wars"
        name = run_dir.name
        if isinstance(cfg.env, KaggricultureEnvConfig):
            project = "kg-v3"
            name = f"ppo-{run_dir.name}"
            init_kwargs.update(
                job_type="ppo",
                group="ppo",
                tags=["kaggriculture-v3", "ppo"],
                mode=wandb_mode,
            )
        elif wandb_mode == "offline":
            init_kwargs["mode"] = wandb_mode
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"
        self._run = wandb.init(
            project=project,
            dir=run_dir,
            name=name,
            config=cfg.model_dump(mode="json"),
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
    resume_run_id: str | None = None,
    wandb_mode: WandbMode = "online",
) -> MetricLogger:
    match log_mode:
        case LogMode.DEBUG:
            return DebugLogger()
        case LogMode.WANDB:
            return WandbLogger(
                run_dir, cfg, resume_run_id=resume_run_id, wandb_mode=wandb_mode
            )
        case _:
            assert_never(log_mode)
