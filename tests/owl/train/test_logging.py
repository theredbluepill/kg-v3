import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from owl.train import FullConfig
from owl.train.logging import (
    DebugLogger,
    LogMode,
    WandbLogger,
    create_logger,
    wandb_init_identity,
)

_CONFIGS = Path(__file__).parents[3] / "configs"


def test_wandb_logger_reports_exit_code() -> None:
    finished: list[int] = []

    class _Run:
        def finish(self, *, exit_code: int = 0) -> None:
            finished.append(exit_code)

    logger = WandbLogger.__new__(WandbLogger)
    logger._run = _Run()
    logger.close(exit_code=1)
    logger.close()

    assert finished == [1, 0]


def test_debug_logger_accepts_exit_code() -> None:
    logger = DebugLogger()

    logger.close(exit_code=1)
    logger.close()


def test_kaggriculture_runs_publish_under_the_v3_wandb_project() -> None:
    kaggriculture = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")
    orbit = FullConfig.from_file(_CONFIGS / "scaling_6m.yaml")

    assert wandb_init_identity(kaggriculture) == {
        "project": "kg-v3",
        "job_type": "ppo",
        "group": "ppo",
        "tags": ["kaggriculture-v3", "ppo"],
    }
    # Isaiah's retained Orbit runs keep his project.
    assert wandb_init_identity(orbit) == {"project": "orbit-wars"}


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_wandb_logger_initializes_the_v3_run_in_the_requested_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    inits: list[dict[str, Any]] = []

    def init(**kwargs: Any) -> SimpleNamespace:
        inits.append(kwargs)
        return SimpleNamespace(id="run-id")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    cfg = FullConfig.from_file(_CONFIGS / "kaggriculture_2rank.yaml")

    logger = create_logger(
        LogMode.WANDB, tmp_path, cfg, wandb_mode=mode, resume_run_id="abc"
    )

    assert isinstance(logger, WandbLogger)
    assert logger.run_id == "run-id"
    (kwargs,) = inits
    assert kwargs["project"] == "kg-v3"
    assert kwargs["tags"] == ["kaggriculture-v3", "ppo"]
    assert kwargs["mode"] == mode
    assert (kwargs["id"], kwargs["resume"]) == ("abc", "must")
    assert kwargs["dir"] == tmp_path
    assert kwargs["config"] == cfg.model_dump(mode="json")
