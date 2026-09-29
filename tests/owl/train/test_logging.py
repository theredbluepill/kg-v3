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
    WandbMode,
    create_logger,
)

_REPO_ROOT = Path(__file__).parents[3]


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    inits: list[dict[str, Any]] = []
    logs: list[tuple[dict[str, float], int]] = []
    finishes: list[int] = []
    run = SimpleNamespace(
        id="test-run-id",
        summary={},
        finish=lambda *, exit_code=0: finishes.append(exit_code),
    )

    def init(**kwargs: Any) -> SimpleNamespace:
        inits.append(kwargs)
        return run

    module = SimpleNamespace(
        init=init,
        log=lambda metrics, *, step: logs.append((metrics, step)),
        run=run,
        inits=inits,
        logs=logs,
        finishes=finishes,
    )
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


@pytest.mark.parametrize("resume_run_id", [None, "existing-orbit-run"])
def test_orbit_wandb_init_kwargs_are_unchanged(
    tmp_path: Path, fake_wandb: SimpleNamespace, resume_run_id: str | None
) -> None:
    cfg = FullConfig.from_file(_REPO_ROOT / "configs/scaling_6m.yaml")

    WandbLogger(tmp_path, cfg, resume_run_id=resume_run_id)

    expected = {
        "project": "orbit-wars",
        "dir": tmp_path,
        "name": tmp_path.name,
        "config": cfg.model_dump(mode="json"),
    }
    if resume_run_id is not None:
        expected.update(id=resume_run_id, resume="must")
    assert fake_wandb.inits == [expected]


@pytest.mark.parametrize("wandb_mode", ["online", "offline"])
def test_kaggriculture_wandb_init_uses_v3_identity_and_mode(
    tmp_path: Path, fake_wandb: SimpleNamespace, wandb_mode: WandbMode
) -> None:
    cfg = FullConfig.from_file(_REPO_ROOT / "configs/kaggriculture.yaml")

    logger = create_logger(LogMode.WANDB, tmp_path, cfg, wandb_mode=wandb_mode)

    assert fake_wandb.inits == [
        {
            "project": "kg-v3",
            "dir": tmp_path,
            "name": f"ppo-{tmp_path.name}",
            "config": cfg.model_dump(mode="json"),
            "job_type": "ppo",
            "group": "ppo",
            "tags": ["kaggriculture-v3", "ppo"],
            "mode": wandb_mode,
        }
    ]
    assert logger.run_id == "test-run-id"
    metrics = {"train/loss": 0.25, "eval/win_rate": 0.5, "eval/bank_margin": 0.0}
    logger.log(metrics, step=12)
    logger.set_summary("last_checkpoint_step", 12)
    logger.close(exit_code=1)
    assert fake_wandb.logs == [(metrics, 12)]
    assert fake_wandb.run.summary == {"last_checkpoint_step": 12}
    assert fake_wandb.finishes == [1]


def test_orbit_offline_wandb_keeps_run_artifacts_in_run_dir(
    tmp_path: Path, fake_wandb: SimpleNamespace
) -> None:
    cfg = FullConfig.from_file(_REPO_ROOT / "configs/scaling_6m.yaml")

    WandbLogger(tmp_path, cfg, wandb_mode="offline")

    assert fake_wandb.inits == [
        {
            "project": "orbit-wars",
            "dir": tmp_path,
            "name": tmp_path.name,
            "config": cfg.model_dump(mode="json"),
            "mode": "offline",
        }
    ]


def test_kaggriculture_online_wandb_resumes_same_run(
    tmp_path: Path, fake_wandb: SimpleNamespace
) -> None:
    cfg = FullConfig.from_file(_REPO_ROOT / "configs/kaggriculture.yaml")

    WandbLogger(tmp_path, cfg, resume_run_id="existing-v3-run")

    assert fake_wandb.inits[0]["id"] == "existing-v3-run"
    assert fake_wandb.inits[0]["resume"] == "must"
    assert fake_wandb.inits[0]["project"] == "kg-v3"


@pytest.mark.parametrize("config", ["scaling_6m", "kaggriculture"])
def test_offline_wandb_rejects_resume_before_initialization(
    tmp_path: Path, fake_wandb: SimpleNamespace, config: str
) -> None:
    cfg = FullConfig.from_file(_REPO_ROOT / f"configs/{config}.yaml")

    with pytest.raises(ValueError, match=r"offline.*resume|resume.*offline"):
        WandbLogger(tmp_path, cfg, resume_run_id="existing-run", wandb_mode="offline")

    assert fake_wandb.inits == []


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
