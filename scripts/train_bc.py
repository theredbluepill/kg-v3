"""Behavior cloning warm start for Kaggriculture PPO (rebuild plan Task 5.2).

Fresh run:  train_bc.py CONFIG --data DATASET --output-dir DIR
Resume:     train_bc.py RUN_DIR --data DATASET

Launch multi-GPU runs with torchrun like ``scripts/run_ppo.py``. The run
directory holds the PPO ``config.yaml`` the checkpoint belongs to, the resolved
``bc_config.yaml``, ``bc_attempts.jsonl`` (one record per launch or resume),
``checkpoint_bc_best.pt`` (the run_ppo checkpoint schema; start PPO with
``--load-model-weights``) and its ``checkpoint_bc_best.json`` record,
``bc_state.pt`` for restarts, ``bc_history.jsonl`` (the held-out NLL curve) and
``bc_result.json``.

Every resume is a new attempt. It reads its own source identity (``git`` or
``--source-commit``) and records it with the parent ``bc_state.pt`` SHA-256 and
every earlier attempt's source, so records written after a resume never carry
the launch's source alone. The resumed settings must match the saved trajectory
(``check_resume_compatible``); only ``max_steps`` in ``bc_config.yaml`` may be
raised.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from owl.kaggriculture.bc_data import load_bc_dataset
from owl.model.compile_gemm import (
    check_compile_stack,
    gemm_backend_claim,
    installed_compile_stack,
)
from owl.model.kaggriculture_workload import headroom_log_lines
from owl.train import configure_torch
from owl.train.bc import (
    BC_CONFIG_NAME,
    BC_STATE,
    PPO_CONFIG_NAME,
    BCConfig,
    BCResumeState,
    Provenance,
    build_bc_model,
    check_bc_workload,
    check_resume_compatible,
    create_bc_logger,
    file_sha256,
    load_bc_configs,
    load_bc_state,
    restore_bc_state,
    train_bc,
    wrap_bc_model_for_distributed,
)
from owl.train.config import FullConfig
from owl.train.distributed import broadcast_object, distributed_session
from owl.train.logging import LogMode, MetricLogger
from owl.train.optimizer import create_lr_scheduler, create_optimizer
from owl.train.utils import configure_model_compile

ATTEMPTS = "bc_attempts.jsonl"


class _NoopLogger(MetricLogger):
    """Non-main ranks: W&B belongs to rank 0 only."""

    def __init__(self, run_id: str | None) -> None:
        self._run_id = run_id

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        return None

    def set_summary(
        self,
        key: str,  # noqa: ARG002
        value: int | float | str,  # noqa: ARG002
    ) -> None:
        return None

    def close(self, *, exit_code: int = 0) -> None:  # noqa: ARG002
        return None


def main() -> None:
    args = _parse_args()
    configure_torch()
    with distributed_session() as context:
        resume_dir = None if args.output_dir is not None else args.target
        if resume_dir is None:
            bc_config, ppo_config = load_bc_configs(
                args.target, _parse_overrides(args.overrides)
            )
        else:
            bc_config, ppo_config = load_bc_configs(resume_dir / BC_CONFIG_NAME)
        headroom = check_bc_workload(bc_config, ppo_config)
        if ppo_config.rl.model_compile != "none":
            report = check_compile_stack(installed_compile_stack())
            if context.is_main_process:
                print(
                    f"Compile stack check: torch {report.torch}; triton "
                    f"{report.triton}; NVIDIA driver {report.nvidia_driver}"
                )
        if context.is_main_process:
            for line in headroom_log_lines(headroom):
                print(line)

        dataset = load_bc_dataset(
            args.data, rank=context.rank, world_size=context.world_size
        )
        resume: BCResumeState | None = None
        if resume_dir is not None:
            resume = load_bc_state(resume_dir / BC_STATE)
            check_resume_compatible(
                resume,
                dataset=dataset,
                world_size=context.world_size,
                config=bc_config,
                ppo_config=ppo_config,
            )
            run_dir = resume_dir
        else:
            created = (
                _create_run_dir(args.output_dir) if context.is_main_process else None
            )
            run_dir = broadcast_object(created, context)
            if context.is_main_process:
                ppo_config.to_file(run_dir / PPO_CONFIG_NAME)
                bc_config.model_copy(
                    update={"ppo_config": Path(PPO_CONFIG_NAME)}
                ).to_file(run_dir / BC_CONFIG_NAME)
        attempt = (
            _start_attempt(
                run_dir,
                source_commit=args.source_commit or _git_head(),
                data=args.data,
                dataset_manifest_sha256=dataset.manifest_sha256,
                world_size=context.world_size,
                resume=resume,
            )
            if context.is_main_process
            else None
        )
        provenance: Provenance = broadcast_object(attempt, context)
        context.barrier()

        model = build_bc_model(ppo_config, device=context.device, seed=bc_config.seed)
        compiled = configure_model_compile(model, ppo_config.rl)
        claim = gemm_backend_claim()
        if context.is_main_process:
            print(f"Compiled model regions: {compiled}")
            if claim is not None:
                fields = ", ".join(f"{k}={v}" for k, v in claim.summary().items())
                print(f"Compiled GEMM backends: {fields}")
        optimizer = create_optimizer(model, bc_config.optimizer)
        lr_scheduler = create_lr_scheduler(optimizer, bc_config.optimizer.lr_schedule)
        if resume is not None:
            restore_bc_state(
                resume, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
            )
        wrapped = wrap_bc_model_for_distributed(model, context)

        logger: MetricLogger
        if context.is_main_process:
            logger = create_bc_logger(
                args.log_mode,
                run_dir,
                config=_logger_config(bc_config, ppo_config, provenance),
                wandb_mode=args.wandb_mode,
                resume_run_id=None if resume is None else resume.wandb_run_id,
            )
            run_id = logger.run_id
        else:
            run_id = None
        run_id = broadcast_object(run_id, context)
        if not context.is_main_process:
            logger = _NoopLogger(run_id)
        with _logger_session(logger):
            result = train_bc(
                config=bc_config,
                ppo_config=ppo_config,
                dataset=dataset,
                model=wrapped,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                context=context,
                run_dir=run_dir,
                logger=logger,
                provenance=provenance,
                resume=resume,
                max_runtime_seconds=(
                    None
                    if args.max_runtime_hours is None
                    else args.max_runtime_hours * 3600.0
                ),
            )
        if context.is_main_process:
            print(json.dumps({"run_dir": str(run_dir), **result.__dict__}))


@contextmanager
def _logger_session(logger: MetricLogger) -> Iterator[MetricLogger]:
    """Close the metric run as failed when training raises (as run_ppo)."""
    try:
        yield logger
    except BaseException:
        logger.close(exit_code=1)
        raise
    logger.close(exit_code=0)


def _logger_config(
    bc_config: BCConfig, ppo_config: FullConfig, provenance: Provenance
) -> dict[str, Any]:
    return {
        "bc": bc_config.model_dump(mode="json"),
        "ppo": ppo_config.model_dump(mode="json"),
        "provenance": dict(provenance),
        "method": "BC teacher-forced replay NLL + winner CE on raw final banks",
    }


def _start_attempt(
    run_dir: Path,
    *,
    source_commit: str,
    data: Path,
    dataset_manifest_sha256: str,
    world_size: int,
    resume: BCResumeState | None,
) -> dict[str, object]:
    """Append this launch's attempt record; it is the provenance of its outputs.

    A resume names the ``bc_state.pt`` it continues (SHA-256 read before this
    attempt overwrites it) and carries every earlier attempt's source, since the
    weights it continues were trained under them.
    """
    path = run_dir / ATTEMPTS
    earlier = _read_attempts(path) if resume is not None else []
    if resume is None and path.exists():
        raise FileExistsError(f"fresh BC run already has attempts: {path}")
    if resume is not None and not earlier:
        raise ValueError(f"resume needs the run's attempt records: {path}")
    record: dict[str, object] = {
        "attempt": len(earlier),
        "source_commit": source_commit,
        "attempt_source_commits": [
            *(str(a["source_commit"]) for a in earlier),
            source_commit,
        ],
        "start_step": 0 if resume is None else resume.step,
        "parent_state_sha256": (
            None if resume is None else file_sha256(run_dir / BC_STATE)
        ),
        "dataset_root": str(data.resolve()),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "world_size": world_size,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _read_attempts(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"BC attempt records missing: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("attempt") != index:
            raise ValueError(f"{path} line {index + 1} is not attempt {index}")
        if not isinstance(record.get("source_commit"), str):
            raise ValueError(f"{path} attempt {index} has no source_commit")
    return records


def _git_head() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cannot read the source commit with git; pass --source-commit"
        ) from error
    return f"{commit}-dirty" if status else commit


def _create_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_dir = output_dir / datetime.now().strftime("bc-%Y%m%d-%H%M%S")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            time.sleep(1.0)


def _parse_overrides(raw: list[list[str]] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in itertools.chain.from_iterable(raw or []):
        field_path, separator, value = item.partition("=")
        if not separator or not field_path:
            raise ValueError(f"override must be field.path=value, got {item!r}")
        if field_path in overrides:
            raise ValueError(f"duplicate override field {field_path!r}")
        overrides[field_path] = yaml.safe_load(value)
    return overrides


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target", type=Path, help="BC config YAML, or a BC run dir to resume"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Task 5.1 BC dataset root"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Parent of a fresh run dir"
    )
    parser.add_argument(
        "--log-mode", type=LogMode, choices=list(LogMode), default=LogMode.WANDB
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
        help="offline keeps telemetry local for a later `wandb sync`",
    )
    parser.add_argument("--max-runtime-hours", type=float, default=None)
    parser.add_argument(
        "--source-commit",
        default=None,
        help="This attempt's source identity when the checkout has no git metadata",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="+",
        action="append",
        default=None,
        metavar="field.path=value",
    )
    args = parser.parse_args()
    fresh = args.output_dir is not None
    if fresh and not args.target.is_file():
        raise ValueError(f"fresh BC config does not exist: {args.target}")
    if not fresh:
        if not args.target.is_dir():
            raise ValueError(f"resume needs a BC run directory, got {args.target}")
        if args.overrides is not None:
            raise ValueError("resume launches cannot use config overrides")
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0.0:
        raise ValueError("--max-runtime-hours must be positive")
    return args


if __name__ == "__main__":
    main()
