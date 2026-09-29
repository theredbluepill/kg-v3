"""Behavior cloning warm start for Kaggriculture PPO (rebuild plan Task 5.2).

Fresh run:  train_bc.py CONFIG --data DATASET --output-dir DIR
Resume:     train_bc.py RUN_DIR --data DATASET

Launch multi-GPU runs with torchrun like ``scripts/run_ppo.py``. The run
directory holds the PPO ``config.yaml`` the checkpoint belongs to, the resolved
``bc_config.yaml``, ``launch.json`` provenance, ``checkpoint_bc_best.pt`` (the
run_ppo checkpoint schema; start PPO with ``--load-model-weights``) and its
``checkpoint_bc_best.json`` record, ``bc_state.pt`` for restarts,
``bc_history.jsonl`` (the held-out NLL curve) and ``bc_result.json``.
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
    build_bc_model,
    check_bc_workload,
    check_resume_compatible,
    create_bc_logger,
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

LAUNCH_RECORD = "launch.json"


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
                resume, dataset=dataset, world_size=context.world_size
            )
            run_dir = resume_dir
            provenance = _read_provenance(run_dir)
        else:
            provenance = {
                "source_commit": args.source_commit or _git_head(),
                "dataset_root": str(args.data.resolve()),
                "dataset_manifest_sha256": dataset.manifest_sha256,
                "world_size": context.world_size,
            }
            created = (
                _create_run_dir(args.output_dir) if context.is_main_process else None
            )
            run_dir = broadcast_object(created, context)
            if context.is_main_process:
                ppo_config.to_file(run_dir / PPO_CONFIG_NAME)
                bc_config.model_copy(
                    update={"ppo_config": Path(PPO_CONFIG_NAME)}
                ).to_file(run_dir / BC_CONFIG_NAME)
                (run_dir / LAUNCH_RECORD).write_text(
                    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
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
    bc_config: BCConfig, ppo_config: FullConfig, provenance: dict[str, str | int]
) -> dict[str, Any]:
    return {
        "bc": bc_config.model_dump(mode="json"),
        "ppo": ppo_config.model_dump(mode="json"),
        "provenance": provenance,
        "method": "BC teacher-forced replay NLL + winner CE on raw final banks",
    }


def _read_provenance(run_dir: Path) -> dict[str, str | int]:
    record = json.loads((run_dir / LAUNCH_RECORD).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{run_dir / LAUNCH_RECORD} must hold a JSON object")
    return {str(k): v for k, v in record.items() if isinstance(v, str | int)}


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
        help="Recorded source identity when the checkout has no git metadata",
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
        if args.source_commit is not None:
            raise ValueError("resume keeps the run's recorded source commit")
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0.0:
        raise ValueError("--max-runtime-hours must be positive")
    return args


if __name__ == "__main__":
    main()
