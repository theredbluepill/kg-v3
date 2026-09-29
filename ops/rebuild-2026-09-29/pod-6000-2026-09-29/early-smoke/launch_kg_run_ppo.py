"""Run scripts/run_ppo.py's own main() with receipt-only instrumentation.

Pre-landing pod diagnostics (branch kg/pod-ppo-prelanding, pending Codex review).
The trainer, model, env and logger code paths are unchanged; this wrapper only
observes them:

- wandb.init: requires project "kg-v3" (set by owl.train.logging for
  Kaggriculture) and refuses a run that is offline or disabled; prints the URL.
- PPOTrainer._collect_rollout / _precompute_teacher_targets / _update and
  run_ppo._evaluate_against_last_best: per-phase wall time and CUDA peak
  allocated/reserved bytes (peak stats reset at each phase start; synchronize
  before and after, which adds a small per-phase cost).
- KaggricultureVectorizedEnv.__init__/step: seed and stride per construction,
  native step wall time (synchronous: fence + native step + buffer publish),
  and terminal-record counts plus the first terminal metrics sample.
- KG_PROBE_FAIL_BEFORE_UPDATE=1: raise inside the first _update call before
  any minibatch runs (isolated failure-status probe; no model or native state
  is corrupted).

Every record is one line "[kg-probe] {json}" on stdout, tagged with RANK.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import wandb

RANK = int(os.environ.get("RANK", "0"))
_FAIL_BEFORE_UPDATE = os.environ.get("KG_PROBE_FAIL_BEFORE_UPDATE") == "1"
_PHASE_PEAKS: dict[str, dict[str, int]] = {}
_STEP = {"calls": 0, "seconds": 0.0, "terminal_records": 0, "sample_logged": 0}
_ITER = {"n": 0}


def emit(kind: str, **fields: Any) -> None:
    record = {"kind": kind, "rank": RANK, "t": time.time(), **fields}
    print("[kg-probe] " + json.dumps(record, default=str), flush=True)


_original_init = wandb.init


def _checked_init(*args: Any, **kwargs: Any) -> Any:
    if kwargs.get("project") != "kg-v3":
        raise RuntimeError(f"unexpected W&B project {kwargs.get('project')!r}")
    run = _original_init(*args, **kwargs)
    if run.offline or run.disabled:
        raise RuntimeError("W&B run is not online; refusing to continue")
    emit("wandb", url=run.url, id=run.id, name=run.name, project=kwargs["project"])
    return run


wandb.init = _checked_init


def _cuda() -> bool:
    return torch.cuda.is_available()


def _phase(name: str, fn: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if _cuda():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        steps_before = _STEP["calls"]
        seconds_before = _STEP["seconds"]
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if _cuda():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record: dict[str, Any] = {"phase": name, "iteration": _ITER["n"], "seconds": elapsed}
        if _cuda():
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            record.update(
                peak_allocated=allocated,
                peak_reserved=reserved,
                device_total=total,
                reserved_fraction=reserved / total,
            )
            peaks = _PHASE_PEAKS.setdefault(name, {"allocated": 0, "reserved": 0})
            peaks["allocated"] = max(peaks["allocated"], allocated)
            peaks["reserved"] = max(peaks["reserved"], reserved)
        record["native_steps"] = _STEP["calls"] - steps_before
        record["native_step_seconds"] = _STEP["seconds"] - seconds_before
        emit("phase", **record)
        return result

    return wrapped


def _instrument_trainer() -> None:
    from owl.train import ppo

    trainer = ppo.PPOTrainer
    trainer._collect_rollout = _phase("rollout", trainer._collect_rollout)
    trainer._precompute_teacher_targets = _phase(
        "teacher_precompute", trainer._precompute_teacher_targets
    )
    original_update = trainer._update

    def update(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _FAIL_BEFORE_UPDATE:
            emit("probe_failure", message="intentional failure before the first update")
            raise RuntimeError("KG_PROBE_FAIL_BEFORE_UPDATE: intentional pre-update failure")
        return original_update(self, *args, **kwargs)

    trainer._update = _phase("update", update)
    original_iteration = trainer.train_iteration

    def train_iteration(self: Any) -> Any:
        _ITER["n"] += 1
        start = time.perf_counter()
        metrics = original_iteration(self)
        keep = dict(metrics)
        emit(
            "iteration",
            iteration=_ITER["n"],
            wall_seconds=time.perf_counter() - start,
            optimizer_steps=self.optimizer_steps,
            metrics=keep,
        )
        return metrics

    trainer.train_iteration = train_iteration


def _instrument_env() -> None:
    from owl.kaggriculture import env as kenv

    cls = kenv.KaggricultureVectorizedEnv
    original_init = cls.__init__
    original_step = cls.step

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        emit(
            "env_construct",
            n_envs=kwargs.get("n_envs"),
            seed=kwargs.get("seed"),
            seed_stride=kwargs.get("seed_stride"),
            native_threads=kwargs.get("native_threads"),
            pin_memory=kwargs.get("pin_memory"),
            transfer_device=str(kwargs.get("transfer_device")),
            episode_steps=kwargs["config"].episode_steps if "config" in kwargs else None,
        )
        original_init(self, *args, **kwargs)

    def step(self: Any, actions: Any) -> Any:
        start = time.perf_counter()
        result = original_step(self, actions)
        _STEP["seconds"] += time.perf_counter() - start
        _STEP["calls"] += 1
        metrics = result[3]
        lengths = {k: len(v) for k, v in metrics.items()}
        if lengths:
            _STEP["terminal_records"] += max(lengths.values())
            if _STEP["sample_logged"] < 2 and max(lengths.values()) > 0:
                _STEP["sample_logged"] += 1
                emit(
                    "terminal_sample",
                    step_call=_STEP["calls"],
                    metrics={k: list(v)[:4] for k, v in metrics.items()},
                    counts=lengths,
                    dones_any=bool(result[2].any()),
                )
        return result

    cls.__init__ = init
    cls.step = step


def _report_exit() -> None:
    record: dict[str, Any] = {
        "native_step_calls": _STEP["calls"],
        "native_step_seconds": _STEP["seconds"],
        "native_step_mean_ms": 1000 * _STEP["seconds"] / max(_STEP["calls"], 1),
        "terminal_records": _STEP["terminal_records"],
        "phase_peaks": _PHASE_PEAKS,
    }
    try:
        from torch._inductor import config as inductor_config

        record["inductor_max_autotune_gemm_backends"] = (
            inductor_config.max_autotune_gemm_backends
        )
    except Exception as error:  # receipt only
        record["inductor_config_error"] = repr(error)
    emit("exit", **record)


atexit.register(_report_exit)


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    _instrument_trainer()
    _instrument_env()
    spec = importlib.util.spec_from_file_location("run_ppo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ppo"] = module
    spec.loader.exec_module(module)
    module._evaluate_against_last_best = _phase(
        "evaluation", module._evaluate_against_last_best
    )
    emit("start", argv=sys.argv, torch=torch.__version__, wandb=wandb.__version__)
    module.main()


if __name__ == "__main__":
    main()
