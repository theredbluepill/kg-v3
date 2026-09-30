"""Run scripts/run_ppo.py's main() with observe-only throughput receipts.

native_threads sweep, pod abl4mvr5w1mmn4 (see run-statement.md). The trainer,
model, env and logger code paths are unchanged. Records are single lines
"[nt-probe] {json}" on stdout, tagged with RANK.

- KG_NT_NUMA=1: before torch loads, bind this rank's CPU affinity and memory
  policy (MPOL_BIND via set_mempolicy) to its GPU's NUMA node. The pod's
  container returns EPERM for set_mempolicy. KG_NT_NUMA=cpu binds the CPU
  affinity only.
- KaggricultureVectorizedEnv.step: wall time per call (synchronous: fence +
  native step + buffer publish).
- PPOTrainer.train_iteration: the trainer's metrics subset, native step time
  in the iteration, process CPU seconds and machine /proc/stat busy/total.
- wandb.init: refuses a run that is not online in project kg-v3.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
# Pod abl4mvr5w1mmn4 topology (env/pod/hardware.txt).
GPU_NODE = {0: 0, 1: 0, 2: 1, 3: 1}
NODE_CPUS = {
    0: [*range(0, 64), *range(128, 192)],
    1: [*range(64, 128), *range(192, 256)],
}


def emit(kind: str, **fields: Any) -> None:
    record = {"kind": kind, "rank": RANK, "t": time.time(), **fields}
    print("[nt-probe] " + json.dumps(record, default=str), flush=True)


def _bind_numa(*, memory: bool) -> None:
    node = GPU_NODE[LOCAL_RANK]
    os.sched_setaffinity(0, NODE_CPUS[node])
    if not memory:
        emit("numa_bind", local_rank=LOCAL_RANK, node=node,
             cpus=len(os.sched_getaffinity(0)), mempolicy_mode=None)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(1 << node)
    # x86_64: set_mempolicy = 238, get_mempolicy = 239; MPOL_BIND = 2.
    if libc.syscall(238, 2, ctypes.byref(mask), 64) != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    mode = ctypes.c_int(-1)
    got = ctypes.c_ulong(0)
    if libc.syscall(239, ctypes.byref(mode), ctypes.byref(got), 64, None, 0) != 0:
        raise OSError(ctypes.get_errno(), "get_mempolicy failed")
    emit(
        "numa_bind",
        local_rank=LOCAL_RANK,
        node=node,
        cpus=len(os.sched_getaffinity(0)),
        mempolicy_mode=mode.value,
        mempolicy_nodemask=got.value,
    )


# "1": CPU affinity + MPOL_BIND (EPERM in this container); "cpu": affinity only.
if os.environ.get("KG_NT_NUMA") in ("1", "cpu"):
    _bind_numa(memory=os.environ["KG_NT_NUMA"] == "1")

import torch  # noqa: E402
import wandb  # noqa: E402

_STEP = {"calls": 0, "seconds": 0.0}
_ITER = {"n": 0}
_original_init = wandb.init


def _checked_init(*args: Any, **kwargs: Any) -> Any:
    if kwargs.get("project") != "kg-v3":
        raise RuntimeError(f"unexpected W&B project {kwargs.get('project')!r}")
    run = _original_init(*args, **kwargs)
    if run.offline or run.disabled:
        raise RuntimeError("W&B run is not online; refusing to continue")
    emit("wandb", url=run.url, id=run.id, name=run.name, group=kwargs.get("group"))
    return run


wandb.init = _checked_init


def _proc_stat() -> tuple[int, int]:
    with open("/proc/stat") as handle:
        fields = [int(x) for x in handle.readline().split()[1:]]
    idle = fields[3] + fields[4]
    total = sum(fields[:8])
    return total - idle, total


_KEEP = (
    "time/",
    "perf/",
    "train/player_step_total",
    "train/total_games_played",
    "optimizer/steps",
)


def _instrument() -> None:
    from owl.kaggriculture import env as kenv
    from owl.train import ppo

    cls = kenv.KaggricultureVectorizedEnv
    original_step = cls.step
    original_init = cls.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        emit(
            "env_construct",
            n_envs=kwargs.get("n_envs"),
            native_threads=kwargs.get("native_threads"),
            transfer_device=str(kwargs.get("transfer_device")),
            affinity_cpus=len(os.sched_getaffinity(0)),
        )
        original_init(self, *args, **kwargs)

    def step(self: Any, actions: Any) -> Any:
        start = time.perf_counter()
        result = original_step(self, actions)
        _STEP["seconds"] += time.perf_counter() - start
        _STEP["calls"] += 1
        return result

    cls.__init__ = init
    cls.step = step

    original_iteration = ppo.PPOTrainer.train_iteration

    def train_iteration(self: Any) -> Any:
        _ITER["n"] += 1
        calls0, secs0 = _STEP["calls"], _STEP["seconds"]
        cpu0 = os.times()
        start = time.perf_counter()
        metrics = original_iteration(self)
        end = time.perf_counter()
        cpu1 = os.times()
        busy, total = _proc_stat()
        emit(
            "iteration",
            iteration=_ITER["n"],
            perf_end=end,
            wall_seconds=end - start,
            native_step_calls=_STEP["calls"] - calls0,
            native_step_seconds=_STEP["seconds"] - secs0,
            proc_cpu_seconds=(cpu1.user + cpu1.system) - (cpu0.user + cpu0.system),
            proc_cpu_total=cpu1.user + cpu1.system,
            stat_busy=busy,
            stat_total=total,
            metrics={k: v for k, v in metrics.items() if k.startswith(_KEEP)},
        )
        return metrics

    ppo.PPOTrainer.train_iteration = train_iteration


def main() -> None:
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    _instrument()
    spec = importlib.util.spec_from_file_location("run_ppo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ppo"] = module
    spec.loader.exec_module(module)
    emit(
        "start",
        argv=sys.argv,
        torch=torch.__version__,
        omp=os.environ.get("OMP_NUM_THREADS"),
        numa=os.environ.get("KG_NT_NUMA"),
    )
    module.main()


if __name__ == "__main__":
    main()
