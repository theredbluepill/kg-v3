"""Launch scripts/run_ppo.py unchanged, with W&B routed to the v3 project.

python/owl/train/logging.py at 994818b hard-codes ``project="orbit-wars"`` in
``wandb.init``; init arguments take precedence over WANDB_PROJECT in wandb
0.26.1, so the env var cannot route the run. This shim rewrites only that one
argument (fail-fast if it is not the expected value), refuses a non-online run,
and prints the process's CUDA peak memory at exit. Everything else is
run_ppo.py's own code path, executed via runpy as __main__.
"""

from __future__ import annotations

import atexit
import json
import runpy
import sys

import torch
import wandb

EXPECTED_PROJECT = "orbit-wars"
V3_PROJECT = "kg-v3"
_original_init = wandb.init


def _v3_init(*args, **kwargs):
    if kwargs.get("project") != EXPECTED_PROJECT:
        raise RuntimeError(f"unexpected wandb project argument: {kwargs.get('project')!r}")
    kwargs["project"] = V3_PROJECT
    run = _original_init(*args, **kwargs)
    if run.offline or run.disabled:
        raise RuntimeError("W&B run is not online; refusing to continue")
    print(f"[launch-v3] wandb online run url={run.url}", flush=True)
    return run


wandb.init = _v3_init


def _report_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    report = {
        f"cuda:{i}": {
            "max_memory_allocated": torch.cuda.max_memory_allocated(i),
            "max_memory_reserved": torch.cuda.max_memory_reserved(i),
        }
        for i in range(torch.cuda.device_count())
    }
    print(f"[launch-v3] peak_memory {json.dumps(report)}", flush=True)


atexit.register(_report_peak_memory)

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
