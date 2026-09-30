"""Plan 6.3b component probe: NCCL all-reduce of one gradient-sized buffer.

A fallback for the all-reduce share when ``nsys`` is absent, and a
cross-check when it is present. Run on the idle 8-GPU pod:

    uv run --no-sync torchrun --nproc-per-node 8 \
        ops/rebuild-2026-09-29/8rank-run/allreduce_bench.py --out RECEIPT_DIR

It builds the real Kaggriculture model from the config to count trainable
parameters, then times ``dist.all_reduce`` of one FP32 buffer of that size
(the gradient bytes DDP reduces per optimizer step) with CUDA events: 10
warm-up and 50 timed calls. Rank 0 writes ``allreduce.json`` with the
per-call median and the per-iteration cost at 16 optimizer steps.

This is a component measurement on the pod's interconnect. It is not the
exposed cost inside a training step: DDP splits the gradient into buckets and
overlaps them with the backward pass, so the share of ``time/update_seconds``
it implies is an upper bound on the exposed all-reduce cost.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from owl.model import create_model
from owl.train import FullConfig

OPTIMIZER_STEPS_PER_ITERATION = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kaggriculture_8rank_bc_finetune.yaml"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    cfg = FullConfig.from_file(args.config)
    model = create_model(
        cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
    )
    numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer = torch.ones(numel, dtype=torch.float32, device="cuda")
    for _ in range(10):
        dist.all_reduce(buffer)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(buffer)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    medians = [None] * dist.get_world_size()
    dist.all_gather_object(medians, statistics.median(times))
    if rank == 0:
        median = max(m for m in medians if m is not None)
        report = {
            "world_size": dist.get_world_size(),
            "trainable_params": numel,
            "bytes": numel * 4,
            "median_ms_per_call_by_rank": medians,
            "slowest_rank_median_ms": median,
            "algbw_GBps": numel * 4 / (median / 1e3) / 1e9,
            "ms_per_iteration_at_16_steps": median * OPTIMIZER_STEPS_PER_ITERATION,
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "allreduce.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
