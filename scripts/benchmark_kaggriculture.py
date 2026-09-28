"""Measure complete shared-PPO updates on a bounded, source-bound CUDA run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from owl import rs
from owl.model.attn import flash_attn_available
from owl.train import FullConfig


def summarize_updates(
    updates: list[dict[str, float]], *, warmup: int, measured: int
) -> dict[str, Any]:
    if len(updates) != warmup + measured:
        raise ValueError(f"expected {warmup + measured} updates, got {len(updates)}")
    if any(not math.isfinite(value) for item in updates for value in item.values()):
        raise ValueError("non-finite training metrics cannot qualify throughput")
    selected = updates[warmup:]
    previous = updates[warmup - 1] if warmup else {}
    elapsed = sum(item["time/iteration_seconds"] for item in selected)
    steps = selected[-1]["train/env_steps"] - previous.get("train/env_steps", 0)
    turns = selected[-1]["train/player_step_total"] - previous.get(
        "train/player_step_total", 0
    )
    return {
        "warmup_updates_excluded": warmup,
        "measured_updates": measured,
        "game_transitions": steps,
        "learner_seat_turns": turns,
        "complete_update_seconds": elapsed,
        "game_sps": steps / elapsed,
        "learner_turns_per_second": turns / elapsed,
        "phase_median_seconds": {
            name: statistics.median(item[f"time/{name}_seconds"] for item in selected)
            for name in ("rollout", "update", "iteration")
        },
        "iteration_seconds_range": [
            min(item["time/iteration_seconds"] for item in selected),
            max(item["time/iteration_seconds"] for item in selected),
        ],
        "scope": "Native collection + model + PPO optimizer, excluding startup, "
        "warmup, evaluation and checkpoint file writes; no playing-strength claim.",
    }


def _source_hashes() -> dict[str, str]:
    files = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], text=True
    ).split("\0")
    prefixes = ("python/", "src/", "engine_rs/", "scripts/", "configs/")
    roots = {"Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock"}
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in sorted(set(files))
        if name
        and (name.startswith(prefixes) or name in roots)
        and Path(name).is_file()
    }


def _runtime_identity(*, dtype: str) -> dict[str, Any]:
    native_path = Path(rs.__file__).resolve()
    has_flash_attn = flash_attn_available()
    nsys_path = shutil.which("nsys")
    return {
        "native_extension": {
            "path": str(native_path),
            "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        },
        "attention": {
            "flash_attn_package_available": has_flash_attn,
            "expected_trunk_api": (
                "flash_attn_varlen_func"
                if has_flash_attn and dtype == "bfloat16"
                else "torch.nn.functional.scaled_dot_product_attention"
            ),
            "kernel_qualification": "Dispatch expectation from CUDA config/package "
            "availability; executed kernels require timeline verification.",
        },
        "nsys": {
            "path": nsys_path,
            "version": (
                subprocess.check_output([nsys_path, "--version"], text=True).strip()
                if nsys_path is not None
                else None
            ),
        },
    }


def _optimization_work(
    updates: list[dict[str, float]], *, warmup: int
) -> dict[str, float | str]:
    selected = updates[warmup:]
    previous_steps = updates[warmup - 1]["optimizer/steps"] if warmup else 0
    return {
        "optimizer_steps_per_rank": selected[-1]["optimizer/steps"] - previous_steps,
        "target_kl_stopped_updates": sum(
            item["policy/target_kl_exceeded"] for item in selected
        ),
        "optimization_work_note": "Target-KL stopping can shorten an update; "
        "optimizer/steps measures executed synchronized optimizer steps per rank. "
        "optimizer/minibatches_per_update is the planned count, not executed work.",
    }


def _observation_work(
    updates: list[dict[str, float]], *, warmup: int
) -> dict[str, float | str]:
    selected = updates[warmup:]
    previous = updates[warmup - 1] if warmup else {}
    game_steps = selected[-1]["train/env_steps"] - previous.get("train/env_steps", 0)
    seat_turns = selected[-1]["train/player_step_total"] - previous.get(
        "train/player_step_total", 0
    )
    actor_occurrences = selected[-1]["train/total_active_entities"] - previous.get(
        "train/total_active_entities", 0
    )
    action_frames = sum(item["train/action_frames"] for item in selected)
    if game_steps <= 0 or seat_turns <= 0:
        raise ValueError("workload means require positive game and learner-seat counts")
    return {
        "observed_seat_observations": 2 * game_steps,
        "observed_own_actor_occurrences": actor_occurrences,
        "mean_own_actors_per_seat_observation": actor_occurrences / (2 * game_steps),
        "max_own_actors_per_seat_observation": max(
            item["train/max_entities"] for item in selected
        ),
        "action_frames": action_frames,
        "mean_action_frames_per_learner_seat_turn": action_frames / seat_turns,
        "observation_work_note": "Actor occurrences sum own-actor entity masks over "
        "all observed seats, including any inactive seats; they are not unique "
        "actors or an executable-action count. Frame lengths include actor/market "
        "frames and STOP, excluding zero padding; their denominator is active "
        "learner-seat turns. These aggregates do not establish a distribution.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/kaggriculture_2rank.yaml")
    )
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--measured-updates", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture an nsys run including startup; profiling changes timing.",
    )
    args = parser.parse_args()
    if args.ranks < 1 or args.warmup_updates < 1 or args.measured_updates < 1:
        parser.error("ranks, warmup-updates and measured-updates must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.ranks:
        parser.error(
            f"requires {args.ranks} CUDA devices; CPU diagnostics cannot qualify SPS"
        )
    if args.profile and shutil.which("nsys") is None:
        parser.error("--profile requires Nsight Systems (nsys)")
    config = FullConfig.from_file(
        args.config,
        overrides={"rl.checkpoint_freq": None, "runtime.n_runtime_gpus": args.ranks},
    )
    if config.model.model_arch != "kaggriculture_transformer":
        parser.error("config must select the Kaggriculture model")
    output = args.output or Path("runs") / (
        "perf-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    steps = (
        (args.warmup_updates + args.measured_updates)
        * args.ranks
        * config.env.n_envs
        * config.rl.horizon
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.ranks}",
        "scripts/run_ppo.py",
        str(args.config),
        str(output / "training"),
        "--log-mode",
        "debug",
        "--max-env-steps",
        str(steps),
        "-o",
        "rl.checkpoint_freq=null",
    ]
    if args.profile:
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--kill=none",
            "--wait=all",
            "--output=" + str(output / "timeline"),
            *command,
        ]
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "torch": str(torch.__version__),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "ranks": args.ranks,
        "devices": [torch.cuda.get_device_name(i) for i in range(args.ranks)],
        "profiled": args.profile,
        "config": config.model_dump(mode="json"),
        "source_hashes": _source_hashes(),
        **_runtime_identity(dtype=config.rl.dtype),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit",
                "--format=csv",
            ],
            text=True,
        ).strip(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    updates: list[dict[str, float]] = []
    received_at: list[float] = []
    with (output / "stdout.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            start = line.find("{'loss/")
            if start >= 0:
                metrics = ast.literal_eval(line[start:].strip())
                if "time/iteration_seconds" in metrics:
                    updates.append(metrics)
                    received_at.append(perf_counter())
        returncode = process.wait()
    if returncode:
        raise SystemExit(
            f"trainer failed ({returncode}); inspect {output / 'stdout.log'}"
        )
    result = summarize_updates(
        updates, warmup=args.warmup_updates, measured=args.measured_updates
    )
    result.update(_optimization_work(updates, warmup=args.warmup_updates))
    result.update(_observation_work(updates, warmup=args.warmup_updates))
    # Consecutive flushed metrics bracket the entire steady-state loop, including
    # trainer metric reductions and logging omitted by inner phase timers.
    wall_seconds = received_at[-1] - received_at[args.warmup_updates - 1]
    result["observed_process_wall_seconds"] = wall_seconds
    result["observed_game_sps"] = result["game_transitions"] / wall_seconds
    result["observed_learner_turns_per_second"] = (
        result["learner_seat_turns"] / wall_seconds
    )
    result["phase_timing_note"] = (
        "Inherited host timers; use the nsys timeline for CUDA phase attribution."
    )
    result["profiled"] = args.profile
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
