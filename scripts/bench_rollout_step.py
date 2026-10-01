"""Bounded, W&B-free Kaggriculture rollout component A/B (not learner SPS).

Default: 40 seat rows, one switch at a time and all switches. CUDA uses the
supplied config's precision and trunk compile; CPU uses FP32 and an eager
trunk. Pass --with-env to include the native step, telemetry and next-observation
copy. Every arm recreates the same seeded model/env; warmup and the first step
(including lazy compilation) are separate from steady timing. No optimizer,
teacher or checkpoint is involved. Action/state/RNG hashes compare equivalent
trajectories; this short random-initialization workload is not trained-policy SPS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from owl import rs
from owl.game import create_env
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.types import KaggricultureActions
from owl.model.attn import flash_attn_available
from owl.model.compile_gemm import installed_compile_stack
from owl.model.factory import create_model
from owl.model.kaggriculture import KaggricultureTransformerConfig
from owl.train.config import FullConfig
from owl.train.ppo import (
    PinnedActionTransfer,
    _actions_to_cpu,
    _copy_obs_to_device_,
    _map_observation,
    _obs_to_device,
)
from owl.train.utils import autocast_context, configure_model_compile, configure_torch

_VARIANTS = ("baseline", "actor", "packing", "d2h", "telemetry", "all")


def _parity_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if not arms:
        return {"reference_variant": None, "comparison_count": 0, "comparisons": []}
    reference = next((arm for arm in arms if arm["variant"] == "baseline"), arms[0])
    fields = (
        "action_sha256",
        "final_observation_sha256",
        "cpu_rng_sha256",
        "cuda_rng_sha256",
    )
    comparisons = [
        {
            "variant": arm["variant"],
            "matches": {field: arm[field] == reference[field] for field in fields},
            "all_exact": all(arm[field] == reference[field] for field in fields),
        }
        for arm in arms
        if arm is not reference
    ]
    return {
        "reference_variant": reference["variant"],
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_arm(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    device = torch.device(args.device)
    enabled = variant == "all"
    overrides: dict[str, Any] = {
        "env.n_envs": args.seat_rows // 2,
        "env.seed": args.seed,
        "env.pin_memory": device.type == "cuda",
        "env.opponent_mix": None,
        "env.skip_reward_telemetry_validation": enabled or variant == "telemetry",
        "rl.segments_per_minibatch": 1,
        "rl.gradient_accumulation_steps": 1,
        "rl.eval_replay_games": 0,
        "rl.compile_actor_heads": enabled or variant == "actor",
        "rl.rollout_packing": enabled or variant == "packing",
        "rl.pinned_action_d2h": enabled or variant == "d2h",
    }
    if device.type == "cpu":
        overrides |= {
            "model.force_flash_attn": False,
            "rl.model_compile": "none",
            "rl.dtype": "float32",
        }
    cfg = FullConfig.from_file(args.config, overrides)
    if not isinstance(cfg.env, KaggricultureEnvConfig) or not isinstance(
        cfg.model, KaggricultureTransformerConfig
    ):
        raise ValueError("benchmark requires a Kaggriculture config")
    torch.manual_seed(args.seed)
    env = create_env(
        cfg.env,
        n_envs=cfg.env.n_envs,
        base_seed=args.seed,
        rank=0,
        world_size=1,
        pin_memory=device.type == "cuda",
        transfer_device=device,
    )
    assert isinstance(env, KaggricultureVectorizedEnv)
    model = (
        create_model(
            cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec
        )
        .to(device)
        .eval()
    )
    configure_model_compile(model, cfg.rl)
    transfer = (
        PinnedActionTransfer(env.n_envs, device) if cfg.rl.pinned_action_d2h else None
    )
    host = env.reset()
    obs = _obs_to_device(host, device, non_blocking=device.type == "cuda")
    action_hash = hashlib.sha256()

    def step() -> KaggricultureActions:
        context = (
            model.rollout_packing(host) if cfg.rl.rollout_packing else nullcontext()
        )
        with context, autocast_context(cfg.rl, device):
            output = model(obs)
        actions = (
            transfer.to_cpu(output.actions)
            if transfer is not None
            else _actions_to_cpu(output.actions)
        )
        if not isinstance(actions, KaggricultureActions):
            raise TypeError("Kaggriculture model must produce KaggricultureActions")
        if args.with_env:
            # Native grammar decode, execution and reward telemetry are inside.
            next_obs, _, _, _ = env.step(actions)
            _copy_obs_to_device_(obs, next_obs, non_blocking=device.type == "cuda")
        return actions

    def record_actions(actions: KaggricultureActions) -> None:
        # The actions are already completed CPU copies; hashing stays outside
        # every timed interval and precedes reusable pinned-buffer overwrite.
        action_hash.update(actions.tokens.numpy().tobytes())
        action_hash.update(actions.lengths.numpy().tobytes())

    _synchronize(device)
    with torch.no_grad():
        cold_start = perf_counter()
        actions = step()
        _synchronize(device)
        cold_seconds = perf_counter() - cold_start
        record_actions(actions)
        for _ in range(args.warmup):
            record_actions(step())
        _synchronize(device)
        times = []
        for _ in range(args.steps):
            start = perf_counter()
            actions = step()
            _synchronize(device)
            times.append(perf_counter() - start)
            record_actions(actions)
        # Outside timing; bind all final observation fields, not just banks
        # (banks alone are often still zero even after divergent trajectories).
        observation_hash = hashlib.sha256()

        def record_observation(tensor: torch.Tensor) -> torch.Tensor:
            observation_hash.update(str((tensor.shape, tensor.dtype)).encode())
            observation_hash.update(tensor.numpy().tobytes())
            return tensor

        _map_observation(env.observations, record_observation)
        cpu_rng_hash = hashlib.sha256(torch.random.get_rng_state().numpy().tobytes())
        cuda_rng_hash = (
            hashlib.sha256(
                torch.cuda.get_rng_state(device).numpy().tobytes()
            ).hexdigest()
            if device.type == "cuda"
            else None
        )
    mean = statistics.mean(times)
    return {
        "variant": variant,
        "resolved_config": cfg.model_dump(mode="json"),
        "cold_first_step_seconds": cold_seconds,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "step_seconds": times,
        "mean_step_seconds": mean,
        "median_step_seconds": statistics.median(times),
        "seat_rows_per_second": args.seat_rows / mean,
        "game_steps_per_second": env.n_envs / mean if args.with_env else None,
        "hashed_steps": 1 + args.warmup + args.steps,
        "action_sha256": action_hash.hexdigest(),
        "final_observation_sha256": observation_hash.hexdigest(),
        "cpu_rng_sha256": cpu_rng_hash.hexdigest(),
        "cuda_rng_sha256": cuda_rng_hash,
        "actor_compile_effective": cfg.rl.compile_actor_heads,
        "packing_effective": cfg.rl.rollout_packing
        and device.type == "cuda"
        and cfg.rl.dtype == "bfloat16"
        and flash_attn_available(),
        "d2h_effective": cfg.rl.pinned_action_d2h and device.type == "cuda",
        "telemetry_effective": cfg.env.skip_reward_telemetry_validation
        and args.with_env
        and (
            cfg.env.reward_shaping.econ_bank_weight > 0
            or cfg.env.reward_shaping.econ_margin_weight > 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kaggriculture_4rank_bank_critic.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seat-rows", type=int, default=40)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--with-env", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", choices=_VARIANTS, default=list(_VARIANTS)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.seat_rows < 2
        or args.seat_rows % 2
        or args.steps < 1
        or args.warmup < 0
        or args.threads < 1
    ):
        parser.error(
            "seat-rows must be positive/even, steps/threads positive, "
            "warmup nonnegative"
        )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    configure_torch()
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "scripts/bench_rollout_step.py",
        "python/owl/train/ppo.py",
        "python/owl/model/kaggriculture.py",
        "python/owl/model/kaggriculture_actor.py",
        "python/owl/model/stateless_transformer_v1.py",
        "python/owl/kaggriculture/env.py",
        "python/owl/kaggriculture/rewards.py",
        "python/owl/train/utils.py",
        "python/owl/train/config.py",
        "python/owl/model/compile_gemm.py",
        "python/owl/kaggriculture/config.py",
        "python/owl/game.py",
    ]
    try:
        flash_version = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        flash_version = None
    native_path = Path(rs.__file__).resolve()
    result: dict[str, Any] = {
        "scope": (
            "forward + within-turn decode + completed CPU action transfer; "
            "optional native step/H2D; no PPO update"
        ),
        "device": args.device,
        "host": platform.platform(),
        "torch": str(torch.__version__),
        "compile_stack": asdict(installed_compile_stack()),
        "flash_attn_version": flash_version,
        "torch_cuda_version": torch.version.cuda,
        "native_binary_path": str(native_path),
        "native_binary_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "threads": args.threads,
        "seed": args.seed,
        "seat_rows": args.seat_rows,
        "with_env": args.with_env,
        "input_config": str(args.config.resolve()),
        "input_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths
        },
        "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "arms": [],
    }
    for variant in args.variants:
        arm = benchmark_arm(args, variant)
        result["arms"].append(arm)
        result["parity"] = _parity_summary(result["arms"])
        print(
            json.dumps(
                {
                    k: arm[k]
                    for k in (
                        "variant",
                        "cold_first_step_seconds",
                        "mean_step_seconds",
                        "seat_rows_per_second",
                    )
                }
            ),
            flush=True,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.output is None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
