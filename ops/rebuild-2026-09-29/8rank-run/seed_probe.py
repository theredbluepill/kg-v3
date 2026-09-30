"""Plan 6.3b: rank seed streams at world size 8 are disjoint (CPU probe).

run_ppo does not log per-rank seeds, so this probe builds each rank's native
Kaggriculture env exactly as run_ppo's fresh launch does (``owl.game.create_env``
with ``run_ppo._kaggriculture_rollout_base_seed``, ranks 0..world_size-1, run
one at a time on the CPU) and reads the native ``seed_state``. It checks that
every rank's construction seeds lie in its residue class ``base + rank`` modulo
the world size, that no seed repeats across ranks, and that each rank's next
seed is the one after its last. Resets and resumes are covered by
``tests/kaggriculture/test_native_env.py::test_seed_partition_*[8]`` and
``tests/scripts/test_run_ppo.py``'s resume-seed tests.

    uv run --no-sync python ops/rebuild-2026-09-29/8rank-run/seed_probe.py \
        configs/kaggriculture_8rank_bc_finetune.yaml --world-size 8 [--start-env-steps 0]

``--start-env-steps`` is the loaded checkpoint's ``env_steps`` (0 for the BC best,
the checkpoint's step for a resume). Prints one JSON object; exits nonzero on a
violation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from owl.game import create_env
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.train import FullConfig

REPO = Path(__file__).resolve().parents[3]


def _run_ppo() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_ppo", REPO / "scripts" / "run_ppo.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/run_ppo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ppo"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--start-env-steps", type=int, default=0)
    args = parser.parse_args()
    cfg = FullConfig.from_file(args.config)
    env_config = cfg.env
    if not isinstance(env_config, KaggricultureEnvConfig):
        raise TypeError("the seed probe needs a Kaggriculture config")
    base = _run_ppo()._kaggriculture_rollout_base_seed(
        env_config.seed, start_env_steps=args.start_env_steps
    )
    ranks: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    violations: list[str] = []
    for rank in range(args.world_size):
        env = create_env(
            env_config,
            n_envs=env_config.n_envs,
            base_seed=base,
            rank=rank,
            world_size=args.world_size,
            pin_memory=False,
            transfer_device=torch.device("cpu"),
        )
        assert isinstance(env, KaggricultureVectorizedEnv)
        next_seed, seeds = env.seed_state()
        del env
        for seed in seeds:
            if (seed - base - rank) % args.world_size != 0:
                violations.append(f"rank {rank} seed {seed} outside its residue")
            if seed in seen:
                violations.append(f"seed {seed} on ranks {seen[seed]} and {rank}")
            seen[seed] = rank
        if next_seed != max(seeds) + args.world_size:
            violations.append(f"rank {rank} next seed {next_seed} is not contiguous")
        ranks.append(
            {
                "rank": rank,
                "n_envs": len(seeds),
                "first": min(seeds),
                "last": max(seeds),
                "next": next_seed,
            }
        )
    report = {
        "config": str(args.config),
        "world_size": args.world_size,
        "env_seed": env_config.seed,
        "start_env_steps": args.start_env_steps,
        "base_seed": base,
        "distinct_seeds": len(seen),
        "ranks": ranks,
        "violations": violations,
        "ok": not violations and len(seen) == args.world_size * env_config.n_envs,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
