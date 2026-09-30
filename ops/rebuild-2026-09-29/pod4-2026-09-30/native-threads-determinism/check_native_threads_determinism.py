"""native_threads determinism check: same seed, different rayon pool sizes.

Question: does ``native_threads`` change any native Kaggriculture output? If the
outputs are byte-identical, the 4-rank native_threads sweep is a pure
throughput choice with no semantic consequence.

Method (CPU only, tiny): build the env through ``owl.game.create_env`` (the
Task 1.5 adapter factory) once per thread count with the same seed and n_envs,
drive every env with identical legal actions sampled by a fixed-seed tiny
grammar model from the reference (native_threads=1) env's observation, and
compare every caller-owned output buffer byte for byte after every call, plus
the step metrics dict, terminal_metrics(i), state_snapshot(i) and seed_state().
A short episodeSteps makes games finish and auto-reset inside the window, and
one mid-run truncate_envs call covers the time-limit reset path.

Run from a checkout with owl.rs built, e.g.
  .venv/bin/python <this file> --steps 200 --threads 1 2 4 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from owl import rs
from owl.game import create_env
from owl.kaggriculture.config import KaggricultureEnvConfig
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.types import KaggricultureActions
from owl.model.kaggriculture import (
    KaggricultureTransformer,
    KaggricultureTransformerConfig,
)


def env_config(*, n_envs: int, threads: int, episode_steps: int) -> KaggricultureEnvConfig:
    return KaggricultureEnvConfig.model_validate(
        {
            "n_envs": n_envs,
            "config": {"episodeSteps": episode_steps},
            "reward_mode": "win_loss",
            # The 4-rank recipe's reward coefficients (configs/kaggriculture_4rank.yaml).
            "reward_shaping": {
                "econ_shaping": 0.2,
                "econ_starvation_weight": 4.0,
                "econ_drought_weight": 1.0,
                "econ_cap": 0.25,
                "econ_ineffective_weight": 0.0,
                "econ_ineffective_cap": 0.1,
            },
            "pin_memory": False,
            "native_threads": threads,
        }
    )


def build(n_envs: int, threads: int, seed: int, episode_steps: int) -> KaggricultureVectorizedEnv:
    env = create_env(
        env_config(n_envs=n_envs, threads=threads, episode_steps=episode_steps),
        n_envs=n_envs,
        base_seed=seed,
        rank=1,
        world_size=4,
        pin_memory=False,
        transfer_device=torch.device("cpu"),
    )
    assert isinstance(env, KaggricultureVectorizedEnv)
    return env


def buffers(env: KaggricultureVectorizedEnv) -> dict[str, torch.Tensor]:
    obs = env.observations
    out = {
        name: getattr(obs, name)
        for name in type(obs).model_fields
        if name != "action_mask"
    }
    out["can_act"] = obs.action_mask.can_act
    out["rewards"] = env.rewards
    out["dones"] = env.dones
    out["transition_banks_before"] = env.transition_banks_before
    out["transition_banks_after"] = env.transition_banks_after
    out["transition_econ_before"] = env.transition_econ_before
    out["transition_econ_after"] = env.transition_econ_after
    return out


def canon(value: Any) -> Any:
    """Exact, order-preserving canonical form (floats by their bit pattern)."""
    if isinstance(value, dict):
        return [(k, canon(v)) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, float):
        return ("f", value.hex())
    if hasattr(value, "tobytes"):
        return ("a", str(value.dtype), tuple(value.shape), value.tobytes())
    return value


def first_diff(
    ref: KaggricultureVectorizedEnv,
    other: KaggricultureVectorizedEnv,
    *,
    seed_state: bool = True,
) -> str | None:
    a, b = buffers(ref), buffers(other)
    for name in a:
        ta, tb = a[name], b[name]
        if ta.numpy().tobytes() != tb.numpy().tobytes():
            mism = (ta.view(torch.uint8) if ta.dtype == torch.bool else ta) != (
                tb.view(torch.uint8) if tb.dtype == torch.bool else tb
            )
            idx = mism.nonzero()[0].tolist() if mism.any() else "bitwise-only (NaN/-0)"
            return f"buffer {name} first differing index {idx}"
    for i in range(ref.n_envs):
        if canon(ref.terminal_metrics(i)) != canon(other.terminal_metrics(i)):
            return f"terminal_metrics({i})"
        if ref.state_snapshot(i) != other.state_snapshot(i):
            return f"state_snapshot({i})"
    if seed_state and ref.seed_state() != other.seed_state():
        return "seed_state()"
    return None


def digest(env: KaggricultureVectorizedEnv) -> str:
    h = hashlib.sha256()
    for name, t in buffers(env).items():
        h.update(name.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--episode-steps", type=int, default=60)
    p.add_argument("--truncate-at", type=int, default=97)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert args.threads[0] == 1, "the first thread count is the serial reference"

    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    spec = env_config(n_envs=args.n_envs, threads=1, episode_steps=args.episode_steps)
    model = KaggricultureTransformer(
        KaggricultureTransformerConfig(
            embed_dim=16, depth=1, n_heads=1, mlp_ratio=1, n_scratch_tokens=0
        ),
        obs_spec=spec.obs_spec,
        action_spec=spec.action_spec,
    ).eval()

    envs = [build(args.n_envs, t, args.seed, args.episode_steps) for t in args.threads]
    ref = envs[0]
    # Candidates: each other thread count, plus a negative control (serial, seed + 1)
    # driven by the same actions, which the comparator must report as different.
    candidates: list[tuple[str, KaggricultureVectorizedEnv]] = [
        (f"threads={t}", e) for t, e in zip(args.threads[1:], envs[1:])
    ]
    candidates.append(
        ("control:seed+1", build(args.n_envs, 1, args.seed + 1, args.episode_steps))
    )
    result: dict[str, Any] = {
        "question": "does native_threads change any native Kaggriculture output?",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "owl_rs": rs.__file__,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "rank_world": [1, 4],
        "episode_steps": args.episode_steps,
        "steps": args.steps,
        "truncate_at": args.truncate_at,
        "threads": args.threads,
        "compared": sorted(buffers(ref)) + [
            "step metrics dict",
            "terminal_metrics(i)",
            "state_snapshot(i)",
            "seed_state()",
        ],
    }
    first: dict[str, str] = {}

    def check(label: str) -> None:
        for name, env in candidates:
            if name in first:
                continue
            where = first_diff(ref, env)
            if where is not None:
                first[name] = f"{label}: {where}"
        # The control's seed_state differs from construction; also record when
        # its buffers/snapshots first differ, so buffer sensitivity is shown.
        if "control:buffers" not in first:
            where = first_diff(ref, candidates[-1][1], seed_state=False)
            if where is not None:
                first["control:buffers"] = f"{label}: {where}"

    check("construction observe")
    ref.reset()
    for _, env in candidates:
        env.reset()
    check("reset")

    dones_total = 0
    terminals = 0
    nonpass_rows = 0
    for step in range(args.steps):
        with torch.no_grad():
            sampled = model(ref.observations).actions
        tokens = sampled.tokens.to(torch.int64).contiguous()
        lengths = sampled.lengths.to(torch.int64).contiguous()
        nonpass_rows += int((lengths > 2).sum())
        m_ref = ref.step(KaggricultureActions(tokens.clone(), lengths.clone()))[3]
        for name, env in candidates:
            m = env.step(KaggricultureActions(tokens.clone(), lengths.clone()))[3]
            if name not in first and canon(m) != canon(m_ref):
                first[name] = f"step {step}: step metrics dict"
        dones_total += int(ref.dones.sum())
        terminals += sum(ref.terminal_metrics(i) is not None for i in range(ref.n_envs))
        check(f"step {step}")
        if step == args.truncate_at:
            mask = torch.zeros(ref.n_envs, dtype=torch.bool)
            mask[::3] = True
            ref.truncate_envs(mask.clone())
            for _, env in candidates:
                env.truncate_envs(mask.clone())
            check(f"truncate_envs after step {step}")

    control_seen = "control:seed+1" in first and "control:buffers" in first
    thread_names = [name for name, _ in candidates[:-1]]
    result.update(
        {
            "done_flags_seen": dones_total,
            "terminal_records_seen": terminals,
            "seat_rows_longer_than_pass_stop": nonpass_rows,
            "final_buffer_sha256": {
                f"threads={t}": digest(e) for t, e in zip(args.threads, envs)
            },
            "identical": {name: name not in first for name in thread_names},
            "first_difference": {name: first.get(name) for name, _ in candidates},
            "negative_control_detected": control_seen,
            "negative_control_buffers_differ": first.get("control:buffers"),
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    ok = control_seen and all(result["identical"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
