"""Model-only SPS ceiling probe (COMPONENT measurement, not end-to-end SPS).

Times the four model workloads of one PPO update for KaggricultureTransformer
(preset configs/model/kaggriculture.yaml) on one GPU with synthetic,
contract-valid observations (tests/kaggriculture/conftest.py::make_obs) and the
synthetic ``expected_grammar_tables``. Engine stepping, host<->device copies,
GAE, logging and DDP all-reduce are NOT included.

Run from the v3 checkout root on the pod with CUDA_VISIBLE_DEVICES=0:
    .venv/bin/python <this> --density {sparse,mid,dense} --out <dir>

Workloads (rows = seat rows; lead shape [envs, 2]):
  A rollout sampling  : model.forward (sample + values), 256 rows, no_grad
  B PPO train step    : evaluate_actions on 1,024 rows with saved actions,
                        clipped-ratio + value + entropy loss, backward,
                        clip_grad_norm_(10), Muon step (create_optimizer)
  C teacher precompute: evaluate_actions under no_grad on 16,384 rows (proxy:
                        this model has no compute_teacher_distillation_targets)
  D critic bootstrap  : compute_value on 256 rows, no_grad
Run-statement: ops/rebuild-2026-09-29/run-statements/model-sps-ceiling.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

DENSITIES: dict[str, dict[str, int]] = {
    # own actors, rival actors, shops per env (same for all 128 envs)
    "sparse": {"own": 1, "rival": 1, "shops": 1},
    "mid": {"own": 40, "rival": 40, "shops": 4},
    "dense": {"own": 241, "rival": 241, "shops": 8},
}
ENVS = 128  # rollout envs per rank -> 256 seat rows
HORIZON = 64
SPM = 8  # segments per minibatch -> 8 x 64 x 2 = 1,024 rows
WARMUP = 5
TIMED = 20
GIB = 1024**3


def _to(obj: BaseModel, device: torch.device) -> BaseModel:
    fields: dict[str, Any] = {}
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        if isinstance(value, torch.Tensor):
            fields[name] = value.to(device)
        elif isinstance(value, BaseModel):
            fields[name] = _to(value, device)
        else:
            fields[name] = value
    return type(obj)(**fields)


def _tile(obj: BaseModel, k: int) -> BaseModel:
    """Repeat every tensor k times along the leading (env) dim."""
    fields: dict[str, Any] = {}
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        if isinstance(value, torch.Tensor):
            fields[name] = value.repeat(k, *([1] * (value.dim() - 1)))
        elif isinstance(value, BaseModel):
            fields[name] = _tile(value, k)
        else:
            fields[name] = value
    return type(obj)(**fields)


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def _time(
    fn: Callable[[], Any], counters: dict[str, Any], tag: str
) -> dict[str, Any]:
    """First call (includes compile), WARMUP, then TIMED iterations (CUDA events)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    counters["pack_calls"] = []
    counters["flash_flags"] = []
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - t0
    first_pack = list(counters["pack_calls"])
    first_flash = list(counters["flash_flags"])
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    counters["pack_calls"] = []
    counters["flash_flags"] = []
    ev_ms: list[float] = []
    host_ms: list[float] = []
    for _ in range(TIMED):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        h0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        host_ms.append((time.perf_counter() - h0) * 1e3)
        ev_ms.append(start.elapsed_time(end))
    pack_calls = counters["pack_calls"]
    result = {
        "workload": tag,
        "first_call_s_incl_compile": first_s,
        "first_call_pack_calls": first_pack,
        "first_call_use_flash_attn": first_flash,
        "warmup_iters": WARMUP,
        "timed_iters": TIMED,
        "cuda_event_ms": ev_ms,
        "median_ms": statistics.median(ev_ms),
        "p90_ms": _pct(ev_ms, 0.9),
        "min_ms": min(ev_ms),
        "max_ms": max(ev_ms),
        "host_wall_median_ms": statistics.median(host_ms),
        "pack_sequence_calls_per_iter": len(pack_calls) / TIMED,
        "packed_tokens_per_chunk_first_iter": [
            n for n, _ in pack_calls[: max(1, len(pack_calls) // TIMED)]
        ],
        "use_flash_attn_all_true": bool(counters["flash_flags"])
        and all(counters["flash_flags"]),
        "use_flash_attn_calls_per_iter": len(counters["flash_flags"]) / TIMED,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }
    print(
        f"[{tag}] first={first_s:.2f}s median={result['median_ms']:.2f}ms "
        f"p90={result['p90_ms']:.2f}ms peak_alloc="
        f"{result['max_memory_allocated_gib']:.2f}GiB "
        f"chunks/iter={result['pack_sequence_calls_per_iter']} "
        f"flash={result['use_flash_attn_all_true']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", choices=sorted(DENSITIES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import flash_attn
    import triton
    from owl.kaggriculture import types as kt
    from owl.model import kaggriculture as km
    from owl.model.attn import flash_attn_available
    from owl.train.optimizer import MuonConfig, create_optimizer
    from owl.train.utils import configure_torch

    from tests.kaggriculture.conftest import make_obs

    assert torch.cuda.is_available()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    configure_torch()  # TF32 on, cudnn.benchmark, as run_ppo
    assert flash_attn_available(), "flash-attn not importable"
    config = km.KaggricultureTransformerConfig.from_file(
        ROOT / "configs/model/kaggriculture.yaml"
    )
    assert config.force_flash_attn is True

    counters: dict[str, Any] = {"pack_calls": [], "flash_flags": []}
    real_pack = km.pack_sequence
    real_use_flash = km.use_flash_attn

    def counting_pack(x: torch.Tensor, mask: torch.Tensor, **kw: Any) -> Any:
        px, packed = real_pack(x, mask, **kw)
        counters["pack_calls"].append((int(px.shape[0]), int(packed.max_seqlen)))
        return px, packed

    def counting_use_flash(x: torch.Tensor) -> bool:
        flag = real_use_flash(x)
        counters["flash_flags"].append(bool(flag))
        return flag

    km.pack_sequence = counting_pack
    km.use_flash_attn = counting_use_flash

    torch.manual_seed(0)
    model = km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs")
    model.train()

    d = DENSITIES[args.density]
    t0 = time.time()
    obs_cpu = make_obs(
        ENVS, own_actors=d["own"], rival_actors=d["rival"], shops=d["shops"]
    )
    make_obs_s = time.time() - t0
    obs = _to(obs_cpu, device)
    assert isinstance(obs, kt.KaggricultureObsBatch)
    seq = obs.actor_mask.sum(-1) + obs.shop_mask.sum(-1)
    fixed_tokens = km.sequence_length(config) - kt.ACTOR_SLOTS - kt.SHOP_SLOTS
    tokens_per_row = (seq + fixed_tokens).flatten()

    def autocast() -> Any:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    info: dict[str, Any] = {
        "density": args.density,
        "density_spec": d,
        "envs": ENVS,
        "rows_A_D": ENVS * kt.PLAYERS,
        "rows_B": SPM * HORIZON * kt.PLAYERS,
        "rows_C": ENVS * HORIZON * kt.PLAYERS,
        "tokens_per_row_min": int(tokens_per_row.min()),
        "tokens_per_row_max": int(tokens_per_row.max()),
        "padded_seq_len": km.sequence_length(config),
        "make_obs_s": make_obs_s,
        "n_params": n_params,
        "config": config.model_dump(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "flash_attn": flash_attn.__version__,
        "triton": triton.__version__,
        "device_name": torch.cuda.get_device_name(device),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "trunk_compile": "torch.compile(mode=max-autotune-no-cudagraphs, dynamic=True)",
        "heads": "eager",
        "autocast": "bfloat16, fp32 params",
        "grammar_tables": "expected_grammar_tables (synthetic)",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory / GIB,
    }
    print(json.dumps({k: v for k, v in info.items() if k != "config"}), flush=True)
    results: dict[str, Any] = {}

    # --- A: rollout sampling ----------------------------------------------------
    def run_a() -> Any:
        with torch.no_grad(), autocast():
            return model(obs)

    results["A"] = _time(run_a, counters, "A_rollout_forward_256")
    with torch.no_grad(), autocast():
        sampled = model(obs)
    actions_256 = sampled.actions
    old_logp_256 = sampled.log_probs.event.float().sum(dim=(-1, -2))

    # --- D: critic bootstrap ------------------------------------------------------
    def run_d() -> Any:
        with torch.no_grad(), autocast():
            return model.compute_value(obs)

    results["D"] = _time(run_d, counters, "D_compute_value_256")

    # --- B: PPO train step ----------------------------------------------------------
    k_b = SPM * HORIZON // ENVS  # 4 -> 512 envs x 2 seats = 1,024 rows
    obs_b = _tile(obs, k_b)
    assert isinstance(obs_b, kt.KaggricultureObsBatch)
    actions_b = kt.KaggricultureActions(
        tokens=actions_256.tokens.repeat(k_b, 1, 1, 1),
        lengths=actions_256.lengths.repeat(k_b, 1),
    )
    old_logp_b = old_logp_256.repeat(k_b, 1)
    gen = torch.Generator(device=device).manual_seed(3)
    adv_b = torch.randn(old_logp_b.shape, generator=gen, device=device)
    ret_b = torch.rand(old_logp_b.shape, generator=gen, device=device) * 2 - 1
    optimizer = create_optimizer(model, MuonConfig())

    def run_b() -> Any:
        with autocast():
            ev = model.evaluate_actions(obs_b, actions_b)
        new_logp = ev.log_probs.event.float().sum(dim=(-1, -2))
        ratio = (new_logp - old_logp_b).exp()
        pg = -torch.min(ratio * adv_b, ratio.clamp(0.8, 1.2) * adv_b).mean()
        v_loss = 0.5 * (ev.values.float() - ret_b).pow(2).mean()
        ent = ev.entropies.event.float().sum(dim=(-1, -2)).mean()
        loss = pg + 0.5 * v_loss - 0.01 * ent
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        return loss

    results["B"] = _time(run_b, counters, "B_ppo_train_step_1024")
    del obs_b, actions_b

    # --- C: teacher precompute proxy --------------------------------------------------
    k_c = HORIZON  # 64 -> 8,192 envs x 2 seats = 16,384 rows
    obs_c = _tile(obs, k_c)
    assert isinstance(obs_c, kt.KaggricultureObsBatch)
    actions_c = kt.KaggricultureActions(
        tokens=actions_256.tokens.repeat(k_c, 1, 1, 1),
        lengths=actions_256.lengths.repeat(k_c, 1),
    )
    info["C_head_rows_per_chunk"] = km.head_rows_per_chunk(config)
    info["C_head_chunks"] = -(-ENVS * HORIZON * kt.PLAYERS // km.head_rows_per_chunk(config))

    def run_c() -> Any:
        with torch.no_grad(), autocast():
            return model.evaluate_actions(obs_c, actions_c)

    results["C"] = _time(run_c, counters, "C_teacher_forward_16384")

    # --- derived ceiling ------------------------------------------------------------
    t = {k: results[k]["median_ms"] / 1e3 for k in "ABCD"}
    t90 = {k: results[k]["p90_ms"] / 1e3 for k in "ABCD"}
    steps = ENVS * HORIZON
    wall = HORIZON * t["A"] + t["C"] + (HORIZON * ENVS // (SPM * HORIZON)) * t["B"] + t["D"]
    wall90 = HORIZON * t90["A"] + t90["C"] + 16 * t90["B"] + t90["D"]
    derived = {
        "formula": "update_wall = 64*t_A + t_C + 16*t_B + t_D; ceiling_sps_rank = 8192/update_wall",
        "env_steps_per_update_per_rank": steps,
        "minibatches_per_update": HORIZON * ENVS // (SPM * HORIZON),
        "update_wall_s_median": wall,
        "update_wall_s_p90": wall90,
        "share": {
            "A_64x": HORIZON * t["A"] / wall,
            "B_16x": 16 * t["B"] / wall,
            "C": t["C"] / wall,
            "D": t["D"] / wall,
        },
        "ceiling_sps_per_rank_median": steps / wall,
        "ceiling_sps_per_rank_p90": steps / wall90,
        "ceiling_sps_2_ranks_approx_median": 2 * steps / wall,
        "model_s_per_env_step": wall / steps,
    }
    out = {"info": info, "results": results, "derived": derived}
    path = args.out / f"results_{args.density}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(derived, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
