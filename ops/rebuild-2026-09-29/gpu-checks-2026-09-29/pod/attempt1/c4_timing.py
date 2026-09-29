"""Check 4: per-rank component timings for 2/4/8-rank Isaiah splits (COMPONENT only).

bench_model_sps.py-style CUDA-event timing (first call incl. compile, 5 warmup,
20 timed; medians), one density per process, ATEN-only via the wrapper.
Isaiah split (plan I2): the global config is fixed at 256 envs x horizon 64 =
16,384 env steps per update; per rank envs = 256 / ranks and spm = 16 / ranks,
so every rank runs 16 minibatches of spm x 64 x 2 rows.
  ranks 8: A/D 64 rows,  B 256 rows,   C 4,096 rows
  ranks 4: A/D 128 rows, B 512 rows,   C 8,192 rows
  ranks 2: A/D 256 rows, B 1,024 rows, C 16,384 rows (the earlier bench shape)
Workloads (8fde43c; the Phase 4 teacher path replaces the earlier C proxy):
  A rollout sampling  : model(obs), no_grad
  D critic bootstrap  : compute_value(obs), no_grad
  B PPO train step    : evaluate_actions_with_cached_teacher on the tiled
                        minibatch (targets precomputed, untimed), clipped-ratio
                        + 0.5 MSE value - 0.01 entropy + 0.005 teacher KL +
                        0.005 teacher value CE, backward, clip_grad_norm_(10),
                        Muon step (create_optimizer(MuonConfig()))
  C teacher precompute: compute_teacher_distillation_targets on the rollout
                        (obs tiled x64 = one rank's 64-step rollout)
update wall = 64 t_A + t_C + 16 t_B + t_D; per-rank SPS = envs x 64 / wall;
global SPS = ranks x envs x 64 / wall = 16,384 / wall. Engine stepping,
host<->device copies, GAE, logging and all-reduce are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import common as C

HORIZON = 64
GLOBAL_ENVS = 256
MINIBATCHES = 16
WARMUP = 2 if C.DRYRUN else 5
TIMED = 2 if C.DRYRUN else 20
SPLITS = (8, 4, 2)  # small first


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fh = open(args.out, "a")
    C.configure()
    dev = C.device()
    import owl.model.kaggriculture as km
    from owl.kaggriculture import types as kt
    from owl.train.optimizer import MuonConfig, create_optimizer

    counters: dict[str, list[Any]] = {"pack": [], "flash": []}
    real_pack, real_use_flash = km.pack_sequence, km.use_flash_attn

    def counting_pack(x: torch.Tensor, mask: torch.Tensor, **kw: Any) -> Any:
        px, packed = real_pack(x, mask, **kw)
        counters["pack"].append(int(px.shape[0]))
        return px, packed

    def counting_use_flash(x: torch.Tensor) -> bool:
        flag = real_use_flash(x)
        counters["flash"].append(bool(flag))
        return flag

    km.pack_sequence = counting_pack
    km.use_flash_attn = counting_use_flash
    C.emit(fh, C.process_record(args.case))
    model = C.build_model(dev).train()
    C.compile_trunk(model)
    optimizer = create_optimizer(model, MuonConfig())
    width = km.trunk_gemm_width(model.config)
    fixed_tokens = km.sequence_length(model.config) - kt.ACTOR_SLOTS - kt.SHOP_SLOTS
    t0 = time.time()
    envs_max = 4 if C.DRYRUN else GLOBAL_ENVS // min(SPLITS)
    obs_all = C.to_dev(C.make_obs_batch(envs_max, args.density), dev)
    C.emit(fh, {"event": "obs", "case": args.case, "make_obs_s": time.time() - t0,
                "envs": envs_max})

    def expected_chunks(obs: Any) -> int:
        rows = obs.still_playing.numel()
        tokens = km.sequence_length(model.config)
        if rows * tokens * width < km._GEMM_ELEMENT_LIMIT:
            return 1
        per_row = (obs.actor_mask.reshape(rows, -1).sum(-1)
                   + obs.shop_mask.reshape(rows, -1).sum(-1) + fixed_tokens)
        return len(km.packed_row_chunks([int(n) for n in per_row.tolist()],
                                        width=width))

    def timed(fn: Callable[[], Any], tag: str, extra: dict[str, Any]) -> dict[str, Any]:
        C.sync(dev)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        counters["pack"].clear(); counters["flash"].clear()
        h0 = time.perf_counter()
        last = fn()
        C.sync(dev)
        first_s = time.perf_counter() - h0
        first_pack = list(counters["pack"])
        for _ in range(WARMUP):
            last = fn()
        C.sync(dev)
        counters["pack"].clear(); counters["flash"].clear()
        ev_ms: list[float] = []
        host_ms: list[float] = []
        for _ in range(TIMED):
            if dev.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                h = time.perf_counter()
                start.record()
                last = fn()
                end.record()
                end.synchronize()
                host_ms.append((time.perf_counter() - h) * 1e3)
                ev_ms.append(start.elapsed_time(end))
            else:
                h = time.perf_counter()
                last = fn()
                host_ms.append((time.perf_counter() - h) * 1e3)
                ev_ms.append(host_ms[-1])
        rec = {
            "event": "timing", "case": args.case, "workload": tag, **extra,
            "first_call_s_incl_compile": first_s,
            "first_call_pack_calls": first_pack,
            "warmup_iters": WARMUP, "timed_iters": TIMED,
            "cuda_event_ms": ev_ms, "median_ms": statistics.median(ev_ms),
            "p90_ms": pct(ev_ms, 0.9), "min_ms": min(ev_ms), "max_ms": max(ev_ms),
            "host_wall_median_ms": statistics.median(host_ms),
            "pack_calls_per_iter": len(counters["pack"]) / TIMED,
            "packed_tokens_per_chunk_first_iter": counters["pack"][
                : max(1, len(counters["pack"]) // TIMED)],
            "use_flash_attn_all_true": bool(counters["flash"]) and all(counters["flash"]),
            **C.mem(dev),
        }
        if isinstance(last, torch.Tensor) and last.numel() == 1:
            rec["last_scalar"] = float(last)
            rec["last_scalar_finite"] = bool(torch.isfinite(last))
        C.emit(fh, rec)
        return rec

    derived: dict[str, Any] = {}
    for ranks in SPLITS:
        envs = 2 if C.DRYRUN else GLOBAL_ENVS // ranks
        spm = MINIBATCHES // ranks  # segments per minibatch; 16 minibatches/rank
        obs = C.first_envs(obs_all, envs)
        k_b = 2 if C.DRYRUN else spm * HORIZON // envs  # = 4 for every split
        k_c = 2 if C.DRYRUN else HORIZON
        tag = f"r{ranks}"
        res: dict[str, Any] = {}

        def run_a() -> Any:
            with torch.no_grad(), C.amp(dev):
                return model(obs)

        res["A"] = timed(run_a, f"{tag}_A_rollout_{2 * envs}",
                         {"ranks": ranks, "rows": 2 * envs,
                          "expected_chunks": expected_chunks(obs)})
        torch.manual_seed(1)
        with torch.no_grad(), C.amp(dev):
            sampled = model(obs)
        actions = sampled.actions
        old_logp = sampled.log_probs.event.float().sum(dim=(-1, -2))

        def run_d() -> Any:
            with torch.no_grad(), C.amp(dev):
                return model.compute_value(obs)

        res["D"] = timed(run_d, f"{tag}_D_value_{2 * envs}",
                         {"ranks": ranks, "rows": 2 * envs,
                          "expected_chunks": expected_chunks(obs)})

        obs_b = C.tile(obs, k_b)
        actions_b = C.tile_actions(actions, k_b)
        old_b = old_logp.repeat(k_b, 1)
        with C.amp(dev):
            targets_b = model.compute_teacher_distillation_targets(obs_b, actions_b)
        gen = torch.Generator(device=dev).manual_seed(3)
        adv = torch.randn(old_b.shape, generator=gen, device=dev)
        ret = torch.rand(old_b.shape, generator=gen, device=dev) * 2 - 1

        def run_b() -> Any:
            with C.amp(dev):
                ev = model.evaluate_actions_with_cached_teacher(obs_b, actions_b,
                                                                targets_b)
            new_logp = ev.student.log_probs.event.float().sum(dim=(-1, -2))
            ratio = (new_logp - old_b).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            v_loss = 0.5 * (ev.student.values.float() - ret).pow(2).mean()
            ent = ev.student.entropies.event.float().sum(dim=(-1, -2)).mean()
            kl = ev.action_kl.event.float().sum(dim=(-1, -2)).mean()
            vce = model.teacher_value_cross_entropy(
                ev.student_winner_log_probabilities, ev.teacher_winner_probabilities,
                value_mask=obs_b.still_playing).mean()
            loss = pg + 0.5 * v_loss - 0.01 * ent + 0.005 * kl + 0.005 * vce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            return loss.detach()

        res["B"] = timed(run_b, f"{tag}_B_train_{obs_b.still_playing.numel()}",
                         {"ranks": ranks, "rows": obs_b.still_playing.numel(),
                          "spm": spm, "expected_chunks": expected_chunks(obs_b)})
        del obs_b, actions_b, targets_b

        obs_c = C.tile(obs, k_c)
        actions_c = C.tile_actions(actions, k_c)

        def run_c() -> Any:
            with C.amp(dev):
                return model.compute_teacher_distillation_targets(obs_c, actions_c)

        res["C"] = timed(run_c, f"{tag}_C_teacher_{obs_c.still_playing.numel()}",
                         {"ranks": ranks, "rows": obs_c.still_playing.numel(),
                          "expected_chunks": expected_chunks(obs_c),
                          "head_chunks_calc": -(-obs_c.still_playing.numel()
                                                // km.head_rows_per_chunk(model.config))})
        del obs_c, actions_c
        t = {k: res[k]["median_ms"] / 1e3 for k in "ABCD"}
        t90 = {k: res[k]["p90_ms"] / 1e3 for k in "ABCD"}
        wall = HORIZON * t["A"] + t["C"] + MINIBATCHES * t["B"] + t["D"]
        wall90 = HORIZON * t90["A"] + t90["C"] + MINIBATCHES * t90["B"] + t90["D"]
        steps_rank = envs * HORIZON
        derived[str(ranks)] = {
            "ranks": ranks, "envs_per_rank": envs, "spm": spm,
            "rows": {k: res[k]["rows"] for k in "ABCD"},
            "median_ms": {k: res[k]["median_ms"] for k in "ABCD"},
            "p90_ms": {k: res[k]["p90_ms"] for k in "ABCD"},
            "update_wall_s_median": wall, "update_wall_s_p90_weighted": wall90,
            "share": {"A_64x": HORIZON * t["A"] / wall, "B_16x": MINIBATCHES * t["B"] / wall,
                      "C": t["C"] / wall, "D": t["D"] / wall},
            "per_rank_sps": steps_rank / wall,
            "global_sps": ranks * steps_rank / wall,
            "label": "COMPONENT ONLY: engine, host copies, GAE, logging, all-reduce excluded",
        }
    out = {"event": "derived", "case": args.case, "density": args.density,
           "formula": "update_wall = 64*t_A + t_C + 16*t_B + t_D; "
                      "global_sps = ranks*envs*64/update_wall",
           "splits": derived}
    C.emit(fh, out)
    Path(args.out).with_suffix(".derived.json").write_text(json.dumps(out, indent=1))
    C.emit(fh, {"event": "done", "case": args.case})
    fh.close()


if __name__ == "__main__":
    main()
