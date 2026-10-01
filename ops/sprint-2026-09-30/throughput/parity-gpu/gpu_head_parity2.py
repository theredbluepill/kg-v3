"""Follow-up (diagnostic only): fp64-heads reference and rollout-vs-replay attribution.

Run 1 found compiled-vs-eager bf16 head drift (per-player logp median 0.006,
grad global rel err 1-9%) and a LARGER stored-rollout-vs-replay logp gap
(median 0.021, max 0.67) that is the same for compiled and eager replay.
This run asks: (A) is compiled-vs-eager drift the same size as eager-bf16 vs
an fp64 head reference (i.e. ordinary bf16 rounding)? (B) is the rollout vs
replay gap a batch-shape effect (40-row rollout batch vs 1440-row replay) in
the heads or the trunk? No teacher (plain evaluate_actions path; run 1 showed
the cached-teacher path gives identical student logp).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from pathlib import Path

import torch
from owl.game import create_env
from owl.model.factory import create_model
from owl.train.config import FullConfig
from owl.train.optimizer import create_lr_scheduler, create_optimizer
from owl.train.ppo import (
    PPOTrainer,
    _actions_index,
    _flatten_actions_time,
    _flatten_obs_time,
    _map_observation,
    _map_action_bundle,
    _model_evaluate_actions,
    _obs_index,
    _output_logp_for_clip_mode,
    _policy_mask,
)
from owl.train.utils import autocast_context, configure_model_compile, configure_torch

OUT = Path("/root/parity")
CKPT = Path("/root/ckpt/checkpoint_00_110_015_744.pt")
CFG = Path("/root/ckpt/config.yaml")
N_ENVS, HORIZON, SEED = 20, 720, 4243


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dist(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().double().flatten()
    a = x.abs()
    q = torch.quantile(a, torch.tensor([0.5, 0.99, 0.999], dtype=torch.float64, device=a.device))
    return {"n": int(a.numel()), "max_abs": float(a.max()), "p999_abs": float(q[2]),
            "p99_abs": float(q[1]), "median_abs": float(q[0]), "mean_abs": float(a.mean()),
            "mean_signed": float(x.mean()), "frac_gt_1e-2": float((a > 1e-2).double().mean()),
            "nonfinite": int((~torch.isfinite(x)).sum())}


def main() -> None:
    torch.set_num_threads(16)
    torch.manual_seed(SEED)
    configure_torch()
    device = torch.device("cuda")
    cfg = FullConfig.from_file(CFG, {"env.seed": SEED})
    rl = cfg.rl
    env = create_env(cfg.env, n_envs=N_ENVS, base_seed=SEED, rank=0, world_size=1,
                     pin_memory=True, transfer_device=device)
    model = create_model(cfg.model, obs_spec=cfg.env.obs_spec,
                         action_spec=cfg.env.action_spec).to(device)
    configure_model_compile(model, rl)
    optimizer = create_optimizer(model, cfg.optimizer)
    trainer = PPOTrainer(config=rl, env=env, model=model, optimizer=optimizer,
                         lr_scheduler=create_lr_scheduler(optimizer, cfg.optimizer.lr_schedule),
                         device=device)
    trainer.load_model_weights(CKPT)
    compiled_core = model._compiled_actor_core
    compiled_trunk = model._compiled_transformer_trunk
    assert compiled_core is not None and compiled_trunk is not None
    log("model ready")
    with torch.no_grad():
        last_values = trainer._collect_rollout()
    log("rollout done")
    segments = trainer.rollout.segment_major()
    value_mask = segments.obs.still_playing
    policy_mask = _policy_mask(segments.obs)
    advantages, returns = trainer._compute_gae(
        rewards=segments.rewards, values=segments.values, dones=segments.dones,
        last_values=last_values, gamma=rl.gamma, gae_lambda=rl.gae_lambda,
        truncated=segments.truncated, bootstrap_values=segments.bootstrap_values)
    winner_targets = trainer._compute_winner_targets(segments, last_values)
    stored = segments.logp.float()

    actor64 = copy.deepcopy(model.actor).double()

    def core64(*args):
        conv = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                for a in args]
        with torch.autocast("cuda", enabled=False):
            r = actor64.policy_core(*conv)
        return dataclasses.replace(r, log_probs=r.log_probs.float(),
                                   entropies=r.entropies.float())

    def set_arm(heads: str, trunk: str = "compiled") -> None:
        model._compiled_actor_core = {"compiled": compiled_core, "eager": None,
                                      "fp64": core64}[heads]
        model._compiled_transformer_trunk = compiled_trunk if trunk == "compiled" else None

    res: dict = {"config": {"n_envs": N_ENVS, "horizon": HORIZON, "seed": SEED}}
    model.train()
    # ---- (A1) full-segment replay (update shape: 1440 rows) ----
    arms = [("compiled", "compiled"), ("eager", "compiled"), ("fp64", "compiled"),
            ("eager", "eager"), ("fp64", "eager")]
    seg_logp: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for heads, trunk in arms:
            set_arm(heads, trunk)
            outs = []
            for k in range(N_ENVS):
                idx = torch.tensor([k], device=device)
                obs = _flatten_obs_time(_obs_index(segments.obs, idx))
                act = _flatten_actions_time(_actions_index(segments.actions, idx))
                with autocast_context(rl, device):
                    out = _model_evaluate_actions(model, obs, act, hidden_state=None,
                                                  dones=segments.dones[idx])
                outs.append(_output_logp_for_clip_mode(out, rl.ppo_clip_mode)
                            .view_as(segments.logp[idx]).float())
            seg_logp[f"{heads}/{trunk}"] = torch.cat(outs)
            log(f"segment replay {heads}/{trunk}")
    # ---- (B) rollout-shaped replay: all envs at one timestep (40 rows) ----
    step_logp: dict[str, torch.Tensor] = {}
    ts = list(range(0, HORIZON, 4))
    with torch.no_grad():
        for heads, trunk in arms:
            set_arm(heads, trunk)
            cols = []
            for t in ts:
                obs_t = _map_observation(segments.obs, lambda x, t=t: x[:, t])
                act_t = _map_action_bundle(segments.actions, lambda x, t=t: x[:, t])
                with autocast_context(rl, device):
                    out = _model_evaluate_actions(model, obs_t, act_t, hidden_state=None,
                                                  dones=None)
                cols.append(_output_logp_for_clip_mode(out, rl.ppo_clip_mode).float()
                            .view_as(stored[:, t]))
            step_logp[f"{heads}/{trunk}"] = torch.stack(cols, dim=1)
            log(f"step replay {heads}/{trunk}")
    m = policy_mask
    ms = policy_mask[:, ts]
    ss = stored[:, ts]
    ref = seg_logp["fp64/compiled"]
    res["segment_replay_1440rows"] = {
        "compiled_minus_eager(trunk compiled)": dist((seg_logp["compiled/compiled"] - seg_logp["eager/compiled"])[m]),
        "compiled_minus_fp64heads": dist((seg_logp["compiled/compiled"] - ref)[m]),
        "eager_minus_fp64heads": dist((seg_logp["eager/compiled"] - ref)[m]),
        "eager_trunk_eager_minus_fp64heads_trunk_eager": dist((seg_logp["eager/eager"] - seg_logp["fp64/eager"])[m]),
        "trunk_compiled_minus_trunk_eager(fp64 heads)": dist((ref - seg_logp["fp64/eager"])[m]),
        **{f"stored_minus_{k}": dist((stored - v)[m]) for k, v in seg_logp.items()},
    }
    res["step_replay_40rows"] = {
        **{f"stored_minus_{k}": dist((ss - v)[ms]) for k, v in step_logp.items()},
        **{f"step40_minus_segment1440_{k}": dist((v - seg_logp[k][:, ts])[ms]) for k, v in step_logp.items()},
        "compiled_minus_eager(trunk compiled)": dist((step_logp["compiled/compiled"] - step_logp["eager/compiled"])[ms]),
        "compiled_minus_fp64heads": dist((step_logp["compiled/compiled"] - step_logp["fp64/compiled"])[ms]),
        "eager_minus_fp64heads": dist((step_logp["eager/compiled"] - step_logp["fp64/compiled"])[ms]),
    }
    log("replay stats done")

    # ---- (A2) minibatch grads vs fp64-head reference ----
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    p64 = dict(actor64.named_parameters())

    def run_mb(heads: str, k: int) -> tuple[dict, dict]:
        set_arm(heads, "compiled")
        model.zero_grad(set_to_none=True)
        actor64.zero_grad(set_to_none=True)
        upd = trainer._update_minibatch(
            segments, advantages, returns, policy_mask, value_mask,
            torch.tensor([k], device=device), teacher_targets=None,
            winner_targets=winner_targets, value_clip_anchor=segments.values.clone(),
            loss_scale=1.0, step_optimizer=False)
        g = {}
        for n, p in params:
            if heads == "fp64" and n.startswith("actor."):
                q = p64[n[len("actor."):]]
                g[n] = None if q.grad is None else q.grad.detach().float().clone()
            else:
                g[n] = None if p.grad is None else p.grad.detach().float().clone()
        mt = upd.metrics
        return {f: float(getattr(mt, f)) for f in ("loss", "policy_loss", "logratio_mean",
                                                   "logratio_abs_max", "clipfrac")}, g

    def compare(ga: dict, gb: dict) -> dict:
        rel, worst = [], []
        keys = [n for n, _ in params if ga[n] is not None and gb[n] is not None]
        for n in keys:
            nb = gb[n].norm()
            r = float((ga[n] - gb[n]).norm() / nb) if nb > 0 else 0.0
            rel.append(r)
            worst.append((r, n))
        t = torch.tensor(rel, dtype=torch.float64)
        fa = torch.cat([ga[n].flatten() for n in keys])
        fb = torch.cat([gb[n].flatten() for n in keys])
        actor_keys = [n for n in keys if n.startswith("actor.")]
        fa_a = torch.cat([ga[n].flatten() for n in actor_keys])
        fb_a = torch.cat([gb[n].flatten() for n in actor_keys])
        worst.sort(reverse=True)
        return {"n_tensors": len(rel), "rel_err_max": float(t.max()),
                "rel_err_median": float(t.median()),
                "global_rel_err": float((fa - fb).norm() / fb.norm()),
                "actor_params_global_rel_err": float((fa_a - fb_a).norm() / fb_a.norm()),
                "trunk_and_rest_global_rel_err": float(
                    (torch.cat([ga[n].flatten() for n in keys if not n.startswith("actor.")])
                     - torch.cat([gb[n].flatten() for n in keys if not n.startswith("actor.")])).norm()
                    / torch.cat([gb[n].flatten() for n in keys if not n.startswith("actor.")]).norm()),
                "cosine": float(torch.nn.functional.cosine_similarity(fa, fb, dim=0)),
                "worst3": worst[:3],
                "nonfinite": int(sum((~torch.isfinite(ga[n])).sum() for n in keys))}

    mb = {}
    for k in (0, 6, 13, 19):
        mc, gc = run_mb("compiled", k)
        me, ge = run_mb("eager", k)
        m64, g64 = run_mb("fp64", k)
        mb[str(k)] = {"metrics": {"compiled": mc, "eager": me, "fp64heads": m64},
                      "compiled_vs_eager": compare(gc, ge),
                      "compiled_vs_fp64heads": compare(gc, g64),
                      "eager_vs_fp64heads": compare(ge, g64)}
        log(f"mb {k}: c-vs-e {mb[str(k)]['compiled_vs_eager']['global_rel_err']:.3e} "
            f"c-vs-64 {mb[str(k)]['compiled_vs_fp64heads']['global_rel_err']:.3e} "
            f"e-vs-64 {mb[str(k)]['eager_vs_fp64heads']['global_rel_err']:.3e}")
    res["minibatch"] = mb
    (OUT / "results2.json").write_text(json.dumps(res, indent=2, default=str))
    log("wrote results2.json")


if __name__ == "__main__":
    main()
