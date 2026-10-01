"""GPU bf16 parity: compiled actor heads vs eager heads (diagnostic only, not shipped).

Question: on CUDA bf16 autocast, how far does the compiled grammar core
(rl.compile_actor_heads) drift from the eager core, on real observations and
the same stored actions, for (1) teacher-forced replay logp/entropy/values,
(2) PPO minibatch loss and parameter gradients, (3) sampling with identical
exponential noise? And how does the stored rollout logp (compiled sampling)
compare with compiled / eager replay logp (the production PPO ratio at update
start)?

Arms toggle ``model._compiled_actor_core`` between the compiled callable and
None on ONE model (same weights, same compiled trunk), so only the heads differ.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from owl.game import create_env
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.model.factory import create_model
from owl.model.kaggriculture_actor import sample_policy_exponentials
from owl.train.config import FullConfig
from owl.train.optimizer import create_lr_scheduler, create_optimizer
from owl.train.ppo import (
    PPOTrainer,
    _actions_index,
    _flatten_actions_time,
    _flatten_obs_time,
    _map_observation,
    _model_evaluate_actions,
    _model_evaluate_actions_with_cached_teacher,
    _obs_index,
    _output_entropy_for_clip_mode,
    _output_logp_for_clip_mode,
    _output_values,
    _policy_mask,
)
from owl.train.utils import autocast_context, configure_model_compile, configure_torch

OUT = Path("/root/parity")
CKPT = Path("/root/ckpt/checkpoint_00_110_015_744.pt")
CFG = Path("/root/ckpt/config.yaml")
N_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 720
SEED = 4242


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dist(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().double().flatten()
    if x.numel() == 0:
        return {"n": 0}
    a = x.abs()
    q = torch.quantile(a[:1_000_000] if a.numel() > 1_000_000 else a,
                       torch.tensor([0.5, 0.99, 0.999], dtype=torch.float64, device=a.device))
    return {
        "n": int(a.numel()),
        "max_abs": float(a.max()),
        "p999_abs": float(q[2]),
        "p99_abs": float(q[1]),
        "median_abs": float(q[0]),
        "mean_abs": float(a.mean()),
        "mean_signed": float(x.mean()),
        "frac_gt_1e-2": float((a > 1e-2).double().mean()),
        "frac_gt_1e-3": float((a > 1e-3).double().mean()),
        "frac_exact_zero": float((a == 0).double().mean()),
        "nonfinite": int((~torch.isfinite(x)).sum()),
    }


def main() -> None:
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    torch.manual_seed(SEED)
    configure_torch()
    device = torch.device("cuda")
    cfg = FullConfig.from_file(
        CFG,
        {
            "env.n_envs": N_ENVS,
            "env.seed": SEED,
            "rl.horizon": HORIZON,
        },
    )
    rl = cfg.rl
    log(f"rl.compile_actor_heads={rl.compile_actor_heads} model_compile={rl.model_compile} "
        f"mode={rl.model_compile_mode} dtype={rl.dtype} clip_mode={rl.ppo_clip_mode} "
        f"teacher_mode={rl.teacher_mode} kl={rl.teacher_kl_coef} tv={rl.teacher_value_coef}")
    env = create_env(cfg.env, n_envs=N_ENVS, base_seed=SEED, rank=0, world_size=1,
                     pin_memory=True, transfer_device=device)
    assert isinstance(env, KaggricultureVectorizedEnv)
    log("env created")
    model = create_model(cfg.model, obs_spec=cfg.env.obs_spec,
                         action_spec=cfg.env.action_spec).to(device)
    log("model created")
    n_compiled = configure_model_compile(model, rl)
    log(f"configure_model_compile -> {n_compiled}; compiled core set: "
        f"{model._compiled_actor_core is not None}")
    optimizer = create_optimizer(model, cfg.optimizer)
    sched = create_lr_scheduler(optimizer, cfg.optimizer.lr_schedule)
    trainer = PPOTrainer(config=rl, env=env, model=model, optimizer=optimizer,
                         lr_scheduler=sched, device=device)
    meta = trainer.load_model_weights(CKPT)
    log(f"loaded weights env_steps={meta.env_steps}")
    # Teacher as a fresh launch with teacher_mode last_best: a compiled copy.
    teacher = create_model(cfg.model, obs_spec=cfg.env.obs_spec,
                           action_spec=cfg.env.action_spec).to(device)
    teacher.load_state_dict(model.state_dict())
    configure_model_compile(teacher, rl)
    teacher.eval()
    trainer.set_teacher_model(teacher, active=True)
    compiled_core = model._compiled_actor_core
    assert compiled_core is not None

    # ---- rollout (production path: compiled heads + compiled trunk) ----
    t0 = time.time()
    with torch.no_grad():
        last_values = trainer._collect_rollout()
    torch.cuda.synchronize()
    log(f"rollout {HORIZON}x{N_ENVS} done in {time.time() - t0:.1f}s")
    segments = trainer.rollout.segment_major()
    assert segments.learner is None
    value_mask = segments.obs.still_playing
    policy_mask = _policy_mask(segments.obs)
    advantages, returns = trainer._compute_gae(
        rewards=segments.rewards, values=segments.values, dones=segments.dones,
        last_values=last_values, gamma=rl.gamma, gae_lambda=rl.gae_lambda,
        truncated=segments.truncated, bootstrap_values=segments.bootstrap_values)
    winner_targets = trainer._compute_winner_targets(segments, last_values)
    t0 = time.time()
    teacher_targets = trainer._precompute_teacher_targets(segments)
    log(f"teacher targets in {time.time() - t0:.1f}s; "
        f"policy_mask frac={policy_mask.float().mean().item():.3f} "
        f"logp shape={tuple(segments.logp.shape)}")
    results: dict = {"config": {"n_envs": N_ENVS, "horizon": HORIZON, "seed": SEED,
                                "torch": torch.__version__,
                                "ckpt_env_steps": meta.env_steps}}

    def set_arm(arm: str) -> None:
        model._compiled_actor_core = compiled_core if arm == "compiled" else None

    # ---- (1) teacher-forced replay, per segment, both update paths ----
    per = {arm: {"plain": [], "teacher": []} for arm in ("compiled", "eager")}
    model.train()
    with torch.no_grad():
        for arm in ("compiled", "eager"):
            set_arm(arm)
            t0 = time.time()
            for k in range(N_ENVS):
                idx = torch.tensor([k], device=device)
                seg_obs = _obs_index(segments.obs, idx)
                seg_act = _actions_index(segments.actions, idx)
                obs = _flatten_obs_time(seg_obs)
                act = _flatten_actions_time(seg_act)
                old = segments.logp[idx]
                with autocast_context(rl, device):
                    out_p = _model_evaluate_actions(model, obs, act, hidden_state=None,
                                                    dones=segments.dones[idx])
                    tev = _model_evaluate_actions_with_cached_teacher(
                        model, seg_obs, seg_act, teacher_targets.index(idx),
                        hidden_state=None, dones=segments.dones[idx],
                        compute_teacher_action_kl=True, compute_teacher_value=True)
                for name, out in (("plain", out_p), ("teacher", tev.student)):
                    per[arm][name].append({
                        "logp": _output_logp_for_clip_mode(out, rl.ppo_clip_mode).view_as(old).float(),
                        "ent": _output_entropy_for_clip_mode(out, old, rl.ppo_clip_mode).float(),
                        "event_logp": out.log_probs.event.float(),
                        "event_ent": out.entropies.event.float(),
                        "values": _output_values(out).float().view_as(segments.values[idx]),
                        "kl": None if name == "plain" else tev.action_kl.per_player_entity.float(),
                    })
            torch.cuda.synchronize()
            log(f"replay arm {arm}: {time.time() - t0:.1f}s")

    pm = policy_mask  # [N, T, P]
    stored = segments.logp.float()

    def cat(arm: str, path: str, key: str) -> torch.Tensor:
        return torch.cat([d[key] for d in per[arm][path]], dim=0)

    rep: dict = {}
    for path in ("plain", "teacher"):
        lc, le = cat("compiled", path, "logp"), cat("eager", path, "logp")
        ec, ee = cat("compiled", path, "ent"), cat("eager", path, "ent")
        vc, ve = cat("compiled", path, "values"), cat("eager", path, "values")
        m = pm
        d = (lc - le)[m]
        ev_c, ev_e = cat("compiled", path, "event_logp"), cat("eager", path, "event_logp")
        en_c, en_e = cat("compiled", path, "event_ent"), cat("eager", path, "event_ent")
        ev_mask = (ev_c != 0) | (ev_e != 0) | (en_c != 0) | (en_e != 0)
        ratio = d.exp()
        # late game = last 20% of the horizon
        T = lc.shape[1]
        late = torch.zeros_like(m)
        late[:, int(0.8 * T):] = True
        rep[path] = {
            "per_player_logp_compiled_minus_eager": dist(d),
            "ratio_exp_logp_c_minus_logp_e": {"min": float(ratio.min()), "max": float(ratio.max()),
                                              "mean": float(ratio.mean())},
            "per_player_logp_diff_late_game": dist((lc - le)[m & late]),
            "per_player_entropy_diff": dist((ec - ee)[m]),
            "per_event_logp_diff": dist((ev_c - ev_e)[ev_mask]),
            "per_event_entropy_diff": dist((en_c - en_e)[ev_mask]),
            "values_diff": dist((vc - ve)[value_mask]),
            "eager_logp_abs_mean": float(le[m].abs().mean()),
            "nonfinite_compiled_logp": int((~torch.isfinite(lc[m])).sum()),
            "nonfinite_eager_logp": int((~torch.isfinite(le[m])).sum()),
            # production ratio at update start: stored rollout logp (compiled sampling)
            "stored_rollout_minus_compiled_replay": dist((stored - lc)[m]),
            "stored_rollout_minus_eager_replay": dist((stored - le)[m]),
            "prod_ratio_exp_compiled_replay_minus_stored": {
                "min": float((lc - stored)[m].exp().min()),
                "max": float((lc - stored)[m].exp().max()),
                "mean": float((lc - stored)[m].exp().mean()),
                "logratio_mean": float((lc - stored)[m].mean())},
            "eager_ratio_exp_eager_replay_minus_stored": {
                "min": float((le - stored)[m].exp().min()),
                "max": float((le - stored)[m].exp().max()),
                "mean": float((le - stored)[m].exp().mean()),
                "logratio_mean": float((le - stored)[m].mean())},
        }
        if path == "teacher":
            kc = torch.cat([d_["kl"].float().reshape(-1) for d_ in per["compiled"][path]])
            ke = torch.cat([d_["kl"].float().reshape(-1) for d_ in per["eager"][path]])
            rep[path]["teacher_action_kl_diff"] = dist(kc - ke)
    results["replay"] = rep
    log("replay stats done")

    # ---- (2) PPO minibatch loss + gradients ----
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def run_minibatch(arm: str, k: int) -> tuple[dict, dict[str, torch.Tensor]]:
        set_arm(arm)
        model.zero_grad(set_to_none=True)
        upd = trainer._update_minibatch(
            segments, advantages, returns, policy_mask, value_mask,
            torch.tensor([k], device=device), teacher_targets=teacher_targets,
            winner_targets=winner_targets, value_clip_anchor=segments.values.clone(),
            loss_scale=1.0, step_optimizer=False)
        mt = upd.metrics
        met = {f: float(getattr(mt, f)) for f in (
            "loss", "policy_loss", "value_loss", "entropy_loss", "teacher_kl_loss",
            "teacher_value_loss", "entropy", "approx_kl", "clipfrac", "ratio_mean",
            "ratio_max", "logratio_mean", "logratio_abs_max")}
        grads = {n: (p.grad.detach().float().clone() if p.grad is not None else None)
                 for n, p in params}
        return met, grads

    def compare(ga: dict, gb: dict) -> dict:
        rel = []
        worst = []
        nonfinite = 0
        missing = 0
        for n, _ in params:
            a, b = ga[n], gb[n]
            if a is None or b is None:
                missing += int((a is None) != (b is None))
                continue
            nonfinite += int((~torch.isfinite(a)).sum() + (~torch.isfinite(b)).sum())
            nb = b.norm()
            r = float((a - b).norm() / nb) if nb > 0 else (0.0 if a.norm() == 0 else float("inf"))
            rel.append(r)
            worst.append((r, n, float(nb)))
        t = torch.tensor(rel, dtype=torch.float64)
        flat_a = torch.cat([ga[n].flatten() for n, _ in params if ga[n] is not None and gb[n] is not None])
        flat_b = torch.cat([gb[n].flatten() for n, _ in params if ga[n] is not None and gb[n] is not None])
        worst.sort(reverse=True)
        return {"n_tensors": len(rel), "rel_err_max": float(t.max()),
                "rel_err_median": float(t.median()), "rel_err_mean": float(t.mean()),
                "global_rel_err": float((flat_a - flat_b).norm() / flat_b.norm()),
                "cosine": float(torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0)),
                "grad_norm_a": float(flat_a.norm()), "grad_norm_b": float(flat_b.norm()),
                "worst5": [{"name": n, "rel": r, "ref_norm": nn} for r, n, nn in worst[:5]],
                "nonfinite": nonfinite, "missing_mismatch": missing}

    mb = {}
    for k in sorted({0, N_ENVS // 3, (2 * N_ENVS) // 3, N_ENVS - 1}):
        mc1, gc1 = run_minibatch("compiled", k)
        me1, ge1 = run_minibatch("eager", k)
        mc2, gc2 = run_minibatch("compiled", k)
        me2, ge2 = run_minibatch("eager", k)
        mb[str(k)] = {
            "metrics": {"compiled": mc1, "eager": me1, "compiled_repeat": mc2, "eager_repeat": me2},
            "loss_rel_diff_c_vs_e": abs(mc1["loss"] - me1["loss"]) / max(abs(me1["loss"]), 1e-12),
            "grad_compiled_vs_eager": compare(gc1, ge1),
            "grad_compiled_repeat_noise": compare(gc2, gc1),
            "grad_eager_repeat_noise": compare(ge2, ge1),
        }
        log(f"minibatch seg {k}: loss c={mc1['loss']:.6g} e={me1['loss']:.6g} "
            f"grad rel c-vs-e global={mb[str(k)]['grad_compiled_vs_eager']['global_rel_err']:.3e} "
            f"eager-noise={mb[str(k)]['grad_eager_repeat_noise']['global_rel_err']:.3e}")
    results["minibatch"] = mb
    model.zero_grad(set_to_none=True)

    # ---- (3) sampling with identical exponentials, rollout-shaped batches ----
    samp = {"rows": 0, "rows_token_mismatch": 0, "timesteps": []}
    lp_diffs = []
    model.eval()
    with torch.no_grad():
        for t in range(0, HORIZON, max(1, HORIZON // 60)):
            obs_t = _map_observation(segments.obs, lambda x, t=t: x[:, t])
            with autocast_context(rl, device):
                enc = model.encode_observations(obs_t)
                ctx = model._grammar_context(obs_t)
                rows = ctx.live.shape[0]
                ui, mi = model._actor_inputs(enc, slice(0, rows))
                exps = sample_policy_exponentials(ui)
                args = (ui, mi, ctx.actor_counts, ctx.order_limits, ctx.live,
                        model.action_spec.hire_limit, None, None, False, None, False, exps)
                rc = compiled_core(*args)
                re_ = model.actor.policy_core(*args)
            same = ((rc.tokens == re_.tokens).flatten(1).all(1)) & (rc.lengths == re_.lengths)
            samp["rows"] += rows
            samp["rows_token_mismatch"] += int((~same).sum())
            if (~same).any():
                samp["timesteps"].append({"t": t, "mismatch_rows": int((~same).sum())})
            if same.any():
                lp_diffs.append((rc.log_probs.float() - re_.log_probs.float())[same].flatten())
    samp["same_token_rows_event_logp_diff"] = dist(torch.cat(lp_diffs))
    results["sampling_identical_noise"] = samp
    counters = {}
    try:
        from torch._dynamo.utils import counters as dc
        counters = {k: dict(v) for k, v in dc.items() if k in ("stats", "frames", "recompiles", "graph_break")}
    except Exception as exc:  # diagnostic only
        counters = {"error": repr(exc)}
    results["dynamo_counters"] = json.loads(json.dumps(counters, default=str))
    out = OUT / "results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
