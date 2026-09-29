"""Check 3: full-model GPU smoke - sampling, replay, teacher KL, values, loss backward.

One density per process; the Inductor GEMM backend comes from the wrapper.
  sample-256 : model(obs) under no_grad + BF16 autocast, 128 envs = 256 rows.
  replay-256 : evaluate_actions(obs, sampled actions), grad enabled (as PPO).
  replay-1024: obs and actions tiled x4 (the 2-rank PPO minibatch shape).
The log-ratio per seat row is sum(replay event log-probs) - sum(sampled event
log-probs), i.e. ppo_clip_mode "per_player" as the first-minibatch alarm
reads it (rl.first_minibatch_logratio_limit 0.05 nats, |weighted mean|).
Teacher (Phase 4): combined path with the model as its own teacher, cached
path (compute_teacher_distillation_targets -> evaluate_actions_with_cached_
teacher) with itself, and both paths against a perturbed copy (every
parameter + N(0, (0.05 std(p))^2), compiled through the same path).
Loss backward: PPO-shaped loss at 1,024 rows with the perturbed teacher's
cached targets (teacher_kl_coef 0.005, teacher_value_coef 0.005), backward,
gradients finite. No optimizer step.

Pre-declared pass criteria (judged by driver.py from the result record):
  replay accepted at 256 and 1,024 (no GrammarReplayError);
  |mean per-row log-ratio| <= 0.05 at 256 and 1,024; all log-probs finite;
  self KL finite, per-row mean <= 1e-3, per-row max <= 1e-2, per-event min >= -1e-4;
  perturbed KL finite, per-row mean > 0 and > self per-row mean, per-event min >= -1e-4;
  values finite and in [-1, 1]; loss and every gradient finite.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

import common as C

SEED = 0
PERTURB = 0.05


def stats(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().float()
    return {"mean": float(t.mean()), "abs_mean": float(t.abs().mean()),
            "abs_max": float(t.abs().max()), "min": float(t.min()),
            "max": float(t.max()), "nonfinite": int((~torch.isfinite(t)).sum()),
            "n": t.numel()}


def replay_metrics(old_event: torch.Tensor, ev: Any, old_values: torch.Tensor,
                   old_entropy: torch.Tensor) -> dict[str, Any]:
    from owl.kaggriculture import types as kt

    new_event = ev.log_probs.event.float()
    old = old_event.float()
    logratio = new_event.sum(dim=(-1, -2)) - old.sum(dim=(-1, -2))
    per_slot = (new_event - old).abs().amax(dim=(0, 1, 2))
    return {
        "logratio_per_row": stats(logratio),
        "per_slot_max_abs_diff": {kt.SLOT_NAMES[s]: float(per_slot[s])
                                  for s in range(kt.ACTION_SLOTS)},
        "event_max_abs_diff": float((new_event - old).abs().max()),
        "new_logp_nonfinite": int((~torch.isfinite(new_event)).sum()),
        "new_logp_max": float(new_event.max()),
        "entropy_max_abs_diff": float(
            (ev.entropies.event.float() - old_entropy.float()).abs().max()),
        "values_max_abs_diff": float((ev.values.float() - old_values.float()).abs().max()),
        "values": stats(ev.values),
        "joint_logp_row": stats(old.sum(dim=(-1, -2))),
    }


def kl_metrics(tev: Any, obs: Any, model: Any) -> dict[str, Any]:
    from owl.kaggriculture import types as kt

    event = tev.action_kl.event.float()
    per_row = event.sum(dim=(-1, -2))
    per_slot = event.sum(dim=(0, 1, 2)) / per_row.numel()
    tw = tev.teacher_winner_probabilities.float()
    sw = tev.student_winner_log_probabilities.float()
    ce = model.teacher_value_cross_entropy(sw, tw, value_mask=obs.still_playing)
    teacher_entropy = -(tw * tw.clamp_min(1e-30).log()).sum(-1).mean(-1)
    return {
        "kl_per_row": stats(per_row),
        "kl_event_min": float(event.min()),
        "kl_nonfinite": int((~torch.isfinite(event)).sum()),
        "kl_per_slot_mean_per_row": {kt.SLOT_NAMES[s]: float(per_slot[s])
                                     for s in range(kt.ACTION_SLOTS)},
        "value_ce": stats(ce),
        "value_kl_ce_minus_teacher_entropy": stats(ce - teacher_entropy),
        "student_values": stats(tev.student.values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=2 if C.DRYRUN else 128)
    args = ap.parse_args()
    fh = open(args.out, "a")
    C.configure()
    dev = C.device()
    import owl.model.kaggriculture as km
    from owl.kaggriculture import types as kt

    flash_flags: list[bool] = []
    real_use_flash = km.use_flash_attn

    def counting_use_flash(x: torch.Tensor) -> bool:
        flag = real_use_flash(x)
        flash_flags.append(bool(flag))
        return flag

    km.use_flash_attn = counting_use_flash
    C.emit(fh, C.process_record(args.case))
    model = C.build_model(dev, seed=SEED).train()
    C.compile_trunk(model)
    compiled = model._compiled_transformer_trunk
    trunk_calls = [0]

    def counting(*a: Any) -> torch.Tensor:
        trunk_calls[0] += 1
        return compiled(*a)

    if compiled is not None:
        model._compiled_transformer_trunk = counting
    obs = C.to_dev(C.make_obs_batch(args.envs, args.density), dev)
    rec: dict[str, Any] = {"event": "result", "case": args.case,
                           "density": args.density, "rows_sample": 2 * args.envs,
                           "rows_replay_tiled": 8 * args.envs}
    steps: dict[str, str] = {}

    def step(name: str) -> None:
        steps[name] = "started"
        rec["steps"] = steps

    try:
        # --- sampling --------------------------------------------------------
        step("sample_256")
        torch.manual_seed(SEED + 1)
        with torch.no_grad(), C.amp(dev):
            s = model(obs)
        actions = s.actions
        old_event = s.log_probs.event.float()
        rec["sample"] = {
            "log_probs": stats(old_event), "entropies": stats(s.entropies.event),
            "values": stats(s.values), "winner_probabilities": stats(
                s.winner_probabilities),
            "lengths": stats(actions.lengths.float()),
            "old_logp_max": float(old_event.max()),
        }
        steps["sample_256"] = "ok"
        # --- replay at 256 ---------------------------------------------------
        step("replay_256")
        with C.amp(dev):
            ev = model.evaluate_actions(obs, actions)
        rec["replay_256"] = replay_metrics(old_event, ev, s.values, s.entropies.event)
        steps["replay_256"] = "ok"
        del ev
        # --- replay at 1,024 (tiled x4) ---------------------------------------
        step("replay_1024")
        obs4 = C.tile(obs, 4)
        actions4 = C.tile_actions(actions, 4)
        with C.amp(dev):
            ev4 = model.evaluate_actions(obs4, actions4)
        rec["replay_1024"] = replay_metrics(
            old_event.repeat(4, 1, 1, 1), ev4, s.values.repeat(4, 1),
            s.entropies.event.repeat(4, 1, 1, 1))
        steps["replay_1024"] = "ok"
        del ev4
        # --- compute_value ---------------------------------------------------
        step("compute_value_256")
        with torch.no_grad(), C.amp(dev):
            v = model.compute_value(obs)
        rec["compute_value_256"] = {"values": stats(v), "max_abs_diff_vs_sample": float(
            (v.float() - s.values.float()).abs().max())}
        steps["compute_value_256"] = "ok"
        # --- teacher KL: self --------------------------------------------------
        step("teacher_self_combined")
        with C.amp(dev):
            tev = model.evaluate_actions_with_teacher(obs, actions, model)
        rec["teacher_self_combined"] = kl_metrics(tev, obs, model)
        steps["teacher_self_combined"] = "ok"
        step("teacher_self_cached")
        with C.amp(dev):
            targets = model.compute_teacher_distillation_targets(obs, actions)
            tev_c = model.evaluate_actions_with_cached_teacher(obs, actions, targets)
        rec["teacher_self_cached"] = kl_metrics(tev_c, obs, model)
        rec["teacher_self_cached_vs_combined_kl_max_abs"] = float(
            (tev_c.action_kl.event.float() - tev.action_kl.event.float()).abs().max())
        steps["teacher_self_cached"] = "ok"
        del tev, tev_c, targets
        # --- teacher KL: perturbed copy ----------------------------------------
        step("teacher_perturbed")
        teacher = C.build_model(dev, seed=SEED)
        teacher.load_state_dict(model.state_dict())
        gen = torch.Generator(device=dev).manual_seed(123)
        with torch.no_grad():
            for p in teacher.parameters():
                scale = float(p.std()) if p.numel() > 1 else 1.0
                p.add_(torch.randn(p.shape, device=dev, generator=gen) * PERTURB * scale)
        teacher.eval()
        C.compile_trunk(teacher)
        with C.amp(dev):
            tev_p = model.evaluate_actions_with_teacher(obs, actions, teacher)
            targets_p = teacher.compute_teacher_distillation_targets(obs, actions)
            tev_pc = model.evaluate_actions_with_cached_teacher(obs, actions, targets_p)
        rec["teacher_perturbed_combined"] = kl_metrics(tev_p, obs, model)
        rec["teacher_perturbed_cached"] = kl_metrics(tev_pc, obs, model)
        rec["teacher_perturbed_cached_vs_combined_kl_max_abs"] = float(
            (tev_pc.action_kl.event.float() - tev_p.action_kl.event.float()).abs().max())
        steps["teacher_perturbed"] = "ok"
        del tev_p, tev_pc, targets_p
        # --- PPO-shaped loss backward at 1,024 --------------------------------
        step("loss_backward_1024")
        with C.amp(dev):
            targets4 = teacher.compute_teacher_distillation_targets(obs4, actions4)
            tev4 = model.evaluate_actions_with_cached_teacher(obs4, actions4, targets4)
        new_logp = tev4.student.log_probs.event.float().sum(dim=(-1, -2))
        old_logp = old_event.repeat(4, 1, 1, 1).sum(dim=(-1, -2))
        g = torch.Generator(device=dev).manual_seed(3)
        adv = torch.randn(old_logp.shape, generator=g, device=dev)
        ret = torch.rand(old_logp.shape, generator=g, device=dev) * 2 - 1
        ratio = (new_logp - old_logp).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
        v_loss = 0.5 * (tev4.student.values.float() - ret).pow(2).mean()
        ent = tev4.student.entropies.event.float().sum(dim=(-1, -2)).mean()
        kl = tev4.action_kl.event.float().sum(dim=(-1, -2)).mean()
        vce = model.teacher_value_cross_entropy(
            tev4.student_winner_log_probabilities, tev4.teacher_winner_probabilities,
            value_mask=obs4.still_playing).mean()
        loss = pg + 0.5 * v_loss - 0.01 * ent + 0.005 * kl + 0.005 * vce
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        nonfinite = sum(int((~torch.isfinite(gr)).sum()) for gr in grads)
        gnorm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gr.float()) for gr in grads]))
        rec["loss_backward_1024"] = {
            "loss": float(loss), "pg": float(pg), "v_loss": float(v_loss),
            "entropy": float(ent), "teacher_kl": float(kl), "teacher_value_ce": float(vce),
            "params_with_grad": len(grads),
            "params_total": sum(1 for _ in model.parameters()),
            "grad_nonfinite": nonfinite, "grad_norm": float(gnorm),
        }
        model.zero_grad(set_to_none=True)
        steps["loss_backward_1024"] = "ok"
    except Exception as exc:  # recorded, then re-raised for a nonzero exit
        rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        rec["use_flash_attn_all_true"] = bool(flash_flags) and all(flash_flags)
        rec["compiled_trunk_calls"] = trunk_calls[0]
        rec["memory"] = C.mem(dev)
        C.emit(fh, rec)
        fh.close()
        raise
    rec["use_flash_attn_all_true"] = bool(flash_flags) and all(flash_flags)
    rec["use_flash_attn_calls"] = len(flash_flags)
    rec["compiled_trunk_calls"] = trunk_calls[0]
    rec["memory"] = C.mem(dev)
    rec["max_frames"] = kt.MAX_FRAMES
    C.emit(fh, rec)
    C.emit(fh, {"event": "done", "case": args.case})
    fh.close()


if __name__ == "__main__":
    main()
