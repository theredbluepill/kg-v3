"""Case F (control): Isaiah's stateless_transformer_6m on real Orbit Wars states.

Config: configs/scaling_6m.yaml (FullConfig: env entity_based_ext_v2 /
discrete_targets, model configs/model/stateless_transformer_6m.yaml) at 8fde43c.
Fresh weights (seed 0), .train(), fp32 parameters, BF16 autocast, TF32 on
(configure_torch). Trunk compiled through configure_model_compile
(model_compile "trunk", max-autotune-no-cudagraphs -> compile_transformer_trunk,
dynamic=True) exactly as PPOConfig's defaults, or left eager (--trunk eager).

States: Isaiah's Rust VectorizedEnv (n_envs = rows, two_player_weight from the
config), reset, then STEPS steps with actions sampled by this same fresh model
(near-uniform at head gain 0.01), rollout-style (no_grad + autocast). The last
observation per row count is saved to --obs-dir on first use and loaded by
later processes, so every setting sees identical states.

Per row count (256, 1,024): S = model(obs) under no_grad (his rollout path,
ppo.py _collect_rollout); R = model.evaluate_actions(obs, S.actions) with grad
enabled (his PPO path). Gaps on present players (still_playing): values,
winner log-probs, centred winner log-probs (= centred critic logits), per-player
joint log-probs (per_player_entity summed, the per_player PPO ratio input),
per-entity log-probs, entropies. Supplementary cells as kg_gap.py, and (case E
analogue) trunk hidden states at all / critic-value / plan tokens under both
grad modes plus a critic head swap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import common as C

SEED = 0
STEPS = 4 if C.DRYRUN else 24


def playing_centered(logp: torch.Tensor, playing: torch.Tensor) -> torch.Tensor:
    lp = logp.float().masked_fill(~playing, 0.0)
    mean = lp.sum(-1, keepdim=True) / playing.sum(-1, keepdim=True)
    return (lp - mean).masked_fill(~playing, 0.0)


def pair_metrics(s: Any, r: Any, playing: torch.Tensor) -> dict[str, Any]:
    s_lp = s.winner_probabilities.float().clamp_min(1e-30).log()
    r_lp = (r.winner_log_probabilities.float() if getattr(r, "winner_log_probabilities", None)
            is not None else r.winner_probabilities.float().clamp_min(1e-30).log())
    old_ppe = s.log_probs.per_player_entity.float()
    new_ppe = r.log_probs.per_player_entity.float()
    old_pp = old_ppe.sum(-1)
    new_pp = new_ppe.sum(-1)
    return {
        "values": C.diff(r.values, s.values, playing),
        "winner_logp": C.diff(r_lp, s_lp, playing),
        "winner_logp_centered": C.diff(playing_centered(r_lp, playing),
                                       playing_centered(s_lp, playing), playing),
        "logp_per_player": C.diff(new_pp, old_pp, playing),
        "logratio_per_player": C.stats((new_pp - old_pp)[playing]),
        "logp_per_player_entity": C.diff(new_ppe, old_ppe),
        "entropy_per_player_entity": C.diff(r.entropies.per_player_entity,
                                            s.entropies.per_player_entity),
        "r_values_requires_grad": bool(r.values.requires_grad),
        "s_values": C.stats(s.values[playing]),
        "s_logp_per_player": C.stats(old_pp[playing]),
        "players_per_row": C.stats(playing.sum(-1).float()),
    }


def run_pair(model: Any, obs: Any, dev: torch.device) -> dict[str, Any]:
    playing = obs.still_playing.bool()
    torch.manual_seed(SEED + 1)
    with torch.no_grad(), C.amp(dev, "bf16"):
        s = model(obs)
    with C.amp(dev, "bf16"):
        r = model.evaluate_actions(obs, s.actions)
    rec: dict[str, Any] = {"pair": pair_metrics(s, r, playing)}
    with torch.no_grad(), C.amp(dev, "bf16"):
        e_ng = model.evaluate_actions(obs, s.actions)
    rec["eval_nograd_vs_sample"] = pair_metrics(s, e_ng, playing)
    with C.amp(dev, "bf16"):
        v_g = model.compute_value(obs)
    rec["compute_value_grad_vs_sample"] = {
        "values": C.diff(v_g, s.values, playing), "requires_grad": bool(v_g.requires_grad)}
    with torch.no_grad(), C.amp(dev, "bf16"):
        v_ng = model.compute_value(obs)
    rec["compute_value_nograd_vs_sample"] = {"values": C.diff(v_ng, s.values, playing)}
    with C.amp(dev, "bf16"):
        r2 = model.evaluate_actions(obs, s.actions)
    rec["replay_grad_repeat"] = {"values": C.diff(r2.values, r.values, playing)}
    return rec


def hidden_metrics(model: Any, obs: Any, dev: torch.device) -> dict[str, Any]:
    from owl.model.stateless_transformer_v1 import _action_entity_slots_from_mask

    slots = _action_entity_slots_from_mask(obs.action_mask)
    playing = obs.still_playing.bool()
    with torch.no_grad(), C.amp(dev, "bf16"):
        enc_ng = model.encode_observations(obs, action_entity_slots=slots)
    with C.amp(dev, "bf16"):
        enc_g = model.encode_observations(obs, action_entity_slots=slots)
    rec: dict[str, Any] = {
        "all": C.diff(enc_g.hidden, enc_ng.hidden, enc_ng.token_mask.bool()),
        "critic": C.diff(enc_g.critic_value_hidden, enc_ng.critic_value_hidden, playing),
        "plan": C.diff(enc_g.actor_plan_hidden, enc_ng.actor_plan_hidden),
        "token_mask_equal": bool(torch.equal(enc_g.token_mask, enc_ng.token_mask)),
    }
    with torch.no_grad(), C.amp(dev, "bf16"):
        v_ng, _ = model._critic(enc_ng.critic_value_hidden, playing)
        v_g, _ = model._critic(enc_g.critic_value_hidden, playing)
    rec["head_swap_values"] = C.diff(v_g, v_ng, playing)
    # fp32 critic head (autocast off, TF32 off) on each path's hidden states
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        v32_ng, _ = model._critic(enc_ng.critic_value_hidden.float(), playing)
        v32_g, _ = model._critic(enc_g.critic_value_hidden.float(), playing)
        logits = model.critic_head(enc_ng.critic_value_hidden.float()).squeeze(-1)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    rec["head_swap_fp32_head_values"] = C.diff(v32_g, v32_ng, playing)
    rec["fp32_head_vs_bf16_head_values_nograd"] = C.diff(v32_ng, v_ng, playing)
    rec["critic_logit_centered_fp32_head"] = C.stats(
        playing_centered(logits, playing)[playing])
    return rec


def get_states(cfg: Any, model: Any, rows: int, dev: torch.device,
               obs_dir: Path) -> tuple[Any, dict[str, Any]]:
    from owl.rl import VectorizedEnv
    from owl.train.ppo import _actions_to_cpu, _obs_to_device

    path = obs_dir / f"orbit_obs_{rows}.pt"
    if path.exists():
        obs = torch.load(path, weights_only=False)
        return _obs_to_device(obs, dev), {"source": "loaded", "path": str(path)}
    env = VectorizedEnv(
        n_envs=rows,
        obs_spec=cfg.env.obs_spec,
        action_spec=cfg.env.action_spec,
        two_player_weight=cfg.env.two_player_weight,
        reward_mode=cfg.env.reward_mode,
        pin_memory=dev.type == "cuda",
    )
    obs = env.reset()
    torch.manual_seed(SEED + 2)
    launches = 0
    for _ in range(STEPS):
        with torch.no_grad(), C.amp(dev, "bf16"):
            out = model(_obs_to_device(obs, dev))
        launches += int(out.actions.launch.sum())
        obs, _, _, _ = env.step(_actions_to_cpu(out.actions))
    cpu = _obs_to_device(obs, torch.device("cpu"))  # cpu target clones the buffers
    obs_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu, path)
    info = {"source": "generated", "path": str(path), "steps": STEPS,
            "sampled_launches": launches,
            "entity_mask_true": int(cpu.entity_mask.sum()),
            "still_playing_per_row": C.stats(cpu.still_playing.sum(-1).float())}
    return _obs_to_device(cpu, dev), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--trunk", choices=("compiled", "eager"), required=True)
    ap.add_argument("--rows", default="4,8" if C.DRYRUN else "256,1024")
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fh = open(args.out, "a")
    C.configure("bf16")
    dev = C.device()
    import owl.model.stateless_transformer_v1 as st
    from owl.model import create_model
    from owl.train import FullConfig

    # Amendment 3: attempt 3 wrapped st.use_flash_attn with a list-appending
    # closure. Isaiah's attention calls use_flash_attn inside the compiled trunk,
    # so Dynamo guarded on the list length, recompiled on every call and hit
    # recompile_limit (8), running the 1,024-row cells eagerly. No wrapper now:
    # force_flash_attn=True makes his trunk raise on any non-flash call
    # (_requires_flash_attn), and a probe records use_flash_attn on a CUDA bf16
    # tensor; the driver also fails a stage whose log hits recompile_limit.
    flash_flags: list[bool] = []
    C.emit(fh, C.process_record(args.case))
    if not C.DRYRUN:
        flash_flags.append(bool(st.use_flash_attn(
            torch.zeros(1, 8, 8, 32, device=dev, dtype=torch.bfloat16))))
    cfg = FullConfig.from_file(C.ROOT / "configs/scaling_6m.yaml")
    torch.manual_seed(SEED)
    model = create_model(cfg.model, obs_spec=cfg.env.obs_spec,
                         action_spec=cfg.env.action_spec).to(dev).train()
    model.reset_parameters()  # as run_ppo._create_training_model_for_config
    head_gains = C.output_layer_gains(model, model.critic_head.out)
    trunk_calls = [0]
    if args.trunk == "compiled":
        C.compile_trunk(model)
        compiled = model._compiled_transformer_trunk

        def counting(*a: Any) -> torch.Tensor:
            trunk_calls[0] += 1
            return compiled(*a)

        model._compiled_transformer_trunk = counting
    rows = [int(r) for r in args.rows.split(",")]
    rec: dict[str, Any] = {
        "event": "result", "case": args.case, "trunk": args.trunk, "rows": rows,
        "config": {"model": cfg.model.model_dump(mode="json"),
                   "obs_spec": cfg.env.obs_spec.model_dump(mode="json"),
                   "action_spec": cfg.env.action_spec.model_dump(mode="json"),
                   "two_player_weight": cfg.env.two_player_weight,
                   "rl_model_compile": cfg.rl.model_compile,
                   "rl_model_compile_mode": cfg.rl.model_compile_mode,
                   "rl_dtype": cfg.rl.dtype},
        "output_layer_spectral_norms": head_gains,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }
    steps: dict[str, str] = {}
    rec["steps"] = steps
    try:
        for r in rows:
            steps[f"states_{r}"] = "started"
            obs, info = get_states(cfg, model, r, dev, Path(args.obs_dir))
            rec[f"states_{r}"] = info
            steps[f"states_{r}"] = "ok"
            steps[f"pair_{r}"] = "started"
            rec[f"default_gain_{r}"] = run_pair(model, obs, dev)
            steps[f"pair_{r}"] = "ok"
            steps[f"hidden_{r}"] = "started"
            rec[f"hidden_{r}"] = hidden_metrics(model, obs, dev)
            steps[f"hidden_{r}"] = "ok"
            del obs
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        raise
    finally:
        rec["use_flash_attn_flags"] = sorted(set(flash_flags))
        rec["use_flash_attn_calls"] = len(flash_flags)
        rec["force_flash_attn"] = cfg.model.force_flash_attn
        rec["flash_check"] = "probe + force_flash_attn (Amendment 3)"
        rec["compiled_trunk_calls"] = trunk_calls[0]
        rec["memory"] = C.mem(dev)
        try:
            from torch._dynamo.utils import counters

            rec["dynamo_counters"] = {k: dict(v) for k, v in counters.items()}
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            rec["dynamo_counters"] = f"unavailable: {exc}"
        C.emit(fh, rec)
    C.emit(fh, {"event": "done", "case": args.case})
    fh.close()


if __name__ == "__main__":
    main()
