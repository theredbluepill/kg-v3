"""Kaggriculture cases A-E of the value-gap diagnostic.

One (precision, trunk) setting per process; the Inductor GEMM backend comes from
gemm_backend_wrap.py. Model: preset configs/model/kaggriculture.yaml at 8fde43c,
fresh weights (seed 0), .train() as PPO, fp32 parameters. Observations:
tests/kaggriculture/conftest.py::make_obs at mid density (40/40 actors, 4 shops),
envs = rows / 2 (two seat rows per env).

Per row count R (256, 1,024):
  pair  : S = model(obs) under no_grad (the rollout path);
          R = model.evaluate_actions(obs, S.actions) with grad enabled (PPO).
          Gaps R - S: values, critic logit difference log p(self) - log p(opp),
          event log-probs, per-row joint log-ratio, entropies.
  suppl : evaluate_actions under no_grad vs S; compute_value with grad vs S;
          compute_value under no_grad vs S; a second grad replay vs the first;
          a second no_grad sample (same seed) vs S.
  hidden: (case E) encode_observations under no_grad vs with grad: trunk input,
          trunk output at all present tokens, critic-value, plan and own-actor
          tokens; each vs an fp32 eager padded-SDPA reference of the trunk
          (TF32 off, autocast off, same trunk input); head swap = the critic
          head applied (no_grad) to each path's hidden states.
Then (case D) the actor head .out layers are re-initialised with gain 1.0
(critic head, trunk and stems untouched) and the pair is repeated.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from pydantic import BaseModel

import common as C

SEED = 0


def map_tensors(obj: BaseModel, fn: Any) -> Any:
    fields: dict[str, Any] = {}
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        if isinstance(value, torch.Tensor):
            fields[name] = fn(value)
        elif isinstance(value, BaseModel):
            fields[name] = map_tensors(value, fn)
        else:
            fields[name] = value
    return type(obj)(**fields)


def build_model(dev: torch.device, precision: str) -> Any:
    from owl.kaggriculture import types as kt
    from owl.model import kaggriculture as km

    # force_flash_attn must be off in fp32: flash-attn needs bf16/fp16 inputs.
    overrides = None if precision == "bf16" else {"force_flash_attn": False}
    config = km.KaggricultureTransformerConfig.from_file(
        C.ROOT / "configs/model/kaggriculture.yaml", overrides
    )
    torch.manual_seed(SEED)
    model = km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).to(dev)
    return model.train()


def make_obs(rows: int, dev: torch.device) -> Any:
    from tests.kaggriculture.conftest import make_obs as mk

    return map_tensors(mk(rows // 2, own_actors=40, rival_actors=40, shops=4),
                       lambda t: t.to(dev))


def logit_diff(p_or_logp: torch.Tensor, is_log: bool) -> torch.Tensor:
    lp = p_or_logp.float() if is_log else p_or_logp.float().log()
    return lp[..., 0] - lp[..., 1]


def pair_metrics(s: Any, r: Any, *, r_is_eval: bool = True) -> dict[str, Any]:
    old = s.log_probs.event.float()
    new = r.log_probs.event.float()
    rl = (r.winner_log_probabilities if r_is_eval and r.winner_log_probabilities
          is not None else r.winner_probabilities.float().log())
    return {
        "values": C.diff(r.values, s.values),
        "critic_logit_diff": C.diff(logit_diff(rl, True),
                                    logit_diff(s.winner_probabilities, False)),
        "logp_event": C.diff(new, old),
        "logratio_row": C.stats(new.sum(dim=(-1, -2)) - old.sum(dim=(-1, -2))),
        "entropy_event": C.diff(r.entropies.event, s.entropies.event),
        "r_values_requires_grad": bool(r.values.requires_grad),
        "r_values": C.stats(r.values),
        "s_values": C.stats(s.values),
        "s_logp_event": C.stats(old),
        "s_joint_logp_row": C.stats(old.sum(dim=(-1, -2))),
    }


def run_pair(model: Any, obs: Any, dev: torch.device, prec: str,
             suppl: bool) -> dict[str, Any]:
    torch.manual_seed(SEED + 1)
    with torch.no_grad(), C.amp(dev, prec):
        s = model(obs)
    with C.amp(dev, prec):
        r = model.evaluate_actions(obs, s.actions)
    rec = {"pair": pair_metrics(s, r)}
    if suppl:
        with torch.no_grad(), C.amp(dev, prec):
            e_ng = model.evaluate_actions(obs, s.actions)
        rec["eval_nograd_vs_sample"] = pair_metrics(s, e_ng)
        with C.amp(dev, prec):
            v_g = model.compute_value(obs)
        rec["compute_value_grad_vs_sample"] = {
            "values": C.diff(v_g, s.values), "requires_grad": bool(v_g.requires_grad)}
        with torch.no_grad(), C.amp(dev, prec):
            v_ng = model.compute_value(obs)
        rec["compute_value_nograd_vs_sample"] = {"values": C.diff(v_ng, s.values)}
        with C.amp(dev, prec):
            r2 = model.evaluate_actions(obs, s.actions)
        rec["replay_grad_repeat"] = {
            "values": C.diff(r2.values, r.values),
            "logp_event": C.diff(r2.log_probs.event, r.log_probs.event)}
        torch.manual_seed(SEED + 1)
        with torch.no_grad(), C.amp(dev, prec):
            s2 = model(obs)
        rec["sample_nograd_repeat"] = {
            "values": C.diff(s2.values, s.values),
            "actions_equal": bool(torch.equal(s2.actions.tokens, s.actions.tokens)),
            "logp_event": C.diff(s2.log_probs.event, s.log_probs.event)}
        del e_ng, v_g, v_ng, r2, s2
    del s, r
    return rec


def hidden_metrics(model: Any, obs: Any, dev: torch.device, prec: str) -> dict[str, Any]:
    from owl.kaggriculture import types as kt
    from owl.model import kaggriculture as km

    with torch.no_grad(), C.amp(dev, prec):
        x_ng, mask = model._assemble_tokens(obs)
        enc_ng = model.encode_observations(obs)
    with C.amp(dev, prec):
        x_g, _ = model._assemble_tokens(obs)
        enc_g = model.encode_observations(obs)
    rec: dict[str, Any] = {"trunk_input": C.diff(x_g, x_ng, mask),
                           "x_dtype": str(x_ng.dtype)}
    h_ng, h_g = enc_ng.hidden.detach(), enc_g.hidden.detach()
    critic_start = h_ng.shape[1] - kt.PLAYERS
    plan = slice(critic_start - km._PLAN_TOKENS, critic_start)
    groups = {
        "all": (slice(None), mask),
        "critic": (slice(critic_start, None), mask[:, critic_start:]),
        "plan": (plan, mask[:, plan]),
        "own_actor": (slice(0, kt.MAX_ACTORS), mask[:, : kt.MAX_ACTORS]),
    }
    # fp32 reference: eager padded SDPA, autocast off, TF32 off, same input.
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    chunks = []
    with torch.no_grad():
        for i in range(0, x_ng.shape[0], 128):
            out = model._forward_transformer_trunk(
                x_ng[i:i + 128].float(), mask[i:i + 128], None)
            chunks.append(out.masked_fill(~mask[i:i + 128].unsqueeze(-1), 0.0))
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    h_ref = torch.cat(chunks)
    for name, (sl, m) in groups.items():
        rec[name] = {
            "grad_vs_nograd": C.diff(h_g[:, sl], h_ng[:, sl], m),
            "nograd_vs_fp32": C.diff(h_ng[:, sl], h_ref[:, sl], m),
            "grad_vs_fp32": C.diff(h_g[:, sl], h_ref[:, sl], m),
        }
    # head swap: the same (no_grad) critic head on each path's hidden states
    with torch.no_grad(), C.amp(dev, prec):
        lp_ng = model._winner_log_probabilities(enc_ng)
        lp_g = model._winner_log_probabilities(enc_g)
    v_ng = 2 * lp_ng[..., 0].exp() - 1
    v_g = 2 * lp_g[..., 0].exp() - 1
    rec["head_swap_values"] = C.diff(v_g, v_ng)
    rec["head_swap_critic_logit_diff"] = C.diff(lp_g[..., 0] - lp_g[..., 1],
                                                lp_ng[..., 0] - lp_ng[..., 1])
    rec["critic_logit_bf16_head"] = C.stats(lp_ng[..., 0] - lp_ng[..., 1])
    # fp32 critic head (autocast off, TF32 off) on each path's hidden states and
    # on the fp32 reference hidden: separates trunk noise from head rounding.
    cmask = mask[:, critic_start:]

    def fp32_head_values(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model.critic_head(h.float()).float().squeeze(-1)
        lp = torch.log_softmax(logits.masked_fill(
            ~cmask, torch.finfo(torch.float32).min), dim=-1)
        return 2 * lp[..., 0].exp() - 1, lp[..., 0] - lp[..., 1]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        v_ref, _ = fp32_head_values(h_ref[:, critic_start:])
        v32_ng, d32_ng = fp32_head_values(h_ng[:, critic_start:])
        v32_g, d32_g = fp32_head_values(h_g[:, critic_start:])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    rec["head_swap_fp32_head_values"] = C.diff(v32_g, v32_ng)
    rec["head_swap_fp32_head_critic_logit_diff"] = C.diff(d32_g, d32_ng)
    rec["fp32_head_vs_bf16_head_values_nograd"] = C.diff(v32_ng, v_ng)
    rec["values_nograd_vs_fp32_head"] = C.diff(v_ng, v_ref)
    rec["values_grad_vs_fp32_head"] = C.diff(v_g, v_ref)
    del enc_ng, enc_g, h_ref, chunks
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    ap.add_argument("--trunk", choices=("compiled", "eager"), required=True)
    ap.add_argument("--rows", default="4,8" if C.DRYRUN else "256,1024")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--gain-swap", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fh = open(args.out, "a")
    C.configure(args.precision)
    dev = C.device()
    import owl.model.kaggriculture as km
    from owl.model.stateless_transformer_v1 import _init_linear

    flash_flags: list[bool] = []
    real_use_flash = km.use_flash_attn

    def counting_use_flash(x: torch.Tensor) -> bool:
        flag = real_use_flash(x)
        flash_flags.append(bool(flag))
        return flag

    km.use_flash_attn = counting_use_flash
    C.emit(fh, C.process_record(args.case))
    model = build_model(dev, args.precision)
    trunk_calls = [0]
    if args.trunk == "compiled":
        C.compile_trunk(model)
        compiled = model._compiled_transformer_trunk

        def counting(*a: Any) -> torch.Tensor:
            trunk_calls[0] += 1
            return compiled(*a)

        model._compiled_transformer_trunk = counting
    rows = [int(r) for r in args.rows.split(",")]
    rec: dict[str, Any] = {"event": "result", "case": args.case,
                           "output_layer_spectral_norms": C.output_layer_gains(
                               model, model.critic_head.out),
                           "precision": args.precision, "trunk": args.trunk,
                           "rows": rows, "force_flash_attn": model.config.force_flash_attn,
                           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    steps: dict[str, str] = {}
    rec["steps"] = steps
    try:
        for r in rows:
            obs = make_obs(r, dev)
            steps[f"pair_{r}"] = "started"
            rec[f"default_gain_{r}"] = run_pair(model, obs, dev, args.precision, True)
            steps[f"pair_{r}"] = "ok"
            if args.hidden:
                steps[f"hidden_{r}"] = "started"
                rec[f"hidden_{r}"] = hidden_metrics(model, obs, dev, args.precision)
                steps[f"hidden_{r}"] = "ok"
            del obs
        if args.gain_swap:
            layers = model.actor.get_output_layers()
            before = [float(l.weight.detach().float().norm()) for l in layers]
            torch.manual_seed(SEED + 7)
            for layer in layers:
                _init_linear(layer, gain=1.0)
            rec["gain_swap"] = {
                "layers": len(layers),
                "weight_norm_before": before,
                "weight_norm_after": [float(l.weight.detach().float().norm())
                                      for l in layers],
                "critic_out_is_swapped": any(l is model.critic_head.out for l in layers),
            }
            for r in rows:
                obs = make_obs(r, dev)
                steps[f"gain1_pair_{r}"] = "started"
                rec[f"actor_gain1_{r}"] = run_pair(model, obs, dev, args.precision, False)
                steps[f"gain1_pair_{r}"] = "ok"
                del obs
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        raise
    finally:
        rec["use_flash_attn_flags"] = sorted(set(flash_flags))
        rec["use_flash_attn_calls"] = len(flash_flags)
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
