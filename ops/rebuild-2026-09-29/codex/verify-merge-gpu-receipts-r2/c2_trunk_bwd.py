"""Check 2: real-trunk forward + backward above the int32 GEMM bound, compiled vs eager.

Adapted from ops/rebuild-2026-09-29/aten-gemm-ab-2026-09-29/scripts/
probe_trunk_e1458d2.py (sha256 f700787b...), forward-only there. Unchanged:
guard bypass kaggriculture._GEMM_ELEMENT_LIMIT = 2**62, packed mask specs,
random BF16 input x [rows, T, 256], compiled small first (dense:8), eager
reference via _run_trunk with the compiled callable removed, a valid token is
wrong when any output channel has |d| > 0.25. Changes: 8fde43c model; trunk
compiled through the registered path (configure_model_compile); depth from
--depth (the preset's 8 does not fit: see the run statement); x requires grad;
loss = sum(out * g) with a fixed random BF16 g; the input gradient and every
parameter gradient are compared; a second eager run gives the eager-vs-eager
noise floor (flash-attn backward is not bitwise deterministic).

Pre-declared tolerances (judged by driver.py):
  output : 0 wrong tokens (|d| > 0.25 on any channel), 0 non-finite
  dX     : 0 non-finite; 0 valid tokens with ||d_t|| > 0.5 ||ref_t||;
           max|d| / max|ref| <= 0.05; exactly 0 at masked tokens
  params : every max|d| / max|ref| <= 0.05, 0 non-finite
Amendment 1 (attempt 2; run statement "Amendment 1"): attn.k.bias has an
analytically zero gradient (softmax is invariant to the per-query constant
q.b_k), so its relative error is judged against the sibling attn.q.bias
gradient scale instead (driver.py). Absolute magnitudes are now recorded.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

import common as C

ROW_CHUNK = 512
OUT_WRONG = 0.25
TOKEN_REL = 0.5


def make_mask(spec: str, tokens: int) -> torch.Tensor:
    kind, _, value = spec.partition(":")
    if kind == "dense":
        return torch.ones((int(value), tokens), dtype=torch.bool)
    if kind == "packed":
        total = int(value)
        rows = -(-total // tokens)
        mask = torch.ones((rows, tokens), dtype=torch.bool)
        mask[-1, total - tokens * (rows - 1):] = False
        return mask
    raise ValueError(spec)


def cmp_out(out: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    max_abs = 0.0
    wrong = nonfinite = 0
    first = last = None
    for s in range(0, out.shape[0], ROW_CHUNK):
        m = mask[s:s + ROW_CHUNK]
        o = out[s:s + ROW_CHUNK].float()
        d = (o - ref[s:s + ROW_CHUNK].float()).abs()
        nonfinite += int((~torch.isfinite(o[m])).sum())
        d = torch.nan_to_num(d, nan=1e30, posinf=1e30).amax(-1)
        d = torch.where(m, d, torch.zeros_like(d))
        max_abs = max(max_abs, float(d.max()))
        bad = d > OUT_WRONG
        wrong += int(bad.sum())
        rows_bad = bad.any(-1).nonzero().flatten()
        if rows_bad.numel():
            first = first if first is not None else int(rows_bad[0]) + s
            last = int(rows_bad[-1]) + s
    return {"max_abs": max_abs, "wrong_tokens": wrong, "nonfinite": nonfinite,
            "first_bad_row": first, "last_bad_row": last}


def cmp_dx(dx: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    max_d = max_r = 0.0
    nonfinite = masked_nonzero = bad_tokens = 0
    sum_d2 = sum_r2 = 0.0
    tok_rel_max = 0.0
    first = last = None
    rels = []
    for s in range(0, dx.shape[0], ROW_CHUNK):
        m = mask[s:s + ROW_CHUNK]
        a = dx[s:s + ROW_CHUNK].float()
        b = ref[s:s + ROW_CHUNK].float()
        nonfinite += int((~torch.isfinite(a[m])).sum())
        masked_nonzero += int((a[~m] != 0).sum()) + int((b[~m] != 0).sum())
        a = torch.nan_to_num(a, nan=1e30, posinf=1e30, neginf=-1e30)
        d = (a - b)
        dm = d[m]
        bm = b[m]
        max_d = max(max_d, float(dm.abs().max()) if dm.numel() else 0.0)
        max_r = max(max_r, float(bm.abs().max()) if bm.numel() else 0.0)
        sum_d2 += float(dm.double().pow(2).sum())
        sum_r2 += float(bm.double().pow(2).sum())
        tn = d.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)
        tn = torch.where(m, tn, torch.zeros_like(tn))
        tok_rel_max = max(tok_rel_max, float(tn.max()))
        rels.append(tn[m].float().cpu())
        bad = tn > TOKEN_REL
        bad_tokens += int(bad.sum())
        rows_bad = bad.any(-1).nonzero().flatten()
        if rows_bad.numel():
            first = first if first is not None else int(rows_bad[0]) + s
            last = int(rows_bad[-1]) + s
    allrel = torch.cat(rels)
    q = torch.quantile(allrel[torch.randperm(allrel.numel())[:1_000_000]],
                       torch.tensor([0.5, 0.99, 0.999]))
    return {"max_abs": max_d, "ref_max_abs": max_r,
            "rel_max": max_d / max_r if max_r else None,
            "rel_fro": (sum_d2 / sum_r2) ** 0.5 if sum_r2 else None,
            "token_rel_l2_max": tok_rel_max,
            "token_rel_l2_p50_p99_p999": [float(v) for v in q],
            "tokens_rel_gt_0.5": bad_tokens, "nonfinite": nonfinite,
            "masked_nonzero": masked_nonzero, "first_bad_row": first,
            "last_bad_row": last}


def cmp_params(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    out = {}
    for name, ref in b.items():
        g = a[name].float()
        r = ref.float()
        d = (g - r).abs()
        out[name] = {
            "max_abs_diff": float(d.max()),
            "ref_max_abs": float(r.abs().max()),
            "out_max_abs": float(g.abs().max()),
            "rel_max": float(d.max() / r.abs().max().clamp_min(1e-30)),
            "rel_fro": float((g - r).norm() / r.norm().clamp_min(1e-30)),
            "nonfinite": int((~torch.isfinite(g)).sum()),
        }
    missing = sorted(set(b) ^ set(a))
    if missing:
        out["_missing"] = missing
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="';'-separated mask specs")
    ap.add_argument("--bypass-guard", action="store_true")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fh = open(args.out, "a")
    C.configure()
    dev = C.device()
    import owl.model.kaggriculture as km

    if args.bypass_guard:
        km._GEMM_ELEMENT_LIMIT = 2**62
    C.emit(fh, {**C.process_record(args.case), "depth": args.depth,
                "bypass_guard": args.bypass_guard})
    model = C.build_model(dev, depth=args.depth).train()
    tokens = km.sequence_length(model.config)
    width = km.trunk_gemm_width(model.config)
    C.compile_trunk(model)
    compiled = model._compiled_transformer_trunk
    calls: list[int] = []

    def recording(x: torch.Tensor, token_mask: Any, packed: Any) -> torch.Tensor:
        calls.append(int(x.shape[0]) if x.dim() == 2 else int(x.shape[0] * x.shape[1]))
        return compiled(x, token_mask, packed)

    def run(x: torch.Tensor, g: torch.Tensor, mask: torch.Tensor,
            use_compiled: bool) -> dict[str, Any]:
        model.zero_grad(set_to_none=True)
        x.grad = None
        calls.clear()
        model._compiled_transformer_trunk = recording if use_compiled else None
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        C.sync(dev)
        t0 = time.perf_counter()
        with C.amp(dev):
            out = model._run_trunk(x, mask)
        loss = (out * g).sum(dtype=torch.float32)
        loss.backward()
        C.sync(dev)
        res = {
            "s": time.perf_counter() - t0,
            "loss": float(loss),
            "out": out.detach(),
            "dx": x.grad,
            "params": {n: p.grad.detach().clone()
                       for n, p in model.named_parameters() if p.grad is not None},
            "calls": list(calls),
            "memory": C.mem(dev),
        }
        x.grad = None
        model.zero_grad(set_to_none=True)
        model._compiled_transformer_trunk = compiled
        return res

    specs = ["dense:8"] + [p for p in args.points.split(";") if p]
    for i, spec in enumerate(specs):
        label = "warm" if i == 0 else "target"
        mask_cpu = make_mask(spec, tokens)
        rows = mask_cpu.shape[0]
        packed_tokens = int(mask_cpu.sum())
        C.emit(fh, {"event": "start", "case": args.case, "label": label,
                    "spec": spec, "rows": rows, "packed_tokens": packed_tokens})
        mask = mask_cpu.to(dev)
        gen = torch.Generator(device=dev).manual_seed(i)
        x = torch.randn((rows, tokens, model.config.embed_dim), device=dev,
                        dtype=torch.bfloat16, generator=gen).requires_grad_(True)
        g = torch.randn((rows, tokens, model.config.embed_dim), device=dev,
                        dtype=torch.bfloat16, generator=gen)
        rec: dict[str, Any] = {
            "event": "result", "case": args.case, "label": label, "spec": spec,
            "rows": rows, "packed_tokens": packed_tokens,
            "packed_tokens_x_width_over_2p31": packed_tokens * width / 2**31,
        }
        stage = "compiled"
        try:
            comp = run(x, g, mask, True)
            rec["compiled"] = {"status": "ok", "s": comp["s"], "loss": comp["loss"],
                               "trunk_call_M": comp["calls"], "memory": comp["memory"],
                               "param_grads": len(comp["params"])}
            stage = "eager"
            eag = run(x, g, mask, False)
            rec["eager"] = {"status": "ok", "s": eag["s"], "loss": eag["loss"],
                            "trunk_call_M": eag["calls"], "memory": eag["memory"]}
            rec["out_compiled_vs_eager"] = cmp_out(comp["out"], eag["out"], mask)
            rec["dx_compiled_vs_eager"] = cmp_dx(comp["dx"], eag["dx"], mask)
            rec["params_compiled_vs_eager"] = cmp_params(comp["params"], eag["params"])
            del comp
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            stage = "eager2"
            eag2 = run(x, g, mask, False)
            rec["floor_out_eager2_vs_eager"] = cmp_out(eag2["out"], eag["out"], mask)
            rec["floor_dx_eager2_vs_eager"] = cmp_dx(eag2["dx"], eag["dx"], mask)
            rec["floor_params_eager2_vs_eager"] = cmp_params(eag2["params"],
                                                             eag["params"])
            del eag, eag2
        except (ValueError, RuntimeError) as exc:
            rec[stage] = {"status": f"{type(exc).__name__}: {exc}"[:2000]}
        C.emit(fh, rec)
        del x, g, mask
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    C.emit(fh, {"event": "done", "case": args.case})
    fh.close()


if __name__ == "__main__":
    main()
