"""Check 1: compiled / eager / padded-SDPA BF16 trunk outputs vs an fp32 reference.

Question: are the 0.017-0.021 % out-of-tolerance trunk elements of results.md
"Phase 6.0" (b) BF16 rounding common to all paths, or specific to one path?

Per density (mid, dense, mixed; 128 envs = 256 seat rows), same weights, same
input: x = model._assemble_tokens(obs) in fp32, rounded once to BF16 (x_bf);
every path consumes x_bf (the reference consumes x_bf.float()), so input
rounding is common and excluded.
  ref     : _forward_transformer_trunk(x_bf.float(), mask, None), autocast OFF,
            TF32 OFF, padded SDPA forced to the MATH backend, 32-row chunks.
  ref_eff : same but SDPA backend auto-selected (fp32 reference noise floor).
  compiled: _run_trunk(x_bf, mask) under BF16 autocast, compiled trunk
            (registered path), packed flash.
  eager   : same with the compiled callable removed (packed flash, eager).
  padded  : _forward_transformer_trunk(x_bf, mask, None) under BF16 autocast,
            eager, padded SDPA (backend auto-selected).
Metrics on present tokens only (token_mask). Tolerance as Phase 6.0:
|d| <= 0.02 + 0.02 |ref|. Retains the 32 worst elements per path and every
outlier coordinate (row, token, channel) per path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import common as C

ATOL, RTOL = 0.02, 0.02
WORST = 32
REF_BUCKETS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf"))


def bf16_ulp(ref: torch.Tensor) -> torch.Tensor:
    """BF16 spacing at |ref| (8 significant bits): 2**(floor(log2|ref|) - 7)."""
    a = ref.abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def compare(
    out: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, *, retain: bool
) -> dict[str, Any]:
    """|out - ref| over present tokens; out/ref [rows, T, D]."""
    o = out.float()
    m3 = mask[..., None].expand_as(ref)
    d = (o - ref).abs()
    finite = torch.isfinite(o) | ~m3
    dm = torch.where(m3, d, torch.zeros_like(d))
    tol = ATOL + RTOL * ref.abs()
    outlier = (dm > tol) & m3
    n = int(m3.sum())
    rec: dict[str, Any] = {
        "elements": n,
        "nonfinite": int((~finite).sum()),
        "max_abs": float(dm.max()),
        "mean_abs": float(dm.sum() / n),
        "rms": float((dm.pow(2).sum() / n).sqrt()),
        "outside_tol": int(outlier.sum()),
        "outside_tol_frac": float(outlier.sum()) / n,
    }
    if not retain:
        return rec
    ulp = bf16_ulp(ref)
    in_ulp = torch.where(m3, dm / ulp, torch.zeros_like(dm))
    rec["mean_err_in_bf16_ulp_of_ref"] = float(in_ulp.sum() / n)
    rec["frac_err_gt_1_ulp"] = float(((in_ulp > 1) & m3).sum()) / n
    rec["frac_err_gt_4_ulp"] = float(((in_ulp > 4) & m3).sum()) / n
    rec["ref_abs_max"] = float(torch.where(m3, ref.abs(), 0).max())
    # outliers and elements per |ref| bucket
    buckets = []
    ra = ref.abs()
    for lo, hi in zip(REF_BUCKETS[:-1], REF_BUCKETS[1:], strict=True):
        sel = m3 & (ra >= lo) & (ra < hi)
        ne = int(sel.sum())
        buckets.append({
            "ref_abs": [lo, hi],
            "elements": ne,
            "outliers": int((outlier & sel).sum()),
            "mean_abs": float(dm[sel].mean()) if ne else None,
        })
    rec["by_ref_abs"] = buckets
    # outliers and elements per token group
    groups = []
    for name, lo, hi in C.token_groups():
        sel = m3[:, lo:hi]
        ne = int(sel.sum())
        groups.append({
            "group": name,
            "elements": ne,
            "outliers": int(outlier[:, lo:hi].sum()),
            "mean_abs": float(dm[:, lo:hi][sel].mean()) if ne else None,
        })
    rec["by_token_group"] = groups
    per_channel = outlier.sum(dim=(0, 1))
    top = torch.topk(per_channel, 8)
    rec["top_outlier_channels"] = [
        [int(c), int(v)] for v, c in zip(top.values, top.indices, strict=True)
    ]
    per_row = outlier.sum(dim=(1, 2))
    rec["rows_with_outliers"] = int((per_row > 0).sum())
    rec["max_outliers_in_one_row"] = int(per_row.max())
    # worst elements
    flat = dm.flatten()
    worst = torch.topk(flat, WORST)
    rows_len = mask.sum(dim=1)
    T, D = ref.shape[1], ref.shape[2]
    rec["worst"] = []
    for v, i in zip(worst.values.tolist(), worst.indices.tolist(), strict=True):
        r, rem = divmod(i, T * D)
        t, c = divmod(rem, D)
        rv = float(ref[r, t, c])
        rec["worst"].append({
            "row": r, "token": t, "channel": c, "group": C.group_of(t),
            "row_tokens": int(rows_len[r]), "ref": rv, "out": float(o[r, t, c]),
            "abs_diff": v, "rel_diff": v / max(abs(rv), 1e-12),
            "ulps": v / float(bf16_ulp(torch.tensor(rv))),
        })
    coords = outlier.nonzero()[:50_000]  # capped; the count above is exact
    rec["outlier_coords_retained"] = int(coords.shape[0])
    rec["_outlier_coords"] = coords.cpu().tolist()
    return rec


def jaccard(a: list[list[int]], b: list[list[int]]) -> dict[str, Any]:
    sa, sb = {tuple(x) for x in a}, {tuple(x) for x in b}
    inter = len(sa & sb)
    union = len(sa | sb)
    return {"a": len(sa), "b": len(sb), "intersection": inter,
            "jaccard": inter / union if union else None}


def verdict(paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pre-declared rule: a path is an outlier when its mean |d| or its
    outside-tol fraction vs fp32 exceeds 2x the median of the three paths on
    that metric; otherwise the three are 'similar' (supports BF16 rounding)."""
    out: dict[str, Any] = {"rule": "outlier if metric > 2 x median(3 paths)"}
    flagged = []
    for metric in ("mean_abs", "outside_tol_frac", "max_abs"):
        vals = {p: paths[p][metric] for p in paths}
        med = sorted(vals.values())[1]
        out[metric] = {"values": vals, "median": med,
                       "ratio_to_median": {p: (v / med if med else None)
                                           for p, v in vals.items()}}
        if metric != "max_abs":
            flagged += [f"{p}:{metric}" for p, v in vals.items() if med and v > 2 * med]
    out["flagged"] = flagged
    out["result"] = "path_specific" if flagged else "similar"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", default="mid,dense,mixed")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=2 if C.DRYRUN else 128)
    args = ap.parse_args()
    out_path = Path(args.out)
    fh = open(out_path, "a")
    C.configure()
    dev = C.device()
    import owl.model.kaggriculture as km

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
    model = C.build_model(dev).eval()
    C.compile_trunk(model)
    compiled_fn = model._compiled_transformer_trunk
    retained: dict[str, Any] = {}

    for density in args.densities.split(","):
        obs = C.to_dev(C.make_obs_batch(args.envs, density), dev)
        with torch.no_grad():
            x32, mask = model._assemble_tokens(obs)
            x_bf = x32.to(torch.bfloat16)
            x_in = x_bf.float()
            rows = x_bf.shape[0]
            rec: dict[str, Any] = {
                "event": "result", "case": args.case, "density": density,
                "rows": rows, "padded_len": int(x_bf.shape[1]),
                "present_tokens": int(mask.sum()),
                "row_tokens_min": int(mask.sum(1).min()),
                "row_tokens_max": int(mask.sum(1).max()),
            }
            # fp32 references: autocast off, TF32 off.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model._compiled_transformer_trunk = None
            ref = torch.empty_like(x_in)
            ref_eff = torch.empty_like(x_in)
            with sdpa_kernel(SDPBackend.MATH):
                for s in range(0, rows, 32):
                    ref[s:s + 32] = model._forward_transformer_trunk(
                        x_in[s:s + 32], mask[s:s + 32], None)
            for s in range(0, rows, 32):
                ref_eff[s:s + 32] = model._forward_transformer_trunk(
                    x_in[s:s + 32], mask[s:s + 32], None)
            rec["ref_dtype"] = str(ref.dtype)
            rec["ref_tf32_during"] = [torch.backends.cuda.matmul.allow_tf32,
                                      torch.backends.cudnn.allow_tf32]
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            rec["ref_nonfinite"] = int((~torch.isfinite(ref[mask])).sum())
            rec["ref_eff_vs_ref_math"] = compare(ref_eff, ref, mask, retain=False)
            del ref_eff
            # BF16 paths.
            outs: dict[str, torch.Tensor] = {}
            with C.amp(dev):
                model._compiled_transformer_trunk = compiled_fn
                model._run_trunk(x_bf[:8], mask[:8])  # compiled small first
                counters["pack"].clear(); counters["flash"].clear()
                outs["compiled"] = model._run_trunk(x_bf, mask)
                rec["compiled_pack_calls"] = list(counters["pack"])
                rec["compiled_use_flash"] = list(counters["flash"])
                again = model._run_trunk(x_bf, mask)
                rec["compiled_repeat_max_abs"] = float(
                    (again.float() - outs["compiled"].float()).abs().max())
                del again
                model._compiled_transformer_trunk = None
                counters["pack"].clear(); counters["flash"].clear()
                outs["eager"] = model._run_trunk(x_bf, mask)
                rec["eager_pack_calls"] = list(counters["pack"])
                rec["eager_use_flash"] = list(counters["flash"])
                outs["padded"] = model._forward_transformer_trunk(x_bf, mask, None)
                model._compiled_transformer_trunk = compiled_fn
            rec["out_dtypes"] = {k: str(v.dtype) for k, v in outs.items()}
            paths = {}
            for name, out in outs.items():
                paths[name] = compare(out, ref, mask, retain=True)
                retained[f"{density}/{name}"] = paths[name].pop("_outlier_coords")
            rec["vs_fp32"] = paths
            rec["pairwise"] = {
                "compiled_vs_eager": compare(outs["compiled"], outs["eager"].float(),
                                             mask, retain=False),
                "eager_vs_padded": compare(outs["eager"], outs["padded"].float(),
                                           mask, retain=False),
                "compiled_vs_padded": compare(outs["compiled"],
                                              outs["padded"].float(), mask,
                                              retain=False),
            }
            rec["outlier_overlap"] = {
                f"{a}&{b}": jaccard(retained[f"{density}/{a}"],
                                    retained[f"{density}/{b}"])
                for a, b in (("compiled", "eager"), ("eager", "padded"),
                             ("compiled", "padded"))
            }
            rec["verdict"] = verdict(paths)
            rec["memory"] = C.mem(dev)
        C.emit(fh, rec)
        del outs, ref, x32, x_bf, x_in, mask, obs
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    coords_path = out_path.with_suffix(".outliers.json")
    coords_path.write_text(json.dumps(
        {"format": "[row, token, channel] per density/path", **retained}))
    C.emit(fh, {"event": "done", "case": args.case,
                "outlier_file": str(coords_path)})
    fh.close()


if __name__ == "__main__":
    main()
