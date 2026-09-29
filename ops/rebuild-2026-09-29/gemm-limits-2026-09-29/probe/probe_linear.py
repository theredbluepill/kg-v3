"""Synthetic compiled-GEMM limit probe (one case per process).

Mirrors the training precision: fp32 parameters and fp32 inputs under
torch.autocast(bfloat16) with TF32 enabled, torch.compile(mode=
"max-autotune-no-cudagraphs", dynamic=True), compiled at a small M first.
For each target M: eager reference (cuBLAS) vs compiled output, element
check |d| > 0.02 + 0.02*|ref|, bad-row count/range, input/reference clobber
checks. Writes one JSON line per event to --out and flushes before each
launch so a CUDA fault is attributable to the M that was running.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch import nn

SMALL_M = 4096
ROW_CHUNK = 1 << 18


class Mlp(nn.Module):
    def __init__(self, k: int, h: int, n: int) -> None:
        super().__init__()
        self.up = nn.Linear(k, h)
        self.down = nn.Linear(h, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


def build(case: str) -> tuple[nn.Module, int, bool]:
    kind, *dims = case.split(":")
    sizes = [int(d) for d in dims]
    if kind == "lin":
        k, n = sizes
        return nn.Linear(k, n), k, False
    if kind == "mlp":
        k, h, n = sizes
        return Mlp(k, h, n), k, False
    if kind == "mlpbwd":
        k, h, n = sizes
        return Mlp(k, h, n), k, True
    raise ValueError(f"unknown case {case}")


def emit(fh, record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def compare(out: torch.Tensor, ref: torch.Tensor) -> dict:
    rows = out.shape[0]
    max_abs = 0.0
    bad_rows: list[torch.Tensor] = []
    nonfinite = 0
    for s in range(0, rows, ROW_CHUNK):
        o = out[s : s + ROW_CHUNK].float()
        r = ref[s : s + ROW_CHUNK].float()
        d = (o - r).abs()
        nonfinite += int((~torch.isfinite(o)).sum().item())
        d = torch.nan_to_num(d, nan=1e30, posinf=1e30)
        max_abs = max(max_abs, float(d.max().item()))
        bad = (d > 0.02 + 0.02 * r.abs()).any(dim=1)
        idx = bad.nonzero().flatten()
        if idx.numel():
            bad_rows.append(idx + s)
    if bad_rows:
        all_bad = torch.cat(bad_rows)
        n_bad, first, last = int(all_bad.numel()), int(all_bad[0]), int(all_bad[-1])
    else:
        n_bad, first, last = 0, None, None
    return {
        "max_abs_diff": max_abs,
        "bad_rows": n_bad,
        "first_bad_row": first,
        "last_bad_row": last,
        "nonfinite": nonfinite,
    }


def run_point(model, fn, m: int, k: int, backward: bool, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed + m)
    x = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=gen)
    x_sum = float(x.double().sum().item())
    rec: dict = {"M": m}
    if not backward:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ref = model(x)
            ref_sum = float(ref.double().sum().item())
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(x)
            torch.cuda.synchronize()
            rec["compiled_s"] = time.perf_counter() - t0
        rec["ref_clobbered"] = float(ref.double().sum().item()) != ref_sum
        rec.update(compare(out, ref))
        rec["out_dtype"] = str(out.dtype)
        del out, ref
    else:
        g = torch.randn(
            (m, model.down.out_features), device="cuda", generator=gen
        ).to(torch.bfloat16)
        grads = {}
        for tag, f in (("eager", model), ("compiled", fn)):
            model.zero_grad(set_to_none=True)
            xg = x.clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = f(xg)
            y.backward(g)
            torch.cuda.synchronize()
            rec[f"{tag}_s"] = time.perf_counter() - t0
            grads[tag] = (
                y.detach(),
                xg.grad.detach(),
                {n: p.grad.detach().clone() for n, p in model.named_parameters()},
            )
            del xg, y
        rec["fwd"] = compare(grads["compiled"][0], grads["eager"][0])
        rec["dx"] = compare(grads["compiled"][1], grads["eager"][1])
        rel = {}
        for name, ge in grads["eager"][2].items():
            gc = grads["compiled"][2][name]
            rel[name] = float(
                ((gc - ge).abs().max() / ge.abs().max().clamp_min(1e-30)).item()
            )
        rec["param_grad_rel_max"] = rel
        del grads
    rec["input_clobbered"] = float(x.double().sum().item()) != x_sum
    del x
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    model, k, backward = build(args.case)
    model = model.cuda()
    if not backward:
        model.eval()
    fn = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=True)
    with open(args.out, "a") as fh:
        emit(fh, {"event": "process", "case": args.case, "torch": torch.__version__,
                  "device": torch.cuda.get_device_name(0)})
        for label, m in [("warm", SMALL_M)] + [
            ("target", int(v)) for v in args.ms.split(",")
        ]:
            emit(fh, {"event": "start", "case": args.case, "label": label, "M": m})
            t0 = time.perf_counter()
            rec = run_point(model, fn, m, k, backward, args.seed)
            rec.update(event="result", case=args.case, label=label,
                       wall_s=time.perf_counter() - t0)
            emit(fh, rec)


if __name__ == "__main__":
    main()
