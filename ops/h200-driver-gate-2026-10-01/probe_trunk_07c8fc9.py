"""Real KaggricultureTransformer trunk correctness probe at 07c8fc9 (one process).

H200 re-probe (2026-10-01, run statement run-statement.md beside this file) of
probe_trunk_e1458d2.py (sha256 f700787b...). Adaptations to 07c8fc9, and nothing
else: ROOT is the pod checkout /root/kg-v3; since 07c8fc9's
compile_transformer_trunk claims the GEMM backends and checks the probed stack
(which rejects driver 570.211.01, the question under test), the probe replaces
owl.model.compile_gemm.check_compile_stack with a recorder that returns the
installed versions unchecked. The claim still sets "ATEN" and
require_compiled_gemm_backends still runs on every compiled trunk call; the
process record carries the claim, the real check's error, and the backends
before and after the run. The --stack-check-bypass flag is required so the
bypass cannot happen silently.

Original docstring follows.

Real KaggricultureTransformer trunk correctness probe at e1458d2 (one process).

Adapted from ops/rebuild-2026-09-29/gemm-limits-2026-09-29/probe/probe_trunk.py
(sha256 5ce3f09a...) for the ATEN-only GEMM A/B
(run statement ops/rebuild-2026-09-29/run-statements/aten-gemm-ab.md).
Differences from the original, and nothing else:

- the model is imported from the pod checkout /workspace/kg-v3-rebuild at
  e1458d2 (installed venv), not from the 1ddc71d source copy;
- the preset config configs/model/kaggriculture.yaml is loaded
  (force_flash_attn: true) and the REAL flash-attn 2.8.3 varlen kernel is used;
  the original's aten varlen shim is not installed (flash-attn is now present);
- only the packed path is supported; the helper names follow e1458d2
  (trunk_gemm_width replaces gemm_kmax);
- the process record carries torch._inductor.config.max_autotune_gemm_backends.

Unchanged: fp32 parameters, bf16 input x [rows, 709, 256] under
torch.autocast(bfloat16), TF32 on, compile_transformer_trunk(mode=
"max-autotune-no-cudagraphs") (dynamic=True), compiled small first (dense:8),
eager reference via _run_trunk with the compiled callable removed, the guard
bypass (kaggriculture._GEMM_ELEMENT_LIMIT = 2**62), the mask specs and the
comparison (|d| > 0.25 on any channel of a valid token counts as wrong).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

import owl
import owl.model.attn as attn_mod

ROOT = Path("/root/kg-v3")
TOKENS = 709
ROW_CHUNK = 512


def emit(fh, record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def make_mask(spec: str) -> torch.Tensor:
    kind, _, value = spec.partition(":")
    if kind == "dense":  # every actor slot and shop live: 709 tokens per row
        return torch.ones((int(value), TOKENS), dtype=torch.bool)
    if kind == "packed":  # dense rows, total valid tokens exactly value
        total = int(value)
        rows = -(-total // TOKENS)
        mask = torch.ones((rows, TOKENS), dtype=torch.bool)
        last = total - TOKENS * (rows - 1)
        mask[-1, last:] = False
        return mask
    raise ValueError(spec)


def compare(out: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> dict:
    max_abs = 0.0
    bad25 = bad100 = nonfinite = 0
    first_bad_row = last_bad_row = None
    for s in range(0, out.shape[0], ROW_CHUNK):
        m = mask[s:s + ROW_CHUNK]
        d = (out[s:s + ROW_CHUNK].float() - ref[s:s + ROW_CHUNK].float()).abs()
        nonfinite += int((~torch.isfinite(out[s:s + ROW_CHUNK][m])).sum())
        d = torch.nan_to_num(d, nan=1e30, posinf=1e30).amax(-1)  # per token
        d = torch.where(m, d, torch.zeros_like(d))
        max_abs = max(max_abs, float(d.max()))
        bad25 += int((d > 0.25).sum())
        bad100 += int((d > 1.0).sum())
        rows_bad = (d > 0.25).any(-1).nonzero().flatten()
        if rows_bad.numel():
            if first_bad_row is None:
                first_bad_row = int(rows_bad[0]) + s
            last_bad_row = int(rows_bad[-1]) + s
    return {"max_abs_diff": max_abs, "tokens_diff_gt_0.25": bad25,
            "tokens_diff_gt_1.0": bad100, "nonfinite": nonfinite,
            "first_bad_row": first_bad_row, "last_bad_row": last_bad_row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="';'-separated mask specs")
    ap.add_argument("--bypass-guard", action="store_true")
    ap.add_argument("--stack-check-bypass", action="store_true", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    import owl.model.kaggriculture as kg
    from owl.kaggriculture import types as kt

    if not kg.__file__.startswith(str(ROOT / "python")):
        raise RuntimeError(f"model imported from the wrong tree: {kg.__file__}")
    if not attn_mod.flash_attn_available():
        raise RuntimeError("flash-attn is not importable; the packed path needs it")

    import owl.model.compile_gemm as cg

    installed = cg.installed_compile_stack()
    try:
        cg.check_compile_stack(installed)
        real_check = "passed"
    except RuntimeError as exc:
        real_check = f"RuntimeError: {exc}"

    def unchecked_stack(inst, probed=cg.KAGGRICULTURE_PROBED_COMPILE_STACK):
        return cg.CompileStackReport(
            torch=inst.torch, triton=str(inst.triton),
            nvidia_driver="UNCHECKED-PROBE:" + ",".join(inst.nvidia_drivers or ()))

    cg.check_compile_stack = unchecked_stack
    fh = open(args.out, "a")
    if args.bypass_guard:
        kg._GEMM_ELEMENT_LIMIT = 2**62
    cfg = kg.KaggricultureTransformerConfig.from_file(
        ROOT / "configs/model/kaggriculture.yaml")
    assert cfg.force_flash_attn is True
    assert kg.sequence_length(cfg) == TOKENS
    torch.manual_seed(0)
    model = kg.KaggricultureTransformer(
        cfg, obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).cuda().eval()
    model.compile_transformer_trunk(mode="max-autotune-no-cudagraphs")
    compiled = model._compiled_transformer_trunk
    calls: list[int] = []

    def recording(x, token_mask, packed):
        calls.append(int(x.shape[0]) if x.dim() == 2 else int(x.shape[0] * x.shape[1]))
        return compiled(x, token_mask, packed)

    width = kg.trunk_gemm_width(cfg)
    emit(fh, {"event": "process", "case": args.case, "torch": torch.__version__,
              "owl_file": owl.__file__, "kg_file": kg.__file__,
              "flash_attn_varlen_func": repr(attn_mod._FLASH_ATTN_VARLEN_FUNC),
              "max_autotune_gemm_backends": inductor_config.max_autotune_gemm_backends,
              "gemm_limit": kg._GEMM_ELEMENT_LIMIT,
              "trunk_gemm_width": width,
              "use_flash_attn_bf16": attn_mod.use_flash_attn(
                  torch.empty(1, device="cuda", dtype=torch.bfloat16)),
              "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability(0)),
              "installed_stack": repr(installed),
              "real_stack_check": real_check,
              "claim": repr(cg.gemm_backend_claim()),
              "backends_after_claim": inductor_config.max_autotune_gemm_backends})
    specs = ["dense:8"] + [p for p in args.points.split(";") if p]
    for i, spec in enumerate(specs):
        label = "warm" if i == 0 else "target"
        mask_cpu = make_mask(spec)
        rows = mask_cpu.shape[0]
        emit(fh, {"event": "start", "case": args.case, "label": label, "spec": spec,
                  "rows": rows, "packed_tokens": int(mask_cpu.sum())})
        mask = mask_cpu.cuda()
        x = torch.randn((rows, TOKENS, cfg.embed_dim), device="cuda",
                        dtype=torch.bfloat16,
                        generator=torch.Generator(device="cuda").manual_seed(i))
        rec: dict = {"event": "result", "case": args.case, "label": label,
                     "spec": spec, "rows": rows,
                     "packed_tokens": int(mask_cpu.sum()),
                     "padded_token_rows": rows * TOKENS}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model._compiled_transformer_trunk = None
            try:
                ref = model._run_trunk(x, mask)
                rec["eager"] = "ok"
            except ValueError as exc:
                ref = None
                rec["eager"] = f"ValueError: {exc}"
            calls.clear()
            model._compiled_transformer_trunk = recording
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                out = model._run_trunk(x, mask)
                torch.cuda.synchronize()
                rec["compiled"] = "ok"
            except ValueError as exc:
                out = None
                rec["compiled"] = f"ValueError: {exc}"
            rec["compiled_s"] = time.perf_counter() - t0
            rec["trunk_call_M"] = list(calls)
            rec["trunk_call_M_x_width_over_2p31"] = [c * width / 2**31 for c in calls]
        if out is not None and ref is not None:
            rec.update(compare(out, ref, mask))
        emit(fh, rec)
        del x, mask, ref, out
        torch.cuda.empty_cache()
    emit(fh, {"event": "process_end", "case": args.case,
              "max_autotune_gemm_backends": inductor_config.max_autotune_gemm_backends,
              "claim": repr(cg.gemm_backend_claim())})
    fh.close()


if __name__ == "__main__":
    main()
