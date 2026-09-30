"""Plan 6.3b setup gate: flash-attn's varlen kernel runs on every GPU (sm_120).

Run on the pod from the repository root with the project venv:
    uv run --no-sync python ops/rebuild-2026-09-29/8rank-run/check_flash_all_gpus.py \
        --expect-gpus 8

For each visible GPU it checks the compute capability (12.0 for RTX PRO 6000
Blackwell), that ``owl.model.attn`` finds flash-attn, and that
``flash_attn_varlen_func`` on packed BF16 q/k/v (8 heads, head dim 32, the
model's shape) agrees with per-sequence fp32 SDPA within ``0.02 + 0.02|ref|``
on every element. It also lists the extension's cubins with ``cuobjdump`` when
the tool exists. It prints one JSON object and exits nonzero on any failure.
The Phase 6.0 smoke (``ops/rebuild-2026-09-29/flash-attn-setup-2026-09-29/``)
covers the model's packed path on one GPU; this gate covers every GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from owl.model.attn import flash_attn_available

HEADS = 8
HEAD_DIM = 32
SEQ_LENS = (214, 709, 377, 1, 512, 64)


def _cubins() -> dict[str, Any]:
    import flash_attn_2_cuda  # type: ignore[import-not-found]

    so = Path(flash_attn_2_cuda.__file__)
    tool = shutil.which("cuobjdump")
    if tool is None:
        return {"so": str(so), "cuobjdump": "absent", "sm_120": None}
    listing = subprocess.run(
        [tool, "--list-elf", str(so)], capture_output=True, text=True, check=False
    ).stdout
    return {"so": str(so), "cuobjdump": tool, "sm_120": "sm_120" in listing}


def _check_device(index: int) -> dict[str, Any]:
    from flash_attn import flash_attn_varlen_func

    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    generator = torch.Generator(device=device).manual_seed(index)
    total = sum(SEQ_LENS)
    q, k, v = (
        torch.randn(total, HEADS, HEAD_DIM, device=device, generator=generator).to(
            torch.bfloat16
        )
        for _ in range(3)
    )
    offsets = [0]
    for length in SEQ_LENS:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device=device, dtype=torch.int32)
    out = flash_attn_varlen_func(
        q, k, v, cu, cu, max(SEQ_LENS), max(SEQ_LENS), causal=False
    )
    torch.cuda.synchronize(device)
    worst = 0.0
    outside = 0
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        ref = F.scaled_dot_product_attention(
            *(t[start:end].float().transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        )[0].transpose(0, 1)
        got = out[start:end].float()
        diff = (got - ref).abs()
        worst = max(worst, float(diff.max()))
        outside += int((diff > 0.02 + 0.02 * ref.abs()).sum())
    finite = bool(torch.isfinite(out).all())
    return {
        "gpu": index,
        "name": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "max_abs_diff": worst,
        "elements_outside_tolerance": outside,
        "finite": finite,
        "ok": capability == (12, 0) and finite and outside == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-gpus", type=int, required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "flash_attn_available": flash_attn_available(),
    }
    ok = (
        report["cuda_available"]
        and report["device_count"] == args.expect_gpus
        and report["flash_attn_available"]
    )
    if ok:
        report["cubins"] = _cubins()
        report["devices"] = [_check_device(i) for i in range(torch.cuda.device_count())]
        ok = all(d["ok"] for d in report["devices"]) and report["cubins"]["sm_120"] in (
            True,
            None,
        )
    report["ok"] = bool(ok)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
