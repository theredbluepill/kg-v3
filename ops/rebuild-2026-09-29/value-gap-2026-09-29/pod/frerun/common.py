"""Shared helpers for the value-gap diagnostic (run statement
ops/rebuild-2026-09-29/run-statements/value-gap-diagnostic.md).

Source under test: kg/rebuild-trainer-model @ 8fde43c, checked out detached in
the pod checkout ROOT (unchanged by this run). Adapted from the GPU checks
bundle's common.py (kg/rebuild-gpu-checks, 5f2ee2d): same model construction,
same registered trunk-compile path, same make_obs observations.

VGAP_DRYRUN=1 is a local CPU code-path check only (tiny shapes, no compile, no
flash); it produces no evidence and is never used on the pod.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(os.environ.get("VGAP_ROOT", "/workspace/kg-v3-rebuild"))
DRYRUN = os.environ.get("VGAP_DRYRUN") == "1"
EXPECTED_HEAD = "8fde43cd7408c9c4f9147eeb8f916bd08f8266ed"
COMPILE_MODE = "max-autotune-no-cudagraphs"
sys.path.insert(0, str(ROOT))
GIB = 1024**3


def device() -> torch.device:
    if DRYRUN:
        return torch.device("cpu")
    assert torch.cuda.is_available(), "CUDA is required"
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    return dev


def configure(precision: str) -> None:
    """configure_torch() as run_ppo does; fp32 cases then turn TF32 off."""
    from owl.train.utils import configure_torch

    configure_torch()
    if precision == "fp32":
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    elif precision != "bf16":
        raise ValueError(precision)


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: object) -> None:
        return None


def amp(dev: torch.device, precision: str) -> Any:
    if precision == "bf16":
        return torch.autocast(dev.type, dtype=torch.bfloat16)
    return _Null()


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def emit(fh: Any, record: dict[str, Any]) -> None:
    record = {"unix": round(time.time(), 3), **record}
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    print(json.dumps(record)[:1500], flush=True)


def git_head() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    if DRYRUN:
        return {"head": "dryrun", "tree": "dryrun", "porcelain": ""}
    return {
        "head": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "porcelain": run("status", "--porcelain"),
    }


def process_record(case: str) -> dict[str, Any]:
    import torch._inductor.config as inductor_config

    import owl.model.attn as attn_mod
    import owl.model.stateless_transformer_v1 as st

    rec: dict[str, Any] = {
        "event": "process",
        "case": case,
        "dryrun": DRYRUN,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_runtime": torch.version.cuda,
        "model_file": st.__file__,
        "flash_attn_available": attn_mod.flash_attn_available(),
        "max_autotune_gemm_backends": inductor_config.max_autotune_gemm_backends,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "actor_head_init_gain_src": st._ACTOR_HEAD_INIT_GAIN,
        "critic_head_init_gain_src": st._CRITIC_HEAD_INIT_GAIN,
        "git": git_head(),
    }
    if not DRYRUN:
        import flash_attn
        import triton

        rec.update(
            flash_attn=flash_attn.__version__,
            triton=triton.__version__,
            device_name=torch.cuda.get_device_name(0),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if not st.__file__.startswith(str(ROOT / "python")):
            raise RuntimeError(f"model imported from the wrong tree: {st.__file__}")
        if rec["git"]["head"] != EXPECTED_HEAD:
            raise RuntimeError(f"pod HEAD {rec['git']['head']} != {EXPECTED_HEAD}")
        if not attn_mod.flash_attn_available():
            raise RuntimeError("flash-attn is not importable")
    return rec


class _CompileCfg:
    model_compile = "trunk"
    model_compile_mode = COMPILE_MODE


def compile_trunk(model: Any) -> int:
    """The model's registered compile path, as run_ppo reaches it."""
    if DRYRUN:  # eager stand-in so the call sites run; no evidence
        model._compiled_transformer_trunk = model._forward_transformer_trunk
        return 0
    from owl.train.utils import configure_model_compile

    n = configure_model_compile(model, _CompileCfg())
    if n != 1 or model._compiled_transformer_trunk is None:
        raise RuntimeError(f"registered trunk compile returned {n}")
    return n


def _subsample(d: torch.Tensor) -> torch.Tensor:
    """Every k-th element so torch.quantile's input-size limit is respected."""
    flat = d.flatten()
    step = -(-flat.numel() // 2**24)
    return flat[::step]


def diff(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    """|a - b| statistics (b is the reference), optionally over mask."""
    a = a.detach().float()
    b = b.detach().float()
    if mask is not None:
        a = a[mask]
        b = b[mask]
    d = (a - b).abs()
    n = d.numel()
    if n == 0:
        return {"n": 0}
    ref = b.abs()
    return {
        "n": n,
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rms": float(d.pow(2).mean().sqrt()),
        "p99_abs": float(torch.quantile(_subsample(d), 0.99)),
        "exact_equal_frac": float((d == 0).float().mean()),
        "ref_max_abs": float(ref.max()),
        "ref_mean_abs": float(ref.mean()),
        "rel_mean": float(d.mean() / ref.mean().clamp_min(1e-30)),
        "rel_max": float(d.max() / ref.max().clamp_min(1e-30)),
        "nonfinite": int((~torch.isfinite(a)).sum() + (~torch.isfinite(b)).sum()),
    }


def stats(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().float()
    return {"mean": float(t.mean()), "abs_mean": float(t.abs().mean()),
            "abs_max": float(t.abs().max()), "min": float(t.min()),
            "max": float(t.max()), "nonfinite": int((~torch.isfinite(t)).sum()),
            "n": t.numel()}


def output_layer_gains(model: Any, critic_out: Any) -> dict[str, Any]:
    """Spectral norm of every output layer (= the orthogonal-init gain)."""
    actor, critic = [], []
    for layer in model.get_output_layers():
        norm = float(torch.linalg.matrix_norm(layer.weight.detach().float(), ord=2))
        (critic if layer is critic_out else actor).append(round(norm, 6))
    return {"critic": critic, "actor": actor}


def mem(dev: torch.device) -> dict[str, float]:
    if dev.type != "cuda":
        return {}
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated(dev) / GIB,
        "max_reserved_gib": torch.cuda.max_memory_reserved(dev) / GIB,
    }
