"""Shared helpers for the GPU checks bundle (run statement
ops/rebuild-2026-09-29/run-statements/gpu-checks-bundle.md).

Source under test: kg/rebuild-trainer-model @ 8fde43c, checked out detached in
the pod checkout ROOT. Every check builds the preset model
(configs/model/kaggriculture.yaml, force_flash_attn true), keeps fp32
parameters, runs BF16 autocast with TF32 on (owl.train.utils.configure_torch)
and compiles the trunk through the model's registered path
(owl.train.utils.configure_model_compile, model_compile="trunk",
mode max-autotune-no-cudagraphs -> compile_transformer_trunk, dynamic=True).
Observations are tests/kaggriculture/conftest.py::make_obs.

GPUCHK_DRYRUN=1 is a local CPU code-path check only (tiny shapes, no compile,
no flash); it produces no evidence and is never used on the pod.
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
from pydantic import BaseModel

ROOT = Path(os.environ.get("GPUCHK_ROOT", "/workspace/kg-v3-rebuild"))
DRYRUN = os.environ.get("GPUCHK_DRYRUN") == "1"
EXPECTED_HEAD = "8fde43cd7408c9c4f9147eeb8f916bd08f8266ed"
COMPILE_MODE = "max-autotune-no-cudagraphs"
sys.path.insert(0, str(ROOT))

# own actors, rival actors, shops per env (bench_model_sps.py DENSITIES)
DENSITIES: dict[str, dict[str, Any]] = {
    "mid": {"own": 40, "rival": 40, "shops": 4},
    "dense": {"own": 241, "rival": 241, "shops": 8},
}
GIB = 1024**3


def mixed_counts(envs: int) -> dict[str, list[int]]:
    """Variable row lengths: own = rival actors 1..241 across envs, shops 1..8."""
    own = [1 + (240 * i) // max(envs - 1, 1) for i in range(envs)]
    return {"own": own, "rival": own, "shops": [1 + i % 8 for i in range(envs)]}


def device() -> torch.device:
    if DRYRUN:
        return torch.device("cpu")
    assert torch.cuda.is_available(), "CUDA is required"
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    return dev


def amp(dev: torch.device) -> Any:
    return torch.autocast(dev.type, dtype=torch.bfloat16)


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def emit(fh: Any, record: dict[str, Any]) -> None:
    record = {"unix": round(time.time(), 3), **record}
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    print(json.dumps(record)[:2000], flush=True)


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
    import owl.model.kaggriculture as km

    rec: dict[str, Any] = {
        "event": "process",
        "case": case,
        "dryrun": DRYRUN,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_runtime": torch.version.cuda,
        "kg_file": km.__file__,
        "flash_attn_available": attn_mod.flash_attn_available(),
        "max_autotune_gemm_backends": inductor_config.max_autotune_gemm_backends,
        "gemm_limit": km._GEMM_ELEMENT_LIMIT,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
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
        if not km.__file__.startswith(str(ROOT / "python")):
            raise RuntimeError(f"model imported from the wrong tree: {km.__file__}")
        if rec["git"]["head"] != EXPECTED_HEAD:
            raise RuntimeError(f"pod HEAD {rec['git']['head']} != {EXPECTED_HEAD}")
        if not attn_mod.flash_attn_available():
            raise RuntimeError("flash-attn is not importable")
    return rec


def configure() -> None:
    from owl.train.utils import configure_torch

    configure_torch()


def build_model(dev: torch.device, *, seed: int = 0, depth: int | None = None) -> Any:
    from owl.kaggriculture import types as kt
    from owl.model import kaggriculture as km

    overrides = None if depth is None else {"depth": depth}
    config = km.KaggricultureTransformerConfig.from_file(
        ROOT / "configs/model/kaggriculture.yaml", overrides
    )
    assert config.force_flash_attn is True
    torch.manual_seed(seed)
    return km.KaggricultureTransformer(
        config,
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).to(dev)


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


def make_obs_batch(envs: int, density: str) -> Any:
    from tests.kaggriculture.conftest import make_obs

    spec = mixed_counts(envs) if density == "mixed" else DENSITIES[density]
    return make_obs(
        envs, own_actors=spec["own"], rival_actors=spec["rival"], shops=spec["shops"]
    )


def map_tensors(obj: BaseModel, fn: Any) -> BaseModel:
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


def to_dev(obj: BaseModel, dev: torch.device) -> Any:
    return map_tensors(obj, lambda t: t.to(dev))


def tile(obj: BaseModel, k: int) -> Any:
    """Repeat every tensor k times along the leading (env) dim."""
    return map_tensors(obj, lambda t: t.repeat(k, *([1] * (t.dim() - 1))))


def first_envs(obj: BaseModel, n: int) -> Any:
    return map_tensors(obj, lambda t: t[:n].clone())


def tile_actions(actions: Any, k: int) -> Any:
    from owl.kaggriculture import types as kt

    return kt.KaggricultureActions(
        tokens=actions.tokens.repeat(k, 1, 1, 1), lengths=actions.lengths.repeat(k, 1)
    )


def token_groups() -> list[tuple[str, int, int]]:
    """Token index ranges of one seat row (KaggricultureTransformer order)."""
    from owl.kaggriculture import types as kt

    sizes = [
        ("own_actor", kt.MAX_ACTORS),
        ("rival_actor", kt.ACTOR_SLOTS - kt.MAX_ACTORS),
        ("tile", kt.TILES),
        ("shop", kt.SHOP_SLOTS),
        ("market", kt.PRODUCT_COUNT),
        ("player", kt.PLAYERS),
        ("global", 1),
        ("board", 4),
        ("plan", 1),
        ("critic", kt.PLAYERS),
    ]
    out, start = [], 0
    for name, n in sizes:
        out.append((name, start, start + n))
        start += n
    return out


def group_of(token: int) -> str:
    for name, lo, hi in token_groups():
        if lo <= token < hi:
            return name
    return "?"


def mem(dev: torch.device) -> dict[str, float]:
    if dev.type != "cuda":
        return {}
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated(dev) / GIB,
        "max_reserved_gib": torch.cuda.max_memory_reserved(dev) / GIB,
    }
