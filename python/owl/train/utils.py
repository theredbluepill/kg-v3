from __future__ import annotations

import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Literal, Protocol, assert_never

import torch
import torch._dynamo
from torch import nn

from owl.model import (
    KaggricultureTransformer,
    RecurrentTransformerV1,
    TrunkCompileAPI,
)
from owl.model.compile_gemm import CompileGame, claim_gemm_backends

ModelCompileTarget = Literal["none", "mlp", "trunk"]
ModelCompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
TrainingDType = Literal["float32", "bfloat16"]


class DTypeConfig(Protocol):
    @property
    def dtype(self) -> TrainingDType: ...


class ModelCompileConfig(Protocol):
    @property
    def compile_actor_heads(self) -> bool: ...

    @property
    def model_compile(self) -> ModelCompileTarget: ...

    @property
    def model_compile_mode(self) -> ModelCompileMode: ...


def configure_torch() -> None:
    # PyTorch 2.9 Inductor still reads the legacy matmul allow_tf32 flag during
    # lowering. Mixing the new fp32_precision setters with that read raises at
    # compile time, so keep this on one API family until Inductor moves over.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Please use the new API settings to control TF32 behavior.*",
            category=UserWarning,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # DDPOptimizer splits a torch.compile'd module under DDP into one subgraph per
    # ~25MB gradient bucket. For the 1B trunk that is ~150 subgraphs, and a bucket
    # boundary that falls mid-block trips an AOTAutograd functionalization bug
    # ("tensor does not have a device") that desyncs ranks on the next collective.
    # Compiling the trunk as a single graph avoids the split bug; we give up some
    # comm/compute overlap but gradients are identical to the split path.
    torch._dynamo.config.optimize_ddp = False


def autocast_context(
    cfg: DTypeConfig,
    device: torch.device,
) -> AbstractContextManager[None]:
    match cfg.dtype:
        case "bfloat16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        case "float32":
            return nullcontext()
        case _:
            assert_never(cfg.dtype)


def _compile_game(model: nn.Module) -> CompileGame:
    return "kaggriculture" if isinstance(model, KaggricultureTransformer) else "orbit"


def configure_model_compile(model: nn.Module, cfg: ModelCompileConfig) -> int:
    """Compile the configured model region in place; never the whole model.

    ``trunk`` dispatches through ``TrunkCompileAPI``, so each model compiles
    only its trunk and keeps its own dispatch (packing, chunking, overflow
    guards) in front of the compiled callable. ``mlp`` compiles the trunk's
    block MLP modules in place. Before any compile, the game claims the
    process-global GEMM backends (cuBLAS only for Kaggriculture; unchanged for
    Orbit): here for ``mlp``, inside ``compile_transformer_trunk`` for ``trunk``.
    One process never compiles both games.
    """
    compiled_heads = 0
    if cfg.compile_actor_heads:
        if not isinstance(model, KaggricultureTransformer):
            raise ValueError("rl.compile_actor_heads requires a Kaggriculture model")
        compiled_heads = model.compile_actor_heads(mode=cfg.model_compile_mode)
    return compiled_heads + _configure_trunk_compile(model, cfg)


def _configure_trunk_compile(model: nn.Module, cfg: ModelCompileConfig) -> int:
    match cfg.model_compile:
        case "none":
            return 0
        case "mlp":
            claim_gemm_backends(_compile_game(model))
            compiled = _compile_transformer_mlp_modules(
                model,
                mode=cfg.model_compile_mode,
            )
            if isinstance(model, KaggricultureTransformer):
                model.compiled_regions_require_gemm_backends = True
            return compiled
        case "trunk":
            if isinstance(model, RecurrentTransformerV1):
                raise RuntimeError(
                    "rl.model_compile='trunk' does not support recurrent_transformer_v1"
                )
            if not isinstance(model, TrunkCompileAPI):
                raise RuntimeError(
                    "rl.model_compile='trunk' requires a model implementing "
                    f"TrunkCompileAPI, got {type(model).__name__}"
                )
            # Each implementation claims its game's GEMM backends itself, so
            # direct calls are held to the same rules as this one.
            return model.compile_transformer_trunk(mode=cfg.model_compile_mode)
        case _:
            assert_never(cfg.model_compile)


def _compile_transformer_mlp_modules(
    model: nn.Module,
    *,
    mode: ModelCompileMode,
) -> int:
    compiled = 0
    for name, module in model.named_modules():
        if not _is_transformer_mlp_module_name(name):
            continue
        module.compile(mode=mode, dynamic=True)
        compiled += 1
    if compiled == 0:
        raise RuntimeError(
            "rl.model_compile='mlp' found no transformer MLP modules to compile"
        )
    return compiled


def _is_transformer_mlp_module_name(name: str) -> bool:
    return (
        name.startswith("blocks.")
        or (".blocks." in name and name.startswith("player_count_adapters."))
    ) and name.endswith(".mlp")


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    if torch.isfinite(tensor).all():
        return
    raise ValueError(f"{name} must contain only finite values")


def require_same_shape(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape == right.shape:
        return
    raise ValueError(
        f"{right_name} must match {left_name} shape {left.shape}, got {right.shape}"
    )


def require_segment_time_major(tensor: torch.Tensor, name: str) -> None:
    """Require segment-major/time-second layout: [N, T, ...]."""
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have shape [N, T, ...], got {tensor.shape}")


def require_probability_range(value: float, name: str) -> None:
    if 0.0 <= value <= 1.0:
        return

    raise ValueError(f"{name} must be between 0 and 1")
