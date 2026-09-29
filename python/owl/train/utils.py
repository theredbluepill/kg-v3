from __future__ import annotations

import shutil
import subprocess
import warnings
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from importlib import metadata
from typing import Literal, Protocol, assert_never

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch import nn

from owl.model import (
    KaggricultureTransformer,
    RecurrentTransformerV1,
    TrunkCompileAPI,
)
from owl.model.kaggriculture import (
    COMPILED_GEMM_BACKENDS,
    require_compiled_gemm_backends,
)

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


CompileGame = Literal["kaggriculture", "orbit"]
_GEMM_BACKENDS_DECISION = (
    "cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md"
)


@dataclass(frozen=True)
class ProbedCompileStack:
    """The software stack on which the cuBLAS-only GEMM setting was measured."""

    torch: str
    triton: str
    nvidia_drivers: tuple[str, ...]


# The ATEN-only GEMM A/B (ops/rebuild-2026-09-29/results.md, "ATEN-only GEMM
# A/B") ran on torch 2.9.0+cu128, triton 3.5.0 and NVIDIA driver 595.91.07.
# Another version must repeat that A/B before it is added here.
KAGGRICULTURE_PROBED_COMPILE_STACK = ProbedCompileStack(
    torch="2.9.0",
    triton="3.5.0",
    nvidia_drivers=("595.91.07",),
)


@dataclass(frozen=True)
class InstalledCompileStack:
    """Versions read from this host; ``None`` means not installed or no GPU."""

    torch: str
    triton: str | None
    cuda_available: bool
    nvidia_drivers: tuple[str, ...] | None


@dataclass(frozen=True)
class CompileStackReport:
    """Checked versions, or ``skipped: <reason>`` where a host cannot check."""

    torch: str
    triton: str
    nvidia_driver: str


@dataclass(frozen=True)
class GemmBackendClaim:
    """The one game whose compile settings own this process's Inductor config."""

    game: CompileGame
    gemm_backends: str
    stack: CompileStackReport | None

    def summary(self) -> dict[str, str]:
        """Telemetry fields for the run summary and startup log."""
        fields = {
            "compile_gemm_game": self.game,
            "compile_gemm_backends": self.gemm_backends,
        }
        if self.stack is not None:
            fields |= {
                "compile_stack_torch": self.stack.torch,
                "compile_stack_triton": self.stack.triton,
                "compile_stack_nvidia_driver": self.stack.nvidia_driver,
            }
        return fields


# ``torch._inductor.config`` is process-global and Inductor reads it when a
# compiled callable first runs or recompiles, not at ``torch.compile``. The first
# compile claims it for one game; compiling the other game in the same process
# raises instead of silently sharing (or overwriting) GEMM backends.
_GEMM_BACKEND_CLAIM: GemmBackendClaim | None = None


def installed_compile_stack() -> InstalledCompileStack:
    """Read torch, triton and (on CUDA hosts) the NVIDIA driver versions.

    ``torch.cuda.is_available()`` decides whether this is a GPU host; the
    driver version comes from ``nvidia-smi`` because torch exposes only the
    CUDA driver API level. Without CUDA (the owner's Mac) no driver is read.
    """
    try:
        triton_version: str | None = metadata.version("triton")
    except metadata.PackageNotFoundError:
        triton_version = None
    cuda_available = torch.cuda.is_available()
    return InstalledCompileStack(
        torch=str(torch.__version__),
        triton=triton_version,
        cuda_available=cuda_available,
        nvidia_drivers=_nvidia_driver_versions() if cuda_available else None,
    )


def _nvidia_driver_versions() -> tuple[str, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError(
            "CUDA is available but nvidia-smi is not on PATH, so the NVIDIA "
            "driver of the compiled Kaggriculture stack cannot be checked"
        )
    completed = subprocess.run(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    drivers = tuple(line.strip() for line in completed.stdout.splitlines())
    if not drivers or not all(drivers):
        raise RuntimeError(
            f"nvidia-smi reported no driver version: {completed.stdout!r}"
        )
    return drivers


def check_compile_stack(
    installed: InstalledCompileStack,
    probed: ProbedCompileStack = KAGGRICULTURE_PROBED_COMPILE_STACK,
) -> CompileStackReport:
    """Reject a torch, triton or NVIDIA driver version the A/B did not probe.

    The GEMM-backend setting's name, its source gating of the Triton templates
    and the overflow it avoids are version-specific, so an unprobed stack fails
    before anything compiles. A host without CUDA cannot run Triton kernels or
    report a driver; those two checks are skipped with an explicit reason and
    happen on GPU hosts.
    """
    torch_base = installed.torch.split("+", 1)[0]
    if torch_base != probed.torch:
        raise RuntimeError(_unprobed("torch", installed.torch, probed.torch))
    if installed.triton is not None:
        if installed.triton != probed.triton:
            raise RuntimeError(_unprobed("triton", installed.triton, probed.triton))
        triton = installed.triton
    elif installed.cuda_available:
        raise RuntimeError(
            "CUDA is available but triton is not installed; the compiled "
            f"Kaggriculture stack was probed with triton {probed.triton}"
        )
    else:
        triton = (
            "skipped: triton is not installed and torch.cuda.is_available() is "
            "False, so Inductor emits no Triton kernels on this host"
        )
    if installed.cuda_available:
        if installed.nvidia_drivers is None:
            raise RuntimeError("CUDA is available but no NVIDIA driver was read")
        unprobed = sorted(set(installed.nvidia_drivers) - set(probed.nvidia_drivers))
        if unprobed:
            raise RuntimeError(
                _unprobed(
                    "NVIDIA driver",
                    ", ".join(unprobed),
                    ", ".join(probed.nvidia_drivers),
                )
            )
        nvidia_driver = ", ".join(sorted(set(installed.nvidia_drivers)))
    else:
        nvidia_driver = (
            "skipped: torch.cuda.is_available() is False; the driver is checked "
            "with nvidia-smi on GPU hosts"
        )
    return CompileStackReport(
        torch=installed.torch, triton=triton, nvidia_driver=nvidia_driver
    )


def _unprobed(component: str, installed: str, probed: str) -> str:
    return (
        f"unprobed {component} {installed} for compiled Kaggriculture regions "
        f"(probed: {probed}); repeat the ATEN-only GEMM A/B on this stack before "
        f"adding it to KAGGRICULTURE_PROBED_COMPILE_STACK (see "
        f"{_GEMM_BACKENDS_DECISION})"
    )


def gemm_backend_claim() -> GemmBackendClaim | None:
    """This process's GEMM-backend claim, or ``None`` before any compile."""
    return _GEMM_BACKEND_CLAIM


def _claim_gemm_backends(model: nn.Module) -> GemmBackendClaim:
    """Check and apply the compiling game's GEMM backends before ``torch.compile``.

    Kaggriculture restricts Inductor to cuBLAS GEMMs (``"ATEN"``) for every
    compile mode: only max-autotune modes consider Triton GEMM templates, but
    ``TORCHINDUCTOR_MAX_AUTOTUNE`` can enable them under any mode. Isaiah's
    Orbit models keep the backends they find.
    """
    global _GEMM_BACKEND_CLAIM
    game: CompileGame = (
        "kaggriculture" if isinstance(model, KaggricultureTransformer) else "orbit"
    )
    current = _GEMM_BACKEND_CLAIM
    if current is not None and current.game != game:
        raise RuntimeError(
            f"this process already compiled a {current.game} model with "
            f"max_autotune_gemm_backends={current.gemm_backends!r}; compiling a "
            f"{game} model here would change or inherit that process-global "
            f"setting. Compile each game in its own process (see "
            f"{_GEMM_BACKENDS_DECISION})"
        )
    if game == "orbit":
        claim = GemmBackendClaim(
            game=game,
            gemm_backends=inductor_config.max_autotune_gemm_backends,
            stack=None,
        )
    else:
        stack = check_compile_stack(installed_compile_stack())
        inductor_config.max_autotune_gemm_backends = COMPILED_GEMM_BACKENDS
        require_compiled_gemm_backends()
        claim = GemmBackendClaim(
            game=game, gemm_backends=COMPILED_GEMM_BACKENDS, stack=stack
        )
    _GEMM_BACKEND_CLAIM = claim
    return claim


def configure_model_compile(model: nn.Module, cfg: ModelCompileConfig) -> int:
    """Compile the configured model region in place; never the whole model.

    ``trunk`` dispatches through ``TrunkCompileAPI``, so each model compiles
    only its trunk and keeps its own dispatch (packing, chunking, overflow
    guards) in front of the compiled callable. ``mlp`` compiles the trunk's
    block MLP modules in place. Before any compile, the game claims the
    process-global GEMM backends (cuBLAS only for Kaggriculture; unchanged for
    Orbit), and one process never compiles both games.
    """
    match cfg.model_compile:
        case "none":
            return 0
        case "mlp":
            _claim_gemm_backends(model)
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
            _claim_gemm_backends(model)
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
