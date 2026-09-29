"""Process-global Inductor GEMM backends, claimed for one game per process.

Compiled Kaggriculture regions lower GEMMs to cuBLAS only (``"ATEN"``) on the
probed torch/triton/driver stack; Isaiah's Orbit models keep the backends they
find. ``torch._inductor.config`` is process-global and Inductor reads it when a
compiled callable first runs or recompiles, not at ``torch.compile``, so every
public compile entry point (``configure_model_compile`` and each model's
``compile_transformer_trunk``) claims the setting for its game before compiling,
and compiling the other game in the same process raises.

Evidence: cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from importlib import metadata
from typing import Literal

import torch
import torch._inductor.config as inductor_config

CompileGame = Literal["kaggriculture", "orbit"]
COMPILED_GEMM_BACKENDS = "ATEN"
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


_GEMM_BACKEND_CLAIM: GemmBackendClaim | None = None
# Serializes check-and-set, so a concurrent claim for the other game raises
# instead of landing between this claim's check and its write.
_GEMM_BACKEND_CLAIM_LOCK = threading.Lock()


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
    before anything compiles. An installed triton is always checked. A GPU host
    must have triton; a host without CUDA may lack it, and the triton check is
    then skipped with an explicit reason. The NVIDIA driver is read and checked
    only on CUDA hosts, and skipped with a reason elsewhere.
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


def claim_gemm_backends(game: CompileGame) -> GemmBackendClaim:
    """Check and apply ``game``'s GEMM backends; call before ``torch.compile``.

    Kaggriculture checks the probed stack and restricts Inductor to cuBLAS GEMMs
    (``"ATEN"``) for every compile mode: only max-autotune modes consider Triton
    GEMM templates, but ``TORCHINDUCTOR_MAX_AUTOTUNE`` can enable them under any
    mode. Isaiah's Orbit models keep the backends they find. Claiming the game
    that already holds the claim is allowed (the trainer's eval and last-best
    models); claiming the other game raises before anything changes. Claims
    are serialized by a lock, so this holds for concurrent claims from threads.
    """
    global _GEMM_BACKEND_CLAIM
    with _GEMM_BACKEND_CLAIM_LOCK:
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


def require_compiled_gemm_backends() -> None:
    """Raise unless Inductor's GEMM backends are Kaggriculture's cuBLAS-only set."""
    backends = inductor_config.max_autotune_gemm_backends
    if backends != COMPILED_GEMM_BACKENDS:
        raise RuntimeError(
            "compiled Kaggriculture regions require "
            f"torch._inductor.config.max_autotune_gemm_backends="
            f"{COMPILED_GEMM_BACKENDS!r}, found {backends!r}; something reset it "
            "after the compile claimed it (see "
            f"{_GEMM_BACKENDS_DECISION})"
        )
