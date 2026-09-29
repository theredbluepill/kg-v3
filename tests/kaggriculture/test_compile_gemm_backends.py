"""Compiled Kaggriculture regions lower GEMMs to cuBLAS only; Orbit keeps its own.

Evidence: ops/rebuild-2026-09-29/results.md, "ATEN-only GEMM A/B"; decision
cookbook/decisions/kaggriculture-compiles-gemms-with-cublas-only.md.
"""

from __future__ import annotations

import re
from typing import Any, get_args

import pytest
import torch
import torch._inductor.config as inductor_config
from owl.kaggriculture import types as kt
from owl.model import StatelessTransformerV1
from owl.model import compile_gemm as cg
from owl.model import kaggriculture as km
from owl.model.compile_gemm import (
    COMPILED_GEMM_BACKENDS,
    KAGGRICULTURE_PROBED_COMPILE_STACK,
    CompileStackReport,
    GemmBackendClaim,
    InstalledCompileStack,
    check_compile_stack,
    gemm_backend_claim,
)
from owl.model.stateless_transformer_v1 import StatelessTransformerV1Config
from owl.rl import ActionPureConfig, EntityBasedConfig
from owl.train import PPOConfig
from owl.train.utils import ModelCompileMode, configure_model_compile
from torch import nn

from tests.kaggriculture.conftest import PROBED_GPU_STACK, make_obs

ISAIAH_DEFAULT_BACKENDS = "ATEN,TRITON,CPP"
MODES: tuple[ModelCompileMode, ...] = get_args(ModelCompileMode)
PROBED_DRIVER = KAGGRICULTURE_PROBED_COMPILE_STACK.nvidia_drivers[0]


@pytest.fixture(autouse=True)
def _isaiah_default_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inductor_config, "max_autotune_gemm_backends", ISAIAH_DEFAULT_BACKENDS
    )


def _kaggriculture() -> km.KaggricultureTransformer:
    torch.manual_seed(3)
    return km.KaggricultureTransformer(
        km.KaggricultureTransformerConfig(
            embed_dim=16, depth=2, n_heads=2, mlp_ratio=2.0, n_scratch_tokens=1
        ),
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).eval()


def _orbit() -> StatelessTransformerV1:
    return StatelessTransformerV1(
        StatelessTransformerV1Config(embed_dim=32, depth=2, n_heads=4, mlp_ratio=1.0),
        obs_spec=EntityBasedConfig(max_entities=64),
        action_spec=ActionPureConfig(max_per_planet_launches=1),
    )


def _record_backends_at_compile(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace both compile entry points; record the backends each one sees."""
    seen: list[str] = []

    def fake_torch_compile(fn: Any, **_: object) -> Any:
        seen.append(inductor_config.max_autotune_gemm_backends)
        return fn

    def fake_module_compile(_self: nn.Module, *_a: object, **_k: object) -> None:
        seen.append(inductor_config.max_autotune_gemm_backends)

    monkeypatch.setattr(torch, "compile", fake_torch_compile)
    monkeypatch.setattr(nn.Module, "compile", fake_module_compile)
    return seen


def test_the_probed_stack_is_one_constant() -> None:
    assert KAGGRICULTURE_PROBED_COMPILE_STACK.torch == "2.9.0"
    assert KAGGRICULTURE_PROBED_COMPILE_STACK.triton == "3.5.0"
    assert KAGGRICULTURE_PROBED_COMPILE_STACK.nvidia_drivers == ("595.91.07",)
    assert COMPILED_GEMM_BACKENDS == "ATEN"


@pytest.mark.usefixtures("probed_compile_stack")
@pytest.mark.parametrize("target", ["trunk", "mlp"])
@pytest.mark.parametrize("mode", MODES)
def test_kaggriculture_compiles_see_cublas_only_gemm_backends(
    monkeypatch: pytest.MonkeyPatch, target: str, mode: ModelCompileMode
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    model = _kaggriculture()

    compiled = configure_model_compile(
        model, PPOConfig(model_compile=target, model_compile_mode=mode)
    )

    # Set before every compile call, for max-autotune and every other mode.
    assert seen
    assert len(seen) == compiled
    assert set(seen) == {"ATEN"}
    assert inductor_config.max_autotune_gemm_backends == "ATEN"
    assert model.compiled_regions_require_gemm_backends
    claim = gemm_backend_claim()
    assert claim == GemmBackendClaim(
        game="kaggriculture",
        gemm_backends="ATEN",
        stack=CompileStackReport(
            torch=PROBED_GPU_STACK.torch, triton="3.5.0", nvidia_driver=PROBED_DRIVER
        ),
    )
    assert claim.summary() == {
        "compile_gemm_game": "kaggriculture",
        "compile_gemm_backends": "ATEN",
        "compile_stack_torch": "2.9.0+cu128",
        "compile_stack_triton": "3.5.0",
        "compile_stack_nvidia_driver": PROBED_DRIVER,
    }


@pytest.mark.parametrize("target", ["trunk", "mlp"])
@pytest.mark.parametrize("mode", ["max-autotune", "max-autotune-no-cudagraphs"])
def test_orbit_compiles_keep_isaiahs_gemm_backends(
    monkeypatch: pytest.MonkeyPatch, target: str, mode: ModelCompileMode
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    monkeypatch.setattr(
        cg,
        "installed_compile_stack",
        lambda: pytest.fail("Orbit compiles do not check the Kaggriculture stack"),
    )

    configure_model_compile(
        _orbit(), PPOConfig(model_compile=target, model_compile_mode=mode)
    )

    assert seen
    assert set(seen) == {ISAIAH_DEFAULT_BACKENDS}
    assert inductor_config.max_autotune_gemm_backends == ISAIAH_DEFAULT_BACKENDS
    assert gemm_backend_claim() == GemmBackendClaim(
        game="orbit", gemm_backends=ISAIAH_DEFAULT_BACKENDS, stack=None
    )


@pytest.mark.parametrize("model_factory", [_kaggriculture, _orbit])
def test_no_compile_target_claims_nothing(
    monkeypatch: pytest.MonkeyPatch, model_factory: Any
) -> None:
    _record_backends_at_compile(monkeypatch)
    compiled = configure_model_compile(model_factory(), PPOConfig(model_compile="none"))
    assert compiled == 0
    assert gemm_backend_claim() is None
    assert inductor_config.max_autotune_gemm_backends == ISAIAH_DEFAULT_BACKENDS


@pytest.mark.usefixtures("probed_compile_stack")
def test_one_process_never_compiles_both_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    cfg = PPOConfig(model_compile="trunk", model_compile_mode="max-autotune")
    configure_model_compile(_kaggriculture(), cfg)
    # The trainer's eval and last-best models are the same game: allowed.
    configure_model_compile(_kaggriculture(), cfg)

    with pytest.raises(RuntimeError, match="already compiled a kaggriculture model"):
        configure_model_compile(_orbit(), cfg)
    assert seen == ["ATEN", "ATEN"]


@pytest.mark.usefixtures("probed_compile_stack")
def test_kaggriculture_after_orbit_raises_before_changing_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    cfg = PPOConfig(model_compile="mlp", model_compile_mode="max-autotune")
    configure_model_compile(_orbit(), cfg)
    orbit_compiles = len(seen)

    with pytest.raises(RuntimeError, match="already compiled a orbit model"):
        configure_model_compile(_kaggriculture(), cfg)
    # Orbit's lazily compiled graphs would read "ATEN" on first run otherwise.
    assert inductor_config.max_autotune_gemm_backends == ISAIAH_DEFAULT_BACKENDS
    assert len(seen) == orbit_compiles


# --- direct compile entry points (not through configure_model_compile) --------


@pytest.mark.usefixtures("probed_compile_stack")
def test_direct_kaggriculture_trunk_compile_claims_cublas_only_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    model = _kaggriculture()

    assert model.compile_transformer_trunk(mode="max-autotune") == 1

    assert seen == ["ATEN"]
    assert model.compiled_regions_require_gemm_backends
    claim = gemm_backend_claim()
    assert claim is not None
    assert claim.game == "kaggriculture"
    assert claim.stack == CompileStackReport(
        torch=PROBED_GPU_STACK.torch, triton="3.5.0", nvidia_driver=PROBED_DRIVER
    )


def test_direct_kaggriculture_trunk_compile_checks_the_stack_with_aten_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting "ATEN" by hand does not skip the probed-stack check or the claim."""
    seen = _record_backends_at_compile(monkeypatch)
    monkeypatch.setattr(inductor_config, "max_autotune_gemm_backends", "ATEN")
    monkeypatch.setattr(
        cg, "installed_compile_stack", lambda: _installed(torch="2.10.0+cu128")
    )
    model = _kaggriculture()

    with pytest.raises(RuntimeError, match=re.escape("unprobed torch 2.10.0")):
        model.compile_transformer_trunk(mode="max-autotune")

    assert seen == []
    assert model._compiled_transformer_trunk is None
    assert not model.compiled_regions_require_gemm_backends
    assert gemm_backend_claim() is None


def test_direct_orbit_trunk_compile_claims_orbit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    monkeypatch.setattr(
        cg,
        "installed_compile_stack",
        lambda: pytest.fail("Orbit compiles do not check the Kaggriculture stack"),
    )

    assert _orbit().compile_transformer_trunk(mode="max-autotune") == 1

    assert seen == [ISAIAH_DEFAULT_BACKENDS]
    assert gemm_backend_claim() == GemmBackendClaim(
        game="orbit", gemm_backends=ISAIAH_DEFAULT_BACKENDS, stack=None
    )


def _configured_trunk(model: nn.Module) -> None:
    configure_model_compile(
        model, PPOConfig(model_compile="trunk", model_compile_mode="max-autotune")
    )


def _configured_mlp(model: nn.Module) -> None:
    configure_model_compile(
        model, PPOConfig(model_compile="mlp", model_compile_mode="max-autotune")
    )


def _direct_trunk(model: nn.Module) -> None:
    assert isinstance(model, km.KaggricultureTransformer | StatelessTransformerV1)
    model.compile_transformer_trunk(mode="max-autotune")


COMPILE_ENTRIES = {
    "configured-trunk": _configured_trunk,
    "configured-mlp": _configured_mlp,
    "direct-trunk": _direct_trunk,
}


@pytest.mark.usefixtures("probed_compile_stack")
@pytest.mark.parametrize("first", COMPILE_ENTRIES)
@pytest.mark.parametrize("second", COMPILE_ENTRIES)
def test_orbit_then_kaggriculture_raises_through_any_entry(
    monkeypatch: pytest.MonkeyPatch, first: str, second: str
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    COMPILE_ENTRIES[first](_orbit())
    orbit_compiles = list(seen)
    kaggriculture = _kaggriculture()

    with pytest.raises(RuntimeError, match="already compiled a orbit model"):
        COMPILE_ENTRIES[second](kaggriculture)

    # Orbit's lazily compiled graphs would read "ATEN" on first run otherwise.
    assert inductor_config.max_autotune_gemm_backends == ISAIAH_DEFAULT_BACKENDS
    assert seen == orbit_compiles
    assert kaggriculture._compiled_transformer_trunk is None
    claim = gemm_backend_claim()
    assert claim is not None
    assert claim.game == "orbit"


@pytest.mark.usefixtures("probed_compile_stack")
@pytest.mark.parametrize("first", COMPILE_ENTRIES)
@pytest.mark.parametrize("second", COMPILE_ENTRIES)
def test_kaggriculture_then_orbit_raises_through_any_entry(
    monkeypatch: pytest.MonkeyPatch, first: str, second: str
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    COMPILE_ENTRIES[first](_kaggriculture())
    kaggriculture_compiles = list(seen)
    orbit = _orbit()

    with pytest.raises(RuntimeError, match="already compiled a kaggriculture model"):
        COMPILE_ENTRIES[second](orbit)

    assert inductor_config.max_autotune_gemm_backends == "ATEN"
    assert seen == kaggriculture_compiles
    assert orbit._compiled_transformer_trunk is None


@pytest.mark.usefixtures("probed_compile_stack")
@pytest.mark.parametrize("target", ["trunk", "mlp"])
def test_compiled_trunk_calls_recheck_the_backends(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A compiled region never runs after something resets the backends.

    Inductor compiles lazily and recompiles on new shapes, reading the config
    at that point rather than at ``torch.compile``.
    """
    _record_backends_at_compile(monkeypatch)
    model = _kaggriculture()
    configure_model_compile(
        model, PPOConfig(model_compile=target, model_compile_mode="max-autotune")
    )
    obs = make_obs(envs=2)
    with torch.no_grad():
        model.encode_observations(obs)  # runs while the backends are cuBLAS-only

    block_calls: list[int] = []
    handle = model.blocks[0].register_forward_pre_hook(
        lambda _m, args: block_calls.append(args[0].shape[0])
    )
    monkeypatch.setattr(
        inductor_config, "max_autotune_gemm_backends", ISAIAH_DEFAULT_BACKENDS
    )
    try:
        with (
            pytest.raises(RuntimeError, match="found 'ATEN,TRITON,CPP'"),
            torch.no_grad(),
        ):
            model.encode_observations(obs)
    finally:
        handle.remove()
    assert block_calls == []


def test_eager_models_run_under_any_backends() -> None:
    model = _kaggriculture()
    assert not model.compiled_regions_require_gemm_backends
    with torch.no_grad():
        model.encode_observations(make_obs())


# --- probed-stack check --------------------------------------------------------


def _installed(**overrides: Any) -> InstalledCompileStack:
    fields: dict[str, Any] = {
        "torch": "2.9.0+cu128",
        "triton": "3.5.0",
        "cuda_available": True,
        "nvidia_drivers": (PROBED_DRIVER,),
    } | overrides
    return InstalledCompileStack(**fields)


def test_stack_check_accepts_the_probed_gpu_stack() -> None:
    assert check_compile_stack(_installed()) == CompileStackReport(
        torch="2.9.0+cu128", triton="3.5.0", nvidia_driver=PROBED_DRIVER
    )


@pytest.mark.parametrize(
    ("overrides", "component"),
    [
        ({"torch": "2.10.0+cu128"}, "torch 2.10.0"),
        ({"torch": "2.9.1"}, "torch 2.9.1"),
        ({"torch": "2.8.0+cu128"}, "torch 2.8.0"),
        ({"triton": "3.4.0"}, "triton 3.4.0"),
        ({"triton": "3.6.0", "cuda_available": False}, "triton 3.6.0"),
        ({"nvidia_drivers": ("570.86.10",)}, "NVIDIA driver 570.86.10"),
        (
            {"nvidia_drivers": (PROBED_DRIVER, "580.65.06")},
            "NVIDIA driver 580.65.06",
        ),
    ],
)
def test_stack_check_rejects_unprobed_versions(
    overrides: dict[str, Any], component: str
) -> None:
    with pytest.raises(RuntimeError, match=re.escape(f"unprobed {component}")):
        check_compile_stack(_installed(**overrides))


def test_stack_check_rejects_a_gpu_host_without_triton() -> None:
    with pytest.raises(RuntimeError, match="triton is not installed"):
        check_compile_stack(_installed(triton=None))


def test_stack_check_rejects_a_gpu_host_whose_driver_was_not_read() -> None:
    with pytest.raises(RuntimeError, match="no NVIDIA driver was read"):
        check_compile_stack(_installed(nvidia_drivers=None))


def test_stack_check_skips_driver_and_triton_without_cuda_with_reasons() -> None:
    """The owner's Mac: no CUDA, no triton. The driver is checked on GPU hosts."""
    report = check_compile_stack(
        _installed(
            torch="2.9.0", triton=None, cuda_available=False, nvidia_drivers=None
        )
    )
    assert report.torch == "2.9.0"
    assert report.triton.startswith("skipped: triton is not installed")
    assert report.nvidia_driver == (
        "skipped: torch.cuda.is_available() is False; the driver is checked "
        "with nvidia-smi on GPU hosts"
    )


def test_kaggriculture_compile_rejects_an_unprobed_torch_before_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_backends_at_compile(monkeypatch)
    monkeypatch.setattr(
        cg,
        "installed_compile_stack",
        lambda: _installed(torch="2.10.0+cu128"),
    )
    with pytest.raises(RuntimeError, match=re.escape("unprobed torch 2.10.0")):
        configure_model_compile(
            _kaggriculture(),
            PPOConfig(model_compile="trunk", model_compile_mode="max-autotune"),
        )
    assert seen == []
    assert gemm_backend_claim() is None
    assert inductor_config.max_autotune_gemm_backends == ISAIAH_DEFAULT_BACKENDS


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only host path")
def test_this_cpu_host_reads_no_driver() -> None:
    installed = cg.installed_compile_stack()
    assert not installed.cuda_available
    assert installed.nvidia_drivers is None
    report = check_compile_stack(installed)
    assert report.nvidia_driver.startswith("skipped: ")
