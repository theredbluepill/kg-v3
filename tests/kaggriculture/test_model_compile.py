"""Task 3.1 (model side): trainer compile targets reach only the Kaggriculture trunk."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from owl.kaggriculture import types as kt
from owl.model import TrunkCompileAPI
from owl.model import kaggriculture as km
from owl.train import PPOConfig
from owl.train.utils import configure_model_compile
from torch import nn

from tests.kaggriculture.conftest import make_obs

ROOT = Path(__file__).resolve().parents[2]
MODE = "max-autotune-no-cudagraphs"
_TrunkFn = Callable[[torch.Tensor, torch.Tensor | None, object], torch.Tensor]


def _tiny() -> km.KaggricultureTransformer:
    torch.manual_seed(11)
    return km.KaggricultureTransformer(
        km.KaggricultureTransformerConfig(
            embed_dim=16, depth=2, n_heads=2, mlp_ratio=2.0, n_scratch_tokens=1
        ),
        obs_spec=kt.KaggricultureObsConfig(),
        action_spec=kt.KaggricultureActionConfig(),
    ).eval()


def _is_trunk_module(name: str) -> bool:
    return name == "final_norm" or name == "blocks" or name.startswith("blocks.")


class _CompileRecorder:
    """Fake ``torch.compile`` that marks when the compiled callable is running."""

    def __init__(self) -> None:
        self.targets: list[Any] = []
        self.kwargs: list[dict[str, object]] = []
        self.calls: list[int] = []
        self.inside = False

    def __call__(self, fn: Any, **kwargs: object) -> Any:
        self.targets.append(fn)
        self.kwargs.append(kwargs)

        def compiled(
            x: torch.Tensor, mask: torch.Tensor | None, packed: object
        ) -> torch.Tensor:
            self.calls.append(x.shape[0])
            self.inside = True
            try:
                return fn(x, mask, packed)
            finally:
                self.inside = False

        return compiled


@contextmanager
def _module_calls(
    model: nn.Module, inside: Callable[[], bool]
) -> Iterator[list[tuple[str, bool]]]:
    """Record (module name, ran inside the compiled region) for every call."""
    calls: list[tuple[str, bool]] = []
    handles = [
        module.register_forward_pre_hook(
            lambda _m, _a, name=name: calls.append((name, inside()))
        )
        for name, module in model.named_modules()
        if name
    ]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def _exercise(model: km.KaggricultureTransformer) -> None:
    """Sampling, replay and value paths, as rollout and PPO use them."""
    obs = make_obs(envs=2)
    with torch.no_grad():
        sampled = model(obs)
        model.evaluate_actions(obs, sampled.actions)
        model.compute_value(obs)


def test_kaggriculture_implements_the_trunk_compile_interface() -> None:
    model = _tiny()
    assert isinstance(model, TrunkCompileAPI)


def test_trunk_target_compiles_exactly_the_trunk_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    recorder = _CompileRecorder()
    module_compiles: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", recorder)
    monkeypatch.setattr(
        nn.Module, "compile", lambda self, *_a, **_k: module_compiles.append(self)
    )
    state_keys = set(model.state_dict())

    compiled = configure_model_compile(
        model, PPOConfig(model_compile="trunk", model_compile_mode=MODE)
    )

    assert compiled == 1
    assert module_compiles == []
    [target] = recorder.targets
    assert target.__self__ is model
    assert target.__func__ is km.KaggricultureTransformer._forward_transformer_trunk
    assert recorder.kwargs == [{"mode": MODE, "dynamic": True}]
    assert model._compiled_transformer_trunk is not None
    assert model._compiled_actor_core is None
    assert set(model.state_dict()) == state_keys


def test_trunk_compiled_region_holds_only_blocks_and_final_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    recorder = _CompileRecorder()
    monkeypatch.setattr(torch, "compile", recorder)
    configure_model_compile(
        model, PPOConfig(model_compile="trunk", model_compile_mode=MODE)
    )

    with _module_calls(model, lambda: recorder.inside) as calls:
        _exercise(model)

    assert recorder.calls, "the compiled trunk never ran"
    inside = {name for name, flag in calls if flag}
    outside = {name for name, flag in calls if not flag}
    trunk = {name for name, _ in model.named_modules() if _is_trunk_module(name)}
    # Every block submodule and the final norm run, and only inside the region.
    assert inside == trunk - {"blocks"}
    assert not any(_is_trunk_module(name) for name in outside)
    # Stems, critic, actor projection and heads run eagerly outside it.
    for eager in ("tile_proj", "global_proj", "critic_head", "actor_input_proj"):
        assert eager in outside
    assert any(name.startswith("actor.heads.") for name in outside)


def test_trunk_guard_stays_in_front_of_the_compiled_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    recorder = _CompileRecorder()
    monkeypatch.setattr(torch, "compile", recorder)
    configure_model_compile(
        model, PPOConfig(model_compile="trunk", model_compile_mode=MODE)
    )
    obs = make_obs(envs=2)
    rows = obs.still_playing.numel()

    monkeypatch.setattr(km, "rows_per_chunk", lambda **_: 1)
    with torch.no_grad():
        model.encode_observations(obs)
    assert recorder.calls == [1] * rows

    recorder.calls.clear()
    monkeypatch.setattr(km, "rows_per_chunk", lambda **_: 0)
    with pytest.raises(ValueError, match="compiled-gemm-template-overflows"):
        model.encode_observations(obs)
    assert recorder.calls == []


def test_mlp_target_compiles_only_trunk_block_mlps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    compiled: list[tuple[nn.Module, dict[str, object]]] = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda self, *_a, **k: compiled.append((self, k))
    )
    monkeypatch.setattr(torch, "compile", lambda *_a, **_k: pytest.fail("compile"))

    count = configure_model_compile(
        model, PPOConfig(model_compile="mlp", model_compile_mode=MODE)
    )

    block_mlps = [block.mlp for block in model.blocks]
    assert count == len(block_mlps) == model.config.depth
    assert [module for module, _ in compiled] == block_mlps
    assert all(kwargs == {"mode": MODE, "dynamic": True} for _, kwargs in compiled)


def test_mlp_target_is_reached_only_through_the_guarded_trunk_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    compiled: list[nn.Module] = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda self, *_a, **_k: compiled.append(self)
    )
    configure_model_compile(
        model, PPOConfig(model_compile="mlp", model_compile_mode=MODE)
    )
    in_run_trunk = [False]
    original = model._run_trunk

    def run_trunk(x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        in_run_trunk[0] = True
        try:
            return original(x, token_mask)
        finally:
            in_run_trunk[0] = False

    monkeypatch.setattr(model, "_run_trunk", run_trunk)
    mlp_calls: list[bool] = []
    handles = [
        module.register_forward_pre_hook(
            lambda _m, _a: mlp_calls.append(in_run_trunk[0])
        )
        for module in compiled
    ]
    try:
        _exercise(model)
        assert mlp_calls
        assert all(mlp_calls)

        mlp_calls.clear()
        monkeypatch.setattr(km, "rows_per_chunk", lambda **_: 0)
        with pytest.raises(ValueError, match="compiled-gemm-template-overflows"):
            model.encode_observations(make_obs())
        assert mlp_calls == []
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("target", ["none", "mlp", "trunk"])
def test_trainer_compile_never_compiles_the_whole_model(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    model = _tiny()
    compiled_fns: list[Any] = []
    compiled_modules: list[nn.Module] = []
    monkeypatch.setattr(torch, "compile", lambda fn, **_: compiled_fns.append(fn))
    monkeypatch.setattr(
        nn.Module, "compile", lambda self, *_a, **_k: compiled_modules.append(self)
    )

    configure_model_compile(
        model, PPOConfig(model_compile=target, model_compile_mode=MODE)
    )

    whole = (model, model.forward, model.evaluate_actions, model.compute_value)
    assert not any(fn in whole for fn in compiled_fns)
    assert model not in compiled_modules
    trunk = {id(m) for n, m in model.named_modules() if _is_trunk_module(n)}
    assert all(id(module) in trunk for module in compiled_modules)


def _compile_targets(path: Path) -> list[str]:
    """Source of every ``torch.compile(x)`` / ``x.compile(...)`` target in a file."""
    targets = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "compile":
            continue
        owner = ast.unparse(node.func.value)
        if owner == "re":
            continue
        targets.append(ast.unparse(node.args[0]) if owner == "torch" else owner)
    return targets


def test_trainer_path_compile_sites_never_take_a_model() -> None:
    """Every compile call site on the trainer and model paths, pinned by source.

    The trainer compiles models only through ``configure_model_compile``; the
    remaining sites compile pure tensor functions.
    """
    files = [
        ROOT / "scripts" / "run_ppo.py",
        *sorted((ROOT / "python" / "owl" / "train").glob("*.py")),
        *sorted((ROOT / "python" / "owl" / "model").glob("*.py")),
    ]
    found = {
        str(path.relative_to(ROOT)): targets
        for path in files
        if (targets := _compile_targets(path))
    }
    assert found == {
        "python/owl/model/kaggriculture.py": ["self._forward_transformer_trunk"],
        "python/owl/model/stateless_transformer_v1.py": [
            "self._forward_transformer_trunk"
        ],
        "python/owl/train/advantages.py": ["_compute_gae_tensors"],
        "python/owl/train/ppo.py": ["_ppo_loss_components"],
        "python/owl/train/utils.py": ["module"],
    }


def test_trunk_target_rejects_models_without_the_interface() -> None:
    with pytest.raises(RuntimeError, match="requires a model implementing"):
        configure_model_compile(nn.Linear(2, 2), PPOConfig(model_compile="trunk"))
