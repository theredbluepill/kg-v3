"""Opt-in actor compilation preserves eager sampling and its RNG stream."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch._inductor.config as inductor_config
from owl.model import compile_gemm
from owl.model import kaggriculture as km

from tests.kaggriculture.conftest import make_obs
from tests.kaggriculture.helpers import _obs_double, _tiny

pytestmark = pytest.mark.usefixtures("probed_compile_stack")
MODE = "max-autotune-no-cudagraphs"


def _assert_outputs(actual: Any, expected: Any, *, atol: float) -> None:
    assert torch.equal(actual.actions.tokens, expected.actions.tokens)
    assert torch.equal(actual.actions.lengths, expected.actions.lengths)
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.log_probs.event, expected.log_probs.event, rtol=0, atol=atol
    )
    torch.testing.assert_close(
        actual.entropies.event, expected.entropies.event, rtol=0, atol=atol
    )


@pytest.mark.parametrize("mode", ["default", MODE])
def test_actor_entrypoint_claims_backends_and_compiles_only_policy_core(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    model = _tiny()
    calls: list[tuple[Any, dict[str, Any]]] = []

    def compile_(fn: Any, **kwargs: Any) -> Any:
        assert inductor_config.max_autotune_gemm_backends == "ATEN"
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", compile_)
    state_keys = set(model.state_dict())
    assert model.compile_actor_heads(mode=mode) == 1
    assert calls == [
        (model.actor.policy_core, {"mode": mode, "fullgraph": True, "dynamic": False})
    ]
    assert model._compiled_transformer_trunk is None
    assert model.compiled_regions_require_gemm_backends
    assert set(model.state_dict()) == state_keys
    claim = compile_gemm.gemm_backend_claim()
    assert claim is not None
    assert claim.game == "kaggriculture"


def test_actor_entrypoint_rejects_an_orbit_claim_before_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_gemm.claim_gemm_backends("orbit")
    monkeypatch.setattr(torch, "compile", lambda *_a, **_k: pytest.fail("compile"))
    with pytest.raises(RuntimeError, match=r"already compiled an? orbit"):
        _tiny().compile_actor_heads(mode=MODE)


@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune", "unknown"])
def test_actor_entrypoint_rejects_unqualified_modes_before_claim_or_compile(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setattr(torch, "compile", lambda *_a, **_k: pytest.fail("compile"))
    monkeypatch.setattr(
        km, "claim_gemm_backends", lambda *_a: pytest.fail("backend claim")
    )
    with pytest.raises(ValueError, match="CUDA graph modes are not qualified"):
        _tiny().compile_actor_heads(mode=mode)


def test_actor_entrypoint_checks_the_stack_even_if_aten_was_set_manually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inductor_config.max_autotune_gemm_backends = "ATEN"
    monkeypatch.setattr(
        compile_gemm,
        "installed_compile_stack",
        lambda: compile_gemm.InstalledCompileStack("0.0.0", None, False, None),
    )
    monkeypatch.setattr(torch, "compile", lambda *_a, **_k: pytest.fail("compile"))
    with pytest.raises(RuntimeError, match="unprobed torch"):
        _tiny().compile_actor_heads(mode=MODE)
    assert compile_gemm.gemm_backend_claim() is None


def test_actor_dispatch_rechecks_backends_without_a_trunk_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny()
    obs = make_obs()
    with torch.no_grad():
        encoded = model.encode_observations(obs)
    monkeypatch.setattr(torch, "compile", lambda fn, **_k: fn)
    model.compile_actor_heads(mode=MODE)
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    with pytest.raises(RuntimeError, match="compiled Kaggriculture regions require"):
        model._policy(encoded, model._grammar_context(obs), None, deterministic=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_external_noise_is_bit_exact_including_chunk_rng_order(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    model = _tiny(hire_limit=4).eval()
    obs = make_obs(envs=2, own_actors=(3, 4), rival_actors=(4, 2), order_limit=10)
    obs.still_playing[1, 1] = False
    if dtype == torch.float64:
        model = model.double()
        obs = _obs_double(obs)
    monkeypatch.setattr(km, "head_rows_per_chunk", lambda _config: 2)
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16),
    ):
        torch.manual_seed(913)
        eager = model(obs)
        rng = torch.random.get_rng_state()
        monkeypatch.setattr(torch, "compile", lambda fn, **_k: fn)
        model.compile_actor_heads(mode=MODE)
        torch.manual_seed(913)
        external = model(obs)
        assert torch.equal(torch.random.get_rng_state(), rng)
    _assert_outputs(external, eager, atol=0)


def test_cpu_inductor_replay_backward_matches_eager() -> None:
    """The PPO replay path still differentiates through every actor stage."""
    torch._dynamo.reset()
    model = _tiny(hire_limit=4).eval()
    obs = make_obs(own_actors=3, rival_actors=4, order_limit=10)
    with torch.no_grad():
        torch.manual_seed(782)
        actions = model(obs).actions

    def backward() -> dict[str, torch.Tensor]:
        evaluation = model.evaluate_actions(obs, actions)
        loss = (
            evaluation.log_probs.event.sum()
            + 0.1 * evaluation.entropies.event.sum()
            + evaluation.values.square().sum()
        )
        loss.backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        model.zero_grad(set_to_none=True)
        return gradients

    eager = backward()
    model.compile_actor_heads(mode=MODE)
    compiled = backward()
    assert eager.keys() == compiled.keys()
    for name in eager:
        torch.testing.assert_close(
            compiled[name], eager[name], rtol=2e-4, atol=2e-5, msg=name
        )
    torch._dynamo.reset()


def test_cpu_inductor_sampling_replay_and_determinism_match_eager() -> None:
    """Real CPU code generation; CUDA/bf16 qualification remains pod work."""
    torch._dynamo.reset()
    model = _tiny(hire_limit=4).eval()
    obs = make_obs(envs=2, own_actors=(3, 4), rival_actors=(4, 2), order_limit=10)
    obs.still_playing[1, 1] = False
    seeds = (19, 43, 987)
    expected = []
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            expected.append((model(obs), torch.random.get_rng_state()))
        greedy = model(obs, deterministic=True)
        model.compile_actor_heads(mode=MODE)
        for seed, (eager, rng) in zip(seeds, expected, strict=True):
            torch.manual_seed(seed)
            compiled = model(obs)
            assert torch.equal(torch.random.get_rng_state(), rng)
            _assert_outputs(compiled, eager, atol=2e-6)
        before = torch.random.get_rng_state()
        _assert_outputs(model(obs, deterministic=True), greedy, atol=2e-6)
        assert torch.equal(torch.random.get_rng_state(), before)
        replay = model.evaluate_actions(obs, compiled.actions)
        torch.testing.assert_close(
            replay.log_probs.event, compiled.log_probs.event, rtol=0, atol=2e-6
        )
        torch.testing.assert_close(
            replay.entropies.event, compiled.entropies.event, rtol=0, atol=2e-6
        )
        assert torch.equal(torch.random.get_rng_state(), before)
    torch._dynamo.reset()


def test_compiled_sampling_core_contains_no_random_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent a future slot from silently drawing inside Inductor again."""
    model = _tiny().eval()
    captured = []

    def backend(graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]) -> Any:
        captured.append(graph)
        return graph.forward

    real_compile = torch.compile

    def compile_(fn: Any, **_kwargs: Any) -> Any:
        return real_compile(fn, backend=backend, fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile_)
    model.compile_actor_heads(mode=MODE)
    with torch.no_grad():
        result = model(make_obs())
    assert result.actions.tokens.numel() > 0
    assert len(captured) == 1
    assert not any(
        random in str(node.target)
        for node in captured[0].graph.nodes
        if node.op in ("call_function", "call_method")
        for random in ("exponential", "rand", "multinomial")
    )
    torch._dynamo.reset()
