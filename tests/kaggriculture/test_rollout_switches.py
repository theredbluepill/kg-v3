"""Default recipe identity and canonical PPO rollout optimization seams."""

from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any

import pytest
import torch
import torch._inductor.config as inductor_config
from owl.kaggriculture.types import KaggricultureActions
from owl.model import compile_gemm
from owl.train import ppo
from owl.train.config import FullConfig
from owl.train.distributed import DistributedContext
from owl.train.logging import config_sha256
from owl.train.ppo import PinnedActionTransfer, PPOConfig
from owl.train.utils import configure_model_compile

from scripts import run_ppo
from tests.kaggriculture.test_reward_telemetry_fast_path import _config, _env
from tests.kaggriculture.test_training_smoke import _tiny_model, _trainer

_FLAGS = ("compile_actor_heads", "rollout_packing", "pinned_action_d2h")


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_pinned_action_transfer_cpu_matches_legacy_bytes(noncontiguous: bool) -> None:
    generator = torch.Generator().manual_seed(91)
    tokens = torch.randint(0, 32, (2, 2, 252, 24), generator=generator)[..., ::2]
    lengths = torch.randint(0, 252, (2, 4), generator=generator)[:, ::2]
    if not noncontiguous:
        tokens, lengths = tokens.contiguous(), lengths.contiguous()
    actions = KaggricultureActions(tokens, lengths)
    expected = ppo._actions_to_cpu(actions)
    transfer = PinnedActionTransfer(2, torch.device("cpu"))
    actual = transfer.to_cpu(actions)
    assert isinstance(expected, KaggricultureActions)
    for got, want in (
        (actual.tokens, expected.tokens),
        (actual.lengths, expected.lengths),
    ):
        assert got.is_contiguous()
        assert got.numpy().tobytes() == want.numpy().tobytes()
    assert transfer.buffers is None
    assert transfer.ready is None


def test_pinned_action_transfer_cuda_orders_both_copies_before_native_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fake only CUDA allocation/transport. The constructor, admission, event
    # sequencing and _step_env call are the real code, runnable on CPU hosts.
    log: list[object] = []

    class FakeTensor:
        def __init__(self, shape: tuple[int, ...], device: str) -> None:
            self.shape = shape
            self.dtype = torch.int64
            self.device = torch.device(device)

        def copy_(self, source: FakeTensor, *, non_blocking: bool) -> FakeTensor:
            log.append(("copy", self.shape, source.device, non_blocking))
            return self

    class FakeEvent:
        def record(self, stream: object) -> None:
            log.append(("record", stream))

        def synchronize(self) -> None:
            log.append("synchronize")

    def allocate(
        shape: tuple[int, ...], *, dtype: torch.dtype, pin_memory: bool
    ) -> FakeTensor:
        assert dtype == torch.int64
        assert pin_memory
        log.append(("allocate", shape))
        return FakeTensor(shape, "cpu")

    def current_stream(device: torch.device) -> str:
        assert device == torch.device("cuda:1")
        return "producer-stream"

    class Reader:
        def step(self, actions: KaggricultureActions) -> tuple[object, ...]:
            assert log[-1] == "synchronize"
            log.append("native-read")
            return (actions,)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    transfer = PinnedActionTransfer(2, torch.device("cuda"))
    assert transfer.device == torch.device("cuda:1")
    buffers = transfer.buffers
    shapes = ((2, 2, 252, 12), (2, 2))
    assert log == [("allocate", shape) for shape in shapes]
    for _ in range(2):
        actions = KaggricultureActions(
            FakeTensor(shapes[0], "cuda:1"),  # type: ignore[arg-type]
            FakeTensor(shapes[1], "cuda:1"),  # type: ignore[arg-type]
        )
        result = ppo._step_env(Reader(), actions, action_transfer=transfer)  # type: ignore[arg-type]
        assert result[0] is buffers
    one_transfer: list[object] = [
        *(("copy", shape, torch.device("cuda:1"), True) for shape in shapes),
        ("record", "producer-stream"),
        "synchronize",
        "native-read",
    ]
    assert log[2:] == one_transfer * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_action_transfer_cuda_exact_bytes_and_reused_storage() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    transfer = PinnedActionTransfer(2, torch.device("cuda"))
    pointers: tuple[int, int] | None = None
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        for offset in (0, 113):
            expected = KaggricultureActions(
                torch.arange(2 * 2 * 252 * 12).reshape(2, 2, 252, 12) + offset,
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int64) + offset,
            )
            actions = KaggricultureActions(
                torch.arange(2 * 2 * 252 * 12, device=device).reshape(2, 2, 252, 12)
                + offset,
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device=device)
                + offset,
            )
            actual = transfer.to_cpu(actions)
            assert actual.tokens.is_pinned()
            assert actual.lengths.is_pinned()
            assert torch.equal(actual.tokens, expected.tokens)
            assert torch.equal(actual.lengths, expected.lengths)
            current = (actual.tokens.data_ptr(), actual.lengths.data_ptr())
            if pointers is not None:
                assert current == pointers
            pointers = current


def test_pinned_action_transfer_rejects_wrong_shapes_dtype_and_device() -> None:
    transfer = PinnedActionTransfer(2, torch.device("cpu"))
    for tokens, lengths, match in (
        (torch.zeros(2, 2, 252, 12), torch.zeros(2, 2, dtype=torch.int64), "int64"),
        (
            torch.zeros(2, 2, 252, 12, dtype=torch.int64),
            torch.zeros(1, 2, dtype=torch.int64),
            "shape",
        ),
        (
            torch.zeros(2, 2, 252, 12, dtype=torch.int64, device="meta"),
            torch.zeros(2, 2, dtype=torch.int64),
            "transfer device",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            transfer.to_cpu(KaggricultureActions(tokens, lengths))


def test_rollout_switches_default_preserves_prechange_config_hash() -> None:
    cfg = FullConfig.from_file(Path("configs/kaggriculture.yaml"))
    # Recorded on the task's base before any optimization was edited.
    assert config_sha256(cfg) == (
        "79ab336b7e60bd6284185a34cb025f92efaa611aae129e532f75d1b4eb504f3a"
    )
    explicit = FullConfig.from_file(
        Path("configs/kaggriculture.yaml"),
        {f"rl.{flag}": False for flag in _FLAGS}
        | {"env.skip_reward_telemetry_validation": False},
    )
    assert explicit.model_dump_json() == cfg.model_dump_json()
    assert not set(_FLAGS).intersection(cfg.model_dump()["rl"])
    enabled = FullConfig.from_file(
        Path("configs/kaggriculture.yaml"), {f"rl.{flag}": True for flag in _FLAGS}
    )
    assert all(enabled.model_dump()["rl"][flag] is True for flag in _FLAGS)
    assert FullConfig.model_validate(enabled.model_dump()) == enabled
    assert config_sha256(enabled) != config_sha256(cfg)


@pytest.mark.parametrize("flag", _FLAGS)
def test_rollout_switches_reject_orbit_config(flag: str) -> None:
    with pytest.raises(ValueError, match="Kaggriculture-only"):
        FullConfig.from_file(Path("configs/scaling_6m.yaml"), {f"rl.{flag}": True})


@pytest.mark.usefixtures("probed_compile_stack")
def test_configure_actor_only_claims_atten_backend_before_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def compile_core(fn: Any, **kwargs: object) -> Any:
        assert inductor_config.max_autotune_gemm_backends == "ATEN"
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", compile_core)
    model = _tiny_model()
    count = configure_model_compile(
        model, PPOConfig(model_compile="none", compile_actor_heads=True)
    )
    assert count == 1
    assert calls == [
        (
            model.actor.policy_core,
            {"mode": "max-autotune-no-cudagraphs", "fullgraph": True, "dynamic": False},
        )
    ]
    assert model._compiled_transformer_trunk is None
    assert model._compiled_actor_core is not None
    assert model.compiled_regions_require_gemm_backends
    claim = compile_gemm.gemm_backend_claim()
    assert claim is not None
    assert claim.gemm_backends == "ATEN"


def test_startup_stack_check_covers_actor_only_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    bad_stack = compile_gemm.InstalledCompileStack(
        torch="0.0.0", triton=None, cuda_available=False, nvidia_drivers=()
    )
    monkeypatch.setattr(run_ppo, "installed_compile_stack", lambda: bad_stack)
    distributed = DistributedContext(torch.device("cpu"), 0, 0, 1, False)
    with pytest.raises(RuntimeError, match="torch"):
        run_ppo._check_compile_stack(
            cfg.model,
            rl=PPOConfig(model_compile="none", compile_actor_heads=True),
            distributed=distributed,
        )
    assert (
        run_ppo._check_compile_stack(
            cfg.model, rl=PPOConfig(model_compile="none"), distributed=distributed
        )
        is None
    )


def test_two_native_ppo_updates_are_exact_with_rollout_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_forward = ppo._model_forward
    outputs: list[dict[str, bytes]] = []
    packing_contexts: list[bool] = []
    original_transfer = PinnedActionTransfer.to_cpu
    transfers: list[int] = []

    def transfer(
        self: PinnedActionTransfer, actions: KaggricultureActions
    ) -> KaggricultureActions:
        transfers.append(id(self))
        return original_transfer(self, actions)

    def forward(*args: Any, **kwargs: Any) -> Any:
        packing_contexts.append(args[0]._rollout_token_mask_cpu is not None)
        output = original_forward(*args, **kwargs)
        assert isinstance(output.actions, KaggricultureActions)
        tensors = {
            "tokens": output.actions.tokens,
            "lengths": output.actions.lengths,
            "logp": output.log_probs.per_player_entity,
            "logp_launch": output.log_probs.launch,
            "logp_event": output.log_probs.event,
            "entropy": output.entropies.per_player_entity,
            "entropy_launch": output.entropies.launch,
            "entropy_event": output.entropies.event,
            "values": output.values,
            **output.entropies.components,
        }
        outputs.append({key: t.numpy().tobytes() for key, t in tensors.items()})
        return output

    monkeypatch.setattr(ppo, "_model_forward", forward)
    monkeypatch.setattr(PinnedActionTransfer, "to_cpu", transfer)
    results = []
    for fast in (False, True):
        torch.manual_seed(317)
        # A deterministic clock isolates all metric arithmetic from wall time.
        ticks = count()
        monkeypatch.setattr(ppo, "perf_counter", lambda ticks=ticks: float(next(ticks)))
        model = _tiny_model()
        env = _env(_config(fast))
        trainer = _trainer(env, model, rollout_packing=fast, pinned_action_d2h=fast)
        outputs.clear()
        packing_contexts.clear()
        transfers.clear()
        metrics = [trainer.train_iteration() for _ in range(2)]
        assert packing_contexts == [fast] * 4
        assert len(transfers) == (4 if fast else 0)
        if fast:
            assert len(set(transfers)) == 1
        rollout = trainer.rollout
        assert isinstance(rollout.actions, KaggricultureActions)
        state = {
            name: tensor.detach().numpy().tobytes()
            for name, tensor in model.state_dict().items()
        }
        buffers = {
            name: tensor.numpy().tobytes()
            for name, tensor in {
                "tokens": rollout.actions.tokens,
                "lengths": rollout.actions.lengths,
                "logp": rollout.logp,
                "entity_logp": rollout.entity_logp,
                "values": rollout.values,
                "rewards": rollout.rewards,
                "dones": rollout.dones,
            }.items()
        }
        results.append((state, buffers, metrics, list(outputs), torch.get_rng_state()))
    before, after = results
    assert before[:-1] == after[:-1]
    assert torch.equal(before[-1], after[-1])
    assert before[3], "the canonical rollout forward must actually be observed"
