"""Stage 1 adapter ownership checks; fake calls do not prove native rollback."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from owl import rs
from owl.kaggriculture.env import (
    KaggricultureVectorizedEnv,
    allocate_observation_buffers,
)
from owl.kaggriculture.rewards import KaggricultureRewardConfig
from owl.kaggriculture.types import (
    KaggricultureActionConfig,
    KaggricultureActionMask,
    KaggricultureActions,
    KaggricultureGameConfig,
    KaggricultureObsBatch,
    KaggricultureObsConfig,
)

from tests.kaggriculture.fake_env import FakeKaggricultureEnv
from tests.kaggriculture.test_observe import _arrays, _header, _tensors, _write

# Independent transcription of the shared ABI: no production schema-derived oracle.
OUTPUTS = {
    "tile_kind": (np.int64, (2, 2, 200)),
    "tile_crop": (np.int64, (2, 2, 200)),
    "tile_animal": (np.int64, (2, 2, 200)),
    "tile_cell": (np.int64, (2, 2, 200)),
    "tile_role": (np.int64, (2, 2, 200)),
    "tiles_int": (np.int64, (2, 2, 200, 7)),
    "tiles_float": (np.float32, (2, 2, 200, 15)),
    "actor_slot": (np.int64, (2, 2, 482)),
    "actor_cell": (np.int64, (2, 2, 482)),
    "actor_role": (np.int64, (2, 2, 482)),
    "actor_mask": (np.bool_, (2, 2, 482)),
    "actor_inventory": (np.int64, (2, 2, 241, 12)),
    "actor_inventory_rank": (np.int64, (2, 2, 241, 12)),
    "actors_float": (np.float32, (2, 2, 482, 26)),
    "player_features": (np.float32, (2, 2, 2, 44)),
    "storage_counts": (np.int64, (2, 2, 17)),
    "storage_rank": (np.int64, (2, 2, 12)),
    "banks": (np.float64, (2, 2, 2)),
    "shop_type": (np.int64, (2, 2, 8)),
    "shop_slot": (np.int64, (2, 2, 8)),
    "shop_mask": (np.bool_, (2, 2, 8)),
    "market_product": (np.int64, (2, 2, 9)),
    "market_float": (np.float32, (2, 2, 9, 2)),
    "market_int": (np.int64, (2, 2, 9, 2)),
    "global_features": (np.float32, (2, 2, 15)),
    "globals_int": (np.int64, (2, 2, 16)),
    "still_playing": (np.bool_, (2, 2)),
    "order_limits": (np.int64, (2, 2)),
    "can_act": (np.bool_, (2, 2, 252)),
    "rewards": (np.float32, (2, 2)),
    "dones": (np.bool_, (2, 2)),
    "transition_banks_before": (np.float64, (2, 2)),
    "transition_banks_after": (np.float64, (2, 2)),
    "transition_econ_before": (np.int64, (2, 2, 32)),
    "transition_econ_after": (np.int64, (2, 2, 32)),
}


def reward_config():
    return KaggricultureRewardConfig(
        econ_shaping=0.2,
        econ_starvation_weight=4,
        econ_drought_weight=1,
        econ_cap=0.25,
        econ_ineffective_weight=0,
        econ_ineffective_cap=0.1,
    )


_CPU_DEVICE = torch.device("cpu")


def make_env(*, pin_memory=False, transfer_device=_CPU_DEVICE):
    return KaggricultureVectorizedEnv(
        n_envs=2,
        seed=41,
        seed_stride=2,
        config=KaggricultureGameConfig(),
        reward_config=reward_config(),
        reward_mode="win_loss",
        native_threads=1,
        pin_memory=pin_memory,
        transfer_device=transfer_device,
        obs_spec=KaggricultureObsConfig(),
        action_spec=KaggricultureActionConfig(),
    )


def pass_actions(n_envs=2):
    tokens = torch.zeros((n_envs, 2, 252, 12), dtype=torch.int64)
    tokens[:, :, 0, 1] = 1
    tokens[:, :, 1, 11] = 1
    return KaggricultureActions(tokens, torch.full((n_envs, 2), 2, dtype=torch.int64))


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(rs, "KaggricultureEnv", FakeKaggricultureEnv, raising=False)
    return make_env()


def output_tensors(env):
    return {
        **_tensors(env.observations),
        "rewards": env.rewards,
        "dones": env.dones,
        "transition_banks_before": env.transition_banks_before,
        "transition_banks_after": env.transition_banks_after,
        "transition_econ_before": env.transition_econ_before,
        "transition_econ_after": env.transition_econ_after,
    }


def test_every_batch_checks_contract(fake):
    env = fake
    obs = env.observations
    assert isinstance(obs, KaggricultureObsBatch)
    assert isinstance(obs.action_mask, KaggricultureActionMask)
    obs.check_contract()
    tensors = output_tensors(env)
    ptrs = {key: value.data_ptr() for key, value in tensors.items()}
    native = env._native
    assert native.calls == ["observe"]  # construction never consumes another reset
    assert env.reset() is obs
    obs.check_contract()
    result, rewards, dones, metrics = env.step(pass_actions())
    assert result is obs
    assert rewards is env.rewards
    assert dones is env.dones
    assert metrics == {"fake_metric": [1.0]}
    obs.check_contract()
    transition_before = {
        k: v.clone() for k, v in tensors.items() if k not in _tensors(obs)
    }
    unselected = {k: v[1].clone() for k, v in _tensors(obs).items()}
    assert env.truncate_envs(torch.tensor([True, False])) is obs
    obs.check_contract()
    for key, tensor in tensors.items():
        assert tensor.data_ptr() == ptrs[key]
    for key, tensor in transition_before.items():
        assert torch.equal(tensors[key], tensor)
    for key, tensor in unselected.items():
        assert torch.equal(tensors[key][1], tensor)
    first = native.outputs[0]
    assert set(first) == set(OUTPUTS)
    assert len(first) == 35
    for arrays in native.outputs:
        for name, (dtype, shape) in OUTPUTS.items():
            arr = arrays[name]
            assert arr is first[name]
            assert arr.dtype == dtype
            assert arr.shape == shape
            assert arr.flags.c_contiguous
            assert arr.flags.writeable
            assert arr.flags.aligned
            assert arr.__array_interface__["data"][0] == tensors[name].data_ptr()
    for index, left in enumerate(first.values()):
        for right in list(first.values())[index + 1 :]:
            assert not np.shares_memory(left, right)
    assert env.n_envs == 2
    assert env.reward_mode == "win_loss"
    assert env.obs_spec == KaggricultureObsConfig()
    assert env.action_spec == KaggricultureActionConfig()
    assert env.terminal_metrics(1) is None
    assert env.state_snapshot(0) == {"public": {"step": 0}}
    assert env.seed_state() == native.seed_state()
    assert native.hire_limit == env.action_spec.hire_limit
    assert json.loads(native.config) == json.loads(
        KaggricultureGameConfig().to_native_json()
    )
    assert native.reward_config == reward_config().to_native_dict("win_loss")


def test_action_and_mask_inputs_are_zero_copy_views(fake, monkeypatch):
    original = torch.Tensor.numpy
    seen = []

    def numpy_spy(tensor):
        seen.append(tensor)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "numpy", numpy_spy)
    for _ in range(2):
        actions = pass_actions()
        mask = torch.tensor([False, True])
        fake.step(actions)
        for view, tensor in zip(
            fake._native.inputs[-1], (actions.tokens, actions.lengths), strict=True
        ):
            assert view.__array_interface__["data"][0] == tensor.data_ptr()
        fake.truncate_envs(mask)
        assert (
            fake._native.inputs[-1][0].__array_interface__["data"][0] == mask.data_ptr()
        )
    assert len(seen) == 6  # exactly one view per fresh input, no output views


@pytest.mark.parametrize(
    "bad",
    [
        "type",
        "dtype",
        "device",
        "shape",
        "stride",
        "length_dtype",
        "length_shape",
        "length_stride",
        "length_device",
    ],
)
def test_invalid_actions_rejected(fake, bad):
    actions = pass_actions()
    if bad == "type":
        actions = (actions.tokens, actions.lengths)
    elif bad == "dtype":
        actions.tokens = actions.tokens.float()
    elif bad == "device":
        actions.tokens = actions.tokens.to("meta")
    elif bad == "shape":
        actions.tokens = actions.tokens[:, :, :251]
    elif bad == "stride":
        actions.tokens = torch.zeros((2, 2, 252, 24), dtype=torch.int64)[..., ::2]
    elif bad == "length_dtype":
        actions.lengths = actions.lengths.bool()
    elif bad == "length_shape":
        actions.lengths = actions.lengths[:, :1]
    elif bad == "length_stride":
        actions.lengths = torch.zeros((2, 4), dtype=torch.int64)[:, ::2]
    elif bad == "length_device":
        actions.lengths = actions.lengths.to("meta")
    with pytest.raises((TypeError, ValueError)):
        fake.step(actions)
    assert fake._native.calls == ["observe"]


@pytest.mark.parametrize(
    "mask",
    [
        [True, False],
        torch.ones(2),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones(4, dtype=torch.bool)[::2],
        torch.ones(2, dtype=torch.bool, device="meta"),
    ],
)
def test_invalid_masks_rejected(fake, mask):
    with pytest.raises((TypeError, ValueError)):
        fake.truncate_envs(mask)
    assert fake._native.calls == ["observe"]


def test_requested_pinning_allocator_spy(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rs, "KaggricultureEnv", FakeKaggricultureEnv, raising=False)
    original = torch.empty
    requests = []

    def allocate(*args, pin_memory=False, **kwargs):
        requests.append(pin_memory)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _tensor: True)
    env = make_env(pin_memory=True)
    assert requests == [True] * 35
    assert env.pin_memory_enabled


def test_requested_pinning_unavailable_fails_fast(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rs, "KaggricultureEnv", FakeKaggricultureEnv, raising=False)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("pin allocator unavailable")

    monkeypatch.setattr(torch, "empty", unavailable)
    with pytest.raises(RuntimeError, match="pin allocator unavailable"):
        make_env(pin_memory=True)


def test_requested_pinning_silent_unpinned_allocation_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original(
            *args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}
        ),
    )
    with pytest.raises(RuntimeError, match="pinned"):
        allocate_observation_buffers(1, pin_memory=True)


def test_fence_precedes_every_native_write(fake, monkeypatch):
    events = fake._native.events
    monkeypatch.setattr(fake, "_fence", lambda: events.append("fence"))
    events.clear()
    fake.reset()
    fake.step(pass_actions())
    fake.truncate_envs(torch.tensor([True, False]))
    assert events == [
        "fence",
        "native.reset",
        "fence",
        "native.step",
        "fence",
        "native.truncate_envs",
    ]


@pytest.mark.parametrize(
    ("device", "pinned", "expected"),
    [("cpu", False, 0), ("cpu", True, 0), ("cuda", False, 0), ("cuda", True, 3)],
)
def test_fence_condition(fake, monkeypatch, device, pinned, expected):
    fake.transfer_device = torch.device(device)
    fake._pin_memory_enabled = pinned
    events = []

    class Stream:
        def synchronize(self):
            events.append("sync")

    def current_stream(device):
        events.append(device)
        return Stream()

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("unexpected CUDA availability probe"),
    )
    fake.reset()
    fake.step(pass_actions())
    fake.truncate_envs(torch.tensor([False, True]))
    assert events == [torch.device(device), "sync"] * expected


@pytest.mark.parametrize("operation", ["reset", "step", "truncate"])
def test_failed_native_call_preserves_buffers(fake, operation):
    before = {
        key: tensor.numpy().tobytes() for key, tensor in output_tensors(fake).items()
    }
    fake._native.fail = True
    operations = {
        "reset": fake.reset,
        "step": lambda: fake.step(pass_actions()),
        "truncate": lambda: fake.truncate_envs(torch.tensor([True, False])),
    }
    with pytest.raises(ValueError, match="injected native failure"):
        operations[operation]()
    for key, tensor in output_tensors(fake).items():
        assert tensor.numpy().tobytes() == before[key]


def test_seat_private_isolation():
    header = _header()
    changed = copy.deepcopy(header)
    changed["initial"]["privates"][1]["inventories"][0] = {"MILK": 11}
    batches = [allocate_observation_buffers(1, pin_memory=False) for _ in range(2)]
    for batch, source in zip(batches, [header, changed], strict=True):
        _write(json.dumps([source]), _arrays(batch))
        batch.check_contract()
    before, after = map(_tensors, batches)
    for name in before:
        assert torch.equal(before[name][:, 0], after[name][:, 0]), name
    assert not torch.equal(
        before["actor_inventory"][:, 1], after["actor_inventory"][:, 1]
    )
    assert not torch.equal(before["actors_float"][:, 1], after["actors_float"][:, 1])


@pytest.mark.skip(reason="needs Task 1.4 binding")
def test_seat_private_isolation_live_actions():
    env = make_env()
    actions = pass_actions()
    actions.tokens[:, 0, 0, 1] = 4  # EAST for seat 0; seat 1 passes
    old = env.observations.actor_cell.clone()
    env.step(actions)
    env.observations.check_contract()
    assert not torch.equal(env.observations.actor_cell[:, 0, 0], old[:, 0, 0])
    assert torch.equal(env.observations.actor_cell[:, 1, 0], old[:, 1, 0])


def test_requested_pinning_without_cuda_rejected_before_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *_args, **_kwargs: pytest.fail("unsafe pinned allocation"),
    )
    with pytest.raises(RuntimeError, match="pinned memory"):
        allocate_observation_buffers(1, pin_memory=True)
