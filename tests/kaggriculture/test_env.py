"""Adapter ownership, fake failure injection and real native lifecycle checks."""

from __future__ import annotations

import copy
import json
from dataclasses import fields

import numpy as np
import pytest
import torch
from owl import rs
from owl.kaggriculture.codec import encode_actions
from owl.kaggriculture.env import (
    KaggricultureVectorizedEnv,
    allocate_observation_buffers,
)
from owl.kaggriculture.rewards import (
    KaggricultureRewardConfig,
    bank_rewards,
    margin_rewards,
)
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
        econ_bank_weight=0.0,
        econ_bank_scale=100_000.0,
        econ_bank_cap=0.0,
        econ_margin_weight=0.0,
        econ_margin_scale=50_000.0,
        econ_margin_cap=0.0,
    )


_CPU_DEVICE = torch.device("cpu")


def make_env(
    *, pin_memory=False, transfer_device=_CPU_DEVICE, config=None, reward=None
):
    return KaggricultureVectorizedEnv(
        n_envs=2,
        seed=41,
        seed_stride=2,
        config=KaggricultureGameConfig() if config is None else config,
        reward_config=reward_config() if reward is None else reward,
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
    assert metrics == {
        "fake_metric": [1.0],
        "reward_bank_mean": [0.0],
        "reward_margin_abs_mean": [0.0],
    }
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
    # 29 observation + 6 transition buffers + the learner mask.
    assert requests == [True] * 36
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


def test_seat_private_isolation_live_actions():
    env = make_env()
    actions = native_actions(env, farmer=["EAST"])
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


@pytest.mark.parametrize("n_envs", [0, -1, True, 1.0, "1"])
def test_allocation_rejects_invalid_env_count_before_allocating(monkeypatch, n_envs):
    monkeypatch.setattr(
        torch, "empty", lambda *_args, **_kwargs: pytest.fail("allocated buffers")
    )
    with pytest.raises(ValueError, match="n_envs must be a positive integer"):
        allocate_observation_buffers(n_envs, pin_memory=False)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_envs", 0),
        ("n_envs", True),
        ("n_envs", 2.0),
        ("seed", -1),
        ("seed", 2**63),
        ("seed", False),
        ("seed_stride", 0),
        ("seed_stride", 2**63),
        ("native_threads", 0),
        ("native_threads", 1.5),
    ],
)
def test_constructor_rejects_invalid_integers_before_allocation_and_native(
    monkeypatch, name, value
):
    def fail_native(*_args, **_kwargs):
        pytest.fail("native environment constructed")

    monkeypatch.setattr(rs, "KaggricultureEnv", fail_native, raising=False)
    monkeypatch.setattr(
        torch, "empty", lambda *_args, **_kwargs: pytest.fail("allocated buffers")
    )
    arguments = {
        "n_envs": 2,
        "seed": 41,
        "seed_stride": 2,
        "native_threads": 1,
    } | {name: value}
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        KaggricultureVectorizedEnv(
            **arguments,
            config=KaggricultureGameConfig(),
            reward_config=reward_config(),
            reward_mode="win_loss",
            pin_memory=False,
            transfer_device=_CPU_DEVICE,
            obs_spec=KaggricultureObsConfig(),
            action_spec=KaggricultureActionConfig(),
        )


def native_actions(env, *, farmer=None, market=None):
    """Cold native encoding uses only each seat's current legal observation."""
    programs = []
    for i in range(env.n_envs):
        seats = []
        for seat in range(2):
            actors = int(env.observations.actor_mask[i, seat, :241].sum())
            seats.append(
                {
                    "farmer": ["PASS"] if farmer is None or seat else farmer,
                    "hands": [["PASS"] for _ in range(actors - 1)],
                    "market": [] if market is None or seat else market,
                }
            )
        programs.append((seats[0], seats[1]))
    return encode_actions(programs, env.observations, action_spec=env.action_spec)


def test_real_binding_every_batch_checks_contract():
    env = make_env()
    assert type(env._native) is rs.KaggricultureEnv
    obs = env.observations
    tensors = output_tensors(env)
    arrays = {
        field.name: getattr(env._arrays, field.name) for field in fields(env._arrays)
    }
    assert set(arrays) == set(tensors) == set(OUTPUTS)
    assert len(arrays) == 35
    assert env.seed_state() == (45, (41, 43))
    assert not env.rewards.any()
    assert not env.dones.any()
    assert not env.transition_econ_before.any()
    assert not env.transition_econ_after.any()
    assert torch.equal(env.transition_banks_before, env.transition_banks_after)
    for name, (dtype, shape) in OUTPUTS.items():
        assert arrays[name].dtype == dtype
        assert arrays[name].shape == shape
        assert arrays[name].flags.c_contiguous
        assert arrays[name].__array_interface__["data"][0] == tensors[name].data_ptr()
    for operation in (
        lambda: env.observations,
        env.reset,
        lambda: env.step(native_actions(env))[0],
        lambda: env.truncate_envs(torch.tensor([False, True])),
    ):
        assert operation() is obs
        obs.check_contract()
        for name, tensor in output_tensors(env).items():
            assert tensor is tensors[name]
            assert getattr(env._arrays, name) is arrays[name]
            assert tensor.data_ptr() == arrays[name].__array_interface__["data"][0]
    assert env.seed_state() == (51, (45, 49))
    assert env.state_snapshot(0)["public"]["step"] == 1
    assert env.state_snapshot(1)["public"]["step"] == 0
    assert env.terminal_metrics(0) is None
    assert env.terminal_metrics(1) is None


def test_real_binding_terminal_step_keeps_completed_transition_and_new_observation():
    env = make_env(config=KaggricultureGameConfig(episode_steps=3))
    obs = env.observations
    result, rewards, dones, metrics = env.step(
        native_actions(env, market=[["BUY_PRODUCT", "WHEAT", 1]])
    )
    assert result is obs
    assert rewards is env.rewards
    assert dones is env.dones
    assert not dones.any()
    # The adapter's own telemetry values are zero with the bank/margin terms off.
    assert metrics.pop("reward_bank_mean") == [0.0]
    assert metrics.pop("reward_margin_abs_mean") == [0.0]
    assert all(value == [] for value in metrics.values())
    assert all(env.terminal_metrics(i) is None for i in range(2))
    final_banks = env.transition_banks_after.clone()
    result, rewards, dones, metrics = env.step(native_actions(env))
    obs.check_contract()
    assert result is obs
    assert rewards is env.rewards
    assert dones is env.dones
    assert dones.all()
    assert env.seed_state() == (49, (45, 47))
    assert not obs.globals_int[..., 0].any()
    assert obs.still_playing.all()
    assert torch.equal(env.transition_banks_after, final_banks)
    assert not torch.equal(obs.banks[..., 0], final_banks)
    records = [env.terminal_metrics(i) for i in range(2)]
    for i, record in enumerate(records):
        assert record is not None
        assert record["episode_steps"] == 2
        assert record["winner"] == 1
        assert [record["bank_0"], record["bank_1"]] == final_banks[i].tolist()
        assert record["margin_0"] == record["bank_0"] - record["bank_1"]
        np.testing.assert_array_equal(record["econ_0"], env.transition_econ_after[i, 0])
        np.testing.assert_array_equal(record["econ_1"], env.transition_econ_after[i, 1])
        assert env.state_snapshot(i)["public"]["step"] == 0
    assert metrics == {
        "total_games_played": [1.0, 1.0],
        "terminal_bank_0": final_banks[:, 0].tolist(),
        "terminal_bank_1": final_banks[:, 1].tolist(),
        "terminal_margin_0": (final_banks[:, 0] - final_banks[:, 1]).tolist(),
        "reward_bank_mean": [0.0],
        "reward_margin_abs_mean": [0.0],
    }
    assert env.reset() is obs
    obs.check_contract()
    assert env.seed_state() == (53, (49, 51))
    assert not env.rewards.any()
    assert not env.dones.any()
    assert not env.transition_econ_before.any()
    assert not env.transition_econ_after.any()
    assert all(env.terminal_metrics(i) is None for i in range(2))


@pytest.mark.parametrize("episode_steps", [2, 4])
@pytest.mark.parametrize("selected", [(False, True), (True, False), (False, False)])
def test_real_binding_truncate_preserves_unselected_and_all_transitions(
    episode_steps, selected
):
    from .test_native_env import buffers

    env = make_env(config=KaggricultureGameConfig(episode_steps=episode_steps))
    env.step(
        native_actions(env, farmer=["HARVEST"], market=[["BUY_PRODUCT", "WHEAT", 1]])
    )
    assert env.dones.all().item() == (episode_steps == 2)
    assert env.transition_econ_after[..., 2].any()
    obs = env.observations
    before = {
        name: tensor.numpy().tobytes() for name, tensor in output_tensors(env).items()
    }
    rows = {
        name: [tensor[i].numpy().tobytes() for i in range(2)]
        for name, tensor in _tensors(obs).items()
    }
    states = [env.state_snapshot(i) for i in range(2)]
    terminals = [env.terminal_metrics(i) for i in range(2)]
    next_seed, seeds = env.seed_state()
    expected_seeds = list(seeds)
    for i, reset in enumerate(selected):
        if reset:
            expected_seeds[i] = next_seed
            next_seed += 2
    assert env.truncate_envs(torch.tensor(selected)) is obs
    obs.check_contract()
    assert env.seed_state() == (next_seed, tuple(expected_seeds))
    fresh = buffers(2)
    env._native.observe(**fresh)
    for name, tensor in output_tensors(env).items():
        if name not in _tensors(obs):
            assert tensor.numpy().tobytes() == before[name], name
        else:
            for i, reset in enumerate(selected):
                expected = fresh[name][i].tobytes() if reset else rows[name][i]
                assert tensor[i].numpy().tobytes() == expected, (name, i)
    for i, reset in enumerate(selected):
        if reset:
            assert env.state_snapshot(i)["public"]["step"] == 0
            assert env.terminal_metrics(i) is None
        else:
            assert env.state_snapshot(i) == states[i]
            terminal = env.terminal_metrics(i)
            if terminals[i] is None:
                assert terminal is None
            else:
                assert terminal is not None
                for name, value in terminals[i].items():
                    np.testing.assert_array_equal(terminal[name], value)


def test_real_binding_invalid_action_preserves_all_35_buffers_and_diagnostics():
    env = make_env(config=KaggricultureGameConfig(episode_steps=2))
    env.step(native_actions(env, farmer=["HARVEST"]))
    tensors = output_tensors(env)
    before = {name: tensor.numpy().tobytes() for name, tensor in tensors.items()}
    states = [env.state_snapshot(i) for i in range(2)]
    seeds = env.seed_state()
    terminals = [env.terminal_metrics(i) for i in range(2)]
    actions = native_actions(env)
    actions.tokens[1, 1, 0, 1] = 2**62
    with pytest.raises(ValueError, match=r"env=1.*seat=1"):
        env.step(actions)
    assert len(before) == 35
    for name, tensor in output_tensors(env).items():
        assert tensor is tensors[name]
        assert tensor.numpy().tobytes() == before[name], name
    assert env.seed_state() == seeds
    assert [env.state_snapshot(i) for i in range(2)] == states
    for i, old in enumerate(terminals):
        current = env.terminal_metrics(i)
        assert old is not None
        assert current is not None
        for name, value in old.items():
            np.testing.assert_array_equal(current[name], value)


def test_real_binding_reports_the_own_bank_reward_mean_per_step():
    # Owner term A on: the adapter's telemetry value is the oracle's mean own
    # bank increment over every seat of the completed transitions.
    reward = reward_config().model_copy(
        update={"econ_bank_weight": 1.0, "econ_bank_cap": 0.25}
    )
    env = make_env(config=KaggricultureGameConfig(episode_steps=3), reward=reward)
    _, _, _, metrics = env.step(
        native_actions(env, market=[["BUY_PRODUCT", "WHEAT", 1]])
    )
    increments = bank_rewards(
        env.transition_banks_before, env.transition_banks_after, reward
    )
    assert increments.any(), "the purchase must move a bank"
    assert metrics["reward_bank_mean"] == [float(increments.mean())]
    _, _, dones, metrics = env.step(native_actions(env))
    assert dones.all()
    assert metrics["reward_bank_mean"] == [
        float(
            bank_rewards(
                env.transition_banks_before, env.transition_banks_after, reward
            ).mean()
        )
    ]


def test_real_binding_reports_the_margin_reward_abs_mean_per_step() -> None:
    # Owner term M on: the adapter's telemetry value is the oracle's mean
    # absolute margin increment over every seat (the signed mean is zero).
    reward = reward_config().model_copy(
        update={
            "econ_shaping": 0.0,
            "econ_margin_weight": 0.5,
            "econ_margin_cap": 0.5,
        }
    )
    assert reward.terminal_scale == 0.5
    env = make_env(config=KaggricultureGameConfig(episode_steps=3), reward=reward)
    _, rewards, _, metrics = env.step(
        native_actions(env, market=[["BUY_PRODUCT", "WHEAT", 1]])
    )
    increments = margin_rewards(
        env.transition_banks_before, env.transition_banks_after, reward
    )
    assert increments.any(), "seat 0's purchase must move the margin"
    assert torch.equal(increments[..., 0], -increments[..., 1])
    assert torch.equal(rewards, increments.float())
    assert metrics["reward_margin_abs_mean"] == [float(increments.abs().mean())]
