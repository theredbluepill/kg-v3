"""Native lifecycle acceptance on one or two games; no adapter implementation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from owl import rs
from owl.kaggriculture.types import _SCHEMA

from .test_observe import _allocate, _arrays

REWARD: rs.KaggricultureRewardDict = {
    "reward_mode": "win_loss",
    "econ_shaping": 0.2,
    "econ_starvation_weight": 4.0,
    "econ_drought_weight": 1.0,
    "econ_cap": 0.25,
    "econ_ineffective_weight": 0.0,
    "econ_ineffective_cap": 0.1,
}
TRANSITIONS = (
    "rewards",
    "dones",
    "transition_banks_before",
    "transition_banks_after",
    "transition_econ_before",
    "transition_econ_after",
)
DESTINATIONS = (*_SCHEMA, "can_act", *TRANSITIONS)


def buffers(n_envs: int) -> dict[str, Any]:
    """Allocate from the existing real schema, with six ABI transition outputs."""
    arrays = _arrays(_allocate(n_envs))
    arrays.update(
        rewards=np.full((n_envs, 2), -17, dtype=np.float32),
        dones=np.ones((n_envs, 2), dtype=np.bool_),
        transition_banks_before=np.full((n_envs, 2), -17, dtype=np.float64),
        transition_banks_after=np.full((n_envs, 2), -17, dtype=np.float64),
        transition_econ_before=np.full((n_envs, 2, 32), -17, dtype=np.int64),
        transition_econ_after=np.full((n_envs, 2, 32), -17, dtype=np.int64),
    )
    return arrays


def make_env(
    n_envs: int = 1,
    seed: int = 11,
    stride: int = 2,
    config: str = "{}",
    threads: int = 1,
) -> rs.KaggricultureEnv:
    return rs.KaggricultureEnv(
        n_envs, seed, stride, config, REWARD, threads, hire_limit=241
    )


def snapshot(arrays: dict[str, Any]) -> dict[str, bytes]:
    return {name: array.tobytes() for name, array in arrays.items()}


def test_constructor_observe_does_not_consume_reset_seed() -> None:
    env = make_env(2)
    batch = _allocate(2)
    arrays = buffers(2)
    arrays.update(_arrays(batch))
    assert env.seed_state() == (15, (11, 13))
    env.observe(**arrays)
    batch.check_contract()
    assert env.seed_state() == (15, (11, 13))
    assert not arrays["dones"].any()
    assert not arrays["rewards"].any()
    np.testing.assert_array_equal(
        arrays["transition_banks_before"], arrays["transition_banks_after"]
    )
    assert not arrays["transition_econ_before"].any()
    assert not arrays["transition_econ_after"].any()
    env.reset(**arrays)
    assert env.seed_state() == (19, (15, 17))
    batch.check_contract()


@pytest.mark.parametrize("name", DESTINATIONS)
@pytest.mark.parametrize(
    "kind", ["shape", "dtype", "readonly", "stride", "fortran", "unaligned", "overlap"]
)
def test_every_destination_rejected_before_any_write(name: str, kind: str) -> None:
    env = make_env(2)
    arrays = buffers(2)
    original = arrays[name]
    if kind == "shape":
        arrays[name] = np.zeros((1, *original.shape[1:]), dtype=original.dtype)
    elif kind == "dtype":
        arrays[name] = np.zeros(original.shape, dtype=np.uint8)
    elif kind == "readonly":
        original.setflags(write=False)
    elif kind == "stride":
        arrays[name] = np.zeros(
            (*original.shape[:-1], original.shape[-1] * 2), dtype=original.dtype
        )[..., ::2]
    elif kind == "fortran":
        arrays[name] = np.asfortranarray(original)
    elif kind == "unaligned":
        if original.dtype.itemsize == 1:
            # bool has alignment one; a nonnative dtype is invalid instead.
            arrays[name] = np.zeros(original.shape, dtype=np.int8)
        else:
            arrays[name] = np.ndarray(
                original.shape,
                dtype=original.dtype,
                buffer=bytearray(original.nbytes + 1),
                offset=1,
            )
    else:
        peer = next(other for other in DESTINATIONS if other != name)
        backing = bytearray(max(original.nbytes, arrays[peer].nbytes))
        arrays[name] = np.ndarray(original.shape, dtype=original.dtype, buffer=backing)
        arrays[peer] = np.ndarray(
            arrays[peer].shape, dtype=arrays[peer].dtype, buffer=backing
        )
    before = snapshot(arrays)
    seeds = env.seed_state()
    state = env.state_snapshot(0)
    with pytest.raises(
        ValueError, match=r"overlap|borrow" if kind == "overlap" else name
    ):
        env.reset(**arrays)
    assert snapshot(arrays) == before
    assert env.seed_state() == seeds
    assert env.state_snapshot(0) == state


@pytest.mark.parametrize(
    ("seed", "stride", "error"),
    [
        (True, 1, ValueError),
        (1.0, 1, ValueError),
        ("1", 1, ValueError),
        (1, True, ValueError),
        (1, 1.0, ValueError),
        (1, "1", ValueError),
        (-1, 1, ValueError),
        (0, 0, ValueError),
        (2**63, 1, OverflowError),
        (0, 2**63, OverflowError),
        (2**63 - 1, 1, OverflowError),
    ],
)
def test_strict_seed_admission(seed: Any, stride: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        make_env(seed=seed, stride=stride)


def test_diagnostics_validate_indices_and_copy_values() -> None:
    env = make_env()
    assert env.terminal_metrics(0) is None
    assert isinstance(json.loads(env.state_snapshot(0)), dict)
    for index in (1, 42):
        with pytest.raises(ValueError, match="env"):
            env.state_snapshot(index)
        with pytest.raises(ValueError, match="env"):
            env.terminal_metrics(index)
    next_seed, seeds = env.seed_state()
    assert isinstance(next_seed, int)
    assert isinstance(seeds, tuple)


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "mode", "nan", "negative", "caps", "inert"]
)
def test_reward_dict_rejects_invalid_config(kind: str) -> None:
    reward: dict[str, Any] = dict(REWARD)
    if kind == "missing":
        del reward["econ_cap"]
    elif kind == "extra":
        reward["unexpected"] = 0
    elif kind == "mode":
        reward["reward_mode"] = "share"
    elif kind == "nan":
        reward["econ_cap"] = float("nan")
    elif kind == "negative":
        reward["econ_drought_weight"] = -1
    elif kind == "caps":
        reward["econ_cap"] = 1.0
    else:
        reward["econ_starvation_weight"] = reward["econ_drought_weight"] = 0
    with pytest.raises(ValueError, match=r"reward|econ|cap"):
        rs.KaggricultureEnv(1, 0, 1, "{}", reward, 1, hire_limit=241)  # type: ignore[arg-type]


def test_reward_mode_wrong_type_is_admission_error() -> None:
    reward: dict[str, Any] = dict(REWARD)
    reward["reward_mode"] = 12
    with pytest.raises(ValueError, match="reward_mode"):
        rs.KaggricultureEnv(1, 0, 1, "{}", reward, 1, hire_limit=241)  # type: ignore[arg-type]


def test_reward_config_requires_plain_dict() -> None:
    class DictSubclass(dict[str, Any]):
        pass

    with pytest.raises(ValueError, match="plain dict"):
        rs.KaggricultureEnv(1, 0, 1, "{}", DictSubclass(REWARD), 1, hire_limit=241)  # type: ignore[arg-type]


@pytest.mark.parametrize(("n_envs", "threads"), [(0, 1), (1, 0), (2**63, 1)])
def test_constructor_rejects_empty_or_unrepresentable_allocation(
    n_envs: int, threads: int
) -> None:
    with pytest.raises(ValueError, match=r"n_envs|native_threads"):
        make_env(n_envs, threads=threads)


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in DESTINATIONS
        if name not in {"actor_mask", "shop_mask", "still_playing", "can_act", "dones"}
    ],
)
def test_foreign_endian_destination_rejected(name: str) -> None:
    env = make_env()
    arrays = buffers(1)
    arrays[name] = arrays[name].astype(arrays[name].dtype.newbyteorder("S"))
    before = snapshot(arrays)
    with pytest.raises(ValueError, match=name):
        env.observe(**arrays)
    assert snapshot(arrays) == before


def pass_actions(n_envs: int) -> tuple[Any, Any]:
    """One farmer PASS frame and a separate STOP; this is a fixed test program."""
    tokens = np.zeros((n_envs, 2, 252, 12), dtype=np.int64)
    tokens[:, :, 0, 1] = 1
    tokens[:, :, 1, 11] = 1
    return tokens, np.full((n_envs, 2), 2, dtype=np.int64)


def diagnostics(env: rs.KaggricultureEnv, n_envs: int) -> tuple[Any, ...]:
    metrics = []
    for i in range(n_envs):
        record = env.terminal_metrics(i)
        metrics.append(
            None
            if record is None
            else {
                **record,
                "econ_0": record["econ_0"].tobytes(),
                "econ_1": record["econ_1"].tobytes(),
            }
        )
    return (
        env.seed_state(),
        tuple(env.state_snapshot(i) for i in range(n_envs)),
        metrics,
    )


def test_step_and_malformed_peer_rollback() -> None:
    env = make_env(2)
    arrays = buffers(2)
    env.observe(**arrays)
    tokens, lengths = pass_actions(2)
    assert env.step(tokens, lengths, **arrays) == {
        "total_games_played": [],
        "terminal_bank_0": [],
        "terminal_bank_1": [],
        "terminal_margin_0": [],
    }
    before = snapshot(arrays), diagnostics(env, 2)
    tokens[1, 1, 0, 1] = 2**62
    with pytest.raises(ValueError, match=r"env=1.*seat=1"):
        env.step(tokens, lengths, **arrays)
    assert (snapshot(arrays), diagnostics(env, 2)) == before
    tokens[1, 1, 0, 1] = 1
    env.step(tokens, lengths, **arrays)


def test_reset_truncate_seed_exhaustion_publishes_nothing() -> None:
    env = make_env(2, seed=2**63 - 6, stride=1, config='{"episodeSteps":2}')
    arrays = buffers(2)
    env.observe(**arrays)
    tokens, lengths = pass_actions(2)
    env.step(tokens, lengths, **arrays)
    assert arrays["dones"].all()
    assert env.seed_state() == (2**63 - 2, (2**63 - 4, 2**63 - 3))
    assert all(env.terminal_metrics(i) is not None for i in range(2))
    before = snapshot(arrays), diagnostics(env, 2)
    with pytest.raises(OverflowError):
        env.reset(**arrays)
    assert (snapshot(arrays), diagnostics(env, 2)) == before
    with pytest.raises(OverflowError):
        env.truncate_envs(np.array([True, True]), **arrays)
    assert (snapshot(arrays), diagnostics(env, 2)) == before
    env.truncate_envs(np.array([False, True]), **arrays)
    assert env.seed_state() == (2**63 - 1, (2**63 - 4, 2**63 - 2))
    assert env.terminal_metrics(0) is not None
    assert env.terminal_metrics(1) is None
    for name in TRANSITIONS:
        assert arrays[name].tobytes() == before[0][name]
    fresh = buffers(2)
    env.observe(**fresh)
    for name in set(DESTINATIONS) - set(TRANSITIONS):
        assert arrays[name][1].tobytes() == fresh[name][1].tobytes()


@pytest.mark.parametrize("name", ["tokens", "lengths", "mask"])
@pytest.mark.parametrize("kind", ["dtype", "shape", "stride", "endian", "overlap"])
def test_native_input_admission_preserves_every_output(name: str, kind: str) -> None:
    env = make_env(2)
    arrays = buffers(2)
    env.observe(**arrays)
    tokens, lengths = pass_actions(2)
    inputs = {"tokens": tokens, "lengths": lengths, "mask": np.array([False, True])}
    array = inputs[name]
    if kind == "dtype":
        inputs[name] = array.astype(np.int8)
    elif kind == "shape":
        inputs[name] = array[:1]
    elif kind == "stride":
        inputs[name] = np.zeros(
            (*array.shape[:-1], array.shape[-1] * 2), dtype=array.dtype
        )[..., ::2]
    elif kind == "endian":
        inputs[name] = array.astype(
            np.int8 if name == "mask" else array.dtype.newbyteorder("S")
        )
    else:
        backing = bytearray(max(array.nbytes, arrays["tile_kind"].nbytes))
        inputs[name] = np.ndarray(array.shape, dtype=array.dtype, buffer=backing)
        arrays["tile_kind"] = np.ndarray(
            arrays["tile_kind"].shape, dtype=np.int64, buffer=backing
        )
    before = snapshot(arrays), diagnostics(env, 2)
    if name == "mask":
        with pytest.raises(
            ValueError, match=r"overlap|borrow" if kind == "overlap" else name
        ):
            env.truncate_envs(inputs["mask"], **arrays)
    else:
        with pytest.raises(
            ValueError, match=r"overlap|borrow" if kind == "overlap" else name
        ):
            env.step(inputs["tokens"], inputs["lengths"], **arrays)
    assert (snapshot(arrays), diagnostics(env, 2)) == before


def test_distinct_numpy_bases_with_shared_storage_are_rejected() -> None:
    import torch

    env = make_env()
    arrays = buffers(1)
    storage = torch.zeros((1, 2, 200), dtype=torch.int64)
    arrays["tile_kind"] = storage.numpy()
    arrays["tile_crop"] = storage.view_as(storage).numpy()
    assert arrays["tile_kind"].base is not arrays["tile_crop"].base
    before = snapshot(arrays)
    with pytest.raises(ValueError, match="overlap"):
        env.observe(**arrays)
    assert snapshot(arrays) == before


@pytest.mark.parametrize(
    ("values", "accepted"),
    [
        ((0.2, 4.0, 1.0, 0.25, 0.0, 0.0), True),
        ((0.2, 0.0, 1.0, 0.25, 0.0, 0.0), True),
        ((0.2, 0.0, 0.0, 0.25, 0.0, 0.0), False),
        ((0.2, 4.0, 1.0, 0.0, 0.0, 0.0), False),
        ((1e-300, 1e-300, 1e-300, 0.25, 0.0, 0.0), False),
        ((1e-300, 1e-300, 1.0, 0.25, 0.0, 0.0), True),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), True),
        ((0.0, 4.0, 1.0, 0.25, 0.0, 0.0), True),
        ((0.0, 0.0, 0.0, 0.0, 0.001, 0.0), False),
        ((0.0, 0.0, 0.0, 0.0, 0.001, 0.1), True),
    ],
)
def test_reward_admission_shared_binary64_predicate(
    values: tuple[float, ...], accepted: bool
) -> None:
    shaping, starvation, drought, cap, ineffective, ineffective_cap = values
    reward: rs.KaggricultureRewardDict = {
        "reward_mode": "win_loss",
        "econ_shaping": shaping,
        "econ_starvation_weight": starvation,
        "econ_drought_weight": drought,
        "econ_cap": cap,
        "econ_ineffective_weight": ineffective,
        "econ_ineffective_cap": ineffective_cap,
    }
    if accepted:
        env = rs.KaggricultureEnv(1, 0, 1, "{}", reward, 1, hire_limit=241)
        assert env.seed_state() == (1, (0,))
    else:
        with pytest.raises(ValueError, match="econ_"):
            rs.KaggricultureEnv(1, 0, 1, "{}", reward, 1, hire_limit=241)


def _rank_seed_stream(
    base: int, rank: int, n_envs: int, world_size: int
) -> tuple[int, int]:
    """The approved factory arithmetic, exercised against real native resets."""
    assert 0 <= rank < world_size
    assert n_envs == 2
    return base + rank, world_size


@pytest.mark.parametrize("world_size", [2, 8])
def test_seed_partition_across_constructor_full_partial_and_terminal_resets(
    world_size: int,
) -> None:
    # Ranks run sequentially: only this rank's two native games are live.
    base, n_envs = 17000, 2
    streams: list[list[int]] = []
    for rank in range(world_size):
        initial_seed, stride = _rank_seed_stream(base, rank, n_envs, world_size)
        env = make_env(
            n_envs, seed=initial_seed, stride=stride, config='{"episodeSteps":2}'
        )
        arrays = buffers(n_envs)
        env.observe(**arrays)
        consumed = list(env.seed_state()[1])
        assert consumed == [initial_seed, initial_seed + stride]
        tokens, lengths = pass_actions(n_envs)
        for cycle in range(13):
            before = snapshot(arrays), diagnostics(env, n_envs)
            tokens[1, 1, 0, 1] = 2**62
            with pytest.raises(ValueError, match=r"env=1.*seat=1"):
                env.step(tokens, lengths, **arrays)
            assert (snapshot(arrays), diagnostics(env, n_envs)) == before
            tokens[1, 1, 0, 1] = 1

            next_seed, _ = env.seed_state()
            env.reset(**arrays)
            # Simultaneous resets reserve in ascending environment order.
            assert env.seed_state() == (
                next_seed + 2 * stride,
                (next_seed, next_seed + stride),
            )
            consumed.extend(env.seed_state()[1])
            next_seed, old_seeds = env.seed_state()
            selected = cycle % 2
            mask = np.array([selected == 0, selected == 1], dtype=np.bool_)
            env.truncate_envs(mask, **arrays)
            expected = list(old_seeds)
            expected[selected] = next_seed
            assert env.seed_state() == (next_seed + stride, tuple(expected))
            consumed.append(next_seed)

            next_seed, _ = env.seed_state()
            info = env.step(tokens, lengths, **arrays)
            assert arrays["dones"].all()
            assert info["total_games_played"] == [1.0, 1.0]
            assert env.seed_state() == (
                next_seed + 2 * stride,
                (next_seed, next_seed + stride),
            )
            consumed.extend(env.seed_state()[1])
            before_observe = env.seed_state()
            env.observe(**arrays)
            assert env.seed_state() == before_observe
        assert len(consumed) == 67
        assert len(set(consumed)) == 67
        streams.append(consumed)
        del env  # Do not keep completed ranks' two-game batches alive.

    for rank, stream in enumerate(streams):
        for peer in range(rank):
            assert set(stream).isdisjoint(streams[peer]), (
                f"rank {rank} collides with rank {peer}"
            )
    for rank, stream in enumerate(streams):
        assert stream == [base + rank + k * world_size for k in range(67)]
    # The rejected finite offset collides even at the third rank-0 seed.
    old_rank_zero = {base + k for k in range(67)}
    old_rank_one = {base + n_envs + k for k in range(67)}
    assert base + n_envs in old_rank_zero & old_rank_one


@pytest.mark.parametrize("world_size", [2, 8])
def test_seed_partition_overflow_is_transactional(world_size: int) -> None:
    maximum = 2**63 - 1
    for rank in range(world_size):
        # Last representable member of this rank's residue class.
        final_seed = maximum - (maximum - rank) % world_size
        initial_seed = final_seed - 4 * world_size
        env = make_env(
            2, seed=initial_seed, stride=world_size, config='{"episodeSteps":2}'
        )
        arrays = buffers(2)
        env.observe(**arrays)
        tokens, lengths = pass_actions(2)
        env.step(tokens, lengths, **arrays)
        assert env.seed_state()[0] == final_seed
        before = snapshot(arrays), diagnostics(env, 2)
        for mask in (None, np.array([False, True]), np.array([True, True])):
            if mask is None:
                with pytest.raises(OverflowError):
                    env.reset(**arrays)
            else:
                with pytest.raises(OverflowError):
                    env.truncate_envs(mask, **arrays)
            assert (snapshot(arrays), diagnostics(env, 2)) == before
        with pytest.raises(OverflowError):
            env.step(tokens, lengths, **arrays)
        assert (snapshot(arrays), diagnostics(env, 2)) == before
        del env


def test_default_horizon_terminal_record_and_autoreset() -> None:
    env = make_env()
    arrays = buffers(1)
    env.observe(**arrays)
    initial_banks = arrays["banks"].copy()
    tokens, lengths = pass_actions(1)
    for transition in range(1, 720):
        before_banks = arrays["banks"][0, 0].copy()
        info = env.step(tokens, lengths, **arrays)
        assert arrays["dones"].all() == (transition == 719)
        np.testing.assert_array_equal(
            arrays["transition_banks_before"][0], before_banks
        )
        if transition < 719:
            assert env.seed_state() == (13, (11,))
            assert env.terminal_metrics(0) is None
            assert info["total_games_played"] == []
    assert env.seed_state() == (15, (13,))
    assert arrays["still_playing"].all()
    assert arrays["can_act"].any()
    assert (arrays["globals_int"][:, :, 0] == 0).all()
    np.testing.assert_array_equal(arrays["banks"], initial_banks)
    terminal = env.terminal_metrics(0)
    assert terminal is not None
    assert terminal["episode_steps"] == 719
    final_banks = arrays["transition_banks_after"][0]
    assert [terminal["bank_0"], terminal["bank_1"]] == final_banks.tolist()
    assert terminal["margin_0"] == final_banks[0] - final_banks[1]
    expected_winner = (
        0
        if final_banks[0] > final_banks[1]
        else 1
        if final_banks[1] > final_banks[0]
        else -1
    )
    assert terminal["winner"] == expected_winner
    assert info == {
        "total_games_played": [1.0],
        "terminal_bank_0": [terminal["bank_0"]],
        "terminal_bank_1": [terminal["bank_1"]],
        "terminal_margin_0": [terminal["margin_0"]],
    }
    for seat in range(2):
        econ = terminal["econ_0"] if seat == 0 else terminal["econ_1"]
        assert econ.dtype == np.int64
        assert econ.flags.c_contiguous
        assert econ.shape == (32,)
        np.testing.assert_array_equal(econ, arrays["transition_econ_after"][0, seat])
    terminal["econ_0"][:] = -1
    again = env.terminal_metrics(0)
    assert again is not None
    assert (again["econ_0"] >= 0).all()
    env.step(tokens, lengths, **arrays)
    assert env.terminal_metrics(0) is None


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("mask_values", [(False, True), (True, False), (False, False)])
def test_truncate_writes_only_selected_rows(
    terminal: bool, mask_values: tuple[bool, bool]
) -> None:
    reward: rs.KaggricultureRewardDict = REWARD.copy()
    reward["econ_ineffective_weight"] = 0.001
    config = json.dumps({"episodeSteps": 2 if terminal else 20})
    env = rs.KaggricultureEnv(2, 11, 2, config, reward, 1, hire_limit=241)
    arrays = buffers(2)
    env.observe(**arrays)
    tokens, lengths = pass_actions(2)
    # Fixed canonical program: seat 0's ineffective HARVEST, then a separate
    # BUY_PRODUCT WHEAT(1) frame and STOP. Seat 1 passes; there is no Python codec.
    tokens[:, 0, 0, 1] = 10
    tokens[:, 0, 1, 11] = 0
    tokens[:, 0, 1, 7] = 4
    tokens[:, 0, 1, 8] = 1
    tokens[:, 0, 1, 10] = 1
    tokens[:, 0, 2, 11] = 1
    lengths[:, 0] = 3
    env.step(tokens, lengths, **arrays)
    assert arrays["rewards"].any()
    assert arrays["transition_banks_after"].any()
    assert arrays["transition_econ_after"].any()
    assert arrays["dones"].all() == terminal
    observation_names = [name for name in DESTINATIONS if name not in TRANSITIONS]
    assert len(observation_names) == 29
    for i, selected in enumerate(mask_values):
        if not selected:
            for name in observation_names:
                row = arrays[name][i]
                if row.dtype == np.bool_:
                    row[:] = True
                else:
                    row.view(np.uint8)[:] = 0x5A
    before = snapshot(arrays)
    states_before = diagnostics(env, 2)
    next_seed, old_seeds = env.seed_state()
    mask = np.array(mask_values, dtype=np.bool_)
    env.truncate_envs(mask, **arrays)
    fresh = buffers(2)
    env.observe(**fresh)
    expected_seeds = list(old_seeds)
    for i, selected in enumerate(mask_values):
        if selected:
            expected_seeds[i] = next_seed
            next_seed += 2
            assert env.terminal_metrics(i) is None
            for name in observation_names:
                assert arrays[name][i].tobytes() == fresh[name][i].tobytes(), name
        else:
            assert env.state_snapshot(i) == states_before[1][i]
            assert diagnostics(env, 2)[2][i] == states_before[2][i]
            for name in observation_names:
                # Compare each row from the original full-buffer byte snapshot.
                width = arrays[name][i].nbytes
                assert (
                    arrays[name][i].tobytes()
                    == before[name][i * width : (i + 1) * width]
                ), name
    assert env.seed_state() == (next_seed, tuple(expected_seeds))
    for name in TRANSITIONS:
        assert arrays[name].tobytes() == before[name], name
    if not mask.any():
        assert snapshot(arrays) == before
        assert diagnostics(env, 2) == states_before


def test_one_and_two_thread_pools_publish_identical_batches() -> None:
    # Separate runs keep only E2 live; retained evidence is bytes, not environments.
    def run(threads: int) -> list[Any]:
        env = make_env(
            2, seed=27000, stride=3, config='{"episodeSteps":4}', threads=threads
        )
        arrays = buffers(2)
        env.observe(**arrays)
        evidence = [(snapshot(arrays), diagnostics(env, 2), None)]
        for step in range(9):
            tokens = np.zeros((2, 2, 252, 12), dtype=np.int64)
            lengths = np.zeros((2, 2), dtype=np.int64)
            for index in range(2):
                for seat in range(2):
                    action = {
                        "farmer": ["PASS"],
                        "hands": [],
                        "market": [["BUY_PRODUCT", "WHEAT", index + seat + 1]],
                    }
                    lengths[index, seat] = rs.kaggriculture_encode(
                        json.dumps(action), 1, 10, 241, tokens[index, seat]
                    )
            metrics = env.step(tokens, lengths, **arrays)
            evidence.append((snapshot(arrays), diagnostics(env, 2), metrics))
            if step == 3:
                env.truncate_envs(np.array([False, True]), **arrays)
                evidence.append((snapshot(arrays), diagnostics(env, 2), None))
            if step == 5:
                env.reset(**arrays)
                evidence.append((snapshot(arrays), diagnostics(env, 2), None))
        return evidence

    serial = run(1)
    assert run(2) == serial
