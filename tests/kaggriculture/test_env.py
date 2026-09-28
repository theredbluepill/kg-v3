from __future__ import annotations

import numpy as np
import pytest
import torch
from owl.kaggriculture.actor_codec import SLOT_NAMES, encode_action
from owl.kaggriculture.env import KaggricultureVectorizedEnv, native_seed_header
from owl.kaggriculture.rewards import (
    KaggricultureRewardConfig,
    economic_rewards,
    terminal_rewards,
)
from owl.kaggriculture.types import KaggricultureActions, KaggricultureObsConfig


def actions_for(
    env: KaggricultureVectorizedEnv, *, buy_land: bool = False
) -> KaggricultureActions:
    tokens = torch.zeros((env.n_envs, 2, 252, 12), dtype=torch.int64)
    lengths = torch.zeros((env.n_envs, 2), dtype=torch.int64)
    for game in range(env.n_envs):
        for seat in range(2):
            count = int(env.observations.context[game, 1 + seat])
            action = {
                "farmer": ["PASS"],
                "hands": [["PASS"] for _ in range(count - 1)],
                "market": [["BUY_LAND"]] if buy_land and seat == 0 else [],
            }
            encoded = encode_action(action, hire_limit=241)
            frames = torch.tensor(
                [[frame[s] for s in SLOT_NAMES] for frame in encoded["frames"]]
            )
            tokens[game, seat, : len(frames)] = frames
            lengths[game, seat] = len(frames)
    return KaggricultureActions(tokens, lengths)


@pytest.mark.parametrize(("version", "width"), [(1, 8165), (2, 8176)])
def test_real_native_observations_and_buffer_reuse(version: int, width: int) -> None:
    with KaggricultureVectorizedEnv(
        n_envs=2,
        obs_spec=KaggricultureObsConfig(observation_version=version),
        pin_memory=False,
    ) as env:
        obs = env.reset()
        ptr = obs.features.data_ptr()
        before = obs.features.clone()
        assert obs.features.shape == (2, 2, width)
        assert obs.entity_mask.sum().item() == 4
        assert obs.context.tolist() == [[0, 1, 1, 10], [0, 1, 1, 10]]
        next_obs, reward, done, _ = env.step(actions_for(env))
        assert next_obs.features.data_ptr() == ptr
        assert not torch.equal(before, next_obs.features)
        assert next_obs.context[:, 0].tolist() == [1, 1]
        assert not reward.any()
        assert not done.any()
        assert env.state_snapshot(0)["public"]["step"] == 1


@pytest.mark.parametrize(
    ("mode", "expected"), [("win_loss", [-1.0, 1.0]), ("win_only", [0.0, 1.0])]
)
def test_terminal_reward_and_auto_reset(mode: str, expected: list[float]) -> None:
    with KaggricultureVectorizedEnv(
        n_envs=2, pin_memory=False, reward_mode=mode, configuration={"episodeSteps": 3}
    ) as env:
        env.step(actions_for(env, buy_land=True))
        obs, reward, done, metrics = env.step(actions_for(env))
        assert done.all()
        assert reward.tolist() == [expected, expected]
        assert obs.context[:, 0].tolist() == [0, 0]
        assert obs.still_playing.all()
        assert metrics["total_games_played"] == [1.0, 1.0]
        terminal = env.terminal_snapshot(0)
        assert terminal is not None
        assert terminal["done"]
        assert terminal["public"]["step"] == 2
        assert env.terminal_metrics(0)["margin_0"] == -1000.0
        assert env.state_snapshot(0)["public"]["step"] == 0
        _, reward, done, _ = env.step(actions_for(env))
        assert not done.any()
        assert not reward.any()
        assert env.terminal_snapshot(0) is None


def test_tie_reward_and_selective_truncation() -> None:
    with KaggricultureVectorizedEnv(
        n_envs=2,
        pin_memory=False,
        reward_mode="win_only",
        configuration={"episodeSteps": 3},
    ) as env:
        env.step(actions_for(env))
        env.truncate_envs(torch.tensor([True, False]))
        assert env.observations.context[:, 0].tolist() == [0, 1]
        obs, reward, done, _ = env.step(actions_for(env))
        assert done.tolist() == [[False, False], [True, True]]
        assert reward.tolist() == [[0.0, 0.0], [0.5, 0.5]]
        assert obs.context[:, 0].tolist() == [1, 0]


def test_invalid_action_rolls_back_entire_native_batch() -> None:
    with KaggricultureVectorizedEnv(n_envs=2, pin_memory=False) as env:
        before = env.observations.features.clone()
        actions = actions_for(env)
        actions.tokens[1, 1, 0, 1] = 19  # RESERVED
        with pytest.raises(RuntimeError):
            env.step(actions)
        assert torch.equal(before, env.observations.features)
        assert env.state_snapshot(0)["public"]["step"] == 0
        assert env.state_snapshot(1)["public"]["step"] == 0


def test_reused_narrowing_buffers_reject_overflow_and_replace_previous_actions() -> (
    None
):
    with KaggricultureVectorizedEnv(n_envs=2, pin_memory=False) as env:
        first = actions_for(env, buy_land=True)
        frame_pointer = env._frames.ctypes.data
        length_pointer = env._lengths.ctypes.data
        env.step(first)
        assert env.current_banks[0].tolist() == [2000.0, 3000.0]
        valid = actions_for(env)
        malformed = actions_for(env)
        malformed.tokens[0, 0, 0, 1] = 65537  # would narrow to PASS
        before = env.observations.features.clone()
        with pytest.raises(ValueError, match="signed-int16"):
            env.step(malformed)
        assert torch.equal(before, env.observations.features)
        env.step(valid)
        assert env.current_banks[0].tolist() == [2000.0, 3000.0]
        assert env._frames.ctypes.data == frame_pointer
        assert env._lengths.ctypes.data == length_pointer
        assert not env.rewards.any()
        assert int(env.observations.context[0, 0]) == 2


def test_serial_and_parallel_native_games_match() -> None:
    with (
        KaggricultureVectorizedEnv(
            n_envs=2, pin_memory=False, seed=31, threads=1
        ) as serial,
        KaggricultureVectorizedEnv(
            n_envs=2, pin_memory=False, seed=31, threads=2
        ) as parallel,
    ):
        for _ in range(30):
            a = actions_for(serial)
            serial.step(a)
            parallel.step(a)
            assert torch.equal(
                serial.observations.features, parallel.observations.features
            )
        assert serial.state_snapshot(0) == parallel.state_snapshot(0)


def test_native_header_rejects_invalid_seed_and_private_buffers_do_not_alias() -> None:
    with pytest.raises(ValueError, match="integer"):
        native_seed_header(True, {})
    with KaggricultureVectorizedEnv(n_envs=1, pin_memory=False) as env:
        assert not np.shares_memory(
            env.observations.features[:, 0].numpy(),
            env.observations.features[:, 1].numpy(),
        )
        assert env.observations.features[0, 0, 5273:8165].equal(
            env.observations.features[0, 1, 5273:8165]
        )


def test_reward_definitions_and_separate_cumulative_caps() -> None:
    spec = KaggricultureRewardConfig(econ_shaping=0.01, econ_ineffective_weight=0.001)
    banks = torch.tensor([[6000.0, 3000.0], [0.0, 0.0]], dtype=torch.float64)
    assert terminal_rewards(banks, "margin", spec).tolist() == [[1.0, -1.0], [0.0, 0.0]]
    assert terminal_rewards(banks, "win_loss", spec).tolist() == [
        [1.0, -1.0],
        [0.0, 0.0],
    ]
    assert terminal_rewards(banks, "win_share", spec)[0, 0].item() == pytest.approx(
        0.9 + 0.1 / 3
    )
    before = torch.zeros((1, 2, 32), dtype=torch.float64)
    after = before.clone()
    after[0, 0, 0] = 1  # one own starvation death: .01*4
    after[0, 0, 2] = 1000  # ineffective budget saturates at .10
    assert economic_rewards(before, after, spec).tolist()[0] == pytest.approx(
        [-0.14, 0.14]
    )
    later = after.clone()
    later[0, 0, 1] = 1  # exhausted ineffective budget cannot suppress drought
    assert economic_rewards(after, later, spec).tolist()[0] == pytest.approx(
        [-0.01, 0.01]
    )
    assert spec.terminal_scale == pytest.approx(0.65)


def test_native_ineffective_penalty_and_auto_reset_counters() -> None:
    with KaggricultureVectorizedEnv(
        n_envs=1,
        pin_memory=False,
        configuration={"episodeSteps": 2},
        reward_shaping=KaggricultureRewardConfig(econ_ineffective_weight=0.01),
    ) as env:
        actions = actions_for(env)
        actions.tokens[0, 0, 0, 1] = 10  # HARVEST with no plant
        _, rewards, dones, _ = env.step(actions)
        assert dones.all()
        assert rewards.tolist()[0] == pytest.approx([-0.01, 0.01])
        assert int(env.current_econ[0, 0, 2]) == 1
        assert int(env.previous_econ[0, 0, 2]) == 0
        env.step(actions_for(env))
        assert int(env.previous_econ[0, 0, 2]) == 0
        assert int(env.current_econ[0, 0, 2]) == 0


def test_default_season_has_719_transitions_and_one_terminal_event() -> None:
    with KaggricultureVectorizedEnv(n_envs=1, pin_memory=False) as env:
        action = actions_for(env)
        for _ in range(718):
            _, _, done, _ = env.step(action)
            assert not done.any()
        obs, rewards, done, metrics = env.step(action)
        assert done.all()
        assert not rewards.any()
        assert metrics["total_games_played"] == [1.0]
        terminal = env.terminal_snapshot(0)
        assert terminal is not None
        assert terminal["public"]["step"] == 719
        assert obs.context[0, 0] == 0


def test_own_private_seed_features_remain_seat_scoped() -> None:
    with KaggricultureVectorizedEnv(n_envs=1, pin_memory=False) as env:
        before = env.observations.features.clone()
        action = actions_for(env)
        # Farmer PASS, buy three seeds, STOP. Only seat0 private seeds change.
        action.tokens[0, 0, 1].zero_()
        action.tokens[0, 0, 1, 7] = 3  # BUY_SEED
        action.tokens[0, 0, 1, 8] = 1  # WHEAT
        action.tokens[0, 0, 1, 10] = 3
        action.tokens[0, 0, 2, 11] = 1
        action.lengths[0, 0] = 3
        obs, _, _, _ = env.step(action)
        assert obs.features[0, 0, 872] > before[0, 0, 872]
        assert torch.equal(obs.features[0, 1, 872:889], before[0, 1, 872:889])


def test_rank_seed_streams_stay_disjoint_across_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import owl.kaggriculture.env as env_module

    recorded: list[int] = []
    original = env_module.native_seed_header

    def record(seed: int, configuration: dict[str, object]) -> dict[str, object]:
        recorded.append(seed)
        return original(seed, configuration)

    monkeypatch.setattr(env_module, "native_seed_header", record)
    streams: list[set[int]] = []
    for rank in range(2):
        recorded.clear()
        with KaggricultureVectorizedEnv(
            n_envs=2,
            pin_memory=False,
            seed=100 + rank,
            seed_stride=2,
            configuration={"episodeSteps": 2},
        ) as env:
            for _ in range(3):
                env.reset()
                env.step(actions_for(env))
        assert len(recorded) == len(set(recorded))
        streams.append(set(recorded))
    assert streams[0].isdisjoint(streams[1])


def test_seed_stride_rejects_nonpositive_or_boolean() -> None:
    for stride in (0, -1, True):
        with pytest.raises(ValueError, match="seed_stride"):
            KaggricultureVectorizedEnv(n_envs=1, seed_stride=stride, pin_memory=False)
