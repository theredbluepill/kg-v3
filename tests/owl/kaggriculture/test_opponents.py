"""Learned seat against each native ``opponents_rs`` controller.

The native ``owl.rs.KaggricultureEnv`` hosts a per-environment scripted
controller (``opponent_bot``/``opponent_envs``, driven by ``env.opponent_mix``).
The learned seat submits grammar tokens through the native step (JSON is never
live step transport); the scripted seat's transport is the absent program and
its official JSON comes from the controller inside the step. Opponent names
belong only to collection/evaluator setup and never enter observations,
rewards, normalization or checkpoint selection. Exact action and state
agreement with an independent kernel + controller replay is tested natively
(``src/kaggriculture/opponent_env_tests.rs``).
"""

import json

import numpy as np
import pytest
from owl import rs

from tests.kaggriculture.test_native_env import REWARD, buffers, pass_actions


def _hosted_env(
    opponent: str, n_envs: int = 2, config: str = "{}"
) -> rs.KaggricultureEnv:
    return rs.KaggricultureEnv(
        n_envs,
        31,
        1,
        config,
        REWARD,
        1,
        hire_limit=241,
        opponent_bot=opponent,
        opponent_envs=n_envs,
    )


def _learner_actions(env: rs.KaggricultureEnv, n_envs: int) -> tuple[object, object]:
    tokens, lengths = pass_actions(n_envs)
    mask = env.learner_mask()
    tokens[~mask] = 0
    lengths[~mask] = 0
    return tokens, lengths


@pytest.mark.parametrize("opponent", ["starter", "r04", "ecobot", "e776", "cha22"])
@pytest.mark.parametrize("learned_seat", [0, 1])
def test_learned_seat_plays_opponent_and_preserves_observation_boundary(
    opponent: str, learned_seat: int
) -> None:
    """The learned seat is chosen by env index and episode; the bot plays the other."""
    env = _hosted_env(opponent)
    arrays = buffers(2)
    env.observe(**arrays)
    # Construction is episode 0: env e learns seat e % 2.
    mask = env.learner_mask()
    assert mask.dtype == np.bool_
    assert mask.shape == (2, 2)
    assert mask[learned_seat].tolist() == [seat == learned_seat for seat in range(2)]
    # The observation is identical to a self-play env's at step zero.
    plain = rs.KaggricultureEnv(2, 31, 1, "{}", REWARD, 1, hire_limit=241)
    plain_arrays = buffers(2)
    plain.observe(**plain_arrays)
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, plain_arrays[name], err_msg=name)
    # A program submitted for the scripted seat is refused, whole-batch.
    tokens, lengths = _learner_actions(env, 2)
    bot_seat = 1 - learned_seat
    lengths[learned_seat, bot_seat] = 2
    with pytest.raises(ValueError, match="played by the fixed opponent"):
        env.step(tokens, lengths, **arrays)
    lengths[learned_seat, bot_seat] = 0
    idle = rs.KaggricultureEnv(2, 31, 1, "{}", REWARD, 1, hire_limit=241)
    idle_arrays = buffers(2)
    idle.observe(**idle_arrays)
    idle_tokens, idle_lengths = pass_actions(2)
    for _ in range(12):
        info = env.step(tokens, lengths, **arrays)
        assert info["terminal_learner_seat"] == []
        idle.step(idle_tokens, idle_lengths, **idle_arrays)
    # The scripted seat acted inside the step: the game left the all-PASS path.
    hosted = json.loads(env.state_snapshot(learned_seat))
    passed = json.loads(idle.state_snapshot(learned_seat))
    assert hosted["public"]["step"] == passed["public"]["step"] == 12
    assert hosted != passed


def test_opponent_resets_independently_on_each_environment_auto_reset() -> None:
    """Each env allocates a fresh controller for its new game's scripted seat."""
    env = _hosted_env("starter", n_envs=2, config='{"episodeSteps":3}')
    arrays = buffers(2)
    env.observe(**arrays)
    for game in range(3):
        before = env.learner_mask()
        env.step(*_learner_actions(env, 2), **arrays)
        info = env.step(*_learner_actions(env, 2), **arrays)
        assert arrays["dones"].all()
        assert info["terminal_learner_seat"] == [
            float(before[0].argmax()),
            float(before[1].argmax()),
        ]
        # Stale controllers would refuse the new game's step zero; the flip
        # shows each env started a new episode with its seats swapped.
        np.testing.assert_array_equal(env.learner_mask(), ~before, err_msg=str(game))
    # A truncation resets only the selected env's controller.
    before = env.learner_mask()
    env.truncate_envs(np.array([True, False]), **arrays)
    after = env.learner_mask()
    np.testing.assert_array_equal(after[0], ~before[0])
    np.testing.assert_array_equal(after[1], before[1])
    env.step(*_learner_actions(env, 2), **arrays)


def test_constructor_admits_opponent_arguments_together() -> None:
    for kwargs in (
        {"opponent_bot": "starter"},
        {"opponent_envs": 1},
        {"opponent_bot": "starter", "opponent_envs": 3},
        {"opponent_bot": "unknown", "opponent_envs": 1},
    ):
        with pytest.raises(ValueError, match="opponent"):
            rs.KaggricultureEnv(2, 1, 1, "{}", REWARD, 1, hire_limit=241, **kwargs)
