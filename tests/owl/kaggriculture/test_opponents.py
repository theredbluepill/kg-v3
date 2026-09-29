"""Deferred native-binding checks; no substitute environment is constructed.

Task 1.4 will expose ``owl.rs.KaggricultureEnv``. A learned seat submits official
JSON actions while a per-environment scripted controller owns the other seat.
Opponent names belong only to evaluator setup and must never enter observations.
"""

import pytest


@pytest.mark.parametrize("opponent", ["starter", "r04", "ecobot", "e776"])
@pytest.mark.parametrize("learned_seat", [0, 1])
def test_learned_seat_plays_opponent_and_preserves_observation_boundary(
    opponent: str, learned_seat: int
) -> None:
    """Future API: reset(seed, learned_seat, opponent), then native step(actions)."""
    del opponent, learned_seat
    pytest.skip("needs Task 1.4 binding")


def test_opponent_resets_independently_on_each_environment_auto_reset() -> None:
    """Future vector reset must allocate a fresh controller for each seat/episode."""
    pytest.skip("needs Task 1.4 binding")
