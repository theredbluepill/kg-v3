"""Deferred native-binding checks; no substitute environment is constructed.

Task 1.4 will expose ``owl.rs.KaggricultureEnv``. The learned seat submits grammar
tokens through the native step (JSON is never live step transport); inside the
native environment a per-environment scripted controller owns the other seat and
emits official JSON. Opponent names belong only to evaluator setup and must never
enter observations, rewards, normalization or checkpoint selection.
"""

import pytest


@pytest.mark.parametrize("opponent", ["starter", "r04", "ecobot", "e776"])
@pytest.mark.parametrize("learned_seat", [0, 1])
def test_learned_seat_plays_opponent_and_preserves_observation_boundary(
    opponent: str, learned_seat: int
) -> None:
    """Intended: choose learned seat and opponent at evaluator setup; step tokens."""
    del opponent, learned_seat
    pytest.skip("needs Task 1.4 binding")


def test_opponent_resets_independently_on_each_environment_auto_reset() -> None:
    """Future vector reset must allocate a fresh controller for each seat/episode."""
    pytest.skip("needs Task 1.4 binding")
