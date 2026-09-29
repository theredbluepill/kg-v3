"""Executable placeholders for the native environment's future custody seam."""

import pytest


@pytest.mark.parametrize(
    "_event", ["construction", "explicit_reset", "simultaneous_terminal_reset"]
)
def test_live_env_consumed_seed_custody(_event: str) -> None:
    # Intended API: KaggricultureEnv.resolved_seeds() reports consumed seeds;
    # reset()/step_into() allocate simultaneous replacements in env-index order.
    pytest.skip("needs Task 1.4 binding")


def test_evaluate_games_exports_eight_complete_episodes() -> None:
    # Intended API: _evaluate_games' Kaggriculture branch owns ReplayRecorder,
    # passes cfg.rl.eval_replay_games=8, and produces 8 verified complete episodes.
    pytest.skip("needs Task 1.4 binding")


def test_live_terminal_snapshot_is_captured_before_auto_reset() -> None:
    # Intended API: KaggricultureEnv.terminal_snapshot(env_index) returns the
    # finished native StepSnapshot even though step_into() returns reset obs.
    pytest.skip("needs Task 1.4 binding")
