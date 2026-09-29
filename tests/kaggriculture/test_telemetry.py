"""Learner-perspective raw-bank telemetry math on synthetic terminal records."""

from __future__ import annotations

import math

import pytest
from owl.kaggriculture.telemetry import opponent_bank_metrics, self_play_bank_metrics

SELF_PLAY_KEYS = {
    "train/bank_games",
    "train/own_bank_mean",
    "train/own_bank_p10",
    "train/own_bank_p50",
    "train/own_bank_p90",
    "train/margin_abs_mean",
    "train/margin_abs_p50",
    "train/winner_bank_mean",
    "train/loser_bank_mean",
    "train/draw_rate",
}

OPPONENT_KEYS = {
    "eval/bank_games",
    *(
        f"eval/{name}_{stat}"
        for name in ("own_bank", "opponent_bank", "margin")
        for stat in ("mean", "p10", "p50", "p90")
    ),
}


def _linear_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def test_self_play_metrics_pool_both_learner_seats_with_asymmetric_banks() -> None:
    # Game 0: seat 0 wins 3000 to 1000; game 1: seat 1 wins 5000 to 2000;
    # game 2: a 2500 draw.
    bank_0 = [3000.0, 2000.0, 2500.0]
    bank_1 = [1000.0, 5000.0, 2500.0]
    metrics = self_play_bank_metrics(bank_0, bank_1)

    assert set(metrics) == SELF_PLAY_KEYS
    own = [*bank_0, *bank_1]
    assert metrics["train/bank_games"] == 3.0
    assert metrics["train/own_bank_mean"] == pytest.approx(sum(own) / 6)
    for name, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        assert metrics[f"train/own_bank_{name}"] == pytest.approx(
            _linear_quantile(own, q)
        )
    # |margin| per game is 2000, 3000, 0: seat-0 signs would cancel, these do not.
    assert metrics["train/margin_abs_mean"] == pytest.approx(5000.0 / 3)
    assert metrics["train/margin_abs_p50"] == 2000.0
    # The draw has no winner or loser; decisive games are 0 and 1.
    assert metrics["train/winner_bank_mean"] == pytest.approx(4000.0)
    assert metrics["train/loser_bank_mean"] == pytest.approx(1500.0)
    assert metrics["train/draw_rate"] == pytest.approx(1.0 / 3)


def test_self_play_metrics_are_seat_symmetric() -> None:
    bank_0 = [3000.0, 2000.0, 2500.0]
    bank_1 = [1000.0, 5000.0, 2500.0]
    assert self_play_bank_metrics(bank_0, bank_1) == self_play_bank_metrics(
        bank_1, bank_0
    )


def test_self_play_metrics_with_only_draws_omit_winner_and_loser() -> None:
    metrics = self_play_bank_metrics([2500.0, 0.0], [2500.0, 0.0])
    assert "train/winner_bank_mean" not in metrics
    assert "train/loser_bank_mean" not in metrics
    assert metrics["train/draw_rate"] == 1.0
    assert metrics["train/margin_abs_mean"] == 0.0
    assert metrics["train/bank_games"] == 2.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_empty_interval_logs_only_an_explicit_zero_count() -> None:
    assert self_play_bank_metrics([], []) == {"train/bank_games": 0.0}
    assert opponent_bank_metrics([], [], prefix="eval/") == {"eval/bank_games": 0.0}


def test_single_game_is_finite() -> None:
    metrics = self_play_bank_metrics([100.0], [40.0])
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["train/own_bank_p10"] == pytest.approx(46.0)
    assert metrics["train/own_bank_p90"] == pytest.approx(94.0)


def test_opponent_metrics_keep_the_signed_margin() -> None:
    own = [3000.0, 2500.0, 1000.0, 4000.0]
    opponent = [2000.0, 2500.0, 1500.0, 1000.0]
    metrics = opponent_bank_metrics(own, opponent, prefix="eval/")

    assert set(metrics) == OPPONENT_KEYS
    margin = [a - b for a, b in zip(own, opponent, strict=True)]
    assert metrics["eval/bank_games"] == 4.0
    for name, values in (
        ("own_bank", own),
        ("opponent_bank", opponent),
        ("margin", margin),
    ):
        assert metrics[f"eval/{name}_mean"] == pytest.approx(sum(values) / 4)
        for stat, q in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
            assert metrics[f"eval/{name}_{stat}"] == pytest.approx(
                _linear_quantile(values, q)
            )
    assert metrics["eval/margin_mean"] == pytest.approx(875.0)
    assert metrics["eval/margin_p10"] < 0 < metrics["eval/margin_p90"]


def test_opponent_metrics_prefix_is_only_a_label() -> None:
    metrics = opponent_bank_metrics([10.0], [4.0], prefix="panel/greedy/")
    assert metrics["panel/greedy/margin_mean"] == 6.0
    assert metrics["panel/greedy/bank_games"] == 1.0


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        ([1.0, 2.0], [1.0], "differ in length"),
        ([math.nan], [1.0], "finite"),
        ([1.0], [math.inf], "finite"),
    ],
)
def test_malformed_records_fail_fast(
    first: list[float], second: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        self_play_bank_metrics(first, second)
    with pytest.raises(ValueError, match=match):
        opponent_bank_metrics(first, second, prefix="eval/")
