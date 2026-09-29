"""Raw-bank evaluation outcomes (rebuild Task 3.2, lesson L1)."""

from __future__ import annotations

import math

import pytest
import torch
from owl.kaggriculture.evaluation import terminal_seat_banks


def _metrics(bank_0: float, bank_1: float, **extra: float) -> dict[str, float]:
    return {
        "bank_0": bank_0,
        "bank_1": bank_1,
        "margin_0": bank_0 - bank_1,
        "episode_steps": 719.0,
        **extra,
    }


def test_terminal_seat_banks_returns_raw_float64_banks_in_seat_order() -> None:
    banks = terminal_seat_banks(_metrics(3100.5, 2900.25, winner=0.0))

    assert banks.dtype == torch.float64
    assert banks.tolist() == [3100.5, 2900.25]


def test_terminal_seat_banks_accepts_draws_and_missing_winner_extension() -> None:
    assert terminal_seat_banks(_metrics(3000.0, 3000.0, winner=-1.0)).tolist() == [
        3000.0,
        3000.0,
    ]
    assert terminal_seat_banks(_metrics(1.0, 2.0)).tolist() == [1.0, 2.0]


@pytest.mark.parametrize("missing", ["bank_0", "bank_1", "margin_0"])
def test_terminal_seat_banks_requires_contract_keys(missing: str) -> None:
    metrics = _metrics(3.0, 2.0)
    del metrics[missing]

    with pytest.raises(ValueError, match=f"terminal metrics lack '{missing}'"):
        terminal_seat_banks(metrics)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_terminal_seat_banks_rejects_non_finite_banks(bad: float) -> None:
    metrics = _metrics(3.0, 2.0)
    metrics["bank_1"] = bad

    with pytest.raises(ValueError, match="banks must be finite"):
        terminal_seat_banks(metrics)


def test_terminal_seat_banks_rejects_margin_that_disagrees_with_banks() -> None:
    metrics = _metrics(3.0, 2.0)
    metrics["margin_0"] = 0.5

    with pytest.raises(ValueError, match=r"margin_0=0\.5 disagrees"):
        terminal_seat_banks(metrics)


@pytest.mark.parametrize(
    ("banks", "winner"),
    [((2.0, 1.0), 1.0), ((1.0, 2.0), 0.0), ((1.0, 1.0), 0.0), ((2.0, 1.0), -1.0)],
)
def test_terminal_seat_banks_rejects_winner_that_disagrees_with_banks(
    banks: tuple[float, float], winner: float
) -> None:
    with pytest.raises(ValueError, match="winner="):
        terminal_seat_banks(_metrics(*banks, winner=winner))
