"""Kaggriculture evaluation outcomes decided by the game objective (lesson L1).

Checkpoint promotion compares raw final banks, never the shaped training return.
The training reward scales the terminal win/loss/draw term by ``terminal_scale``
and adds capped penalty differences (``docs/kaggriculture-contract.md``,
"Rewards"), so a shaped return can rank the seats differently from their banks:
equal banks are a draw even when the penalties differ, and a recipe with larger
caps can invert the winner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

_SEAT_BANK_KEYS = ("bank_0", "bank_1")


def terminal_seat_banks(terminal_metrics: Mapping[str, float]) -> torch.Tensor:
    """Raw final banks ``[bank_0, bank_1]`` (float64, seat order) of one game.

    Requires the contract's terminal ``bank_0``, ``bank_1`` and ``margin_0`` and
    checks them, plus the optional ``winner`` extension, against each other.
    """
    for key in (*_SEAT_BANK_KEYS, "margin_0"):
        if key not in terminal_metrics:
            raise ValueError(f"Kaggriculture terminal metrics lack '{key}'")
    bank_0 = float(terminal_metrics["bank_0"])
    bank_1 = float(terminal_metrics["bank_1"])
    if not (math.isfinite(bank_0) and math.isfinite(bank_1)):
        raise ValueError(f"terminal banks must be finite, got {bank_0}, {bank_1}")
    margin = float(terminal_metrics["margin_0"])
    if margin != bank_0 - bank_1:
        raise ValueError(
            f"terminal margin_0={margin} disagrees with banks {bank_0} - {bank_1}"
        )
    if "winner" in terminal_metrics:
        winner = float(terminal_metrics["winner"])
        expected = 0.0 if bank_0 > bank_1 else 1.0 if bank_1 > bank_0 else -1.0
        if winner != expected:
            raise ValueError(
                f"terminal winner={winner} disagrees with banks {bank_0}, {bank_1}"
            )
    return torch.tensor([bank_0, bank_1], dtype=torch.float64)
