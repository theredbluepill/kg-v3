"""Learner-perspective raw-bank diagnostics for Kaggriculture PPO (W&B only).

These metrics are telemetry. They are computed from completed-game terminal
records after the fact and go to the metric logger alone: they never feed model
inputs, rewards, losses, advantage or return normalization, or checkpoint
selection (promotion stays on raw-bank wins, ``owl.kaggriculture.evaluation``).

Units are raw final bank (money), not normalized. Percentiles use linear
interpolation between order statistics (``torch.quantile``'s default).

Why training reports an absolute margin: in self-play both seats of a training
game are the learner, so one game contributes ``+m`` from one seat and ``-m``
from the other. The signed learner margin is identically zero per game and
carries no signal, and the seat-0 margin (``train/terminal_margin_0``) only
measures seat asymmetry. ``|bank_0 - bank_1|`` measures how decisive the games
are. A signed margin is reported only against a distinct opponent (the
last-best evaluation, or a fixed-opponent panel whose name enters the metric
prefix as a label and never the model).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

BANK_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p10", 0.1),
    ("p50", 0.5),
    ("p90", 0.9),
)


def self_play_bank_metrics(
    bank_0: Sequence[float],
    bank_1: Sequence[float],
    *,
    prefix: str = "train/",
) -> dict[str, float]:
    """Both learner seats' raw final banks over self-play games.

    ``bank_0[i]`` and ``bank_1[i]`` are game ``i``'s seat banks. Always returns
    ``{prefix}bank_games`` (the game count, possibly 0). With at least one game
    it adds ``own_bank_{mean,p10,p50,p90}`` over every learner seat (two values
    per game), ``margin_abs_{mean,p50}`` and ``draw_rate``. With at least one
    decisive game it adds ``winner_bank_mean`` and ``loser_bank_mean`` over the
    decisive games; draws (equal banks) have no winner or loser. No key is NaN.
    """
    banks = _paired_banks(bank_0, bank_1, names=("bank_0", "bank_1"))
    games = banks.shape[1]
    metrics = {f"{prefix}bank_games": float(games)}
    if games == 0:
        return metrics
    _add_distribution(metrics, f"{prefix}own_bank", banks.flatten())
    margin_abs = (banks[0] - banks[1]).abs()
    metrics[f"{prefix}margin_abs_mean"] = float(margin_abs.mean())
    metrics[f"{prefix}margin_abs_p50"] = float(torch.quantile(margin_abs, 0.5))
    decisive = banks[0] != banks[1]
    metrics[f"{prefix}draw_rate"] = float((~decisive).to(torch.float64).mean())
    if bool(decisive.any()):
        metrics[f"{prefix}winner_bank_mean"] = float(
            banks.max(dim=0).values[decisive].mean()
        )
        metrics[f"{prefix}loser_bank_mean"] = float(
            banks.min(dim=0).values[decisive].mean()
        )
    return metrics


def opponent_bank_metrics(
    own: Sequence[float],
    opponent: Sequence[float],
    *,
    prefix: str,
) -> dict[str, float]:
    """A policy's raw final banks against a distinct opponent, one entry per game.

    Always returns ``{prefix}bank_games``. With at least one game it adds the
    mean and p10/p50/p90 of ``own_bank``, ``opponent_bank`` and the signed
    ``margin`` (own minus opponent). The prefix carries any panel label; the
    opponent's identity never reaches the policy.
    """
    banks = _paired_banks(own, opponent, names=("own", "opponent"))
    games = banks.shape[1]
    metrics = {f"{prefix}bank_games": float(games)}
    if games == 0:
        return metrics
    _add_distribution(metrics, f"{prefix}own_bank", banks[0])
    _add_distribution(metrics, f"{prefix}opponent_bank", banks[1])
    _add_distribution(metrics, f"{prefix}margin", banks[0] - banks[1])
    return metrics


def _paired_banks(
    first: Sequence[float], second: Sequence[float], *, names: tuple[str, str]
) -> torch.Tensor:
    if len(first) != len(second):
        raise ValueError(
            f"{names[0]} and {names[1]} bank lists differ in length: "
            f"{len(first)} != {len(second)}"
        )
    values = [float(value) for value in (*first, *second)]
    bad = [value for value in values if not math.isfinite(value)]
    if bad:
        raise ValueError(f"terminal banks must be finite, got {bad[:4]}")
    return torch.tensor(values, dtype=torch.float64).reshape(2, len(first))


def _add_distribution(
    metrics: dict[str, float], name: str, values: torch.Tensor
) -> None:
    metrics[f"{name}_mean"] = float(values.mean())
    for suffix, quantile in BANK_PERCENTILES:
        metrics[f"{name}_{suffix}"] = float(torch.quantile(values, quantile))
