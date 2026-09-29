from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, Self

import torch

__all__ = ["TeacherTargets"]


class TeacherTargets(Protocol):
    """Teacher distillation targets precomputed once per rollout.

    The PPO trainer computes them in segment chunks, joins the chunks with
    ``concat`` and slices each update minibatch with ``index``. Both act on the
    leading (segment) dimension of every tensor. Implementations fail fast when
    chunks disagree about which optional targets they carry.
    """

    def index(self, indices: torch.Tensor) -> Self:
        """Select ``indices`` along the leading (segment) dimension."""
        ...

    @classmethod
    def concat(cls, chunks: Sequence[Self]) -> Self:
        """Concatenate chunks along the leading (segment) dimension."""
        ...
