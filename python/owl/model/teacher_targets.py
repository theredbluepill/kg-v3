from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, Self

import torch

__all__ = ["TeacherTargets"]


class TeacherTargets(Protocol):
    """Teacher distillation targets precomputed once per rollout.

    The PPO trainer computes them in segment chunks, joins the chunks with
    ``concat`` and slices each update minibatch with ``index``. Both act on the
    leading (segment) dimension of every tensor. The protocol does not require
    ``concat`` to validate that chunks carry the same optional targets; see each
    implementation. ``CachedTeacherDistillationTargets.concat`` follows the first
    chunk: it raises when a later chunk lacks an optional target the first chunk
    carries, but silently drops a target that only later chunks carry.
    """

    def index(self, indices: torch.Tensor) -> Self:
        """Select ``indices`` along the leading (segment) dimension."""
        ...

    @classmethod
    def concat(cls, chunks: Sequence[Self]) -> Self:
        """Concatenate chunks along the leading (segment) dimension."""
        ...
