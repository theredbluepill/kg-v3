"""Kaggriculture teacher-distillation targets (rebuild Phase 4.2).

``KaggricultureTeacherTargets`` implements Isaiah's ``TeacherTargets`` protocol
for the grammar heads: the frozen teacher's replay-conditioned masked slot
logits and its per-seat winner probabilities, computed once per rollout and
sliced per update minibatch. The targets carry the teacher's
``GrammarSignature``: replay admission cannot detect a teacher grammar that
differs from the student's while still admitting the replayed program, so the
student checks the signature before using the cached logits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, TypeVar

import torch

from owl.kaggriculture import types as kt
from owl.model.kaggriculture_actor import (
    MARKET_POLICY_SLOTS,
    MARKET_POSITIONS,
    POLICY_SLOTS,
    UNIT_POLICY_SLOTS,
)
from owl.model.teacher_targets import TeacherTargets

__all__ = [
    "TEACHER_TARGET_BYTES_PER_ROW",
    "GrammarSignature",
    "KaggricultureTeacherTargets",
    "slot_frames",
]

_FP32_BYTES = 4
_T = TypeVar("_T")


def slot_frames(slot: int) -> int:
    """Frame extent of a policy slot's cached logits: 241 unit frames or 11."""
    if slot in UNIT_POLICY_SLOTS:
        return kt.MAX_ACTORS
    if slot in MARKET_POLICY_SLOTS:
        return MARKET_POSITIONS
    raise ValueError(f"slot {slot} is not a policy slot {POLICY_SLOTS}")


# FP32 cache bytes per seat row: every policy slot's logits plus two winner
# probabilities (102,208 B at contract v4 widths).
TEACHER_TARGET_BYTES_PER_ROW = _FP32_BYTES * (
    sum(slot_frames(slot) * kt.SLOT_WIDTHS[slot] for slot in POLICY_SLOTS) + kt.PLAYERS
)


@dataclass(frozen=True)
class GrammarSignature:
    """Host-side grammar identity: the tables' SHA-256 and ``hire_limit``."""

    tables_sha256: str
    hire_limit: int


@dataclass(frozen=True)
class KaggricultureTeacherTargets(TeacherTargets):
    """Frozen-teacher targets for one rollout, in the observation lead layout.

    ``slot_logits[k]``: density dtype (FP32; FP64 only in FP64 tests),
    ``[*lead, 241, W_k]`` for unit slots or ``[*lead, 11, W_k]`` for market
    slots, ``finfo(dtype).min`` outside the replay-conditioned mask.
    ``winner_probabilities``: FP32 ``[*lead, 2]``, (self, opponent) from each
    seat's view. ``grammar``: the teacher's ``GrammarSignature``. A target
    that was not requested is ``None``.
    """

    slot_logits: dict[int, torch.Tensor] | None
    winner_probabilities: torch.Tensor | None
    grammar: GrammarSignature

    def index(self, indices: torch.Tensor) -> Self:
        """Select ``indices`` along the leading (segment) dimension."""
        return type(self)(
            slot_logits=(
                None
                if self.slot_logits is None
                else {slot: t[indices] for slot, t in self.slot_logits.items()}
            ),
            winner_probabilities=(
                None
                if self.winner_probabilities is None
                else self.winner_probabilities[indices]
            ),
            grammar=self.grammar,
        )

    @classmethod
    def concat(cls, chunks: Sequence[Self]) -> Self:
        """Concatenate chunks along the leading (segment) dimension.

        Validates symmetrically: every chunk must carry the same optional
        targets, the same slot keys and the same grammar, else ``ValueError``
        naming the field. A single chunk is returned as is, without a copy.
        """
        if not chunks:
            raise ValueError(
                "cannot concatenate an empty list of Kaggriculture teacher targets"
            )
        first = chunks[0]
        for position, chunk in enumerate(chunks[1:], start=1):
            if (chunk.slot_logits is None) != (first.slot_logits is None):
                raise ValueError(
                    "inconsistent slot_logits presence across teacher target "
                    f"chunks (chunk 0 vs chunk {position})"
                )
            if (
                chunk.slot_logits is not None
                and first.slot_logits is not None
                and set(chunk.slot_logits) != set(first.slot_logits)
            ):
                raise ValueError(
                    "inconsistent slot_logits keys across teacher target chunks "
                    f"(chunk 0 {sorted(first.slot_logits)} vs chunk {position} "
                    f"{sorted(chunk.slot_logits)})"
                )
            if (chunk.winner_probabilities is None) != (
                first.winner_probabilities is None
            ):
                raise ValueError(
                    "inconsistent winner_probabilities presence across teacher "
                    f"target chunks (chunk 0 vs chunk {position})"
                )
            if chunk.grammar != first.grammar:
                raise ValueError(
                    "inconsistent grammar across teacher target chunks "
                    f"(chunk 0 {first.grammar} vs chunk {position} {chunk.grammar})"
                )
        if len(chunks) == 1:
            return first
        slot_logits: dict[int, torch.Tensor] | None = None
        if first.slot_logits is not None:
            slot_logits = {
                slot: torch.cat([_present(c.slot_logits)[slot] for c in chunks])
                for slot in first.slot_logits
            }
        winner_probabilities: torch.Tensor | None = None
        if first.winner_probabilities is not None:
            winner_probabilities = torch.cat(
                [_present(c.winner_probabilities) for c in chunks]
            )
        return cls(
            slot_logits=slot_logits,
            winner_probabilities=winner_probabilities,
            grammar=first.grammar,
        )

    def nbytes(self) -> int:
        """Bytes held by the cached tensors (metadata only; no device sync)."""
        total = 0
        if self.slot_logits is not None:
            total += sum(t.nbytes for t in self.slot_logits.values())
        if self.winner_probabilities is not None:
            total += self.winner_probabilities.nbytes
        return total


def _present(value: _T | None) -> _T:
    if value is None:
        raise ValueError("a validated teacher target chunk is missing a field")
    return value
