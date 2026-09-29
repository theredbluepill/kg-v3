"""Startup workload check against the Kaggriculture model's GEMM chunking limits.

Every Kaggriculture forward runs one row per seat (``2 x envs`` rows). The model
chunks its compiled trunk (``rows_per_chunk``) and its heads
(``head_rows_per_chunk``) so each call stays below the 2**31 GEMM extent
(cookbook reference compiled-gemm-template-overflows-above-2-21-rows). This
module computes, from config shapes alone, the rows each training forward
issues and how the model's own limits divide them, so a workload the model
cannot service fails at startup (``scripts/run_ppo.py``) before any model is
allocated, and the headroom is printed in the run log.

Trunk figures assume every row at the full padded sequence length. That is the
padded path's exact chunk size; the packed path plans chunks from actual token
counts (``packed_row_chunks``), which never exceed padded tokens, so the reported
trunk calls are an upper bound and the trunk headroom a lower bound. Head
chunking depends only on rows, so head figures are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from owl.kaggriculture import types as kt
from owl.model.kaggriculture import (
    KaggricultureTransformerConfig,
    head_rows_per_chunk,
    rows_per_chunk,
    sequence_length,
    trunk_gemm_width,
)


@dataclass(frozen=True)
class ForwardWorkload:
    """Seat rows one model forward receives, per rank."""

    name: str
    rows: int


@dataclass(frozen=True)
class WorkloadHeadroom:
    name: str
    rows: int
    tokens_per_row: int
    trunk_rows_per_call: int
    head_rows_per_call: int

    @property
    def min_trunk_headroom(self) -> float:
        """Full-padding trunk rows per call over rows; a lower bound.

        Below 1 the trunk chunks when rows are fully padded.
        """
        return self.trunk_rows_per_call / self.rows

    @property
    def head_headroom(self) -> float:
        """Head rows per call over rows; below 1 the heads chunk."""
        return self.head_rows_per_call / self.rows

    @property
    def max_trunk_calls(self) -> int:
        """Trunk calls at full padding; packed batches may need fewer."""
        return math.ceil(self.rows / self.trunk_rows_per_call)

    @property
    def head_calls(self) -> int:
        return math.ceil(self.rows / self.head_rows_per_call)

    def log_line(self) -> str:
        return (
            f"GEMM workload headroom {self.name}: {self.rows} rows x "
            f"{self.tokens_per_row} padded tokens; trunk "
            f"{self.trunk_rows_per_call} rows/call at full padding "
            f"(>= {self.min_trunk_headroom:.4g}x headroom, "
            f"<= {self.max_trunk_calls} call(s)); heads "
            f"{self.head_rows_per_call} rows/call "
            f"({self.head_headroom:.4g}x headroom, {self.head_calls} call(s))"
        )


def ppo_forward_workloads(
    *,
    n_envs: int,
    horizon: int,
    segments_per_minibatch: int,
    teacher_segments_per_minibatch: int | None,
) -> tuple[ForwardWorkload, ...]:
    """Per-rank rows of every PPO forward, from Isaiah's trainer loops.

    - rollout: one step of every env, both seats (``n_envs x 2``).
    - minibatch: one optimizer micro-step (``spm x horizon x 2``); gradient
      accumulation repeats it without widening it.
    - teacher_chunk: ``_precompute_teacher_targets`` slices ``n_envs`` into
      chunks of ``teacher_segments_per_minibatch`` segments; ``None`` means no
      teacher is configured and the workload is omitted.
    - evaluation: ``_evaluate_against_last_best`` plays ``n_envs`` games on
      ``n_envs`` envs; ``n_envs x 2`` bounds either model's rows.
    """
    for name, value in (
        ("n_envs", n_envs),
        ("horizon", horizon),
        ("segments_per_minibatch", segments_per_minibatch),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    workloads = [
        ForwardWorkload("rollout", n_envs * kt.PLAYERS),
        ForwardWorkload("minibatch", segments_per_minibatch * horizon * kt.PLAYERS),
    ]
    if teacher_segments_per_minibatch is not None:
        if teacher_segments_per_minibatch < 1:
            raise ValueError(
                "teacher_segments_per_minibatch must be >= 1, got "
                f"{teacher_segments_per_minibatch}"
            )
        teacher_segments = min(teacher_segments_per_minibatch, n_envs)
        workloads.append(
            ForwardWorkload("teacher_chunk", teacher_segments * horizon * kt.PLAYERS)
        )
    workloads.append(ForwardWorkload("evaluation", n_envs * kt.PLAYERS))
    return tuple(workloads)


def check_workload_headroom(
    config: KaggricultureTransformerConfig,
    workloads: tuple[ForwardWorkload, ...],
) -> tuple[WorkloadHeadroom, ...]:
    """Assert each workload is serviceable within the model's chunking limits.

    Uses the model's own ``rows_per_chunk`` (at the full padded sequence length)
    and ``head_rows_per_chunk``: head calls are the calls the model issues, trunk
    calls are their full-padding upper bound. Raises ``ValueError``
    when a single row cannot fit the trunk or head limit (the model would raise
    at the first forward) or a workload is empty.
    """
    if not workloads:
        raise ValueError("no workloads to check")
    tokens = sequence_length(config)
    width = trunk_gemm_width(config)
    trunk_rows = rows_per_chunk(tokens=tokens, width=width)
    if trunk_rows < 1:
        raise ValueError(
            f"one padded row of {tokens} tokens x trunk width {width} reaches the "
            "2**31 GEMM limit; no trunk chunking can service this model"
        )
    head_rows = head_rows_per_chunk(config)
    reports = []
    for workload in workloads:
        if workload.rows < 1:
            raise ValueError(
                f"workload {workload.name!r} has {workload.rows} rows; expected >= 1"
            )
        reports.append(
            WorkloadHeadroom(
                name=workload.name,
                rows=workload.rows,
                tokens_per_row=tokens,
                trunk_rows_per_call=trunk_rows,
                head_rows_per_call=head_rows,
            )
        )
    return tuple(reports)


def headroom_log_lines(reports: tuple[WorkloadHeadroom, ...]) -> tuple[str, ...]:
    """One run-log line per workload, labelling the padded trunk bounds."""
    return tuple(report.log_line() for report in reports)
