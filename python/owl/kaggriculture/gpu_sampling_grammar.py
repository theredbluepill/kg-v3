"""Device-resident native grammar tables, with no Python grammar reconstruction.

Construction validates and deduplicates native plans before bounded tensor upload.
Hot methods use tensor indexing only. Node -1 is terminal (dummy choice zero);
invalid nodes, wrong slots and illegal choices yield no support / poison node -2.
Slot zero retains the native 241-wide actor vocabulary; callers with smaller heads
may slice its mask to their model vocabulary after admitting compatible shapes.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import torch

from owl.kaggriculture.native_bridge import MyolieSampler

WIDTHS = (241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2)


class GrammarBatch:
    def __init__(
        self,
        native: MyolieSampler,
        shapes: Sequence[tuple[int, int, int]],
        device: torch.device | str,
        max_bytes: int = 256 * 1024**2,
    ) -> None:
        self.shapes = tuple(shapes)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        plans = {}
        total_nodes = total_edges = 0
        for shape in self.shapes:
            if (
                len(shape) != 3
                or any(type(v) is not int for v in shape)
                or not (
                    1 <= shape[0] <= 241
                    and 1 <= shape[1] <= 10
                    and 1 <= shape[2] <= 241
                )
            ):
                raise ValueError("invalid native grammar shape")
            if shape in plans:
                continue
            plan = native.plan(*shape)
            n = len(plan.nodes)
            if not n or not 0 <= plan.start < n:
                raise ValueError("invalid native grammar start")
            for mask, allowed, transitions, pending_hires, slot in plan.nodes:
                if (
                    not 0 <= slot < 12
                    or len(mask) != WIDTHS[slot]
                    or not 0 <= pending_hires <= shape[1]
                ):
                    raise ValueError("invalid native grammar node")
                bits, nxt = np.asarray(mask), np.asarray(transitions)
                if (
                    bits.dtype != np.bool_
                    or nxt.shape != bits.shape
                    or nxt.dtype.kind not in "iu"
                    or not bits.any()
                    or tuple(np.flatnonzero(bits)) != tuple(allowed)
                    or np.any(nxt[~bits] != -2)
                    or np.any((nxt[bits] < -1) | (nxt[bits] >= n))
                ):
                    raise ValueError("invalid native grammar edges")
                for token in allowed:
                    successor = int(nxt[token])
                    if successor == -1:
                        if slot != 11 or token != 1:
                            raise ValueError("invalid native grammar terminal")
                    elif plan.nodes[successor][4] != (slot + 1) % 12:
                        raise ValueError("invalid native grammar successor slot")
                total_edges += len(mask)
            plans[shape] = (plan, total_nodes)
            total_nodes += n
            # Three int64 node columns, bool/int64 edges, starts and cached ranges.
            needed = (
                (total_nodes + 1) * 24
                + (total_edges + max(WIDTHS)) * 9
                + len(self.shapes) * 8
                + sum(WIDTHS) * 8
            )
            if needed > max_bytes:
                raise MemoryError(
                    f"native device grammar needs {needed} bytes; cap is {max_bytes}"
                )
        self.storage_bytes = (
            (total_nodes + 1) * 24
            + (total_edges + max(WIDTHS)) * 9
            + len(self.shapes) * 8
            + sum(WIDTHS) * 8
        )
        if self.storage_bytes > max_bytes:
            raise MemoryError("native device grammar exceeds byte cap")
        offsets = np.empty(total_nodes + 1, dtype=np.int64)
        hires = np.zeros(total_nodes + 1, dtype=np.int64)
        slots = np.full(total_nodes + 1, -1, dtype=np.int64)
        masks = np.zeros(total_edges + max(WIDTHS), dtype=np.bool_)
        next_nodes = np.full(total_edges + max(WIDTHS), -2, dtype=np.int64)
        edge = 0
        for plan, base in plans.values():
            for local, (mask, _, transitions, pending, slot) in enumerate(plan.nodes):
                idx, width = base + local, len(mask)
                offsets[idx], hires[idx], slots[idx] = edge, pending, slot
                masks[edge : edge + width] = mask
                nxt = np.asarray(transitions, dtype=np.int64)
                next_nodes[edge : edge + width] = np.where(nxt >= 0, nxt + base, nxt)
                edge += width
        offsets[-1] = total_edges
        masks[total_edges] = True
        next_nodes[total_edges] = -1
        self.node_count = total_nodes
        self.unique_shapes = tuple(plans)
        self.shape_offsets = {shape: base for shape, (_, base) in plans.items()}
        self.device = torch.device(device)
        self._offsets = torch.tensor(offsets, device=self.device)
        self._hires = torch.tensor(hires, device=self.device)
        self._slots = torch.tensor(slots, device=self.device)
        self._masks = torch.tensor(masks, device=self.device)
        self._next = torch.tensor(next_nodes, device=self.device)
        self._ranges = tuple(torch.arange(w, device=self.device) for w in WIDTHS)
        self.plan_starts = {
            shape: base + plan.start for shape, (plan, base) in plans.items()
        }
        self.starts = torch.tensor(
            [self.plan_starts[s] for s in self.shapes],
            dtype=torch.long,
            device=self.device,
        )
        self.device = (
            self.starts.device
        )  # Canonical CUDA index ("cuda" resolves to e.g. "cuda:0").

    def rebind(self, shapes: Sequence[tuple[int, int, int]]) -> GrammarBatch:
        """Bind existing tables to another list drawn from ``unique_shapes``.

        A collector's seats change actor counts every turn. Reusing tables avoids
        repeating validation and device uploads on each step.
        """
        shapes = tuple(shapes)
        missing = sorted(set(shapes) - set(self.plan_starts))
        if missing:
            raise ValueError(f"shapes outside this grammar batch: {missing}")
        view = copy.copy(self)
        view.shapes = shapes
        view.starts = torch.tensor(
            [self.plan_starts[s] for s in shapes], dtype=torch.long, device=self.device
        )
        return view

    def _rows(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        live = (nodes >= 0) & (nodes < self.node_count)
        rows = torch.where(live, nodes, self.node_count)
        return rows, live

    def hires(self, nodes: torch.Tensor) -> torch.Tensor:
        rows, _ = self._rows(nodes)
        return self._hires[rows]

    def options(self, nodes: torch.Tensor, slot: int) -> torch.Tensor:
        rows, live = self._rows(nodes)
        admitted = (live & (self._slots[rows] == slot)) | (nodes == -1)
        indices = self._offsets[rows, None] + self._ranges[slot]
        return self._masks[indices] & admitted[:, None]

    def advance(
        self, nodes: torch.Tensor, choices: torch.Tensor, slot: int
    ) -> torch.Tensor:
        rows, live = self._rows(nodes)
        admitted = (live & (self._slots[rows] == slot)) | (nodes == -1)
        admitted = admitted & (choices >= 0) & (choices < WIDTHS[slot])
        indices = self._offsets[rows] + choices.clamp(0, WIDTHS[slot] - 1)
        return torch.where(admitted, self._next[indices], -2)
