"""Thin NumPy ownership layer over the starter's owl.rs PyO3 extension.

Array validation and immutable grammar wire parsing are adapted from v2.
All native calls use the same maturin-built extension as the training starter;
there is no ctypes loading, Python engine dependency or build-on-import.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from owl.rs import KaggricultureBatch, kaggriculture_decode, kaggriculture_grammar_plan

MYOLIE_FEATURE_COUNT = 8165
MYOLIE_INVEST_FEATURE_COUNT = 8176
MYOLIE_OBSERVATION_WIDTHS = {1: MYOLIE_FEATURE_COUNT, 2: MYOLIE_INVEST_FEATURE_COUNT}
MYOLIE_MAX_FRAMES = 252
MYOLIE_SLOTS = 12


def _checked_array(
    value: object,
    dtype: Any,
    shape: tuple[int, ...],
    name: str,
    *,
    writable: bool = False,
) -> NDArray[Any]:
    """Refuse silent copies/casts so callers own every byte passed to ctypes."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.dtype(dtype) or not value.dtype.isnative:
        raise TypeError(f"{name} must have native {np.dtype(dtype)} dtype")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _disjoint_arrays(arrays: Sequence[NDArray[Any]]) -> None:
    regions = [
        (array.ctypes.data, array.ctypes.data + array.nbytes) for array in arrays
    ]
    for index, (start, end) in enumerate(regions):
        if any(
            start < other_end and other_start < end
            for other_start, other_end in regions[:index]
        ):
            raise ValueError("Myolie input/output arrays must not overlap")


class MyolieBuffers:
    """Reusable explicit-version observation storage, indexed [game, seat, feature].

    Version 1 retains width 8165; opt-in version 2 appends investment suffix v1.

    Supplied arrays are used without copies (including ``pinned_tensor.numpy()``).
    ``owner`` optionally retains the backing tensors/allocator. Every successful
    observe or step overwrites the same storage; copy historical rows before the
    next call. Views/tensors of these arrays observe those overwrites immediately.
    Context columns: step, seat0 actor count, seat1 actor count, market order limit.
    Banks are raw cash; this ABI does not replace the trainer's reward or GAE.
    """

    def __init__(
        self,
        n_envs: int,
        *,
        features: NDArray[Any] | None = None,
        context: NDArray[Any] | None = None,
        banks: NDArray[Any] | None = None,
        done: NDArray[Any] | None = None,
        owner: object | None = None,
        observation_version: int = 1,
    ) -> None:
        if (
            not isinstance(n_envs, (int, np.integer))
            or isinstance(n_envs, bool)
            or n_envs < 1
        ):
            raise ValueError("n_envs must be a positive integer")
        self.n_envs = int(n_envs)
        if (
            type(observation_version) is not int
            or observation_version not in MYOLIE_OBSERVATION_WIDTHS
        ):
            raise ValueError("unsupported Myolie observation version")
        self.observation_version = observation_version
        self.feature_count = MYOLIE_OBSERVATION_WIDTHS[observation_version]
        self.owner = owner
        self.features = (
            np.empty((self.n_envs, 2, self.feature_count), dtype=np.float32)
            if features is None
            else features
        )
        self.context = (
            np.empty((self.n_envs, 4), dtype=np.int64) if context is None else context
        )
        self.banks = (
            np.empty((self.n_envs, 2), dtype=np.float64) if banks is None else banks
        )
        self.done = np.empty((self.n_envs,), dtype=np.uint8) if done is None else done
        self._validate(self.n_envs)

    def _validate(self, n_envs: int) -> list[NDArray[Any]]:
        if (
            type(self.observation_version) is not int
            or self.observation_version not in MYOLIE_OBSERVATION_WIDTHS
            or self.feature_count != MYOLIE_OBSERVATION_WIDTHS[self.observation_version]
        ):
            raise ValueError("unsupported Myolie observation buffer version")
        arrays = [
            _checked_array(
                self.features,
                np.float32,
                (n_envs, 2, self.feature_count),
                "features",
                writable=True,
            ),
            _checked_array(
                self.context, np.int64, (n_envs, 4), "context", writable=True
            ),
            _checked_array(self.banks, np.float64, (n_envs, 2), "banks", writable=True),
            _checked_array(self.done, np.uint8, (n_envs,), "done", writable=True),
        ]
        _disjoint_arrays(arrays)
        return arrays


class _SamplerPlan:
    """Immutable native masks and token transitions; no Python grammar rules."""

    def __init__(self, payload: bytes) -> None:
        if len(payload) < 20:
            raise RuntimeError("truncated native sampler plan")
        magic, version, count, start, edges = map(
            int, np.frombuffer(payload, "<u4", count=5)
        )
        if (
            magic != 0x4D534731
            or version != 1
            or not 0 < count <= 100_000
            or start >= count
        ):
            raise RuntimeError("unsupported native sampler plan")
        if edges > 2_000_000 or len(payload) != 20 + 16 * count + 5 * edges:
            raise RuntimeError("invalid native sampler plan size")
        metadata = np.frombuffer(payload, "<u4", count=4 * count, offset=20).reshape(
            count, 4
        )
        masks = np.frombuffer(payload, np.uint8, count=edges, offset=20 + 16 * count)
        transitions = np.frombuffer(
            payload, "<i4", count=edges, offset=20 + 16 * count + edges
        )
        widths = (241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2)
        nodes: list[
            tuple[tuple[bool, ...], tuple[int, ...], NDArray[Any], int, int]
        ] = []
        expected_offset = 0
        size = sys.getsizeof(payload) + sys.getsizeof(nodes)
        for slot, hires, offset, width in metadata:
            slot, hires, offset, width = map(int, (slot, hires, offset, width))
            if (
                slot >= 12
                or hires > 10
                or width != widths[slot]
                or offset != expected_offset
                or offset + width > edges
            ):
                raise RuntimeError("invalid native sampler node")
            bits = masks[offset : offset + width]
            next_nodes = transitions[offset : offset + width]
            if np.any(bits > 1) or not np.any(bits):
                raise RuntimeError("invalid native sampler mask")
            if np.any(next_nodes[bits == 0] != -2) or np.any(
                (next_nodes[bits != 0] < -1) | (next_nodes[bits != 0] >= count)
            ):
                raise RuntimeError("invalid native sampler transition")
            terminals = np.flatnonzero(next_nodes == -1)
            if len(terminals) and (slot != 11 or terminals.tolist() != [1]):
                raise RuntimeError("invalid native sampler terminal")
            successors = next_nodes[next_nodes >= 0]
            if np.any(metadata[successors, 0] != (slot + 1) % 12):
                raise RuntimeError("invalid native sampler successor slot")
            mask = tuple(bits.astype(bool).tolist())
            allowed = tuple(np.flatnonzero(bits).tolist())
            node = (mask, allowed, next_nodes, hires, slot)
            nodes.append(node)
            size += (
                sys.getsizeof(node)
                + sys.getsizeof(mask)
                + sys.getsizeof(allowed)
                + sys.getsizeof(next_nodes)
                + 2 * sys.getsizeof(hires)
            )
            expected_offset += width
        if expected_offset != edges:
            raise RuntimeError("native sampler has unused edges")
        self.nodes = tuple(nodes)
        self.start = start
        self.cache_bytes = size + sys.getsizeof(self.nodes)
        self.payload = payload  # Keeps all read-only NumPy transition views alive.


class MyolieSampler:
    """Rust-compiled grammar tables and Rust action decoding for canonical sampling.

    Plans cross the ABI only on a cache miss, finished frames once per action.
    The LRU is bounded by both entry count and an upper estimate of retained
    payload/container bytes; evicted cursors retain their own immutable plan.
    """

    def __init__(
        self, *, cache_entries: int = 16, cache_bytes: int = 32 * 1024 * 1024
    ) -> None:
        if cache_entries < 1 or cache_bytes < 1:
            raise ValueError("native sampler cache bounds must be positive")
        self._cache: OrderedDict[tuple[int, int, int], _SamplerPlan] = OrderedDict()
        self._lock = threading.RLock()
        self.cache_entries, self.cache_limit = cache_entries, cache_bytes
        self.cache_bytes = self.plan_calls = self.decode_calls = 0

    @staticmethod
    def _shape(actors: int, limit: int, hire_limit: int) -> tuple[int, int, int]:
        values = (actors, limit, hire_limit)
        if any(type(v) is not int for v in values) or not (
            1 <= actors <= 241 and 1 <= limit <= 10 and 1 <= hire_limit <= 241
        ):
            raise ValueError(
                "native sampler requires actors 1..241, orders 1..10, HIRE limit 1..241"
            )
        return values

    def plan(self, actors: int, limit: int, hire_limit: int) -> _SamplerPlan:
        key = self._shape(actors, limit, hire_limit)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            plan = _SamplerPlan(kaggriculture_grammar_plan(*key))
            self.plan_calls += 1
            if plan.cache_bytes <= self.cache_limit:
                while self._cache and (
                    len(self._cache) >= self.cache_entries
                    or self.cache_bytes + plan.cache_bytes > self.cache_limit
                ):
                    _, old = self._cache.popitem(last=False)
                    self.cache_bytes -= old.cache_bytes
                self._cache[key] = plan
                self.cache_bytes += plan.cache_bytes
            return plan

    def decode(
        self, actors: int, limit: int, hire_limit: int, frames: NDArray[Any]
    ) -> dict[str, Any]:
        key = self._shape(actors, limit, hire_limit)
        if (
            not isinstance(frames, np.ndarray)
            or frames.ndim != 2
            or frames.shape[1] != 12
        ):
            raise ValueError("native sampler frames must have shape [frames, 12]")
        _checked_array(frames, np.int16, frames.shape, "frames")
        if not 1 <= len(frames) <= 252:
            raise ValueError("native sampler requires 1..252 frames")
        with self._lock:
            result = json.loads(kaggriculture_decode(*key, frames))
            self.decode_calls += 1
            return result


class RustBatch:
    """Own a PyO3 batch; all hot calls use caller-owned arrays."""

    def __init__(self, headers: list[dict[str, Any]], *, threads: int = 1) -> None:
        self._native = KaggricultureBatch(json.dumps(headers), threads)
        self._n_envs = len(headers)

    def reset(self, headers: list[dict[str, Any]]) -> None:
        if len(headers) != self._n_envs:
            raise ValueError("reset must retain the batch size")
        self._native.reset(json.dumps(headers))

    def myolie_observe(self, buffers: MyolieBuffers) -> MyolieBuffers:
        buffers._validate(self._n_envs)
        self._native.observe(
            buffers.features,
            buffers.context,
            buffers.banks,
            buffers.done,
            buffers.observation_version,
        )
        return buffers

    def myolie_step_frames(
        self, frames: NDArray[Any], lengths: NDArray[Any], buffers: MyolieBuffers
    ) -> MyolieBuffers:
        n = self._n_envs
        arrays = buffers._validate(n)
        frames = _checked_array(frames, np.int16, (n, 2, 252, 12), "frames")
        lengths = _checked_array(lengths, np.int32, (n, 2), "lengths")
        _disjoint_arrays([*arrays, frames, lengths])
        self._native.step_frames(
            frames,
            lengths,
            buffers.features,
            buffers.context,
            buffers.banks,
            buffers.done,
            buffers.observation_version,
        )
        return buffers

    def myolie_reset_indices(
        self,
        indices: NDArray[Any],
        headers: list[dict[str, Any]],
        *,
        observation_version: int = 1,
    ) -> None:
        indices = _checked_array(indices, np.uint32, (len(indices),), "indices")
        self._native.reset_indices(indices, json.dumps(headers), observation_version)

    def snapshots(self) -> list[dict[str, Any]]:
        return json.loads(self._native.snapshots())

    def econ(self, counters: NDArray[Any]) -> None:
        _checked_array(
            counters, np.uint64, (self._n_envs, 2, 32), "econ", writable=True
        )
        self._native.econ(counters)

    def close(self) -> None:
        self._native.close()
