"""Kaggriculture BC shards (Task 5.1 output) and their rank-resident loader (5.2).

The shard format is the refreshed Task 5.1 brief's ``kaggriculture-bc-shard-v1``
(``ops/rebuild-2026-09-29/briefs/5.1-bc-data.md``, branch
``kg/rebuild-bc-brief``): one ``np.savez_compressed`` file per episode at
``<split>/<day>-<episode_id>.npz`` holding

- ``schema``: 0-d unicode ``"kaggriculture-bc-shard-v1"``;
- the 28 contract-v4 observation fields at their contract dtypes, ``[T, 2, ...]``;
- ``can_act`` bool ``[T, 2, 252]``, ``tokens`` int64 ``[T, 2, 252, 12]`` and
  ``lengths`` int64 ``[T, 2]`` (STOP included);
- ``policy_seat`` bool ``[T, 2]``: the seat rows whose recorded program is a
  policy target (the episode's winner by final bank, both seats on a draw);
  every row has at least one. Both seat rows always carry the winner target, so
  the critic sees both outcomes;
- ``turn`` int64 ``[T]``: the original turn index, so rejected gaps stay visible.

``manifest.json`` is strict JSON. ``BCManifest`` types and requires the keys BC
reads (schema and version ids; per episode ``episode_id``, ``split``,
``terminal_banks`` in recorded seat order, ``admitted``, ``shard_path``,
``shard_sha256``, ``shard_bytes``); preparation's other custody keys (run and
source identity, rejection and normalization counts, totals) are kept but not
interpreted here. ``write_bc_episode`` / ``write_bc_manifest`` write the same
format, for preparation and for tests.

Residency is 5.2's choice (the brief's dense estimate is about 52 GB for the
reference slice). Each rank keeps rows ``[rank::world_size]`` of every episode
(the reference's split) in host memory, compacted losslessly: exact integer
fields as int32 and tokens as int16, each range-checked, and cast back to int64
when gathered. Loading verifies every shard's SHA-256 and byte count and checks
its full contract once, so gathered batches need no per-batch checks. Episode
ids and splits are custody labels and never reach the model; recorded programs
are admitted by the model's own replay validation (``evaluate_actions``).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field

from owl.kaggriculture import types as kt

SHARD_SCHEMA = "kaggriculture-bc-shard-v1"
CONTRACT_DOCUMENT_VERSION = 4
OBSERVATION_SCHEMA_VERSION = 3
MANIFEST_NAME = "manifest.json"
Split = Literal["train", "validation"]

# The contract's named observation tensors, in KaggricultureObsBatch order.
# Iterating the pinned schema's field names is the sanctioned dynamic case.
OBS_FIELDS: tuple[str, ...] = tuple(
    name for name in kt.KaggricultureObsBatch.model_fields if name != "action_mask"
)
ACTION_FIELDS: tuple[str, ...] = ("can_act", "tokens", "lengths", "policy_seat")
ROW_FIELDS: tuple[str, ...] = (*OBS_FIELDS, *ACTION_FIELDS)
SHARD_ARRAYS: frozenset[str] = frozenset({"schema", *ROW_FIELDS, "turn"})

_ACTION_LAYOUT: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
    "can_act": (torch.bool, (kt.PLAYERS, kt.MAX_FRAMES)),
    "tokens": (torch.int64, (kt.PLAYERS, kt.MAX_FRAMES, kt.ACTION_SLOTS)),
    "lengths": (torch.int64, (kt.PLAYERS,)),
    "policy_seat": (torch.bool, (kt.PLAYERS,)),
}
# Lossless in-memory storage for exact integers; gather casts back to int64.
_TOKEN_STORAGE = torch.int16
_INTEGER_STORAGE = torch.int32
_COMPACT_INTEGERS = (_TOKEN_STORAGE, _INTEGER_STORAGE)


class BCManifestEpisode(BaseModel):
    """One episode's manifest record; keys beyond these are 5.1 custody."""

    model_config = ConfigDict(extra="allow", frozen=True)

    episode_id: str = Field(min_length=1)
    split: Split
    terminal_banks: tuple[float, float]
    admitted: int = Field(ge=0)
    shard_path: str = Field(min_length=1)
    shard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_bytes: int = Field(ge=1)


class BCManifest(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    schema_id: Literal["kaggriculture-bc-shard-v1"] = Field(alias="schema")
    observation_schema_version: Literal[3]
    contract_document_version: Literal[4]
    episodes: tuple[BCManifestEpisode, ...]


@dataclass(frozen=True)
class BCEpisode:
    """One episode's admitted turns, in memory (writer input)."""

    episode_id: str
    split: Split
    day: str
    obs: kt.KaggricultureObsBatch
    actions: kt.KaggricultureActions
    policy_seat: torch.Tensor
    turn: torch.Tensor
    terminal_banks: tuple[float, float]


@dataclass(frozen=True)
class BCBatch:
    """Gathered rows. ``final_banks`` is float64 ``[rows, 2]`` in seat order;
    ``policy_seat`` is bool ``[rows, 2]`` (seat rows that are policy targets)."""

    obs: kt.KaggricultureObsBatch
    actions: kt.KaggricultureActions
    final_banks: torch.Tensor
    policy_seat: torch.Tensor

    def to(self, device: torch.device) -> BCBatch:
        if device.type == "cpu":
            return self
        return BCBatch(
            obs=_map_obs(self.obs, lambda t: _to_device(t, device)),
            actions=kt.KaggricultureActions(
                tokens=_to_device(self.actions.tokens, device),
                lengths=_to_device(self.actions.lengths, device),
            ),
            final_banks=_to_device(self.final_banks, device),
            policy_seat=_to_device(self.policy_seat, device),
        )


class BCSplit:
    """This rank's resident rows of one split, addressed ``0 .. num_rows - 1``.

    ``episode`` (an index into ``episode_ids``) and ``turn`` identify each row.
    ``rank_rows`` holds every rank's row count, computed from the manifest.
    """

    def __init__(
        self,
        split: Split,
        *,
        store: dict[str, torch.Tensor],
        episode: torch.Tensor,
        turn: torch.Tensor,
        episode_ids: tuple[str, ...],
        terminal_banks: torch.Tensor,
        rank_rows: tuple[int, ...],
    ) -> None:
        self.split = split
        self._store = store
        self.episode = episode
        self.turn = turn
        self.episode_ids = episode_ids
        self._terminal_banks = terminal_banks
        self.rank_rows = rank_rows
        self.num_rows = int(episode.shape[0])

    def gather(self, rows: npt.NDArray[np.int64]) -> BCBatch:
        """Rows in the requested order, as contract-dtype CPU tensors."""
        index = torch.tensor(np.asarray(rows, dtype=np.int64))
        if index.ndim != 1 or index.numel() == 0:
            raise ValueError("gather needs a non-empty 1-D row index")
        if int(index.min()) < 0 or int(index.max()) >= self.num_rows:
            raise IndexError(f"{self.split} rows must be in [0, {self.num_rows})")
        out = {
            name: _restore(values.index_select(0, index))
            for name, values in self._store.items()
        }
        return BCBatch(
            obs=kt.KaggricultureObsBatch(
                **{name: out[name] for name in OBS_FIELDS},
                action_mask=kt.KaggricultureActionMask(can_act=out["can_act"]),
            ),
            actions=kt.KaggricultureActions(
                tokens=out["tokens"], lengths=out["lengths"]
            ),
            final_banks=self._terminal_banks.index_select(
                0, self.episode.index_select(0, index)
            ),
            policy_seat=out["policy_seat"],
        )


@dataclass(frozen=True)
class BCDataset:
    root: Path
    manifest: BCManifest
    manifest_sha256: str
    rank: int
    world_size: int
    train: BCSplit
    validation: BCSplit


def shard_path(split: Split, day: str, episode_id: str) -> str:
    return f"{split}/{day}-{episode_id}.npz"


def write_bc_episode(root: Path, episode: BCEpisode) -> BCManifestEpisode:
    """Write one episode shard and return its manifest record; never overwrite."""
    rows = int(episode.turn.shape[0])
    lead = (rows, kt.PLAYERS)
    if (
        tuple(episode.obs.still_playing.shape) != lead
        or tuple(episode.actions.lengths.shape) != lead
        or tuple(episode.policy_seat.shape) != lead
    ):
        raise ValueError(f"episode {episode.episode_id}: rows must lead with {lead}")
    relative = shard_path(episode.split, episode.day, episode.episode_id)
    _check_shard_path(relative, episode.split)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``Any`` values: numpy's stub would bind a bool-typed ``allow_pickle`` key.
    arrays: dict[str, Any] = {
        name: _obs_field(episode.obs, name).detach().cpu().numpy()
        for name in OBS_FIELDS
    }
    arrays["can_act"] = episode.obs.action_mask.can_act.detach().cpu().numpy()
    arrays["tokens"] = episode.actions.tokens.detach().cpu().numpy()
    arrays["lengths"] = episode.actions.lengths.detach().cpu().numpy()
    arrays["policy_seat"] = episode.policy_seat.detach().cpu().numpy()
    arrays["turn"] = episode.turn.detach().cpu().numpy()
    with path.open("xb") as handle:
        np.savez_compressed(handle, schema=np.array(SHARD_SCHEMA), **arrays)
    return BCManifestEpisode(
        episode_id=episode.episode_id,
        split=episode.split,
        terminal_banks=episode.terminal_banks,
        admitted=rows,
        shard_path=relative,
        shard_sha256=_sha256(path),
        shard_bytes=path.stat().st_size,
    )


def write_bc_manifest(
    root: Path, episodes: Sequence[BCManifestEpisode], **custody: object
) -> BCManifest:
    """Write ``manifest.json``; ``custody`` carries preparation's other keys."""
    manifest = BCManifest.model_validate(
        {
            **custody,
            "schema": SHARD_SCHEMA,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "contract_document_version": CONTRACT_DOCUMENT_VERSION,
            "episodes": [e.model_dump(mode="json") for e in episodes],
        }
    )
    with (root / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(by_alias=True, indent=2) + "\n")
    return manifest


def load_bc_dataset(root: Path, *, rank: int = 0, world_size: int = 1) -> BCDataset:
    """Verify custody and every shard's contract; keep rows ``[rank::world]``."""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world size {world_size}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BC dataset manifest not found: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = BCManifest.model_validate_json(manifest_bytes)
    ids = [episode.episode_id for episode in manifest.episodes]
    duplicates = sorted({e for e in ids if ids.count(e) > 1})
    if duplicates:
        raise ValueError(f"BC episodes appear more than once: {duplicates}")
    return BCDataset(
        root=root,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        rank=rank,
        world_size=world_size,
        train=_load_split(root, "train", manifest, rank=rank, world_size=world_size),
        validation=_load_split(
            root, "validation", manifest, rank=rank, world_size=world_size
        ),
    )


def _load_split(
    root: Path,
    split: Split,
    manifest: BCManifest,
    *,
    rank: int,
    world_size: int,
) -> BCSplit:
    records = [e for e in manifest.episodes if e.split == split and e.admitted > 0]
    if not records:
        raise ValueError(f"BC dataset has no admitted {split} rows")
    rank_rows = tuple(
        sum(len(range(r, e.admitted, world_size)) for e in records)
        for r in range(world_size)
    )
    total = rank_rows[rank]
    store: dict[str, torch.Tensor] = {}
    episode = torch.empty(total, dtype=torch.int32)
    turn = torch.empty(total, dtype=torch.int64)
    offset = 0
    mine = slice(rank, None, world_size)
    for index, record in enumerate(records):
        arrays = _read_shard(root, record)
        count = len(range(rank, record.admitted, world_size))
        for name in ROW_FIELDS:
            values = _compact(name, torch.from_numpy(arrays[name][mine]), record)
            if name not in store:
                store[name] = torch.empty(
                    (total, *values.shape[1:]), dtype=values.dtype
                )
            store[name][offset : offset + count] = values
        episode[offset : offset + count] = index
        turn[offset : offset + count] = torch.from_numpy(arrays["turn"][mine])
        offset += count
    return BCSplit(
        split,
        store=store,
        episode=episode,
        turn=turn,
        episode_ids=tuple(e.episode_id for e in records),
        terminal_banks=torch.tensor(
            [e.terminal_banks for e in records], dtype=torch.float64
        ),
        rank_rows=rank_rows,
    )


def _read_shard(
    root: Path, record: BCManifestEpisode
) -> dict[str, npt.NDArray[np.generic]]:
    """One shard, fully checked: custody, schema id, layout and contract."""
    _check_shard_path(record.shard_path, record.split)
    if not all(math.isfinite(bank) for bank in record.terminal_banks):
        raise ValueError(f"episode {record.episode_id} terminal_banks must be finite")
    path = root / record.shard_path
    if not path.is_file():
        raise FileNotFoundError(f"BC shard missing: {path}")
    if path.stat().st_size != record.shard_bytes:
        raise ValueError(f"{path} size differs from the manifest")
    if _sha256(path) != record.shard_sha256:
        raise ValueError(f"{path} SHA-256 does not match the manifest")
    with np.load(path, allow_pickle=False) as shard:
        names = set(shard.files)
        if "schema" not in names or str(shard["schema"][()]) != SHARD_SCHEMA:
            raise ValueError(f"{path} is not a {SHARD_SCHEMA} shard")
        if names != SHARD_ARRAYS:
            raise ValueError(
                f"{path} arrays must be exactly {sorted(SHARD_ARRAYS)}, got "
                f"{sorted(names)}"
            )
        arrays = {name: shard[name] for name in (*ROW_FIELDS, "turn")}
    for name, array in arrays.items():
        if array.shape[:1] != (record.admitted,):
            raise ValueError(f"{path} {name} must lead with {record.admitted} rows")
    turn = arrays["turn"]
    if turn.dtype != np.int64 or turn.ndim != 1:
        raise ValueError(f"{path} turn must be int64 [T]")
    if int(turn[0]) < 0 or not bool(np.all(np.diff(turn) > 0)):
        raise ValueError(f"{path} turn must be non-negative and increasing")
    tensors = {name: torch.from_numpy(arrays[name]) for name in ROW_FIELDS}
    for name, (dtype, trailing) in _ACTION_LAYOUT.items():
        if tensors[name].dtype != dtype or tuple(tensors[name].shape[1:]) != trailing:
            raise ValueError(f"{path} {name} must be {dtype} {trailing}")
    obs = kt.KaggricultureObsBatch(
        **{name: tensors[name] for name in OBS_FIELDS},
        action_mask=kt.KaggricultureActionMask(can_act=tensors["can_act"]),
    )
    try:
        obs.check_contract()
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not bool(obs.still_playing.all()):
        raise ValueError(
            f"{path}: rows are admitted paired turns; every seat must be still_playing"
        )
    lengths = tensors["lengths"]
    if int(lengths.min()) < 1 or bool(
        (lengths > obs.action_mask.can_act.sum(dim=-1)).any()
    ):
        raise ValueError(f"{path}: lengths must be in [1, can_act.sum(-1)]")
    if int(tensors["tokens"].min()) < 0:
        raise ValueError(f"{path}: tokens must be >= 0")
    if not bool(tensors["policy_seat"].any(dim=-1).all()):
        raise ValueError(f"{path}: every row needs at least one policy_seat")
    return arrays


def _compact(
    name: str, values: torch.Tensor, record: BCManifestEpisode
) -> torch.Tensor:
    """Declared lossless storage cast: int64 fields to int32 (tokens int16)."""
    if values.dtype != torch.int64:
        return values
    storage = _TOKEN_STORAGE if name == "tokens" else _INTEGER_STORAGE
    info = torch.iinfo(storage)
    if values.numel() and (
        int(values.min()) < info.min or int(values.max()) > info.max
    ):
        raise ValueError(
            f"episode {record.episode_id} {name} exceeds the {storage} storage range"
        )
    return values.to(storage)


def _restore(values: torch.Tensor) -> torch.Tensor:
    """Undo ``_compact``: every stored integer field is int64 in the contract."""
    return values.to(torch.int64) if values.dtype in _COMPACT_INTEGERS else values


def _check_shard_path(path: str, split: Split) -> None:
    posix = PurePosixPath(path)
    if (
        posix.is_absolute()
        or "\\" in path
        or any(part in ("", ".", "..") for part in posix.parts)
        or len(posix.parts) != 2
        or posix.parts[0] != split
        or posix.suffix != ".npz"
    ):
        raise ValueError(f"shard path must be '{split}/<day>-<episode>.npz': {path!r}")


def _obs_field(obs: kt.KaggricultureObsBatch, name: str) -> torch.Tensor:
    value = getattr(obs, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"observation field {name} must be a tensor")
    return value


def _map_obs(
    obs: kt.KaggricultureObsBatch, fn: Callable[[torch.Tensor], torch.Tensor]
) -> kt.KaggricultureObsBatch:
    return kt.KaggricultureObsBatch(
        **{name: fn(_obs_field(obs, name)) for name in OBS_FIELDS},
        action_mask=kt.KaggricultureActionMask(can_act=fn(obs.action_mask.can_act)),
    )


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
