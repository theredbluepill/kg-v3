"""Kaggle episode archives -> ``kaggriculture-bc-shard-v1`` BC shards (Task 5.1).

Lean preparation for the owner's BC data: every episode of the chosen days
(default 2026-09-21..27), imitating one seat per episode, the winner by final
bank (both seats on a draw).

Input is a directory of day archives ``kaggriculture-episodes-YYYY-MM-DD.zip``
whose members are ``<episode_id>.json`` Kaggle episodes. On the RunPod volume
``4llk4uaf20`` that directory is
``<mount>/kaggriculture-v2/public-episodes-2026-09-14-to-2026-09-27/archives``;
the pod must confirm its actual mount path (``/workspace`` is the RunPod
default, the historical reader used ``/data``).

Per episode (one worker process each, one ZIP member in memory at a time):

1. Envelope: 720 steps of two seats, ``info.EpisodeId`` equal to the member
   name, an integer ``info.seed`` (custody only, never encoded), both final
   statuses ``DONE``, the full Rust config key set.
2. Winner seat(s) by final bank ``farms[s].money`` at the last step.
3. For each kept turn ``t`` in ``0..718`` (all turns unless ``--turn-stride``),
   the observation is ``steps[t]`` (seat 1's shared keys come from seat 0) and
   the label of seat ``s`` is ``steps[t + 1][s].action``: Kaggle records the
   action answering step ``t``'s observation at step ``t + 1``. The explicit
   state header (placeholder seed 0) is encoded by the native Task 1.3 encoder
   into both seat rows.
4. Each policy seat's action is admitted strictly by the native Task 1.4 codec
   (``kaggriculture_encode`` plus a decode round trip); falsy ``hands`` /
   ``market`` values become ``[]`` as in the reference, counted by kind. A turn
   is kept only if every policy seat is admitted; rejections are counted by
   reason. The other seat row keeps its observation (the critic trains on both
   outcomes) and carries an all-PASS placeholder program, which the trainer
   never imitates (``policy_seat`` is false).
5. The episode's split is a deterministic hash of its id (about
   ``--validation-fraction``), and admitted rows go to one shard at
   ``<split>/<day>-<episode_id>.npz`` through ``owl.kaggriculture.bc_data``.

Every finished episode leaves ``records/<day>-<episode_id>.json``; a rerun
skips those (resumable), and ``manifest.json`` is written once at the end.
``records/identity.json`` binds the records to the source checkout, the
label-affecting settings and every archive's SHA-256; a rerun whose identity
differs fails instead of reusing stale labels or winners.
Agent and team names are never read.

``--pairing-sample N`` first steps N episodes through kaggle-environments' own
Kaggriculture interpreter (with the recorded ``info.seed``) from every
``steps[t]`` with ``steps[t + 1].action`` and counts how many transitions
reproduce ``steps[t + 1]``: evidence for the step+1 pairing on this corpus.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from owl import rs
from owl.kaggriculture import bc_data
from owl.kaggriculture import types as kt

EPISODE_STEPS = 720
PAIRED_TURNS = EPISODE_STEPS - 1
HIRE_LIMIT = 241
ENCODE_BATCH = 32
RECORD_DIR = "records"
IDENTITY_NAME = "identity.json"
LABEL_PAIRING = "observation steps[t], action steps[t+1]"
PAIRING_NAME = "pairing.json"
ARCHIVE_PATTERN = "kaggriculture-episodes-{month}-{day:02d}.zip"
PUBLIC_KEYS = ("step", "day", "hour", "farms", "market", "town")
PAIRING_PUBLIC_KEYS = ("day", "hour", "farms", "market", "town")
CONFIG_KEYS = frozenset(
    {
        "episodeSteps",
        "boardSize",
        "startingMoney",
        "maxMarketOrdersPerTurn",
        "turnsPerDay",
        "shedCapacity",
        "weedSpawnChance",
        "townShopUnlockInterval",
        "townShopSellInterval",
        "townCenterSellInterval",
        "farmHandCostMult",
        "marketParams",
    }
)
CONFIG_EXTRA_KEYS = frozenset({"actTimeout", "runTimeout", "seed"})
ACTION_KEYS = frozenset({"farmer", "hands", "market"})
_EPISODE_ID = re.compile(r"^[0-9A-Za-z_-]+$")
_ABSENT = object()


class EpisodeRejected(Exception):
    """The whole episode is unusable; the message is the counted reason."""


@dataclass(frozen=True)
class EpisodeTask:
    archive: str
    member: str
    day: str
    episode_id: str
    split: bc_data.Split
    turn_stride: int


# --- pure helpers (unit tested) ---------------------------------------------------


def validation_split(episode_id: str, fraction: float) -> bc_data.Split:
    """Deterministic episode-level split: a hash of the id, never of its content."""
    bucket = int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16) % 10_000
    return "validation" if bucket < round(fraction * 10_000) else "train"


def kept_turns(episode_id: str, stride: int) -> list[int]:
    """Turns ``t`` in ``0..718`` kept under ``stride``, phase-shifted per episode."""
    if stride < 1:
        raise ValueError(f"turn stride must be >= 1, got {stride}")
    offset = int(hashlib.sha256(f"turn:{episode_id}".encode()).hexdigest()[:8], 16)
    return [t for t in range(PAIRED_TURNS) if (t + offset) % stride == 0]


def winner_seats(banks: Sequence[float]) -> tuple[bool, bool]:
    """Policy seats: the larger final bank, both on a draw."""
    return (banks[0] >= banks[1], banks[1] >= banks[0])


def check_envelope(data: dict[str, Any], episode_id: str) -> None:
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) != EPISODE_STEPS:
        count = len(steps) if isinstance(steps, list) else None
        raise EpisodeRejected(f"steps: expected {EPISODE_STEPS}, got {count}")
    if any(not isinstance(step, list) or len(step) != 2 for step in steps):
        raise EpisodeRejected("steps: every step needs exactly two seats")
    info = data.get("info")
    if not isinstance(info, dict) or str(info.get("EpisodeId")) != episode_id:
        raise EpisodeRejected("info.EpisodeId differs from the member name")
    seed = info.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EpisodeRejected("info.seed is not an integer")
    statuses = [seat.get("status") for seat in steps[-1]]
    if statuses != ["DONE", "DONE"]:
        raise EpisodeRejected(f"final statuses {statuses}")
    config = data.get("configuration")
    if not isinstance(config, dict):
        raise EpisodeRejected("configuration missing")
    missing = CONFIG_KEYS - set(config)
    extra = set(config) - CONFIG_KEYS - CONFIG_EXTRA_KEYS
    if missing or extra:
        raise EpisodeRejected(
            f"configuration keys: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    if config["episodeSteps"] != EPISODE_STEPS:
        raise EpisodeRejected(f"episodeSteps {config['episodeSteps']}")


def final_banks(data: dict[str, Any]) -> tuple[float, float]:
    farms = data["steps"][-1][0]["observation"]["farms"]
    banks = (float(farms[0]["money"]), float(farms[1]["money"]))
    if not all(math.isfinite(bank) for bank in banks):
        raise EpisodeRejected(f"final banks not finite: {banks}")
    return banks


def turn_header(data: dict[str, Any], t: int, *, seed: int = 0) -> dict[str, Any]:
    """Explicit-state header for ``steps[t]``; seat 1's shared keys from seat 0.

    Encoding always uses the placeholder seed 0, so the recorded seed never
    reaches observation buffers.
    """
    obs0 = data["steps"][t][0]["observation"]
    obs1 = data["steps"][t][1]["observation"]
    return {
        "format": "kaggriculture-re-parity-v1",
        "seed": seed,
        "configuration": data["configuration"],
        "shop_schedule": [],
        "rng_schedule": [],
        "initial": {
            "public": {key: obs0[key] for key in PUBLIC_KEYS},
            "privates": [obs0["private"], obs1["private"]],
        },
        "terminal_banks": [],
        "transitions": 0,
    }


def _falsy_kind(value: object) -> str | None:
    """The reference's ``raw.get(k) or []``: which falsy value became ``[]``."""
    if value is _ABSENT:
        return "absent"
    if value is None:
        return "null"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return "zero"
    if isinstance(value, str) and value == "":
        return "empty_string"
    if isinstance(value, dict) and not value:
        return "empty_object"
    return None


def normalize_action(raw: object) -> tuple[dict[str, Any], list[str]]:
    """Reference envelope normalization; raises ``ValueError`` on a bad envelope."""
    if not isinstance(raw, dict) or not set(raw) <= ACTION_KEYS:
        raise ValueError("action envelope")
    action: dict[str, Any] = {"farmer": raw.get("farmer")}
    normalized: list[str] = []
    for key in ("hands", "market"):
        value = raw.get(key, _ABSENT)
        kind = _falsy_kind(value)
        if kind is not None:
            normalized.append(f"normalized_{key}_{kind}")
            value = []
        elif not isinstance(value, list):
            raise ValueError("action envelope")
        action[key] = value
    return action, normalized


def pass_action(actors: int) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [["PASS"]] * (actors - 1), "market": []}


def rejection_reason(error: Exception) -> str:
    return str(error).split(":", 1)[0].strip()[:80] or type(error).__name__


def resident_bytes_per_turn() -> int:
    """Host bytes one turn row occupies in ``load_bc_dataset``'s compact store."""
    sizes = {torch.int64: 4, torch.float32: 4, torch.float64: 8, torch.bool: 1}
    per_seat = sum(
        sizes[dtype] * math.prod(shape) for dtype, shape, _, _ in kt._SCHEMA.values()
    )
    labels = kt.MAX_FRAMES + 2 * kt.MAX_FRAMES * kt.ACTION_SLOTS + 4 + 1
    return kt.PLAYERS * (per_seat + labels) + 4 + 8


def stride_for_budget(episodes: int, budget_gib: float) -> int:
    rows = episodes * PAIRED_TURNS
    budget_rows = int(budget_gib * 2**30) // resident_bytes_per_turn()
    if budget_rows < 1:
        raise ValueError(f"resident budget {budget_gib} GiB holds no turn row")
    return max(1, math.ceil(rows / budget_rows))


# --- archive listing --------------------------------------------------------------


def list_tasks(
    archives: Path,
    *,
    month: str,
    days: Iterable[int],
    validation_fraction: float,
    turn_stride: int,
) -> list[EpisodeTask]:
    tasks: list[EpisodeTask] = []
    seen: set[str] = set()
    for day in days:
        path = archives / ARCHIVE_PATTERN.format(month=month, day=day)
        if not path.is_file():
            raise FileNotFoundError(f"day archive missing: {path}")
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.endswith(".json"))
        for member in names:
            episode_id = member[: -len(".json")]
            if "/" in member or not _EPISODE_ID.match(episode_id):
                raise ValueError(f"{path}: unexpected member {member!r}")
            if episode_id in seen:
                raise ValueError(f"episode {episode_id} appears in two archives")
            seen.add(episode_id)
            tasks.append(
                EpisodeTask(
                    archive=str(path),
                    member=member,
                    day=f"{month}-{day:02d}",
                    episode_id=episode_id,
                    split=validation_split(episode_id, validation_fraction),
                    turn_stride=turn_stride,
                )
            )
    return tasks


# --- one episode ------------------------------------------------------------------


def _read_member(task: EpisodeTask) -> tuple[dict[str, Any], str, int]:
    with zipfile.ZipFile(task.archive) as archive:
        payload = archive.read(task.member)
    data = json.loads(payload)
    return data, hashlib.sha256(payload).hexdigest(), len(payload)


_NUMPY_DTYPES: dict[torch.dtype, type[np.generic]] = {
    torch.int64: np.int64,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: np.bool_,
}


def _allocate(rows: int) -> dict[str, npt.NDArray[Any]]:
    arrays = {
        name: np.zeros((rows, kt.PLAYERS, *shape), dtype=_NUMPY_DTYPES[dtype])
        for name, (dtype, shape, _, _) in kt._SCHEMA.items()
    }
    arrays["can_act"] = np.zeros((rows, kt.PLAYERS, kt.MAX_FRAMES), dtype=np.bool_)
    return arrays


def encode_observations(
    data: dict[str, Any], turns: Sequence[int]
) -> dict[str, npt.NDArray[Any]]:
    """Native Task 1.3 encoding of ``steps[t]`` for every kept turn, both seats."""
    arrays = _allocate(len(turns))
    for start in range(0, len(turns), ENCODE_BATCH):
        chunk = turns[start : start + ENCODE_BATCH]
        headers = json.dumps([turn_header(data, t) for t in chunk])
        view = {name: a[start : start + len(chunk)] for name, a in arrays.items()}
        try:
            rs.encode_kaggriculture_headers_into(headers, **view)
        except ValueError as error:
            raise EpisodeRejected(
                f"encoder: turns {chunk[0]}..{chunk[-1]}: {error}"
            ) from error
    return arrays


def admit_turns(
    data: dict[str, Any],
    turns: Sequence[int],
    arrays: dict[str, npt.NDArray[Any]],
    policy: tuple[bool, bool],
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    Counter[str],
    Counter[str],
]:
    """Strict per-turn admission of the policy seats' step+1 actions."""
    order_limit_config = int(data["configuration"]["maxMarketOrdersPerTurn"])
    rows = len(turns)
    admitted = np.zeros(rows, dtype=np.bool_)
    tokens = np.zeros((rows, 2, kt.MAX_FRAMES, kt.ACTION_SLOTS), dtype=np.int64)
    lengths = np.zeros((rows, 2), dtype=np.int64)
    rejections: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    for row, t in enumerate(turns):
        farms = data["steps"][t][0]["observation"]["farms"]
        ok = True
        for seat in (0, 1):
            actors = int(arrays["actor_mask"][row, seat, :HIRE_LIMIT].sum())
            order_limit = int(arrays["order_limits"][row, seat])
            if actors != 1 + len(farms[seat]["hands"]):
                raise EpisodeRejected(f"turn {t} seat {seat}: actor count mismatch")
            if order_limit != order_limit_config:
                raise EpisodeRejected(f"turn {t} seat {seat}: order limit mismatch")
            out = tokens[row, seat]
            if not policy[seat]:
                try:
                    lengths[row, seat] = rs.kaggriculture_encode(
                        json.dumps(pass_action(actors)),
                        actors,
                        order_limit,
                        HIRE_LIMIT,
                        out,
                    )
                except ValueError as error:
                    rejections[f"placeholder {rejection_reason(error)}"] += 1
                    ok = False
                continue
            # Step+1 pairing: the action answering steps[t] is recorded at steps[t + 1].
            raw = data["steps"][t + 1][seat]["action"]
            try:
                action, kinds = normalize_action(raw)
                length = rs.kaggriculture_encode(
                    json.dumps(action), actors, order_limit, HIRE_LIMIT, out
                )
                decoded = json.loads(
                    rs.kaggriculture_decode(
                        out, length, actors, order_limit, HIRE_LIMIT
                    )
                )
                if decoded != action:
                    raise ValueError("decode round trip")
            except ValueError as error:
                rejections[rejection_reason(error)] += 1
                out[:] = 0
                ok = False
                continue
            normalizations.update(kinds)
            lengths[row, seat] = length
        admitted[row] = ok
    return admitted, tokens, lengths, rejections, normalizations


def process_episode(task: EpisodeTask, out_dir: Path) -> dict[str, Any]:
    """Prepare one episode and write its shard and record; return the record."""
    record: dict[str, Any] = {
        "episode_id": task.episode_id,
        "day": task.day,
        "member": task.member,
        "split": task.split,
    }
    try:
        data, sha256, size = _read_member(task)
        record |= {"source_sha256": sha256, "source_bytes": size}
        check_envelope(data, task.episode_id)
        banks = final_banks(data)
        policy = winner_seats(banks)
        turns = kept_turns(task.episode_id, task.turn_stride)
        arrays = encode_observations(data, turns)
        admitted, tokens, lengths, rejections, normalizations = admit_turns(
            data, turns, arrays, policy
        )
    except EpisodeRejected as error:
        record["rejected"] = str(error)
        _write_record(out_dir, task, record)
        return record
    record |= {
        "info_seed": int(data["info"]["seed"]),
        "terminal_banks": list(banks),
        "policy_seats": [seat for seat in (0, 1) if policy[seat]],
        "paired_turns": PAIRED_TURNS,
        "kept_turns": len(turns),
        "admitted": int(admitted.sum()),
        "rejections": dict(sorted(rejections.items())),
        "normalizations": dict(sorted(normalizations.items())),
    }
    del data
    if record["admitted"] == 0:
        record["rejected"] = "no admitted turns"
        _write_record(out_dir, task, record)
        return record
    keep = np.flatnonzero(admitted)
    tensors = {name: torch.from_numpy(a[keep]) for name, a in arrays.items()}
    can_act = tensors.pop("can_act")
    episode = bc_data.BCEpisode(
        episode_id=task.episode_id,
        split=task.split,
        day=task.day,
        obs=kt.KaggricultureObsBatch(
            **tensors, action_mask=kt.KaggricultureActionMask(can_act=can_act)
        ),
        actions=kt.KaggricultureActions(
            tokens=torch.from_numpy(tokens[keep]),
            lengths=torch.from_numpy(lengths[keep]),
        ),
        policy_seat=torch.tensor([policy] * keep.size, dtype=torch.bool),
        turn=torch.tensor([turns[i] for i in keep], dtype=torch.int64),
        terminal_banks=banks,
    )
    episode.obs.check_contract()
    shard = out_dir / bc_data.shard_path(task.split, task.day, task.episode_id)
    shard.unlink(missing_ok=True)  # a crash before the record left a partial shard
    written = bc_data.write_bc_episode(out_dir, episode)
    record |= {
        "shard_path": written.shard_path,
        "shard_sha256": written.shard_sha256,
        "shard_bytes": written.shard_bytes,
    }
    _write_record(out_dir, task, record)
    return record


def _record_path(out_dir: Path, task: EpisodeTask) -> Path:
    return out_dir / RECORD_DIR / f"{task.day}-{task.episode_id}.json"


def _write_record(out_dir: Path, task: EpisodeTask, record: dict[str, Any]) -> None:
    path = _record_path(out_dir, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _run_task(args: tuple[EpisodeTask, str]) -> dict[str, Any]:
    task, out_dir = args
    return process_episode(task, Path(out_dir))


# --- pairing sanity with Kaggle's own engine --------------------------------------


def pairing_check(data: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Step Kaggle's interpreter from ``steps[t]`` with ``steps[t + 1].action``.

    Returns the number of the 719 transitions whose public state and both
    privates equal ``steps[t + 1]`` (value equality), plus the mismatches.
    """
    # kaggle_environments ships no type information; bind it through importlib.
    interpreter = importlib.import_module(
        "kaggle_environments.envs.kaggriculture.kaggriculture"
    ).interpreter
    utils = importlib.import_module("kaggle_environments.utils")
    Struct, structify = utils.Struct, utils.structify

    env = Struct(
        configuration=structify(copy.deepcopy(data["configuration"])),
        info={"seed": data["info"]["seed"]},
        done=False,
    )
    steps = data["steps"]
    matches = 0
    mismatches: list[dict[str, Any]] = []
    for t in range(PAIRED_TURNS):
        # structify copies every container. Kaggle shares one public
        # farms/market/town object across seats, so both seats get the same one.
        public = structify({k: steps[t][0]["observation"][k] for k in PUBLIC_KEYS})
        state = []
        for seat in (0, 1):
            own = structify(
                {
                    k: v
                    for k, v in steps[t][seat]["observation"].items()
                    if k not in PUBLIC_KEYS
                }
            )
            state.append(
                Struct(
                    observation=Struct(**own, **public),
                    action=structify(steps[t + 1][seat]["action"]),
                    status="ACTIVE",
                    reward=0,
                )
            )
        interpreter(state, env)
        predicted: dict[str, Any] = _plain(
            {
                "public": {k: state[0].observation[k] for k in PAIRING_PUBLIC_KEYS},
                "privates": [state[s].observation["private"] for s in (0, 1)],
            }
        )
        nxt = steps[t + 1][0]["observation"]
        expected: dict[str, Any] = {
            "public": {k: nxt[k] for k in PAIRING_PUBLIC_KEYS},
            "privates": [steps[t + 1][s]["observation"]["private"] for s in (0, 1)],
        }
        if predicted == expected:
            matches += 1
        else:
            differing = [
                k
                for k in PAIRING_PUBLIC_KEYS
                if predicted["public"][k] != expected["public"][k]
            ] + [
                f"private{s}"
                for s in (0, 1)
                if predicted["privates"][s] != expected["privates"][s]
            ]
            mismatches.append({"turn": t, "differs": differing})
    return matches, mismatches


def _plain(value: object) -> Any:
    return json.loads(json.dumps(value))


def _pairing_task(task: EpisodeTask) -> dict[str, Any]:
    data, _, _ = _read_member(task)
    try:
        check_envelope(data, task.episode_id)
    except EpisodeRejected as error:
        return {
            "episode_id": task.episode_id,
            "day": task.day,
            "matches": 0,
            "total": 0,
            "rejected": str(error),
        }
    matches, mismatches = pairing_check(data)
    return {
        "episode_id": task.episode_id,
        "day": task.day,
        "matches": matches,
        "total": PAIRED_TURNS,
        "mismatches": mismatches[:20],
    }


def pairing_sample(
    tasks: Sequence[EpisodeTask], count: int, mapper: Any
) -> dict[str, Any]:
    step = max(1, len(tasks) // count)
    chosen = list(tasks[::step][:count])
    results = list(mapper(_pairing_task, chosen))
    matches = sum(r["matches"] for r in results)
    total = sum(r["total"] for r in results)
    return {
        "engine": _engine_identity(),
        "episodes": results,
        "matches": matches,
        "total": total,
        "fraction": matches / total if total else 0.0,
    }


def _engine_identity() -> dict[str, str]:
    from importlib import metadata

    kaggriculture = importlib.import_module(
        "kaggle_environments.envs.kaggriculture.kaggriculture"
    )

    source = Path(str(kaggriculture.__file__))
    return {
        "kaggle_environments": metadata.version("kaggle-environments"),
        "interpreter_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


# --- manifest ---------------------------------------------------------------------


def build_manifest(
    out_dir: Path, records: Sequence[dict[str, Any]], custody: dict[str, Any]
) -> bc_data.BCManifest:
    kept = sorted(
        (r for r in records if "rejected" not in r),
        key=lambda r: (r["day"], r["episode_id"]),
    )
    rejected = sorted(
        (r for r in records if "rejected" in r),
        key=lambda r: (r["day"], r["episode_id"]),
    )
    totals: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        mine = [r for r in kept if r["split"] == split]
        totals[split] = {
            "episodes": len(mine),
            "admitted_turns": sum(r["admitted"] for r in mine),
            "policy_seat_rows": sum(
                r["admitted"] * len(r["policy_seats"]) for r in mine
            ),
            "draw_episodes": sum(len(r["policy_seats"]) == 2 for r in mine),
        }
    turn_rejections: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    for r in kept:
        turn_rejections.update(r["rejections"])
        normalizations.update(r["normalizations"])
    episode_rejections = Counter(
        rejection_reason(Exception(r["rejected"])) for r in rejected
    )
    episodes = [bc_data.BCManifestEpisode.model_validate(r) for r in kept]
    return bc_data.write_bc_manifest(
        out_dir,
        episodes,
        **custody,
        totals=totals,
        turn_rejections=dict(sorted(turn_rejections.items())),
        normalizations=dict(sorted(normalizations.items())),
        episode_rejections=dict(sorted(episode_rejections.items())),
        rejected_episodes=rejected,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(repo: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"refusing to prepare from a dirty checkout:\n{dirty}")
    extension = Path(str(rs.__file__))
    return {
        "git_head": head,
        "uv_lock_sha256": _sha256(repo / "uv.lock"),
        "cargo_lock_sha256": _sha256(repo / "Cargo.lock"),
        "prepare_sha256": _sha256(Path(__file__)),
        "owl_rs_extension": extension.name,
        "owl_rs_sha256": _sha256(extension),
    }


# --- CLI --------------------------------------------------------------------------


@contextlib.contextmanager
def _mapper(workers: int) -> Iterator[Callable[..., Iterable[Any]]]:
    """In-process ``map`` for one worker, else a bounded spawn pool."""
    if workers == 1:
        yield map
        return
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, maxtasksperchild=64) as pool:
        yield pool.imap_unordered


def _parse_days(text: str) -> list[int]:
    if "-" in text:
        first, last = (int(part) for part in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(part) for part in text.split(",")]


def _bind_identity(out_dir: Path, identity: dict[str, Any]) -> None:
    """Record the preparation identity, or require an exact match on resume."""
    records = out_dir / RECORD_DIR
    path = records / IDENTITY_NAME
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != identity:
            changed = sorted(
                key
                for key in stored.keys() | identity.keys()
                if stored.get(key) != identity.get(key)
            )
            raise RuntimeError(
                f"{path} differs from this run in {changed}; "
                "prepare into a fresh out_dir"
            )
        return
    if records.is_dir() and any(records.glob("*.json")):
        raise RuntimeError(
            f"{records} has records but no {IDENTITY_NAME}; use a fresh out_dir"
        )
    records.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archives", type=Path, help="directory of day archive ZIPs")
    parser.add_argument(
        "out_dir", type=Path, help="dataset root (outside the checkout)"
    )
    parser.add_argument("--month", default="2026-09")
    parser.add_argument("--days", default="22-28", help="e.g. 22-28 or 22,23")
    parser.add_argument(
        "--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2))
    )
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    stride = parser.add_mutually_exclusive_group()
    stride.add_argument("--turn-stride", type=int, default=None)
    stride.add_argument(
        "--resident-budget-gib",
        type=float,
        default=None,
        help="derive the turn stride so all turn rows fit this host-memory budget",
    )
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--pairing-sample", type=int, default=0)
    parser.add_argument("--pairing-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    repo = Path(__file__).resolve().parents[1]
    out_dir: Path = args.out_dir.resolve()
    if out_dir.is_relative_to(repo):
        raise ValueError("out_dir must lie outside the checkout")
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir}/manifest.json exists; dataset is complete")
    run = _source_identity(repo)
    days = _parse_days(args.days)
    tasks = list_tasks(
        args.archives,
        month=args.month,
        days=days,
        validation_fraction=args.validation_fraction,
        turn_stride=1,
    )
    if args.limit_episodes is not None:
        tasks = tasks[: args.limit_episodes]
    if not tasks:
        raise ValueError("no episodes found")
    turn_stride = args.turn_stride or 1
    if args.resident_budget_gib is not None:
        turn_stride = stride_for_budget(len(tasks), args.resident_budget_gib)
    tasks = [
        EpisodeTask(t.archive, t.member, t.day, t.episode_id, t.split, turn_stride)
        for t in tasks
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_sha256 = {
        Path(archive).name: _sha256(Path(archive))
        for archive in sorted({t.archive for t in tasks})
    }
    _bind_identity(
        out_dir,
        {
            "run": run,
            "month": args.month,
            "days": days,
            "validation_fraction": args.validation_fraction,
            "turn_stride": turn_stride,
            "hire_limit": HIRE_LIMIT,
            "label_pairing": LABEL_PAIRING,
            "archive_sha256": archive_sha256,
        },
    )
    print(
        f"episodes={len(tasks)} days={days} turn_stride={turn_stride} "
        f"bytes_per_turn_row={resident_bytes_per_turn()} workers={args.workers}",
        flush=True,
    )
    with _mapper(args.workers) as mapper:
        if args.pairing_sample > 0:
            pairing = pairing_sample(tasks, args.pairing_sample, mapper)
            (out_dir / PAIRING_NAME).write_text(json.dumps(pairing, indent=2) + "\n")
            print(
                f"pairing: {pairing['matches']}/{pairing['total']} transitions match "
                f"({pairing['fraction']:.4f}) over {len(pairing['episodes'])} episodes",
                flush=True,
            )
            if args.pairing_only:
                return 0
        records: list[dict[str, Any]] = []
        todo: list[EpisodeTask] = []
        for task in tasks:
            path = _record_path(out_dir, task)
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                todo.append(task)
        print(f"resuming: {len(records)} done, {len(todo)} to do", flush=True)
        for index, record in enumerate(
            mapper(_run_task, [(t, str(out_dir)) for t in todo]), 1
        ):
            records.append(record)
            if index % 50 == 0 or index == len(todo):
                print(f"episodes {index}/{len(todo)}", flush=True)
    pairing_path = out_dir / PAIRING_NAME
    manifest = build_manifest(
        out_dir,
        records,
        {
            "run": run
            | {
                "command": [sys.executable, *sys.argv],
                "workers": args.workers,
                "hire_limit": HIRE_LIMIT,
                "turn_stride": turn_stride,
                "bytes_per_turn_row": resident_bytes_per_turn(),
                "validation_fraction": args.validation_fraction,
                "label_pairing": LABEL_PAIRING,
                "policy_seats": "winner by final bank; both on a draw",
            },
            "source": {
                "archives": str(args.archives),
                "month": args.month,
                "days": days,
                "archive_sha256": archive_sha256,
                "archive_bytes": {
                    Path(t.archive).name: Path(t.archive).stat().st_size for t in tasks
                },
                "episodes_listed": len(tasks),
            },
            "pairing": (
                json.loads(pairing_path.read_text()) if pairing_path.is_file() else None
            ),
        },
    )
    extra = manifest.model_extra or {}
    print(json.dumps(extra.get("totals"), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
