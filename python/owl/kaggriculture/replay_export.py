"""Opt-in host custody for native Kaggriculture evaluation replays.

This recorder does not call an environment or consume a seed. The caller supplies
its consumed seed and copies diagnostic state before the environment auto-resets.
Nothing in this module feeds model observations. Production wiring awaits Task
1.4's native environment and the Kaggriculture evaluation branch.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from owl import rs

FRAMEWORK_VERSION = "1.32.7"
FRAMEWORK_HASHES = {
    "core.py": "0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e",
    "utils.py": "537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b",
    "envs/kaggriculture/kaggriculture.py": (
        "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
    ),
    "envs/kaggriculture/kaggriculture.json": (
        "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867"
    ),
}


class ReplayConfig(Protocol):
    @property
    def eval_replay_games(self) -> int: ...


def _json(value: Any) -> str:
    # Dict order and arbitrary-width Python ints cross the extension as text.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def load_pinned_framework() -> dict[str, Any]:
    """Load the installed specification through the hash-pinned framework.

    ``state={}`` bypasses Environment.reset; the no-op interpreter is never run.
    Environment itself expands the specification with its own schemas, including
    shared ``step``. No hand-maintained copy of the framework schema is used.
    Every call rechecks the files before importing or deriving the specification.
    """
    try:
        distribution = importlib.metadata.distribution("kaggle-environments")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("pinned kaggle-environments is unavailable") from error
    if distribution.version != FRAMEWORK_VERSION:
        raise ValueError(
            f"kaggle-environments must be {FRAMEWORK_VERSION}, "
            f"got {distribution.version}"
        )
    hashes: dict[str, str] = {}
    verified_bytes: dict[str, bytes] = {}
    for name, expected in FRAMEWORK_HASHES.items():
        path = Path(str(distribution.locate_file(f"kaggle_environments/{name}")))
        try:
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
        except OSError as error:
            raise ValueError(f"pinned framework file is unavailable: {path}") from error
        if actual != expected:
            raise ValueError(
                f"{name} SHA-256 differs: expected {expected}, got {actual}"
            )
        hashes[name] = actual
        verified_bytes[name] = payload

    core = importlib.import_module("kaggle_environments.core")

    raw_specification = json.loads(
        verified_bytes["envs/kaggriculture/kaggriculture.json"]
    )
    environment = core.Environment(
        specification=raw_specification,
        interpreter=lambda state, _environment: state,
        renderer=lambda *_args: "",
        html_renderer=lambda *_args: "",
        state={},
    )
    episode = environment.toJSON()
    return {
        "specification": episode["specification"],
        "default_configuration": episode["configuration"],
        "envelope": {
            key: episode[key]
            for key in (
                "id",
                "name",
                "title",
                "description",
                "version",
                "module_version",
                "schema_version",
            )
        },
        "hashes": hashes,
    }


def _schema_difference(actual: Any, expected: Any, pointer: str) -> str | None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return pointer
        for key, value in expected.items():
            child = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in actual:
                return child
            difference = _schema_difference(actual[key], value, child)
            if difference is not None:
                return difference
        if list(actual) != list(expected):
            return pointer
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return pointer
        if len(actual) != len(expected):
            return pointer
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            difference = _schema_difference(left, right, f"{pointer}/{index}")
            if difference is not None:
                return difference
        return None
    return None if type(actual) is type(expected) and actual == expected else pointer


def validate_native_replay_input(json_text: str, episode: bool) -> None:
    """Guard every PyO3 entry against the installed, hash-verified framework.

    Pure Rust replay accepts its caller-supplied specification as trusted. The
    Python-facing boundary admits only the processed pinned schema and a fully
    resolved configuration; framework time budgets may differ only within that
    schema. This function validates metadata and never transforms game state.
    """
    framework = load_pinned_framework()
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError) as error:
        raise ValueError(f"/ (transition None): invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("/ (transition None): expected replay object")
    if "specification" not in data:
        raise ValueError("/specification (transition None): required pinned schema")
    difference = _schema_difference(
        data["specification"], framework["specification"], "/specification"
    )
    if difference is not None:
        raise ValueError(f"{difference} (transition None): differs from pinned schema")
    envelope = data if episode else data.get("envelope")
    envelope_pointer = "" if episode else "/envelope"
    if not isinstance(envelope, dict):
        raise ValueError(f"{envelope_pointer} (transition None): expected envelope")
    if not isinstance(envelope.get("id"), str):
        raise ValueError(f"{envelope_pointer}/id (transition None): expected string")
    for key, expected in framework["envelope"].items():
        if key == "id":
            continue
        if (
            key not in envelope
            or type(envelope[key]) is not type(expected)
            or envelope[key] != expected
        ):
            raise ValueError(
                f"{envelope_pointer}/{key} (transition None): "
                "differs from pinned envelope"
            )
    configuration = data.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("/configuration (transition None): expected resolved object")
    expected_keys = set(framework["specification"]["configuration"])
    actual_keys = set(configuration)
    if actual_keys != expected_keys:
        key = sorted(actual_keys ^ expected_keys)[0]
        raise ValueError(
            f"/configuration/{key} (transition None): missing or unknown resolved field"
        )
    utils = importlib.import_module("kaggle_environments.utils")
    for key, field_schema in framework["specification"]["configuration"].items():
        error, _processed = utils.process_schema(
            field_schema, deepcopy(configuration[key])
        )
        if error:
            raise ValueError(f"/configuration/{key} (transition None): {error}")
    if configuration["seed"] is not None:
        raise ValueError(
            "/configuration/seed (transition None): resolved seed must be null"
        )


def select_replay_games(
    evaluation_identity: str,
    total_games: int,
    count: int,
    *,
    seat_assignments: Sequence[int] | None = None,
) -> frozenset[int]:
    """Select stable ordinals, stratified by the supplied model-seat schedule.

    The caller supplies the actual evaluation schedule when both assignments
    occur; ordinal parity is never treated as an implicit seat assignment.
    Count zero disables recording. A positive count below the number of seat
    assignments cannot cover both and is rejected explicitly.
    """
    if not evaluation_identity:
        raise ValueError("evaluation_identity must be non-empty")
    if type(total_games) is not int or total_games < 0:
        raise ValueError("total_games must be a non-negative integer")
    if type(count) is not int or not 0 <= count <= total_games:
        raise ValueError("count must be a non-negative integer <= total_games")
    groups: dict[int, list[int]] = {}
    if seat_assignments is not None:
        if len(seat_assignments) != total_games:
            raise ValueError("seat_assignments length must equal total_games")
        for ordinal, seat in enumerate(seat_assignments):
            if type(seat) is not int or seat not in (0, 1):
                raise ValueError("seat_assignments entries must be 0 or 1")
            groups.setdefault(seat, []).append(ordinal)
    if count == 0:
        return frozenset()
    if count < len(groups):
        raise ValueError("count cannot cover both model-seat assignments")
    seed = int.from_bytes(hashlib.sha256(evaluation_identity.encode("utf-8")).digest())
    rng = random.Random(seed)
    selected = {rng.choice(groups[seat]) for seat in sorted(groups)}
    selected.update(
        rng.sample(
            [i for i in range(total_games) if i not in selected], count - len(selected)
        )
    )
    return frozenset(selected)


@dataclass
class _ActiveReplay:
    game_ordinal: int
    env_index: int
    seed_header: dict[str, Any]
    seat_assignment: int
    checkpoint_hashes: dict[str, str]
    versions: dict[str, Any]
    transitions: list[dict[str, Any]] = field(default_factory=list)
    captured: dict[str, Any] = field(default_factory=dict)


class ReplayRecorder:
    """Record selected games and materialize them only after completion/truncation.

    ``config`` is the caller's ``cfg.rl``; its ``eval_replay_games`` is the default
    count (eight in the two supported rank configs). Failed export or evidence
    comparison writes an error custody record and no successful episode.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        evaluation_identity: str,
        total_games: int,
        config: ReplayConfig,
        count: int | None = None,
        seat_assignments: Sequence[int] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.evaluation_identity = evaluation_identity
        self.selected_games = select_replay_games(
            evaluation_identity,
            total_games,
            config.eval_replay_games if count is None else count,
            seat_assignments=seat_assignments,
        )
        self._seat_assignments = (
            tuple(seat_assignments) if seat_assignments is not None else None
        )
        self._active: dict[int, _ActiveReplay] = {}
        self._finished: set[int] = set()
        self._framework = load_pinned_framework() if self.selected_games else None

    def start_game(
        self,
        game_ordinal: int,
        env_index: int,
        resolved_seed: int,
        configuration: Mapping[str, Any],
        seat_assignment: int,
        checkpoint_hashes: Mapping[str, str],
        versions: Mapping[str, Any],
        *,
        captured_initial: Mapping[str, Any] | None = None,
    ) -> None:
        if game_ordinal not in self.selected_games:
            return
        if game_ordinal in self._active or game_ordinal in self._finished:
            raise ValueError(f"game {game_ordinal} was already started")
        if type(resolved_seed) is not int or resolved_seed < 0:
            raise ValueError("resolved_seed must be an exact non-negative integer")
        if type(env_index) is not int or env_index < 0:
            raise ValueError("env_index must be a non-negative integer")
        if type(seat_assignment) is not int or seat_assignment not in (0, 1):
            raise ValueError("seat_assignment must be 0 or 1")
        if (
            self._seat_assignments is not None
            and self._seat_assignments[game_ordinal] != seat_assignment
        ):
            raise ValueError("seat_assignment differs from the selection schedule")
        for key in ("source", "engine", "schema_version"):
            if key not in versions:
                raise ValueError(f"versions lacks {key}")
        for key in ("source", "engine"):
            if not isinstance(versions[key], str) or not versions[key]:
                raise ValueError(f"versions.{key} must identify the source")
        for key in ("candidate", "incumbent"):
            digest = checkpoint_hashes.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in digest)
            ):
                raise ValueError(f"checkpoint_hashes.{key} must be a SHA-256")
        if self._framework is None:
            raise ValueError("selected replay has no pinned framework")
        provenance = {
            key: versions[key] for key in ("source", "engine", "schema_version")
        }
        provenance["target_framework_version"] = FRAMEWORK_VERSION
        envelope = deepcopy(self._framework["envelope"])
        envelope["id"] = f"{self.evaluation_identity}:game:{game_ordinal}"
        resolved_configuration = deepcopy(self._framework["default_configuration"])
        resolved_configuration.update(deepcopy(dict(configuration)))
        resolved_configuration["seed"] = None
        header = {
            "seed": resolved_seed,
            "configuration": resolved_configuration,
            "provenance": deepcopy(provenance),
            "specification": deepcopy(self._framework["specification"]),
            "envelope": envelope,
        }
        captured = (
            {}
            if captured_initial is None
            else {"initial": deepcopy(dict(captured_initial))}
        )
        self._active[game_ordinal] = _ActiveReplay(
            game_ordinal,
            env_index,
            header,
            seat_assignment,
            deepcopy(dict(checkpoint_hashes)),
            deepcopy(dict(versions)),
            captured=captured,
        )

    def record_transition(
        self,
        game_ordinal: int,
        actions: Sequence[Mapping[str, Any]],
        *,
        tokens: Sequence[Mapping[str, Any]] | None = None,
        captured: Mapping[str, Any] | None = None,
    ) -> None:
        if game_ordinal not in self.selected_games:
            return
        replay = self._require_active(game_ordinal)
        if len(actions) != 2:
            message = "transition requires exactly two seat actions"
            self.fail_game(game_ordinal, error=message)
            raise ValueError(message)
        transition: dict[str, Any] = {"actions": deepcopy(list(actions))}
        if tokens is not None:
            if len(tokens) != 2:
                message = "tokens requires exactly two seat rows"
                self.fail_game(game_ordinal, error=message)
                raise ValueError(message)
            transition["tokens"] = deepcopy(list(tokens))
        if captured is not None:
            evidence = deepcopy(dict(captured))
            try:
                if "public" in evidence:
                    banks = [farm["money"] for farm in evidence["public"]["farms"]]
                else:
                    banks = evidence["banks"]
                if not isinstance(banks, list) or len(banks) != 2:
                    raise TypeError("expected two seat banks")
            except (KeyError, TypeError) as error:
                message = f"captured evidence needs a snapshot or two banks: {error}"
                self.fail_game(game_ordinal, error=message)
                raise ValueError(message) from error
            if "public" in evidence:
                replay.captured.setdefault("snapshots", []).append(
                    {"transition": len(replay.transitions), "snapshot": evidence}
                )
            replay.captured.setdefault("banks", []).append(banks)
        replay.transitions.append(transition)

    def finish_game(self, game_ordinal: int, terminal: Mapping[str, Any]) -> None:
        if game_ordinal not in self.selected_games:
            return
        replay = self._require_active(game_ordinal)
        replay.captured["terminal"] = deepcopy(dict(terminal))
        self._write(replay, status="complete")

    def truncate_game(self, game_ordinal: int, *, reason: str) -> None:
        if game_ordinal not in self.selected_games:
            return
        self._write(
            self._require_active(game_ordinal), status="truncated", reason=reason
        )

    def fail_game(self, game_ordinal: int, *, error: str) -> None:
        """Record a caller-observed native failure without pretending it committed."""
        if game_ordinal not in self.selected_games:
            return
        self._write(self._require_active(game_ordinal), status="error", reason=error)

    def _require_active(self, game_ordinal: int) -> _ActiveReplay:
        if game_ordinal not in self._active:
            raise ValueError(f"selected game {game_ordinal} is not active")
        return self._active[game_ordinal]

    def _write(
        self, replay: _ActiveReplay, *, status: str, reason: str | None = None
    ) -> None:
        tape = {"complete": status == "complete", "transitions": replay.transitions}
        sidecar: dict[str, Any] = {
            "schema_version": 1,
            "evaluation_identity": self.evaluation_identity,
            "game_ordinal": replay.game_ordinal,
            "env_index": replay.env_index,
            "status": status,
            "complete": status == "complete",
            "seed_header": replay.seed_header,
            "action_tape": tape,
            "checkpoint_hashes": replay.checkpoint_hashes,
            "seat_assignment": replay.seat_assignment,
            "versions": replay.versions,
            "framework_hashes": self._framework["hashes"] if self._framework else {},
            "captured": replay.captured,
            "producer": "v3 native replay; not executed by the Python framework",
        }
        if reason is not None:
            sidecar["error" if status == "error" else "reason"] = reason
        episode_bytes: bytes | None = None
        if status != "error":
            try:
                episode_json = rs.export_kaggriculture_episode(
                    _json(replay.seed_header), _json(tape)
                )
                report = json.loads(
                    rs.verify_kaggriculture_episode(
                        episode_json, _json(replay.captured)
                    )
                )
                if report["ok"] is not True:
                    raise ValueError("native verification did not report success")
                sidecar["verification"] = {
                    key: value
                    for key, value in report.items()
                    if key != "canonical_json"
                }
                episode_bytes = episode_json.encode("utf-8")
                sidecar["episode_sha256"] = hashlib.sha256(episode_bytes).hexdigest()
            except (ValueError, RuntimeError) as error:
                sidecar.update(status="error", complete=False, error=str(error))
                tape["complete"] = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"game_{replay.game_ordinal:06d}"
        episode_path = self.output_dir / f"{stem}.json"
        custody_path = self.output_dir / f"{stem}.custody.json"
        if episode_path.exists() or custody_path.exists():
            raise FileExistsError(f"replay custody already exists for {stem}")
        # Verify before publishing any successful episode. Exclusive creation
        # protects evidence if a caller accidentally reuses an output directory.
        with custody_path.open("x", encoding="utf-8") as stream:
            stream.write(_json(sidecar))
        if episode_bytes is not None:
            with episode_path.open("xb") as stream:
                stream.write(episode_bytes)
        del self._active[replay.game_ordinal]
        self._finished.add(replay.game_ordinal)
