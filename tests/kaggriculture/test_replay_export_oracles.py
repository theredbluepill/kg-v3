"""Independent Python evidence for Task 7.3's seed replay exporter.

The official fixtures are converted from recorded Python state, never from native
export. One tiny live framework game supplies an independent envelope oracle.
All fixtures run by default: this is bounded replay, not an evaluation panel.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import resource
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from owl import rs

ROOT = Path(__file__).resolve().parents[2]
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
FIXTURE_HASHES = {
    "95324500": "47cdfa489b7a80edf8ec1361f2f55cd033c75824d624cd7c9e9daaa3137affd7",
    "95901360": "e80653f445570a3778a3fb9026a66614b1ecaa2ea417cf8358d3f1850d725281",
    "95921764": "bc3e01cd12ff70fd2f78bfbe7129d5caca468a6324c124c25c9efebc86fbd3f2",
    "95990191": "4bf1a3b09c644719c8b36a619289844e52c3458d0e25dc0a0a1429bc6eaa0d1b",
}
SEED = 2**80 + 19
ENVELOPE_FIELDS = (
    "id",
    "name",
    "title",
    "description",
    "version",
    "module_version",
    "schema_version",
)
PUBLIC_FIELDS = ("step", "day", "hour", "farms", "market", "town")
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _framework() -> Any:
    """Verify bytes before importing the package or invoking its interpreter."""
    distribution = importlib.metadata.distribution("kaggle-environments")
    assert distribution.version == "1.32.7", "pinned framework version differs"
    package = Path(distribution.locate_file("kaggle_environments"))
    for relative, expected in FRAMEWORK_HASHES.items():
        path = package / relative
        assert path.is_file(), f"pinned framework file unavailable: {path}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"pinned framework hash differs: {path}: {actual}"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return importlib.import_module("kaggle_environments")


def _run_framework_game() -> dict[str, Any]:
    """Exactly one live game; nine transitions, three day rolls, under 2min/1GB."""
    framework = _framework()
    started = time.monotonic()
    env = framework.make(
        "kaggriculture",
        configuration={
            "episodeSteps": 10,
            "turnsPerDay": 3,
            "maxMarketOrdersPerTurn": 4,
            "townShopUnlockInterval": 1,
            "weedSpawnChance": 0.2,
            "seed": SEED,
        },
        debug=True,
    )
    for transition in range(9):
        action = copy.deepcopy(PASS)
        if transition == 0:
            action["market"] = [["BUY_PRODUCT", "WHEAT", 2]]
        elif transition == 1:
            action["farmer"] = ["PICKUP", "WHEAT", 0]
            action["market"] = [["SELL", "WHEAT", 0], []]
        elif transition == 2:
            action["farmer"] = ["PICKUP", "WHEAT"]
        env.step([action, copy.deepcopy(PASS)])
        assert time.monotonic() - started < 120, "live oracle exceeded two minutes"
    episode = env.toJSON()
    assert episode["info"]["seed"] == SEED
    assert episode["configuration"]["seed"] is None
    assert episode["statuses"] == ["DONE", "DONE"]
    assert episode["steps"][-1][0]["observation"]["day"] == 3
    assert len(episode["steps"][-1][0]["observation"]["town"]["unlocked_shops"]) == 3
    assert any(
        isinstance(tile, dict) and tile["kind"] == "WEED"
        for farm in episode["steps"][-1][0]["observation"]["farms"]
        for row in farm["tiles"]
        for tile in row
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    assert rss_bytes < 1_000_000_000, f"live oracle process exceeded 1 GB: {rss_bytes}"
    return {
        "episode": episode,
        "seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes,
    }


@pytest.fixture(scope="module")
def framework_episode() -> dict[str, Any]:
    # Isolate the one live game from prior tests' torch allocations and historic
    # process RSS. The child imports this helper without pytest's conftest files.
    command = (
        "import json; "
        "from tests.kaggriculture.test_replay_export_oracles "
        "import _run_framework_game; "
        "print(json.dumps(_run_framework_game(), allow_nan=False))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    result = json.loads(completed.stdout)
    print(
        f"framework oracle: seed={SEED}, 9 transitions, "
        f"{result['seconds']:.6f}s, peak RSS={result['peak_rss_bytes']} bytes"
    )
    return result["episode"]


def _header_tape(episode: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    header = {
        "seed": episode["info"]["seed"],
        "configuration": episode["configuration"],
        "provenance": {
            "source": "Task 7.3 independent Python oracle tests",
            "engine": FRAMEWORK_HASHES["envs/kaggriculture/kaggriculture.py"],
            "schema_version": 1,
            "target_framework_version": "1.32.7",
        },
        "specification": episode["specification"],
        "envelope": {field: episode[field] for field in ENVELOPE_FIELDS},
    }
    tape = {
        "complete": True,
        "transitions": [
            {"actions": [seat["action"] for seat in step]}
            for step in episode["steps"][1:]
        ],
    }
    return header, tape


def _snapshot(step: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "public": {field: step[0]["observation"][field] for field in PUBLIC_FIELDS},
        "privates": [seat["observation"]["private"] for seat in step],
        "done": all(seat["status"] == "DONE" for seat in step),
        "statuses": [seat["status"] for seat in step],
        "rewards": [seat["reward"] for seat in step],
    }


def _captured(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial": _snapshot(episode["steps"][0]),
        "terminal": _snapshot(episode["steps"][-1]),
        "banks": [
            [farm["money"] for farm in step[0]["observation"]["farms"]]
            for step in episode["steps"][1:]
        ],
        "snapshots": [
            {"transition": transition, "snapshot": _snapshot(step)}
            for transition, step in enumerate(episode["steps"][1:])
        ],
    }


def _fixture_episode(episode_id: str, template: dict[str, Any]) -> dict[str, Any]:
    """Build only from the fixture's Python data and the framework envelope."""
    path = ROOT / "engine_rs/fixtures" / f"episode-{episode_id}.jsonl.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_HASHES[episode_id]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    header, transitions = rows[0], rows[1:]
    assert len(transitions) == header["transitions"] == 719
    episode = copy.deepcopy(template)
    episode["id"] = episode_id
    episode["configuration"] = header["configuration"]
    episode["info"] = {"seed": header["seed"]}
    observations = [header["initial"]["public"]] + [r["expected"] for r in transitions]
    privates = [header["initial"]["privates"]] + [r["privates"] for r in transitions]
    episode["steps"] = []
    for index, (public, private) in enumerate(zip(observations, privates, strict=True)):
        assert public["step"] == index
        step = copy.deepcopy(template["steps"][0])
        for seat in range(2):
            observation = step[seat]["observation"]
            # Use the framework's actual serialization layout, including duplicated
            # farms/market/town/day/hour at seat 1 and its omitted shared step.
            for field in list(observation):
                if field in public:
                    observation[field] = public[field]
            observation["private"] = private[seat]
            if index:
                transition = transitions[index - 1]
                assert transition["from_step"] == index - 1
                step[seat]["action"] = transition["actions"][seat]
                step[seat]["reward"] = transition["rewards"][seat]
                step[seat]["status"] = transition["statuses"][seat]
        episode["steps"].append(step)
    episode["rewards"] = transitions[-1]["rewards"]
    episode["statuses"] = transitions[-1]["statuses"]
    assert episode["rewards"] == header["terminal_banks"]
    return episode


def _semantic_payload(episode: dict[str, Any]) -> dict[str, Any]:
    """Only ignore framework metadata/timing; retain seed and every game value.

    core.py toJSON returns arbitrary top-level env.info and per-seat state.info;
    native provenance is different because Python did not execute native replay.
    core.py __loop_through_interpreter subtracts agent-log duration from the
    remainingOverageTime field. The native replay has no agent execution clock.
    No configuration, action, game observation, status, or reward is ignored.
    """
    result = copy.deepcopy(episode)
    result["info"] = {"seed": result["info"]["seed"]}
    for step in result["steps"]:
        for seat in step:
            seat["info"] = {}
            seat["observation"].pop("remainingOverageTime", None)
    return result


def _ordered(value: Any) -> Any:
    """Key order and JSON number kind are observable: 0 and 0.0 differ.

    Kaggle keeps the schema reward default (integer 0) until DONE, then writes
    float(money); Python's ``==`` would otherwise hide an int/float mismatch.
    """
    if isinstance(value, dict):
        return tuple((key, _ordered(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_ordered(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (type(value).__name__, value)
    return value


def _reject(
    episode: dict[str, Any], label: str, captured: dict[str, Any] | None = None
) -> str:
    with pytest.raises(ValueError, match="transition") as error:
        rs.verify_kaggriculture_episode(
            _dump(episode), None if captured is None else _dump(captured)
        )
    message = str(error.value)
    assert "/" in message, "divergence must contain a JSON pointer"
    assert "transition" in message, "divergence must identify a transition"
    print(f"mutation {label}: {message}")
    return message


@pytest.mark.parametrize("episode_id", FIXTURE_HASHES)
def test_official_fixture_round_trip_all_python_states(
    episode_id: str, framework_episode: dict[str, Any]
) -> None:
    episode = _fixture_episode(episode_id, framework_episode)
    started = time.monotonic()
    report = json.loads(
        rs.verify_kaggriculture_episode(_dump(episode), _dump(_captured(episode)))
    )
    assert report["ok"]
    assert report["mode"] == "semantic"
    assert report["transitions"] == 719
    assert report["captured"] == {
        "initial": True,
        "terminal": True,
        "banks": 719,
        "snapshots": 719,
    }
    actual = json.loads(report["canonical_json"])
    assert _ordered(_semantic_payload(actual)) == _ordered(_semantic_payload(episode))
    print(f"fixture {episode_id}: 719 transitions, {time.monotonic() - started:.6f}s")
    episode["steps"][-1][0]["reward"] = 0.8
    episode["rewards"][0] = 0.8
    # Keep the envelope internally consistent so import admission succeeds and
    # the independent native replay comparison must discover the changed bank.
    for seat in episode["steps"][-1]:
        seat["observation"]["farms"][0]["money"] = 0.8
    message = _reject(episode, f"fixture {episode_id} raw reward to shaped reward")
    assert "/steps/719/0/reward" in message
    assert "718" in message


def test_real_framework_export_and_import_are_independent_oracles(
    framework_episode: dict[str, Any],
) -> None:
    header, tape = _header_tape(framework_episode)
    exported = json.loads(rs.export_kaggriculture_episode(_dump(header), _dump(tape)))
    assert _ordered(_semantic_payload(exported)) == _ordered(
        _semantic_payload(framework_episode)
    )
    report = json.loads(
        rs.verify_kaggriculture_episode(
            _dump(framework_episode), _dump(_captured(framework_episode))
        )
    )
    assert report["ok"]
    assert report["mode"] == "semantic"
    assert report["transitions"] == 9
    assert _ordered(
        _semantic_payload(json.loads(report["canonical_json"]))
    ) == _ordered(_semantic_payload(framework_episode))
    mutated = copy.deepcopy(framework_episode)
    mutated["info"]["seed"] += 1
    _reject(mutated, "framework resolved seed changed")


def test_native_byte_round_trip_and_wide_seed(
    framework_episode: dict[str, Any],
) -> None:
    header, tape = _header_tape(framework_episode)
    exported = rs.export_kaggriculture_episode(_dump(header), _dump(tape))
    report = json.loads(rs.verify_kaggriculture_episode(exported))
    assert report["ok"]
    assert report["mode"] == "byte"
    assert report["canonical_json"] == exported
    assert json.loads(exported)["info"]["seed"] == SEED
    mutated = json.loads(exported)
    assert mutated["steps"][2][0]["action"] == tape["transitions"][1]["actions"][0]
    assert mutated["steps"][2][0]["action"]["market"][0][-1] == 0
    for seat in mutated["steps"][0]:
        assert seat["observation"]["farms"][0]["money"] == 3000.0
        seat["observation"]["farms"][0]["money"] = 3000
    message = _reject(mutated, "native byte round trip float money spelled as integer")
    assert "/steps/0/0/observation/farms/0/money" in message


def test_captured_evidence_is_independent_of_replay(
    framework_episode: dict[str, Any],
) -> None:
    header, tape = _header_tape(framework_episode)
    exported = rs.export_kaggriculture_episode(_dump(header), _dump(tape))
    evidence = _captured(framework_episode)
    report = json.loads(rs.verify_kaggriculture_episode(exported, _dump(evidence)))
    assert report["captured"] == {
        "initial": True,
        "terminal": True,
        "banks": 9,
        "snapshots": 9,
    }
    evidence["banks"][1][0] += 1
    message = _reject(
        json.loads(exported), "captured transition bank changed", evidence
    )
    assert "/banks/1/0" in message
    assert "1" in message


def test_shared_fields_restore_but_private_state_does_not(
    framework_episode: dict[str, Any],
) -> None:
    episode = copy.deepcopy(framework_episode)
    shared = [
        key
        for key, field in episode["specification"]["observation"].items()
        if field.get("shared")
    ]
    for step in episode["steps"]:
        for key in shared:
            step[1]["observation"].pop(key, None)
    assert json.loads(rs.verify_kaggriculture_episode(_dump(episode)))["ok"]
    episode["steps"][1][1]["observation"]["private"] = copy.deepcopy(
        episode["steps"][1][0]["observation"]["private"]
    )
    message = _reject(episode, "seat 0 private state leaked into seat 1")
    assert "/steps/1/1/observation/private" in message


def test_inventory_order_is_not_ignored(framework_episode: dict[str, Any]) -> None:
    episode = copy.deepcopy(framework_episode)
    # Seeds are an ordered private map with several keys even in the initial state.
    seeds = episode["steps"][0][0]["observation"]["private"]["seeds"]
    assert len(seeds) > 1
    episode["steps"][0][0]["observation"]["private"]["seeds"] = dict(
        reversed(list(seeds.items()))
    )
    message = _reject(episode, "initial private key order reversed")
    assert "/steps/0/0/observation/private/seeds" in message


def test_official_inventory_key_order_mutation(
    framework_episode: dict[str, Any],
) -> None:
    episode = _fixture_episode("95324500", framework_episode)
    inventory = episode["steps"][22][1]["observation"]["private"]["inventories"][3]
    assert list(inventory) == ["COW", "WHEAT"]
    episode["steps"][22][1]["observation"]["private"]["inventories"][3] = dict(
        reversed(list(inventory.items()))
    )
    message = _reject(episode, "fixture actor inventory key order reversed")
    assert "/steps/22/1/observation/private/inventories/3" in message
    assert "21" in message


def test_framework_reward_number_kind_is_not_ignored(
    framework_episode: dict[str, Any],
) -> None:
    # Claude review: Kaggle writes integer 0 before DONE; a float 0.0 is a
    # representation change the semantic comparison must report.
    assert type(framework_episode["steps"][1][0]["reward"]) is int
    episode = copy.deepcopy(framework_episode)
    episode["steps"][1][0]["reward"] = 0.0
    message = _reject(episode, "framework ACTIVE reward 0 spelled as float 0.0")
    assert "/steps/1/0/reward" in message
