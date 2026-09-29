"""Host-only recorder custody; no replacement for Task 1.4's native env."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from owl.kaggriculture import replay_export


@pytest.fixture
def native_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def export(header_json: str, tape_json: str) -> str:
        header, tape = json.loads(header_json), json.loads(tape_json)
        calls.append({"header": header, "tape": tape})
        return json.dumps(
            {"info": {"seed": header["seed"]}, "steps": tape["transitions"]}
        )

    def verify(episode_json: str, captured_json: str | None = None) -> str:
        calls[-1]["captured"] = json.loads(captured_json or "{}")
        return json.dumps(
            {"ok": True, "transitions": len(json.loads(episode_json)["steps"])}
        )

    monkeypatch.setattr(
        replay_export.rs, "export_kaggriculture_episode", export, raising=False
    )
    monkeypatch.setattr(
        replay_export.rs, "verify_kaggriculture_episode", verify, raising=False
    )
    return calls


def _recorder(
    tmp_path: Path, total_games: int = 2, count: int = 2
) -> replay_export.ReplayRecorder:
    return replay_export.ReplayRecorder(
        output_dir=tmp_path,
        evaluation_identity="checkpoint-sha:evaluation-1000",
        total_games=total_games,
        config=SimpleNamespace(eval_replay_games=count),
        seat_assignments=[i % 2 for i in range(total_games)],
    )


def _snapshot(step: int, *, done: bool = False) -> dict[str, Any]:
    return {
        "public": {"step": step, "farms": [{"money": 3000.25}, {"money": 2999.5}]},
        "privates": [{"shed": {"WHEAT": 0, "CARROT": 2}}, {"shed": {}}],
        "done": done,
        "statuses": ["DONE", "DONE"] if done else ["ACTIVE", "ACTIVE"],
        "rewards": [3000.25, 2999.5] if done else [0, 0],
    }


def _start(
    recorder: replay_export.ReplayRecorder, ordinal: int = 0, seed: int = 2**100 + 19
) -> None:
    recorder.start_game(
        ordinal,
        ordinal,
        seed,
        {"episodeSteps": 3},
        ordinal % 2,
        {"candidate": "a" * 64, "incumbent": "b" * 64},
        {"source": "source-sha", "engine": "engine-sha", "schema_version": 1},
        captured_initial=_snapshot(0),
    )


def _actions() -> list[dict[str, Any]]:
    return [
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 0], []]},
        {"farmer": ["PICKUP", "WHEAT"], "hands": [], "market": []},
    ]


def test_selection_is_reproducible_and_covers_explicit_seat_schedule() -> None:
    schedule = [0] * 7 + [1] * 13
    selected = replay_export.select_replay_games(
        "eval-a", 20, 8, seat_assignments=schedule
    )
    assert selected == replay_export.select_replay_games(
        "eval-a", 20, 8, seat_assignments=schedule
    )
    assert len(selected) == 8
    assert {schedule[i] for i in selected} == {0, 1}
    assert selected != replay_export.select_replay_games(
        "eval-b", 20, 8, seat_assignments=schedule
    )
    assert (
        replay_export.select_replay_games("off", 20, 0, seat_assignments=schedule)
        == frozenset()
    )


@pytest.mark.parametrize(("total", "count"), [(2, 3), (2, -1), (-1, 0)])
def test_selection_rejects_invalid_counts(total: int, count: int) -> None:
    with pytest.raises(ValueError, match=r"count|total_games"):
        replay_export.select_replay_games("eval", total, count)


def test_selection_cannot_claim_both_seats_with_one_replay() -> None:
    with pytest.raises(ValueError, match=r"both.*seat"):
        replay_export.select_replay_games("eval", 2, 1, seat_assignments=[0, 1])


def test_recorder_retains_consumed_seed_actions_tokens_and_custody(
    tmp_path: Path, native_calls: list[dict[str, Any]]
) -> None:
    recorder = _recorder(tmp_path)
    consumed_seed = 2**100 + 19
    _start(recorder, seed=consumed_seed)
    live_next_seed = consumed_seed + 1
    actions = _actions()
    tokens = [{"tokens": [1, 2, 0], "length": 2}, {"tokens": [3, 0], "length": 1}]
    expected_actions, expected_tokens = copy.deepcopy(actions), copy.deepcopy(tokens)
    recorder.record_transition(0, actions, tokens=tokens, captured=_snapshot(1))
    actions[0]["market"].clear()
    tokens[0]["tokens"][0] = 100
    recorder.record_transition(0, _actions(), captured=_snapshot(2, done=True))
    recorder.finish_game(0, _snapshot(2, done=True))
    assert native_calls[0]["header"]["seed"] == consumed_seed != live_next_seed
    assert native_calls[0]["tape"]["transitions"][0]["actions"] == expected_actions
    assert native_calls[0]["tape"]["transitions"][0]["tokens"] == expected_tokens
    assert native_calls[0]["captured"]["banks"] == [[3000.25, 2999.5]] * 2
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "complete"
    assert sidecar["seed_header"]["seed"] == consumed_seed
    assert sidecar["checkpoint_hashes"] == {
        "candidate": "a" * 64,
        "incumbent": "b" * 64,
    }
    assert sidecar["seat_assignment"] == 0
    episode = (tmp_path / "game_000000.json").read_bytes()
    assert sidecar["episode_sha256"] == hashlib.sha256(episode).hexdigest()


def test_simultaneous_finishes_copy_terminal_before_auto_reset(
    tmp_path: Path, native_calls: list[dict[str, Any]]
) -> None:
    recorder = _recorder(tmp_path)
    for ordinal in range(2):
        _start(recorder, ordinal, seed=70 + ordinal)
        recorder.record_transition(
            ordinal, _actions(), captured=_snapshot(1, done=True)
        )
    terminal = _snapshot(1, done=True)
    recorder.finish_game(0, terminal)
    recorder.finish_game(1, terminal)
    terminal["public"]["farms"][0]["money"] = -100
    terminal["rewards"][0] = -1
    terminal["statuses"][0] = "ACTIVE"
    assert [c["header"]["seed"] for c in native_calls] == [70, 71]
    for ordinal in range(2):
        sidecar = json.loads(
            (tmp_path / f"game_{ordinal:06d}.custody.json").read_text()
        )
        assert sidecar["captured"]["terminal"]["rewards"] == [3000.25, 2999.5]
        assert sidecar["captured"]["terminal"]["statuses"] == ["DONE", "DONE"]
        assert (
            native_calls[ordinal]["captured"]["terminal"]["public"]["farms"][0]["money"]
            == 3000.25
        )


def test_truncation_is_partial_and_does_not_fabricate_terminal(
    tmp_path: Path, native_calls: list[dict[str, Any]]
) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.record_transition(0, _actions(), captured=_snapshot(1))
    recorder.truncate_game(0, reason="evaluation budget exhausted")
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "truncated"
    assert native_calls[0]["tape"]["complete"] is False
    assert "terminal" not in sidecar["captured"]
    assert "winner" not in sidecar


def test_native_error_writes_incomplete_custody_without_successful_episode(
    tmp_path: Path, native_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.record_transition(0, _actions())

    def fail(*_args: str) -> str:
        raise ValueError("transition 0 /steps/1/0/action: native step failed")

    monkeypatch.setattr(replay_export.rs, "export_kaggriculture_episode", fail)
    recorder.finish_game(0, _snapshot(1, done=True))
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "error"
    assert sidecar["complete"] is False
    assert sidecar["action_tape"]["complete"] is False
    assert "transition 0 /steps/1/0/action" in sidecar["error"]
    assert not (tmp_path / "game_000000.json").exists()
    assert native_calls == []


def test_captured_evidence_failure_is_not_written_as_success(
    tmp_path: Path, native_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.record_transition(0, _actions(), captured=_snapshot(1, done=True))

    def fail(*_args: str) -> str:
        raise ValueError("transition 0 /captured/banks/0/0 differs")

    monkeypatch.setattr(replay_export.rs, "verify_kaggriculture_episode", fail)
    recorder.finish_game(0, _snapshot(1, done=True))
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "error"
    assert "/captured/banks/0/0" in sidecar["error"]
    assert not (tmp_path / "game_000000.json").exists()
    assert len(native_calls) == 1


def test_explicit_native_failure_preserves_incomplete_tape(
    tmp_path: Path, native_calls: list[dict[str, Any]]
) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    recorder.record_transition(0, _actions())
    recorder.fail_game(0, error="native transaction rejected before commit")
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "error"
    assert sidecar["action_tape"]["complete"] is False
    assert len(sidecar["action_tape"]["transitions"]) == 1
    assert native_calls == []


def test_unselected_games_do_no_per_transition_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path, total_games=10, count=2)
    ordinal = next(i for i in range(10) if i not in recorder.selected_games)

    def forbid(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unselected game touched payload or native interface")

    monkeypatch.setattr(replay_export, "deepcopy", forbid)
    recorder.start_game(ordinal, 0, None, None, None, None, None)  # type: ignore[arg-type]
    recorder.record_transition(ordinal, None)  # type: ignore[arg-type]
    recorder.finish_game(ordinal, None)  # type: ignore[arg-type]
    recorder.truncate_game(ordinal, reason="unused")
    recorder.fail_game(ordinal, error="unused")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seed", [None, 3.0, -1, True, "10"])
def test_recorder_rejects_invalid_resolved_seed(tmp_path: Path, seed: Any) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(ValueError, match="resolved_seed"):
        _start(recorder, seed=seed)


def test_loader_verifies_installed_framework_hashes_and_shared_step() -> None:
    framework = replay_export.load_pinned_framework()
    assert framework["envelope"]["module_version"] == "1.32.7"
    assert framework["specification"]["observation"]["step"]["shared"] is True
    assert framework["specification"]["observation"]["private"]["shared"] is False
    assert (
        framework["hashes"]["core.py"]
        == "0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e"
    )


def test_loader_rejects_changed_pinned_file(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes

    def changed(path: Path) -> bytes:
        return b"tampered" if path.name == "core.py" else original(path)

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match=r"core.py.*SHA-256"):
        replay_export.load_pinned_framework()


def test_rank_configs_supply_eight_replays_by_default(tmp_path: Path) -> None:
    import yaml

    for filename in ("kaggriculture_2rank.yaml", "kaggriculture_4rank.yaml"):
        config = yaml.safe_load((Path("configs") / filename).read_text())
        recorder = replay_export.ReplayRecorder(
            output_dir=tmp_path,
            evaluation_identity=filename,
            total_games=20,
            config=SimpleNamespace(eval_replay_games=config["rl"]["eval_replay_games"]),
            seat_assignments=[0, 1] * 10,
        )
        assert len(recorder.selected_games) == 8


def test_loader_uses_the_bytes_it_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def changed(path: Path, *args: Any, **kwargs: Any) -> str:
        text = original(path, *args, **kwargs)
        if path.name == "kaggriculture.json":
            spec = json.loads(text)
            spec["title"] = "unverified replacement"
            return json.dumps(spec)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    framework = replay_export.load_pinned_framework()
    assert framework["envelope"]["title"] == "Kaggriculture"


@pytest.mark.parametrize("malformed", ["actions", "tokens"])
def test_malformed_selected_transition_creates_error_custody(
    tmp_path: Path, malformed: str
) -> None:
    recorder = _recorder(tmp_path)
    _start(recorder)
    with pytest.raises(ValueError, match="exactly two"):
        recorder.record_transition(
            0,
            _actions()[:1] if malformed == "actions" else _actions(),
            tokens=[] if malformed == "tokens" else None,
        )
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "error"
    assert sidecar["action_tape"]["complete"] is False
    assert not (tmp_path / "game_000000.json").exists()


@pytest.mark.parametrize(
    "checkpoints",
    [{}, {"candidate": "a" * 64}, {"candidate": "not-a-hash", "incumbent": "b" * 64}],
)
def test_start_requires_both_checkpoint_hashes(
    tmp_path: Path, checkpoints: dict[str, str]
) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_hashes"):
        recorder.start_game(
            0,
            0,
            1,
            {},
            0,
            checkpoints,
            {"source": "source", "engine": "engine", "schema_version": 1},
        )


def test_recorder_writes_a_real_native_verified_episode(tmp_path: Path) -> None:
    """One three-transition framework game; no environment binding substitute."""
    import importlib

    framework = importlib.import_module("kaggle_environments")
    seed = 2**100 + 19
    environment = framework.make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": 4, "turnsPerDay": 1},
    )

    def snapshot() -> dict[str, Any]:
        episode = environment.toJSON()
        state = episode["steps"][-1]
        return {
            "public": {
                key: state[0]["observation"][key]
                for key in ("step", "day", "hour", "farms", "market", "town")
            },
            "privates": [seat["observation"]["private"] for seat in state],
            "done": all(seat["status"] == "DONE" for seat in state),
            "statuses": [seat["status"] for seat in state],
            "rewards": [seat["reward"] for seat in state],
        }

    recorder = replay_export.ReplayRecorder(
        output_dir=tmp_path,
        evaluation_identity="real-native-smoke",
        total_games=1,
        config=SimpleNamespace(eval_replay_games=1),
        seat_assignments=[0],
    )
    recorder.start_game(
        0,
        0,
        environment.info["seed"],
        dict(environment.configuration),
        0,
        {"candidate": "a" * 64, "incumbent": "b" * 64},
        {
            "source": "Task-7.3-test",
            "engine": replay_export.FRAMEWORK_HASHES[
                "envs/kaggriculture/kaggriculture.py"
            ],
            "schema_version": 1,
        },
        captured_initial=snapshot(),
    )
    actions = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
    for _ in range(3):
        environment.step(actions)
        recorder.record_transition(0, actions, captured=snapshot())
    recorder.finish_game(0, snapshot())
    sidecar = json.loads((tmp_path / "game_000000.custody.json").read_text())
    assert sidecar["status"] == "complete", sidecar.get("error")
    assert sidecar["verification"]["captured"] == {
        "initial": True,
        "terminal": True,
        "banks": 3,
        "snapshots": 3,
    }
    episode = json.loads((tmp_path / "game_000000.json").read_text())
    assert episode["info"]["seed"] == seed
    assert episode["rewards"] == environment.toJSON()["rewards"]
    assert episode["statuses"] == ["DONE", "DONE"]


def _native_replay_input() -> tuple[dict[str, Any], dict[str, Any]]:
    framework = replay_export.load_pinned_framework()
    configuration = framework["default_configuration"]
    configuration.update(episodeSteps=3, seed=None)
    header = {
        "seed": 71,
        "configuration": configuration,
        "provenance": {
            "source": "test-source",
            "engine": "test-engine",
            "schema_version": 1,
            "target_framework_version": "1.32.7",
        },
        "specification": framework["specification"],
        "envelope": framework["envelope"],
    }
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    tape = {"complete": True, "transitions": [{"actions": [action, action]}] * 2}
    return header, tape


def _call_native(operation: str, data_json: str, tape_json: str) -> str:
    if operation == "export":
        return replay_export.rs.export_kaggriculture_episode(data_json, tape_json)
    return replay_export.rs.verify_kaggriculture_episode(data_json, None)


@pytest.mark.parametrize("operation", ["export", "verify"])
@pytest.mark.parametrize("mutation", ["action_default", "shared_private", "title"])
def test_direct_native_api_rejects_unpinned_schema(
    operation: str, mutation: str
) -> None:
    header, tape = _native_replay_input()
    if operation == "verify":
        data = json.loads(
            replay_export.rs.export_kaggriculture_episode(
                json.dumps(header), json.dumps(tape)
            )
        )
    else:
        data = header
    if mutation == "action_default":
        data["specification"]["action"]["default"]["farmer"] = ["NORTH"]
        if operation == "verify":
            for seat in data["steps"][0]:
                seat["action"]["farmer"] = ["NORTH"]
    elif mutation == "shared_private":
        data["specification"]["observation"]["private"]["shared"] = True
    elif operation == "verify":
        data["title"] = "Unpinned game title"
    else:
        data["envelope"]["title"] = "Unpinned game title"
    with pytest.raises(ValueError, match=r"specification|title|envelope"):
        _call_native(operation, json.dumps(data), json.dumps(tape))


@pytest.mark.parametrize("operation", ["export", "verify"])
@pytest.mark.parametrize("failure", ["missing", "hash"])
def test_direct_native_api_requires_installed_framework_custody(
    monkeypatch: pytest.MonkeyPatch, operation: str, failure: str
) -> None:
    header, tape = _native_replay_input()
    episode_json = replay_export.rs.export_kaggriculture_episode(
        json.dumps(header), json.dumps(tape)
    )

    def failed_loader() -> dict[str, Any]:
        raise ValueError(
            "pinned framework unavailable"
            if failure == "missing"
            else "core.py SHA-256 differs"
        )

    monkeypatch.setattr(replay_export, "load_pinned_framework", failed_loader)
    data_json = json.dumps(header) if operation == "export" else episode_json
    with pytest.raises(ValueError, match=r"unavailable|SHA-256"):
        _call_native(operation, data_json, json.dumps(tape))


@pytest.mark.parametrize("field", ["actTimeout", "runTimeout"])
@pytest.mark.parametrize("value", ["not-a-number", -1])
def test_direct_native_api_rejects_invalid_framework_time_budgets(
    field: str, value: Any
) -> None:
    header, tape = _native_replay_input()
    episode = json.loads(
        replay_export.rs.export_kaggriculture_episode(
            json.dumps(header), json.dumps(tape)
        )
    )
    episode["configuration"][field] = value
    episode["info"].pop("v3_native_replay")
    with pytest.raises(ValueError, match=r"configuration"):
        replay_export.rs.verify_kaggriculture_episode(json.dumps(episode), None)
