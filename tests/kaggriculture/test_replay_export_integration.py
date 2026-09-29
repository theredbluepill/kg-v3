"""Live native-environment custody for Kaggriculture evaluation replays.

These tests drive the Task 1.4 ``owl.rs.KaggricultureEnv`` binding through
``owl.kaggriculture.native_evaluation`` and verify every exported episode with
the native round trip plus captured live evidence. Tiny horizons keep them
within the Mac's bounded-check budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from owl import rs
from owl.kaggriculture import replay_export
from owl.kaggriculture.native_evaluation import (
    NativeGameResult,
    evaluate_native_games,
)

from .test_native_env import REWARD, buffers

HIRE_LIMIT = 241
VERSIONS = {
    "source": "Task-7.3-live-test",
    "engine": replay_export.FRAMEWORK_HASHES["envs/kaggriculture/kaggriculture.py"],
    "schema_version": 1,
}
HASHES = {"candidate": "a" * 64, "incumbent": "b" * 64}


def market_policy(
    arrays: dict[str, Any], candidate_seats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic legal programs that exercise market frames and PASS."""
    n_envs = arrays["dones"].shape[0]
    assert candidate_seats.shape == (n_envs,)
    tokens = np.zeros((n_envs, 2, 252, 12), dtype=np.int64)
    lengths = np.full((n_envs, 2), 2, dtype=np.int64)
    tokens[:, :, 0, 1] = 1  # farmer PASS
    tokens[:, :, 1, 11] = 1  # STOP
    step = arrays["globals_int"][:, 0, 0]
    for env_index in range(n_envs):
        seat = int(step[env_index] % 2)
        # HARVEST, one BUY_PRODUCT/SELL WHEAT(1) frame, then STOP.
        tokens[env_index, seat, 0, 1] = 10
        tokens[env_index, seat, 1, 11] = 0
        tokens[env_index, seat, 1, 7] = 4 if seat == 0 else 6
        tokens[env_index, seat, 1, 8] = 1
        tokens[env_index, seat, 1, 10] = 1
        tokens[env_index, seat, 2, 11] = 1
        lengths[env_index, seat] = 3
    return tokens, lengths


def _recorder(
    tmp_path: Path, n_games: int, count: int, seats: list[int]
) -> replay_export.ReplayRecorder:
    return replay_export.ReplayRecorder(
        output_dir=tmp_path,
        evaluation_identity="live-native-eval",
        total_games=n_games,
        config=SimpleNamespace(eval_replay_games=count),
        seat_assignments=seats,
    )


def _run(
    tmp_path: Path,
    *,
    n_envs: int,
    n_games: int,
    count: int,
    configuration: dict[str, Any],
    seed: int = 91,
    stride: int = 3,
    start: str = "observe",
) -> tuple[rs.KaggricultureEnv, list[NativeGameResult], list[int]]:
    env = rs.KaggricultureEnv(
        n_envs,
        seed,
        stride,
        json.dumps(configuration),
        REWARD,
        1,
        hire_limit=HIRE_LIMIT,
    )
    seats = [ordinal % 2 for ordinal in range(n_games)]
    results = evaluate_native_games(
        env,
        buffers(n_envs),
        market_policy,
        n_games=n_games,
        seat_assignments=seats,
        configuration=configuration,
        hire_limit=HIRE_LIMIT,
        recorder=_recorder(tmp_path, n_games, count, seats),
        checkpoint_hashes=HASHES,
        versions=VERSIONS,
        start=start,
    )
    return env, results, seats


def _custody(tmp_path: Path) -> dict[int, dict[str, Any]]:
    return {
        json.loads(path.read_text())["game_ordinal"]: json.loads(path.read_text())
        for path in sorted(tmp_path.glob("game_*.custody.json"))
    }


@pytest.mark.parametrize(
    "event", ["construction", "explicit_reset", "simultaneous_terminal_reset"]
)
def test_live_env_consumed_seed_custody(tmp_path: Path, event: str) -> None:
    # Two envs, stride 3 from seed 91: construction consumes 91/94, an explicit
    # reset 97/100, and the simultaneous terminal auto-reset of both envs
    # allocates the next pair in env-index order.
    configuration = {"episodeSteps": 3}
    n_games = 2 if event != "simultaneous_terminal_reset" else 4
    env, results, _ = _run(
        tmp_path,
        n_envs=2,
        n_games=n_games,
        count=n_games,
        configuration=configuration,
        start="reset" if event == "explicit_reset" else "observe",
    )
    first = (91, 94) if event != "explicit_reset" else (97, 100)
    expected = {0: first[0], 1: first[1]}
    if event == "simultaneous_terminal_reset":
        expected.update({2: first[1] + 3, 3: first[1] + 6})
    custody = _custody(tmp_path)
    assert sorted(custody) == sorted(expected)
    for ordinal, seed in expected.items():
        record = custody[ordinal]
        assert record["status"] == "complete", record.get("error")
        assert record["seed_header"]["seed"] == seed
        episode = json.loads((tmp_path / f"game_{ordinal:06d}.json").read_text())
        assert episode["info"]["seed"] == seed
        # The captured initial snapshot is the live env's, so a wrong seed
        # header would fail native verification rather than pass silently.
        assert record["verification"]["captured"]["initial"] is True
    assert [(r.game_ordinal, r.seed) for r in results] == sorted(expected.items())
    assert [r.env_index for r in results] == [
        ordinal % 2 for ordinal in sorted(expected)
    ]
    assert env.seed_state()[0] > max(expected.values())


def test_evaluate_games_exports_eight_complete_episodes(tmp_path: Path) -> None:
    configuration = {"episodeSteps": 5, "turnsPerDay": 2}
    n_games = 10
    _, results, seats = _run(
        tmp_path,
        n_envs=2,
        n_games=n_games,
        count=8,
        configuration=configuration,
        seed=2**62 + 5,
        stride=1,
    )
    assert [r.game_ordinal for r in results] == list(range(n_games))
    assert len({r.seed for r in results}) == n_games
    custody = _custody(tmp_path)
    assert len(custody) == 8
    selected = replay_export.select_replay_games(
        "live-native-eval", n_games, 8, seat_assignments=seats
    )
    assert set(custody) == set(selected)
    for ordinal, record in custody.items():
        result = results[ordinal]
        assert record["status"] == "complete", record.get("error")
        assert record["seat_assignment"] == seats[ordinal] == result.candidate_seat
        assert record["verification"]["captured"] == {
            "initial": True,
            "terminal": True,
            "banks": 4,
            "snapshots": 4,
        }
        transitions = record["action_tape"]["transitions"]
        assert len(transitions) == 4
        assert all(len(row["tokens"]) == 2 for row in transitions)
        assert any(row["actions"][0]["market"] for row in transitions)
        assert any(row["actions"][1]["market"] for row in transitions)
        episode_json = (tmp_path / f"game_{ordinal:06d}.json").read_text()
        episode = json.loads(episode_json)
        assert episode["info"]["seed"] == result.seed
        assert episode["statuses"] == ["DONE", "DONE"]
        assert episode["rewards"] == list(result.banks)
        # Independent reverification of the published bytes.
        report = json.loads(rs.verify_kaggriculture_episode(episode_json, None))
        assert report["ok"] is True
        assert report["mode"] == "byte"


def test_live_terminal_snapshot_is_captured_before_auto_reset() -> None:
    env = rs.KaggricultureEnv(
        1, 11, 1, json.dumps({"episodeSteps": 3}), REWARD, 1, hire_limit=HIRE_LIMIT
    )
    arrays = buffers(1)
    env.observe(**arrays)
    tokens, lengths = market_policy(arrays, np.zeros(1, dtype=np.int64))
    assert env.terminal_snapshot(0) is None
    env.step(tokens, lengths, **arrays)
    assert not arrays["dones"].any()
    assert env.terminal_snapshot(0) is None
    env.step(tokens, lengths, **arrays)
    assert arrays["dones"].all()
    finished_json = env.terminal_snapshot(0)
    assert finished_json is not None
    finished = json.loads(finished_json)
    live = json.loads(env.state_snapshot(0))
    # The step returned the replacement game's observation; the terminal
    # snapshot still belongs to the completed game.
    assert live["public"]["step"] == 0
    assert live["done"] is False
    assert finished["done"] is True
    assert finished["public"]["step"] == 2
    assert finished["statuses"] == ["DONE", "DONE"]
    metrics = env.terminal_metrics(0)
    assert metrics is not None
    assert [farm["money"] for farm in finished["public"]["farms"]] == [
        metrics["bank_0"],
        metrics["bank_1"],
    ]
    assert arrays["transition_banks_after"][0].tolist() == [
        metrics["bank_0"],
        metrics["bank_1"],
    ]
    env.step(tokens, lengths, **arrays)
    assert env.terminal_snapshot(0) is None


def test_native_evaluation_rejects_inconsistent_schedule() -> None:
    with pytest.raises(ValueError, match="seat_assignments"):
        evaluate_native_games(
            rs.KaggricultureEnv(1, 11, 1, "{}", REWARD, 1, hire_limit=HIRE_LIMIT),
            buffers(1),
            market_policy,
            n_games=2,
            seat_assignments=[0],
            configuration={},
            hire_limit=HIRE_LIMIT,
            recorder=None,
            checkpoint_hashes=HASHES,
            versions=VERSIONS,
        )
