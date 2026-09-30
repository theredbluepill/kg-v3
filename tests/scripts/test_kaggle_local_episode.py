"""Task 7.4: the local-episode harness records faults Kaggle's statuses erase."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "kaggle_local_episode.py"
_SPEC = importlib.util.spec_from_file_location("kaggle_local_episode", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
sys.modules["kaggle_local_episode"] = harness
_SPEC.loader.exec_module(harness)

STEPS = 8


def _passer(fault_step: int | None = None, returns: Any = None) -> Any:
    def agent(obs: Any, _config: Any) -> Any:
        if obs["step"] == fault_step:
            if returns is None:
                raise RuntimeError("injected final-call fault")
            return returns
        hands = [["PASS"] for _ in obs["farms"][obs["player"]]["hands"]]
        return {"farmer": ["PASS"], "hands": hands, "market": []}

    return agent


def test_clean_episode_counts_every_call() -> None:
    result = harness.run_episode(
        [_passer(), _passer()], seed=5, recorded_seats={0, 1}, episode_steps=STEPS
    )
    summary = harness.summarize(result, recorded_seats={0, 1})
    assert summary["recorded_steps"] == STEPS
    assert summary["bad_status_count"] == 0
    for seat in ("0", "1"):
        assert summary["seats"][seat]["calls"] == STEPS - 1
        assert summary["seats"][seat]["exceptions"] == 0
        assert summary["seats"][seat]["invalid_raw_actions"] == 0


def test_final_call_fault_is_recorded_although_the_status_reads_done() -> None:
    final = STEPS - 2
    result = harness.run_episode(
        [_passer(fault_step=final), _passer()],
        seed=5,
        recorded_seats={0},
        episode_steps=STEPS,
    )
    summary = harness.summarize(result, recorded_seats={0})
    assert summary["final_statuses"] == ["DONE", "DONE"]
    assert summary["bad_status_count"] == 0
    seat = summary["seats"]["0"]
    assert seat["calls"] == STEPS - 1
    assert seat["exceptions"] == 1
    assert seat["first_problems"][0]["step"] == final


def test_non_dict_return_is_flagged_before_kaggle_normalizes_it() -> None:
    result = harness.run_episode(
        [_passer(fault_step=2, returns=["PASS"]), _passer()],
        seed=5,
        recorded_seats={0},
        episode_steps=STEPS,
    )
    summary = harness.summarize(result, recorded_seats={0})
    # Kaggle replaced the list with the default action: no bad status anywhere.
    assert summary["bad_status_count"] == 0
    assert summary["seats"]["0"]["invalid_raw_actions"] == 1
    assert summary["seats"]["0"]["first_problems"][0]["step"] == 2
