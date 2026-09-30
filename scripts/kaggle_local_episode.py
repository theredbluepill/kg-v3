"""One bounded local kaggle-environments episode for the packaged agent (Task 7.4).

Runs the extracted submission's ``main.py`` through Kaggle's real file-agent
loader with ``debug=False`` (the capture, timing and overage path). Because
Kaggle silently turns a non-dict return into PASS and overwrites a final-call
fault to ``DONE``, every ``Agent.act`` result is recorded here before
``Environment.step`` consumes it: duration, exception, and whether the raw
returned action passes the agent's own validator. Every recorded step's status
is inspected, not only the final one.

This proves packaging and legality on the host that runs it, not strength and
not Kaggle-hardware latency. It never uploads or submits anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BAD_STATUSES = {"ERROR", "TIMEOUT", "INVALID"}


@dataclass
class CallRecord:
    seat: int
    step: int
    duration_s: float
    exception: str | None
    default_pass: bool
    action_valid: bool
    invalid_reason: str | None


@dataclass
class EpisodeResult:
    calls: list[CallRecord] = field(default_factory=list)
    statuses: list[list[str]] = field(default_factory=list)
    rewards: list[float | None] = field(default_factory=list)
    banks: list[float] = field(default_factory=list)
    min_overage: list[float] = field(default_factory=list)
    replay: dict[str, Any] = field(default_factory=dict)


def _raw_action_problem(
    action: Any, observation: Any, configuration: Any
) -> str | None:
    """Why a raw returned action fails the agent's validator, or ``None``."""
    import numpy as np
    from owl.kaggriculture.kaggle_agent import (
        ActionValidationError,
        pass_action,
        validate_action,
    )
    from owl.kaggriculture.types import ACTION_SLOTS, MAX_ACTORS, MAX_FRAMES

    if isinstance(action, BaseException):
        return f"exception {type(action).__name__}"
    if action == pass_action():
        # The schema default the guard returns; legal to Kaggle, counted apart.
        return None
    seat = observation["player"]
    try:
        validate_action(
            action,
            actors=len(observation["farms"][seat]["hands"]) + 1,
            order_limit=int(configuration["maxMarketOrdersPerTurn"]),
            hire_limit=MAX_ACTORS,
            scratch=np.empty((MAX_FRAMES, ACTION_SLOTS), dtype=np.int64),
        )
    except ActionValidationError as error:
        return str(error)
    return None


def run_episode(
    agents: list[Any],
    *,
    seed: int,
    recorded_seats: set[int],
    episode_steps: int | None = None,
) -> EpisodeResult:
    """Run one episode, recording each call of the seats in ``recorded_seats``."""
    # kaggle_environments ships no type information; bind it through importlib.
    kaggle_agent = importlib.import_module("kaggle_environments.agent")
    make = importlib.import_module("kaggle_environments").make
    from owl.kaggriculture.kaggle_agent import pass_action

    configuration: dict[str, Any] = {"seed": seed}
    if episode_steps is not None:
        configuration["episodeSteps"] = episode_steps
    env = make("kaggriculture", configuration=configuration, debug=False)
    result = EpisodeResult()
    seats_by_agent: dict[int, int] = {}
    original_act = kaggle_agent.Agent.act

    def recording_act(self: Any, observation: Any) -> Any:
        action, log = original_act(self, observation)
        seat = int(observation["player"])
        seats_by_agent[id(self)] = seat
        if seat in recorded_seats:
            problem = _raw_action_problem(action, observation, env.configuration)
            result.calls.append(
                CallRecord(
                    seat=seat,
                    step=int(observation["step"]),
                    duration_s=float(log["duration"]),
                    exception=(
                        f"{type(action).__name__}: {action}"
                        if isinstance(action, BaseException)
                        else None
                    ),
                    default_pass=action == pass_action(),
                    action_valid=problem is None,
                    invalid_reason=problem,
                )
            )
        return action, log

    kaggle_agent.Agent.act = recording_act
    try:
        env.run(agents)
    finally:
        kaggle_agent.Agent.act = original_act
    result.statuses = [[state["status"] for state in step] for step in env.steps]
    final = env.steps[-1]
    result.rewards = [state["reward"] for state in final]
    shared = final[0]["observation"]
    result.banks = [float(farm["money"]) for farm in shared["farms"]]
    result.min_overage = [
        min(
            float(step[seat]["observation"]["remainingOverageTime"])
            for step in env.steps
        )
        for seat in range(2)
    ]
    result.replay = env.toJSON()
    return result


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]

    return {
        "n": len(ordered),
        "p50": pick(0.5),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def summarize(result: EpisodeResult, *, recorded_seats: set[int]) -> dict[str, Any]:
    bad = [
        {"step": index, "seat": seat, "status": status}
        for index, step in enumerate(result.statuses)
        for seat, status in enumerate(step)
        if status in BAD_STATUSES
    ]
    per_seat: dict[str, Any] = {}
    for seat in sorted(recorded_seats):
        calls = [call for call in result.calls if call.seat == seat]
        steady = [call.duration_s for call in calls if call.step > 0]
        per_seat[str(seat)] = {
            "calls": len(calls),
            "exceptions": sum(call.exception is not None for call in calls),
            "invalid_raw_actions": sum(not call.action_valid for call in calls),
            "default_pass_returns": sum(call.default_pass for call in calls),
            "turn0_duration_s": next(
                (call.duration_s for call in calls if call.step == 0), None
            ),
            "steady_duration_s": _quantiles(steady),
            "steady_excess_over_1s_sum": sum(max(0.0, value - 1.0) for value in steady),
            "min_remaining_overage_s": result.min_overage[seat],
            "first_problems": [
                asdict(call)
                for call in calls
                if call.exception is not None or not call.action_valid
            ][:5],
        }
    return {
        "recorded_steps": len(result.statuses),
        "bad_statuses": bad[:20],
        "bad_status_count": len(bad),
        "final_statuses": result.statuses[-1] if result.statuses else [],
        "final_rewards": result.rewards,
        "final_banks": result.banks,
        "seats": per_seat,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--opponent",
        default="self",
        help="'self' (a second copy of main.py, like Kaggle's validation episode) "
        "or a built-in agent name such as 'starter'",
    )
    parser.add_argument("--agent-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    agent_dir = args.agent_dir.resolve()
    main_path = agent_dir / "main.py"
    # Kaggle's image has no other owl; make the extracted package win here too.
    sys.path.insert(0, str(agent_dir))
    os.environ["KAGGRICULTURE_AGENT_STRICT"] = "1"
    import torch

    kaggle_environments = importlib.import_module("kaggle_environments")

    if args.opponent == "self":
        agents: list[Any] = [str(main_path), str(main_path)]
        recorded = {0, 1}
    else:
        agents = [args.opponent, args.opponent]
        agents[args.agent_seat] = str(main_path)
        recorded = {args.agent_seat}
    started = time.time()
    result = run_episode(agents, seed=args.seed, recorded_seats=recorded)
    wall_s = time.time() - started
    import owl
    import owl.rs

    owl_file = Path(owl.__file__).resolve()
    rs_file = Path(owl.rs.__file__).resolve()
    for path in (owl_file, rs_file):
        if agent_dir not in path.parents:
            raise RuntimeError(f"{path} was not loaded from the extracted agent")

    replay = json.dumps(result.replay, separators=(",", ":")).encode()
    args.replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = args.replay_dir / f"replay-seed{args.seed}-{args.opponent}.json"
    replay_path.write_bytes(replay)
    if kaggle_environments.__file__ is None:
        raise RuntimeError("kaggle_environments has no source file")
    kaggriculture_py = (
        Path(kaggle_environments.__file__).parent
        / "envs"
        / "kaggriculture"
        / "kaggriculture.py"
    )
    summary = summarize(result, recorded_seats=recorded)
    expected_calls = len(result.statuses) - 1
    qualified = summary["bad_status_count"] == 0 and all(
        seat["calls"] == expected_calls
        and seat["exceptions"] == 0
        and seat["invalid_raw_actions"] == 0
        for seat in summary["seats"].values()
    )
    manifest_path = agent_dir / "manifest.json"
    receipt = {
        "qualified": qualified,
        "expected_calls_per_seat": expected_calls,
        "seed": args.seed,
        "opponent": args.opponent,
        "agent_seats": sorted(recorded),
        "strict": True,
        "wall_s": round(wall_s, 3),
        "summary": summary,
        "replay": {
            "path": str(replay_path),
            "bytes": len(replay),
            "sha256": _sha256_bytes(replay),
        },
        "loaded_from": {"owl": str(owl_file), "owl.rs": str(rs_file)},
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes())
        if manifest_path.is_file()
        else None,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "kaggle_environments": kaggle_environments.__version__,
            "kaggriculture_py_sha256": _sha256_bytes(kaggriculture_py.read_bytes()),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "host_note": "host CPU without a vCPU quota; not Kaggle hardware",
        "telemetry": "not wired (no W&B run for this harness)",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("qualified", "wall_s")}))
    print(json.dumps(summary, indent=2))
    if not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
