"""One Kaggle-mode episode of the unpacked submission against a built-in agent.

Kaggle mode: the agent is passed to ``kaggle_environments.make(...).run`` as the
path of the unpacked ``main.py``; ``debug=False`` (stdout captured, timing and
overage enforced by ``Agent.act``); ``KAGGRICULTURE_AGENT_STRICT`` is NOT set, so
the agent's guarded fallback is live exactly as on Kaggle; the agent directory is
NOT put on ``sys.path`` (Kaggle's loader appends it only during the ``exec`` of
``main.py``); the working directory is not the agent directory.

``Agent.act`` is wrapped to record, for the packaged seat, every call's duration,
exception, raw action, the agent's own validator verdict on that raw action, and
the fallback lines the agent prints (caught-error PASS, overage-budget PASS).
After the episode the agent's own counters (``calls``, ``caught_errors``,
``budget_passes``) are read from the loaded ``main.py`` globals. Process memory is
sampled from ``/proc/self/status`` and ``getrusage``.

This is an ops receipt script for Task 7.4 validation; it uploads nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BAD_STATUSES = {"ERROR", "TIMEOUT", "INVALID"}


def _proc_status_kib(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise RuntimeError(f"/proc/self/status has no {field}")


def _mem() -> dict[str, float]:
    return {
        "rss_mib": round(_proc_status_kib("VmRSS") / 1024, 1),
        "hwm_mib": round(_proc_status_kib("VmHWM") / 1024, 1),
        "ru_maxrss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 6),
        "p50": pick(0.5),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--agent-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if "KAGGRICULTURE_AGENT_STRICT" in os.environ:
        raise SystemExit("KAGGRICULTURE_AGENT_STRICT must be unset for Kaggle mode")
    agent_dir = args.agent_dir.resolve()
    if Path.cwd().resolve() == agent_dir:
        raise SystemExit("run from outside the agent directory")
    main_path = agent_dir / "main.py"
    mem_start = _mem()

    import kaggle_environments
    from kaggle_environments import agent as kaggle_agent
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"seed": args.seed}, debug=False)
    config = env.configuration
    act_timeout = float(config.actTimeout)
    initial_overage = float(env.state[0].observation.remainingOverageTime)
    run_timeout = float(config.runTimeout)
    episode_steps = int(config.episodeSteps)

    calls: list[dict[str, Any]] = []
    rss_after_call: list[float] = []
    original_act = kaggle_agent.Agent.act

    def recording_act(self: Any, observation: Any) -> Any:
        action, log = original_act(self, observation)
        if int(observation["player"]) != args.agent_seat:
            return action, log
        # main.py imported these during its exec; Kaggle never puts the agent
        # directory on sys.path afterwards, so take them from sys.modules.
        kaggle_agent_mod = sys.modules["owl.kaggriculture.kaggle_agent"]
        types_mod = sys.modules["owl.kaggriculture.types"]
        import numpy as np

        problem = None
        exception = None
        is_pass = False
        if isinstance(action, BaseException):
            exception = f"{type(action).__name__}: {action}"
            problem = "exception"
        else:
            is_pass = action == kaggle_agent_mod.pass_action()
            seat = int(observation["player"])
            try:
                kaggle_agent_mod.validate_action(
                    action,
                    actors=len(observation["farms"][seat]["hands"]) + 1,
                    order_limit=int(config["maxMarketOrdersPerTurn"]),
                    hire_limit=types_mod.MAX_ACTORS,
                    scratch=np.empty(
                        (types_mod.MAX_FRAMES, types_mod.ACTION_SLOTS), dtype=np.int64
                    ),
                )
            except kaggle_agent_mod.ActionValidationError as error:
                problem = str(error)
        stdout = log.get("stdout", "")
        calls.append(
            {
                "step": int(observation["step"]),
                "duration_s": float(log["duration"]),
                "remaining_overage_s": float(observation["remainingOverageTime"]),
                "exception": exception,
                "invalid_reason": problem,
                "pass_action": is_pass,
                "caught_error_line": "caught_errors=" in stdout,
                "budget_pass_line": ": PASS" in stdout and "overage=" in stdout
                and "below" in stdout,
                "stderr_nonempty": bool(log.get("stderr", "").strip()),
            }
        )
        rss_after_call.append(_proc_status_kib("VmRSS") / 1024)
        return action, log

    agents: list[Any] = [args.opponent, args.opponent]
    agents[args.agent_seat] = str(main_path)
    kaggle_agent.Agent.act = recording_act
    started = time.time()
    try:
        env.run(agents)
    finally:
        kaggle_agent.Agent.act = original_act
    wall_s = time.time() - started
    mem_end = _mem()

    import owl
    import owl.rs

    loaded = {"owl": owl.__file__, "owl.rs": owl.rs.__file__}
    for path in loaded.values():
        if agent_dir not in Path(str(path)).resolve().parents:
            raise RuntimeError(f"{path} was not loaded from the unpacked agent")

    # The packaged agent's own counters, from the exec'd main.py globals.
    counters = None
    import gc

    for obj in gc.get_objects():
        if type(obj).__name__ == "KaggricultureAgent":
            counters = {
                "calls": obj.calls,
                "caught_errors": obj.caught_errors,
                "budget_passes": obj.budget_passes,
                "strict": obj.strict,
            }
            break

    statuses = [[s["status"] for s in step] for step in env.steps]
    bad = [
        {"step": i, "seat": seat, "status": status}
        for i, step in enumerate(statuses)
        for seat, status in enumerate(step)
        if status in BAD_STATUSES
    ]
    final = env.steps[-1]
    rewards = [s["reward"] for s in final]
    banks = [float(farm["money"]) for farm in final[0]["observation"]["farms"]]
    winner = (
        "draw"
        if banks[0] == banks[1]
        else ("seat0" if banks[0] > banks[1] else "seat1")
    )
    min_overage = min(
        float(step[args.agent_seat]["observation"]["remainingOverageTime"])
        for step in env.steps
    )
    durations = [c["duration_s"] for c in calls]
    steady = [c["duration_s"] for c in calls if c["step"] > 0]
    turn0 = next((c["duration_s"] for c in calls if c["step"] == 0), None)
    over_timeout = [c for c in calls if c["duration_s"] > act_timeout]
    replay = json.dumps(env.toJSON(), separators=(",", ":")).encode()
    args.replay.parent.mkdir(parents=True, exist_ok=True)
    args.replay.write_bytes(replay)

    fallback = {
        "agent_counters": counters,
        "pass_action_returns": sum(c["pass_action"] for c in calls),
        "caught_error_lines": sum(c["caught_error_line"] for c in calls),
        "budget_pass_lines": sum(c["budget_pass_line"] for c in calls),
        "stderr_nonempty_calls": sum(c["stderr_nonempty"] for c in calls),
    }
    expected_calls = len(statuses) - 1
    ok = (
        not bad
        and all(s == "DONE" for s in statuses[-1])
        and len(calls) == expected_calls
        and not any(c["exception"] or c["invalid_reason"] for c in calls)
        and counters is not None
        and counters["caught_errors"] == 0
        and counters["budget_passes"] == 0
        and fallback["caught_error_lines"] == 0
        and fallback["budget_pass_lines"] == 0
    )
    ke_file = Path(str(kaggle_environments.__file__)).parent
    import torch

    receipt = {
        "ok": ok,
        "mode": "kaggle (debug=False, strict unset, agent dir not on sys.path)",
        "seed": args.seed,
        "agent_seat": args.agent_seat,
        "opponent": args.opponent,
        "wall_s": round(wall_s, 3),
        "limits": {
            "episodeSteps": episode_steps,
            "actTimeout_s": act_timeout,
            "initial_remainingOverageTime_s": initial_overage,
            "runTimeout_s": run_timeout,
        },
        "completion": {
            "recorded_steps": len(statuses),
            "final_statuses": statuses[-1],
            "bad_status_count": len(bad),
            "bad_statuses": bad[:20],
        },
        "legality": {
            "calls": len(calls),
            "expected_calls": expected_calls,
            "exceptions": sum(c["exception"] is not None for c in calls),
            "invalid_raw_actions": sum(c["invalid_reason"] is not None for c in calls),
            "first_problems": [
                c for c in calls if c["exception"] or c["invalid_reason"]
            ][:5],
            "env_note": "kaggriculture.py treats illegal unit actions as silent "
            "no-ops (no penalty, no status); framework INVALID/ERROR/TIMEOUT "
            "statuses are counted under completion",
        },
        "timing_s": {
            "all_calls": _stats(durations),
            "turn0": turn0,
            "steady": _stats(steady),
            "calls_over_actTimeout": len(over_timeout),
            "overage_used_s": round(initial_overage - min_overage, 6),
            "min_remaining_overage_s": min_overage,
        },
        "memory_mib": {
            "process_start": mem_start,
            "process_end": mem_end,
            "rss_after_agent_calls": _stats(rss_after_call),
            "note": "one process holds the env, the opponent and the packaged "
            "agent (Kaggle's in-process file-agent path)",
        },
        "result": {
            "final_rewards": rewards,
            "final_banks": banks,
            "packaged_seat_bank": banks[args.agent_seat],
            "opponent_bank": banks[1 - args.agent_seat],
            "winner": winner,
            "packaged_agent_won": winner == f"seat{args.agent_seat}",
            "note": "one game; a packaging check, not a strength claim",
        },
        "fallback": fallback,
        "replay": {
            "path": str(args.replay),
            "bytes": len(replay),
            "sha256": hashlib.sha256(replay).hexdigest(),
        },
        "loaded_from": loaded,
        "manifest_sha256": hashlib.sha256(
            (agent_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "kaggle_environments": kaggle_environments.__version__,
            "kaggriculture_py_sha256": hashlib.sha256(
                (ke_file / "envs" / "kaggriculture" / "kaggriculture.py").read_bytes()
            ).hexdigest(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nice": os.nice(0),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("ok", "wall_s", "result", "fallback")}))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
