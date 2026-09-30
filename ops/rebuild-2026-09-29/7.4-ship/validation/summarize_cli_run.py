"""Summarize a `kaggle-environments run` self-play of the packaged agent.

Reads the CLI's --out replay and --log agent logs; counts bad statuses, the
agent's fallback lines (caught-error PASS, overage-budget PASS), stderr, PASS
returns and per-seat call durations. Ops receipt helper for Task 7.4.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]  # noqa: E731
    return {"n": len(ordered), "mean": round(statistics.fmean(ordered), 6),
            "p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99), "max": ordered[-1]}


def main() -> None:
    out_path, log_path, hwm_kib, receipt_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    raw = Path(out_path).read_bytes()
    replay = json.loads(raw)
    logs = json.loads(Path(log_path).read_text())
    steps = replay["steps"]
    config = replay["configuration"]
    statuses = [[s["status"] for s in step] for step in steps]
    bad = [{"step": i, "seat": j, "status": st} for i, step in enumerate(statuses)
           for j, st in enumerate(step) if st in {"ERROR", "TIMEOUT", "INVALID"}]
    banks = [float(f["money"]) for f in steps[-1][0]["observation"]["farms"]]
    seats = {}
    for seat in (0, 1):
        entries = [(i, step[seat]) for i, step in enumerate(logs) if seat < len(step) and step[seat]]
        durations = [e["duration"] for _, e in entries]
        # logs[0] is the empty initial step; the first logged call is turn 0.
        steady = durations[1:]
        actions = [step[seat]["action"] for step in steps[1:]]
        seats[str(seat)] = {
            "calls_logged": len(entries),
            "turn0_duration_s": entries[0][1]["duration"] if entries else None,
            "all_calls_s": stats(durations),
            "steady_s": stats(steady),
            "calls_over_actTimeout": sum(d > float(config["actTimeout"]) for d in durations),
            "min_remaining_overage_s": min(float(step[seat]["observation"]["remainingOverageTime"]) for step in steps),
            "caught_error_lines": sum("caught_errors=" in e.get("stdout", "") for _, e in entries),
            "budget_pass_lines": sum("below" in e.get("stdout", "") and ": PASS" in e.get("stdout", "") for _, e in entries),
            "stderr_nonempty_calls": sum(bool(e.get("stderr", "").strip()) for _, e in entries),
            "pass_action_returns": sum(a == PASS for a in actions),
            "non_dict_actions": sum(not isinstance(a, dict) for a in actions),
        }
    ok = (not bad and all(s == "DONE" for s in statuses[-1])
          and all(v["calls_logged"] == len(steps) - 1 and v["caught_error_lines"] == 0
                  and v["budget_pass_lines"] == 0 and v["non_dict_actions"] == 0
                  for v in seats.values()))
    receipt = {
        "ok": ok,
        "path": "kaggle-environments CLI `run` (kaggle_environments.main.action_run), self-play of the unpacked main.py, debug unset (False)",
        "note": "kaggle-environments 1.32.7 has no dedicated validate/test action (main.py actions: list, evaluate, act, step, run, load, dispose, http-server); `run` of the submission against itself is the shape of Kaggle's validation episode. The agents run non-strict; statuses are recorded per step.",
        "limits": {"episodeSteps": config["episodeSteps"], "actTimeout_s": config["actTimeout"]},
        "recorded_steps": len(steps),
        "final_statuses": statuses[-1],
        "bad_status_count": len(bad),
        "bad_statuses": bad[:20],
        "final_rewards": [s["reward"] for s in steps[-1]],
        "final_banks": banks,
        "winner": "draw" if banks[0] == banks[1] else ("seat0" if banks[0] > banks[1] else "seat1"),
        "seats": seats,
        "process_vmhwm_mib": round(hwm_kib / 1024, 1),
        "process_vmhwm_note": "polled once per second from /proc/<pid>/status; one process holds the env and both agent copies",
        "replay": {"path": out_path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "logs_sha256": hashlib.sha256(Path(log_path).read_bytes()).hexdigest(),
    }
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("ok", "final_banks", "winner", "bad_status_count")}))
    print(json.dumps(seats, indent=1))


if __name__ == "__main__":
    main()
