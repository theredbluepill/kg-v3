"""Run every probe case in fresh subprocesses on GPU 0 within a wall budget.

A CUDA illegal memory access poisons the process's context, so each case runs
in its own process; when a process dies mid-point the driver records a fault
for that point and resumes the remaining points in a new process (the
per-case Inductor cache keeps the restart warm). No case starts after
BUDGET_S - RESERVE_S; every process is killed at the budget.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
PY = "/workspace/kg-v3/.venv/bin/python"
SRC = "/workspace/gemm-limits-src-1ddc71d/python"
BUDGET_S = 45 * 60
RESERVE_S = 4 * 60
MAX_RESTARTS = 3

L512 = 2**31 // 512
L256 = 2**31 // 256
L4096 = 2**31 // 4096
L768 = 2**31 // 768


def pts(limit: int) -> list[str]:
    return [str(limit - 4096), str(limit), str(limit + 1), str(limit + 4096),
            str(2 * limit)]


CASES: list[tuple[str, str, list[str], list[str]]] = [
    ("trunk_packed", "trunk",
     ["--path", "packed"],
     ["sparse:256", "dense:256", "dense:5914", "dense:5915", "packed:4194303",
      "packed:4194304", "dense:5916", "dense:11830"]),
    ("mlp_256_512_256", "lin", ["--case", "mlp:256:512:256"], pts(L512)),
    ("lin_256_512", "lin", ["--case", "lin:256:512"], pts(L512)),
    ("lin_512_256", "lin", ["--case", "lin:512:256"], pts(L512)),
    ("trunk_packed_bypass", "trunk",
     ["--path", "packed", "--bypass-guard"],
     ["packed:4194305", "packed:4198400", "dense:5916", "dense:11830",
      "dense:11832"]),
    ("trunk_padded", "trunk",
     ["--path", "padded"],
     ["sparse:256", "dense:256", "dense:5914", "dense:5915", "dense:5916",
      "dense:11830"]),
    ("trunk_padded_bypass", "trunk",
     ["--path", "padded", "--bypass-guard"], ["dense:5916", "dense:11830"]),
    ("lin_16_4096", "lin", ["--case", "lin:16:4096"], pts(L4096)),
    ("lin_4096_16", "lin", ["--case", "lin:4096:16"], pts(L4096)),
    ("lin_256_256", "lin", ["--case", "lin:256:256"], pts(L256)),
    ("lin_768_256", "lin", ["--case", "lin:768:256"], pts(L768)),
    ("mlpbwd_256_512_256", "lin", ["--case", "mlpbwd:256:512:256"],
     [str(L512 - 4096), str(L512 + 1), str(2 * L512)]),
]


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    t_start = time.monotonic()
    only = set(sys.argv[1:])
    summary = open(RUN / "driver.jsonl", "a")

    def log(rec: dict) -> None:
        rec["t"] = round(time.monotonic() - t_start, 1)
        summary.write(json.dumps(rec) + "\n")
        summary.flush()
        print(json.dumps(rec), flush=True)

    for name, kind, extra, points in CASES:
        if only and name not in only:
            continue
        out = RUN / f"{name}.jsonl"
        remaining = list(points)
        for attempt in range(MAX_RESTARTS + 1):
            elapsed = time.monotonic() - t_start
            if elapsed > BUDGET_S - RESERVE_S:
                log({"case": name, "status": "skipped_budget", "remaining": remaining})
                break
            env = dict(os.environ)
            env.update(
                CUDA_VISIBLE_DEVICES="0",
                PYTHONPATH=SRC,
                OWL_NATIVE_MODULE_DIR="/workspace/gemm-limits-src-1ddc71d/native",
                TORCHINDUCTOR_CACHE_DIR=str(RUN / "inductor_cache" / name),
                TORCH_LOGS="recompiles" if kind == "trunk" else "recompiles,guards",
                PYTHONUNBUFFERED="1",
            )
            if kind == "trunk":
                cmd = [PY, str(RUN / "probe_trunk.py"), *extra,
                       "--points", ";".join(remaining), "--out", str(out)]
            else:
                cmd = [PY, str(RUN / "probe_linear.py"), *extra,
                       "--ms", ",".join(remaining), "--out", str(out)]
            n_before = len(read_events(out))
            log({"case": name, "attempt": attempt, "cmd": cmd, "status": "launch"})
            with open(RUN / f"{name}.attempt{attempt}.log", "w") as err:
                try:
                    proc = subprocess.run(cmd, env=env, stdout=err, stderr=err,
                                          timeout=max(60, BUDGET_S - elapsed))
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            events = read_events(out)[n_before:]
            done = [e for e in events if e.get("event") == "result"
                    and e.get("label") == "target"]
            starts = [e for e in events if e.get("event") == "start"]
            results = [e for e in events if e.get("event") == "result"]
            log({"case": name, "attempt": attempt, "rc": rc,
                 "targets_done": len(done), "status": "exit"})
            if rc == 0:
                break
            if len(starts) > len(results):
                crashed = starts[-1]
                tail = (RUN / f"{name}.attempt{attempt}.log").read_text()[-3000:]
                fault = {"event": "fault", "case": name, "label": crashed["label"],
                         "rc": rc, "at": {k: v for k, v in crashed.items()
                                          if k not in ("event",)},
                         "stderr_tail": tail}
                with open(out, "a") as fh:
                    fh.write(json.dumps(fault) + "\n")
                log({"case": name, "status": "fault", "at": fault["at"]})
                if crashed["label"] == "warm":
                    break
                key = crashed.get("spec", str(crashed.get("M")))
                idx = remaining.index(key)
                remaining = remaining[idx + 1:]
            else:
                break
            if not remaining:
                break
    log({"status": "driver_done", "elapsed_s": round(time.monotonic() - t_start, 1)})


if __name__ == "__main__":
    main()
