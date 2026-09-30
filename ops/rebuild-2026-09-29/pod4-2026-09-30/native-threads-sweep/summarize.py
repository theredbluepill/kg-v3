"""Summarize the native_threads sweep receipts (iterations 3-10) into a table.

Usage: python summarize.py RECEIPTS_DIR  (one subdirectory per row with run.log)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROWS = ["nt2", "nt4", "nt8", "nt16", "nt8-numa", "nt8-cpubind", "nt4-r2", "nt8-r2", "nt2-r2"]
FIRST, LAST = 3, 10
STEPS_PER_ITER = 16_384


def records(log: Path) -> list[dict]:
    # Ranks share stdout, so records can be concatenated on one line.
    decoder = json.JSONDecoder()
    text = log.read_text(errors="replace")
    out = []
    tag = "[nt-probe] "
    pos = text.find(tag)
    while pos != -1:
        record, _ = decoder.raw_decode(text, pos + len(tag))
        out.append(record)
        pos = text.find(tag, pos + len(tag))
    return out


def warnings(log: Path) -> list[str]:
    seen: list[str] = []
    for line in log.read_text(errors="replace").splitlines():
        low = line.lower()
        if ("warn" in low or "error" in low or "traceback" in low) and "[nt-probe]" not in line:
            key = line.strip()[:160]
            if key not in seen:
                seen.append(key)
    return seen


def summarize(row_dir: Path) -> dict:
    recs = records(row_dir / "run.log")
    its = {(r["rank"], r["iteration"]): r for r in recs if r["kind"] == "iteration"}
    r0 = {i: its[(0, i)] for i in range(1, LAST + 1) if (0, i) in its}
    if FIRST - 1 not in r0 or LAST not in r0:
        return {"row": row_dir.name, "complete": False, "iterations": len(r0)}
    window = range(FIRST, LAST + 1)
    wall = r0[LAST]["perf_end"] - r0[FIRST - 1]["perf_end"]
    m = {i: r0[i]["metrics"] for i in window}
    seats = r0[LAST]["metrics"]["train/player_step_total"] - r0[FIRST - 1]["metrics"]["train/player_step_total"]
    ranks = sorted({r for r, _ in its})
    step_ms = {}
    cores = {}
    for rank in ranks:
        calls = sum(its[(rank, i)]["native_step_calls"] for i in window)
        secs = sum(its[(rank, i)]["native_step_seconds"] for i in window)
        step_ms[rank] = 1000 * secs / calls
        cpu = its[(rank, LAST)]["proc_cpu_total"] - its[(rank, FIRST - 1)]["proc_cpu_total"]
        cores[rank] = cpu / wall
    busy = r0[LAST]["stat_busy"] - r0[FIRST - 1]["stat_busy"]
    total = r0[LAST]["stat_total"] - r0[FIRST - 1]["stat_total"]
    numa = [r for r in recs if r["kind"] == "numa_bind"]
    wb = [r for r in recs if r["kind"] == "wandb"]
    return {
        "row": row_dir.name,
        "complete": True,
        "iterations": len(r0),
        "iter1_seconds": r0[1]["metrics"]["time/iteration_seconds"],
        "mean_iteration_seconds": mean(m[i]["time/iteration_seconds"] for i in window),
        "wall_3_10": wall,
        "game_sps": len(window) * STEPS_PER_ITER / wall,
        "trainer_sps_mean": mean(m[i]["perf/steps_per_second"] for i in window),
        "seat_sps": seats / wall,
        "rollout_s": mean(m[i]["time/rollout_seconds"] for i in window),
        "teacher_s": mean(m[i]["time/teacher_seconds"] for i in window),
        "update_s": mean(m[i]["time/update_seconds"] for i in window),
        "native_step_ms": step_ms,
        "native_step_share_of_rollout": (
            max(step_ms.values()) * 64 / 1000 / mean(m[i]["time/rollout_seconds"] for i in window)
        ),
        "proc_cores": cores,
        "machine_cpu_pct": 100 * busy / total,
        "numa": numa,
        "wandb": wb[0]["url"] if wb else None,
        "warnings": warnings(row_dir / "run.log"),
    }


def main() -> None:
    root = Path(sys.argv[1])
    out = [summarize(root / row) for row in ROWS if (root / row / "run.log").exists()]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
