"""Bound the canonical smoke's process tree; use the existing psutil dependency."""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COMMAND = [sys.executable, "-m", "pytest", "tests/kaggriculture/test_training_smoke.py::test_no_teacher_two_updates", "-q"]
MAX_SECONDS = 115.0
MAX_BYTES = 960 * 1024**2


def main() -> int:
    start = time.monotonic()
    parent_process = psutil.Process()
    peak_bytes = 0
    stopped = None
    with (OUT / "smoke-watchdog.log").open("w") as output:
        child = subprocess.Popen(COMMAND, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        child_process = psutil.Process(child.pid)
        while child.poll() is None:
            rss = 0
            for proc in (parent_process, child_process):
                try:
                    rss += proc.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak_bytes = max(peak_bytes, rss)
            elapsed = time.monotonic() - start
            if elapsed >= MAX_SECONDS or rss >= MAX_BYTES:
                stopped = "wall limit" if elapsed >= MAX_SECONDS else "RSS limit"
                for proc in (child_process,):
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.wait()
                break
            time.sleep(0.01)
    result = {
        "command": COMMAND,
        "exit_code": child.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "peak_aggregate_rss_bytes": peak_bytes,
        "child_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "rss_scope": "watchdog Python plus direct pytest child, sampled each 10 ms; smoke creates no descendants; uv launcher excluded; process enumeration blocked by sandbox",
        "max_seconds": MAX_SECONDS,
        "max_aggregate_rss_bytes": MAX_BYTES,
        "stopped": stopped,
    }
    (OUT / "smoke-resources.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if stopped is not None else child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
