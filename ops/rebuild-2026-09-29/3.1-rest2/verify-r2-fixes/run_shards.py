"""Run the Python suite in bounded shards on the Mac; log pass counts and max RSS."""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPLIT = {"tests/kaggriculture/test_model_heads.py", "tests/kaggriculture/test_teacher.py"}
ENV = {**os.environ, "OMP_NUM_THREADS": "2"}


def run(label: str, args: list[str]) -> None:
    cmd = ["/usr/bin/time", "-l", "uv", "run", "pytest", "-q", "-p", "no:cacheprovider",
           "-m", "not slow", *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
    lines = [l for l in r.stdout.splitlines() if re.search(r"\d+ (passed|failed|skipped|error)", l)]
    rss = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    print(f"== {label} | rc={r.returncode} | {lines[-1] if lines else ''} | "
          f"maxrss={rss.group(1) if rss else '?'}", flush=True)
    if r.returncode not in (0, 5):
        print(r.stdout[-3000:], flush=True)


def ids(path: str) -> list[str]:
    r = subprocess.run(["uv", "run", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider",
                        path], cwd=ROOT, capture_output=True, text=True, env=ENV)
    return [l for l in r.stdout.splitlines() if "::" in l]


run("tests --ignore=tests/kaggriculture", ["tests", "--ignore=tests/kaggriculture"])
for path in sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests/kaggriculture").glob("test_*.py")):
    if path in SPLIT:
        node_ids = ids(path)
        half = len(node_ids) // 2
        for i, chunk in enumerate((node_ids[:half], node_ids[half:]), 1):
            run(f"{path} [{i}/2]", chunk)
    else:
        run(path, [path])
