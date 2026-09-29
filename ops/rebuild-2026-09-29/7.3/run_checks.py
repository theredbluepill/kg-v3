"""Bounded offline Task 7.3 check receipts; no Git mutations or package sync."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COMMANDS = {
    "native": ["cargo", "test", "--offline", "--lib", "kaggriculture::replay_export"],
    "cargo": ["cargo", "test", "--offline"],
    "engine": ["cargo", "test", "--locked", "--offline", "--manifest-path", "engine_rs/Cargo.toml"],
    "trim": ["uv", "run", "--offline", "python", "scripts/check_engine_trim.py"],
    "maturin": ["uv", "run", "--offline", "maturin", "develop"],
    "focused": ["uv", "run", "--offline", "pytest", "tests/kaggriculture/test_replay_export.py", "tests/kaggriculture/test_replay_export_integration.py", "tests/owl/test_replay.py", "tests/tools/test_check_engine_trim.py", "-q"],
    "rs-prepare": ["uvx", "--offline", "--from", "rust-just", "just", "rs-prepare"],
    "py-prepare": ["uvx", "--offline", "--from", "rust-just", "just", "py-prepare"],
    "prepare": ["uvx", "--offline", "--from", "rust-just", "just", "prepare"],
}

def main() -> int:
    for key, expected in {"CARGO_BUILD_JOBS":"2", "CARGO_NET_OFFLINE":"true", "UV_OFFLINE":"true", "UV_NO_SYNC":"1", "RAYON_NUM_THREADS":"2", "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"1"}.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"missing required environment {key}={expected}")
    os.environ["RUST_TEST_THREADS"] = "2"
    path = OUT / "final-results.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    if "commands" not in result:
        result["commands"] = {}
    failed = False
    for name in sys.argv[1:]:
        command = COMMANDS[name]
        started = time.monotonic()
        with (OUT / f"final-{name}.log").open("w") as log:
            print("$ " + " ".join(command), file=log, flush=True)
            status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic() - started
        result["commands"][name] = {"command": " ".join(command), "exit_status": status, "seconds": elapsed, "log": f"final-{name}.log"}
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: exit {status}, {elapsed:.3f}s", flush=True)
        failed |= status != 0
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
