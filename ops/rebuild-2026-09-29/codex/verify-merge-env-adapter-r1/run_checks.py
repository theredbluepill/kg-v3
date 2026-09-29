"""Read-only verification commands with bounded CPU and captured receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ENV = os.environ | {"CARGO_BUILD_JOBS": "2", "OMP_NUM_THREADS": "2"}
CHECKS = [
    ("engine-tests", ["cargo", "test", "--manifest-path", "engine_rs/Cargo.toml", "--locked", "--offline"]),
    ("root-tests", ["cargo", "test", "--locked", "--offline"]),
    ("engine-trim", ["uv", "run", "python", "scripts/check_engine_trim.py"]),
    ("pytest", ["uv", "run", "pytest", "tests", "-m", "not slow", "-q"]),
    ("mypy", ["uv", "run", "mypy", "python/owl", "scripts"]),
    ("docs-fresh", ["uv", "run", "python", "scripts/check_doc_freshness.py"]),
]
rows = []
for name, argv in CHECKS:
    started = time.time()
    print(f"START {name}", flush=True)
    with (OUT / f"{name}.log").open("w") as log:
        result = subprocess.run(argv, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    row = {"name": name, "argv": argv, "cwd": str(ROOT), "returncode": result.returncode,
           "elapsed_seconds": round(time.time() - started, 3), "environment": {k: ENV[k] for k in ("CARGO_BUILD_JOBS", "OMP_NUM_THREADS")},
           "log_sha256": hashlib.sha256((OUT / f"{name}.log").read_bytes()).hexdigest()}
    rows.append(row)
    (OUT / "checks.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    print((OUT / f"{name}.log").read_text()[-1500:], flush=True)
