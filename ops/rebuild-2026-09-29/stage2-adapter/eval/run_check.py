"""Bound each offline Stage 2 eval-seam check and keep its real exit status."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

name, *command = sys.argv[1:]
root = Path(__file__).parent
started = time.monotonic()
check_env = dict(os.environ, OMP_NUM_THREADS="2", CARGO_BUILD_JOBS="2", CARGO_NET_OFFLINE="true", UV_OFFLINE="true")
with (root / f"{name}.log").open("w") as log:
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=check_env, timeout=115)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
record = {"command": command, "exit_code": code, "seconds": time.monotonic() - started}
(root / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
print((root / f"{name}.log").read_text()[-6000:])
sys.exit(code)
