"""Run one offline Task 1.1 check and retain its actual exit status."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

directory = Path(__file__).resolve().parent
label, expected, *command = sys.argv[1:]
environment = dict(os.environ, CARGO_BUILD_JOBS="3", CARGO_NET_OFFLINE="true", UV_OFFLINE="true")
started = time.time()
with (directory / f"{label}.log").open("w") as output:
    output.write(f"$ {' '.join(command)}\nCARGO_BUILD_JOBS=3 CARGO_NET_OFFLINE=true UV_OFFLINE=true\n")
    output.flush()
    result = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    output.write(f"\nEXIT_CODE={result.returncode}\n")
record = dict(command=command, exit_code=result.returncode, expected_exit_code=int(expected),
              elapsed_seconds=round(time.time()-started, 3), log=f"{label}.log")
(directory / f"{label}.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record))
raise SystemExit(0 if result.returncode == int(expected) else 1)
