"""Execute a bounded check with a prior question and an unmasked exit receipt."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("id")
    parser.add_argument("--question", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--stop", required=True)
    separator = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:separator])
    command = sys.argv[separator + 1:]
    if not command:
        parser.error("command required")
    folder = Path(__file__).resolve().parent
    record = {
        "id": args.id,
        "question": args.question,
        "inputs": args.inputs,
        "expected_discriminator": args.expect,
        "stopping_condition": args.stop,
        "command": command,
        "environment": {"CARGO_BUILD_JOBS": "2", "CARGO_NET_OFFLINE": "true", "UV_OFFLINE": "true"},
        "log": f"logs/{args.id}.log",
        "exit_code": None,
    }
    receipt = folder / "checks.json"
    def update(done: bool) -> None:
        with receipt.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            content = stream.read()
            rows = json.loads(content) if content else []
            if done:
                rows = [record if row["id"] == args.id else row for row in rows]
            else:
                if any(row["id"] == args.id for row in rows):
                    raise ValueError(f"duplicate check id: {args.id}")
                rows.append(record)
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps(rows, indent=2) + "\n")
    update(False)
    env = os.environ.copy()
    env.update(record["environment"])
    start = time.monotonic()
    with (folder / record["log"]).open("w", encoding="utf-8") as log:
        log.write("$ CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true UV_OFFLINE=true " + shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nEXIT_CODE={result.returncode}\n")
    record["exit_code"] = result.returncode
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    update(True)
    print(f"{args.id}: exit {result.returncode}; {record['log']}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
