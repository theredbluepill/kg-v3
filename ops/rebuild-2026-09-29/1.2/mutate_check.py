"""Run a single declared negative control, preserving and restoring exact bytes."""

import argparse
import difflib
import hashlib
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("label")
    parser.add_argument("target", type=Path)
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args(sys.argv[1 : sys.argv.index("--")])
    command = sys.argv[sys.argv.index("--") + 1 :]
    before = args.target.read_bytes()
    old, new = args.old.encode(), args.new.encode()
    if before.count(old) != 1:
        raise ValueError("negative control must replace exactly one source fragment")
    after = before.replace(old, new)
    controls = Path(__file__).resolve().parent / "controls"
    saved = controls / f"{args.label}.before"
    if saved.exists():
        raise ValueError("negative control label already used")
    saved.write_bytes(before)
    (controls / f"{args.label}.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.decode().splitlines(keepends=True),
                after.decode().splitlines(keepends=True),
                fromfile=str(args.target),
                tofile=str(args.target),
            )
        )
    )
    print("before_sha256", hashlib.sha256(before).hexdigest(), flush=True)
    print("mutated_sha256", hashlib.sha256(after).hexdigest(), flush=True)
    try:
        args.target.write_bytes(after)
        result = subprocess.run(command, check=False)
    finally:
        args.target.write_bytes(before)
        assert args.target.read_bytes() == saved.read_bytes()
        print(
            "restored_sha256",
            hashlib.sha256(args.target.read_bytes()).hexdigest(),
            flush=True,
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
