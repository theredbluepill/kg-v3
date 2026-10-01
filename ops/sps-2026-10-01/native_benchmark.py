"""Record/replay native parity or time native step; choose binary via PYTHONPATH.

Example: PYTHONPATH=/private/tmp/kg-sps-baseline-b2276bc/python .venv/bin/python
ops/sps-2026-10-01/native_benchmark.py record --output baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from owl import rs

from tests.kaggriculture.native_step_oracle import (
    BASE_COMMIT,
    GOLDEN,
    REWARD,
    SEED,
    STRIDE,
    benchmark,
    trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify", "benchmark"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.mode != "record" and args.output.resolve() == GOLDEN.resolve():
        parser.error("only pristine-source record mode may replace the golden")
    binary = Path(rs.__file__)
    source = binary.parents[2]
    files = ["Cargo.lock", "Cargo.toml", "rust-toolchain.toml"]
    for directory in ("src", "engine_rs/src", "opponents_rs/src"):
        files.extend(
            str(path.relative_to(source))
            for path in sorted((source / directory).rglob("*.rs"))
        )
    identity = {
        "platform": platform.platform(),
        "python": sys.version,
        "binary_path": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in files
        },
        "helper_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[2]
                / "tests/kaggriculture/native_step_oracle.py"
            ).read_bytes()
        ).hexdigest(),
    }
    if args.mode == "record":
        archive = subprocess.check_output(
            ["git", "archive", BASE_COMMIT, *files],
            cwd=Path(__file__).resolve().parents[2],
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as exported:
            for name, digest in identity["source_sha256"].items():
                member = exported.extractfile(name)
                assert member is not None
                assert hashlib.sha256(member.read()).hexdigest() == digest, (
                    f"golden recording requires pristine {BASE_COMMIT}: {name} differs"
                )
    if args.mode == "benchmark":
        result = {"identity": identity, "benchmark": benchmark(args.repetitions)}
    else:
        actual = trajectory(args.threads)
        if args.mode == "verify":
            expected = json.loads(GOLDEN.read_text())["trajectory"]
            assert actual == expected, {"actual": actual, "expected": expected}
        result = {
            "format": "native-step-byte-golden-v1",
            "base_commit": BASE_COMMIT,
            "seed": SEED,
            "stride": STRIDE,
            "reward": REWARD,
            "configuration": {},
            "native_threads": args.threads,
            "identity": identity,
            "trajectory": actual,
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "identity"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
