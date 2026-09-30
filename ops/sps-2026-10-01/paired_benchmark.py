"""Alternate independent old/new extension processes for equal-work CPU timing."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python-source", type=Path, required=True)
    parser.add_argument("--candidate-python-source", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1:
        raise ValueError("pairs must be positive")
    sources = {
        "baseline": args.baseline_python_source.resolve(),
        "candidate": args.candidate_python_source.resolve(),
    }
    runner = Path(__file__).with_name("native_benchmark.py")
    samples: list[dict[str, Any]] = []
    identities: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="kg-native-paired-") as tmp:
        for pair in range(args.pairs):
            order = ["baseline", "candidate"]
            if pair % 2:
                order.reverse()
            for arm in order:
                output = Path(tmp) / f"{pair}-{arm}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "benchmark",
                        "--repetitions",
                        "1",
                        "--output",
                        str(output),
                    ],
                    env=dict(os.environ, PYTHONPATH=str(sources[arm])),
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                receipt = json.loads(output.read_text())
                identity = receipt["identity"]
                if arm in identities and identities[arm] != identity:
                    raise ValueError(f"{arm} source/binary changed during timing")
                identities[arm] = identity
                samples.append({"pair": pair, "arm": arm, **receipt["benchmark"]})
    for key in ("actions_sha256", "final_sha256"):
        if len({sample[key] for sample in samples}) != 1:
            raise ValueError(f"before/after work differs: {key}")
    medians = {
        arm: statistics.median(
            sample["median_step_seconds"] for sample in samples if sample["arm"] == arm
        )
        for arm in sources
    }
    result = {
        "identities": identities,
        "samples": samples,
        "median_step_seconds": medians,
        "median_speed_ratio": medians["baseline"] / medians["candidate"],
        "median_time_reduction_percent": 100
        * (1 - medians["candidate"] / medians["baseline"]),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("identities", "samples")
            }
        )
    )


if __name__ == "__main__":
    main()
