"""Replay thirteen frozen observations through original R04, with read-only probes."""

from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "oracle_generator", ROOT / "scripts/kaggriculture_parity/generate_traces.py"
)
assert SPEC is not None
assert SPEC.loader is not None
GEN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEN
SPEC.loader.exec_module(GEN)


def main() -> None:
    path = ROOT / "opponents_rs/fixtures/oracle/oracle-00-starter-vs-r04.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        records = [json.loads(line) for line in handle]
    policy = GEN.SiblingPolicy("r04")
    module = policy.modules[policy.name]
    original_jobs = module.build_jobs
    original_pairs = module.global_assignment_pairs
    current: dict[str, Any] = {}

    def traced_jobs(state: Any, world: Any) -> Any:
        jobs = original_jobs(state, world)
        current.update(
            {
                "units": world.units,
                "targets_before": [
                    [unit, *point] for unit, point in state.targets.items()
                ],
                "jobs": [
                    {
                        "tile": job.tile,
                        "actions": job.actions,
                        "prio": job.prio,
                        "need": job.need,
                        "kind": job.kind,
                    }
                    for job in jobs
                ],
            }
        )
        return jobs

    def traced_pairs(pairs: Any, n_units: int, n_jobs: int) -> Any:
        selected = original_pairs(pairs, n_units, n_jobs)
        current["pairs"] = pairs
        current["selected"] = selected
        return selected

    module.build_jobs = traced_jobs
    module.global_assignment_pairs = traced_pairs
    output = []
    public = records[0]["initial"]["public"]
    privates = records[0]["initial"]["privates"]
    try:
        for record in records[1:14]:
            current = {"step": record["from_step"]}
            obs = {"player": 1, **public, "private": privates[1]}
            action = policy(obs, records[0]["configuration"], random.Random(0)).action
            assert action == record["actions"][1], (
                f"Python replay drift at {record['from_step']}"
            )
            state = module._STATES[1]
            current.update(
                {
                    "action": action,
                    "roles": [[*point, role] for point, role in state.roles.items()],
                    "anchors": [
                        [unit, *point] for unit, point in state.anchors.items()
                    ],
                    "targets": [
                        [unit, *point] for unit, point in state.targets.items()
                    ],
                }
            )
            output.append(current)
            public, privates = record["expected"], record["privates"]
        native = json.loads(
            (Path(__file__).parent / "r04-native-debug.json").read_text()
        )
        sequential = GEN.SiblingPolicy("r04")
        sequential_module = sequential.modules[sequential.name]
        anchors_function = sequential_module.compute_anchors

        def linear_sum(items: Any) -> Any:
            total = 0
            for item in items:
                total += item
            return total

        sequential_module.compute_anchors = FunctionType(
            anchors_function.__code__,
            {**anchors_function.__globals__, "sum": linear_sum},
            anchors_function.__name__,
            anchors_function.__defaults__,
            anchors_function.__closure__,
        )
        counterfactual = []
        public = records[0]["initial"]["public"]
        privates = records[0]["initial"]["privates"]
        try:
            for record, native_step in zip(records[1:14], native, strict=True):
                obs = {"player": 1, **public, "private": privates[1]}
                action = sequential(
                    obs, records[0]["configuration"], random.Random(0)
                ).action
                state = sequential_module._STATES[1]
                anchors = [[unit, *point] for unit, point in state.anchors.items()]
                counterfactual.append(
                    {
                        "step": record["from_step"],
                        "action": action,
                        "equals_native_action": action == native_step["native_action"],
                        "anchors_equal_native": anchors
                        == native_step["debug"]["anchors"],
                    }
                )
                public, privates = record["expected"], record["privates"]
        finally:
            sequential.close()
        result = {
            "source_commit": GEN.SIBLING_COMMIT,
            "source_sha256": GEN.SIBLING_ENTRY_SHA256["r04"],
            "fixture_sha256": GEN.sha256_file(path),
            "globals": {
                "GLOBAL_ASSIGN": module.GLOBAL_ASSIGN,
                "ASSIGN_LOOKAHEAD": module.ASSIGN_LOOKAHEAD,
                "ASSIGN_JOINT_ROUTES": module.ASSIGN_JOINT_ROUTES,
            },
            "observations_replayed": len(output),
            "actions_equal_fixture": len(output),
            "steps": output,
            "counterfactual_sequential_sum_only_in_compute_anchors": counterfactual,
            "floating_reduction": {
                "python_version": sys.version,
                "python_sum_25_times_0_35": sum([0.35] * 25),
                "sequential_sum_25_times_0_35": linear_sum([0.35] * 25),
                "python_per_5_hands": sum([0.35] * 25) / 5,
                "sequential_per_5_hands": linear_sum([0.35] * 25) / 5,
                "sequential_15_weights": linear_sum([0.35] * 15),
            },
        }
        (Path(__file__).parent / "r04-python-debug.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
