"""Register Task 7.1's stopped import diagnostic; preserve all engine entries.

Unlike a completed import, no excluded reason may claim an opponents_rs copy.
Accept only the exact integration input or this script's exact output.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "b8747b6e8acece5f561d09a75bb914364a60ac05"
MANIFEST = "engine_rs/TRIM_MANIFEST.json"
OPS = "ops/rebuild-2026-09-29/7.1"
CHANGES = {
    "docs/rules-parity-coverage.md": (
        "document the compile blocker and zero opponent coverage."
    ),
    "cookbook/references/frozen-engine-api-blocks-standalone-opponent-import.md": (
        "record the verified public-API blocker and reopening condition."
    ),
    "cookbook/references/index.md": "index the stopped-import Reference.",
    "cookbook/log.md": "prepend the stopped-import finding.",
    **{
        f"{OPS}/{name}": reason
        for name, reason in {
            "plan.md": "separate diagnostic expectations from actual results.",
            "results.md": "report actual commands, scope, coverage and remaining work.",
            "native-api-probe.md": "describe the compile failure and exact reproducer.",
            "native-api-probe.json": (
                "pin compile inputs, support source and compiler-log hash."
            ),
            "native-api-probe-red.log": (
                "retain the compiler's inaccessible-item diagnostics."
            ),
            "native-api-probe-red.exit": "retain compile exit status 101.",
            "commands.json": (
                "record requested final commands and actual exit statuses."
            ),
            "opponents-test.log": "record missing crate after the mandated stop.",
            "engine-test.log": "record unchanged engine regression results.",
            "engine-trim.log": "record the engine custody check.",
            "opponent-import.log": "record missing opponent checker after the stop.",
            "targeted-tests.log": (
                "record requested pytest collection failure after the stop."
            ),
            "prepare.log": "record the requested repository preparation result.",
            "closure.log": "record bookkeeping, documentation and final byte checks.",
            "python-oracle-source-audit.json": (
                "record read-only original-submission hashes and notice gaps."
            ),
            "update_trim_manifest.py": (
                "idempotently register only non-engine diagnostic changes."
            ),
            "test_update_trim_manifest.py": (
                "test preservation, idempotency and unexpected-input refusal."
            ),
            "updater-red.log": "retain the test-first missing-updater failure.",
            "updater-green.log": (
                "record bookkeeping test results after implementation."
            ),
        }.items()
    },
}


def render(current: bytes, baseline: bytes, changes: dict[str, str]) -> bytes:
    manifest = json.loads(baseline)
    entries = manifest["non_engine_changes"]
    existing = {entry["path"]: entry for entry in entries}
    for path, reason in changes.items():
        if path in existing:
            existing[path]["reason"] += f" Task 7.1: {reason}"
        else:
            entries.append({"path": path, "reason": f"Task 7.1: {reason}"})
    result = (json.dumps(manifest, indent=2) + "\n").encode()
    if current not in (baseline, result):
        raise ValueError("unexpected trim manifest input; refusing to overwrite")
    return result


def main() -> None:
    baseline = subprocess.run(
        ["git", "show", f"{BASE}:{MANIFEST}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    target = ROOT / MANIFEST
    target.write_bytes(render(target.read_bytes(), baseline, CHANGES))
    print("Task 7.1: non-engine diagnostic inventory updated; engine entries unchanged")


if __name__ == "__main__":
    main()
