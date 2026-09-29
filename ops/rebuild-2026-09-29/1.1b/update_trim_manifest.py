"""Refresh Task 1.1b authored hashes and inventory in engine_rs/TRIM_MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "engine_rs/TRIM_MANIFEST.json"
AUTHORED = {
    "engine_rs/tests/replay_parity.rs": (
        "Direct native-reset parity replay over official, committed generated and "
        "KAGG_PARITY_TRACES traces; comparator regressions; perturbation and "
        "expected-divergence checks (Task 1.1b)"
    ),
    "engine_rs/fixtures/generated/MANIFEST.json": (
        "Pins each Task 1.1b trace generated live from kaggle-environments 1.32.7 "
        "(SHA-256, size, policies, seeds, config, documented expected divergences)"
    ),
}
NON_ENGINE = {
    "scripts/kaggriculture_parity/generate_traces.py": (
        "Task 1.1b hash-guarded live trace generator over Kaggle's Python engine"
    ),
    "scripts/kaggriculture_parity/sweep.py": (
        "Task 1.1b local differential sweep runner and JSON summary"
    ),
    "tests/scripts/test_kaggriculture_parity.py": (
        "Task 1.1b hash guard, trace-format round trip, coverage and optional live "
        "regeneration tests"
    ),
    "ops/rebuild-2026-09-29/1.1b": (
        "Task 1.1b receipts: sweep summary, first failing traces and command logs"
    ),
    "cookbook/references/live-differential-parity-checks-the-rust-kernel.md": (
        "Task 1.1b Reference for the live differential parity check and findings"
    ),
    "cookbook/references/index.md": "List the Task 1.1b parity Reference",
}
APPEND = {
    "scripts/check_engine_trim.py": (
        " Task 1.1b: also validates the generated-trace manifest, inventory, "
        "hashes, engine pin and 4 MB budget."
    ),
    "tests/tools/test_check_engine_trim.py": (
        " Task 1.1b: generated-manifest drift cases."
    ),
    "docs/rules-parity-coverage.md": (
        " Task 1.1b: live differential parity coverage, divergences and limits."
    ),
    "cookbook/log.md": " Task 1.1b: prepend the live parity record.",
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["authored"] = [
        {
            "path": path,
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "reason": reason,
        }
        for path, reason in AUTHORED.items()
    ]
    changes = manifest["non_engine_changes"]
    known = {entry["path"] for entry in changes}
    for entry in changes:
        suffix = APPEND.get(entry["path"])
        if suffix and not entry["reason"].endswith(suffix):
            entry["reason"] += suffix
    changes += [
        {"path": path, "reason": reason}
        for path, reason in NON_ENGINE.items()
        if path not in known
    ]
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
