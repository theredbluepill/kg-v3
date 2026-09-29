"""Retire the Task 1.2 grammar bridge registration in engine_rs/TRIM_MANIFEST.json.

Contract v4.1: at the first production root -> engine dependency (Task 1.3's
Cargo.toml edge), kernel acceptance/replay-state tests move into root
integration (src/kaggriculture/grammar_kernel_tests.rs) and the temporary
engine_rs/tests/grammar_kernel.rs plus its authored registration are removed.
Retained kernel bytes and their hashes are untouched. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "engine_rs/TRIM_MANIFEST.json"
BRIDGE = "engine_rs/tests/grammar_kernel.rs"
NON_ENGINE = {
    "src/kaggriculture/grammar_kernel_tests.rs": (
        "Task 1.3: root integration home of the nine kernel acceptance/replay-state "
        "tests moved from the retired engine grammar bridge (contract v4.1)."
    ),
    "src/kaggriculture/mod.rs": (
        "Task 1.3: register the root grammar kernel acceptance tests."
    ),
    "ops/rebuild-2026-09-29/1.3": (
        "Task 1.3 receipts, including this grammar-bridge retirement generator."
    ),
}
APPEND = {
    "scripts/check_engine_trim.py": (
        " Task 1.3: authored set drops the retired grammar bridge."
    ),
    "tests/tools/test_check_engine_trim.py": (
        " Task 1.3: retired-bridge rejection cases."
    ),
    "src/kaggriculture/grammar_tests.rs": (
        " Task 1.3: root package only after the engine bridge retired."
    ),
    "docs/rules-parity-coverage.md": " Task 1.3: grammar bridge retirement.",
    "cookbook/log.md": " Task 1.3: prepend the bridge retirement record.",
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["authored"] = [
        entry for entry in manifest["authored"] if entry["path"] != BRIDGE
    ]
    changes = manifest["non_engine_changes"]
    known = {entry["path"] for entry in changes}
    for entry in changes:
        suffix = APPEND.get(entry["path"])
        if suffix and not entry["reason"].endswith(suffix):
            entry["reason"] += suffix
    missing = sorted(set(APPEND) - known)
    if missing:
        raise SystemExit(f"non_engine_changes lacks expected paths: {missing}")
    changes += [
        {"path": path, "reason": reason}
        for path, reason in NON_ENGINE.items()
        if path not in known
    ]
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
