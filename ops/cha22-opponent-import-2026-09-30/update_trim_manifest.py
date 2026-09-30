"""Record Cha22's import in engine_rs/TRIM_MANIFEST.json (idempotent).

Mirrors Task 7.1: each excluded reference path copied into opponents_rs gains a
pointer to its destination, and the non-engine change ledger lists new paths.
Run from the repository root: uv run --offline python <this file>.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_opponent_import", ROOT / "scripts/check_opponent_import.py"
)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)
TAG = "(Cha22 import, 2026-09-30)"


def main() -> None:
    path = ROOT / "engine_rs/TRIM_MANIFEST.json"
    manifest = json.loads(path.read_text())
    excluded = {entry["path"]: entry for entry in manifest["excluded"]}
    copies = {
        **{p: "byte-exact copy" for p in checker.IMPORT_HASHES},
        **{p: "Game-view-adapted copy" for p in checker.ADAPTED_HASHES},
    }
    for reference, kind in copies.items():
        if (
            TAG in excluded[reference]["reason"]
            or "Task 7.1" in excluded[reference]["reason"]
        ):
            continue
        destination = reference.replace("engine_rs/", "opponents_rs/", 1)
        excluded[reference]["reason"] += (
            f"; {kind} imported to {destination} under OPPONENT_MANIFEST.json {TAG}"
        )
    ledger = manifest["non_engine_changes"]
    listed = {entry["path"] for entry in ledger}
    manifest_entries = json.loads((ROOT / checker.MANIFEST).read_text())
    additions = {
        **{
            e["path"]: "preserve pinned Cha22-closure bytes."
            for e in manifest_entries["imported"]
        },
        **{
            e["path"]: "Cha22 closure with Game-view accessor lines only."
            for e in manifest_entries["adapted"]
        },
        **{
            e["path"]: "retain Cha22 upstream Apache-2.0 notices."
            for e in manifest_entries["notices"]
        },
        **{
            e[
                "path"
            ]: "freeze original-submission Cha22 observations and actions (CPython 3.11)."
            for e in manifest_entries["oracle_traces"]
            if checker.CHA22_ORACLE_DIR in e["path"]
        },
        f"{checker.CHA22_ORACLE_DIR}/MANIFEST.json": "Cha22 oracle inventory and coverage.",
    }
    for change, reason in additions.items():
        if change not in listed:
            ledger.append({"path": change, "reason": f"Cha22 import: {reason}"})
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
