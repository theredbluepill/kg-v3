"""Regenerate engine_rs/TRIM_MANIFEST.json for the Task 1.2 merge.

Three-way merge of the manifest at the merge base (90ed86c), the integration
side (Task 1.1b + 2.3, e1458d2) and Task 1.2 (7877c46), following the Task 1.1b
updater pattern (ops/rebuild-2026-09-29/1.1b/update_trim_manifest.py):

- ``retained`` and ``excluded`` must be byte-identical on all three sides and are
  copied unchanged (no vendored bytes or reference inventory change).
- ``authored`` is the ordered union of both sides; every SHA-256 is recomputed
  from the merged working tree, never copied.
- ``reason`` text is taken from whichever side changed it; a path whose reason
  both sides changed must appear in ``BOTH_CHANGED`` (and nothing else may), so
  an unexpected three-way conflict fails loudly instead of being guessed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = "engine_rs/TRIM_MANIFEST.json"
BASE, OURS, THEIRS = "90ed86c", "e1458d2", "7877c46"
BOTH_CHANGED = {
    "scripts/check_engine_trim.py": (
        "Python provenance checker required by Claude review; replaces planned "
        "Node tool. Task 1.1b: also validates the generated-trace manifest, "
        "inventory, hashes, engine pin and 4 MB budget. Task 1.2: extends the "
        "fixed authored inventory to exactly the replay and grammar kernel tests "
        "plus the generated-trace manifest."
    ),
    "tests/tools/test_check_engine_trim.py": (
        "Test-first pytest coverage for provenance and inventory drift. Task 1.1b: "
        "generated-manifest drift cases. Task 1.2: exact authored-set inventory, "
        "omissions, third-file and authored-hash attacks."
    ),
    "docs/rules-parity-coverage.md": (
        "Document current Kaggriculture replay and unit coverage without changing "
        "Orbit coverage. Task 1.1b: live differential parity coverage, divergences "
        "and limits. Task 1.2: scope historical Task 1.1 counts and record "
        "independent grammar and kernel qualification boundaries."
    ),
    "cookbook/log.md": (
        "Prepend material Task 1.1 adaptation record. Task 1.1b: prepend the live "
        "parity record. Task 1.2: prepend the native grammar record."
    ),
    "cookbook/references/index.md": (
        "List the Task 1.1b parity Reference; keep the native semantics Reference "
        "description aligned with current rebuild scope (Task 1.2)."
    ),
}


def load(rev: str) -> dict:
    out = subprocess.run(
        ["git", "show", f"{rev}:{PATH}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return json.loads(out)


def pick(before: str | None, mine: str | None, other: str | None) -> str | None:
    """Three-way pick of one reason; None means both sides changed it."""
    if mine == other or other is None or other == before:
        return mine
    if mine is None or mine == before:
        return other
    return None


def merge_entries(key: str, sides: tuple[dict, dict, dict]) -> list[dict]:
    base, ours, theirs = ({e["path"]: e for e in side[key]} for side in sides)
    for path in base:
        if path not in ours or path not in theirs:
            raise SystemExit(f"{key}: {path} removed on one side; resolve by hand")
    merged: list[dict] = []
    both: set[str] = set()
    for path in [*ours, *(p for p in theirs if p not in ours)]:
        reason = pick(
            base.get(path, {}).get("reason"),
            ours.get(path, {}).get("reason"),
            theirs.get(path, {}).get("reason"),
        )
        if reason is None:
            both.add(path)
            reason = BOTH_CHANGED[path]
        entry = {"path": path}
        if key == "authored":
            # Hashes are recomputed from the merged tree, never copied.
            entry["sha256"] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        entry["reason"] = reason
        merged.append(entry)
    if key == "non_engine_changes" and both != set(BOTH_CHANGED):
        raise SystemExit(f"{key}: both-changed {sorted(both)} != declared")
    if key == "authored" and both:
        raise SystemExit(f"authored: both sides changed {sorted(both)}")
    return merged


def main() -> None:
    base, ours, theirs = load(BASE), load(OURS), load(THEIRS)
    for key in ("schema_version", "reference_commit", "retained", "excluded"):
        if not base[key] == ours[key] == theirs[key]:
            raise SystemExit(f"{key} differs between sides; vendored inventory drift")
    manifest = {
        "schema_version": ours["schema_version"],
        "reference_commit": ours["reference_commit"],
        "retained": ours["retained"],
        "excluded": ours["excluded"],
        "authored": merge_entries("authored", (base, ours, theirs)),
        "non_engine_changes": merge_entries(
            "non_engine_changes", (base, ours, theirs)
        ),
    }
    (ROOT / PATH).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
