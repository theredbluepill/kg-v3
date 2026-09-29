"""Register Task 7.3 outside the frozen engine, from one exact integration base.

The retained/excluded/authored inventories are never recomputed or edited. Input
must be the original manifest or this generator's exact output, allowing an
earlier subset of receipt files while the task is still running. Removing an
already registered receipt or changing an earlier reason is an error.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "0b8cf98ef57fc49a329dca4c8368c630586c4752"
PATH = "engine_rs/TRIM_MANIFEST.json"
RECEIPTS = "ops/rebuild-2026-09-29/7.3"
DEADLINE_TEST = "tests/tools/test_observation_oracle_custody.py"
DECLARED = {
    "src/kaggriculture/mod.rs": (
        "register stateless replay PyO3 entry points in the root extension"
    ),
    "src/kaggriculture/replay_export.rs": (
        "native seed replay, Kaggle envelope, strict import and "
        "divergence/evidence checks"
    ),
    "src/kaggriculture/replay_export_tests.rs": (
        "test-first native replay contract and negative controls"
    ),
    "python/owl/rs.pyi": "type the two JSON-text replay extension functions",
    "python/owl/kaggriculture/replay_export.py": (
        "hash-pinned framework schema, deterministic recorder and custody writer"
    ),
    "tests/kaggriculture/test_replay_export.py": (
        "synthetic recorder, seed custody and explicit incomplete/error tests"
    ),
    "tests/kaggriculture/test_replay_export_oracles.py": (
        "independent official fixture and real-framework round trips "
        "with mutation controls"
    ),
    "tests/kaggriculture/test_replay_export_integration.py": (
        "explicit Task 1.4 binding-dependent evaluation and seed-lifecycle skips"
    ),
    "tests/tools/test_replay_trim_manifest.py": (
        "test-first updater inventory, idempotence and drift rejection"
    ),
    DEADLINE_TEST: (
        "reset each custody unit test's deadline after full default replay tests; "
        "production timeout enforcement remains unchanged"
    ),
    "docs/rl-api-specs.md": (
        "document stateless replay text APIs without an environment substitute"
    ),
    "docs/rules-parity-coverage.md": (
        "document replay/export oracle coverage and remaining limits"
    ),
    "cookbook/references/native-replay-export-preserves-kaggle-episodes.md": (
        "record the replay adaptation, independent evidence and open bindings"
    ),
    "cookbook/references/rebuild-data-preparation-preserves-replay-identity.md": (
        "scope the earlier replay deferral to preparation and link Task 7.3 "
        "runtime qualification"
    ),
    "cookbook/references/index.md": "index the native replay-export Reference",
    "cookbook/log.md": "prepend the Task 7.3 adaptation record",
}
RECEIPT_REASON = (
    "Task 7.3: bounded implementation/check receipts, source custody, updater and "
    "test-first or mutation evidence; no vendored engine content"
)


def _validate_path(path: str) -> None:
    if (
        "\\" in path
        or "\x00" in path
        or any(part in ("", ".", "..", ".git") for part in path.split("/"))
    ):
        raise ValueError(f"unsafe path: {path!r}")


def _receipt(path: str) -> bool:
    return path.startswith(f"{RECEIPTS}/")


def _build(
    baseline: dict, receipts: set[str], *, include_deadline_test: bool = True
) -> dict:
    result = copy.deepcopy(baseline)
    entries = result["non_engine_changes"]
    by_path = {entry["path"]: entry for entry in entries}
    for path, purpose in DECLARED.items():
        if path == DEADLINE_TEST and not include_deadline_test:
            continue
        reason = f"Task 7.3: {purpose}."
        if path in by_path:
            by_path[path]["reason"] = (
                f"{reason} Prior declaration: {by_path[path]['reason']}"
            )
        else:
            entries.append({"path": path, "reason": reason})
    entries.extend(
        {"path": path, "reason": RECEIPT_REASON} for path in sorted(receipts)
    )
    return result


def regenerate(baseline: dict, current: dict, changed_paths: set[str]) -> dict:
    """Return the declared update, rejecting frozen/input/worktree drift."""
    for path in changed_paths:
        _validate_path(path)
        if path != PATH and path not in DECLARED and not _receipt(path):
            raise ValueError(f"undeclared task path: {path}")
    receipts = {path for path in changed_paths if _receipt(path)}
    if current != baseline:
        prior = {
            entry["path"]
            for entry in current["non_engine_changes"]
            if _receipt(entry["path"])
        }
        # One precise prior generated form predates the broad-suite discovery
        # that collection-time oracle deadlines expire during full replay tests.
        # No other omitted/revised fixed declaration is a recognized migration.
        if current not in (
            _build(baseline, prior),
            _build(baseline, prior, include_deadline_test=False),
        ):
            raise ValueError(
                "unexpected manifest input: not base or generated Task 7.3"
            )
        if not prior <= receipts:
            raise ValueError(f"missing prior receipt: {sorted(prior - receipts)}")
    return _build(baseline, receipts)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout


def changed_paths() -> set[str]:
    """Include ignored receipt logs, excluding generated Python bytecode only."""
    paths = {
        path.decode("utf-8")
        for data in (
            _git("diff", "--name-only", "-z", BASE),
            _git("ls-files", "--others", "--exclude-standard", "-z"),
        )
        for path in data.split(b"\0")
        if path
    }
    for file in (ROOT / RECEIPTS).rglob("*"):
        if "__pycache__" in file.parts or file.suffix == ".pyc":
            continue
        if file.is_symlink():
            raise ValueError(f"receipt symlink is not allowed: {file}")
        if file.is_file():
            paths.add(file.relative_to(ROOT).as_posix())
    return paths


def main() -> None:
    baseline = json.loads(_git("show", f"{BASE}:{PATH}"))
    current = json.loads((ROOT / PATH).read_text(encoding="utf-8"))
    manifest = regenerate(baseline, current, changed_paths())
    (ROOT / PATH).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "Task 7.3: registered non-engine changes; retained/excluded/authored "
        "inventories unchanged"
    )


if __name__ == "__main__":
    main()
