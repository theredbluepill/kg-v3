"""The replay manifest updater admits only the declared, non-engine task delta."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "replay_trim_updater",
    Path(__file__).parents[2] / "ops/rebuild-2026-09-29/7.3/update_trim_manifest.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
updater = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(updater)


def baseline() -> dict:
    return {
        "schema_version": 1,
        "reference_commit": "unchanged",
        "retained": [{"path": "engine_rs/src/lib.rs", "sha256": "frozen"}],
        "excluded": [{"path": "engine_rs/src/ffi.rs", "reason": "excluded"}],
        "authored": [{"path": "engine_rs/tests/replay_parity.rs", "sha256": "frozen"}],
        "non_engine_changes": [
            {"path": "docs/rl-api-specs.md", "reason": "Earlier declaration"}
        ],
    }


def test_only_declared_delta_and_idempotent_receipt_extension() -> None:
    before = baseline()
    paths = {"src/kaggriculture/replay_export.rs", f"{updater.RECEIPTS}/red.log"}
    first = updater.regenerate(before, before, paths)
    assert updater.regenerate(before, first, paths) == first
    extended = updater.regenerate(
        before, first, paths | {f"{updater.RECEIPTS}/green.log"}
    )
    for key in (
        "schema_version",
        "reference_commit",
        "retained",
        "excluded",
        "authored",
    ):
        assert extended[key] == before[key]
    entries = {
        entry["path"]: entry["reason"] for entry in extended["non_engine_changes"]
    }
    assert paths <= entries.keys()
    assert entries["docs/rl-api-specs.md"].startswith("Task 7.3:")
    assert "Earlier declaration" in entries["docs/rl-api-specs.md"]
    assert entries[f"{updater.RECEIPTS}/red.log"].startswith("Task 7.3:")
    assert json.dumps(before) == json.dumps(baseline())


@pytest.mark.parametrize("section", ["retained", "excluded", "authored"])
def test_frozen_inventory_drift_is_rejected(section: str) -> None:
    before = baseline()
    changed = copy.deepcopy(before)
    changed[section][0]["path"] = "engine_rs/changed"
    with pytest.raises(ValueError, match="unexpected manifest input"):
        updater.regenerate(before, changed, set())


def test_prior_generated_reason_drift_is_rejected() -> None:
    before = baseline()
    first = updater.regenerate(before, before, {f"{updater.RECEIPTS}/red.log"})
    first["non_engine_changes"][-1]["reason"] = "unreviewed replacement"
    with pytest.raises(ValueError, match="unexpected manifest input"):
        updater.regenerate(before, first, {f"{updater.RECEIPTS}/red.log"})


@pytest.mark.parametrize(
    "path",
    [
        "python/owl/model/kaggriculture.py",
        "engine_rs/src/lib.rs",
        "ops/rebuild-2026-09-29/7.3/../escape",
        "/outside",
        "ops/rebuild-2026-09-29/7.3//red.log",
    ],
)
def test_undeclared_or_unsafe_worktree_change_is_rejected(path: str) -> None:
    before = baseline()
    with pytest.raises(ValueError, match=r"undeclared task path|unsafe path"):
        updater.regenerate(before, before, {path})


def test_missing_previous_receipt_is_rejected() -> None:
    before = baseline()
    first = updater.regenerate(before, before, {f"{updater.RECEIPTS}/red.log"})
    with pytest.raises(ValueError, match="missing prior receipt"):
        updater.regenerate(before, first, set())


def test_pre_deadline_isolation_manifest_is_an_explicit_migration() -> None:
    before = baseline()
    path = "tests/tools/test_observation_oracle_custody.py"
    previous = updater.regenerate(before, before, set())
    previous["non_engine_changes"] = [
        entry for entry in previous["non_engine_changes"] if entry["path"] != path
    ]
    updated = updater.regenerate(before, previous, {path})
    entries = {
        entry["path"]: entry["reason"] for entry in updated["non_engine_changes"]
    }
    assert entries[path].startswith("Task 7.3:")
    assert "deadline" in entries[path]


def test_migration_does_not_admit_other_missing_declarations() -> None:
    before = baseline()
    previous = updater.regenerate(before, before, set())
    previous["non_engine_changes"] = [
        entry
        for entry in previous["non_engine_changes"]
        if entry["path"] != "src/kaggriculture/replay_export.rs"
    ]
    with pytest.raises(ValueError, match="unexpected manifest input"):
        updater.regenerate(before, previous, set())
