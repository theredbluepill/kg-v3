from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_engine_trim", Path(__file__).parents[2] / "scripts/check_engine_trim.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


def fixture() -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    kept = "engine_rs/src/py_random.rs"
    omitted = "engine_rs/src/ffi.rs"
    authored = "engine_rs/tests/replay_parity.rs"
    originals = {kept: b"original\n", omitted: b"ffi\n"}
    current = {kept: originals[kept], authored: b"test\n"}
    manifest = {
        "schema_version": 1,
        "reference_commit": checker.PIN,
        "retained": [
            {
                "path": kept,
                "reference_sha256": checker.sha(originals[kept]),
                "sha256": checker.sha(current[kept]),
                "edits": [],
            }
        ],
        "excluded": [
            {
                "path": omitted,
                "reference_sha256": checker.sha(originals[omitted]),
                "reason": "C5 excluded ABI",
            }
        ],
        "authored": [
            {
                "path": authored,
                "sha256": checker.sha(current[authored]),
                "reason": "replay regression",
            }
        ],
        "non_engine_changes": [
            {"path": "justfile", "reason": "Separate engine package preparation"}
        ],
    }
    return manifest, originals, current


def test_valid_full_inventory_passes() -> None:
    checker.verify(*fixture())


def test_task_authored_inventory_accepts_replay_test() -> None:
    manifest, _, _ = fixture()
    checker.verify_task_authored(manifest["authored"])


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        ([], r"Task 1.1 authored set: missing=.*replay_parity"),
        (
            ["engine_rs/tests/replay_parity.rs", "engine_rs/tests/other.rs"],
            r"Task 1.1 authored set: .*extra=.*other.rs",
        ),
    ],
)
def test_task_authored_inventory_requires_exact_replay_test(
    paths: list[str], message: str
) -> None:
    authored = [
        {"path": path, "sha256": checker.sha(b"test\n"), "reason": "test"}
        for path in paths
    ]
    with pytest.raises(ValueError, match=message):
        checker.verify_task_authored(authored)


def test_declared_line_removal_reconstructs_exact_original_bytes() -> None:
    path = "engine_rs/Cargo.toml"
    before = b"keep\nremove\nkeep2\n"
    after = b"keep\nkeep2\n"
    manifest = {
        "schema_version": 1,
        "reference_commit": checker.PIN,
        "excluded": [],
        "authored": [],
        "non_engine_changes": [],
        "retained": [
            {
                "path": path,
                "reference_sha256": checker.sha(before),
                "sha256": checker.sha(after),
                "edits": [
                    {
                        "start_line": 2,
                        "delete_lines": 1,
                        "insert": "",
                        "reason": "remove excluded target",
                    }
                ],
            }
        ],
    }
    checker.verify(manifest, {path: before}, {path: after})
    manifest["retained"][0]["edits"][0]["start_line"] = 1
    with pytest.raises(ValueError, match="undeclared edit"):
        checker.verify(manifest, {path: before}, {path: after})


@pytest.mark.parametrize(
    "mutation",
    [
        "modified retained bytes",
        "wrong reference hash",
        "undeclared edit despite updated hash",
        "extra excluded module",
        "missing retained file",
        "missing authored file",
        "unaccounted reference file",
        "duplicate entry",
        "unsafe path",
        "unexpected schema key",
    ],
)
def test_drift_is_rejected(mutation: str) -> None:
    manifest, originals, current = fixture()
    kept = manifest["retained"][0]["path"]
    if mutation == "modified retained bytes":
        current[kept] = b"changed\n"
    elif mutation == "wrong reference hash":
        manifest["retained"][0]["reference_sha256"] = "0" * 64
    elif mutation == "undeclared edit despite updated hash":
        current[kept] = b"changed\n"
        manifest["retained"][0]["sha256"] = checker.sha(current[kept])
    elif mutation == "extra excluded module":
        current[manifest["excluded"][0]["path"]] = b"ffi\n"
    elif mutation == "missing retained file":
        del current[kept]
    elif mutation == "missing authored file":
        del current[manifest["authored"][0]["path"]]
    elif mutation == "unaccounted reference file":
        manifest["excluded"].pop()
    elif mutation == "duplicate entry":
        manifest["retained"].append(manifest["retained"][0])
    elif mutation == "unsafe path":
        manifest["retained"][0]["path"] = "engine_rs/../escape"
    else:
        manifest["fallback"] = True
    with pytest.raises(
        ValueError, match=r"hash|edit|inventory|duplicate|path|schema keys"
    ):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    "path",
    [
        "/outside",
        "../escape",
        "a//b",
        "a/./b",
        "a\\b",
        "engine_rs/src/lib.rs",
        ".git/config",
    ],
)
def test_invalid_non_engine_path_is_rejected(path: str) -> None:
    manifest, originals, current = fixture()
    manifest["non_engine_changes"][0]["path"] = path
    with pytest.raises(ValueError, match="path"):
        checker.verify(manifest, originals, current)


def test_duplicate_non_engine_change_is_rejected() -> None:
    manifest, originals, current = fixture()
    manifest["non_engine_changes"].append(manifest["non_engine_changes"][0])
    with pytest.raises(ValueError, match="duplicate"):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    "field", ["retained", "excluded", "authored", "non_engine_changes"]
)
def test_inventory_entry_schema_is_strict(field: str) -> None:
    manifest, originals, current = fixture()
    manifest[field][0]["unexpected"] = "value"
    with pytest.raises(ValueError, match="schema keys"):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize("value", [True, "1", 1.0, None])
def test_schema_version_requires_an_integer(value: object) -> None:
    manifest, originals, current = fixture()
    manifest["schema_version"] = value
    with pytest.raises(ValueError, match="schema_version"):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize("value", ["", "   ", 4, None])
def test_non_engine_reason_is_required(value: object) -> None:
    manifest, originals, current = fixture()
    manifest["non_engine_changes"][0]["reason"] = value
    with pytest.raises(ValueError, match="reason"):
        checker.verify(manifest, originals, current)


def test_retained_rust_cannot_be_edited_even_when_declared() -> None:
    manifest, originals, current = fixture()
    retained = manifest["retained"][0]
    current[retained["path"]] = b"changed\n"
    retained["sha256"] = checker.sha(b"changed\n")
    retained["edits"] = [
        {"start_line": 1, "delete_lines": 1, "insert": "changed\n", "reason": "change"}
    ]
    with pytest.raises(ValueError, match="retained Rust must be unchanged"):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("start_line", True, "start_line"),
        ("start_line", 0, "edit range"),
        ("start_line", 3, "edit range"),
        ("delete_lines", False, "delete_lines"),
        ("delete_lines", -1, "edit range"),
        ("delete_lines", 2, "edit end"),
        ("insert", None, "insert"),
        ("reason", "", "reason"),
        ("unexpected", "value", "schema keys"),
    ],
)
def test_invalid_edit_is_rejected(field: str, value: object, message: str) -> None:
    manifest, originals, current = fixture()
    manifest["retained"][0]["edits"] = [
        {"start_line": 1, "delete_lines": 0, "insert": "", "reason": "test edit"}
    ]
    manifest["retained"][0]["edits"][0][field] = value
    with pytest.raises(ValueError, match=message):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    "path", ["engine_rs/LICENSE", "engine_rs/fixtures/episode.jsonl.gz"]
)
def test_binary_edits_are_rejected(path: str) -> None:
    original = b"\xff\xfe\n"
    manifest = {
        "schema_version": 1,
        "reference_commit": checker.PIN,
        "excluded": [],
        "authored": [],
        "non_engine_changes": [],
        "retained": [
            {
                "path": path,
                "reference_sha256": checker.sha(original),
                "sha256": checker.sha(original),
                "edits": [
                    {"start_line": 1, "delete_lines": 0, "insert": "", "reason": "test"}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match=r"UTF-8|binary fixtures"):
        checker.verify(manifest, {path: original}, {path: original})
