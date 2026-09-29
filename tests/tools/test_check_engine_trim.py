from __future__ import annotations

import importlib.util
import json
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


def test_task_authored_inventory_accepts_replay_test_and_generated_manifest() -> None:
    authored = [
        {"path": path, "sha256": checker.sha(b"x"), "reason": "authored"}
        for path in ("engine_rs/tests/replay_parity.rs", checker.GENERATED_MANIFEST)
    ]
    checker.verify_task_authored(authored)


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        ([], r"Task 1.1 authored set: missing=.*MANIFEST.json.*replay_parity"),
        (
            [
                "engine_rs/tests/replay_parity.rs",
                checker.GENERATED_MANIFEST,
                "engine_rs/tests/other.rs",
            ],
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


PIN_PAIR = ("1.32.7", "a" * 64)


def generated_fixture() -> tuple[dict[str, Any], dict[str, bytes]]:
    traces = {"gen-a.jsonl.gz": b"trace a", "divergence-b.jsonl.gz": b"trace b"}
    entries: list[dict[str, Any]] = [
        {
            "path": name,
            "sha256": checker.sha(data),
            "bytes": len(data),
            "seed": 1,
            "policies": ["random", "edge"],
            "policy_seed": 2,
            "config_variant": "default",
            "transitions": 719,
            "rejected": 0,
        }
        for name, data in traces.items()
    ]
    entries[1]["probe"] = '{"farmer":["PLANT",["WHEAT"]]}'
    entries[1]["expected_divergence"] = {
        "line": 3,
        "from_step": 2,
        "kind": "rust_accepted",
        "field": "step",
        "reason": "D2: documented",
    }
    manifest = {
        "schema_version": 1,
        "format": checker.TRACE_FORMAT,
        "generator": "scripts/kaggriculture_parity/generate_traces.py",
        "kaggle_environments_version": PIN_PAIR[0],
        "python_engine_sha256": PIN_PAIR[1],
        "traces": entries,
    }
    return manifest, traces


def _verify_generated(manifest: dict[str, Any], traces: dict[str, bytes]) -> int:
    return int(
        checker.verify_generated(json.dumps(manifest).encode(), traces, PIN_PAIR)
    )


def test_generated_manifest_pins_every_trace() -> None:
    assert _verify_generated(*generated_fixture()) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("edited trace", "trace hash"),
        ("wrong size", "trace size"),
        ("unlisted trace", "generated trace inventory"),
        ("missing trace", "is missing"),
        ("other engine", "Cargo engine pin"),
        ("over budget", "budget"),
        ("no reason", "reason"),
        ("probe without expectation", "go together"),
        ("nested path", "plain"),
        ("duplicate", "duplicate"),
        ("extra key", "schema keys"),
    ],
)
def test_generated_manifest_drift_is_rejected(mutation: str, message: str) -> None:
    manifest, traces = generated_fixture()
    first = manifest["traces"][0]
    if mutation == "edited trace":
        traces[first["path"]] = b"trace A"
    elif mutation == "wrong size":
        first["bytes"] += 1
    elif mutation == "unlisted trace":
        traces["gen-c.jsonl.gz"] = b"unlisted"
    elif mutation == "missing trace":
        del traces[first["path"]]
    elif mutation == "other engine":
        manifest["python_engine_sha256"] = "b" * 64
    elif mutation == "over budget":
        traces[first["path"]] = b"x" * (checker.GENERATED_BUDGET_BYTES + 1)
        first["sha256"] = checker.sha(traces[first["path"]])
        first["bytes"] = len(traces[first["path"]])
    elif mutation == "no reason":
        manifest["traces"][1]["expected_divergence"]["reason"] = " "
    elif mutation == "probe without expectation":
        del manifest["traces"][1]["expected_divergence"]
    elif mutation == "nested path":
        first["path"] = "sub/gen-a.jsonl.gz"
    elif mutation == "duplicate":
        manifest["traces"].append(dict(first))
    else:
        first["note"] = "unexpected"
    with pytest.raises(ValueError, match=message):
        _verify_generated(manifest, traces)


def test_generated_traces_are_split_from_engine_inventory() -> None:
    current = {
        "engine_rs/src/lib.rs": b"lib",
        checker.GENERATED_MANIFEST: b"{}",
        f"{checker.GENERATED_DIR}/gen-a.jsonl.gz": b"trace",
    }
    engine, generated = checker.split_generated(current)
    assert set(engine) == {"engine_rs/src/lib.rs", checker.GENERATED_MANIFEST}
    assert generated == {"gen-a.jsonl.gz": b"trace"}


def test_engine_pin_reads_cargo_metadata() -> None:
    cargo = (Path(__file__).parents[2] / "engine_rs/Cargo.toml").read_bytes()
    assert checker.engine_pin(cargo) == (
        "1.32.7",
        "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e",
    )
