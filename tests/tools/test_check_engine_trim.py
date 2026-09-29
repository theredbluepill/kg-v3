from __future__ import annotations

import importlib.util
import json
import re
import sys
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
    authored = (
        "engine_rs/tests/replay_parity.rs",
        "engine_rs/tests/grammar_kernel.rs",
    )
    originals = {kept: b"original\n", omitted: b"ffi\n"}
    current = {kept: originals[kept], **{path: b"test\n" for path in authored}}
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
                "path": path,
                "sha256": checker.sha(current[path]),
                "reason": "authored kernel regression",
            }
            for path in authored
        ],
        "non_engine_changes": [
            {"path": "justfile", "reason": "Separate engine package preparation"}
        ],
    }
    return manifest, originals, current


def test_valid_full_inventory_passes() -> None:
    checker.verify(*fixture())


def test_task_authored_inventory_accepts_two_tests_and_generated_manifest() -> None:
    authored = [
        {"path": path, "sha256": checker.sha(b"x"), "reason": "authored"}
        for path in (
            "engine_rs/tests/replay_parity.rs",
            "engine_rs/tests/grammar_kernel.rs",
            checker.GENERATED_MANIFEST,
        )
    ]
    checker.verify_task_authored(authored)


@pytest.mark.parametrize(
    ("paths", "message"),
    [
        (
            [],
            r"Task 1.2 authored set: missing=.*MANIFEST.json.*grammar_kernel"
            r".*replay_parity",
        ),
        (
            ["engine_rs/tests/replay_parity.rs", checker.GENERATED_MANIFEST],
            r"Task 1.2 authored set: missing=.*grammar_kernel",
        ),
        (
            ["engine_rs/tests/grammar_kernel.rs", checker.GENERATED_MANIFEST],
            r"Task 1.2 authored set: missing=.*replay_parity",
        ),
        (
            ["engine_rs/tests/replay_parity.rs", "engine_rs/tests/grammar_kernel.rs"],
            r"Task 1.2 authored set: missing=.*generated/MANIFEST.json",
        ),
        (
            [
                "engine_rs/tests/replay_parity.rs",
                "engine_rs/tests/grammar_kernel.rs",
                checker.GENERATED_MANIFEST,
                "engine_rs/tests/other.rs",
            ],
            r"Task 1.2 authored set: .*extra=.*other.rs",
        ),
    ],
)
def test_task_authored_inventory_requires_exact_authored_set(
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


def test_retained_file_cannot_be_edited_even_when_declared() -> None:
    manifest, originals, current = fixture()
    retained = manifest["retained"][0]
    current[retained["path"]] = b"changed\n"
    retained["sha256"] = checker.sha(b"changed\n")
    retained["edits"] = [
        {"start_line": 1, "delete_lines": 1, "insert": "changed\n", "reason": "change"}
    ]
    with pytest.raises(ValueError, match="retained file must be unchanged"):
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


# --- Task 1.1 check()-level regressions against the real pinned reference ---

_REPO = Path(__file__).parents[2]
_LOCK = "engine_rs/Cargo.lock"
_VENDORED = "engine_rs/VENDORED_FROM.md"


@pytest.fixture(scope="module")
def reference() -> dict[str, bytes]:
    return checker.reference_files(_REPO)


@pytest.fixture
def trimmed(
    tmp_path: Path, reference: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A copy of the committed engine package whose reference comes from the pin."""
    for relative, data in checker._current_files(_REPO).items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = _REPO / "engine_rs/TRIM_MANIFEST.json"
    (tmp_path / "engine_rs/TRIM_MANIFEST.json").write_bytes(manifest.read_bytes())
    monkeypatch.setattr(checker, "reference_files", lambda _root: reference)
    return tmp_path


def _declare(root: Path, path: str, current: bytes, edits: list[Any]) -> None:
    """Write new bytes and update the manifest so they are fully declared."""
    (root / path).write_bytes(current)
    manifest_path = root / "engine_rs/TRIM_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (entry,) = [e for e in manifest["retained"] if e["path"] == path]
    entry["sha256"] = checker.sha(current)
    entry["edits"] = edits
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _whole_file_edit(original: bytes, current: bytes) -> list[Any]:
    lines = len(original.decode("utf-8").splitlines())
    return [
        {
            "start_line": 1,
            "delete_lines": lines,
            "insert": current.decode("utf-8"),
            "reason": "declared rewrite",
        }
    ]


def test_committed_package_passes_check() -> None:
    checker.check(_REPO)


def test_check_entry_point_passes_on_unmodified_copy(trimmed: Path) -> None:
    checker.check(trimmed)


@pytest.mark.parametrize("name", ["replay_parity.rs", "grammar_kernel.rs"])
def test_check_rejects_omitted_authored_test(trimmed: Path, name: str) -> None:
    """Deleting both the test and its declaration cannot weaken the inventory."""
    path = f"engine_rs/tests/{name}"
    (trimmed / path).unlink()
    manifest_path = trimmed / "engine_rs/TRIM_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["authored"] = [
        entry for entry in manifest["authored"] if entry["path"] != path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Task 1.2 authored set: missing=.*{name}"):
        checker.check(trimmed)


def test_check_rejects_self_declared_third_authored_source(trimmed: Path) -> None:
    path = "engine_rs/tests/third.rs"
    data = b"// Extra authored source is outside the fixed inventory.\n"
    (trimmed / path).write_bytes(data)
    manifest_path = trimmed / "engine_rs/TRIM_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["authored"].append(
        {"path": path, "sha256": checker.sha(data), "reason": "self-declared third"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Task 1.2 authored set: .*extra=.*third.rs"):
        checker.check(trimmed)


@pytest.mark.parametrize("name", ["replay_parity.rs", "grammar_kernel.rs"])
def test_check_rejects_wrong_authored_hash(trimmed: Path, name: str) -> None:
    path = f"engine_rs/tests/{name}"
    manifest_path = trimmed / "engine_rs/TRIM_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (entry,) = [entry for entry in manifest["authored"] if entry["path"] == path]
    entry["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}: authored hash")):
        checker.check(trimmed)


def test_main_reports_failure_exit_code(
    trimmed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (trimmed / "engine_rs/src/py_random.rs").write_bytes(b"changed\n")
    monkeypatch.setattr(sys, "argv", ["check_engine_trim.py", str(trimmed)])
    assert checker.main() == 1


def test_declared_license_edit_is_rejected(
    trimmed: Path, reference: dict[str, bytes]
) -> None:
    path = "engine_rs/LICENSE"
    changed = reference[path] + b"Additional terms.\n"
    _declare(trimmed, path, changed, _whole_file_edit(reference[path], changed))
    with pytest.raises(
        ValueError, match=re.escape("LICENSE: retained file must be unchanged")
    ):
        checker.check(trimmed)


def test_declared_removal_of_trim_provenance_is_rejected(
    trimmed: Path, reference: dict[str, bytes]
) -> None:
    original = reference[_VENDORED]
    lines = len(original.decode("utf-8").splitlines())
    _declare(trimmed, _VENDORED, original, [])
    with pytest.raises(
        ValueError, match=re.escape("VENDORED_FROM.md: Task 1.1 trim appendix")
    ):
        checker.check(trimmed)
    # Also reject a rewritten appendix that keeps the heading.
    tampered = (
        original
        + b"\n## Task 1.1 rules-only trim \xe2\x80\x94 2026-09-29\n\nRewritten.\n"
    )
    appended = tampered[len(original) :].decode("utf-8")
    _declare(
        trimmed,
        _VENDORED,
        tampered,
        [
            {
                "start_line": lines + 1,
                "delete_lines": 0,
                "insert": appended,
                "reason": "rewrite",
            }
        ],
    )
    with pytest.raises(
        ValueError, match=re.escape("VENDORED_FROM.md: Task 1.1 trim appendix")
    ):
        checker.check(trimmed)


def test_later_provenance_append_is_accepted(
    trimmed: Path, reference: dict[str, bytes]
) -> None:
    original = reference[_VENDORED]
    current = (trimmed / _VENDORED).read_bytes()
    appended = current + b"\n## Later task\n\nMore provenance.\n"
    lines = len(original.decode("utf-8").splitlines())
    _declare(
        trimmed,
        _VENDORED,
        appended,
        [
            {
                "start_line": lines + 1,
                "delete_lines": 0,
                "insert": appended[len(original) :].decode("utf-8"),
                "reason": "append-only provenance",
            }
        ],
    )
    checker.check(trimmed)


def test_declared_extra_lockfile_change_is_rejected(
    trimmed: Path, reference: dict[str, bytes]
) -> None:
    current = (trimmed / _LOCK).read_bytes()
    changed = current.replace(b'version = "1.0.229"', b'version = "1.0.230"', 1)
    assert changed != current
    _declare(trimmed, _LOCK, changed, _whole_file_edit(reference[_LOCK], changed))
    with pytest.raises(
        ValueError, match=re.escape("Cargo.lock: Rayon closure removal only")
    ):
        checker.check(trimmed)


def test_declared_eighth_lib_removal_is_rejected(
    trimmed: Path, reference: dict[str, bytes]
) -> None:
    path = "engine_rs/src/lib.rs"
    original = reference[path]
    lines = original.splitlines(keepends=True)
    removed = {19, 21, 22, 23, 24, 25, 27, 20}
    changed = b"".join(
        line for number, line in enumerate(lines, 1) if number not in removed
    )
    _declare(trimmed, path, changed, _whole_file_edit(original, changed))
    with pytest.raises(ValueError, match=re.escape("lib.rs: exact seven removals")):
        checker.check(trimmed)


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


# --- Merged Task 1.1 + 1.1b: generated traces through the full check() path ---


def test_check_rejects_edited_generated_trace(trimmed: Path) -> None:
    trace = sorted((trimmed / checker.GENERATED_DIR).glob("gen-*.jsonl.gz"))[0]
    trace.write_bytes(trace.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match=re.escape(f"{trace.name}: trace hash")):
        checker.check(trimmed)


def test_check_rejects_unlisted_generated_trace(trimmed: Path) -> None:
    (trimmed / checker.GENERATED_DIR / "gen-unlisted.jsonl.gz").write_bytes(b"x")
    with pytest.raises(ValueError, match="generated trace inventory"):
        checker.check(trimmed)
