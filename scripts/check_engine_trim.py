"""Verify the pinned rules-kernel inventory and every declared byte-level edit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

PIN = "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0"
LIB_SHA256 = "c4b9bac5057be3a435d2f1035aae17bcd15e7f95ea8557322e4929877c8231fd"
# Only these retained files may carry declared edits; every other retained file
# (LICENSE, RNG/attribution sources, RNG tests, fixtures) stays byte-identical.
EDITABLE = frozenset(
    {
        "engine_rs/Cargo.lock",
        "engine_rs/Cargo.toml",
        "engine_rs/VENDORED_FROM.md",
        "engine_rs/src/lib.rs",
    }
)
# Packages that `cargo remove rayon` prunes from the pinned lockfile.
RAYON_CLOSURE = frozenset(
    {
        "crossbeam-deque",
        "crossbeam-epoch",
        "crossbeam-utils",
        "either",
        "rayon",
        "rayon-core",
    }
)
TRIM_APPENDIX_HEADING = "\n## Task 1.1 rules-only trim \u2014 2026-09-29\n".encode()
# SHA-256 of the Task 1.1 appendix section, from its heading up to the next
# `## ` section (later provenance may be appended after it, never inside it).
TRIM_APPENDIX_SHA256 = (
    "1d089760261e8fef7ed0fddcaaa1a60776cb9abbb65386ef08d529f863d76c93"
)


class Edit(TypedDict):
    start_line: int
    delete_lines: int
    insert: str
    reason: str


class Retained(TypedDict):
    path: str
    reference_sha256: str
    sha256: str
    edits: list[Edit]


class Excluded(TypedDict):
    path: str
    reference_sha256: str
    reason: str


class Authored(TypedDict):
    path: str
    sha256: str
    reason: str


class NonEngineChange(TypedDict):
    path: str
    reason: str


class Manifest(TypedDict):
    schema_version: int
    reference_commit: str
    retained: list[Retained]
    excluded: list[Excluded]
    authored: list[Authored]
    non_engine_changes: list[NonEngineChange]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: object, keys: Sequence[str], label: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{label}: expected object")
    result = cast(dict[str, object], value)
    _require(set(result) == set(keys), f"{label}: schema keys must be {list(keys)}")
    return result


def _array(value: object, label: str) -> list[object]:
    _require(isinstance(value, list), f"{label}: expected array")
    return cast(list[object], value)


def _string(value: object, label: str) -> str:
    _require(isinstance(value, str), f"{label}: expected string")
    return cast(str, value)


def _integer(value: object, label: str) -> int:
    _require(type(value) is int, f"{label}: expected integer")
    return cast(int, value)


def _reason(value: object, label: str) -> str:
    reason = _string(value, f"{label}: reason")
    _require(bool(reason.strip()), f"{label}: reason must not be empty")
    return reason


def _path(value: object, *, engine: bool) -> str:
    path = _string(value, "path")
    parts = path.split("/")
    _require(
        "\\" not in path
        and "\x00" not in path
        and not any(part in ("", ".", "..", ".git") for part in parts),
        f"{path!r}: unsafe path",
    )
    if engine:
        _require(
            len(parts) > 1
            and parts[0] == "engine_rs"
            and parts[1] not in ("target", "TRIM_MANIFEST.json"),
            f"{path}: invalid engine path",
        )
    else:
        _require(parts[0] != "engine_rs", f"{path}: invalid non-engine path")
    return path


def _digest(value: object, label: str) -> str:
    digest = _string(value, label)
    _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"{label}: SHA-256")
    return digest


def _edit(value: object, path: str) -> Edit:
    entry = _object(value, ("start_line", "delete_lines", "insert", "reason"), path)
    return {
        "start_line": _integer(entry["start_line"], f"{path}: start_line"),
        "delete_lines": _integer(entry["delete_lines"], f"{path}: delete_lines"),
        "insert": _string(entry["insert"], f"{path}: insert"),
        "reason": _reason(entry["reason"], path),
    }


def parse_manifest(value: object) -> Manifest:
    raw = _object(
        value,
        (
            "schema_version",
            "reference_commit",
            "retained",
            "excluded",
            "authored",
            "non_engine_changes",
        ),
        "manifest",
    )
    version = _integer(raw["schema_version"], "schema_version")
    _require(version == 1, "schema_version must be 1")
    pin = _string(raw["reference_commit"], "reference_commit")
    _require(pin == PIN, f"reference_commit must be {PIN}")
    retained: list[Retained] = []
    for value in _array(raw["retained"], "retained"):
        entry = _object(
            value, ("path", "reference_sha256", "sha256", "edits"), "retained"
        )
        path = _path(entry["path"], engine=True)
        retained.append(
            {
                "path": path,
                "reference_sha256": _digest(entry["reference_sha256"], path),
                "sha256": _digest(entry["sha256"], path),
                "edits": [_edit(edit, path) for edit in _array(entry["edits"], path)],
            }
        )
    excluded: list[Excluded] = []
    for value in _array(raw["excluded"], "excluded"):
        entry = _object(value, ("path", "reference_sha256", "reason"), "excluded")
        path = _path(entry["path"], engine=True)
        excluded.append(
            {
                "path": path,
                "reference_sha256": _digest(entry["reference_sha256"], path),
                "reason": _reason(entry["reason"], path),
            }
        )
    authored: list[Authored] = []
    for value in _array(raw["authored"], "authored"):
        entry = _object(value, ("path", "sha256", "reason"), "authored")
        path = _path(entry["path"], engine=True)
        authored.append(
            {
                "path": path,
                "sha256": _digest(entry["sha256"], path),
                "reason": _reason(entry["reason"], path),
            }
        )
    changes: list[NonEngineChange] = []
    for value in _array(raw["non_engine_changes"], "non_engine_changes"):
        entry = _object(value, ("path", "reason"), "non_engine_changes")
        path = _path(entry["path"], engine=False)
        changes.append({"path": path, "reason": _reason(entry["reason"], path)})
    return {
        "schema_version": version,
        "reference_commit": pin,
        "retained": retained,
        "excluded": excluded,
        "authored": authored,
        "non_engine_changes": changes,
    }


def _same_inventory(actual: Sequence[str], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    _require(not missing and not extra, f"{label}: missing={missing}; extra={extra}")


def verify_task_authored(authored: Sequence[Authored]) -> None:
    _same_inventory(
        [entry["path"] for entry in authored],
        ["engine_rs/tests/replay_parity.rs", "engine_rs/tests/grammar_kernel.rs"],
        "Task 1.2 authored set",
    )


def _reconstruct(original: bytes, edits: Sequence[Edit], path: str) -> bytes:
    if not edits:
        return original
    _require(not path.endswith(".gz"), f"{path}: binary fixtures cannot have edits")
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: edits require UTF-8") from error
    lines = re.findall(r"[^\n]*\n|[^\n]+$", text)
    cursor = 0
    parts: list[str] = []
    for edit in edits:
        start = edit["start_line"] - 1
        end = start + edit["delete_lines"]
        _require(
            cursor <= start <= len(lines) and edit["delete_lines"] >= 0,
            f"{path}: edit range",
        )
        _require(end <= len(lines), f"{path}: edit end")
        parts.extend(("".join(lines[cursor:start]), edit["insert"]))
        cursor = end
    parts.append("".join(lines[cursor:]))
    try:
        return "".join(parts).encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{path}: inserted text requires UTF-8") from error


def verify(
    value: object, originals: Mapping[str, bytes], current: Mapping[str, bytes]
) -> None:
    manifest = parse_manifest(value)
    entries = [*manifest["retained"], *manifest["excluded"], *manifest["authored"]]
    seen: set[str] = set()
    for entry in [*entries, *manifest["non_engine_changes"]]:
        path = entry["path"]
        _require(path not in seen, f"duplicate: {path}")
        seen.add(path)
    _same_inventory(
        [entry["path"] for entry in [*manifest["retained"], *manifest["excluded"]]],
        list(originals),
        "reference inventory",
    )
    _same_inventory(
        [entry["path"] for entry in [*manifest["retained"], *manifest["authored"]]],
        list(current),
        "current inventory",
    )
    for excluded in manifest["excluded"]:
        path = excluded["path"]
        _require(
            sha(originals[path]) == excluded["reference_sha256"],
            f"{path}: reference hash",
        )
        _require(path not in current, f"{path}: excluded file present")
    for retained in manifest["retained"]:
        path = retained["path"]
        original, actual = originals[path], current[path]
        _require(
            sha(original) == retained["reference_sha256"], f"{path}: reference hash"
        )
        _require(sha(actual) == retained["sha256"], f"{path}: current hash")
        _require(
            actual == _reconstruct(original, retained["edits"], path),
            f"{path}: undeclared edit",
        )
        if path not in EDITABLE:
            _require(not retained["edits"], f"{path}: retained file must be unchanged")
    for authored in manifest["authored"]:
        path = authored["path"]
        _require(sha(current[path]) == authored["sha256"], f"{path}: authored hash")


def _current_files(root: Path) -> dict[str, bytes]:
    current: dict[str, bytes] = {}

    def walk(directory: Path) -> None:
        _require(not directory.is_symlink(), f"{directory}: symlink")
        for child in sorted(directory.iterdir()):
            relative = child.relative_to(root).as_posix()
            if relative in ("engine_rs/target", "engine_rs/TRIM_MANIFEST.json"):
                continue
            _require(not child.is_symlink(), f"{relative}: symlink")
            if child.is_dir():
                walk(child)
            else:
                _require(child.is_file(), f"{relative}: not a regular file")
                current[relative] = child.read_bytes()

    walk(root / "engine_rs")
    return current


def reference_files(root: Path) -> dict[str, bytes]:
    """Read every pinned-reference ``engine_rs`` blob from the Git object store."""

    def git(*args: str) -> bytes:
        # check_output has no Node-style 1 MiB limit; the largest blob is 5,099,830 B.
        return subprocess.check_output(["git", *args], cwd=root)

    paths = git("ls-tree", "-r", "--name-only", PIN, "engine_rs").decode().splitlines()
    return {path: git("show", f"{PIN}:{path}") for path in paths}


def check(root: Path) -> None:
    root = root.resolve()
    manifest_path = root / "engine_rs/TRIM_MANIFEST.json"
    _require(not manifest_path.is_symlink(), f"{manifest_path}: symlink")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_task(value, reference_files(root), _current_files(root))


def verify_task(
    value: object, originals: Mapping[str, bytes], current: Mapping[str, bytes]
) -> None:
    """Apply the generic manifest checks plus the fixed Task 1.1 trim contract."""
    manifest = parse_manifest(value)
    verify(manifest, originals, current)
    retained = [
        "Cargo.toml",
        "Cargo.lock",
        "LICENSE",
        "VENDORED_FROM.md",
        "src/lib.rs",
        "src/py_random.rs",
        "src/econ_attrib.rs",
        "tests/py_random.rs",
        *[
            f"fixtures/episode-{episode}.jsonl.gz"
            for episode in (95324500, 95901360, 95921764, 95990191)
        ],
    ]
    _same_inventory(
        [entry["path"] for entry in manifest["retained"]],
        [f"engine_rs/{path}" for path in retained],
        "Task 1.1 retained set",
    )
    verify_task_authored(manifest["authored"])
    _require(
        "justfile" in {entry["path"] for entry in manifest["non_engine_changes"]},
        "Task 1.1 non_engine_changes must include justfile",
    )
    removed = {19, 21, 22, 23, 24, 25, 27}
    original_lines = originals["engine_rs/src/lib.rs"].splitlines(keepends=True)
    expected_lib = b"".join(
        line
        for number, line in enumerate(original_lines, start=1)
        if number not in removed
    )
    _require(
        current["engine_rs/src/lib.rs"] == expected_lib, "lib.rs: exact seven removals"
    )
    _require(sha(expected_lib) == LIB_SHA256, "lib.rs: expected trim SHA-256")
    cargo = (
        originals["engine_rs/Cargo.toml"]
        .replace(b'crate-type = ["rlib", "cdylib"]\n', b"")
        .replace(b'[[bin]]\nname = "re_engine"\npath = "src/main.rs"\n\n', b"")
        .replace(b'rayon = "1.12.0"\n', b"")
    )
    _require(
        current["engine_rs/Cargo.toml"] == cargo,
        "Cargo.toml: target/dependency removals only",
    )
    _require(
        current["engine_rs/Cargo.lock"]
        == _expected_lock(originals["engine_rs/Cargo.lock"]),
        "Cargo.lock: Rayon closure removal only",
    )
    _verify_provenance(
        originals["engine_rs/VENDORED_FROM.md"], current["engine_rs/VENDORED_FROM.md"]
    )


def _expected_lock(original: bytes) -> bytes:
    """Derive the trimmed lockfile: drop the Rayon closure and the Rayon edge."""
    separator = b"\n\n[[package]]\n"
    header, *packages = original.split(separator)
    names = [package.split(b"\n", 1)[0] for package in packages]
    pruned = {f'name = "{name}"'.encode() for name in RAYON_CLOSURE}
    _require(pruned <= set(names), "Cargo.lock: reference lacks the Rayon closure")
    kept = [
        package
        for package, name in zip(packages, names, strict=True)
        if name not in pruned
    ]
    lock = separator.join([header, *kept])
    edge = b'\n "rayon",\n'
    _require(lock.count(edge) == 1, "Cargo.lock: expected one Rayon dependency edge")
    return lock.replace(edge, b"\n")


def _verify_provenance(original: bytes, current: bytes) -> None:
    """Historical bytes stay intact and the exact Task 1.1 appendix follows them."""
    _require(current.startswith(original), "VENDORED_FROM.md: append-only provenance")
    tail = current[len(original) :]
    end = tail.find(b"\n## ", 1)
    section = tail if end < 0 else tail[:end]
    _require(
        tail.startswith(TRIM_APPENDIX_HEADING) and sha(section) == TRIM_APPENDIX_SHA256,
        "VENDORED_FROM.md: Task 1.1 trim appendix must follow the history unchanged",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        check(args.root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    print("engine trim manifest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
