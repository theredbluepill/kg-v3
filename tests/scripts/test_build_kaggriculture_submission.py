"""Task 7.4: the Kaggriculture submission archive and its manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.kaggriculture.kaggle_fixtures import tiny_state, write_model_root

_SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "build_kaggriculture_submission.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "build_kaggriculture_submission", _SCRIPT_PATH
)
assert _SPEC is not None
assert _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
sys.modules["build_kaggriculture_submission"] = builder
_SPEC.loader.exec_module(builder)

IDENTITY = builder.SourceIdentity(commit="a" * 40, tree="b" * 40)


def _elf(glibc: str = "2.17", machine: int = 62) -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4], header[5] = 2, 1
    header[18:20] = machine.to_bytes(2, "little")
    return bytes(header) + f"\x00GLIBC_2.2.5\x00GLIBC_{glibc}\x00".encode()


def _inputs(tmp_path: Path, *, glibc: str = "2.17") -> dict[str, Any]:
    root = write_model_root(
        tmp_path / "run",
        checkpoint={"model": tiny_state(), "optimizer": {"state": {}}, "env_steps": 1},
    )
    native = tmp_path / "rs.abi3.so"
    native.write_bytes(_elf(glibc))
    return {
        "checkpoint": root / "checkpoint.pt",
        "config": root / "config.yaml",
        "native_module": native,
        "output": tmp_path / "out" / "submission.tar.gz",
        "identity": IDENTITY,
        "runtime_receipt": {"python": "3.11.13"},
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_archive_layout_manifest_and_slim_checkpoint(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    original_sha = _sha(inputs["checkpoint"].read_bytes())
    manifest = builder.build(**inputs, expected_checkpoint_sha256=original_sha)
    with tarfile.open(inputs["output"]) as tar:
        members = {member.name: member for member in tar.getmembers()}
        contents = {
            name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for name, member in members.items()
            if member.isfile()
        }
    assert "main.py" in contents
    assert "manifest.json" in contents
    assert json.loads(contents["manifest.json"]) == manifest
    repo = Path(__file__).parents[2]
    assert (
        contents["main.py"] == (repo / "python" / "kaggriculture_main.py").read_bytes()
    )
    assert contents["owl/rs.abi3.so"] == inputs["native_module"].read_bytes()
    assert [name for name in contents if name.endswith(".so")] == ["owl/rs.abi3.so"]
    assert not [
        name for name in members if "__pycache__" in name or name.endswith(".pyc")
    ]
    assert "owl/kaggriculture/kaggle_warmup.json" in contents
    for name, entry in manifest["files"].items():
        assert _sha(contents[name]) == entry["sha256"], name
        assert len(contents[name]) == entry["bytes"], name
    assert set(manifest["files"]) == set(contents) - {"manifest.json"}
    slim = torch.load(io.BytesIO(contents["models/primary/checkpoint.pt"]))
    assert set(slim) == {"model"}
    assert manifest["checkpoint"]["original_sha256"] == original_sha
    # Only the file name is recorded, never the builder's local path.
    assert manifest["checkpoint"]["original_name"] == inputs["checkpoint"].name
    assert "original_path" not in manifest["checkpoint"]
    assert manifest["source"] == {"commit": IDENTITY.commit, "tree": IDENTITY.tree}
    assert manifest["native_module"]["glibc_symbol_max"] == "2.17"
    assert manifest["entrypoint"]["deterministic"] is True
    assert manifest["runtime_target"] == {"python": "3.11.13"}
    for member in members.values():
        assert (member.uid, member.gid, member.mtime) == (0, 0, 0)


def test_native_receipt_is_bound_to_the_staged_module(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    module_sha = _sha(inputs["native_module"].read_bytes())
    receipt = {"source_commit": "c" * 40, "module": {"sha256": module_sha}}
    manifest = builder.build(**inputs, native_receipt=receipt)
    assert manifest["native_module"]["build"] == receipt
    assert manifest["native_module"]["sha256"] == module_sha
    other = _inputs(tmp_path / "other")
    with pytest.raises(ValueError, match="native receipt names module"):
        builder.build(**other, native_receipt={"module": {"sha256": "0" * 64}})
    assert not other["output"].exists()


def test_archive_bytes_are_reproducible_apart_from_build_time(tmp_path: Path) -> None:
    first = _inputs(tmp_path / "a")
    second = _inputs(tmp_path / "b")
    builder.build(**first)
    builder.build(**second)

    def files(path: Path) -> dict[str, bytes]:
        with tarfile.open(path) as tar:
            return {
                member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
                for member in tar.getmembers()
                if member.isfile() and member.name != "manifest.json"
            }

    assert files(first["output"]) == files(second["output"])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"expected_checkpoint_sha256": "0" * 64}, "expected"),
        ({"size_limit_bytes": 1024}, "over the 1024 limit"),
        ({"max_glibc": "2.16"}, "GLIBC_2.17"),
    ],
)
def test_build_refuses_bad_inputs(
    tmp_path: Path, change: dict[str, Any], match: str
) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match=match):
        builder.build(**inputs, **change)
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"\xcf\xfa\xed\xfe" + bytes(60), "not an ELF"),
        (_elf(machine=183), "x86-64"),
    ],
)
def test_native_module_must_be_linux_x86_64(data: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        builder.check_elf_x86_64(data)


def test_glibc_requirement_reads_the_highest_symbol_version() -> None:
    assert builder.glibc_requirement(_elf("2.34")) == "2.34"
    assert builder.glibc_requirement(b"GLIBC_2.9\x00GLIBC_2.10") == "2.10"
    assert builder.glibc_requirement(b"no versions") is None


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_source_identity_refuses_dirty_trees(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
    identity = builder.source_identity(repo)
    assert len(identity.commit) == 40
    assert len(identity.tree) == 40
    (repo / "untracked.txt").write_text("b")
    with pytest.raises(RuntimeError, match="dirty"):
        builder.source_identity(repo)


def test_bulk_outputs_are_ignored_by_git() -> None:
    repo = Path(__file__).parents[2]
    for path in ("artifacts/7.4/submission.tar.gz", "submission.tar.gz"):
        subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", path], check=True)
    assert repo / "artifacts" / "7.4" / "submission.tar.gz" == builder.DEFAULT_OUTPUT
