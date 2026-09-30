"""Build the Kaggriculture Kaggle submission tarball (Task 7.4).

The archive has ``main.py`` at its root (``python/kaggriculture_main.py``), the
``owl`` package with a prebuilt Linux x86-64 ``rs.abi3.so`` in place of any
local extension, ``models/primary/{checkpoint.pt,config.yaml}`` with a slim
model-only checkpoint, and ``manifest.json`` binding every file to its SHA-256,
the source commit, the original checkpoint hash and the runtime target.

The native module is built separately for the Kaggle runtime (CPython 3.11
abi3, Linux x86-64, glibc no newer than the image's) and passed in; this script
checks its ELF header and its highest required glibc symbol version. The
working tree must be clean: the manifest's source identity is the checked-out
commit. Nothing here uploads or submits.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "7.4" / "submission.tar.gz"
SIZE_LIMIT_MIB = 100
MAX_GLIBC = "2.35"
ENTRYPOINT = Path("python") / "kaggriculture_main.py"
EXCLUDED_NAMES = {"__pycache__", ".DS_Store"}
EXCLUDED_SUFFIXES = (".pyc", ".so", ".dylib", ".pyd")
ELF_X86_64 = 62


@dataclass(frozen=True)
class SourceIdentity:
    commit: str
    tree: str


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def source_identity(repo: Path) -> SourceIdentity:
    """Return HEAD and its tree; refuse a tree with tracked or untracked changes."""
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            f"refusing to build from a dirty tree at {repo}:\n{status[:2000]}"
        )
    return SourceIdentity(
        commit=_git(repo, "rev-parse", "HEAD"),
        tree=_git(repo, "rev-parse", "HEAD^{tree}"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_elf_x86_64(data: bytes) -> None:
    """Require a little-endian ELF64 shared object for x86-64."""
    if data[:4] != b"\x7fELF":
        raise ValueError("native module is not an ELF file (build it for Linux)")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("native module must be little-endian ELF64")
    machine = int.from_bytes(data[18:20], "little")
    if machine != ELF_X86_64:
        raise ValueError(f"native module e_machine is {machine}, expected x86-64 (62)")


def glibc_requirement(data: bytes) -> str | None:
    """Highest ``GLIBC_x.y[.z]`` version string referenced by the module."""
    versions = {
        tuple(int(part) for part in match.group(1).split(b"."))
        for match in re.finditer(rb"GLIBC_(\d+\.\d+(?:\.\d+)?)", data)
    }
    if not versions:
        return None
    return ".".join(str(part) for part in max(versions))


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _load_extract_model_weights() -> Any:
    path = REPO_ROOT / "scripts" / "extract_model_weights.py"
    spec = importlib.util.spec_from_file_location("extract_model_weights", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_model_weights"] = module
    spec.loader.exec_module(module)
    return module


def _copy_package(source: Path, destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        del directory
        return {
            name
            for name in names
            if name in EXCLUDED_NAMES or name.endswith(EXCLUDED_SUFFIXES)
        }

    shutil.copytree(source, destination, ignore=ignore)


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if info.isdir() or info.name.endswith(".so"):
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def write_archive(stage: Path, output: Path) -> None:
    """Deterministic gzip tarball of ``stage`` with sorted entries at the root."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(stage.rglob("*")):
            tar.add(
                path,
                arcname=path.relative_to(stage).as_posix(),
                recursive=False,
                filter=_tar_filter,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as compressed,
    ):
        compressed.write(buffer.getvalue())


def build(
    *,
    checkpoint: Path,
    config: Path,
    native_module: Path,
    output: Path,
    identity: SourceIdentity,
    runtime_receipt: dict[str, Any] | None,
    expected_checkpoint_sha256: str | None = None,
    max_glibc: str = MAX_GLIBC,
    size_limit_bytes: int = SIZE_LIMIT_MIB * 1024 * 1024,
) -> dict[str, Any]:
    """Stage, check and write the archive; return its manifest."""
    checkpoint_sha256 = sha256_file(checkpoint)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            f"checkpoint SHA-256 {checkpoint_sha256} != expected "
            f"{expected_checkpoint_sha256}"
        )
    native = native_module.read_bytes()
    check_elf_x86_64(native)
    glibc = glibc_requirement(native)
    if glibc is not None and _version_tuple(glibc) > _version_tuple(max_glibc):
        raise ValueError(
            f"native module requires GLIBC_{glibc}, "
            f"newer than the runtime's {max_glibc}"
        )
    if output.exists():
        raise ValueError(f"output already exists: {output}")

    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / "submission"
        stage.mkdir()
        shutil.copy2(REPO_ROOT / ENTRYPOINT, stage / "main.py")
        _copy_package(REPO_ROOT / "python" / "owl", stage / "owl")
        shutil.copy2(native_module, stage / "owl" / "rs.abi3.so")
        model_root = stage / "models" / "primary"
        model_root.mkdir(parents=True)
        _load_extract_model_weights().extract_model_weights(
            checkpoint, model_root / "checkpoint.pt"
        )
        shutil.copy2(config, model_root / "config.yaml")

        files = {
            path.relative_to(stage).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, Any] = {
            "format": "kaggriculture-kaggle-submission-v1",
            "built_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": {"commit": identity.commit, "tree": identity.tree},
            "checkpoint": {
                "original_path": str(checkpoint),
                "original_sha256": checkpoint_sha256,
                "slim_sha256": files["models/primary/checkpoint.pt"]["sha256"],
                "config_sha256": files["models/primary/config.yaml"]["sha256"],
                "load": "model_only strict; force_flash_attn overridden to false",
            },
            "native_module": {
                "sha256": files["owl/rs.abi3.so"]["sha256"],
                "elf": "ELF64 x86-64",
                "glibc_symbol_max": glibc,
                "glibc_limit": max_glibc,
            },
            "entrypoint": {
                "source": ENTRYPOINT.as_posix(),
                "sha256": files["main.py"]["sha256"],
                "deterministic": True,
                "threads": 1,
            },
            "locks": {
                "Cargo.lock": sha256_file(REPO_ROOT / "Cargo.lock"),
                "uv.lock": sha256_file(REPO_ROOT / "uv.lock"),
            },
            "builder": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "runtime_target": runtime_receipt,
            "files": files,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        write_archive(stage, output)
    size = output.stat().st_size
    if size > size_limit_bytes:
        output.unlink()
        raise ValueError(f"archive is {size} bytes, over the {size_limit_bytes} limit")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="training config.yaml; defaults to the checkpoint's sibling",
    )
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--runtime-receipt",
        type=Path,
        help="JSON receipt of the Kaggle runtime image, copied into the manifest",
    )
    parser.add_argument("--max-glibc", default=MAX_GLIBC)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    identity = source_identity(REPO_ROOT)
    config = args.config or args.checkpoint.parent / "config.yaml"
    receipt = (
        json.loads(args.runtime_receipt.read_text())
        if args.runtime_receipt is not None
        else None
    )
    manifest = build(
        checkpoint=args.checkpoint.resolve(),
        config=config.resolve(),
        native_module=args.native_module.resolve(),
        output=args.output.resolve(),
        identity=identity,
        runtime_receipt=receipt,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        max_glibc=args.max_glibc,
    )
    if source_identity(REPO_ROOT) != identity:
        raise RuntimeError("source tree changed during the build")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
                "source_commit": manifest["source"]["commit"],
                "checkpoint_sha256": manifest["checkpoint"]["original_sha256"],
                "native_glibc_symbol_max": manifest["native_module"][
                    "glibc_symbol_max"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
