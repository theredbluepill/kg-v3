"""Stage a LOCAL macOS copy of the Task 7.4 Kaggle package (not for upload).

Mirrors scripts/build_kaggriculture_submission.py (kg/rebuild-7-4-ship 619349f):
main.py <- python/kaggriculture_main.py, the owl package minus built extensions,
owl/rs.abi3.so, models/primary/{checkpoint.pt (slim model-only), config.yaml},
manifest.json. Differences, on purpose: the native module is a macOS arm64
build (maturin, same source) instead of the Linux ELF, so the ELF/glibc checks
are skipped; the source is a scratch no-commit merge of the ship branch into the
training commit 0f70773 (the ship branch alone cannot validate the run's
reward_shaping config keys), so the identity records both parents and the
resolved conflict instead of one clean commit. Nothing is uploaded or submitted.

usage: python pkg_local_mac.py SRC SO CHECKPOINT CONFIG OUT_DIR EXPECTED_SHA256
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EXCLUDED_NAMES = {"__pycache__", ".DS_Store"}
EXCLUDED_SUFFIXES = (".pyc", ".so", ".dylib", ".pyd")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    src, so, checkpoint, config, out, expected = sys.argv[1:7]
    src_p, so_p, ckpt_p, cfg_p, out_p = map(Path, (src, so, checkpoint, config, out))
    actual = sha(ckpt_p)
    if actual != expected:
        raise SystemExit(f"checkpoint sha256 {actual} != expected {expected}")
    if out_p.exists():
        raise SystemExit(f"output exists: {out_p}")
    out_p.mkdir(parents=True)
    shutil.copy2(src_p / "python" / "kaggriculture_main.py", out_p / "main.py")
    shutil.copytree(
        src_p / "python" / "owl",
        out_p / "owl",
        ignore=lambda _d, names: {
            n for n in names if n in EXCLUDED_NAMES or n.endswith(EXCLUDED_SUFFIXES)
        },
    )
    shutil.copy2(so_p, out_p / "owl" / "rs.abi3.so")
    model_root = out_p / "models" / "primary"
    model_root.mkdir(parents=True)
    sys.path.insert(0, str(out_p))
    sys.path.insert(1, str(src_p / "scripts"))
    from extract_model_weights import extract_model_weights  # noqa: PLC0415

    extract_model_weights(ckpt_p, model_root / "checkpoint.pt")
    shutil.copy2(cfg_p, model_root / "config.yaml")
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(src_p), *a], check=True, capture_output=True, text=True
    ).stdout.strip()
    import torch  # noqa: PLC0415

    files = {
        p.relative_to(out_p).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(out_p.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    manifest = {
        "format": "kaggriculture-local-mac-package-v1 (NOT a submission)",
        "built_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "head": git("rev-parse", "HEAD"),
            "merging": (
                subprocess.run(["git", "-C", str(src_p), "rev-parse", "-q", "--verify",
                                "MERGE_HEAD"], capture_output=True, text=True).stdout.strip()
                or None
            ),
            "note": "if merging is set: scratch no-commit merge of kg/rebuild-7-4-ship "
            "619349f into training commit 0f70773 (observe.rs additive conflict resolved "
            "by keeping both impl blocks; python/owl/model identical to 0f70773); "
            "otherwise a clean detached checkout of head",
            "diff_vs_head_sha256": hashlib.sha256(
                git("diff", "HEAD").encode()
            ).hexdigest(),
        },
        "checkpoint": {
            "original_path": str(ckpt_p),
            "original_sha256": actual,
            "slim_sha256": files["models/primary/checkpoint.pt"]["sha256"],
            "config_sha256": files["models/primary/config.yaml"]["sha256"],
            "load": "model_only strict; force_flash_attn overridden to false",
        },
        "native_module": {
            "sha256": files["owl/rs.abi3.so"]["sha256"],
            "format": "Mach-O arm64 (maturin build --release, abi3 cp311)",
        },
        "entrypoint": {"source": "python/kaggriculture_main.py", "threads": 1,
                       "deterministic": True},
        "builder": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform()},
        "files": files,
    }
    (out_p / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out_p), "checkpoint_sha256": actual,
                      "slim_sha256": manifest["checkpoint"]["slim_sha256"]}))


if __name__ == "__main__":
    main()
