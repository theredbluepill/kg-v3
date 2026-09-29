"""Merge-seam mutations for the W&B landing (kg/merge-wandb-c).

Each mutation edits one resolution the merge made, runs the targeted tests,
expects at least one failure, and restores the file byte for byte.
Run from the worktree root: uv run python ops/rebuild-2026-09-29/merge-wandb-c/mutations.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

TESTS = ["tests/owl/train/test_logging.py", "tests/scripts/test_run_ppo.py"]

MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "M1 drop the offline-resume check in _validate_args",
        "scripts/run_ppo.py",
        "    if args.output_dir is None and args.wandb_mode == WandbMode.OFFLINE:\n",
        "    if False:\n",
    ),
    (
        "M2 drop the offline-resume check in WandbLogger",
        "python/owl/train/logging.py",
        "        if mode is WandbMode.OFFLINE and resume_run_id is not None:\n",
        "        if False:\n",
    ),
    (
        "M3 run name back to the bare run directory",
        "python/owl/train/logging.py",
        '            name=f"{identity.job_type}-{run_dir.name}",\n',
        "            name=run_dir.name,\n",
    ),
    (
        "M4 drop Task 3.1's offline run-dir line",
        "scripts/run_ppo.py",
        "        if identity.telemetry is TelemetryMode.WANDB_OFFLINE:\n",
        "        if False:\n",
    ),
    (
        "M5 send Orbit back to orbit-wars",
        "python/owl/train/logging.py",
        "            project=WANDB_PROJECT,\n",
        '            project=WANDB_PROJECT if game == "kaggriculture" else "orbit-wars",\n',
    ),
    (
        "M6 let debug logging accept --wandb-mode offline",
        "python/owl/train/logging.py",
        "            if wandb_mode != WandbMode.ONLINE:\n                raise ValueError(",
        "            if False:\n                raise ValueError(",
    ),
    (
        "M7 group back to the constant ppo",
        "python/owl/train/logging.py",
        "            group=identity.experiment_id,\n",
        '            group="ppo",\n',
    ),
]


def main() -> int:
    survivors = 0
    for name, path_text, old, new in MUTATIONS:
        path = Path(path_text)
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        text = original.decode()
        if text.count(old) != 1:
            print(f"{name}: anchor not found exactly once", flush=True)
            return 2
        path.write_text(text.replace(old, new, 1))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_bytes(original)
        restored = hashlib.sha256(path.read_bytes()).hexdigest() == digest
        summary = result.stdout.strip().splitlines()[-1] if result.stdout else ""
        killed = result.returncode != 0
        survivors += not killed
        print(
            f"{name}: {'KILLED' if killed else 'SURVIVED'} ({summary}); "
            f"restored={restored}",
            flush=True,
        )
        if not restored:
            return 3
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
