"""Merge-seam mutations for kg/merge-3-1-3-5-c; each is applied, tested, restored."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TESTS = [
    "tests/owl/train/test_logging.py",
    "tests/scripts/test_run_ppo.py",
    "tests/kaggriculture/test_teacher.py::test_run_ppo_resume_restores_the_teacher_from_checkpoint_last_best",
    "tests/kaggriculture/test_teacher.py::test_run_ppo_fresh_launch_from_weights_activates_the_last_best_teacher",
]
MUTATIONS = [
    (
        "offline-resume-allowed",
        "python/owl/train/logging.py",
        'if wandb_mode == "offline" and resume_run_id is not None:',
        "if False:",
    ),
    (
        "kaggriculture-name-unprefixed",
        "python/owl/train/logging.py",
        'name = f"ppo-{run_dir.name}"',
        "name = run_dir.name",
    ),
    (
        "orbit-online-gets-mode",
        "python/owl/train/logging.py",
        'elif wandb_mode == "offline":',
        "else:",
    ),
    (
        "teacher-source-check-dropped",
        "scripts/run_ppo.py",
        "        _require_kaggriculture_teacher_source(cfg, launch)\n",
        "",
    ),
    (
        "startup-env-steps-read-dropped",
        "scripts/run_ppo.py",
        "            _checkpoint_env_steps(start_checkpoint_path)\n",
        "            0\n",
    ),
    (
        "teacher-init-not-loaded",
        "scripts/run_ppo.py",
        'if cfg.rl.teacher_mode != "last_best" or cfg.rl.teacher_init is None:',
        "if True:",
    ),
]


def main() -> None:
    for name, rel, old, new in MUTATIONS:
        path = ROOT / rel
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        text = original.decode()
        assert text.count(old) == 1, (name, text.count(old))
        path.write_text(text.replace(old, new))
        try:
            result = subprocess.run(
                ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_bytes(original)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        summary = result.stdout.strip().splitlines()[-1]
        failed = [
            line.split(" ")[1]
            for line in result.stdout.splitlines()
            if line.startswith("FAILED ")
        ]
        status = "KILLED" if result.returncode != 0 else "SURVIVED"
        print(f"{name}: {status} ({summary}); restored sha256 {digest[:12]}")
        for test in failed:
            print(f"  {test}")


if __name__ == "__main__":
    main()
