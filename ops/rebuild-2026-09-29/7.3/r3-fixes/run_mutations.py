"""Apply each publication-guard mutation to replay_export.py, run the focused
tests, and restore the source byte-for-byte. Each mutation must fail a test."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/owl/kaggriculture/replay_export.py"
TESTS = [
    "tests/kaggriculture/test_replay_export.py",
    "tests/kaggriculture/test_replay_export_integration.py",
]
MUTATIONS = {
    "custody_published_before_episode": (
        "files.insert(0, (episode_path, episode_bytes))",
        "files.append((episode_path, episode_bytes))",
    ),
    "rename_replaces_existing_path": (
        "os.link(temporary, path)",
        "os.replace(temporary, path)",
    ),
    "no_fsync_before_link": (
        "            os.fsync(stream.fileno())\n",
        "",
    ),
    "published_files_not_removed_on_failure": (
        "            path.unlink(missing_ok=True)\n        raise",
        "            pass\n        raise",
    ),
    "no_error_custody_after_failed_publication": (
        'if sidecar["status"] != "error":',
        "if False:",
    ),
    "staging_file_left_behind": (
        "        temporary.unlink(missing_ok=True)\n",
        "        pass\n",
    ),
    "error_custody_keeps_episode_hash": (
        'if key not in ("episode_sha256", "verification")',
        'if key != "verification"',
    ),
    "unpublished_error_custody_retires_game": (
        "            )\n            return\n        self._retire(replay)",
        "            )\n        self._retire(replay)",
    ),
}


def main() -> int:
    original = SOURCE.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    results = {}
    try:
        for name, (old, new) in MUTATIONS.items():
            text = original.decode()
            assert text.count(old) == 1, name
            SOURCE.write_text(text.replace(old, new))
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            tail = run.stdout.strip().splitlines()[-1]
            failed = [
                line.split(" ", 1)[1].split(" - ")[0]
                for line in run.stdout.splitlines()
                if line.startswith("FAILED ")
            ]
            results[name] = {"exit": run.returncode, "summary": tail, "failed": failed}
            SOURCE.write_bytes(original)
    finally:
        SOURCE.write_bytes(original)
    restored = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = {"source_sha256": digest, "restored_sha256": restored, "mutations": results}
    print(json.dumps(report, indent=2))
    detected = all(result["exit"] != 0 for result in results.values())
    return 0 if detected and restored == digest else 1


if __name__ == "__main__":
    raise SystemExit(main())
