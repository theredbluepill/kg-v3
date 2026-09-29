"""Replace each recorder size/hash require() with a no-op, run the size/hash tests, restore.

Run from the repository root: uv run python ops/rebuild-2026-09-29/1.4/verify-r2-fixes/size_hash_guard_mutations.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

src = Path("scripts/record_kaggriculture_env_reference.py")
orig = src.read_bytes()
text = orig.decode()
selector = "size_budget or size_mismatch or archive_corruption or digest_mismatch"
messages = [
    "compressed size budget exceeded",
    "expanded size budget exceeded",
    "fixture compressed size differs",
    "fixture hash differs",
    "expanded size budget differs",
    "expanded fixture hash differs",
]
out: dict[str, object] = {
    "recorder_sha256_before": hashlib.sha256(orig).hexdigest(),
    "test": f"tests/tools/test_record_kaggriculture_env_reference.py -k '{selector}'",
    "mutation": "replace one require( call with a no-op, run, restore",
    "cases": {},
}


def run() -> dict[str, object]:
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/tools/test_record_kaggriculture_env_reference.py",
         "-q", "-k", selector, "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    return {"returncode": r.returncode, "summary": r.stdout.strip().splitlines()[-1]}


try:
    out["baseline"] = run()
    for message in messages:
        key = f'"{message}"'
        assert text.count(key) == 1, message
        i = text.index(key)
        j = text.rindex("require(", 0, i)
        src.write_text(text[:j] + "(lambda *_: None)(" + text[j + len("require("):])
        out["cases"][message] = run()  # type: ignore[index]
        print(message, "->", out["cases"][message], flush=True)  # type: ignore[index]
finally:
    src.write_bytes(orig)
out["recorder_sha256_restored"] = hashlib.sha256(src.read_bytes()).hexdigest()
out["restored"] = out["recorder_sha256_restored"] == out["recorder_sha256_before"]
print(json.dumps(out, indent=1))
