from pathlib import Path
import json
import subprocess
results = []
for path in sorted(Path("tests/owl").rglob("test_*.py")):
    name = "codex-verify-orbit-" + str(path.relative_to("tests/owl")).replace("/", "-").removesuffix(".py")
    result = subprocess.run(["python3", "ops/rebuild-2026-09-29/1.3/bounded.py", "--name", name,
                             "--", "uv", "run", "pytest", str(path), "-q"], check=False)
    results.append({"path": str(path), "status": result.returncode})
    print(json.dumps(results[-1]), flush=True)
Path("ops/rebuild-2026-09-29/1.3/codex-verify-orbit-shards.json").write_text(json.dumps(results, indent=2)+"\n")
