"""Read-only source/receipt checks for the Task 1.3 working-tree handoff."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True)


run("git", "diff", "--check")
assert not run("git", "status", "--porcelain", "--", "engine_rs").strip()
assert not (ROOT / "python/owl/kaggriculture/types.py").exists()
assert not (ROOT / "tests/fixtures/kaggriculture/observation-v3").exists()

notes = [
    "cookbook/references/structured-observations-preserve-legal-state-and-order.md",
    "cookbook/decisions/restart-the-port-from-isaiahs-clean-base.md",
]
payload = {
    "cwd": str(ROOT),
    "tool_name": "apply_patch",
    "tool_input": {"files": [{"file_path": path} for path in notes]},
}
lint = subprocess.check_output(
    ["node", ".codex/hooks/cookbook-lint.mjs"],
    input=json.dumps(payload), cwd=ROOT, text=True,
)
assert json.loads(lint) == {}, lint
sources = set()
for path in notes:
    note = (ROOT / path).read_text()
    assert 'tags: ["kaggriculture-v3"' in note
    for source in re.findall(r'"repository:([^"\n]+)"', note):
        assert (ROOT / source).exists(), source
        sources.add(source)

mutations = json.loads((WORK / "h13-mutations.json").read_text())
assert len(mutations["cases"]) == 16
assert mutations["original_sha256"] == mutations["restored_sha256"]
assert all(case["semantic_failure_observed"] for case in mutations["cases"])
assert "drop(self.acquire_snapshot())" not in (ROOT / "src/kaggriculture/observe.rs").read_text()

changed = set(run("git", "ls-files", "--modified", "--others", "--exclude-standard", "-z").split("\0")) - {""}
code = sorted(path for path in changed if path.startswith(("src/", "python/", "scripts/", "tests/", "docs/", "cookbook/")) or path in ("Cargo.toml", "Cargo.lock"))
for path in code:
    data = (ROOT / path).read_bytes()
    assert data.endswith(b"\n"), path
    assert all(line.rstrip(b" \t") == line for line in data.splitlines()), path
(WORK / "final-source.sha256").write_text("".join(
    f"{hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}  {path}\n" for path in code
))
print(json.dumps({
    "git_diff_check": "pass", "engine_worktree_changes": 0,
    "real_schema_present": False, "qualified_corpus_present": False,
    "cookbook_working_note_lint": "pass", "existing_repository_sources": len(sources),
    "restored_H_mutations": 16, "snapshot_mutation_absent": True,
    "source_files_hashed": len(code), "source_whitespace": "pass",
}))
