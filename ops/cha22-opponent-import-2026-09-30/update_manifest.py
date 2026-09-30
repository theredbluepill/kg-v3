"""Register Cha22's import in opponents_rs/OPPONENT_MANIFEST.json (idempotent).

Run from the repository root after every authored edit:
    uv run --offline python ops/cha22-opponent-import-2026-09-30/update_manifest.py
It takes pins from scripts/check_opponent_import.py, recomputes authored hashes,
and reads compared-action counts from this folder's cha22-parity.json (written by
the Rust oracle test through CHA22_PARITY_REPORT). It never reads a sibling repo.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_opponent_import", ROOT / "scripts/check_opponent_import.py"
)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

REASONS = {
    "opponents_rs/README.md": (
        "Public controller-view API, default-config support, information boundary, "
        "provenance notices, Cha22 closure/fixtures/notices and parity limits."
    ),
    "opponents_rs/src/lib.rs": (
        "V3 controller view owns a frozen engine plus refreshed public snapshot; "
        "private fib compatibility helper; configuration() for the v2-imported "
        "Cha22 closure."
    ),
    "opponents_rs/src/native_agents.rs": (
        "Explicit controller module declarations (four 7.1 bots, Cha22 and its "
        "unregistered dependency closure) and narrow imported-code lint allowances."
    ),
    "opponents_rs/src/registry.rs": (
        "Five-key registry and independent seat/episode controller lifecycle."
    ),
    "opponents_rs/src/view_tests.rs": (
        "Snapshot synchronization and rival-private perturbation with positive "
        "controls for every registered bot."
    ),
    "opponents_rs/tests/lifecycle.rs": (
        "Registry, reset, seat isolation, determinism and runner checks; Cha22 "
        "full matches against Starter in both seats."
    ),
    "opponents_rs/tests/oracle_parity.rs": (
        "Original Python action and state comparisons, continuous and resumed "
        "after mid-episode reconstruction, with localization and mutation checks; "
        "separate three-game Cha22 corpus."
    ),
    "opponents_rs/fixtures/oracle-cha22/MANIFEST.json": (
        "Original Python Cha22 oracle trace inventory (CPython 3.11), available "
        "actions and observed coverage."
    ),
}


def main() -> None:
    path = ROOT / checker.MANIFEST
    old = json.loads(path.read_text())
    current = checker.current_files(ROOT)
    originals = checker.reference_files(ROOT)

    def dest(reference: str) -> str:
        return reference.replace("engine_rs/", "opponents_rs/", 1)

    imported = [
        {
            "path": dest(reference),
            "reference_path": reference,
            "reference_sha256": digest,
            "sha256": checker.sha(current[dest(reference)]),
        }
        for reference, digest in checker.IMPORT_HASHES.items()
    ]
    adapted = [
        {
            "path": dest(reference),
            "reference_path": reference,
            "reference_sha256": digest,
            "sha256": checker.sha(current[dest(reference)]),
            "adaptation": checker.VIEW_ADAPTATION,
        }
        for reference, digest in checker.ADAPTED_HASHES.items()
    ]
    for entry in adapted:
        assert current[entry["path"]] == checker.view_adapted(
            originals[entry["reference_path"]], entry["path"]
        ), entry["path"]
    notices = [
        {
            "path": notice,
            "source_repo": repo,
            "source_commit": version,
            "source_path": source,
            "sha256": digest,
        }
        for notice, (repo, version, source, digest) in checker.NOTICES.items()
    ]
    authored_paths = [e["path"] for e in old["authored"]]
    if checker.CHA22_ORACLE_DIR + "/MANIFEST.json" not in authored_paths:
        authored_paths.append(checker.CHA22_ORACLE_DIR + "/MANIFEST.json")
    previous_reasons = {e["path"]: e["reason"] for e in old["authored"]}
    authored = [
        {
            "path": entry,
            "sha256": checker.sha(current[entry]),
            "reason": REASONS.get(entry, previous_reasons.get(entry, "")),
        }
        for entry in authored_paths
    ]
    python_oracles = [e for e in old["python_oracles"] if e["bot"] != "cha22"]
    python_oracles.append(
        {
            "bot": "cha22",
            "source_repo": checker.CHA22_SOURCE_REPO,
            "source_commit": checker.CHA22_SOURCE_VERSION,
            "files": [
                {
                    "path": checker.CHA22_SOURCE_PATH,
                    "sha256": checker.CHA22_SOURCE_SHA256,
                }
            ],
            "provenance": (
                "Public notebook output main.py, Apache-2.0 with its upstream "
                "notices (opponents_rs/notices/cha22); full ig_agent entry. Not "
                "copied here."
            ),
        }
    )
    parity = json.loads((Path(__file__).parent / "cha22-parity.json").read_text())
    counts = {entry["path"]: entry for entry in parity}
    generated = json.loads(
        current[checker.CHA22_ORACLE_DIR + "/MANIFEST.json"].decode()
    )
    traces = [e for e in old["oracle_traces"] if "/oracle-cha22/" not in e["path"]]
    for entry in generated["traces"]:
        report = counts[entry["path"]]
        assert (
            report["mismatch"] is None
            and report["matched_actions"] == report["compared_actions"]
        ), report
        traces.append(
            {
                "path": f"{checker.CHA22_ORACLE_DIR}/{entry['path']}",
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "seed": entry["seed"],
                "policies": entry["policies"],
                "transitions": entry["transitions"],
                "compared_actions": report["compared_actions"],
            }
        )
    manifest = {
        "schema_version": 1,
        "reference_commit": checker.PIN,
        "imported": imported,
        "adapted": adapted,
        "notices": notices,
        "authored": authored,
        "python_oracles": python_oracles,
        "oracle_traces": traces,
        "trace_budget_bytes": checker.TRACE_BUDGET,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
