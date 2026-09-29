"""Refresh Task 7.1 custody only for its explicitly reviewed crate inventory."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "check_opponent_import", ROOT / "scripts/check_opponent_import.py"
)
assert SPEC is not None
assert SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

AUTHORED = {
    "opponents_rs/README.md": (
        "Public controller-view API, default-config support, information "
        "boundary, provenance notices and unresolved parity limits."
    ),
    "opponents_rs/Cargo.toml": (
        "Standalone evaluation crate generated with offline Cargo dependency commands."
    ),
    "opponents_rs/Cargo.lock": (
        "Cargo-generated independent locked dependency resolution."
    ),
    "opponents_rs/src/lib.rs": (
        "V3 controller view owns a frozen engine plus refreshed public snapshot; "
        "private fib compatibility helper."
    ),
    "opponents_rs/src/native_agents.rs": (
        "Explicit four-controller module declarations and narrow imported-code "
        "lint allowances."
    ),
    "opponents_rs/src/registry.rs": (
        "Four-key registry and independent seat/episode controller lifecycle."
    ),
    "opponents_rs/src/runner.rs": (
        "Default-config-only bounded native match runner and engine acceptance records."
    ),
    "opponents_rs/src/view_tests.rs": (
        "Snapshot synchronization and rival-private perturbation with positive "
        "controls."
    ),
    "opponents_rs/tests/lifecycle.rs": (
        "Registry, reset, seat isolation, determinism and runner checks."
    ),
    "opponents_rs/tests/oracle_parity.rs": (
        "Original Python action and state comparisons, localization and mutation "
        "checks."
    ),
    "opponents_rs/fixtures/oracle/MANIFEST.json": (
        "Original Python oracle trace inventory, available actions and observed "
        "coverage."
    ),
}


def main() -> None:
    originals = checker.reference_files(ROOT)
    current = checker.current_files(ROOT)
    generated = json.loads(current[checker.ORACLE_MANIFEST])
    results = json.loads((Path(__file__).parent / "oracle-parity.json").read_text())
    reports = {entry["path"]: entry for entry in results}
    imported = []
    for reference, data in originals.items():
        path = reference.replace("engine_rs/", "opponents_rs/", 1)
        if (
            current[path] != data
            or checker.sha(data) != checker.IMPORT_HASHES[reference]
        ):
            raise ValueError(f"{path}: refusing changed import")
        imported.append(
            {
                "path": path,
                "reference_path": reference,
                "reference_sha256": checker.sha(data),
                "sha256": checker.sha(current[path]),
            }
        )
    traces = []
    for entry in generated["traces"]:
        report = reports[entry["path"]]
        if any(report[key] != entry[key] for key in ("seed", "policies")):
            raise ValueError(f"{entry['path']}: comparison report identity mismatch")
        traces.append(
            {
                "path": f"{checker.ORACLE_DIR}/{entry['path']}",
                **{
                    key: entry[key]
                    for key in ("sha256", "bytes", "seed", "policies", "transitions")
                },
                "compared_actions": report["compared_actions"],
            }
        )
    expected = {
        *AUTHORED,
        *(entry["path"] for entry in imported),
        *(entry["path"] for entry in traces),
    }
    if set(current) != expected:
        raise ValueError(f"unexpected inventory: {sorted(set(current) ^ expected)}")
    manifest = {
        "schema_version": 1,
        "reference_commit": checker.PIN,
        "imported": imported,
        "authored": [
            {"path": path, "sha256": checker.sha(current[path]), "reason": reason}
            for path, reason in AUTHORED.items()
        ],
        "python_oracles": checker.python_oracle_receipts(),
        "oracle_traces": traces,
        "trace_budget_bytes": checker.TRACE_BUDGET,
    }
    checker.verify(manifest, originals, current)
    (ROOT / checker.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {checker.MANIFEST}; parity outcome is unchanged.")


if __name__ == "__main__":
    main()
