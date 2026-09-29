"""Register Task 7.1's standalone import without changing frozen engine entries.

Accept only the pinned integration, committed run-1 stop, or exact final output.
Preserve run-1 receipts and retained/authored inventory; update exactly five
excluded reasons to identify the copies now owned by the opponent manifest.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "b8747b6e8acece5f561d09a75bb914364a60ac05"
STOPPED = "21d0f45effdf76302cc4bf3b163601038d0d2f8e"
MANIFEST = "engine_rs/TRIM_MANIFEST.json"
OPS = "ops/rebuild-2026-09-29/7.1"
CHANGES = {
    "docs/rules-parity-coverage.md": (
        "document snapshot-view, lifecycle and original-Python parity coverage."
    ),
    "cookbook/references/frozen-engine-api-blocks-standalone-opponent-import.md": (
        "retire the stop-only note in favor of the snapshot-view Reference."
    ),
    "cookbook/references/snapshot-view-isolates-byte-exact-evaluation-opponents.md": (
        "record implemented import and qualification limits; correct stop attribution."
    ),
    "cookbook/references/index.md": "index the current opponent-import Reference.",
    "cookbook/log.md": "record the import and correct the historical stop attribution.",
    "justfile": "check opponent byte custody and standalone fmt, Clippy and tests.",
    "scripts/check_opponent_import.py": (
        "validate strict opponent import and trace custody."
    ),
    "tests/tools/test_check_opponent_import.py": (
        "exercise opponent custody mutation attacks."
    ),
    "scripts/kaggriculture_parity/generate_traces.py": (
        "load pinned original Python submissions with isolated per-seat lifecycle."
    ),
    "tests/scripts/test_kaggriculture_parity.py": (
        "test submission custody and independent original-Python agent instances."
    ),
    "tests/owl/kaggriculture/test_opponents.py": (
        "declare learned-seat checks skipped until the Task 1.4 binding exists."
    ),
    **{
        f"opponents_rs/{name}": reason
        for name, reason in {
            "Cargo.toml": "declare the standalone edition-2024 evaluation crate.",
            "Cargo.lock": "pin the opponent crate's offline dependency closure.",
            "README.md": "document view, execution and parity qualification limits.",
            "OPPONENT_MANIFEST.json": "pin imported/authored and oracle custody.",
            "src/lib.rs": "isolate engine ownership behind a current snapshot view.",
            "src/native_agents.rs": "declare only four byte-exact controller modules.",
            "src/registry.rs": "enforce per-seat, per-episode controller lifecycle.",
            "src/runner.rs": "execute bounded default-config evaluation matches.",
            "src/view_tests.rs": "check the view and private-state visibility.",
            "tests/lifecycle.rs": "check lifecycle, determinism and config bounds.",
            "tests/oracle_parity.rs": "compare original Python actions and state.",
            "fixtures/e776-kenjo-trace.json": "preserve E776 executable policy data.",
            "fixtures/oracle/MANIFEST.json": "freeze generated Python oracle metadata.",
            "fixtures/oracle/oracle-00-starter-vs-r04.jsonl.gz": (
                "freeze original Starter/R04 observations and actions."
            ),
            **{
                f"src/native_agents/{bot}.rs": "preserve pinned controller bytes."
                for bot in ("starter", "r04", "ecobot", "e776")
            },
        }.items()
    },
    **{
        f"{OPS}/{name}": reason
        for name, reason in {
            "plan.md": "separate planned checks from actual results.",
            "results.md": "report actual commands, scope, coverage and remaining work.",
            "update_trim_manifest.py": (
                "register import reasons and task inventory from pinned inputs."
            ),
            "test_update_trim_manifest.py": (
                "test preservation, committed inputs, idempotency and drift refusal."
            ),
        }.items()
    },
    **{
        f"{OPS}/run2/{name}": reason
        for name, reason in {
            "clippy-first.log": "retain imported-code lint findings before allowances.",
            "clippy-second.log": (
                "retain authored-code lint findings before correction."
            ),
            "clippy-green.log": "record initial standalone Clippy success.",
            "clippy-final.log": "record final standalone Clippy success.",
            "custody-red.log": "retain missing-checker test-first failure.",
            "custody-content-red.log": "retain strict trace-metadata test failures.",
            "custody-coverage-red.log": "reject invented positive coverage counts.",
            "custody-green.log": "record custody mutation tests and binding skips.",
            "custody-check.log": "record final opponent custody verification.",
            "lifecycle-red.log": (
                "retain registry/lifecycle test-first compile failures."
            ),
            "reset-red.log": "retain same-episode reset bypass failure.",
            "reset-green.log": "record fresh-episode reset enforcement.",
            "rust-first.log": "retain initial view/controller compile findings.",
            "rust-first-green.log": "record initial view/controller test success.",
            "rust-lifecycle-visibility.log": (
                "record bounded lifecycle/visibility checks."
            ),
            "oracle-red.log": "retain original-submission loader test-first failures.",
            "oracle-green.log": "record original-submission loader test success.",
            "oracle-tests.log": "record trace generator and coverage test results.",
            "oracle-mypy.log": "record trace generator static checking.",
            "oracle-comparator-red.log": "retain comparator test-first failure.",
            "oracle-comparator-green.log": "record comparator regression success.",
            "oracle-first-generation.log": "record bounded original-Python generation.",
            "oracle-first-summary.json": "record first Python game's runtime/counts.",
            "oracle-first-parity.log": (
                "retain typed-reward comparator harness failure."
            ),
            "oracle-second-parity.log": "record original R04 action mismatch.",
            "oracle-parity.json": "pin actual compared/matched action denominators.",
            "coverage.json": "record exact trace coverage and unmeasured categories.",
            "localize_r04_python.py": "localize original R04 decision-path divergence.",
            "r04-diagnostic.rs": "reproduce native R04 debug decisions in scratch.",
            "r04-native-debug.json": (
                "retain native decision-path localization evidence."
            ),
            "r04-python-debug.json": (
                "retain original and counterfactual Python decisions."
            ),
            "r04-mismatch.md": (
                "explain first R04 mismatch and bounded source attribution."
            ),
            "write_opponent_manifest.py": "freeze explicit opponent custody inventory.",
            "updater-red.log": "retain six failures before resumed-updater support.",
            "updater-green.log": "record six updater tests and 12 mutation subtests.",
            "updater-green-final.log": (
                "record final updater tests and byte idempotency."
            ),
            "opponents-test.log": "record required opponent crate test status.",
            "engine-test.log": "record frozen engine regression results.",
            "engine-trim.log": "record final engine trim verification.",
            "opponent-import.log": "record required opponent custody checker status.",
            "targeted-tests.log": "record required Python checks and binding skips.",
            "prepare.log": "record full requested repository preparation status.",
            "py-prepare.log": "record required Python preparation status.",
            "rs-prepare.log": "record required Rust preparation status.",
            "commands.json": "record final command arguments, statuses and counts.",
            "closure.log": "record final docs, byte preservation and inventory checks.",
            "inventory.json": "list final changed paths and their task purposes.",
        }.items()
    },
}

IMPORTED = {
    **{
        f"engine_rs/src/native_agents/{name}.rs": (
            f"opponents_rs/src/native_agents/{name}.rs"
        )
        for name in ("starter", "r04", "ecobot", "e776")
    },
    "engine_rs/fixtures/e776-kenjo-trace.json": (
        "opponents_rs/fixtures/e776-kenjo-trace.json"
    ),
}


def render(
    current: bytes, baseline: bytes, stopped: bytes, changes: dict[str, str]
) -> bytes:
    manifest = json.loads(stopped)
    original = json.loads(baseline)
    for key in original.keys() | manifest.keys():
        if key != "non_engine_changes" and original[key] != manifest[key]:
            raise ValueError(f"unexpected committed {key} drift")
    excluded = {entry["path"]: entry for entry in manifest["excluded"]}
    if not IMPORTED.keys() <= excluded.keys():
        raise ValueError("missing imported reference path in excluded inventory")
    for path, destination in IMPORTED.items():
        excluded[path]["reason"] += (
            f"; byte-exact copy imported to {destination} "
            "under OPPONENT_MANIFEST.json (Task 7.1)"
        )
    entries = manifest["non_engine_changes"]
    existing = {entry["path"]: entry for entry in entries}
    for path, reason in changes.items():
        if path in existing:
            existing[path]["reason"] += f" Task 7.1: {reason}"
        else:
            entries.append({"path": path, "reason": f"Task 7.1: {reason}"})
    result = (json.dumps(manifest, indent=2) + "\n").encode()
    if current not in (baseline, stopped, result):
        raise ValueError("unexpected trim manifest input; refusing to overwrite")
    return result


def main() -> None:
    def read(revision: str) -> bytes:
        return subprocess.run(
            ["git", "show", f"{revision}:{MANIFEST}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout

    baseline, stopped = read(BASE), read(STOPPED)
    target = ROOT / MANIFEST
    target.write_bytes(render(target.read_bytes(), baseline, stopped, CHANGES))
    print(
        "Task 7.1: import inventory updated; retained/authored engine entries unchanged"
    )


if __name__ == "__main__":
    main()
