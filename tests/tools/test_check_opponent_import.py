from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_opponent_import", ROOT / "scripts/check_opponent_import.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


@lru_cache(maxsize=1)
def original_fixture() -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    originals = checker.reference_files(ROOT)
    current = {
        path.replace("engine_rs/", "opponents_rs/", 1): data
        for path, data in originals.items()
    }
    current["opponents_rs/src/lib.rs"] = b"// authored registry\n"
    with gzip.open(
        ROOT / "engine_rs/fixtures/generated/gen-starter-vs-random.jsonl.gz", "rt"
    ) as stream:
        records = [json.loads(stream.readline()) for _ in range(2)]
    records[0]["source"]["policies"] = ["builtin:starter", "sibling:r04"]
    records[0]["transitions"] = 1
    path = "opponents_rs/fixtures/oracle/test.jsonl.gz"
    current[path] = encode(records)
    trace = {
        "path": path,
        "sha256": checker.sha(current[path]),
        "bytes": len(current[path]),
        "seed": records[0]["seed"],
        "policies": records[0]["source"]["policies"],
        "transitions": 1,
        "compared_actions": [1, 1],
    }
    generated = {
        "schema_version": 1,
        "format": checker.TRACE_FORMAT,
        "generator": checker.GENERATOR,
        "kaggle_environments_version": checker.KAGGLE_VERSION,
        "python_engine_sha256": checker.PYTHON_ENGINE_SHA256,
        "byte_budget": checker.TRACE_BUDGET,
        "traces": [
            {
                **{
                    key: value
                    for key, value in trace.items()
                    if key != "compared_actions"
                },
                "available_actions": [1, 1],
                "path": "test.jsonl.gz",
                "policy_seed": records[0]["source"]["policy_seed"],
                "config_variant": "default",
                "python_runtime": "3.11.15",
                "rejected": 0,
                "coverage": [
                    {**dict.fromkeys(checker.COVERAGE_KEYS, 0), "openings": 1}
                    for _ in range(2)
                ],
            }
        ],
    }
    current[checker.ORACLE_MANIFEST] = json.dumps(generated).encode()
    manifest = {
        "schema_version": 1,
        "reference_commit": checker.PIN,
        "imported": [
            {
                "path": path.replace("engine_rs/", "opponents_rs/", 1),
                "reference_path": path,
                "reference_sha256": checker.sha(data),
                "sha256": checker.sha(data),
            }
            for path, data in originals.items()
        ],
        "authored": [
            {"path": path, "sha256": checker.sha(current[path]), "reason": "test"}
            for path in ["opponents_rs/src/lib.rs", checker.ORACLE_MANIFEST]
        ],
        # Committed receipts: the default check must not need the sibling repo.
        "python_oracles": json.loads((ROOT / checker.MANIFEST).read_bytes())[
            "python_oracles"
        ],
        "oracle_traces": [trace],
        "trace_budget_bytes": checker.TRACE_BUDGET,
    }
    return manifest, originals, current


def fixture() -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    return copy.deepcopy(original_fixture())


def encode(records: list[dict[str, Any]]) -> bytes:
    return gzip.compress(
        ("\n".join(json.dumps(record) for record in records) + "\n").encode(), mtime=0
    )


def test_valid_inventory_and_trace_content_pass() -> None:
    checker.verify(*fixture())


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("reference_drift", "reference hash"),
        ("extra_file", "current inventory"),
        ("missing_file", "current inventory"),
        ("edited_import", "import hash"),
        ("edited_import_and_hash", "byte-exact"),
        ("authored_hash", "authored hash"),
        ("trace_hash", "trace hash"),
        ("trace_size", "trace size"),
        ("budget_override", "budget must remain"),
        ("over_budget", "budget"),
        ("pin_drift", "reference_commit"),
        ("unknown_key", "schema keys"),
        ("duplicate", "duplicate"),
        ("extra_import", "import inventory"),
        ("python_hash", "Python source hash"),
        ("python_missing", "Python source inventory"),
        ("python_pin", "source_commit"),
        ("python_repo", "source_repo"),
        ("no_provenance", "provenance"),
        ("trace_seed", "seed"),
        ("trace_policies", "policies"),
        ("trace_count", "transitions"),
        ("compared_count", "compared_actions"),
        ("bool_count", "integer"),
    ],
)
def test_custody_mutations_fail(attack: str, message: str) -> None:
    manifest, originals, current = fixture()
    imported = manifest["imported"][0]
    trace = manifest["oracle_traces"][0]
    if attack == "reference_drift":
        originals[imported["reference_path"]] += b"edited"
    elif attack == "extra_file":
        current["opponents_rs/surprise.txt"] = b"surprise"
    elif attack == "missing_file":
        del current[imported["path"]]
    elif attack in {"edited_import", "edited_import_and_hash"}:
        current[imported["path"]] += b"edited"
        if attack == "edited_import_and_hash":
            imported["sha256"] = checker.sha(current[imported["path"]])
    elif attack == "authored_hash":
        current["opponents_rs/src/lib.rs"] += b"edited"
    elif attack == "trace_hash":
        current[trace["path"]] += b"edited"
    elif attack == "trace_size":
        trace["bytes"] += 1
    elif attack == "budget_override":
        manifest["trace_budget_bytes"] += 1
    elif attack == "over_budget":
        current[trace["path"]] += b"x" * checker.TRACE_BUDGET
        trace["sha256"] = checker.sha(current[trace["path"]])
        trace["bytes"] = len(current[trace["path"]])
    elif attack == "pin_drift":
        manifest["reference_commit"] = "0" * 40
    elif attack == "unknown_key":
        manifest["unknown"] = True
    elif attack == "duplicate":
        manifest["authored"].append(manifest["authored"][0])
    elif attack == "extra_import":
        imported["reference_path"] = "engine_rs/src/native_agents/mod.rs"
    elif attack == "python_hash":
        manifest["python_oracles"][1]["files"][0]["sha256"] = "0" * 64
    elif attack == "python_missing":
        manifest["python_oracles"][1]["files"].pop()
    elif attack == "python_pin":
        manifest["python_oracles"][1]["source_commit"] = "0" * 40
    elif attack == "python_repo":
        manifest["python_oracles"][1]["source_repo"] = "/other"
    elif attack == "no_provenance":
        manifest["python_oracles"][2]["provenance"] = ""
    elif attack == "trace_seed":
        trace["seed"] += 1
    elif attack == "trace_policies":
        trace["policies"].reverse()
    elif attack == "trace_count":
        trace["transitions"] += 1
    elif attack == "compared_count":
        trace["compared_actions"][0] += 1
    elif attack == "bool_count":
        trace["transitions"] = True
    with pytest.raises(ValueError, match=message):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/x",
        "../x",
        "opponents_rs/../x",
        "opponents_rs//x",
        "opponents_rs/./x",
        "opponents_rs\\x",
        "opponents_rs/.git/x",
        "opponents_rs/target/x",
    ],
)
def test_unsafe_inventory_paths_fail(path: str) -> None:
    manifest, originals, current = fixture()
    manifest["authored"][0]["path"] = path
    with pytest.raises(ValueError, match="path"):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    "section", ["imported", "authored", "python_oracles", "oracle_traces"]
)
def test_nested_unknown_keys_fail(section: str) -> None:
    manifest, originals, current = fixture()
    manifest[section][0]["unknown"] = True
    with pytest.raises(ValueError, match="schema keys"):
        checker.verify(manifest, originals, current)


def test_symlink_inventory_fails(tmp_path: Path) -> None:
    (tmp_path / "opponents_rs").mkdir()
    (tmp_path / "opponents_rs/link").symlink_to(ROOT / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        checker.current_files(tmp_path)


def test_target_is_the_only_ignored_directory(tmp_path: Path) -> None:
    (tmp_path / "opponents_rs/target").mkdir(parents=True)
    (tmp_path / "opponents_rs/target/arbitrary").write_text("ignored")
    (tmp_path / "opponents_rs/.gitignore").write_text("visible")
    assert checker.current_files(tmp_path) == {"opponents_rs/.gitignore": b"visible"}


def test_checker_cli_reports_failure(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "python",
            str(ROOT / "scripts/check_opponent_import.py"),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "opponent import check failed" in completed.stderr


def refresh_authored(manifest: dict[str, Any], current: dict[str, bytes]) -> None:
    for entry in manifest["authored"]:
        entry["sha256"] = checker.sha(current[entry["path"]])


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("unknown_key", "schema keys"),
        ("missing_trace", "inventory"),
        ("wrong_hash", "sha256"),
        ("bool_count", "integer"),
        ("wrong_budget", "budget"),
        ("negative_coverage", "negative coverage"),
        ("invented_coverage", "coverage differs"),
        ("invented_replay", "coverage differs"),
        ("unsafe_trace", "path"),
        ("python_312_runtime", r"CPython 3\.11"),
        ("missing_runtime", "schema keys"),
    ],
)
def test_rehashed_oracle_manifest_attacks_fail(attack: str, message: str) -> None:
    manifest, originals, current = fixture()
    generated = json.loads(current[checker.ORACLE_MANIFEST])
    entry = generated["traces"][0]
    if attack == "unknown_key":
        entry["unknown"] = 1
    elif attack == "missing_trace":
        generated["traces"] = []
    elif attack == "wrong_hash":
        entry["sha256"] = "0" * 64
    elif attack == "bool_count":
        entry["transitions"] = True
    elif attack == "wrong_budget":
        generated["byte_budget"] += 1
    elif attack == "negative_coverage":
        entry["coverage"][0]["hires"] = -1
    elif attack == "invented_coverage":
        entry["coverage"][0]["hires"] = 1
    elif attack == "invented_replay":
        entry["coverage"][0]["mid_episode_replay"] = 1
    elif attack == "unsafe_trace":
        entry["path"] = "../test.jsonl.gz"
    elif attack == "python_312_runtime":
        entry["python_runtime"] = "3.12.13"
    elif attack == "missing_runtime":
        del entry["python_runtime"]
    current[checker.ORACLE_MANIFEST] = json.dumps(generated).encode()
    refresh_authored(manifest, current)
    with pytest.raises(ValueError, match=message):
        checker.verify(manifest, originals, current)


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("wrong_seed", "seed"),
        ("wrong_policies", "policies"),
        ("bool_transitions", "integer"),
        ("custom_configuration", "non-default configuration"),
        ("malformed_configuration", "configuration: expected object"),
        ("wrong_step", "non-contiguous"),
        ("unknown_record", "unknown record"),
        ("wrong_action_seats", "both seats"),
        ("missing_record", "transitions"),
    ],
)
def test_rehashed_trace_content_attacks_fail(attack: str, message: str) -> None:
    manifest, originals, current = fixture()
    entry = manifest["oracle_traces"][0]
    records = [
        json.loads(line)
        for line in gzip.decompress(current[entry["path"]]).splitlines()
    ]
    if attack == "wrong_seed":
        records[0]["seed"] += 1
    elif attack == "wrong_policies":
        records[0]["source"]["policies"].reverse()
    elif attack == "bool_transitions":
        records[0]["transitions"] = True
    elif attack == "custom_configuration":
        records[0]["configuration"]["farmHandCostMult"] = 2
    elif attack == "malformed_configuration":
        records[0]["configuration"] = []
    elif attack == "wrong_step":
        records[1]["from_step"] += 1
    elif attack == "unknown_record":
        records[1]["type"] = "unknown"
    elif attack == "wrong_action_seats":
        records[1]["actions"].pop()
    elif attack == "missing_record":
        records.pop()
    current[entry["path"]] = encode(records)
    entry["sha256"] = checker.sha(current[entry["path"]])
    entry["bytes"] = len(current[entry["path"]])
    generated = json.loads(current[checker.ORACLE_MANIFEST])
    generated["traces"][0]["sha256"] = entry["sha256"]
    generated["traces"][0]["bytes"] = entry["bytes"]
    current[checker.ORACLE_MANIFEST] = json.dumps(generated).encode()
    refresh_authored(manifest, current)
    with pytest.raises(ValueError, match=message):
        checker.verify(manifest, originals, current)


def test_duplicate_json_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        checker._loads('{"schema_version": 1, "schema_version": 1}')


def test_nonfinite_json_values_are_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        checker._loads('{"reward": NaN}')


SIBLING_ABSENT = not Path(checker.PYTHON_REPO, ".git").exists()


def test_default_check_needs_no_sibling_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pods and containers run `just prepare` without the owner's sibling tree."""

    def unavailable() -> None:
        raise OSError("sibling repository unavailable")

    monkeypatch.setattr(checker, "_python_files", unavailable)
    manifest, originals, current = fixture()
    checker.verify(manifest, originals, current)
    with pytest.raises(OSError, match="unavailable"):
        checker.verify(manifest, originals, current, original_sources=True)


@pytest.mark.skipif(SIBLING_ABSENT, reason="original sources live in the sibling repo")
def test_original_sources_mode_checks_every_dependency_hash() -> None:
    manifest, originals, current = fixture()
    checker.verify(manifest, originals, current, original_sources=True)
    e776 = next(e for e in manifest["python_oracles"] if e["bot"] == "e776")
    dependency = next(f for f in e776["files"] if "/agents/" in f["path"])
    dependency["sha256"] = "0" * 64
    checker.verify(manifest, originals, current)  # structural pins cannot see it
    with pytest.raises(ValueError, match="Python source hash"):
        checker.verify(manifest, originals, current, original_sources=True)


def test_structural_check_pins_entry_sources() -> None:
    manifest, originals, current = fixture()
    ecobot = next(e for e in manifest["python_oracles"] if e["bot"] == "ecobot")
    main = next(f for f in ecobot["files"] if f["path"].endswith("main.py"))
    main["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="Python source hash"):
        checker.verify(manifest, originals, current)
