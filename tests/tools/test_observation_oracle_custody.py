from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import signal
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "observation_oracle",
    ROOT / "scripts/kaggriculture_observation_oracle/regenerate.py",
)
assert SPEC is not None
assert SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def fixture(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    with gzip.open(ROOT / "engine_rs/fixtures/episode-95324500.jsonl.gz", "rt") as f:
        original = json.loads(next(f))
        transition = json.loads(next(f))
    header = {
        key: original[key]
        for key in (
            "format",
            "seed",
            "configuration",
            "shop_schedule",
            "rng_schedule",
            "initial",
            "terminal_banks",
            "transitions",
        )
    }
    header.update(shop_schedule=[], rng_schedule=[], terminal_banks=[], transitions=0)
    second = copy.deepcopy(header)
    second["initial"] = {
        "public": transition["expected"],
        "privates": transition["privates"],
    }
    rows = [
        {
            "record_id": f"official:95324500:{i}",
            "source": {"kind": "official", "episode": 95324500, "step": i},
            "header": h,
        }
        for i, h in enumerate((header, second))
    ]
    features = b"".join(struct.pack("<f", i + 0.25) * 8176 for i in range(4))
    identity = {
        "root_commit": "a" * 40,
        "dirty_files": {"src/kaggriculture/oracle_corpus.rs": "b" * 64},
        "engine": {k: "b" * 64 for k in ("trim_manifest", "lib", "lock")},
        "reference_commit": "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0",
        "feature_sha256": (
            "24a7d09f9ddf8196aed7e5dc3b18d7590562370d697c8407f95b4cc8b4f9a6ab"
        ),
        "recorder_sha256": "b" * 64,
        "driver_sha256": "b" * 64,
        "producer_sha256": "b" * 64,
        "versions": {k: "test" for k in ("rustc", "cargo", "python")},
        "argv": [["tiny-fixture"]],
        "official_fixtures": {
            str(k): "b" * 64 for k in (95324500, 95901360, 95921764, 95990191)
        },
    }
    manifest = {
        "format": "kaggriculture-observation-oracle-f32le-gzip-v1",
        "schema_version": 1,
        "observation_schema": 3,
        "contract_version": 4,
        "source_identity": identity,
        "profiles": oracle.PROFILES,
        "policy": "observation-corpus-v2",
        "seed_runs": [],
        "source_counts": {"official": 2, "seeded": 0, "dense": 0},
        "records": [
            {
                "record_id": r["record_id"],
                "source": r["source"],
                "header_sha256": sha(encoded(r["header"])),
                "seat_sha256": [
                    sha(features[(2 * i + s) * 32704 : (2 * i + s + 1) * 32704])
                    for s in range(2)
                ],
            }
            for i, r in enumerate(rows)
        ],
        "files": {},
        "reference_shape": [2, 2, 8176],
        "dtype": "<f4",
        "coverage": oracle.count_coverage(rows),
    }
    install(directory, manifest, rows, features)
    return manifest, rows, features


def install(
    directory: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    features: bytes,
) -> None:
    directory.mkdir(exist_ok=True)
    for name, raw in (
        ("states.jsonl.gz", b"".join(encoded(r) + b"\n" for r in rows)),
        ("reference.f32le.gz", features),
    ):
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        (directory / name).write_bytes(compressed)
        manifest["files"][name] = {
            "compressed_size": len(compressed),
            "expanded_size": len(raw),
            "compressed_sha256": sha(compressed),
            "expanded_sha256": sha(raw),
        }
    (directory / "manifest.json").write_bytes(encoded(manifest))


def test_two_record_fixture_validates(tmp_path: Path) -> None:
    fixture(tmp_path)
    oracle.validate_corpus(tmp_path, expected_records=2)


@pytest.mark.parametrize(
    "damage",
    [
        "byte",
        "header_order",
        "duplicate",
        "missing",
        "seat_swap",
        "length",
        "version",
        "unknown",
        "source",
        "duplicate_json_key",
        "coverage",
    ],
)
def test_custody_rejects_corruption_even_with_rehashed_files(
    tmp_path: Path, damage: str
) -> None:
    manifest, rows, features = fixture(tmp_path)
    if damage == "byte":
        features = bytes([features[0] ^ 1]) + features[1:]
    elif damage == "header_order":
        inv = rows[0]["header"]["initial"]["privates"][0]["shed"]
        key = next(iter(inv))
        inv[key] = inv.pop(key)
    elif damage == "duplicate":
        rows[1]["record_id"] = rows[0]["record_id"]
        manifest["records"][1]["record_id"] = rows[0]["record_id"]
    elif damage == "missing":
        rows.pop()
    elif damage == "seat_swap":
        features = features[32704:65408] + features[:32704] + features[65408:]
    elif damage == "length":
        features = features[:-4]
    elif damage == "version":
        manifest["schema_version"] = 2
    elif damage == "unknown":
        manifest["unreviewed"] = True
    elif damage == "source":
        rows[0]["source"]["kind"] = "borrowed"
    elif damage == "coverage":
        manifest["coverage"]["non_synthetic"]["actor_gt16_states"] += 4
    install(tmp_path, manifest, rows, features)
    if damage == "duplicate_json_key":
        text = (tmp_path / "manifest.json").read_text()
        (tmp_path / "manifest.json").write_text(
            text.replace('"schema_version":1', '"schema_version":1,"schema_version":1')
        )
    errors = {
        "byte": "feature block hash mismatch",
        "header_order": "header bytes/key-order hash mismatch",
        "duplicate": "duplicate record id",
        "missing": "missing/extra records",
        "seat_swap": "feature block hash mismatch",
        "length": "reference byte length mismatch",
        "version": "unsupported schema/contract version",
        "unknown": "manifest: exact schema keys",
        "source": "invalid source tag",
        "duplicate_json_key": "duplicate JSON key",
        "coverage": "coverage counts disagree",
    }
    with pytest.raises(ValueError, match=errors[damage]):
        oracle.validate_corpus(tmp_path, expected_records=2)


def test_final_quota_cannot_be_satisfied_by_dense_states() -> None:
    counts = {name: 100 for name in oracle.COVERAGE_TAGS}
    counts["actor_gt16_states"] = 0
    with pytest.raises(ValueError, match=r"actor_gt16_states.*0.*4"):
        oracle.require_coverage(
            {
                "non_synthetic": counts,
                "dense": {name: 100 for name in counts},
                "shed_order_exception": False,
            }
        )


def test_gzip_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"raw\x00\x80" * 100)
    one, two = tmp_path / "one.gz", tmp_path / "two.gz"
    oracle.compress_file(source, one)
    oracle.compress_file(source, two)
    assert one.read_bytes() == two.read_bytes()
    assert gzip.decompress(one.read_bytes()) == source.read_bytes()


@pytest.mark.parametrize(
    "field",
    ["schema_version", "observation_schema", "contract_version", "reference_shape"],
)
def test_manifest_versions_and_dimensions_are_integers(
    tmp_path: Path, field: str
) -> None:
    manifest, rows, features = fixture(tmp_path)
    manifest[field] = (
        True
        if field == "schema_version"
        else (float(manifest[field]) if field != "reference_shape" else [2.0, 2, 8176])
    )
    install(tmp_path, manifest, rows, features)
    with pytest.raises(ValueError, match=r"schema/contract|shape/dtype"):
        oracle.validate_corpus(tmp_path, expected_records=2)


def test_record_source_step_cannot_disagree_with_header(tmp_path: Path) -> None:
    manifest, rows, features = fixture(tmp_path)
    rows[0]["source"]["step"] = 17
    manifest["records"][0]["source"]["step"] = 17
    install(tmp_path, manifest, rows, features)
    with pytest.raises(ValueError, match="source/header step"):
        oracle.validate_corpus(tmp_path, expected_records=2)


def test_header_hash_preserves_numeric_spelling_and_nested_key_order() -> None:
    raw = (
        b'{"source":{},"header":{"number":1e-7,"other":-0.0,'
        b'"keys":{"B":0,"A":1}},"record_id":"test"}\n'
    )
    assert (
        oracle.header_bytes(raw) == b'{"number":1e-7,"other":-0.0,"keys":{"B":0,"A":1}}'
    )
    assert oracle.header_bytes(raw) != encoded(json.loads(raw)["header"])


def test_dense_pair_allows_only_two_remove_reinsert_orders(tmp_path: Path) -> None:
    _, rows, _ = fixture(tmp_path)
    first = rows[0]["header"]
    first["initial"]["privates"][0]["inventories"][0] = {"WOOL": 0, "WHEAT": 7}
    second = copy.deepcopy(first)
    for mapping in (
        second["initial"]["privates"][0]["inventories"][0],
        second["initial"]["privates"][0]["shed"],
    ):
        key = next(iter(mapping))
        mapping[key] = mapping.pop(key)
    oracle.validate_dense_pair(first, second)
    second["initial"]["public"]["farms"][0]["money"] += 1
    with pytest.raises(ValueError, match="dense d30/d31"):
        oracle.validate_dense_pair(first, second)


def test_byteplane_round_trip_preserves_float_bits(tmp_path: Path) -> None:
    raw, planes, restored = tmp_path / "raw", tmp_path / "planes", tmp_path / "restored"
    raw.write_bytes(bytes(range(256)) * 8193)
    oracle.byteplanes(raw, planes, decode=False)
    oracle.byteplanes(planes, restored, decode=True)
    assert restored.read_bytes() == raw.read_bytes()


def test_shuffled_fixture_validates_the_original_seat_bytes(tmp_path: Path) -> None:
    manifest, rows, features = fixture(tmp_path)
    values = len(features) // 4
    planes = b"".join(features[plane::4] for plane in range(4))
    assert len(planes) == values * 4
    manifest["format"] = oracle.SHUFFLED_FORMAT
    install(tmp_path, manifest, rows, planes)
    oracle.validate_corpus(tmp_path, expected_records=2)


def test_bounded_command_reports_process_group_memory_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oracle, "group_rss", lambda _group: 1_000_000_000)
    with pytest.raises(RuntimeError, match="reason=aggregate RSS"):
        oracle.run([sys.executable, "-c", "import time; time.sleep(30)"])


def test_bounded_command_returns_exact_stdout() -> None:
    assert (
        oracle.run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\x00\\x80')"]
        )
        == b"\x00\x80"
    )


def test_preexisting_caller_memory_is_not_charged_to_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A large caller (pytest with torch under `just prepare`) already holds more
    # than the budget before launch; only growth after launch counts.
    real = oracle.group_rss
    caller = os.getpgrp()
    monkeypatch.setattr(
        oracle,
        "group_rss",
        lambda group: 1_500_000_000 if group == caller else real(group),
    )
    assert oracle.run([sys.executable, "-c", "print('ok')"]) == b"ok\n"


def test_caller_growth_after_launch_still_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = os.getpgrp()
    samples = iter([0])
    monkeypatch.setattr(
        oracle,
        "group_rss",
        lambda group: next(samples, 1_000_000_000) if group == caller else 0,
    )
    with pytest.raises(RuntimeError, match="reason=aggregate RSS"):
        oracle.run([sys.executable, "-c", "import time; time.sleep(30)"])


def test_shared_deadline_stops_child_group_before_outer_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = time.monotonic()
    monkeypatch.setattr(oracle, "RUN_DEADLINE", started + 0.08)
    with pytest.raises(RuntimeError, match="shared execution deadline"):
        oracle.run([sys.executable, "-c", "import time; time.sleep(30)"])
    assert time.monotonic() - started < 2


def test_group_rss_measures_a_live_child_group() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; x=bytearray(32*1024**2); "
            "print('ready',flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == b"ready\n"
        assert oracle.group_rss(process.pid) >= 32 * 1024**2
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_producer_failure_never_exports_or_installs_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "CARGO_BUILD_JOBS": "2",
        "CARGO_NET_OFFLINE": "true",
        "UV_OFFLINE": "true",
        "RAYON_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RUST_TEST_THREADS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    calls = []

    def fail_producer(argv: list[str], **_kwargs: object) -> bytes:
        if argv == ["git", "rev-parse", "HEAD"]:
            return b"a" * 40
        calls.append(argv)
        assert argv[:2] == ["cargo", "test"]
        raise RuntimeError("R1 actor_gt16_states: actual 0, required 4")

    monkeypatch.setattr(oracle, "run", fail_producer)

    def forbidden_export(_path: Path) -> dict[str, str]:
        pytest.fail("failed producer must never export the reference")

    monkeypatch.setattr(oracle, "export_reference", forbidden_export)
    output = tmp_path / "final"
    with pytest.raises(RuntimeError, match="R1 actor_gt16_states"):
        oracle.regenerate(output, oracle.PIN)
    assert len(calls) == 1
    assert not output.exists()


@pytest.mark.parametrize(
    "field", ["profile_int", "seed", "sampled_steps", "terminal_step"]
)
def test_recipe_metadata_rejects_float_integer_replacements(
    tmp_path: Path, field: str
) -> None:
    manifest, rows, features = fixture(tmp_path)
    manifest["profiles"] = copy.deepcopy(manifest["profiles"])
    manifest["seed_runs"] = [
        {
            "seed": 11001,
            "profile": 0,
            "sampled_steps": [k * 94 // 15 for k in range(16)],
            "action_sha256": "b" * 64,
            "final_snapshot_sha256": "b" * 64,
            "terminal_step": 95,
        }
    ]
    if field == "profile_int":
        manifest["profiles"][0][0] = 96.0
    elif field == "sampled_steps":
        manifest["seed_runs"][0][field][0] = False
    else:
        manifest["seed_runs"][0][field] = float(manifest["seed_runs"][0][field])
    install(tmp_path, manifest, rows, features)
    with pytest.raises(ValueError, match="integer"):
        oracle.validate_corpus(tmp_path, expected_records=2)


@pytest.mark.parametrize("corrupt", [False, True])
def test_reference_export_checks_archive_bytes_against_git_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    path = "engine_rs/src/myolie_features.rs"
    original = b"reference encoder\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        directory = tarfile.TarInfo("engine_rs/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        file = tarfile.TarInfo(path)
        content = b"changed" if corrupt else original
        file.size = len(content)
        archive.addfile(file, io.BytesIO(content))

    def fake_git(argv: list[str]) -> bytes:
        if argv[1] == "archive":
            return buffer.getvalue()
        if argv[1] == "ls-tree":
            return (path + "\n").encode()
        assert argv[1] == "show"
        return original

    monkeypatch.setattr(oracle, "run", fake_git)
    monkeypatch.setattr(oracle, "FEATURE_SHA", sha(original))
    if corrupt:
        with pytest.raises(ValueError, match="reference export bytes changed"):
            oracle.export_reference(tmp_path)
    else:
        assert oracle.export_reference(tmp_path) == {path: sha(original)}


def regeneration_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift_phase: str = "",
    drift_path: str = "",
    corrupt_recorder: str = "",
) -> tuple[Path, list[str]]:
    """Exercise custody around mocked execution, independently of R1 admission."""
    _manifest, rows, _features = fixture(tmp_path / "headers")
    root = tmp_path / "source"
    paths = [
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
        "src/kaggriculture/mod.rs",
        "src/kaggriculture/config.rs",
        "src/kaggriculture/buffers.rs",
        "src/kaggriculture/observe.rs",
        "src/kaggriculture/tests.rs",
        "src/kaggriculture/oracle_corpus.rs",
        "scripts/kaggriculture_observation_oracle/record.rs",
        "scripts/kaggriculture_observation_oracle/regenerate.py",
        "engine_rs/TRIM_MANIFEST.json",
        "engine_rs/Cargo.toml",
        "engine_rs/Cargo.lock",
        "engine_rs/src/lib.rs",
        "engine_rs/src/econ_attrib.rs",
        "engine_rs/src/py_random.rs",
        *(
            f"engine_rs/fixtures/episode-{episode}.jsonl.gz"
            for episode in oracle.EPISODES
        ),
    ]
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((path + " before execution\n").encode())
    monkeypatch.setattr(oracle, "ROOT", root)
    monkeypatch.setattr(
        oracle,
        "__file__",
        str(root / "scripts/kaggriculture_observation_oracle/regenerate.py"),
    )
    for key, value in {
        "CARGO_BUILD_JOBS": "2",
        "CARGO_NET_OFFLINE": "true",
        "UV_OFFLINE": "true",
        "RAYON_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RUST_TEST_THREADS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    calls: list[str] = []
    revision = "a" * 40

    def change_source(phase: str) -> None:
        nonlocal revision
        if phase == drift_phase:
            if drift_path == "HEAD":
                revision = "b" * 40
            else:
                (root / drift_path).write_bytes(b"changed after execution started\n")

    def fake_run(
        argv: list[str], *, env: dict[str, str] | None = None, **_kwargs: object
    ) -> bytes:
        if argv == ["git", "rev-parse", "HEAD"]:
            calls.append("identity")
            return (revision + "\n").encode()
        if argv == ["rustc", "--version"] or argv == ["cargo", "--version"]:
            return b"mock version\n"
        if argv[:2] == ["cargo", "test"]:
            calls.append("producer")
            assert env is not None
            inputs = Path(env["KG_OBS_ORACLE_OUT"])
            (inputs / "states.jsonl").write_bytes(
                b"".join(encoded(row) + b"\n" for row in rows)
            )
            (inputs / "generation.json").write_bytes(
                encoded({"records": 512, "seed_runs": [], "coverage": {}})
            )
            change_source("producer")
            return b""
        assert argv[:2] == ["cargo", "run"]
        calls.append("reference")
        with Path(argv[-1]).open("wb") as features:
            features.truncate(33488896)
        if corrupt_recorder == "reference":
            manifest = Path(argv[argv.index("--manifest-path") + 1])
            example = manifest.parent / "examples/observation_v3_oracle.rs"
            example.write_bytes(b"different recorder executed\n")
        change_source("reference")
        return b""

    def fake_export(directory: Path) -> dict[str, str]:
        path = "engine_rs/src/lib.rs"
        target = directory / path
        target.parent.mkdir(parents=True)
        target.write_bytes(b"pinned reference source\n")
        change_source("export")
        return {path: oracle.hash_file(target)}

    original_copy = oracle.shutil.copyfile

    def copy_recorder(source: Path, destination: Path) -> Path:
        result = original_copy(source, destination)
        if corrupt_recorder == "copy":
            destination.write_bytes(b"wrong recorder copied\n")
        return result

    def validate_before_install(_directory: Path) -> None:
        calls.append("admission")
        change_source("admission")

    monkeypatch.setattr(oracle, "run", fake_run)
    monkeypatch.setattr(oracle, "export_reference", fake_export)
    monkeypatch.setattr(oracle.shutil, "copyfile", copy_recorder)
    monkeypatch.setattr(oracle, "validate_inputs", lambda _path, _generation: None)
    monkeypatch.setattr(oracle, "validate_corpus", validate_before_install)
    monkeypatch.setattr(
        oracle,
        "compress_file",
        lambda _source, destination: destination.write_bytes(b"gz"),
    )
    return root, calls


@pytest.mark.parametrize(
    ("phase", "path"),
    [
        ("producer", "src/kaggriculture/oracle_corpus.rs"),
        ("export", "scripts/kaggriculture_observation_oracle/record.rs"),
        ("reference", "scripts/kaggriculture_observation_oracle/regenerate.py"),
        ("admission", "Cargo.lock"),
        ("admission", "engine_rs/fixtures/episode-95324500.jsonl.gz"),
        ("producer", "engine_rs/src/py_random.rs"),
        ("admission", "engine_rs/Cargo.toml"),
        ("export", "engine_rs/src/econ_attrib.rs"),
        ("reference", "HEAD"),
    ],
)
def test_regeneration_source_identity_rejects_changes_between_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, path: str
) -> None:
    _root, calls = regeneration_harness(
        tmp_path, monkeypatch, drift_phase=phase, drift_path=path
    )
    output = tmp_path / "final"
    with pytest.raises(ValueError, match=r"source.*changed|source.*identity"):
        oracle.regenerate(output, oracle.PIN)
    assert "producer" in calls
    assert not output.exists()


@pytest.mark.parametrize("phase", ["copy", "reference"])
def test_exported_recorder_must_match_preexecution_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    _root, calls = regeneration_harness(tmp_path, monkeypatch, corrupt_recorder=phase)
    output = tmp_path / "final"
    with pytest.raises(ValueError, match=r"recorder.*changed|recorder.*identity"):
        oracle.regenerate(output, oracle.PIN)
    if phase == "copy":
        assert "reference" not in calls
    assert not output.exists()


def test_regeneration_source_identity_is_captured_before_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, calls = regeneration_harness(tmp_path, monkeypatch)
    recorder = root / "scripts/kaggriculture_observation_oracle/record.rs"
    expected_recorder_sha = oracle.hash_file(recorder)
    output = tmp_path / "final"
    oracle.regenerate(output, oracle.PIN)
    identity = json.loads((output / "manifest.json").read_bytes())["source_identity"]
    assert calls.index("identity") < calls.index("producer")
    assert identity["root_commit"] == "a" * 40
    assert identity["recorder_sha256"] == expected_recorder_sha
    assert (
        identity["dirty_files"][str(recorder.relative_to(root))]
        == expected_recorder_sha
    )


def test_regeneration_identity_records_every_engine_build_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _calls = regeneration_harness(tmp_path, monkeypatch)
    output = tmp_path / "final"
    oracle.regenerate(output, oracle.PIN)
    identity = json.loads((output / "manifest.json").read_bytes())["source_identity"]
    recorded = {**identity["dirty_files"]}
    engine_keys = dict(oracle.ENGINE_PATHS)
    for key, value in identity["engine"].items():
        recorded[engine_keys[key]] = value
    for path in (
        "engine_rs/Cargo.toml",
        "engine_rs/Cargo.lock",
        "engine_rs/src/lib.rs",
        "engine_rs/src/econ_attrib.rs",
        "engine_rs/src/py_random.rs",
    ):
        assert recorded[path] == oracle.hash_file(root / path)


@pytest.mark.parametrize("phase", ["", "producer"])
def test_regeneration_rejects_an_undeclared_engine_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root, calls = regeneration_harness(
        tmp_path,
        monkeypatch,
        drift_phase=phase,
        drift_path="engine_rs/src/undeclared.rs",
    )
    if not phase:
        (root / "engine_rs/src/undeclared.rs").write_bytes(b"mod x;\n")
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="undeclared engine source"):
        oracle.regenerate(output, oracle.PIN)
    assert ("producer" in calls) == bool(phase)
    assert not output.exists()


def test_declared_engine_inputs_cover_the_live_engine_crate() -> None:
    oracle.require_declared_engine_sources(ROOT)


def test_missing_corpus_names_the_manifest_and_regeneration_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="corpus missing") as caught:
        oracle.validate_corpus(tmp_path)
    message = str(caught.value)
    assert "manifest.json" in message
    assert "regenerate.py" in message
    assert "R1" not in message
