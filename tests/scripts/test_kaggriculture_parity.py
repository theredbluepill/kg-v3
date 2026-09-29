from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
FIXTURES = REPO_ROOT / "engine_rs/fixtures"
GENERATED = FIXTURES / "generated"
OFFICIAL = FIXTURES / "episode-95324500.jsonl.gz"
FULL_ENGINE_SHA = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"


def _load(name: str) -> ModuleType:
    path = REPO_ROOT / f"scripts/kaggriculture_parity/{name}.py"
    spec = importlib.util.spec_from_file_location(f"kaggriculture_parity_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load("generate_traces")
sweep = _load("sweep")


def _records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _manifest() -> dict[str, Any]:
    return dict(json.loads((GENERATED / "MANIFEST.json").read_text(encoding="utf-8")))


# ------------------------------------------------------------------ hash guard


def test_pin_is_read_from_cargo_metadata() -> None:
    pin = generator.pinned_engine()
    assert pin.version == "1.32.7"
    assert pin.sha256 == FULL_ENGINE_SHA


def test_hash_guard_accepts_only_the_pinned_engine(tmp_path: Path) -> None:
    engine = tmp_path / "kaggriculture.py"
    engine.write_bytes(b"pinned engine bytes\n")
    digest = hashlib.sha256(engine.read_bytes()).hexdigest()
    pin = generator.EnginePin(version="1.32.7", sha256=digest)
    assert generator.verify_engine(engine, "1.32.7", pin) == digest

    engine.write_bytes(b"pinned engine bytes, edited\n")
    with pytest.raises(generator.ParityGeneratorError, match="Refusing to generate"):
        generator.verify_engine(engine, "1.32.7", pin)


def test_hash_guard_rejects_another_package_version(tmp_path: Path) -> None:
    engine = tmp_path / "kaggriculture.py"
    engine.write_bytes(b"x")
    pin = generator.EnginePin("1.32.7", hashlib.sha256(b"x").hexdigest())
    with pytest.raises(generator.ParityGeneratorError, match=r"1\.29\.0 is installed"):
        generator.verify_engine(engine, "1.29.0", pin)


def test_generator_refuses_the_project_kaggle_environments() -> None:
    """The project lock pins 1.29.0 (no kaggriculture env): the guard must refuse."""
    code = generator.main(["--preset", "committed", "--out", os.devnull])
    assert code == 2


def test_cargo_pin_must_be_hex(tmp_path: Path) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        '[package.metadata.kaggriculture]\nkaggle-environments-version = "1.32.7"\n'
        'python-engine-sha256 = "not-a-digest"\n',
        encoding="utf-8",
    )
    with pytest.raises(generator.ParityGeneratorError, match="not hex"):
        generator.pinned_engine(cargo)


# --------------------------------------------------------- format round trip


def test_generated_traces_use_the_official_trace_schema() -> None:
    official = _records(OFFICIAL)
    header, transition = official[0], official[1]
    manifest = _manifest()
    assert manifest["python_engine_sha256"] == FULL_ENGINE_SHA
    kinds = set()
    for entry in manifest["traces"]:
        records = _records(GENERATED / entry["path"])
        generated_header = records[0]
        assert list(generated_header) == list(header)
        assert generated_header["format"] == header["format"]
        assert set(generated_header["configuration"]) == set(header["configuration"])
        assert list(generated_header["initial"]) == list(header["initial"])
        assert list(generated_header["initial"]["public"]) == list(
            header["initial"]["public"]
        )
        source = generated_header["source"]
        assert source["module_version"] == header["source"]["module_version"]
        assert source["engine_sha256"] == header["source"]["engine_sha256"]
        for day in generated_header["rng_schedule"]:
            assert set(day) == set(header["rng_schedule"][0])
        assert generated_header["transitions"] == entry["transitions"]
        for record in records[1:]:
            kinds.add(record["type"])
            if record["type"] == "transition":
                assert list(record) == list(transition)
                assert list(record["expected"]) == list(transition["expected"])
            else:
                assert list(record) == ["type", "from_step", "actions", "python_error"]
    assert kinds == {"transition", "rejected"}


def test_trace_encoding_round_trips_committed_bytes() -> None:
    for entry in _manifest()["traces"]:
        path = GENERATED / entry["path"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert generator.encode_trace(_records(path)) == data, entry["path"]


def test_committed_set_covers_policies_seeds_and_edge_cases() -> None:
    traces = _manifest()["traces"]
    games = [t for t in traces if "expected_divergence" not in t]
    assert 6 <= len(games) <= 8
    policies = {policy for t in games for policy in t["policies"]}
    assert {"random", "edge", "builtin:starter", "builtin:random"} <= policies
    assert any(len(set(t["policies"])) == 2 for t in games), "mixed seats"
    assert len({t["seed"] for t in games}) == len(games)
    assert {t["config_variant"] for t in games} >= {"default", "free-hire", "rich"}
    assert sum(t["rejected"] for t in games) > 0, "Python-rejected inputs"
    free_hire = next(t for t in games if t["config_variant"] == "free-hire")
    records = _records(GENERATED / free_hire["path"])
    actors = max(
        1 + len(farm["hands"])
        for record in records[1:]
        if record["type"] == "transition"
        for farm in record["expected"]["farms"]
    )
    assert actors > 241, "HIRE until and past the 241-actor capacity"


def test_known_divergences_are_recorded_expected_failures() -> None:
    by_name = {Path(t["path"]).name: t for t in _manifest()["traces"]}
    for name, action, kind, field, reason in generator.KNOWN_DIVERGENCES:
        entry = by_name[f"{name}.jsonl.gz"]
        assert json.loads(entry["probe"]) == action
        assert entry["expected_divergence"] == {
            "line": 3,
            "from_step": 2,
            "kind": kind,
            "field": field,
            "reason": reason,
        }
        assert sweep.classify([action, generator.PASS_ACTION]) is not None


def test_edge_policy_excludes_known_divergent_inputs_by_default() -> None:
    obs = _records(GENERATED / "gen-edge-vs-edge.jsonl.gz")[0]["initial"]
    observation = {
        "player": 0,
        **obs["public"],
        "private": obs["privates"][0],
    }
    config = {"boardSize": 10, "maxMarketOrdersPerTurn": 10, "farmHandCostMult": 1}
    for include in (False, True):
        policy = generator.make_edge_policy(include)
        rng = random.Random(3)
        classes = {
            sweep.classify([policy(observation, config, rng).action])
            for _ in range(3000)
        }
        assert classes - {None} == ({"D1", "D2"} if include else set())


def test_sweep_summary_reports_first_divergence(tmp_path: Path) -> None:
    trace = tmp_path / "probe-000-unit.jsonl.gz"
    action = {"farmer": ["PLANT", ["WHEAT"]], "hands": [], "market": []}
    trace.write_bytes(
        generator.encode_trace(
            [
                {"type": "header"},
                {"type": "rejected", "from_step": 0, "actions": [action, {}]},
            ]
        )
    )
    generated = [
        {
            "path": trace.name,
            "seed": 7,
            "policies": ["script", "builtin:pass"],
            "config_variant": "probe",
            "probe": json.dumps(action),
            "transitions": 5,
            "rejected": 1,
        }
    ]
    divergence = {"line": 1, "from_step": 0, "kind": "rust_accepted", "field": "step"}
    report = [{"file": trace.name, "ok": False, "divergence": divergence}]
    summary = sweep.summarize(generated, report, tmp_path)
    assert summary["traces_diverging"] == 1
    assert summary["divergences"][0]["first_divergence"] == divergence
    assert summary["divergences_by_class"] == {"D2": 1}
    assert summary["new_divergences"] == 0
    assert sweep.classify([{"market": [["SELL", "WHEAT", "٣"]]}]) == "D1"
    assert sweep.classify([{"market": [["SELL", "WHEAT", "3"]]}]) is None


# ------------------------------------------------- optional live regeneration


def _isolated_env_available() -> str | None:
    if shutil.which("uv") is None:
        return "uv is not on PATH"
    probe = subprocess.run(
        [*sweep.isolated_python(), "-c", "import kaggle_environments"],
        cwd=REPO_ROOT,
        env={**os.environ, "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return (
            "an isolated kaggle-environments==1.32.7 env is not available offline "
            f"(uv exit {probe.returncode}); live regeneration is skipped"
        )
    return None


def test_live_kaggle_engine_regenerates_committed_traces(tmp_path: Path) -> None:
    """Rerun Kaggle's engine and require byte-identical committed traces."""
    reason = _isolated_env_available()
    if reason is not None:
        pytest.skip(reason)
    subprocess.run(
        [
            *sweep.isolated_python(),
            str(REPO_ROOT / "scripts/kaggriculture_parity/generate_traces.py"),
            "--preset",
            "committed",
            "--out",
            str(tmp_path),
            "--manifest",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "UV_OFFLINE": "1"},
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "MANIFEST.json").read_bytes() == (
        GENERATED / "MANIFEST.json"
    ).read_bytes()
    for entry in _manifest()["traces"]:
        assert (tmp_path / entry["path"]).read_bytes() == (
            GENERATED / entry["path"]
        ).read_bytes(), entry["path"]
