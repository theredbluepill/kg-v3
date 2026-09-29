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


def test_generator_refuses_before_writing_when_the_pin_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() exits 2 and writes nothing when the installed engine misses the pin."""
    other = generator.EnginePin(version="0.0.0", sha256=FULL_ENGINE_SHA)
    monkeypatch.setattr(generator, "pinned_engine", lambda: other)
    out = tmp_path / "out"
    assert generator.main(["--preset", "committed", "--out", str(out)]) == 2
    assert not out.exists()


def test_project_environment_satisfies_the_engine_pin() -> None:
    """The project lock pins kaggle-environments 1.32.7, whose engine is the pin."""
    _, _, digest = generator.load_pinned_kaggle(generator.pinned_engine())
    assert digest == FULL_ENGINE_SHA


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
        # Minimal repros: the divergent action is the first step of a one-step game.
        assert entry["expected_divergence"] == {
            "line": 1,
            "from_step": 0,
            "kind": kind,
            "field": field,
            "reason": reason,
        }
        assert entry["transitions"] == 1
        records = _records(GENERATED / entry["path"])
        assert records[1]["from_step"] == 0
        assert records[1]["actions"] == [action, generator.PASS_ACTION]
        assert sweep.input_classes([action, generator.PASS_ACTION])


def test_known_divergence_actions_carry_no_unused_fields() -> None:
    for name, action, *_ in generator.KNOWN_DIVERGENCES:
        assert len(action) == 1, name
        assert "PASS" not in json.dumps(action), name


def test_scripted_null_action_is_submitted_not_replaced() -> None:
    """A probe of a null whole action must submit null, not PASS."""
    spec = generator._probe_spec("probe-null", None)
    choices = generator.scripted_choices(spec, len(generator.PROBE_PREAMBLE))
    assert [choice.action for choice in choices] == [None, generator.PASS_ACTION]
    assert all(choice.fallback == generator.PASS_ACTION for choice in choices)
    after = generator.scripted_choices(spec, len(generator.PROBE_PREAMBLE) + 1)
    assert [choice.action for choice in after] == [generator.PASS_ACTION] * 2
    labels = dict(generator.probe_cases())
    assert any(action is None for kind, action in generator.probe_cases()), labels


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
            cls
            for _ in range(3000)
            for cls in sweep.input_classes([policy(observation, config, rng).action])
        }
        assert classes == ({"D1", "D2"} if include else set())


def _summary_case(
    tmp_path: Path,
    record: dict[str, Any],
    divergence: dict[str, Any],
    recheck: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trace = tmp_path / "probe-000-unit.jsonl.gz"
    trace.write_bytes(generator.encode_trace([{"type": "header"}, record]))
    generated = [
        {
            "path": trace.name,
            "seed": 7,
            "policies": ["script", "builtin:pass"],
            "config_variant": "probe",
            "probe": json.dumps(record["actions"][0]),
            "transitions": 5,
            "rejected": 1,
        }
    ]
    report = [{"file": trace.name, "ok": False, "divergence": divergence}]
    return dict(sweep.summarize(generated, report, tmp_path, recheck))


D2_ACTION = {"farmer": ["PLANT", ["WHEAT"]], "hands": [], "market": []}
D2_REJECTED = {
    "type": "rejected",
    "from_step": 0,
    "actions": [D2_ACTION, {}],
    "python_error": "TypeError: unhashable type: 'list'",
}
D1_ACTION = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", "٣"]]}
D1_TRANSITION = {"type": "transition", "from_step": 0, "actions": [D1_ACTION, {}]}


def _divergence(kind: str, field: str, line: int = 1) -> dict[str, Any]:
    return {"line": line, "from_step": 0, "kind": kind, "field": field}


def test_sweep_summary_reports_first_divergence(tmp_path: Path) -> None:
    divergence = _divergence("rust_accepted", "step")
    summary = _summary_case(tmp_path, D2_REJECTED, divergence)
    assert summary["traces_diverging"] == 1
    assert summary["divergences"][0]["first_divergence"] == divergence
    assert summary["divergences_by_class"] == {"D2": 1}
    assert summary["new_divergences"] == 0
    assert sweep.input_classes([{"market": [["SELL", "WHEAT", "٣"]]}]) == ["D1"]
    assert sweep.input_classes([{"market": [["SELL", "WHEAT", "3"]]}]) == []


def test_d2_class_requires_its_observed_signature(tmp_path: Path) -> None:
    """Recognizing a D2 input is not enough: the mismatch must be D2's."""
    cases = [
        # Python accepted the step and the states differ: not D2.
        ({**D2_REJECTED, "type": "transition"}, _divergence("public state", "x")),
        # Python raised something other than an unhashable-type TypeError.
        ({**D2_REJECTED, "python_error": "ValueError: x"}, None),
        # Rust rejected too but then changed its state.
        (D2_REJECTED, _divergence("rust_mutated public state", "public.day")),
    ]
    for index, (record, divergence) in enumerate(cases):
        case = tmp_path / str(index)
        case.mkdir()
        divergence = divergence or _divergence("rust_accepted", "step")
        summary = _summary_case(case, record, divergence)
        assert summary["divergences_by_class"] == {"unclassified": 1}, record
        assert summary["new_divergences"] == 1


def test_d1_class_requires_the_ascii_recheck_to_pass_the_line(tmp_path: Path) -> None:
    """A D1 input on a line whose state was corrupted must stay unclassified."""
    divergence = _divergence("public state", "public.day")
    unconfirmed = [
        None,
        [{"file": "probe-000-unit.jsonl.gz", "ok": False, "divergence": divergence}],
    ]
    for index, recheck in enumerate(unconfirmed):
        case = tmp_path / f"unconfirmed-{index}"
        case.mkdir()
        summary = _summary_case(case, D1_TRANSITION, divergence, recheck)
        assert summary["divergences_by_class"] == {"unclassified": 1}
        assert summary["new_divergences"] == 1
    later = _divergence("public state", "public.day", line=2)
    for index, recheck in enumerate(
        [
            [{"file": "probe-000-unit.jsonl.gz", "ok": True}],
            [{"file": "probe-000-unit.jsonl.gz", "ok": False, "divergence": later}],
        ]
    ):
        case = tmp_path / f"confirmed-{index}"
        case.mkdir()
        summary = _summary_case(case, D1_TRANSITION, divergence, recheck)
        assert summary["divergences_by_class"] == {"D1": 1}
        assert summary["new_divergences"] == 0


def test_d1_recheck_rewrites_only_the_divergent_line(tmp_path: Path) -> None:
    records = [
        {"type": "header"},
        {**D1_TRANSITION, "from_step": 0},
        {**D1_TRANSITION, "from_step": 1},
    ]
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t.jsonl.gz").write_bytes(generator.encode_trace(records))
    report = [
        {
            "file": "t.jsonl.gz",
            "ok": False,
            "divergence": _divergence("rust_error", "step"),
        }
    ]
    recheck = tmp_path / "recheck"
    assert sweep.write_d1_rechecks(report, traces, recheck) == ["t.jsonl.gz"]
    rewritten = _records(recheck / "t.jsonl.gz")
    assert rewritten[1]["actions"][0]["market"] == [["SELL", "WHEAT", "3"]]
    assert rewritten[2] == records[2]
    assert rewritten[0] == records[0]


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
    if sys.platform == "darwin":
        pytest.skip(
            "Mac bound: committed regeneration exceeds two live games; run on pod"
        )
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


# ---------------------------------------------------- original opponent oracles


def test_sibling_oracle_rejects_changed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator, "_sibling_blob", lambda _path: b"changed source")
    with pytest.raises(generator.ParityGeneratorError, match="SHA-256"):
        generator.sibling_oracle_files("r04")


def test_sibling_oracle_rejects_unknown_bot() -> None:
    with pytest.raises(generator.ParityGeneratorError, match="unknown sibling"):
        generator.sibling_oracle_files("other")


def test_sibling_oracle_instances_have_fresh_module_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        b"count = 0\ndef agent(obs, config=None):\n global count\n"
        b" count += 1\n return {'count': count}\n"
    )
    monkeypatch.setattr(
        generator, "sibling_oracle_files", lambda _bot: {"agents/r04/main.py": source}
    )
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    first = generator.SiblingPolicy("r04")
    second = generator.SiblingPolicy("r04")
    rng = random.Random(0)
    try:
        assert first({}, {}, rng).action == {"count": 1}
        assert first({}, {}, rng).action == {"count": 2}
        assert second({}, {}, rng).action == {"count": 1}
        assert first({}, {}, rng).action == {"count": 3}
        assert sys.path == original_path
        assert not any(
            key.startswith("_kagg_oracle_")
            for key in set(sys.modules) - original_modules
        )
    finally:
        first.close()
        second.close()
    assert not first.root.exists()
    assert not second.root.exists()


def test_opponent_specs_cover_both_seats_and_enforce_live_batch_bound() -> None:
    specs = generator.opponent_specs(0, 2, 20260929) + generator.opponent_specs(
        2, 2, 20260929
    )
    assert len(specs) == 4
    for seat in range(2):
        assert {spec.policies[seat] for spec in specs} == {
            "builtin:starter",
            "sibling:r04",
            "sibling:ecobot",
            "sibling:e776",
        }
    assert all(spec.variant == "default" for spec in specs)
    with pytest.raises(generator.ParityGeneratorError, match="one or two"):
        generator.opponent_specs(0, 3, 20260929)


def test_opponent_coverage_counts_actual_events() -> None:
    public: dict[str, Any] = {
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [
            {"hands": [], "tiles": [[{"kind": "WEED"}]]},
            {"hands": [], "tiles": [[None]]},
        ],
        "market": {"inventory": {"WHEAT": 0}},
    }
    records = [
        {"initial": {"public": public}},
        {
            "type": "transition",
            "from_step": 0,
            "actions": [{"market": [["HIRE"], ["BUY_PRODUCT", "WHEAT", 1]]}, {}],
            "expected": {
                **public,
                "step": 1,
                "farms": [
                    {**public["farms"][0], "hands": [[5, 4]]},
                    public["farms"][1],
                ],
            },
        },
    ]
    counts = generator.opponent_coverage(records)
    assert counts[0]["openings"] == counts[1]["openings"] == 1
    assert counts[0]["hires"] == 1
    assert counts[1]["hires"] == 0
    assert counts[0]["weed_presence"] == 1
    assert counts[0]["buy_quantity_above_inventory_index"] == 1
    assert counts[0]["mid_episode_replay"] == 0


def test_e776_package_modules_are_independent_per_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "agents/e776/main.py": (
            b"from e776_pkg.entry import policy\n"
            b"def kaggriculture_e776_agent(obs, config=None): return policy(obs)\n"
        ),
        "agents/e776/e776_pkg/__init__.py": b"",
        "agents/e776/e776_pkg/entry.py": (
            b"count = 0\ndef policy(obs):\n global count\n count += 1\n"
            b" return {'count': count}\n"
        ),
    }
    monkeypatch.setattr(generator, "sibling_oracle_files", lambda _bot: files)
    first, second = generator.SiblingPolicy("e776"), generator.SiblingPolicy("e776")
    try:
        rng = random.Random(0)
        assert first({}, {}, rng).action == {"count": 1}
        assert first({}, {}, rng).action == {"count": 2}
        assert second({}, {}, rng).action == {"count": 1}
        assert "e776_pkg.entry" not in sys.modules
    finally:
        first.close()
        second.close()


def test_opponent_deadline_cannot_be_swallowed_by_original_agent() -> None:
    def simulate_original_agent() -> None:
        with generator.opponent_deadline():
            # R04 catches Exception around its entire original agent. The deadline
            # must escape that handler and become an explicit generator failure.
            try:
                raise generator._OpponentDeadline
            except Exception:
                pytest.fail("original agent swallowed the deadline")

    with pytest.raises(generator.ParityGeneratorError, match="Mac bound"):
        simulate_original_agent()


def test_opponent_custody_failure_preserves_existing_manifest(tmp_path: Path) -> None:
    manifest = {
        "traces": [{"path": "missing.jsonl.gz", "sha256": "0" * 64, "bytes": 0}]
    }
    path = tmp_path / "MANIFEST.json"
    frozen = json.dumps(manifest)
    path.write_text(frozen)
    with pytest.raises(FileNotFoundError):
        generator.write_opponent_manifest(tmp_path, [], generator.pinned_engine())
    assert path.read_text() == frozen


def test_opponent_oracle_requires_the_competition_python_runtime() -> None:
    """R04 sums floats; CPython 3.12's compensated sum() changes its decisions."""
    assert generator.oracle_python_runtime((3, 11, 15)) == "3.11.15"
    for version in [(3, 12, 13), (3, 10, 14), (3, 13, 0)]:
        with pytest.raises(generator.ParityGeneratorError, match=r"CPython 3\.11"):
            generator.oracle_python_runtime(version)


def test_opponent_preset_refuses_other_runtimes_before_playing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(generator.sys, "version_info", (3, 12, 13, "final", 0))
    out = tmp_path / "oracle"
    status = generator.main(["--preset", "opponents", "--out", str(out)])
    assert status == 2
    assert "CPython 3.11" in capsys.readouterr().err
    assert not out.exists()


# ------------------------------------------- original mid-episode replay oracle


@pytest.fixture(scope="module")
def starter_trace() -> tuple[list[dict[str, Any]], ModuleType, ModuleType]:
    """One live default game (the Mac bound allows two) for replay unit checks."""
    pin = generator.pinned_engine()
    kaggle, module, digest = generator.load_pinned_kaggle(pin)
    spec = generator.GameSpec(
        # Not "oracle-": that prefix enables the per-process 1 GB generation bound.
        name="replay-unit",
        seed=20261001,
        policies=("builtin:starter", "builtin:starter"),
        variant="default",
        policy_seed=20261001,
    )
    result = generator.play(spec, kaggle, module, pin, digest)
    return result.records, kaggle, module


def test_replay_resumes_fresh_controllers_after_the_recorded_prefix(
    starter_trace: tuple[list[dict[str, Any]], ModuleType, ModuleType],
) -> None:
    records, kaggle, module = starter_trace
    case = generator.replay_case(records, 30, 5, kaggle, module)
    assert case["source"] == "replay-unit.jsonl.gz"
    assert case["policies"] == ["builtin:starter", "builtin:starter"]
    assert (case["reconstruct_step"], case["resume_steps"]) == (30, 5)
    assert case["resumed_actions"] == [
        records[1 + step]["actions"] for step in range(30, 35)
    ]
    assert case["continuous_equal_actions"] == [5, 5]
    assert case["final"]["public"] == records[35]["expected"]
    assert case["final"]["privates"] == records[35]["privates"]
    assert case["final"]["statuses"] == records[35]["statuses"]


def test_replay_refuses_a_prefix_the_fresh_controller_did_not_choose(
    starter_trace: tuple[list[dict[str, Any]], ModuleType, ModuleType],
) -> None:
    records, kaggle, module = starter_trace
    altered = json.loads(json.dumps(records))
    altered[1 + 7]["actions"][1] = {"farmer": ["TASK_7_1_MUTATION"]}
    with pytest.raises(generator.ParityGeneratorError, match="prefix step 7 seat 1"):
        generator.replay_case(altered, 30, 5, kaggle, module)


def test_replay_refuses_a_prefix_the_engine_does_not_reproduce(
    starter_trace: tuple[list[dict[str, Any]], ModuleType, ModuleType],
) -> None:
    records, kaggle, module = starter_trace
    altered = json.loads(json.dumps(records))
    altered[1 + 11]["expected"]["hour"] += 1
    with pytest.raises(generator.ParityGeneratorError, match="reconstruction step 11"):
        generator.replay_case(altered, 30, 5, kaggle, module)


def test_replay_window_must_resume_inside_the_recorded_episode() -> None:
    generator.replay_window(695, 24, 719)
    for point, resume in [(0, 24), (37, 0), (700, 24)]:
        with pytest.raises(generator.ParityGeneratorError, match="replay window"):
            generator.replay_window(point, resume, 719)


def test_replay_points_cover_mid_day_day_reset_and_final_day() -> None:
    hours = [point % 24 for point in generator.REPLAY_POINTS]
    assert hours[0] not in (0, 23)
    assert 0 in hours
    last = generator.REPLAY_POINTS[-1]
    assert last + generator.REPLAY_RESUME_STEPS == 719
    assert (last + 1) // 24 == 29


def test_replay_fixture_refuses_changed_oracle_custody(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "a.jsonl.gz").write_bytes(b"changed")
    manifest = {"traces": [{"path": "a.jsonl.gz", "sha256": "0" * 64, "bytes": 7}]}
    (oracle / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(generator.ParityGeneratorError, match="oracle custody"):
        generator.oracle_sources(oracle)


def test_replay_fixture_refuses_to_overwrite_a_different_frozen_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / generator.REPLAY_FILE
    path.write_bytes(b"frozen")
    with pytest.raises(generator.ParityGeneratorError, match="frozen replay"):
        generator.write_replay_fixture(tmp_path, {"cases": []})
    assert path.read_bytes() == b"frozen"


def test_replay_preset_refuses_other_runtimes_before_playing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(generator.sys, "version_info", (3, 12, 13, "final", 0))
    out = tmp_path / "replay"
    status = generator.main(["--preset", "opponent-replay", "--out", str(out)])
    assert status == 2
    assert "CPython 3.11" in capsys.readouterr().err
    assert not out.exists()
