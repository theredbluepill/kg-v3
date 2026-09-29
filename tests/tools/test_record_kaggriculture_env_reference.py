from __future__ import annotations

import copy
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "env_reference", ROOT / "scripts/record_kaggriculture_env_reference.py"
)
assert SPEC is not None
assert SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def example() -> tuple[dict, dict[str, np.ndarray]]:
    arrays = recorder.empty_arrays()
    arrays["lengths"].fill(2)
    arrays["program_offsets"] = np.arange(16 * 719 * 2 + 1, dtype=np.int64) * 2
    arrays["tokens"] = np.zeros((16 * 719 * 2 * 2, 12), dtype=np.int64)
    arrays["tokens"][::2, 1] = 1
    arrays["tokens"][1::2, 11] = 1
    arrays["dones"][:, -1, :] = True
    arrays["econ_after"][:, :, 0, :3] = 1
    arrays["econ_after"][:, :, 1, 8:10] = 2
    arrays["econ_before"][:, 1:] = arrays["econ_after"][:, :-1]
    arrays["terminal_econ"][:] = arrays["econ_after"][:, -1]
    arrays["terminal_steps"].fill(719)
    arrays["terminal_winner"].fill(-1)
    coverage = [
        dict(
            starvation=1,
            drought=1,
            ineffective=1,
            hires=1,
            animal_placements=1,
            sell_units=2,
            sell_cash=2,
        )
        for _ in range(16)
    ]
    manifest = recorder.make_manifest(arrays, coverage, recorder.source_identity())
    return manifest, arrays


def publish_example(tmp_path: Path) -> Path:
    manifest, arrays = example()
    target = tmp_path / "oracle.npz"
    recorder.publish_fixture(target, manifest, arrays)
    return target


def test_deterministic_packed_fixture_round_trip(tmp_path: Path) -> None:
    first = publish_example(tmp_path)
    original = first.read_bytes(), first.with_suffix(".json").read_bytes()
    manifest, arrays = recorder.load_fixture(first)
    recorder.publish_fixture(first, manifest, arrays)
    assert original == (first.read_bytes(), first.with_suffix(".json").read_bytes())
    assert arrays["tokens"].shape == (16 * 719 * 2 * 2, 12)
    assert manifest["compressed_bytes"] <= 8 * 1024**2
    assert manifest["expanded_bytes"] <= 256 * 1024**2
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(row.date_time == (1980, 1, 1, 0, 0, 0) for row in archive.infolist())


@pytest.mark.parametrize(
    "field",
    [
        "starvation",
        "drought",
        "ineffective",
        "hires",
        "animal_placements",
        "sell_units",
        "sell_cash",
    ],
)
def test_every_game_requires_all_coverage(field: str) -> None:
    manifest, arrays = example()
    manifest["coverage"][7][field] = 0
    with pytest.raises(ValueError, match="coverage"):
        recorder.validate_arrays(manifest, arrays)


@pytest.mark.parametrize(
    "field",
    [
        "reference",
        "policy_sha256",
        "recorder_sha256",
        "rust_recorder_sha256",
        "archive_sha256",
        "grammar_fixture_sha256",
        "grammar_manifest_sha256",
    ],
)
def test_source_drift_rejected(field: str, tmp_path: Path) -> None:
    path = publish_example(tmp_path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    manifest["sources"][field] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source"):
        recorder.load_fixture(path)


@pytest.mark.parametrize(
    "field",
    [
        "lengths",
        "program_offsets",
        "transition_indices",
        "terminal_steps",
        "autoreset_seed",
        "next_seed",
        "terminal_econ",
        "terminal_banks",
        "dones",
    ],
)
def test_transition_and_seed_inventory_is_complete(field: str) -> None:
    manifest, arrays = example()
    arrays[field].flat[-1] += 1 if arrays[field].dtype != np.bool_ else False
    if field == "dones":
        arrays[field][0, 0, 0] = True
    with pytest.raises(ValueError, match="custody"):
        recorder.validate_arrays(manifest, arrays)


def test_expanded_and_compressed_caps_are_enforced(tmp_path: Path) -> None:
    path = publish_example(tmp_path)
    for field, size in [
        ("compressed_bytes", 8 * 1024**2 + 1),
        ("expanded_bytes", 256 * 1024**2 + 1),
    ]:
        manifest = json.loads(path.with_suffix(".json").read_text())
        manifest[field] = size
        path.with_suffix(".json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=r"size|budget"):
            recorder.load_fixture(path)
        path = publish_example(tmp_path)


def test_fixture_hash_is_verified_before_numpy_load(
    tmp_path: Path, monkeypatch
) -> None:
    path = publish_example(tmp_path)
    path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(
        np, "load", lambda *_a, **_k: pytest.fail("numpy loaded before custody")
    )
    with pytest.raises(ValueError, match=r"hash|size"):
        recorder.load_fixture(path)


def test_duplicate_or_object_zip_members_rejected(tmp_path: Path) -> None:
    path = publish_example(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        content = io.BytesIO()
        np.save(content, np.array([object()], dtype=object))
        archive.writestr("unexpected.npy", content.getvalue())
    manifest = json.loads(path.with_suffix(".json").read_text())
    manifest["compressed_bytes"] = path.stat().st_size
    manifest["fixture_sha256"] = recorder.sha(path.read_bytes())
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        recorder.load_fixture(path)


def test_failed_validation_publishes_no_partial_fixture(tmp_path: Path) -> None:
    manifest, arrays = example()
    manifest["coverage"][0]["hires"] = 0
    target = tmp_path / "oracle.npz"
    with pytest.raises(ValueError, match="coverage"):
        recorder.publish_fixture(target, manifest, arrays)
    assert not target.exists()
    assert not target.with_suffix(".json").exists()


def test_watchdog_charges_supervisor_and_children_and_kills_group(
    tmp_path: Path, monkeypatch
) -> None:
    class Process:
        pid = 12345
        stopped = False

        def poll(self):
            return -9 if self.stopped else None

        def wait(self):
            self.stopped = True
            return -9

    process = Process()
    killed = []
    monkeypatch.setattr(recorder.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(recorder, "OPS", tmp_path)
    monkeypatch.setattr(
        recorder, "group_rss", lambda group: 700_000 if group == 12345 else 0
    )
    monkeypatch.setattr(recorder, "resident_bytes", lambda _pid: 400_000)
    monkeypatch.setattr(
        recorder.os, "killpg", lambda group, sig: killed.append((group, sig))
    )
    with pytest.raises(SystemExit, match="reason=memory"):
        recorder.supervise(["fake-worker"], 115, 1)
    assert killed == [(12345, recorder.signal.SIGKILL)]
    receipt = json.loads((tmp_path / "reference-recording-attempt.json").read_text())
    assert receipt["sampled_process_group_peak_rss_bytes"] == 1_100_000
    assert receipt["exit_status"] == -9


def test_worker_failure_never_publishes_target(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "oracle.npz"
    monkeypatch.setattr(recorder.sys, "argv", ["recorder", "--output", str(target)])

    def fail(*_args):
        raise SystemExit("mock worker failure")

    monkeypatch.setattr(recorder, "supervise", fail)
    with pytest.raises(SystemExit, match="mock worker failure"):
        recorder.main()
    assert not target.exists()
    assert not target.with_suffix(".json").exists()


def test_reference_source_member_drift_rejected(tmp_path: Path) -> None:
    path = publish_example(tmp_path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    manifest["sources"]["reference_sha256"]["engine_rs/src/training.rs"] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source custody"):
        recorder.load_fixture(path)


@pytest.mark.parametrize("corrupt", [False, True])
def test_exported_source_drift_rechecked_before_publication(
    tmp_path: Path, monkeypatch, corrupt: bool
) -> None:
    relative = "engine_rs/src/training.rs"
    source = b"unchanged pinned TrainingBatch source"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(source + b"unexpected edit" if corrupt else source)
    monkeypatch.setattr(
        recorder, "reference_sources", lambda: {relative: recorder.sha(source)}
    )
    if corrupt:
        with pytest.raises(ValueError, match="export source drift"):
            recorder.verify_export(tmp_path)
    else:
        recorder.verify_export(tmp_path)


def test_cannot_reduce_sixteen_game_recipe(monkeypatch) -> None:
    monkeypatch.setattr(recorder.sys, "argv", ["recorder", "--games", "2"])
    with pytest.raises(ValueError, match="exact pinned 16-game"):
        recorder.main()


def test_policy_is_observation_local_and_exact() -> None:
    policy = recorder.policy
    public = {"step": 0, "farms": [{"hands": []}, {"hands": [[0, 0]]}]}
    before = copy.deepcopy(public)
    action = policy.action(public, 0)
    assert action == {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["BUY_SEED", "WHEAT", 2],
            ["BUY_ANIMAL", "GOOSE", 1],
            ["BUY_PRODUCT", "WHEAT", 3],
            ["HIRE"],
            ["BUY_LAND"],
        ],
    }
    assert policy.action(public, 1)["hands"] == [["PASS"]]
    for step, farmer in enumerate(
        [
            ["PICKUP", "GOOSE"],
            ["BUILD_COOP"],
            ["PLACE", "GOOSE"],
            ["NORTH"],
            ["PLANT", "WHEAT"],
            ["WATER"],
        ],
        1,
    ):
        public["step"] = step
        assert policy.action(public, 0)["farmer"] == farmer
    public["step"] = 8
    assert policy.action(public, 0)["farmer"] == ["HARVEST"]
    public["step"] = 10
    assert policy.action(public, 0)["market"] == [[], ["BUY_PRODUCT", "WHEAT", 0]]
    public["step"] = 2
    assert policy.action(public, 1)["market"] == [["SELL", "WHEAT", 2]]
    public["step"] = 0
    assert public == before
    assert policy.reward_config(0)["econ_shaping"] == 0.02
    assert policy.reward_config(8)["econ_shaping"] == 0.2
    assert policy.reward_config(8)["econ_ineffective_weight"] == 0


def test_frozen_fixture_custody_and_complete_coverage() -> None:
    manifest, arrays = recorder.load_fixture()
    recorder.validate_arrays(manifest, arrays)
    assert arrays["lengths"].shape == (16, 719, 2)
