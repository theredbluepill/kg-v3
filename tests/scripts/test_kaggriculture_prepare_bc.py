"""Lean Task 5.1 preparation on tiny synthetic Kaggle episodes (CPU only).

The end-to-end episode is played by kaggle-environments' own Kaggriculture
interpreter (the project dependency), so its step+1 pairing is real.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from owl.kaggriculture.bc_data import MANIFEST_NAME, load_bc_dataset

_SCRIPT = Path(__file__).parents[2] / "scripts" / "kaggriculture_prepare_bc.py"
_SPEC = importlib.util.spec_from_file_location("kaggriculture_prepare_bc", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
prepare = importlib.util.module_from_spec(_SPEC)
sys.modules["kaggriculture_prepare_bc"] = prepare
_SPEC.loader.exec_module(prepare)

_SECRET_NAME = "Sentinel Team Name"


@pytest.fixture(scope="module")
def played() -> dict[str, Any]:
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"seed": 12345})
    env.run(["starter", "pass"])
    data: dict[str, Any] = env.toJSON()
    data["info"] |= {"TeamNames": [_SECRET_NAME, "B"], "Agents": [_SECRET_NAME]}
    return data


def _episode(played: dict[str, Any], episode_id: str) -> dict[str, Any]:
    data = copy.deepcopy(played)
    data["info"]["EpisodeId"] = int(episode_id)
    return data


def _ids(split: str, count: int) -> list[str]:
    found = []
    for n in range(10_000, 20_000):
        if prepare.validation_split(str(n), 0.03) == split:
            found.append(str(n))
            if len(found) == count:
                return found
    raise AssertionError("no ids")


def _archive(root: Path, day: int, episodes: dict[str, dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"kaggriculture-episodes-2026-09-{day:02d}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for episode_id, data in episodes.items():
            archive.writestr(f"{episode_id}.json", json.dumps(data))


# --- pure rules -------------------------------------------------------------------


def test_split_is_a_deterministic_episode_hash() -> None:
    ids = [str(n) for n in range(20_000)]
    splits = [prepare.validation_split(i, 0.03) for i in ids]
    assert splits == [prepare.validation_split(i, 0.03) for i in ids]
    fraction = splits.count("validation") / len(ids)
    assert 0.025 < fraction < 0.035


def test_turn_stride_keeps_a_per_episode_phase() -> None:
    assert prepare.kept_turns("7", 1) == list(range(719))
    kept = prepare.kept_turns("7", 5)
    assert kept == prepare.kept_turns("7", 5)
    assert len(kept) in (143, 144) and np.all(np.diff(kept) == 5)
    phases = {prepare.kept_turns(str(n), 5)[0] for n in range(40)}
    assert len(phases) > 1
    with pytest.raises(ValueError):
        prepare.kept_turns("7", 0)


def test_winner_is_the_larger_final_bank_and_draws_keep_both() -> None:
    assert prepare.winner_seats((10.0, 5.0)) == (True, False)
    assert prepare.winner_seats((5.0, 10.0)) == (False, True)
    assert prepare.winner_seats((7.0, 7.0)) == (True, True)


@pytest.mark.parametrize(
    ("value", "kind"),
    [(None, "null"), (False, "false"), (0, "zero"), (0.0, "zero"), ("", "empty_string"), ({}, "empty_object")],
)
@pytest.mark.parametrize("key", ["hands", "market"])
def test_falsy_hands_and_market_normalize_to_empty_lists_counted(
    key: str, value: object, kind: str
) -> None:
    action, kinds = prepare.normalize_action({"farmer": ["PASS"], key: value})
    other = "market" if key == "hands" else "hands"
    assert action == {"farmer": ["PASS"], "hands": [], "market": []}
    assert kinds == sorted([f"normalized_{key}_{kind}", f"normalized_{other}_absent"])


def test_bad_action_envelopes_are_rejected() -> None:
    for raw in (None, [], {"farmer": ["PASS"], "extra": 1}, {"hands": "EAST"}):
        with pytest.raises(ValueError, match="action envelope"):
            prepare.normalize_action(raw)
    action, kinds = prepare.normalize_action({"hands": [], "market": []})
    assert action["farmer"] is None and kinds == []


def test_envelope_checks(played: dict[str, Any]) -> None:
    good = _episode(played, "123")
    prepare.check_envelope(good, "123")
    with pytest.raises(prepare.EpisodeRejected, match="EpisodeId"):
        prepare.check_envelope(good, "124")
    short = copy.deepcopy(good)
    short["steps"] = short["steps"][:-1]
    with pytest.raises(prepare.EpisodeRejected, match="steps"):
        prepare.check_envelope(short, "123")
    boolean = copy.deepcopy(good)
    boolean["info"]["seed"] = True
    with pytest.raises(prepare.EpisodeRejected, match="seed"):
        prepare.check_envelope(boolean, "123")
    errored = copy.deepcopy(good)
    errored["steps"][-1][1]["status"] = "ERROR"
    with pytest.raises(prepare.EpisodeRejected, match="final statuses"):
        prepare.check_envelope(errored, "123")
    partial = copy.deepcopy(good)
    del partial["configuration"]["shedCapacity"]
    with pytest.raises(prepare.EpisodeRejected, match="shedCapacity"):
        prepare.check_envelope(partial, "123")


def test_header_copies_shared_keys_from_seat_zero(played: dict[str, Any]) -> None:
    assert "step" not in played["steps"][5][1]["observation"]
    header = prepare.turn_header(played, 5)
    assert header["seed"] == 0
    assert header["initial"]["public"]["step"] == 5
    assert header["initial"]["privates"][1] == (
        played["steps"][5][1]["observation"]["private"]
    )
    assert header["configuration"] is played["configuration"]


def test_resident_budget_sets_the_turn_stride() -> None:
    per_turn = prepare.resident_bytes_per_turn()
    assert 150_000 < per_turn < 300_000
    rows = 100 * 719
    assert prepare.stride_for_budget(100, rows * per_turn / 2**30) == 1
    assert prepare.stride_for_budget(100, rows * per_turn / 2**30 / 4.5) == 5


# --- Kaggle pairing ---------------------------------------------------------------


def test_step_plus_one_pairing_reproduces_kaggle_and_shifted_does_not(
    played: dict[str, Any],
) -> None:
    matches, mismatches = prepare.pairing_check(played)
    assert (matches, mismatches) == (719, [])
    shifted = copy.deepcopy(played)
    for t in range(719, 0, -1):
        for seat in (0, 1):
            shifted["steps"][t][seat]["action"] = played["steps"][t - 1][seat]["action"]
    matches, mismatches = prepare.pairing_check(shifted)
    assert matches < 719 and mismatches


# --- end to end -------------------------------------------------------------------


def test_end_to_end_writes_loadable_winner_shards_and_resumes(
    played: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare, "_source_identity", lambda repo: {"git_head": "t"})
    train_id, valid_id = _ids("train", 1)[0], _ids("validation", 1)[0]
    bad = _episode(played, "999")
    bad["steps"] = bad["steps"][:700]
    archives = tmp_path / "archives"
    _archive(archives, 21, {train_id: _episode(played, train_id), "999": bad})
    _archive(archives, 22, {valid_id: _episode(played, valid_id)})
    out = tmp_path / "bc"
    argv = [str(archives), str(out), "--days", "21-22", "--workers", "1",
            "--turn-stride", "3", "--pairing-sample", "3"]
    assert prepare.main(argv) == 0

    manifest_text = (out / MANIFEST_NAME).read_text()
    assert _SECRET_NAME not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["pairing"]["matches"] == manifest["pairing"]["total"] == 2 * 719
    assert [r["episode_id"] for r in manifest["rejected_episodes"]] == ["999"]
    assert manifest["episode_rejections"] == {"steps": 1}
    records = {r["episode_id"]: r for r in manifest["episodes"]}
    banks = played["steps"][-1][0]["observation"]["farms"]
    winner = [s for s in (0, 1) if banks[s]["money"] == max(f["money"] for f in banks)]
    for episode_id, split in ((train_id, "train"), (valid_id, "validation")):
        record = records[episode_id]
        assert record["split"] == split and record["policy_seats"] == winner
        assert record["kept_turns"] == len(prepare.kept_turns(episode_id, 3))
        assert record["admitted"] == record["kept_turns"]
    assert manifest["run"]["turn_stride"] == 3

    dataset = load_bc_dataset(out)
    batch = dataset.train.gather(np.arange(dataset.train.num_rows))
    expected = [s in winner for s in (0, 1)]
    assert batch.policy_seat.tolist() == [expected] * dataset.train.num_rows
    assert dataset.train.turn.tolist() == prepare.kept_turns(train_id, 3)
    from owl import rs

    seat = winner[0]
    for row, t in enumerate(dataset.train.turn.tolist()):
        decoded = rs.kaggriculture_decode(
            batch.actions.tokens[row, seat].numpy(),
            int(batch.actions.lengths[row, seat]),
            int(batch.obs.actor_mask[row, seat, :241].sum()),
            int(batch.obs.order_limits[row, seat]),
            241,
        )
        recorded = played["steps"][t + 1][seat]["action"]
        assert json.loads(decoded) == prepare.normalize_action(recorded)[0], t

    # Resume: finished episodes are skipped and their shards stay byte-identical.
    shards = {p: p.read_bytes() for p in out.glob("*/*.npz")}
    (out / MANIFEST_NAME).unlink()
    monkeypatch.setattr(
        prepare, "process_episode", lambda *a: pytest.fail("finished episode redone")
    )
    assert prepare.main(argv[:-2]) == 0
    assert {p: p.read_bytes() for p in out.glob("*/*.npz")} == shards
    with pytest.raises(FileExistsError):
        prepare.main(argv)
