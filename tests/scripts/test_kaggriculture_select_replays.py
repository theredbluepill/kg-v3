from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "kaggriculture_bc" / "select_replays.py"
)
_SPEC = importlib.util.spec_from_file_location("select_replays", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    archives = root / "archives"
    archives.mkdir(parents=True)
    for day in range(21, 28):
        with zipfile.ZipFile(
            archives / f"kaggriculture-episodes-2026-09-{day}.zip", "w"
        ) as archive:
            # Deliberately reverse ZIP order; the historical selector sorts first.
            for index in reversed(range(40)):
                archive.writestr(
                    f"{day}0{index:03}.json",
                    json.dumps({"info": {"EpisodeId": day * 10000 + index}}),
                )
            archive.writestr("README.txt", "not an episode")
    return root


@pytest.mark.parametrize(
    ("day", "first_six", "last"),
    [
        (21, [3, 7, 21, 24, 25, 35], 37),
        (27, [37, 21, 13, 1, 2, 38], 10),
    ],
)
def test_reference_seeded_order(day: int, first_six: list[int], last: int) -> None:
    # Golden output from the reference's sorted/shuffle/slice recipe, not from
    # the port. This covers the validation/train boundary and the final selection.
    selected = selector.select_members(
        [f"{day}0{index:03}.json" for index in reversed(range(40))], day
    )
    assert selected[:6] == [f"{day}0{index:03}.json" for index in first_six]
    assert selected[-1] == f"{day}0{last:03}.json"
    assert len(selected) == 36


def test_export_manifest_preserves_split_and_payloads(
    corpus: Path, tmp_path: Path
) -> None:
    before = {path: path.read_bytes() for path in (corpus / "archives").iterdir()}
    output = tmp_path / "selected.zip"
    manifest = selector.select_replays(corpus, output)
    assert manifest["volume"] == "4llk4uaf20"
    assert manifest["source_root"] == str(corpus)
    assert len(manifest["episodes"]) == 252
    with zipfile.ZipFile(output) as archive:
        assert json.loads(archive.read("selection.json")) == manifest
        assert len(archive.namelist()) == 253
        counts: Counter[tuple[str, str]] = Counter()
        identities: dict[str, set[str]] = {"train": set(), "validation": set()}
        for record in manifest["episodes"]:
            day, member = record["member"].split("/", 1)
            payload = archive.read(record["member"])
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
            assert len(payload) == record["bytes"]
            assert member == record["source_member"]
            assert (
                record["source_archive"] == f"kaggriculture-episodes-2026-09-{day}.zip"
            )
            with zipfile.ZipFile(
                corpus / "archives" / record["source_archive"]
            ) as source:
                assert payload == source.read(member)
            counts[day, record["split"]] += 1
            identities[record["split"]].add(member)
        assert counts == Counter(
            {
                (str(day), split): count
                for day in range(21, 28)
                for split, count in (("train", 32), ("validation", 4))
            }
        )
        assert identities["train"].isdisjoint(identities["validation"])
    assert before == {path: path.read_bytes() for path in before}
    repeated = selector.select_replays(corpus, tmp_path / "repeated.zip")
    assert repeated == manifest


@pytest.mark.parametrize(
    "members",
    [
        ["21/123.json", "22/123.json"],
        ["123.json", "nested/123.json"],
        ["123.json", "123.json"],
    ],
)
def test_duplicate_episode_identity_rejected(members: list[str]) -> None:
    with pytest.raises(ValueError, match=r"Duplicate episode identity: 123\.json"):
        selector.validate_episode_identity(members)


@pytest.mark.parametrize("member", ["../123.json", "/123.json", "123.txt"])
def test_invalid_member_rejected(member: str) -> None:
    with pytest.raises(ValueError, match="Invalid episode member"):
        selector.validate_episode_identity([member])


def test_duplicate_across_dates_fails_before_export(
    corpus: Path, tmp_path: Path
) -> None:
    archive_path = corpus / "archives" / "kaggriculture-episodes-2026-09-22.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index in range(36):
            archive.writestr(f"210{index:03}.json", "{}")
    output = tmp_path / "selected.zip"
    with pytest.raises(ValueError, match="Duplicate episode identity"):
        selector.select_replays(corpus, output)
    assert not output.exists()


def test_insufficient_day_fails_before_export(corpus: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(
        corpus / "archives" / "kaggriculture-episodes-2026-09-27.zip", "w"
    ) as archive:
        archive.writestr("270001.json", "{}")
    output = tmp_path / "selected.zip"
    with pytest.raises(ValueError, match="September 27: expected at least 36 episodes"):
        selector.select_replays(corpus, output)
    assert not output.exists()


def test_duplicate_zip_entries_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate JSON ZIP member"):
        selector.select_members(["123.json", "123.json"], 21)


def test_unsupported_day_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported September source day"):
        selector.select_members([], 20)


def test_output_never_overwritten(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing.zip"
    output.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        selector.select_replays(corpus, output)
    assert output.read_bytes() == b"preserve me"


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_copy_failure_removes_partial_output(
    corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    original_read = zipfile.ZipFile.read
    calls = 0

    def failing_read(
        archive: zipfile.ZipFile, name: str, pwd: bytes | None = None
    ) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("injected copy failure")
        return original_read(archive, name, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", failing_read)
    output = tmp_path / "partial.zip"
    with pytest.raises(error_type, match="injected copy failure"):
        selector.select_replays(corpus, output)
    assert calls == 2
    assert not output.exists()


def test_cli_exports_without_engine(corpus: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli.zip"
    result = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), str(corpus), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"episodes": 252, "output": str(output)}
