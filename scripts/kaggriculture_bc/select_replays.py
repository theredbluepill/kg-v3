"""Select the audited public replay slice without importing the game engine.

The source root must contain archives/kaggriculture-episodes-2026-09-DD.zip.
Only the selected JSON payloads are copied; source archives are opened read-only.
The embedded selection.json preserves the reference manifest's field names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal, TypedDict

SOURCE_VOLUME = "4llk4uaf20"
DAYS = tuple(range(21, 28))
SEED_BASE = 20260929
TRAIN_PER_DAY = 32
VALIDATION_PER_DAY = 4
SAMPLING = (
    "seeded32 train+4 validation episodes/day,2026-09-21..27; "
    "both seats; no outcome filtering"
)


class SelectedEpisode(TypedDict):
    member: str
    split: Literal["train", "validation"]
    source_archive: str
    source_member: str
    sha256: str
    bytes: int


class SelectionManifest(TypedDict):
    volume: str
    source_root: str
    sampling: str
    episodes: list[SelectedEpisode]


def select_members(names: list[str], day: int) -> list[str]:
    """Apply the reference sort, per-day seed and 36-episode cutoff."""
    if day not in DAYS:
        raise ValueError(f"Unsupported September source day: {day}")
    members = sorted(name for name in names if name.endswith(".json"))
    if len(set(members)) != len(members):
        raise ValueError(f"Duplicate JSON ZIP member on September {day}")
    count = TRAIN_PER_DAY + VALIDATION_PER_DAY
    if len(members) < count:
        raise ValueError(
            f"September {day}: expected at least {count} episodes, got {len(members)}"
        )
    random.Random(SEED_BASE + day).shuffle(members)
    return members[:count]


def validate_episode_identity(members: list[str]) -> None:
    """Reject repeated episode basenames, including across dates and splits.

    Public archives name episodes by ID (for example 111677381.json). This is
    the identity check previously performed by the reference prepare.py.
    """
    seen: set[str] = set()
    for member in members:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ValueError(f"Invalid episode member: {member!r}")
        identity = path.name
        if identity in seen:
            raise ValueError(f"Duplicate episode identity: {identity}")
        seen.add(identity)


def select_replays(source_root: Path, output: Path) -> SelectionManifest:
    """Export all seven days and return their payload-hashed selection manifest.

    Refuse incomplete daily samples and identity leakage before writing. Existing
    outputs are never replaced; a failed copy removes its incomplete output.
    """
    planned: list[tuple[Path, list[str], int]] = []
    all_members: list[str] = []
    for day in DAYS:
        archive = source_root / "archives" / f"kaggriculture-episodes-2026-09-{day}.zip"
        with zipfile.ZipFile(archive) as source:
            members = select_members(source.namelist(), day)
        planned.append((archive, members, day))
        all_members.extend(members)
    validate_episode_identity(all_members)

    manifest: SelectionManifest = {
        "volume": SOURCE_VOLUME,
        "source_root": str(source_root),
        "sampling": SAMPLING,
        "episodes": [],
    }
    # Exclusive creation also prevents accidentally overwriting a source archive.
    with output.open("xb") as output_file:
        try:
            with zipfile.ZipFile(
                output_file,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
            ) as destination:
                for archive, members, day in planned:
                    with zipfile.ZipFile(archive) as source:
                        for index, member in enumerate(members):
                            payload = source.read(member)
                            target = f"{day}/{member}"
                            destination.writestr(target, payload)
                            manifest["episodes"].append(
                                {
                                    "member": target,
                                    "split": (
                                        "validation"
                                        if index < VALIDATION_PER_DAY
                                        else "train"
                                    ),
                                    "source_archive": archive.name,
                                    "source_member": member,
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                    "bytes": len(payload),
                                }
                            )
                destination.writestr("selection.json", json.dumps(manifest, indent=2))
        except BaseException:
            output.unlink()
            raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Read-only source corpus root")
    parser.add_argument("output", type=Path, help="New selected replay ZIP path")
    args = parser.parse_args()
    try:
        manifest = select_replays(args.source_root, args.output)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(
        json.dumps({"episodes": len(manifest["episodes"]), "output": str(args.output)})
    )


if __name__ == "__main__":
    main()
