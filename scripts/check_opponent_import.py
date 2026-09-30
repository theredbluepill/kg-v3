"""Verify byte-exact opponent imports, authored inventory and Python oracle custody.

Imports come from this repository's own pinned history. Cha22's closure adds a
second, narrow category: files whose only edits are the v3 Game-view accessor
lines, re-derived here from the pinned blob. Cha22's upstream notices are
pinned by hash to kaggriculture-v2 and to the original submission.

The manifest records actual comparison counts separately from the available actions
in generated traces. A successful custody check does not establish action parity.
Original Python submissions are never copied into this repository. The default
check pins their recorded entry hashes structurally, so it runs anywhere (pods,
containers). ``--original-sources`` additionally re-reads every original file from
its pinned Git object store in the owner's sibling repository and the Starter
source from the installed pinned package, without importing or executing Kaggle
or any controller.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

PIN = "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0"
PYTHON_REPO = "/Users/poonszesen/kaggriculture"
PYTHON_PIN = "e8884aae82eddeb7a1aeae99ecceeca7c830d67e"
KAGGLE_VERSION = "1.32.7"
PYTHON_ENGINE_SHA256 = (
    "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
)
STARTER_PATH = "envs/kaggriculture/kaggriculture.py"
E776_MANIFEST_SHA256 = (
    "55dcc45b4c56324599d7832dfbc53ce6f1ce86aa44524fcb5a59a3c694857d58"
)
MANIFEST = "opponents_rs/OPPONENT_MANIFEST.json"
ORACLE_DIR = "opponents_rs/fixtures/oracle"
ORACLE_MANIFEST = f"{ORACLE_DIR}/MANIFEST.json"
CHA22_ORACLE_DIR = "opponents_rs/fixtures/oracle-cha22"
ORACLE_DIRS = (ORACLE_DIR, CHA22_ORACLE_DIR)
TRACE_FORMAT = "kaggriculture-re-parity-v1"
GENERATOR = "scripts/kaggriculture_parity/generate_traces.py"
TRACE_BUDGET = 4_000_000
MAX_TRACE_EXPANSION = 128_000_000
BOTS = ("starter", "r04", "ecobot", "e776", "cha22")
POLICIES = (
    "builtin:starter",
    "sibling:r04",
    "sibling:ecobot",
    "sibling:e776",
    "upstream:cha22",
)
# Cha22's original is the public notebook abhinav0370/cha22-agent (ID 135642255)
# output main.py, retrieved 2026-09-24. No local Git repository holds it;
# ``--original-sources`` reads a caller-supplied copy (the generator's variable).
CHA22_SOURCE_REPO = "kaggle-notebook:abhinav0370/cha22-agent"
CHA22_SOURCE_VERSION = "135642255"
CHA22_SOURCE_PATH = "main.py"
CHA22_SOURCE_SHA256 = "127ed3e62988c0474d386db6527ae8ca9de9bb1fe7004128557ddef67126c652"
CHA22_SOURCE_ENV = "KAGG_CHA22_SOURCE"
V2_REPO = "/Users/poonszesen/kaggriculture-v2"
V2_PIN = "30a3ac47f8a19725c2a0da762f9ffdf87c9cd389"
NOTICE_DIR = "opponents_rs/notices/cha22"
# Every comment line of the original submission, in order: its Apache-2.0
# license text and the attribution notices its layers carry.
COMMENT_LINES = "main.py#comment-lines"
NOTICES = {
    f"{NOTICE_DIR}/NOTICE.md": (
        V2_REPO,
        V2_PIN,
        "opponents/cha22/NOTICE.md",
        "9c52f8bdfdcc4786e0f1786d446a760ec8ed9fd52123025ed4f8d723b8242074",
    ),
    f"{NOTICE_DIR}/UPSTREAM-NOTEBOOK.md": (
        V2_REPO,
        V2_PIN,
        "opponents/cha22/UPSTREAM-NOTEBOOK.md",
        "6e3f4bf38ad43e9ffd8ca3a5ef32acb7078fe5fa6628db02cfca34dc7db7a9f4",
    ),
    f"{NOTICE_DIR}/UPSTREAM_README.md": (
        V2_REPO,
        V2_PIN,
        "opponents/cha22/UPSTREAM_README.md",
        "b7bba756b2533ddd5429bb9fa00e6f78d7ab8b0851f81d13ba96a390c618cece",
    ),
    f"{NOTICE_DIR}/UPSTREAM-SOURCE-COMMENTS.txt": (
        CHA22_SOURCE_REPO,
        CHA22_SOURCE_VERSION,
        COMMENT_LINES,
        "06c1701e76cc897e8af93a7ee065d7eafaf61ed94e341266294112c8f435d31d",
    ),
}
NOTICE_KEYS = ("path", "source_repo", "source_commit", "source_path", "sha256")
# The v2 controllers read ``game.privates``/``game.config`` fields; the v3 view
# exposes accessors. Each substitution must occur exactly once per adapted file.
VIEW_ADAPTATION = "v3-game-view-accessors"
VIEW_EDITS = (
    (
        b"serde_json::to_value(game.snapshot().public)",
        b"serde_json::to_value(&game.snapshot().public)",
    ),
    (
        b"serde_json::to_value(&game.privates[",
        b"serde_json::to_value(&game.privates()[",
    ),
    (
        b"serde_json::to_value(&game.config)",
        b"serde_json::to_value(game.configuration())",
    ),
)
ADAPTED_KEYS = ("path", "reference_path", "reference_sha256", "sha256", "adaptation")
COVERAGE_KEYS = (
    "openings",
    "day_resets",
    "weed_presence",
    "buy_quantity_above_inventory_index",
    "rejected_steps",
    "hires",
    "final_day_sell_orders",
    "mid_episode_replay",
)
IMPORT_HASHES = {
    "engine_rs/src/native_agents/starter.rs": (
        "01b4de943e7ea9df425b97355f8451ce48547b3666cc1c3aa3faa5649eda22e0"
    ),
    "engine_rs/src/native_agents/r04.rs": (
        "a80f130c636fc9def610027b6a5593c67db3a4ddb5d63ac972d80c4d2ba31431"
    ),
    "engine_rs/src/native_agents/ecobot.rs": (
        "8794e7cf57ef6663c1578464d8e5a1d2749736bc21ba52d45a2301a58e338ce8"
    ),
    "engine_rs/src/native_agents/e776.rs": (
        "02166970e2fc8345178d29ea863fe8532c9a4fa7d227651d0ebc57b0df82b9fe"
    ),
    "engine_rs/fixtures/e776-kenjo-trace.json": (
        "da0d5d1bd326cb5bf068c2065ba1fe8f7e644107db806d7f9a1eae4dafd89692"
    ),
    # Cha22 (full ig_agent) and its dependency closure, byte-exact.
    "engine_rs/src/native_agents/cha22/early.rs": (
        "2c9c4658d8d45c0fae65172cbc29caed7eaf1729bf608b72bae16f45b542779e"
    ),
    "engine_rs/src/native_agents/cha22/execution.rs": (
        "b6933502cb655fca7de2e1c057269d8e2b19014f71b49a5b3c5eb17cf62ac53a"
    ),
    "engine_rs/src/native_agents/cha22/market.rs": (
        "a60fc74266fb5a3392c08306805d2fae586a18934052b4844898c5b4d168b97a"
    ),
    "engine_rs/src/native_agents/cha22/tail.rs": (
        "837dc61172a299d5a14758511bf7ae4150d0cb529f5074ebdaffe1503a91db43"
    ),
    "engine_rs/src/native_agents/farm2945/capharv.rs": (
        "327765645193594df0bed214e71508a9917367e5504ffecbbe389eb78be66761"
    ),
    "engine_rs/src/native_agents/farm2945/carrot2.rs": (
        "ec6dd1b6db37af53d0803aaa04a56afb57e393aa227630d690f405fbff84a768"
    ),
    "engine_rs/src/native_agents/farm2945/early.rs": (
        "e7b48136a19dc060565a4b7573d308be4eb58a62a053d9efd7bfad7afe5bd2ff"
    ),
    "engine_rs/src/native_agents/farm2945/herd2.rs": (
        "253dd24d1b0513791feada2fb54e744b085c9ca26eb2921d3a2e955f1d05a425"
    ),
    "engine_rs/src/native_agents/farm2945/market.rs": (
        "6de2753da7f59ed4f0d59eb7760bc6e25acf069be94abc43672c8a4c02fe6f32"
    ),
    "engine_rs/src/native_agents/farm2945/orderpri2.rs": (
        "f05ecb3260475076f022006893b0a7e86e05b6110917fa4b1a08cd6003ec8c30"
    ),
    "engine_rs/src/native_agents/farm2945/race.rs": (
        "c104e52c102668763937b359ff9a0481b119f00f017e0e4250d0af3b6f8068e6"
    ),
    "engine_rs/src/native_agents/farm2945/sheep.rs": (
        "b5a9129d3d4823ecd7a454037935a959098d85d2fcd429ad8c0d5f9088bc53cd"
    ),
    "engine_rs/src/native_agents/farm2945/util.rs": (
        "adeaa37bb9ae7b551cc508bd5952fcd04a05e0f0f46cc7e14c152caffae356f5"
    ),
    "engine_rs/src/native_agents/metav4/library_lengths.rs": (
        "bfb174fa470cc2e78dd382f5883fdca5a50b2dc02e07b41d11f57fd6450ca75f"
    ),
    "engine_rs/src/native_agents/v43/common.rs": (
        "9928b24f592746a4e33905b86039cd7313195b7a4eb7eeb11a84ea0db5ec458d"
    ),
    "engine_rs/src/native_agents/v43/contracts.rs": (
        "ca13f6e147d96fa847c58e5db9cc8beb9ceb38dc8b864513be6e137ce7c75f2a"
    ),
    "engine_rs/src/native_agents/v43/core.rs": (
        "fa8c916b80f9526c5b2cedb976b7f6d59dedaaa29e13f41e28de3d4b1af68bdf"
    ),
    "engine_rs/src/native_agents/v43/late.rs": (
        "1066a2b5dbec33ba4f84cf916f2c6e48a54bef0b10af66042d80706c3fd018e4"
    ),
    "engine_rs/src/native_agents/v43/production.rs": (
        "a7c43b240dd58dc7b4c4284a9da319b31594406cdbdaa4102a1f4773a5ec862d"
    ),
    "engine_rs/src/native_agents/v43/terminal.rs": (
        "cef1571aef93de57c382b94dbceace6e6431442fd2350e608b89448be36e0088"
    ),
    "engine_rs/src/native_agents/v47/advance.rs": (
        "4759ff16b9cb7652e314d46152cb40d87e1e0fd62dd3a5c1e1c972ffe273b614"
    ),
    "engine_rs/src/native_agents/v47/herd.rs": (
        "c2892be9caedbf97b021b61ebdb634581e831f773c8d8b6c19a1a85580260e8f"
    ),
    "engine_rs/src/native_agents/v47/lockstep.rs": (
        "8d5759ddb51baf97649c7be6ad2b8d4b257dc70aabffbb3621eb6703b4bf46eb"
    ),
    "engine_rs/src/native_agents/v47/opening.rs": (
        "b8b53987f1f0597ae5b165b6af7139c5c3067ac02b8736e844771118a466ad35"
    ),
    "engine_rs/src/native_agents/v47/preguard.rs": (
        "b09682d386a438f6d1e3765725efeb27155cffc220ca7192cd6038014e07b9f0"
    ),
    "engine_rs/src/native_agents/v47/race.rs": (
        "483ac19861dea65d16a711982bd2d1cc566897c4d7bf03383a926ec4c430b305"
    ),
    "engine_rs/src/native_agents/v48/compact.rs": (
        "7de1fd89a128f3548e51a720c0926ed10ec59a1a062fb180cb3aa19875b872e6"
    ),
    "engine_rs/fixtures/v43-routes.json": (
        "8dcd59e074e2842d781b6b00769cb060c94af1ee39e36a78a3457c6f165000c9"
    ),
    "engine_rs/fixtures/farm2945-sell-library.bin": (
        "1ad6523dd8fb09439340100a2dc0a6083d7accd91bc54915667a1e54da91ebd6"
    ),
    "engine_rs/fixtures/metav4-sell-library.bin": (
        "f5c3d06b214d369768a92e6dfb8e07aa5e5da6cffe15bd423af4160261ee73d4"
    ),
}
# Cha22 closure files whose only edits are the v3 Game-view accessor lines.
ADAPTED_HASHES = {
    "engine_rs/src/native_agents/cha22/execution_tests.rs": (
        "6ef8cf8ef2a49c0c519f6095145dc34c72ac38fd8d945ef4b29b60d957afe1b6"
    ),
    "engine_rs/src/native_agents/cha22/mod.rs": (
        "801eb7bf33d3c61fdfcddbf4af5ffc13cc8bee99e4fd181e4b65eb47c8baa269"
    ),
    "engine_rs/src/native_agents/farm2945/mod.rs": (
        "adba16f6d7994711917e7f086a574b49639f33ce6d57c43d766b32fc4ba7cb0c"
    ),
    "engine_rs/src/native_agents/metav4/mod.rs": (
        "7fed16b8b2714536e71add914674916cfdb5707f19346f217b0b1c6477330ea5"
    ),
    "engine_rs/src/native_agents/pipe16.rs": (
        "a13fba34d0e743551cbf3a6e99d7aa1117632d95929864eb1d0a816ada7ff615"
    ),
    "engine_rs/src/native_agents/v43/mod.rs": (
        "1ef789938e5ee571081cbe05e50a5311e5f97cb676ce884e065693e1f3e42e77"
    ),
    "engine_rs/src/native_agents/v47/mod.rs": (
        "bf6b19129f2808ca890fd3832ab74bd189f4e325a3dff0a41d01d34c05409932"
    ),
    "engine_rs/src/native_agents/v48/mod.rs": (
        "5cfe000bbe28d9ba5059739d9b11caa6da6ea7f2e1b157e1510677e81cb26517"
    ),
}
PYTHON_MAIN_HASHES = {
    "r04": "22d074391822206872448a6114a37ba2a4a39eda2ddbbcdc0bc8aa8cc6a64188",
    "ecobot": "0dc02e03c94ef60c06b5093efc2e2fd0530aa6eea20df507a90b90d6651bd067",
    "e776": "0cc2a88594f82b6c8d3cb15fcd4aa2df2bdd2a0a7fb0133d95a3204dd2d6ba38",
}
DEFAULT_CONFIGURATION = {
    "actTimeout": 1,
    "boardSize": 10,
    "episodeSteps": 720,
    "farmHandCostMult": 1,
    "marketParams": {},
    "maxMarketOrdersPerTurn": 10,
    "runTimeout": 1200,
    "seed": None,
    "shedCapacity": 100,
    "startingMoney": 3000,
    "townCenterSellInterval": 24,
    "townShopSellInterval": 4,
    "townShopUnlockInterval": 3,
    "turnsPerDay": 24,
    "weedSpawnChance": 0.005,
}
TRACE_KEYS = (
    "path",
    "sha256",
    "bytes",
    "seed",
    "policies",
    "transitions",
    "compared_actions",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: object, keys: Sequence[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label}: expected object")
    result = cast(dict[str, Any], value)
    _require(set(result) == set(keys), f"{label}: schema keys must be {list(keys)}")
    return result


def _array(value: object, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label}: expected array")
    return cast(list[Any], value)


def _string(value: object, label: str) -> str:
    _require(isinstance(value, str), f"{label}: expected string")
    return cast(str, value)


def _integer(value: object, label: str) -> int:
    _require(type(value) is int, f"{label}: expected integer")
    return cast(int, value)


def _text(value: object, label: str) -> str:
    text = _string(value, label)
    _require(bool(text.strip()), f"{label}: must not be empty")
    return text


def _digest(value: object, label: str) -> str:
    digest = _string(value, label)
    _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"{label}: SHA-256")
    return digest


def _path(value: object, *, crate: bool = False) -> str:
    path = _string(value, "path")
    parts = path.split("/")
    _require(
        "\\" not in path
        and "\x00" not in path
        and not any(part in ("", ".", "..", ".git") for part in parts),
        f"{path!r}: unsafe path",
    )
    if crate:
        _require(
            len(parts) > 1
            and parts[0] == "opponents_rs"
            and parts[1] not in ("target", "OPPONENT_MANIFEST.json"),
            f"{path}: invalid crate path",
        )
    return path


def _inventory(actual: Sequence[str], expected: Sequence[str], label: str) -> None:
    _require(len(actual) == len(set(actual)), f"{label}: duplicate path")
    missing, extra = (
        sorted(set(expected) - set(actual)),
        sorted(set(actual) - set(expected)),
    )
    _require(not missing and not extra, f"{label}: missing={missing}; extra={extra}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON duplicate key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _loads(data: bytes | str) -> Any:
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=_invalid_constant)


def reference_files(root: Path) -> dict[str, bytes]:
    """Read exactly the selected (imported and adapted) blobs at the pinned commit."""
    return {
        path: subprocess.check_output(["git", "show", f"{PIN}:{path}"], cwd=root)
        for path in [*IMPORT_HASHES, *ADAPTED_HASHES]
    }


def view_adapted(reference: bytes, label: str) -> bytes:
    """Apply exactly the declared Game-view accessor edits to a pinned blob."""
    adapted = reference
    for old, new in VIEW_EDITS:
        _require(
            adapted.count(old) == 1,
            f"{label}: view adaptation expects one {old.decode()!r}",
        )
        adapted = adapted.replace(old, new)
    return adapted


def comment_lines(source: bytes) -> bytes:
    """Every line of the original submission that starts with ``#``, in order."""
    return b"".join(
        line for line in source.splitlines(keepends=True) if line.startswith(b"#")
    )


def cha22_source() -> bytes:
    """Read the caller-supplied original Cha22 submission and check its hash."""
    location = os.environ.get(CHA22_SOURCE_ENV)
    _require(
        bool(location),
        f"set {CHA22_SOURCE_ENV} to the original Cha22 main.py "
        f"(SHA-256 {CHA22_SOURCE_SHA256})",
    )
    data = Path(cast(str, location)).read_bytes()
    _require(sha(data) == CHA22_SOURCE_SHA256, "Cha22 Python source hash changed")
    return data


@lru_cache(maxsize=1)
def _python_files() -> dict[str, dict[str, bytes]]:
    """Resolve the independent pinned Python dependency closure, without execution."""
    distribution = importlib.metadata.distribution("kaggle-environments")
    _require(distribution.version == KAGGLE_VERSION, "Starter package version changed")
    starter = Path(
        str(distribution.locate_file(f"kaggle_environments/{STARTER_PATH}"))
    ).read_bytes()
    _require(
        sha(starter) == PYTHON_ENGINE_SHA256, "Starter package source hash changed"
    )

    def blob(path: str) -> bytes:
        return subprocess.check_output(
            ["git", "show", f"{PYTHON_PIN}:{path}"], cwd=PYTHON_REPO
        )

    sources: dict[str, dict[str, bytes]] = {"starter": {STARTER_PATH: starter}}
    for bot, digest in PYTHON_MAIN_HASHES.items():
        path = f"agents/{bot}/main.py"
        data = blob(path)
        _require(sha(data) == digest, f"{bot}: pinned Python main source hash changed")
        sources[bot] = {path: data}
    provenance = "agents/ecobot/PROVENANCE.md"
    sources["ecobot"][provenance] = blob(provenance)
    manifest_path = "agents/e776/MANIFEST.sha256"
    manifest = blob(manifest_path)
    _require(sha(manifest) == E776_MANIFEST_SHA256, "E776 source manifest hash changed")
    sources["e776"][manifest_path] = manifest
    for line in manifest.decode().splitlines():
        digest, relative = line.split("  ", 1)
        path = f"agents/e776/{_path(relative)}"
        data = blob(path)
        _require(sha(data) == _digest(digest, path), f"{path}: E776 dependency hash")
        sources["e776"][path] = data
    sources["cha22"] = {CHA22_SOURCE_PATH: cha22_source()}
    return sources


def _python_origin(bot: str) -> tuple[str, str]:
    """The pinned (repository, version) that holds each original Python bot."""
    if bot == "starter":
        return "kaggle-environments", KAGGLE_VERSION
    if bot == "cha22":
        return CHA22_SOURCE_REPO, CHA22_SOURCE_VERSION
    return PYTHON_REPO, PYTHON_PIN


def python_oracle_receipts() -> list[dict[str, Any]]:
    """Build source metadata for the manifest; no original source is copied."""
    notices = {
        "starter": (
            "Built-in starter_agent in pinned Kaggle engine package; "
            "engine/package notices apply."
        ),
        "r04": (
            "Original agents/r04/main.py at pinned sibling commit; no agent "
            "PROVENANCE.md. Original notice/license custody remains unresolved."
        ),
        "ecobot": (
            "Original PROVENANCE.md declares no software license; do not "
            "redistribute without resolving that license gap."
        ),
        "e776": (
            "Original PROVENANCE.md attributes tapes as CC0, but declares no "
            "software license for the submission; do not redistribute without "
            "resolving that license gap."
        ),
        "cha22": (
            "Public notebook output main.py, Apache-2.0 with its upstream notices "
            "(opponents_rs/notices/cha22); full ig_agent entry. Not copied here."
        ),
    }
    return [
        {
            "bot": bot,
            "source_repo": _python_origin(bot)[0],
            "source_commit": _python_origin(bot)[1],
            "files": [
                {"path": path, "sha256": sha(data)}
                for path, data in sorted(files.items())
            ],
            "provenance": notices[bot],
        }
        for bot, files in _python_files().items()
    ]


def _structural_python_pins(bot: str, files: Mapping[str, str]) -> None:
    """Hashes this checker pins itself; needs no source repository."""
    pins = {
        "starter": {STARTER_PATH: PYTHON_ENGINE_SHA256},
        "e776": {
            "agents/e776/main.py": PYTHON_MAIN_HASHES["e776"],
            "agents/e776/MANIFEST.sha256": E776_MANIFEST_SHA256,
        },
        "cha22": {CHA22_SOURCE_PATH: CHA22_SOURCE_SHA256},
    }.get(bot, {f"agents/{bot}/main.py": PYTHON_MAIN_HASHES.get(bot, "")})
    for path, digest in pins.items():
        _require(path in files, f"{bot}: Python source inventory missing={path}")
        _require(files[path] == digest, f"{path}: Python source hash")


def _verify_python(value: object, *, original_sources: bool) -> None:
    entries = _array(value, "python_oracles")
    expected = _python_files() if original_sources else None
    seen = []
    for value in entries:
        entry = _object(
            value,
            ("bot", "source_repo", "source_commit", "files", "provenance"),
            "Python oracle",
        )
        bot = _string(entry["bot"], "bot")
        _require(bot in BOTS, f"unknown Python oracle: {bot}")
        seen.append(bot)
        repo, version = _python_origin(bot)
        _require(entry["source_repo"] == repo, f"{bot}: source_repo")
        _require(entry["source_commit"] == version, f"{bot}: source_commit")
        _text(entry["provenance"], f"{bot}: provenance")
        files: dict[str, str] = {}
        for value in _array(entry["files"], f"{bot}: files"):
            source = _object(value, ("path", "sha256"), f"{bot}: file")
            path = _path(source["path"])
            _require(path not in files, f"{bot}: duplicate Python source {path}")
            files[path] = _digest(source["sha256"], path)
        _structural_python_pins(bot, files)
        if expected is not None:
            for path, digest in files.items():
                _require(
                    path in expected[bot],
                    f"{bot}: Python source inventory extra={path}",
                )
                _require(
                    digest == sha(expected[bot][path]), f"{path}: Python source hash"
                )
            _inventory(
                list(files), list(expected[bot]), f"{bot}: Python source inventory"
            )
    _inventory(seen, list(BOTS), "Python oracle inventory")


def _trace_entry(value: object) -> dict[str, Any]:
    entry = _object(value, TRACE_KEYS, "oracle trace")
    path = _path(entry["path"], crate=True)
    _require(
        any(
            re.fullmatch(
                re.escape(directory) + r"/[a-z0-9][a-z0-9.-]*\.jsonl\.gz", path
            )
            for directory in ORACLE_DIRS
        ),
        f"{path}: trace path",
    )
    _digest(entry["sha256"], path)
    _require(_integer(entry["bytes"], f"{path}: bytes") > 0, f"{path}: bytes")
    _integer(entry["seed"], f"{path}: seed")
    policies = _array(entry["policies"], f"{path}: policies")
    _require(
        len(policies) == 2 and all(p in POLICIES for p in policies), f"{path}: policies"
    )
    transitions = _integer(entry["transitions"], f"{path}: transitions")
    _require(0 < transitions <= 719, f"{path}: transitions outside default episode")
    counts = _array(entry["compared_actions"], f"{path}: compared_actions")
    _require(len(counts) == 2, f"{path}: compared_actions must have both seats")
    for count in counts:
        _require(
            0 <= _integer(count, f"{path}: compared_actions") <= transitions,
            f"{path}: compared_actions outside available trace",
        )
    return entry


def _trace_coverage(records: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    """Recount stored observations independently of the generator's counters.

    The inventory-index comparison is a diagnostic proxy, not market stock or
    rejected-order evidence. Sales are submitted final-day SELL orders; hires
    are observed hand-count increases. These contiguous traces never reset or
    replay their controllers within an episode.
    """
    result = [dict.fromkeys(COVERAGE_KEYS, 0) for _ in range(2)]
    before = records[0]["initial"]["public"]
    for record in records[1:]:
        if record["type"] == "rejected":
            for count in result:
                count["rejected_steps"] += 1
            continue
        after = record["expected"]
        for seat in range(2):
            count = result[seat]
            farm = before["farms"][seat]
            count["openings"] += int(record["from_step"] == 0)
            count["day_resets"] += int(record["from_step"] > 0 and before["hour"] == 0)
            weeds = [
                tile
                for row in farm["tiles"]
                for tile in row
                if isinstance(tile, dict) and tile["kind"] == "WEED"
            ]
            count["weed_presence"] += int(bool(weeds))
            count["hires"] += max(
                0, len(after["farms"][seat]["hands"]) - len(farm["hands"])
            )
            # Cha22 submits empty orders, which the engine ignores.
            for order in filter(None, record["actions"][seat]["market"]):
                count["final_day_sell_orders"] += int(
                    before["day"] == 29 and order[0] == "SELL"
                )
                if order[0] == "BUY_PRODUCT":
                    inventory = before["market"]["inventory"]
                    index = inventory.get(order[1], 0)
                    count["buy_quantity_above_inventory_index"] += int(
                        int(order[2]) > index
                    )
        before = after
    return result


def _trace_content(
    data: bytes, entry: Mapping[str, Any], generated: Mapping[str, Any]
) -> None:
    path = entry["path"]
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        expanded = stream.read(MAX_TRACE_EXPANSION + 1)
    _require(
        len(expanded) <= MAX_TRACE_EXPANSION,
        f"{path}: trace expansion exceeds safety bound",
    )
    records = [_loads(line) for line in expanded.splitlines()]
    _require(bool(records), f"{path}: empty trace")
    header = _object(
        records[0],
        (
            "type",
            "format",
            "source",
            "seed",
            "configuration",
            "shop_schedule",
            "rng_schedule",
            "initial",
            "terminal_banks",
            "transitions",
        ),
        f"{path}: header",
    )
    _require(
        header["type"] == "header" and header["format"] == TRACE_FORMAT,
        f"{path}: trace format",
    )
    source = _object(
        header["source"],
        (
            "generator",
            "module_version",
            "engine_sha256",
            "name",
            "policies",
            "policy_seed",
            "config_variant",
        ),
        f"{path}: source",
    )
    _require(
        source["generator"] == GENERATOR
        and source["module_version"] == KAGGLE_VERSION
        and source["engine_sha256"] == PYTHON_ENGINE_SHA256,
        f"{path}: source pin",
    )
    _require(header["seed"] == entry["seed"], f"{path}: seed differs from header")
    _require(
        source["policies"] == entry["policies"], f"{path}: policies differ from header"
    )
    _require(
        source["config_variant"] == "default", f"{path}: config_variant must be default"
    )
    _require(source["policy_seed"] == generated["policy_seed"], f"{path}: policy_seed")
    _integer(header["seed"], f"{path}: header seed")
    _integer(header["transitions"], f"{path}: header transitions")
    actual_configuration = _object(
        header["configuration"], tuple(DEFAULT_CONFIGURATION), f"{path}: configuration"
    )
    configuration = dict(DEFAULT_CONFIGURATION)
    if actual_configuration["seed"] is not None:
        _integer(actual_configuration["seed"], f"{path}: configuration seed")
        configuration["seed"] = entry["seed"]
    _require(
        header["configuration"] == configuration, f"{path}: non-default configuration"
    )
    transitions, rejected = 0, 0
    for line, value in enumerate(records[1:], start=2):
        _require(isinstance(value, dict), f"{path}:{line}: record object")
        if value["type"] == "transition":
            record = _object(
                value,
                (
                    "type",
                    "from_step",
                    "actions",
                    "expected",
                    "privates",
                    "statuses",
                    "rewards",
                ),
                f"{path}:{line}",
            )
            _require(
                record["from_step"] == transitions,
                f"{path}:{line}: non-contiguous step",
            )
            transitions += 1
        elif value["type"] == "rejected":
            record = _object(
                value,
                ("type", "from_step", "actions", "python_error"),
                f"{path}:{line}",
            )
            _require(
                record["from_step"] == transitions, f"{path}:{line}: rejected step"
            )
            rejected += 1
        else:
            raise ValueError(f"{path}:{line}: unknown record type")
        _require(
            len(_array(record["actions"], f"{path}:{line}: actions")) == 2,
            f"{path}:{line}: actions need both seats",
        )
    _require(
        transitions == entry["transitions"] == header["transitions"],
        f"{path}: transitions differ from content",
    )
    _require(
        rejected == generated["rejected"],
        f"{path}: rejected count differs from content",
    )
    _require(
        _trace_coverage(records) == generated["coverage"],
        f"{path}: coverage differs from stored observations/actions",
    )


def _verify_traces(entries: list[dict[str, Any]], current: Mapping[str, bytes]) -> None:
    total = 0
    for entry in entries:
        path, data = entry["path"], current[entry["path"]]
        _require(sha(data) == entry["sha256"], f"{path}: trace hash")
        _require(len(data) == entry["bytes"], f"{path}: trace size")
        total += len(data)
    _require(
        total <= TRACE_BUDGET,
        f"oracle traces use {total:,} B; budget is {TRACE_BUDGET:,} B",
    )
    for directory in ORACLE_DIRS:
        listed = [e for e in entries if str(Path(e["path"]).parent) == directory]
        if listed:
            _verify_oracle_manifest(f"{directory}/MANIFEST.json", listed, current)


def _verify_oracle_manifest(
    manifest: str, entries: list[dict[str, Any]], current: Mapping[str, bytes]
) -> None:
    """One generated corpus: its MANIFEST lists exactly the traces beside it."""
    raw = _object(
        _loads(current[manifest]),
        (
            "schema_version",
            "format",
            "generator",
            "kaggle_environments_version",
            "python_engine_sha256",
            "byte_budget",
            "traces",
        ),
        "oracle MANIFEST",
    )
    _require(
        type(raw["schema_version"]) is int and raw["schema_version"] == 1,
        "oracle schema_version",
    )
    _require(
        raw["format"] == TRACE_FORMAT
        and raw["generator"] == GENERATOR
        and raw["kaggle_environments_version"] == KAGGLE_VERSION
        and raw["python_engine_sha256"] == PYTHON_ENGINE_SHA256,
        "oracle MANIFEST source pins",
    )
    _require(
        type(raw["byte_budget"]) is int and raw["byte_budget"] == TRACE_BUDGET,
        "oracle budget must remain 4,000,000 B",
    )
    by_name = {Path(entry["path"]).name: entry for entry in entries}
    listed = []
    for value in _array(raw["traces"], "oracle MANIFEST traces"):
        generated = _object(
            value,
            (
                *(key for key in TRACE_KEYS if key != "compared_actions"),
                "available_actions",
                "policy_seed",
                "config_variant",
                "python_runtime",
                "rejected",
                "coverage",
            ),
            "oracle MANIFEST trace",
        )
        name = _path(generated["path"])
        _require(name in by_name, f"{name}: oracle MANIFEST trace inventory")
        listed.append(name)
        entry = by_name[name]
        for key in ("sha256", "bytes", "seed", "policies", "transitions"):
            _require(
                generated[key] == entry[key], f"{name}: oracle MANIFEST {key} differs"
            )
        for key in ("bytes", "seed", "transitions", "policy_seed", "rejected"):
            _integer(generated[key], f"{name}: {key}")
        for count in _array(
            generated["available_actions"], f"{name}: available_actions"
        ):
            _integer(count, f"{name}: available_actions")
        _integer(generated["policy_seed"], f"{name}: policy_seed")
        _integer(generated["rejected"], f"{name}: rejected")
        _require(
            generated["available_actions"] == [entry["transitions"]] * 2,
            f"{name}: available_actions",
        )
        _require(
            generated["config_variant"] == "default", f"{name}: default config only"
        )
        # The competition runtime; CPython 3.12's compensated float sum() changes
        # R04 decisions, so no other interpreter can serve as the oracle.
        _require(
            re.fullmatch(r"3\.11\.\d+", _string(generated["python_runtime"], name))
            is not None,
            f"{name}: oracle must come from CPython 3.11 (Kaggle runtime)",
        )
        coverage = _array(generated["coverage"], f"{name}: coverage")
        _require(len(coverage) == 2, f"{name}: coverage needs both seats")
        for counts in coverage:
            for key, count in _object(
                counts, COVERAGE_KEYS, f"{name}: coverage"
            ).items():
                _require(
                    _integer(count, f"{name}: {key}") >= 0, f"{name}: negative coverage"
                )
        _trace_content(current[entry["path"]], entry, generated)
    _inventory(listed, list(by_name), "oracle MANIFEST trace inventory")


def _verify_notices(
    entries: list[dict[str, Any]],
    current: Mapping[str, bytes],
    *,
    original_sources: bool,
) -> None:
    """Upstream notices are exact copies (or the declared comment-line excerpt)."""
    _inventory([entry["path"] for entry in entries], list(NOTICES), "notice inventory")
    for entry in entries:
        path = entry["path"]
        repo, version, source, digest = NOTICES[path]
        _require(
            (entry["source_repo"], entry["source_commit"], entry["source_path"])
            == (repo, version, source),
            f"{path}: notice source",
        )
        _require(
            _digest(entry["sha256"], path) == digest == sha(current[path]),
            f"{path}: notice hash",
        )
        if original_sources:
            original = (
                comment_lines(cha22_source())
                if source == COMMENT_LINES
                else subprocess.check_output(
                    ["git", "show", f"{version}:{source}"], cwd=repo
                )
            )
            _require(original == current[path], f"{path}: notice differs from source")


def verify(
    value: object,
    originals: Mapping[str, bytes],
    current: Mapping[str, bytes],
    *,
    original_sources: bool = False,
) -> None:
    """Check an inventory, raising ValueError at the first custody failure.

    ``original_sources`` re-reads the original Python files (sibling repository,
    installed Starter package and the supplied Cha22 source) and the v2 notice
    blobs; the default pins only recorded entry hashes.
    """
    raw = _object(
        value,
        (
            "schema_version",
            "reference_commit",
            "imported",
            "adapted",
            "notices",
            "authored",
            "python_oracles",
            "oracle_traces",
            "trace_budget_bytes",
        ),
        "manifest",
    )
    _require(
        _integer(raw["schema_version"], "schema_version") == 1,
        "schema_version must be 1",
    )
    _require(raw["reference_commit"] == PIN, f"reference_commit must be {PIN}")
    _require(
        _integer(raw["trace_budget_bytes"], "trace_budget_bytes") == TRACE_BUDGET,
        "trace budget must remain 4,000,000 B",
    )
    imports = [
        _object(
            entry, ("path", "reference_path", "reference_sha256", "sha256"), "imported"
        )
        for entry in _array(raw["imported"], "imported")
    ]
    adapted = [
        _object(entry, ADAPTED_KEYS, "adapted")
        for entry in _array(raw["adapted"], "adapted")
    ]
    notices = [
        _object(entry, NOTICE_KEYS, "notice")
        for entry in _array(raw["notices"], "notices")
    ]
    authored = [
        _object(entry, ("path", "sha256", "reason"), "authored")
        for entry in _array(raw["authored"], "authored")
    ]
    traces = [
        _trace_entry(entry) for entry in _array(raw["oracle_traces"], "oracle_traces")
    ]
    _require(bool(traces), "at least one oracle trace is required")
    declared = [
        _path(entry["path"], crate=True)
        for entry in [*imports, *adapted, *notices, *authored, *traces]
    ]
    _inventory(declared, list(current), "current inventory")
    authored_paths = [entry["path"] for entry in authored]
    for directory in {str(Path(entry["path"]).parent) for entry in traces}:
        _require(
            f"{directory}/MANIFEST.json" in authored_paths,
            f"{directory}: oracle MANIFEST must be authored",
        )
    reference_paths = [_path(entry["reference_path"]) for entry in imports]
    _inventory(reference_paths, list(IMPORT_HASHES), "import inventory")
    _inventory(
        [_path(entry["reference_path"]) for entry in adapted],
        list(ADAPTED_HASHES),
        "adapted inventory",
    )
    _inventory(
        list(originals), [*IMPORT_HASHES, *ADAPTED_HASHES], "reference inventory"
    )
    for entry in imports:
        path, reference = entry["path"], entry["reference_path"]
        _require(
            path == reference.replace("engine_rs/", "opponents_rs/", 1),
            f"{path}: import destination",
        )
        _require(
            _digest(entry["reference_sha256"], path)
            == sha(originals[reference])
            == IMPORT_HASHES[reference],
            f"{path}: reference hash",
        )
        _require(
            sha(current[path]) == _digest(entry["sha256"], path), f"{path}: import hash"
        )
        _require(
            current[path] == originals[reference]
            and entry["sha256"] == entry["reference_sha256"],
            f"{path}: import must remain byte-exact",
        )
    for entry in adapted:
        path, reference = entry["path"], entry["reference_path"]
        _require(
            path == reference.replace("engine_rs/", "opponents_rs/", 1),
            f"{path}: adapted destination",
        )
        _require(entry["adaptation"] == VIEW_ADAPTATION, f"{path}: adaptation")
        _require(
            _digest(entry["reference_sha256"], path)
            == sha(originals[reference])
            == ADAPTED_HASHES[reference],
            f"{path}: reference hash",
        )
        _require(
            sha(current[path]) == _digest(entry["sha256"], path),
            f"{path}: adapted hash",
        )
        _require(
            current[path] == view_adapted(originals[reference], path),
            f"{path}: only the Game-view accessor lines may differ",
        )
    for entry in authored:
        path = entry["path"]
        _text(entry["reason"], f"{path}: reason")
        _require(
            sha(current[path]) == _digest(entry["sha256"], path),
            f"{path}: authored hash",
        )
    _verify_python(raw["python_oracles"], original_sources=original_sources)
    _verify_notices(notices, current, original_sources=original_sources)
    _verify_traces(traces, current)


def current_files(root: Path) -> dict[str, bytes]:
    """Read all crate files, refusing symlinks, except build output and manifest."""
    current: dict[str, bytes] = {}

    def walk(directory: Path) -> None:
        _require(not directory.is_symlink(), f"{directory}: symlink")
        for child in sorted(directory.iterdir()):
            relative = child.relative_to(root).as_posix()
            if relative in ("opponents_rs/target", MANIFEST):
                continue
            _require(not child.is_symlink(), f"{relative}: symlink")
            if child.is_dir():
                walk(child)
            else:
                _require(child.is_file(), f"{relative}: not a regular file")
                current[relative] = child.read_bytes()

    walk(root / "opponents_rs")
    return current


def check(root: Path, *, original_sources: bool = False) -> None:
    root = root.resolve()
    manifest_path = root / MANIFEST
    _require(not manifest_path.is_symlink(), f"{manifest_path}: symlink")
    verify(
        _loads(manifest_path.read_bytes()),
        reference_files(root),
        current_files(root),
        original_sources=original_sources,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--original-sources",
        action="store_true",
        help="also re-read every original Python file (needs the sibling repository)",
    )
    args = parser.parse_args(argv)
    try:
        check(args.root, original_sources=args.original_sources)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        print(f"opponent import check failed: {error}", file=sys.stderr)
        return 1
    scope = "original sources re-read" if args.original_sources else "entry pins"
    print(
        f"opponent import check passed ({scope}; custody only; "
        "parity is checked separately)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
