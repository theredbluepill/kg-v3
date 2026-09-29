"""Record the pinned reference encoder; validate custody without rebuilding it.

The Rust input producer owns gameplay and the recipe. This driver only validates
metadata, counts coverage, moves bytes, and invokes the original reference crate.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[2]
PIN = "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0"
FEATURE_SHA = "24a7d09f9ddf8196aed7e5dc3b18d7590562370d697c8407f95b4cc8b4f9a6ab"
FORMAT = "kaggriculture-observation-oracle-f32le-gzip-v1"
SHUFFLED_FORMAT = "kaggriculture-observation-oracle-f32le-byteplanes-gzip-v1"
FEATURES = 8176
SEAT_BYTES = FEATURES * 4
EPISODES = (95324500, 95901360, 95921764, 95990191)
OFFICIAL_STEPS = [*range(32), *range(344, 376), *range(687, 719)]
PROFILES = [
    [96, 24, 10, 1, 100, 3, 4, 24, 3000, 0.005],
    [96, 12, 4, 3, 64, 2, 3, 12, 7500, 0.0],
    [96, 8, 3, 0, 17, 1, 2, 8, 12345, 0.02],
    [96, 6, 1, 7, 256, 5, 7, 9, 40000, 0.01],
    [96, 30, 8, 2, 500, 4, 5, 30, 9000, 0.125],
    [96, 16, 5, 5, 33, 2, 4, 16, 100000, 1.0],
]
ITEMS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
]
COVERAGE_TAGS = [
    "records",
    *[f"tile_{s}" for s in ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")],
    *[f"crop_{s}" for s in ITEMS[:5]],
    *[f"animal_{s}" for s in ITEMS[9:]],
    "fert_current",
    "fert_expired",
    "unwatered",
    "unfed",
    "actor_gt16_states",
    "reordered_inventories",
    "reordered_sheds",
    "shops_ge4_states",
    "both_hires_nonzero_states",
]
MAX_COMPRESSED = 8 * 1024**2
MAX_EXPANDED = 128 * 1024**2
MAX_ROW = 2 * 1024**2
# A shared Unix deadline spans Mac Python 3.9/3.11 process clock differences.
# Convert once to this process's monotonic clock, leaving five seconds to clean up.
RUN_DEADLINE = (
    time.monotonic()
    + min(
        115.0,
        float(os.environ.get("KG_BOUNDED_DEADLINE_UNIX", time.time() + 115))
        - time.time(),
    )
    - 5
)
COMMAND_RECEIPTS: list[dict[str, Any]] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def obj(value: Any, keys: Iterable[str], label: str) -> dict[str, Any]:
    expected = tuple(keys)
    require(
        isinstance(value, dict) and set(value) == set(expected),
        f"{label}: exact schema keys {list(expected)} required",
    )
    return value


def integer(value: Any, label: str, minimum: int = 0) -> int:
    require(
        type(value) is int and value >= minimum,
        f"{label}: integer >= {minimum} required",
    )
    return value


def digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label}: SHA-256 required",
    )
    return value


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse(raw: str | bytes) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def header_bytes(raw: bytes) -> bytes:
    """Retain Rust's number spelling and nested key order, rather than re-encoding."""
    text = raw.decode()
    decoder = json.JSONDecoder()
    at = text.index("{") + 1
    while True:
        while text[at].isspace() or text[at] == ",":
            at += 1
        key, at = decoder.raw_decode(text, at)
        while text[at].isspace():
            at += 1
        require(text[at] == ":", "record separator")
        at += 1
        while text[at].isspace():
            at += 1
        start = at
        _, at = decoder.raw_decode(text, at)
        if key == "header":
            return text[start:at].encode()


def check_source(source: Any) -> dict[str, Any]:
    require(isinstance(source, dict), "record source must be an object")
    kind = source.get("kind")
    if kind == "official":
        obj(source, ("kind", "episode", "step"), "official source")
        require(
            integer(source["episode"], "episode") in EPISODES,
            "unknown official episode",
        )
        integer(source["step"], "step")
    elif kind == "seeded":
        obj(
            source,
            ("kind", "seed", "profile", "step", "policy_sha256"),
            "seeded source",
        )
        profile = integer(source["profile"], "profile")
        integer(source["seed"], "seed")
        require(
            profile < 6 and source["seed"] == 11001 + profile, "seed/profile mismatch"
        )
        integer(source["step"], "step")
        digest(source["policy_sha256"], "policy SHA")
    elif kind == "dense":
        obj(source, ("kind", "case"), "dense source")
        require(integer(source["case"], "dense case") < 32, "dense case outside recipe")
    else:
        raise ValueError(f"invalid source tag {kind!r}")
    return source


def check_header(header: Any) -> None:
    obj(
        header,
        (
            "format",
            "seed",
            "configuration",
            "shop_schedule",
            "rng_schedule",
            "initial",
            "terminal_banks",
            "transitions",
        ),
        "header",
    )
    require(header["format"] == "kaggriculture-re-parity-v1", "header format")
    integer(header["seed"], "resolved seed", -(1 << 63))
    require(
        header["shop_schedule"]
        == header["rng_schedule"]
        == header["terminal_banks"]
        == []
        and header["transitions"] == 0,
        "oracle header must contain no future schedules",
    )
    config = header["configuration"]
    config_keys = {
        "episodeSteps",
        "boardSize",
        "startingMoney",
        "maxMarketOrdersPerTurn",
        "turnsPerDay",
        "shedCapacity",
        "weedSpawnChance",
        "townShopUnlockInterval",
        "townShopSellInterval",
        "townCenterSellInterval",
        "farmHandCostMult",
        "marketParams",
    }
    require(
        isinstance(config, dict)
        and config_keys
        <= set(config)
        <= config_keys | {"actTimeout", "runTimeout", "seed"},
        "complete configuration keys",
    )
    initial = obj(header["initial"], ("public", "privates"), "initial")
    public = obj(
        initial["public"], ("step", "day", "hour", "farms", "market", "town"), "public"
    )
    turns = integer(config["turnsPerDay"], "turnsPerDay", 1)
    step = integer(public["step"], "step")
    require(
        (public["day"], public["hour"]) == divmod(step, turns), "header clock mismatch"
    )
    require(
        len(public["farms"]) == len(initial["privates"]) == 2,
        "two farms/privates required",
    )
    for farm, private in zip(public["farms"], initial["privates"], strict=True):
        obj(
            farm,
            ("money", "tiles", "farmer", "hands", "unlocked_quadrants", "hires_today"),
            "farm",
        )
        obj(private, ("shed", "seeds", "inventories"), "private")
    obj(public["town"], ("unlocked_shops",), "town")
    market = public["market"]
    require(
        set(market) in ({"inventory", "prices"}, {"inventory", "prices", "params"}),
        "market keys",
    )
    require(
        config["marketParams"] == {} and market.get("params") in (None, {}),
        "market parameters must use defaults",
    )


def iter_rows(stream: BinaryIO) -> Iterator[tuple[dict[str, Any], bytes]]:
    total = 0
    while raw := stream.readline(MAX_ROW + 1):
        total += len(raw)
        require(
            len(raw) <= MAX_ROW and total <= MAX_EXPANDED,
            "state expansion budget exceeded",
        )
        require(raw.endswith(b"\n"), "truncated state row")
        row = obj(parse(raw), ("record_id", "source", "header"), "record")
        require(
            isinstance(row["record_id"], str) and bool(row["record_id"]),
            "record id required",
        )
        check_source(row["source"])
        check_header(row["header"])
        if row["source"]["kind"] != "dense":
            require(
                row["source"]["step"] == row["header"]["initial"]["public"]["step"],
                "source/header step mismatch",
            )
        if row["source"]["kind"] == "seeded":
            require(
                row["source"]["seed"] == row["header"]["seed"],
                "source/header seed mismatch",
            )
        yield row, raw


def empty_coverage() -> dict[str, Any]:
    return {
        "non_synthetic": dict.fromkeys(COVERAGE_TAGS, 0),
        "dense": dict.fromkeys(COVERAGE_TAGS, 0),
        "shed_order_exception": False,
    }


def add_coverage(coverage: dict[str, Any], row: dict[str, Any]) -> None:
    count = coverage["dense" if row["source"]["kind"] == "dense" else "non_synthetic"]
    state = row["header"]["initial"]
    public, privates = state["public"], state["privates"]
    count["records"] += 1
    count["actor_gt16_states"] += any(len(f["hands"]) + 1 > 16 for f in public["farms"])
    count["shops_ge4_states"] += len(public["town"]["unlocked_shops"]) >= 4
    count["both_hires_nonzero_states"] += all(
        f["hires_today"] > 0 for f in public["farms"]
    )
    for farm, private in zip(public["farms"], privates, strict=True):
        count["reordered_sheds"] += list(private["shed"]) != sorted(
            private["shed"], key=ITEMS.index
        )
        for inv in private["inventories"]:
            count["reordered_inventories"] += list(inv) != sorted(inv, key=ITEMS.index)
        for tile in (t for line in farm["tiles"] for t in line):
            kind = (
                "EMPTY"
                if tile is None
                else tile
                if isinstance(tile, str)
                else tile["kind"]
            )
            count[f"tile_{kind}"] += 1
            if not isinstance(tile, dict):
                continue
            if kind == "PLANT":
                count[f"crop_{tile['crop']}"] += 1
                until = tile["fertilized_until_day"]
                count["fert_current"] += until >= public["day"]
                count["fert_expired"] += 0 <= until < public["day"]
                count["unwatered"] += tile["consecutive_unwatered"] > 0
            if "animal" in tile:
                count[f"animal_{tile['animal']}"] += 1
                count["unfed"] += tile["consecutive_unfed"] > 0


def count_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    coverage = empty_coverage()
    for row in rows:
        add_coverage(coverage, row)
    coverage["shed_order_exception"] = (
        coverage["non_synthetic"]["reordered_sheds"] == 0
        and coverage["dense"]["reordered_sheds"] > 0
    )
    return coverage


def validate_dense_pair(first: dict[str, Any], second: dict[str, Any]) -> None:
    expected = copy.deepcopy(first)
    private = expected["initial"]["privates"][0]
    for mapping in (private["inventories"][0], private["shed"]):
        require(len(mapping) > 1, "dense d30/d31 needs nontrivial rank change")
        key = next(iter(mapping))
        mapping[key] = mapping.pop(key)
    require(
        json_bytes(expected) == json_bytes(second),
        "dense d30/d31 must differ only by two remove/reinsert orders",
    )
    require(
        list(private["shed"]) != sorted(private["shed"], key=ITEMS.index),
        "dense d31 shed order must differ from Item order",
    )


def require_coverage(coverage: dict[str, Any]) -> None:
    counts = coverage["non_synthetic"]
    minimums = {
        name: 8 if name.startswith("tile_") else 4
        for name in COVERAGE_TAGS
        if name.startswith(("tile_", "crop_", "animal_"))
    }
    minimums.update(
        fert_current=4,
        fert_expired=4,
        unwatered=4,
        unfed=4,
        actor_gt16_states=4,
        reordered_inventories=8,
        shops_ge4_states=8,
        both_hires_nonzero_states=8,
    )
    for name, minimum in minimums.items():
        require(
            counts[name] >= minimum,
            f"R1 coverage {name}: actual {counts[name]}, required {minimum}; "
            "final corpus publication blocked",
        )
    require(
        counts["reordered_sheds"] >= 1 or coverage["shed_order_exception"] is True,
        "R1 reordered sheds needs explicit dense d=31 exception",
    )


def _check_identity(value: Any) -> None:
    identity = obj(
        value,
        (
            "root_commit",
            "dirty_files",
            "engine",
            "reference_commit",
            "feature_sha256",
            "recorder_sha256",
            "driver_sha256",
            "producer_sha256",
            "versions",
            "argv",
            "official_fixtures",
        ),
        "source identity",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", identity["root_commit"]) is not None,
        "root commit",
    )
    require(
        identity["reference_commit"] == PIN
        and identity["feature_sha256"] == FEATURE_SHA,
        "reference source pin mismatch",
    )
    for key in ("recorder_sha256", "driver_sha256", "producer_sha256"):
        digest(identity[key], key)
    obj(identity["engine"], ("trim_manifest", "lib", "lock"), "engine hashes")
    for value in identity["engine"].values():
        digest(value, "engine hash")
    require(isinstance(identity["dirty_files"], dict), "dirty files map")
    for path, value in identity["dirty_files"].items():
        require(
            not Path(path).is_absolute()
            and all(s not in ("", ".", "..", ".git") for s in path.split("/")),
            "unsafe source path",
        )
        digest(value, path)
    obj(identity["versions"], ("rustc", "cargo", "python"), "versions")
    require(
        all(isinstance(v, str) and v for v in identity["versions"].values()),
        "tool versions",
    )
    require(
        isinstance(identity["argv"], list)
        and bool(identity["argv"])
        and all(
            isinstance(a, list) and a and all(isinstance(v, str) for v in a)
            for a in identity["argv"]
        ),
        "exact argv arrays",
    )
    obj(identity["official_fixtures"], map(str, EPISODES), "official hashes")
    for value in identity["official_fixtures"].values():
        digest(value, "official fixture hash")


def _check_recipe(
    manifest: dict[str, Any], sources: list[dict[str, Any]], expected_records: int
) -> None:
    require(isinstance(manifest["profiles"], list), "profiles list required")
    for profile in manifest["profiles"]:
        require(
            isinstance(profile, list) and len(profile) == 10,
            "ten profile values required",
        )
        for value in profile[:9]:
            integer(value, "profile integer")
        require(type(profile[9]) in (float, int), "profile weed must be a number")
    require(
        manifest["profiles"] == PROFILES
        and manifest["policy"] == "observation-corpus-v2",
        "profile/policy recipe changed",
    )
    require(isinstance(manifest["seed_runs"], list), "seed runs array")
    for run in manifest["seed_runs"]:
        obj(
            run,
            (
                "seed",
                "profile",
                "sampled_steps",
                "action_sha256",
                "final_snapshot_sha256",
                "terminal_step",
            ),
            "seed run",
        )
        p = integer(run["profile"], "profile")
        integer(run["seed"], "seed integer")
        integer(run["terminal_step"], "terminal step integer")
        require(isinstance(run["sampled_steps"], list), "sampled steps must be a list")
        for step in run["sampled_steps"]:
            integer(step, "sampled step integer")
        require(p < 6 and run["seed"] == 11001 + p, "seed run identity")
        require(
            run["sampled_steps"] == [k * 94 // 15 for k in range(16)]
            and run["terminal_step"] == 95,
            "sample/terminal steps",
        )
        digest(run["action_sha256"], "action hash")
        digest(run["final_snapshot_sha256"], "final snapshot hash")
    if expected_records != 512:
        return
    require(
        [r["profile"] for r in manifest["seed_runs"]] == list(range(6)),
        "all six seed runs required",
    )
    expected: list[dict[str, Any]] = [
        {"kind": "official", "episode": ep, "step": step}
        for ep in EPISODES
        for step in OFFICIAL_STEPS
    ]
    expected += [
        {
            "kind": "seeded",
            "seed": r["seed"],
            "profile": r["profile"],
            "step": step,
            "policy_sha256": r["action_sha256"],
        }
        for r in manifest["seed_runs"]
        for step in r["sampled_steps"]
    ]
    expected += [{"kind": "dense", "case": d} for d in range(32)]
    require(
        sources == expected,
        "record source order/selection differs from 512-state recipe",
    )
    require_coverage(manifest["coverage"])


def _file_info(path: Path, raw: Path) -> dict[str, Any]:
    return {
        "compressed_size": path.stat().st_size,
        "expanded_size": raw.stat().st_size,
        "compressed_sha256": hash_file(path),
        "expanded_sha256": hash_file(raw),
    }


def compress_file(source: Path, target: Path) -> None:
    with (
        source.open("rb") as src,
        target.open("wb") as dst,
        gzip.GzipFile(
            filename="", fileobj=dst, mode="wb", mtime=0, compresslevel=9
        ) as zipped,
    ):
        shutil.copyfileobj(src, zipped, 1024 * 1024)


def byteplanes(source: Path, target: Path, *, decode: bool) -> None:
    size = source.stat().st_size
    require(size % 4 == 0, "unaligned reference bytes")
    values = size // 4
    with source.open("rb") as src, target.open("w+b") as dst:
        dst.truncate(size)
        for start in range(0, values, 262144):
            length = min(262144, values - start)
            if decode:
                raw = bytearray(length * 4)
                for plane in range(4):
                    src.seek(plane * values + start)
                    raw[plane::4] = src.read(length)
                dst.write(raw)
            else:
                block = src.read(length * 4)
                for plane in range(4):
                    dst.seek(plane * values + start)
                    dst.write(block[plane::4])


def _expand(path: Path, target: Path, info: dict[str, Any]) -> None:
    obj(
        info,
        ("compressed_size", "expanded_size", "compressed_sha256", "expanded_sha256"),
        str(path),
    )
    for key in ("compressed_size", "expanded_size"):
        integer(info[key], key)
    for key in ("compressed_sha256", "expanded_sha256"):
        digest(info[key], key)
    require(
        path.stat().st_size == info["compressed_size"]
        and hash_file(path) == info["compressed_sha256"],
        f"{path.name}: compressed size/hash mismatch",
    )
    total = 0
    try:
        with gzip.open(path, "rb") as src, target.open("wb") as dst:
            while block := src.read(1024 * 1024):
                total += len(block)
                require(
                    total <= MAX_EXPANDED and total <= info["expanded_size"],
                    "expanded byte budget/size exceeded",
                )
                dst.write(block)
    except (OSError, EOFError) as error:
        raise ValueError(f"{path.name}: invalid gzip: {error}") from error
    require(
        total == info["expanded_size"] and hash_file(target) == info["expanded_sha256"],
        f"{path.name}: expanded size/hash mismatch",
    )


def validate_corpus(directory: Path, *, expected_records: int = 512) -> None:
    """Validate all declared bytes; tiny-test scope never bypasses final512 quotas."""
    require(
        (directory / "manifest.json").is_file(),
        "qualified observation corpus missing: "
        f"{directory / 'manifest.json'} not found; regenerate it with `uv run python "
        "scripts/kaggriculture_observation_oracle/regenerate.py --output <dir>`",
    )
    manifest = obj(
        parse((directory / "manifest.json").read_bytes()),
        (
            "format",
            "schema_version",
            "observation_schema",
            "contract_version",
            "source_identity",
            "profiles",
            "policy",
            "seed_runs",
            "source_counts",
            "records",
            "files",
            "reference_shape",
            "dtype",
            "coverage",
        ),
        "manifest",
    )
    require(manifest["format"] in (FORMAT, SHUFFLED_FORMAT), "unsupported format")
    require(
        all(
            type(manifest[key]) is int
            for key in ("schema_version", "observation_schema", "contract_version")
        )
        and manifest["schema_version"] == 1
        and manifest["observation_schema"] == 3
        and manifest["contract_version"] == 4,
        "unsupported schema/contract version",
    )
    require(
        isinstance(manifest["reference_shape"], list)
        and all(type(dim) is int for dim in manifest["reference_shape"])
        and manifest["reference_shape"] == [expected_records, 2, FEATURES]
        and manifest["dtype"] == "<f4",
        "reference shape/dtype mismatch",
    )
    _check_identity(manifest["source_identity"])
    files = obj(manifest["files"], ("states.jsonl.gz", "reference.f32le.gz"), "files")
    require(
        sum(integer(f["compressed_size"], "compressed size") for f in files.values())
        <= MAX_COMPRESSED
        and sum(integer(f["expanded_size"], "expanded size") for f in files.values())
        <= MAX_EXPANDED,
        "fixture storage budget exceeded",
    )
    require(
        isinstance(manifest["records"], list)
        and len(manifest["records"]) == expected_records,
        "record metadata count mismatch",
    )
    counts = obj(
        manifest["source_counts"], ("official", "seeded", "dense"), "source counts"
    )
    for value in counts.values():
        integer(value, "source count")
    coverage = obj(
        manifest["coverage"],
        ("non_synthetic", "dense", "shed_order_exception"),
        "coverage",
    )
    for scope in ("non_synthetic", "dense"):
        obj(coverage[scope], COVERAGE_TAGS, "coverage tags")
        for value in coverage[scope].values():
            integer(value, "coverage count")
    require(type(coverage["shed_order_exception"]) is bool, "shed exception flag")
    with tempfile.TemporaryDirectory(prefix="obs-check-") as tmp:
        work = Path(tmp)
        for name, info in files.items():
            _expand(directory / name, work / name, info)
        feature_path = work / "reference.f32le.gz"
        if manifest["format"] == SHUFFLED_FORMAT:
            byteplanes(feature_path, work / "reference.raw", decode=True)
            feature_path = work / "reference.raw"
        require(
            feature_path.stat().st_size == expected_records * 2 * SEAT_BYTES,
            "reference byte length mismatch",
        )
        found_ids: set[str] = set()
        actual_counts = dict.fromkeys(counts, 0)
        actual_coverage = empty_coverage()
        sources = []
        dense30 = None
        dense_pair_checked = False
        with (
            (work / "states.jsonl.gz").open("rb") as states,
            feature_path.open("rb") as features,
        ):
            n = 0
            for row, raw in iter_rows(states):
                require(n < expected_records, "extra state record")
                meta = obj(
                    manifest["records"][n],
                    ("record_id", "source", "header_sha256", "seat_sha256"),
                    "record metadata",
                )
                require(row["record_id"] not in found_ids, "duplicate record id")
                found_ids.add(row["record_id"])
                require(
                    meta["record_id"] == row["record_id"]
                    and meta["source"] == row["source"],
                    "record identity/source mismatch",
                )
                require(
                    digest(meta["header_sha256"], "header hash")
                    == sha(header_bytes(raw)),
                    "header bytes/key-order hash mismatch",
                )
                require(
                    isinstance(meta["seat_sha256"], list)
                    and len(meta["seat_sha256"]) == 2,
                    "two seat block hashes required",
                )
                for seat in range(2):
                    block = features.read(SEAT_BYTES)
                    require(
                        len(block) == SEAT_BYTES
                        and sha(block)
                        == digest(meta["seat_sha256"][seat], "seat hash"),
                        f"record {row['record_id']} seat {seat}: "
                        "feature block hash mismatch",
                    )
                actual_counts[row["source"]["kind"]] += 1
                add_coverage(actual_coverage, row)
                if row["source"] == {"kind": "dense", "case": 30}:
                    dense30 = row["header"]
                elif row["source"] == {"kind": "dense", "case": 31}:
                    require(dense30 is not None, "dense d31 requires preceding d30")
                    assert dense30 is not None
                    validate_dense_pair(dense30, row["header"])
                    dense_pair_checked = True
                sources.append(row["source"])
                n += 1
            require(
                n == expected_records and not features.read(1),
                "missing/extra records or feature bytes",
            )
        actual_coverage["shed_order_exception"] = (
            actual_coverage["non_synthetic"]["reordered_sheds"] == 0
            and actual_coverage["dense"]["reordered_sheds"] > 0
        )
        require(
            counts == actual_counts and coverage == actual_coverage,
            "declared source/coverage counts disagree with states",
        )
        if expected_records == 512:
            require(
                dense_pair_checked, "full corpus requires verified dense d30/d31 pair"
            )
        _check_recipe(manifest, sources, expected_records)


def group_rss(group: int) -> int:
    """Sample the complete cargo/test process group on Mac or Linux."""
    total = 0
    if sys.platform == "darwin":
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        pids = (ctypes.c_int * 4096)()
        count = libproc.proc_listpgrppids(group, pids, ctypes.sizeof(pids))
        # proc_listpgrppids returns the number of PIDs (not proc_listpids' bytes).
        require(0 <= count < len(pids), "process-group inventory unavailable")
        for pid in pids[:count]:
            info = ctypes.create_string_buffer(96)
            if libproc.proc_pidinfo(pid, 4, 0, info, 96) == 96:
                total += (ctypes.c_uint64 * 2).from_buffer(info)[1]
    elif sys.platform == "linux":
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) == group:
                    total += int((path / "statm").read_text().split()[1]) * os.sysconf(
                        "SC_PAGE_SIZE"
                    )
            except (FileNotFoundError, ProcessLookupError):
                continue
    else:
        raise RuntimeError(
            "bounded oracle execution requires Mac libproc or Linux /proc"
        )
    return total


def run(
    argv: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> bytes:
    """Hard wall/process-group cleanup; sampled aggregate RSS stays below 1 GB.

    The budget charges the child group plus caller-group growth after launch.
    Memory the caller already held (for example pytest with torch loaded under
    `just prepare`) is the caller's, not this command's.
    """
    if time.monotonic() >= RUN_DEADLINE:
        raise RuntimeError("shared oracle execution deadline expired before launch")
    caller_baseline = group_rss(os.getpgrp())
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        start = time.monotonic()
        peak = 0
        reason = None
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            while process.poll() is None:
                # Include the driver/caller group as well as this new child group.
                caller_growth = max(0, group_rss(os.getpgrp()) - caller_baseline)
                peak = max(peak, group_rss(process.pid) + caller_growth)
                now = time.monotonic()
                if peak >= 1_000_000_000 or now - start >= 120 or now >= RUN_DEADLINE:
                    reason = (
                        "aggregate RSS"
                        if peak >= 1_000_000_000
                        else "shared execution deadline"
                        if now >= RUN_DEADLINE
                        else "120-second wall limit"
                    )
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.05)
            code = process.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        stdout.seek(0)
        result = stdout.read()
        stderr.seek(0)
        errors = stderr.read()
        COMMAND_RECEIPTS.append(
            {
                "argv": argv,
                "exit_status": code,
                "wall_seconds": time.monotonic() - start,
                "sampled_child_group_and_caller_growth_peak_rss_bytes": peak,
                "stop_reason": reason,
            }
        )
        if code != 0 or reason is not None:
            raise RuntimeError(
                f"bounded offline command failed: {argv}; exit={code}; "
                f"reason={reason}; peak_rss={peak}; "
                f"stdout={result!r}; stderr={errors!r}"
            )
        return result


def export_reference(directory: Path) -> dict[str, str]:
    """Export all reference engine files and verify each byte against git show."""
    archive = run(["git", "archive", PIN, "engine_rs"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            require(
                (member.name == "engine_rs" or member.name.startswith("engine_rs/"))
                and not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts,
                "unsafe reference archive entry",
            )
            target = directory / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                require(member.isfile(), "reference export requires regular files")
                source = tar.extractfile(member)
                assert source is not None
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as dst:
                    shutil.copyfileobj(source, dst)
    paths = (
        run(["git", "ls-tree", "-r", "--name-only", PIN, "engine_rs"])
        .decode()
        .splitlines()
    )
    hashes = {}
    for path in paths:
        expected = sha(run(["git", "show", f"{PIN}:{path}"]))
        require(
            hash_file(directory / path) == expected,
            f"reference export bytes changed: {path}",
        )
        hashes[path] = expected
    require(
        hashes["engine_rs/src/myolie_features.rs"] == FEATURE_SHA,
        "reference feature source hash changed",
    )
    return hashes


def validate_inputs(path: Path, generation: dict[str, Any]) -> None:
    """Recount the producer's actual stream before spending on a reference build."""
    coverage = empty_coverage()
    sources = []
    ids: set[str] = set()
    dense30 = None
    pair_checked = False
    with path.open("rb") as stream:
        for row, _raw in iter_rows(stream):
            require(row["record_id"] not in ids, "duplicate producer record id")
            ids.add(row["record_id"])
            sources.append(row["source"])
            add_coverage(coverage, row)
            if row["source"] == {"kind": "dense", "case": 30}:
                dense30 = row["header"]
            elif row["source"] == {"kind": "dense", "case": 31}:
                require(dense30 is not None, "dense d31 requires preceding d30")
                assert dense30 is not None
                validate_dense_pair(dense30, row["header"])
                pair_checked = True
    coverage["shed_order_exception"] = (
        coverage["non_synthetic"]["reordered_sheds"] == 0
        and coverage["dense"]["reordered_sheds"] > 0
    )
    require(len(ids) == generation["records"] == 512, "producer record count mismatch")
    require(pair_checked, "producer dense pair missing")
    require(coverage == generation["coverage"], "producer coverage report mismatch")
    _check_recipe(
        {
            "profiles": PROFILES,
            "policy": "observation-corpus-v2",
            "seed_runs": generation["seed_runs"],
            "coverage": coverage,
        },
        sources,
        512,
    )


SOURCE_PATHS = (
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
)
ENGINE_PATHS = (
    ("trim_manifest", "engine_rs/TRIM_MANIFEST.json"),
    ("lib", "engine_rs/src/lib.rs"),
    ("lock", "engine_rs/Cargo.lock"),
)
SourceSnapshot = tuple[str, tuple[tuple[str, str], ...]]


def source_snapshot() -> SourceSnapshot:
    """Pin immutable source identity before either executable is built or run."""
    commit = run(["git", "rev-parse", "HEAD"]).decode().strip()
    paths = (
        *SOURCE_PATHS,
        *(path for _key, path in ENGINE_PATHS),
        *(f"engine_rs/fixtures/episode-{ep}.jsonl.gz" for ep in EPISODES),
    )
    return commit, tuple((path, hash_file(ROOT / path)) for path in paths)


def check_source_snapshot(snapshot: SourceSnapshot) -> None:
    commit, files = snapshot
    require(
        run(["git", "rev-parse", "HEAD"]).decode().strip() == commit,
        "source identity changed: root commit",
    )
    for path, expected in files:
        require(
            (ROOT / path).is_file() and hash_file(ROOT / path) == expected,
            f"source bytes changed during regeneration: {path}",
        )


def regenerate(output: Path, reference: str) -> None:
    require(reference == PIN, "only the reviewed reference commit is supported")
    require(
        not output.exists(),
        "output directory already exists; refusing fixture replacement",
    )
    for name, value in {
        "CARGO_BUILD_JOBS": "2",
        "CARGO_NET_OFFLINE": "true",
        "UV_OFFLINE": "true",
        "RAYON_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RUST_TEST_THREADS": "1",
    }.items():
        require(
            os.environ.get(name) == value,
            f"required bounded environment: {name}={value}",
        )
    snapshot = source_snapshot()
    source_commit, source_files = snapshot
    source_hashes = dict(source_files)
    check_source_snapshot(snapshot)
    with tempfile.TemporaryDirectory(prefix="obs-oracle-") as tmp:
        work = Path(tmp)
        inputs = work / "inputs"
        inputs.mkdir()
        argv = [
            [
                "cargo",
                "test",
                "--locked",
                "--offline",
                "--lib",
                "kaggriculture::oracle_corpus::generate_observation_oracle_inputs",
                "--",
                "--exact",
                "--ignored",
                "--nocapture",
            ]
        ]
        env = os.environ | {"KG_OBS_ORACLE_OUT": str(inputs)}
        # The producer fails R1 before any export/build; retain its actual report.
        try:
            run(argv[0], env=env)
        except RuntimeError:
            report = inputs / "generation.json"
            if report.exists():
                print(report.read_text(), file=sys.stderr)
            raise
        check_source_snapshot(snapshot)
        generation = parse((inputs / "generation.json").read_bytes())
        require(generation["records"] == 512, "producer record count")
        validate_inputs(inputs / "states.jsonl", generation)
        exported = work / "reference"
        exported.mkdir()
        original_hashes = export_reference(exported)
        check_source_snapshot(snapshot)
        example = exported / "engine_rs/examples/observation_v3_oracle.rs"
        example.parent.mkdir(exist_ok=True)
        require(not example.exists(), "recorder would replace reference source")
        recorder = ROOT / "scripts/kaggriculture_observation_oracle/record.rs"
        shutil.copyfile(recorder, example)
        recorder_sha = source_hashes[
            "scripts/kaggriculture_observation_oracle/record.rs"
        ]
        require(
            hash_file(example) == recorder_sha,
            "recorder identity changed while copying pinned source",
        )
        raw_features = work / "reference.f32le"
        argv.append(
            [
                "cargo",
                "run",
                "--locked",
                "--offline",
                "--manifest-path",
                str(exported / "engine_rs/Cargo.toml"),
                "--example",
                "observation_v3_oracle",
                "--",
                str(inputs / "states.jsonl"),
                str(raw_features),
            ]
        )
        try:
            run(argv[-1])
        finally:
            require(
                example.is_file() and hash_file(example) == recorder_sha,
                "recorder bytes changed during reference execution",
            )
            for path, value in original_hashes.items():
                require(
                    hash_file(exported / path) == value,
                    f"reference existing bytes changed during build: {path}",
                )
        check_source_snapshot(snapshot)
        require(
            raw_features.stat().st_size == 33488896,
            "reference must contain exactly 33,488,896 bytes",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".observation-install-", dir=output.parent
        ) as staging:
            staged = Path(staging) / "fixture"
            staged.mkdir()
            compress_file(inputs / "states.jsonl", staged / "states.jsonl.gz")
            compress_file(raw_features, staged / "reference.f32le.gz")
            fmt = FORMAT
            stored_features = raw_features
            if sum(p.stat().st_size for p in staged.iterdir()) > MAX_COMPRESSED:
                stored_features = work / "reference.planes"
                byteplanes(raw_features, stored_features, decode=False)
                compress_file(stored_features, staged / "reference.f32le.gz")
                fmt = SHUFFLED_FORMAT
            identity = {
                "root_commit": source_commit,
                "dirty_files": {p: source_hashes[p] for p in SOURCE_PATHS},
                "engine": {key: source_hashes[path] for key, path in ENGINE_PATHS},
                "reference_commit": PIN,
                "feature_sha256": FEATURE_SHA,
                "recorder_sha256": recorder_sha,
                "driver_sha256": source_hashes[
                    "scripts/kaggriculture_observation_oracle/regenerate.py"
                ],
                "producer_sha256": source_hashes["src/kaggriculture/oracle_corpus.rs"],
                "versions": {
                    "rustc": run(["rustc", "--version"]).decode().strip(),
                    "cargo": run(["cargo", "--version"]).decode().strip(),
                    "python": sys.version,
                },
                "argv": [[sys.executable, *sys.argv], *argv],
                "official_fixtures": {
                    str(ep): source_hashes[f"engine_rs/fixtures/episode-{ep}.jsonl.gz"]
                    for ep in EPISODES
                },
            }
            records = []
            with (
                (inputs / "states.jsonl").open("rb") as state_stream,
                raw_features.open("rb") as features,
            ):
                for row, raw in iter_rows(state_stream):
                    records.append(
                        {
                            "record_id": row["record_id"],
                            "source": row["source"],
                            "header_sha256": sha(header_bytes(raw)),
                            "seat_sha256": [
                                sha(features.read(SEAT_BYTES)) for _ in range(2)
                            ],
                        }
                    )
            manifest = {
                "format": fmt,
                "schema_version": 1,
                "observation_schema": 3,
                "contract_version": 4,
                "source_identity": identity,
                "profiles": PROFILES,
                "policy": "observation-corpus-v2",
                "seed_runs": generation["seed_runs"],
                "source_counts": {"official": 384, "seeded": 96, "dense": 32},
                "records": records,
                "files": {
                    "states.jsonl.gz": _file_info(
                        staged / "states.jsonl.gz", inputs / "states.jsonl"
                    ),
                    "reference.f32le.gz": _file_info(
                        staged / "reference.f32le.gz", stored_features
                    ),
                },
                "reference_shape": [512, 2, FEATURES],
                "dtype": "<f4",
                "coverage": generation["coverage"],
            }
            (staged / "manifest.json").write_bytes(json_bytes(manifest) + b"\n")
            validate_corpus(staged)
            check_source_snapshot(snapshot)
            staged.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=PIN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate_corpus(args.output)
        print("observation oracle custody: OK (512 states, 1,024 seats)")
    else:
        regenerate(args.output, args.reference)


if __name__ == "__main__":
    main()
