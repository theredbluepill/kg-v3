"""Record native reference verdicts without importing the rebuilt grammar."""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import random
import struct
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

PIN = "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0"
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
NAMES = [
    "unit_actor",
    "unit_kind",
    "unit_target",
    "unit_item",
    "unit_quantity_present",
    "unit_quantity_high",
    "unit_quantity",
    "market_kind",
    "market_item",
    "market_quantity_high",
    "market_quantity",
    "stop",
]
WIDTHS = [241, 20, 128, 16, 2, 32, 32, 8, 16, 32, 32, 2]
SHAPES = [
    (1, 1, 241),
    (1, 10, 241),
    (2, 10, 241),
    (17, 10, 16),
    (231, 10, 241),
    (240, 10, 241),
    (241, 1, 241),
    (241, 10, 241),
]
EPISODES = [95324500, 95901360, 95921764, 95990191]
SOURCES = [
    "engine_rs/src/myolie_sampler.rs",
    "engine_rs/src/ffi.rs",
    "python/owl/kaggriculture/actor_codec.py",
    "python/owl/kaggriculture/gpu_sampling_grammar.py",
]
PINS = {
    "engine_rs/src/myolie_sampler.rs": (
        "563793903e43911b372a90b379d617ffdbab505d87af195694c380e75b57ac29"
    ),
    "engine_rs/src/ffi.rs": (
        "84c19507b1945106cbffbc878bb0324ed71f581065c1e1b90c4360fe47df4369"
    ),
    "python/owl/kaggriculture/actor_codec.py": (
        "88ed3d44a7a0dc1cbc386035d323c34b17a3d14a2d4a520ad30240345140e5d6"
    ),
}
TABLE_SHAPES = {
    "unit_kind": (20,),
    "unit_item": (20, 16),
    "unit_quantity_present": (20, 2),
    "unit_quantity_high": (2, 32),
    "unit_quantity_low": (2, 2, 32),
    "market_kind": (8,),
    "market_item": (8, 16),
    "market_quantity": (8, 32),
}
CATEGORIES = [
    "unknown market command",
    "unsupported unit command arguments",
    "market item syntax",
    "extra hand commands",
    "market queue limit",
    "quantity range",
    "hire capacity",
]
COMMON = {
    "type",
    "id",
    "source",
    "shape",
    "tokens",
    "length",
    "padding_edits",
    "sampler",
    "training_decoder",
    "python_codec",
    "v4_expected",
}
EXTRAS = {
    "sampled": {"seed", "selection_mode"},
    "replay": {"episode", "from_step", "seat", "raw_action"},
    "mutation": {"category"},
    "replay_rejected": {"episode", "from_step", "seat", "raw_action", "category"},
}
MANIFEST_KEYS = {
    "schema_version",
    "reference_commit",
    "source_sha256",
    "trace_sha256",
    "recorder_sha256",
    "harness_sha256",
    "append_patch_sha256",
    "cargo_lock_sha256",
    "toolchain_sha256",
    "commands",
    "tools",
    "counts",
    "replay_scan",
    "disagreements",
    "missing_rejection_categories",
    "compressed_sha256",
    "uncompressed_sha256",
    "compressed_bytes",
    "uncompressed_bytes",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{PIN}:{path}"], cwd=ROOT)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
        for row in rows
    )


def deterministic_gzip(payload: bytes) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=target, mtime=0, compresslevel=9
    ) as stream:
        stream.write(payload)
    return target.getvalue()


def output_metadata(payload: bytes) -> dict[str, Any]:
    compressed = deterministic_gzip(payload)
    return {
        "compressed_sha256": sha(compressed),
        "uncompressed_sha256": sha(payload),
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(payload),
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    accepted = [
        r
        for r in rows
        if r["source"] in ("sampled", "replay") and r["v4_expected"]["accepted"]
    ]
    return {
        "accepted": len(accepted),
        "sampled": sum(r["source"] == "sampled" for r in accepted),
        "replay": sum(r["source"] == "replay" for r in accepted),
        "dense": sum(r["shape"]["actors"] == 241 for r in accepted),
        "dense_full_market": sum(
            r["shape"]["actors"] == 241
            and r["shape"]["order_limit"] == 10
            and r["length"] == 252
            for r in accepted
        ),
        "full_market": sum(
            r["length"] == r["shape"]["actors"] + r["shape"]["order_limit"] + 1
            for r in accepted
        ),
        "mutations": sum(r["source"] == "mutation" for r in rows),
        "replay_rejected": sum(r["source"] == "replay_rejected" for r in rows),
    }


def fixture_manifest(payload: bytes) -> dict[str, Any]:
    rows = [
        json.loads(line, object_pairs_hook=strict_object)
        for line in payload.splitlines()
    ][1:]
    manifest = {
        "schema_version": 1,
        "reference_commit": PIN,
        "source_sha256": {p: sha(git_bytes(p)) for p in SOURCES},
        "trace_sha256": {
            str(e): sha(
                (ROOT / f"engine_rs/fixtures/episode-{e}.jsonl.gz").read_bytes()
            )
            for e in EPISODES
        },
        "recorder_sha256": sha(Path(__file__).read_bytes()),
        "harness_sha256": sha((HERE / "reference_harness.rs").read_bytes()),
        "append_patch_sha256": sha((HERE / "scratch/append.patch").read_bytes())
        if (HERE / "scratch/append.patch").exists()
        else sha(b""),
        "cargo_lock_sha256": sha(git_bytes("engine_rs/Cargo.lock")),
        "toolchain_sha256": sha(git_bytes("rust-toolchain.toml")),
        "commands": [],
        "tools": {},
        "counts": counts(rows),
        "replay_scan": {},
        "disagreements": [],
        "missing_rejection_categories": CATEGORIES.copy(),
    }
    manifest.update(output_metadata(payload))
    return manifest


def exact(value: Any, keys: Any, label: str) -> None:
    require(isinstance(value, Mapping) and set(value) == set(keys), f"{label} keys")


def boolean_array(value: Any, shape: tuple[int, ...]) -> None:
    if not shape:
        require(type(value) is bool, "table bit is not boolean")
        return
    require(isinstance(value, list) and len(value) == shape[0], "table dimensions")
    for part in value:
        boolean_array(part, shape[1:])


def oracle_result(value: Any) -> None:
    if isinstance(value, dict) and "not_applicable" in value:
        exact(value, {"not_applicable", "reason"}, "oracle result")
        require(
            value["not_applicable"] is True
            and isinstance(value["reason"], str)
            and bool(value["reason"]),
            "oracle applicability reason",
        )
    else:
        exact(value, {"accepted", "action", "error"}, "oracle result")
        require(type(value["accepted"]) is bool, "oracle acceptance bool")
        require(
            (isinstance(value["action"], dict) and value["error"] is None)
            if value["accepted"]
            else (
                value["action"] is None
                and isinstance(value["error"], str)
                and bool(value["error"])
            ),
            "oracle result contents",
        )


def validate_fixture(payload: bytes, manifest: Mapping[str, object]) -> None:
    exact(manifest, MANIFEST_KEYS, "manifest")
    require(len(payload) <= 4 * 1024 * 1024, "uncompressed size overflow")
    require(len(deterministic_gzip(payload)) <= 512 * 1024, "compressed size overflow")
    require(len(json.dumps(manifest).encode()) <= 64 * 1024, "manifest size overflow")
    require(
        manifest["schema_version"] == 1 and manifest["reference_commit"] == PIN,
        "manifest provenance version",
    )
    require(
        manifest["source_sha256"] == {p: sha(git_bytes(p)) for p in SOURCES},
        "stale source hash",
    )
    for p, pinned in PINS.items():
        require(
            manifest["source_sha256"][p] == pinned, "source differs from reviewed pin"
        )
    require(
        manifest["recorder_sha256"] == sha(Path(__file__).read_bytes()),
        "stale recorder hash",
    )
    require(
        manifest["harness_sha256"] == sha((HERE / "reference_harness.rs").read_bytes()),
        "stale harness hash",
    )
    require(
        manifest["trace_sha256"]
        == {
            str(e): sha(
                (ROOT / f"engine_rs/fixtures/episode-{e}.jsonl.gz").read_bytes()
            )
            for e in EPISODES
        },
        "stale trace hash",
    )
    require(
        manifest["cargo_lock_sha256"] == sha(git_bytes("engine_rs/Cargo.lock"))
        and manifest["toolchain_sha256"] == sha(git_bytes("rust-toolchain.toml")),
        "stale build provenance",
    )
    patch_path = HERE / "scratch/append.patch"
    expected_patch = sha(patch_path.read_bytes()) if patch_path.exists() else sha(b"")
    require(
        manifest["append_patch_sha256"] == expected_patch, "stale append patch hash"
    )
    for key, value in output_metadata(payload).items():
        require(manifest[key] == value, f"fixture {key} mismatch")
    try:
        rows = [
            json.loads(line, object_pairs_hook=strict_object)
            for line in payload.splitlines()
        ]
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid JSONL") from error
    require(bool(rows), "empty fixture")
    header, *programs = rows
    exact(
        header,
        {
            "type",
            "schema_version",
            "reference_commit",
            "slot_names",
            "slot_widths",
            "tables",
        },
        "header",
    )
    require(
        header["type"] == "header"
        and header["schema_version"] == 1
        and header["reference_commit"] == PIN,
        "header provenance",
    )
    require(
        header["slot_names"] == NAMES and header["slot_widths"] == WIDTHS, "slot ABI"
    )
    exact(header["tables"], TABLE_SHAPES, "tables")
    for name, shape in TABLE_SHAPES.items():
        boolean_array(header["tables"][name], shape)
    ids = set()
    scheduled = set()
    strata = Counter()
    for row in programs:
        require(isinstance(row, dict) and row.get("source") in EXTRAS, "program source")
        source = row["source"]
        exact(row, COMMON | EXTRAS[source], "program")
        require(
            row["type"] == "program"
            and isinstance(row["id"], str)
            and row["id"] not in ids,
            "duplicate or invalid id",
        )
        ids.add(row["id"])
        exact(row["shape"], {"actors", "order_limit", "hire_limit"}, "shape")
        a, o, h = (row["shape"][k] for k in ("actors", "order_limit", "hire_limit"))
        require(
            all(type(v) is int for v in (a, o, h))
            and 1 <= a <= 241
            and 1 <= o <= 10
            and 1 <= h <= 241,
            "shape bounds",
        )
        require(
            type(row["length"]) is int and -1 <= row["length"] <= 253,
            "length type/bounds",
        )
        require(
            isinstance(row["tokens"], list)
            and len(row["tokens"]) <= 3024
            and all(type(v) is int and -32768 <= v <= 32767 for v in row["tokens"]),
            "reference token domain",
        )
        require(isinstance(row["padding_edits"], list), "padding edits")
        for edit in row["padding_edits"]:
            require(
                isinstance(edit, list)
                and len(edit) == 3
                and all(type(v) is int for v in edit),
                "padding edit schema",
            )
            f, s, v = edit
            require(
                max(0, row["length"]) <= f < 252
                and 0 <= s < 12
                and -32768 <= v <= 32767,
                "padding bounds",
            )
        for key in ("sampler", "training_decoder", "python_codec"):
            oracle_result(row[key])
        exact(row["v4_expected"], {"accepted", "action", "reason"}, "v4_expected")
        v4 = row["v4_expected"]
        require(
            type(v4["accepted"]) is bool
            and isinstance(v4["reason"], str)
            and bool(v4["reason"]),
            "v4 verdict",
        )
        if v4["accepted"]:
            require(
                len(row["tokens"]) == row["length"] * 12
                and a + 1 <= row["length"] <= a + o + 1,
                "accepted length",
            )
            require(
                all(v == 0 for _, _, v in row["padding_edits"]),
                "accepted nonzero padding",
            )
            for key in ("sampler", "training_decoder", "python_codec"):
                require(
                    row[key].get("accepted") is True
                    and row[key]["action"] == v4["action"],
                    "accepted expected JSON differs",
                )
        else:
            require(v4["action"] is None, "rejected v4 action")
            if row["sampler"].get("accepted"):
                require(
                    source == "mutation" and row["category"].startswith("padding"),
                    "unclassified sampler divergence",
                )
        if source == "sampled":
            require(v4["accepted"], "scheduled sample rejected")
            seed = row["seed"]
            require(
                type(seed) is int and 2026092900 <= seed < 2026093156, "sample seed"
            )
            j, i = divmod(seed - 2026092900, 32)
            require((a, o, h) == SHAPES[j] and seed not in scheduled, "sample schedule")
            scheduled.add(seed)
            require(
                row["selection_mode"] == ("full_market" if i % 2 else "uniform"),
                "selection mode",
            )
            if i % 2:
                require(row["length"] == a + o + 1, "full-market quota")
        if source in ("replay", "replay_rejected"):
            require(
                row["episode"] in EPISODES
                and type(row["from_step"]) is int
                and 0 <= row["from_step"] <= 718
                and row["seat"] in (0, 1),
                "replay identity",
            )
            if source == "replay":
                require(
                    v4["accepted"] and row["raw_action"] == v4["action"],
                    "raw replay identity",
                )
                strata[(row["episode"], row["from_step"] // 180)] += 1
            else:
                require(
                    not v4["accepted"] and row["python_codec"].get("accepted") is False,
                    "replay rejection",
                )
        if source in ("mutation", "replay_rejected"):
            require(
                isinstance(row["category"], str) and bool(row["category"]),
                "negative category",
            )
    exact(manifest["counts"], counts(programs), "manifest counts")
    require(isinstance(manifest["commands"], list), "manifest commands")
    for command in manifest["commands"]:
        require(
            isinstance(command, list) and all(isinstance(v, str) for v in command),
            "manifest command values",
        )
    require(
        isinstance(manifest["tools"], dict)
        and all(
            isinstance(k, str) and isinstance(v, str) and v
            for k, v in manifest["tools"].items()
        ),
        "manifest tool versions",
    )
    require(isinstance(manifest["replay_scan"], dict), "manifest replay scan")
    for episode, scan in manifest["replay_scan"].items():
        require(episode in {str(e) for e in EPISODES}, "manifest replay episode")
        exact(
            scan,
            {
                "candidates",
                "selected",
                "codec_rejections",
                "canonical_layout_exclusions",
            },
            "replay scan",
        )
        require(
            type(scan["candidates"]) is int and scan["candidates"] == 1438,
            "candidate denominator",
        )
        exact(scan["selected"], {"0", "1", "2", "3"}, "replay selected strata")
        require(
            all(type(v) is int and v == 4 for v in scan["selected"].values()),
            "replay selected quota",
        )
        for key in ("codec_rejections", "canonical_layout_exclusions"):
            require(
                isinstance(scan[key], dict)
                and all(
                    isinstance(k, str) and type(v) is int and v > 0
                    for k, v in scan[key].items()
                ),
                "replay rejection counts",
            )
    require(isinstance(manifest["disagreements"], list), "manifest disagreements")
    for difference in manifest["disagreements"]:
        exact(
            difference,
            {"id", "sampler", "training_decoder", "python_codec", "v4_expected"},
            "manifest disagreement",
        )
        require(difference["id"] in ids, "disagreement id")
        original = next(row for row in programs if row["id"] == difference["id"])
        require(
            all(difference[key] == original[key] for key in difference),
            "disagreement evidence mismatch",
        )
    expected_differences = {
        r["id"] for r in programs if r["source"] == "mutation" and verdicts_differ(r)
    }
    require(
        {r["id"] for r in manifest["disagreements"]} == expected_differences,
        "disagreement inventory mismatch",
    )
    require(
        manifest["missing_rejection_categories"]
        == [
            c
            for c in CATEGORIES
            if not any(
                r["source"] == "replay_rejected" and r["category"] == c
                for r in programs
            )
        ],
        "missing rejection category inventory",
    )
    actual = counts(programs)
    require(actual == manifest["counts"], "manifest count mismatch")
    require(
        actual["accepted"] == 320
        and actual["sampled"] == 256
        and actual["replay"] == 64,
        "accepted quotas",
    )
    require(
        actual["dense"] == 64
        and actual["dense_full_market"] >= 16
        and actual["mutations"] >= 40,
        "dense/negative quota",
    )
    require(
        len(scheduled) == 256
        and all(strata[e, s] == 4 for e in EPISODES for s in range(4)),
        "schedule/replay strata quota",
    )


class Plan:
    def __init__(self, values: list[int]) -> None:
        data = bytes(values)
        magic, abi, nodes, start, edges = struct.unpack_from("<5I", data)
        require((magic, abi, start) == (0x4D534731, 1, 0), "reference plan ABI")
        require(len(data) == 20 + 16 * nodes + 5 * edges, "reference plan size")
        self.rows = [struct.unpack_from("<4I", data, 20 + 16 * i) for i in range(nodes)]
        self.masks = data[20 + 16 * nodes : 20 + 16 * nodes + edges]
        self.next = struct.unpack_from(f"<{edges}i", data, 20 + 16 * nodes + edges)
        for slot, _, offset, width in self.rows:
            require(width == WIDTHS[slot] and offset + width <= edges, "plan row")

    def options(self, node: int) -> tuple[int, list[int], int]:
        slot, _, offset, width = self.rows[node]
        return slot, [i for i in range(width) if self.masks[offset + i]], offset

    def after(self, prefix: list[int]) -> list[bool]:
        node = 0
        for token in prefix:
            require(node >= 0, "prefix after STOP")
            _, choices, offset = self.options(node)
            require(token in choices, "reference table probe rejected")
            node = self.next[offset + token]
        require(node >= 0, "table prefix terminated")
        slot, choices, _ = self.options(node)
        return [i in choices for i in range(WIDTHS[slot])]

    def sample(self, seed: int, full: bool) -> list[int]:
        rng = random.Random(seed)
        node = 0
        tokens = []
        while node >= 0:
            slot, choices, offset = self.options(node)
            if slot == 7 and full and len(choices) > 1:
                choices = [c for c in choices if c != 0]
            token = rng.choice(choices)
            tokens.append(token)
            node = self.next[offset + token]
        require(node == -1, "sample ended at invalid edge")
        return tokens


def extract_tables(plan: Plan) -> dict[str, Any]:
    options = plan.after

    def first(mask: list[bool]) -> int:
        return mask.index(True)

    unit = options([0])
    item = [options([0, 1, 0]) for _ in range(20)]
    present = [options([0, 1, 0, 0]) for _ in range(20)]
    for kind, allowed in enumerate(unit):
        if allowed:
            item[kind] = options([0, kind, 0])
            present[kind] = options([0, kind, 0, first(item[kind])])
    high = [options([0, 6, 0, 1, p]) for p in range(2)]
    low = [
        [options([0, 6, 0, 1, p, 0 if zero or not p else 1]) for zero in range(2)]
        for p in range(2)
    ]
    prefix = [0, 1] + [0] * 17
    market = options(prefix)
    market_item = [options([*prefix, k]) for k in range(8)]
    quantities = [options([*prefix, k, first(market_item[k])]) for k in range(8)]
    for kind in range(8):
        for digit, allowed in enumerate(quantities[kind]):
            if allowed:
                require(
                    options([*prefix, kind, first(market_item[kind]), digit])
                    == quantities[kind],
                    "reference market digits differ",
                )
    return dict(
        zip(
            TABLE_SHAPES,
            [unit, item, present, high, low, market, market_item, quantities],
            strict=True,
        )
    )


def scratch_prepare(reference: Path) -> None:
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference)
        .decode()
        .strip()
        == PIN,
        "wrong scratch commit",
    )
    original = git_bytes("engine_rs/src/ffi.rs")
    append = (HERE / "reference_harness.rs").read_bytes()
    require(sha(original) == PINS["engine_rs/src/ffi.rs"], "wrong original FFI blob")
    status = (
        subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=reference)
        .decode()
        .splitlines()
    )
    require(
        set(status) <= {"engine_rs/src/ffi.rs"},
        "unauthorized scratch tracked difference",
    )
    for path in SOURCES:
        data = (reference / path).read_bytes()
        require(
            data in (original, original + append)
            if path == "engine_rs/src/ffi.rs"
            else data == git_bytes(path),
            f"scratch source differs: {path}",
        )
    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    before = scratch / "ffi.rs.before"
    if before.exists():
        require(before.read_bytes() == original, "preserved original differs")
    else:
        before.write_bytes(original)
    patch = "".join(
        difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            (original + append).decode().splitlines(keepends=True),
            fromfile="a/engine_rs/src/ffi.rs",
            tofile="b/engine_rs/src/ffi.rs",
        )
    ).encode()
    prior = scratch / "append.patch"
    if prior.exists():
        require(prior.read_bytes() == patch, "preserved append patch differs")
    else:
        prior.write_bytes(patch)
    if (reference / "engine_rs/src/ffi.rs").read_bytes() == original:
        require(not status, "initial scratch is not clean")
        (reference / "engine_rs/src/ffi.rs").write_bytes(original + append)
    for path in SOURCES:
        (scratch / (Path(path).name + ".original")).write_bytes(git_bytes(path))


def native(
    reference: Path, requests: list[dict[str, Any]], label: str
) -> tuple[list[dict[str, Any]], list[str]]:
    source = HERE / "scratch" / f"{label}.requests.jsonl"
    target = HERE / "scratch" / f"{label}.responses.jsonl"
    source.write_bytes(jsonl(requests))
    env = os.environ.copy()
    env.update(
        CARGO_BUILD_JOBS="2",
        CARGO_NET_OFFLINE="true",
        UV_OFFLINE="true",
        CARGO_TARGET_DIR=str(ROOT / ".codex-tmp/grammar-reference-build-1.2"),
        KG_GRAMMAR_ORACLE_INPUT=str(source),
        KG_GRAMMAR_ORACLE_OUTPUT=str(target),
    )
    cmd = [
        "cargo",
        "test",
        "--manifest-path",
        str(reference / "engine_rs/Cargo.toml"),
        "--offline",
        "--locked",
        "--lib",
        "ffi::task12_oracle::record_requests",
        "--",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    print("Native command:", json.dumps(cmd), flush=True)
    result = subprocess.run(cmd, env=env, cwd=ROOT, check=False)
    require(
        result.returncode == 0,
        f"native oracle exit {result.returncode}; "
        "inspect missing-crate/compile evidence; no network fetch permitted",
    )
    rows = [json.loads(line) for line in target.read_bytes().splitlines()]
    require(
        len(rows) == len(requests)
        and [r["id"] for r in rows] == [r["id"] for r in requests],
        "native response identity",
    )
    return rows, cmd


def shape_value(shape: tuple[int, ...]) -> dict[str, int]:
    return dict(zip(("actors", "order_limit", "hire_limit"), shape, strict=True))


def legal(shape: tuple[int, ...], seat: int = 0) -> dict[str, Any]:
    a, o, _ = shape
    return {
        "observation": {"player": seat, "farms": [{"hands": [None] * (a - 1)}] * 2},
        "configuration": {"maxMarketOrdersPerTurn": o},
    }


def accepted(action: Any) -> dict[str, Any]:
    return {"accepted": True, "action": action, "error": None}


def rejected(error: Any) -> dict[str, Any]:
    return {"accepted": False, "action": None, "error": str(error)}


def na(reason: str) -> dict[str, Any]:
    return {"not_applicable": True, "reason": reason}


def codec_decode(
    codec: ModuleType, shape: tuple[int, ...], tokens: list[int]
) -> dict[str, Any]:
    if len(tokens) % 12:
        return na("incomplete candidate tokens cannot form Python frame dictionaries")
    frames = [
        dict(zip(NAMES, tokens[i : i + 12], strict=True))
        for i in range(0, len(tokens), 12)
    ]
    encoded = {
        "schema": codec.SCHEMA,
        "layout": {"keys": ["farmer", "hands", "market"], "hands_count": shape[0] - 1},
        "frames": frames,
    }
    try:
        return accepted(codec.decode_action(encoded, legal(shape), hire_limit=shape[2]))
    except ValueError as error:
        return rejected(error)


def category(reason: str) -> str:
    mapping = {
        "commands exceed observed hand count": "extra hand commands",
        "market queue exceeds configured limit": "market queue limit",
        "quantity must be an integer in 0..1023": "quantity range",
        "target violates syntax mask: market_item": "market item syntax",
        "target violates syntax mask: market_kind": "hire capacity",
    }
    return mapping.get(reason, reason)


def base_row(
    identity: str, source: str, shape: tuple[int, ...], tokens: list[int], **extra: Any
) -> dict[str, Any]:
    return {
        "type": "program",
        "id": identity,
        "source": source,
        "shape": shape_value(shape),
        "tokens": tokens,
        "length": len(tokens) // 12,
        "padding_edits": [],
        **extra,
    }


def replay_rows(
    codec: ModuleType,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = []
    rejected_rows = []
    scan = {}
    for episode in EPISODES:
        with gzip.open(
            ROOT / f"engine_rs/fixtures/episode-{episode}.jsonl.gz", "rt"
        ) as stream:
            header = json.loads(next(stream))
            public = header["initial"]["public"]
            selected = Counter()
            reasons = Counter()
            canonical = Counter()
            seen = set()
            candidates = 0
            for line in stream:
                transition = json.loads(line)
                step = transition["from_step"]
                for seat, action in enumerate(transition["actions"]):
                    candidates += 1
                    shape = (
                        1 + len(public["farms"][seat]["hands"]),
                        header["configuration"]["maxMarketOrdersPerTurn"],
                        241,
                    )
                    identity = f"replay-{episode}-{step}-{seat}"
                    common = {
                        "episode": episode,
                        "from_step": step,
                        "seat": seat,
                        "raw_action": action,
                    }
                    try:
                        encoded = codec.encode_action(
                            action, legal(shape, seat), hire_limit=241
                        )
                        restored = codec.decode_action(
                            encoded, legal(shape, seat), hire_limit=241
                        )
                        require(restored == action, "reference codec altered replay")
                    except ValueError as error:
                        reason = str(error)
                        cat = category(reason)
                        reasons[cat] += 1
                        if cat not in seen:
                            seen.add(cat)
                            row = base_row(
                                identity,
                                "replay_rejected",
                                shape,
                                [],
                                category=cat,
                                **common,
                            )
                            row.update(
                                sampler=na(
                                    "reference codec rejected raw action; "
                                    "no canonical token program"
                                ),
                                training_decoder=na(
                                    "reference codec rejected raw action; "
                                    "no canonical token program"
                                ),
                                python_codec=rejected(reason),
                                v4_expected={
                                    "accepted": False,
                                    "action": None,
                                    "reason": cat,
                                },
                            )
                            rejected_rows.append(row)
                        continue
                    if not isinstance(action, dict) or set(action) != {
                        "farmer",
                        "hands",
                        "market",
                    }:
                        canonical["incomplete action keys"] += 1
                        continue
                    if len(action["hands"]) != shape[0] - 1:
                        canonical["short hand commands"] += 1
                        continue
                    stratum = step // 180
                    if selected[stratum] >= 4:
                        continue
                    tokens = [
                        frame[name] for frame in encoded["frames"] for name in NAMES
                    ]
                    row = base_row(identity, "replay", shape, tokens, **common)
                    row["python_codec"] = accepted(restored)
                    rows.append(row)
                    selected[stratum] += 1
                public = transition["expected"]
            require(
                [selected[i] for i in range(4)] == [4] * 4,
                f"replay quota {episode}: {selected}",
            )
            scan[str(episode)] = {
                "candidates": candidates,
                "selected": {str(k): v for k, v in sorted(selected.items())},
                "codec_rejections": dict(sorted(reasons.items())),
                "canonical_layout_exclusions": dict(sorted(canonical.items())),
            }
    print("Replay scan:", json.dumps(scan, sort_keys=True), flush=True)
    return rows, rejected_rows, scan


def mutation_rows() -> list[dict[str, Any]]:
    base = [0, 1] + [0] * 21 + [1]
    rows = []

    def add(
        name: str,
        tokens: list[int],
        shape: tuple[int, ...] = (1, 10, 241),
        padding: list[list[int]] | None = None,
    ) -> None:
        row = base_row("mutation-" + name, "mutation", shape, tokens, category=name)
        row["padding_edits"] = [] if padding is None else padding
        rows.append(row)

    for i, (index, value) in enumerate(
        [(0, 1), (1, 0), (1, 19), (1, -1), (2, 1), (3, 1), (4, 1), (11, 1), (23, 0)]
    ):
        tokens = base.copy()
        tokens[index] = value
        add(f"exact-{i}", tokens)
    for slot, width in enumerate(WIDTHS):
        for value in (-1, width):
            tokens = base.copy()
            tokens[slot] = value
            add(f"slot-{slot}-{value}", tokens)
    add("missing-stop", base[:-1])
    add("after-stop", base + [0] * 11 + [1])
    tokens = base.copy()
    tokens[1] = 6
    tokens[3] = 1
    tokens[4] = 1
    add("unit-explicit-zero", tokens)
    for a, o, h, hires in [
        (1, 10, 1, 1),
        (17, 10, 16, 1),
        (240, 10, 241, 2),
        (241, 10, 241, 1),
    ]:
        tokens = (
            [v for actor in range(a) for v in [actor, 1] + [0] * 10]
            + ([0] * 7 + [1] + [0] * 4) * hires
            + [0] * 11
            + [1]
        )
        add(f"hire-capacity-{a}-{h}", tokens, (a, o, h))
    for name, value in [
        ("nonzero", 1),
        ("out-of-width", 20),
        ("negative", -1),
        ("zero", 0),
    ]:
        add("padding-" + name, base.copy(), padding=[[2, 1, value]])
    return rows


def verdicts_differ(row: dict[str, Any]) -> bool:
    sampler = row["sampler"]
    for key in ("training_decoder", "python_codec", "v4_expected"):
        other = row[key]
        if sampler.get("accepted") != other.get("accepted"):
            return True
        if sampler.get("accepted") and sampler["action"] != other.get("action"):
            return True
    return False


def record(reference: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    scratch_prepare(reference)
    spec = importlib.util.spec_from_file_location(
        "task12_reference_codec", reference / "python/owl/kaggriculture/actor_codec.py"
    )
    require(spec is not None and spec.loader is not None, "codec file import")
    codec = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(codec)
    require(
        "torch" not in sys.modules and "owl" not in sys.modules,
        "oracle imported model/package",
    )
    requests = [
        {"op": "plan", "id": f"plan-{i}", "shape": shape_value(s)}
        for i, s in enumerate(SHAPES)
    ]
    plan_rows, plan_cmd = native(reference, requests, label + "-plans")
    plans = []
    for row in plan_rows:
        require(row["error"] is None, "native plan rejected shape")
        plans.append(Plan(row["bytes"]))
    rows = []
    for j, shape in enumerate(SHAPES):
        for i in range(32):
            seed = 2026092900 + 32 * j + i
            tokens = plans[j].sample(seed, bool(i % 2))
            rows.append(
                base_row(
                    f"sampled-{j}-{i}",
                    "sampled",
                    shape,
                    tokens,
                    seed=seed,
                    selection_mode="full_market" if i % 2 else "uniform",
                )
            )
    replay, replay_rejected, scan = replay_rows(codec)
    rows += replay + mutation_rows()
    requests = []
    for row in rows:
        shape = tuple(row["shape"].values())
        if "python_codec" not in row:
            row["python_codec"] = codec_decode(codec, shape, row["tokens"])
        requests.append(
            {
                "op": "decode",
                "id": row["id"],
                "shape": row["shape"],
                "seat": row.get("seat", 0),
                "tokens": row["tokens"],
                "length": row["length"],
                "padding_edits": row["padding_edits"],
            }
        )
    replies, decode_cmd = native(reference, requests, label + "-decode")
    for row, response in zip(rows, replies, strict=True):
        row.update(
            sampler=response["sampler"], training_decoder=response["training_decoder"]
        )
        sampler = row["sampler"]
        padding_bad = any(v != 0 for _, _, v in row["padding_edits"])
        ok = sampler["accepted"] and not padding_bad
        reason = (
            "padding must be zero (v4.1 question 2); "
            "reference verdict covers prefix only"
            if padding_bad
            else ("sampler canonical prefix" if ok else row["category"])
        )
        row["v4_expected"] = {
            "accepted": ok,
            "action": sampler["action"] if ok else None,
            "reason": reason,
        }
        if row["source"] in ("sampled", "replay"):
            require(
                ok
                and row["training_decoder"] == sampler
                and row["python_codec"] == sampler,
                f"accepted schedule decoder disagreement {row['id']}",
            )
    header = {
        "type": "header",
        "schema_version": 1,
        "reference_commit": PIN,
        "slot_names": NAMES,
        "slot_widths": WIDTHS,
        "tables": extract_tables(plans[0]),
    }
    rows += replay_rejected
    payload = jsonl([header, *rows])
    manifest = fixture_manifest(payload)
    manifest["commands"] = [plan_cmd, decode_cmd]
    manifest["tools"] = {
        name: subprocess.check_output(cmd, text=True).strip()
        for name, cmd in {
            "cargo": ["cargo", "--version"],
            "rustc": ["rustc", "--version"],
            "python": [sys.executable, "--version"],
        }.items()
    }
    manifest["replay_scan"] = scan
    manifest["missing_rejection_categories"] = [
        c for c in CATEGORIES if not any(r["category"] == c for r in replay_rejected)
    ]
    manifest["disagreements"] = [
        {
            "id": r["id"],
            "sampler": r["sampler"],
            "training_decoder": r["training_decoder"],
            "python_codec": r["python_codec"],
            "v4_expected": r["v4_expected"],
        }
        for r in rows
        if r["source"] == "mutation" and verdicts_differ(r)
    ]
    validate_fixture(payload, manifest)
    return payload, manifest


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["record", "verify"])
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload, manifest = record(args.reference_root.resolve(), args.mode)
    compressed = deterministic_gzip(payload)
    if args.mode == "record":
        require(args.output is not None, "record requires --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(compressed)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        require(
            sha(args.output.read_bytes()) == manifest["compressed_sha256"],
            "written fixture hash",
        )
    else:
        require(args.fixture is not None, "verify requires --fixture")
        previous = json.loads(args.manifest.read_text())
        validate_fixture(gzip.decompress(args.fixture.read_bytes()), previous)
        require(
            compressed == args.fixture.read_bytes(),
            "second native recording bytes differ",
        )
        require(
            manifest == previous, "second native recording provenance/counts differ"
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "counts": manifest["counts"],
                **output_metadata(payload),
                "missing_rejection_categories": manifest[
                    "missing_rejection_categories"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
