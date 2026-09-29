"""Record the actual pinned TrainingBatch, with strict custody and bounded execution.

The supervised worker exports the reference, compiles a debug example and streams
exactly sixteen complete games through one live TrainingBatch. No fixture is
published until coverage, sizes, schemas and source hashes have been validated.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/kaggriculture_env_reference_v1.npz"
MANIFEST = FIXTURE.with_suffix(".json")
REFERENCE = "65f0eac5bb00b18a9d3acce319c2a231cbd5dff0"
GAMES, STEPS = 16, 719
MAX_COMPRESSED, MAX_EXPANDED = 8 * 1024**2, 256 * 1024**2
OPS = ROOT / "ops/rebuild-2026-09-29/1.4"
RUST_RECORDER = OPS / "reference_recorder.rs"
POLICY = ROOT / "scripts/kaggriculture_env_reference_policy.py"
GRAMMAR = ROOT / "tests/fixtures/kaggriculture/grammar-v4-reference.jsonl.gz"
GRAMMAR_MANIFEST = GRAMMAR.with_name("grammar-v4-reference.manifest.json")
EXPORT_PATHS = (
    "engine_rs",
    "python/owl/kaggriculture/actor_codec.py",
    "rust-toolchain.toml",
)
COVERAGE = (
    "starvation",
    "drought",
    "ineffective",
    "hires",
    "animal_placements",
    "sell_units",
    "sell_cash",
)
Array = NDArray[Any]


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load_module("env_reference_policy", POLICY)


def reward_config(game: int) -> dict[str, Any]:
    return policy.reward_config(game)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@lru_cache(maxsize=1)
def reference_archive() -> bytes:
    resolved = (
        subprocess.check_output(["git", "rev-parse", REFERENCE], cwd=ROOT)
        .decode()
        .strip()
    )
    require(resolved == REFERENCE, "reference source commit differs")
    return subprocess.check_output(
        ["git", "archive", REFERENCE, *EXPORT_PATHS], cwd=ROOT
    )


@lru_cache(maxsize=1)
def reference_sources() -> dict[str, str]:
    result = {}
    with tarfile.open(fileobj=io.BytesIO(reference_archive())) as archive:
        for member in archive:
            path = Path(member.name)
            require(
                not path.is_absolute() and ".." not in path.parts,
                "unsafe reference archive path",
            )
            require(
                member.isdir() or member.isfile(),
                "reference archive requires regular files",
            )
            if member.isfile():
                source = archive.extractfile(member)
                assert source is not None
                result[member.name] = sha(source.read())
    return result


def source_identity() -> dict[str, Any]:
    """Cache only immutable git objects; inspect all mutable sources every time."""
    return {
        "reference": REFERENCE,
        "archive_sha256": sha(reference_archive()),
        "reference_sha256": dict(reference_sources()),
        "recorder_sha256": sha(Path(__file__).read_bytes()),
        "rust_recorder_sha256": sha(RUST_RECORDER.read_bytes()),
        "policy_sha256": sha(POLICY.read_bytes()),
        "grammar_fixture_sha256": sha(GRAMMAR.read_bytes()),
        "grammar_manifest_sha256": sha(GRAMMAR_MANIFEST.read_bytes()),
    }


def empty_arrays() -> dict[str, Array]:
    pair = (GAMES, STEPS, 2)
    seeds = np.arange(17000, 17016, dtype=np.int64)
    return {
        "transition_indices": np.broadcast_to(
            np.arange(STEPS, dtype=np.int64), (GAMES, STEPS)
        ).copy(),
        "lengths": np.zeros(pair, dtype=np.int64),
        "program_offsets": np.zeros(GAMES * STEPS * 2 + 1, dtype=np.int64),
        "tokens": np.empty((0, 12), dtype=np.int64),
        "rewards": np.zeros(pair, dtype=np.float32),
        "dones": np.zeros(pair, dtype=np.bool_),
        "banks_before": np.zeros(pair, dtype=np.float64),
        "banks_after": np.zeros(pair, dtype=np.float64),
        "econ_before": np.zeros((*pair, 32), dtype=np.int64),
        "econ_after": np.zeros((*pair, 32), dtype=np.int64),
        "terminal_banks": np.zeros((GAMES, 2), dtype=np.float64),
        "terminal_econ": np.zeros((GAMES, 2, 32), dtype=np.int64),
        "terminal_steps": np.zeros(GAMES, dtype=np.int64),
        "terminal_winner": np.zeros(GAMES, dtype=np.int64),
        "autoreset_seed": seeds + 1,
        "next_seed": seeds + 2,
    }


def metadata(arrays: dict[str, Array]) -> dict[str, Any]:
    return {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": sha(value.tobytes(order="C")),
        }
        for name, value in sorted(arrays.items())
    }


def make_manifest(
    arrays: dict[str, Array], coverage: list[dict[str, int]], sources: dict[str, Any]
) -> dict[str, Any]:
    return {
        "format": "kaggriculture-env-reference-v1",
        "games": GAMES,
        "steps_per_game": STEPS,
        "configuration": {},
        "reward_configs": [reward_config(game) for game in range(GAMES)],
        "seeds": list(range(17000, 17016)),
        "seed_stride": 1,
        "max_live_envs": 1,
        "policy": policy.POLICY,
        "sources": sources,
        "coverage": coverage,
        "arrays": metadata(arrays),
        "fixture_sha256": "",
        "expanded_sha256": "",
        "compressed_bytes": 0,
        "expanded_bytes": 0,
    }


def validate_arrays(manifest: dict[str, Any], arrays: dict[str, Array]) -> None:
    template = empty_arrays()
    require(set(arrays) == set(template), "array inventory differs")
    for name, array in arrays.items():
        require(
            array.dtype == template[name].dtype and array.flags.c_contiguous,
            f"array dtype/layout differs: {name}",
        )
        require(
            array.shape == template[name].shape
            if name != "tokens"
            else array.ndim == 2 and array.shape[1] == 12,
            f"array shape differs: {name}",
        )
    require(manifest["arrays"] == metadata(arrays), "array hash/shape custody differs")
    lengths, offsets = arrays["lengths"], arrays["program_offsets"]
    require(
        bool(np.all((lengths >= 1) & (lengths <= 252))),
        "program lengths outside 1..252",
    )
    expected_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(lengths.reshape(-1)))
    )
    require(
        np.array_equal(offsets, expected_offsets), "program offset inventory differs"
    )
    require(int(offsets[-1]) == len(arrays["tokens"]), "packed token inventory differs")
    require(
        np.array_equal(arrays["transition_indices"], template["transition_indices"]),
        "transition index inventory differs",
    )
    require(
        np.array_equal(arrays["autoreset_seed"], template["autoreset_seed"])
        and np.array_equal(arrays["next_seed"], template["next_seed"]),
        "seed consumption differs",
    )
    require(bool(np.all(arrays["terminal_steps"] == STEPS)), "terminal steps differ")
    require(
        bool(np.all(arrays["dones"][:, -1])) and not np.any(arrays["dones"][:, :-1]),
        "terminal done schedule differs",
    )
    require(
        np.array_equal(arrays["terminal_banks"], arrays["banks_after"][:, -1])
        and np.array_equal(arrays["terminal_econ"], arrays["econ_after"][:, -1]),
        "terminal values differ",
    )
    banks = arrays["terminal_banks"]
    winners = np.where(
        banks[:, 0] > banks[:, 1], 0, np.where(banks[:, 0] < banks[:, 1], 1, -1)
    )
    require(
        np.array_equal(arrays["terminal_winner"], winners), "terminal winner differs"
    )
    require(
        np.array_equal(arrays["econ_before"][:, 1:], arrays["econ_after"][:, :-1]),
        "economic transition continuity differs",
    )
    require(
        np.array_equal(arrays["banks_before"][:, 1:], arrays["banks_after"][:, :-1]),
        "bank transition continuity differs",
    )
    for name in ("rewards", "banks_before", "banks_after", "terminal_banks"):
        require(bool(np.all(np.isfinite(arrays[name]))), f"nonfinite {name}")
    require(
        bool(np.all(arrays["econ_before"] >= 0))
        and bool(np.all(arrays["econ_after"] >= arrays["econ_before"])),
        "economic counters decrease",
    )
    coverage = manifest["coverage"]
    require(
        isinstance(coverage, list) and len(coverage) == GAMES,
        "coverage game inventory differs",
    )
    for game, values in enumerate(coverage):
        require(set(values) == set(COVERAGE), f"coverage keys differ for game {game}")
        require(
            all(type(v) is int and v > 0 for v in values.values()),
            f"coverage missing for game {game}: {values}",
        )
        econ = arrays["terminal_econ"][game].sum(axis=0)
        for field, counter in (
            ("starvation", 0),
            ("drought", 1),
            ("ineffective", 2),
            ("sell_units", 8),
            ("sell_cash", 9),
        ):
            require(
                values[field] == int(econ[counter]),
                f"coverage counter differs for game {game}: {field}",
            )


def validate_manifest(manifest: dict[str, Any]) -> None:
    expected = make_manifest(empty_arrays(), [], source_identity())
    require(set(manifest) == set(expected), "manifest key inventory differs")
    for key in (
        "format",
        "games",
        "steps_per_game",
        "configuration",
        "reward_configs",
        "seeds",
        "seed_stride",
        "max_live_envs",
        "policy",
    ):
        require(manifest[key] == expected[key], f"manifest recipe differs: {key}")
    require(
        manifest["sources"] == expected["sources"],
        "reference/recorder/policy source custody differs",
    )
    require(
        type(manifest["compressed_bytes"]) is int
        and 0 < manifest["compressed_bytes"] <= MAX_COMPRESSED,
        "compressed size budget exceeded",
    )
    require(
        type(manifest["expanded_bytes"]) is int
        and 0 < manifest["expanded_bytes"] <= MAX_EXPANDED,
        "expanded size budget exceeded",
    )


def deterministic_npz(arrays: dict[str, Array]) -> tuple[bytes, str, int]:
    output = io.BytesIO()
    expanded = hashlib.sha256()
    expanded_size = 0
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in sorted(arrays.items()):
            content = io.BytesIO()
            np.save(content, array, allow_pickle=False)
            raw = content.getvalue()
            expanded.update(raw)
            expanded_size += len(raw)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, raw, compresslevel=9)
    return output.getvalue(), expanded.hexdigest(), expanded_size


def publish_fixture(
    path: Path, manifest: dict[str, Any], arrays: dict[str, Array]
) -> None:
    validate_arrays(manifest, arrays)
    payload, expanded_sha, expanded_size = deterministic_npz(arrays)
    manifest = manifest | {
        "fixture_sha256": sha(payload),
        "compressed_bytes": len(payload),
        "expanded_sha256": expanded_sha,
        "expanded_bytes": expanded_size,
    }
    validate_manifest(manifest)
    scratch = ROOT / ".codex-tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="env-fixture-", dir=scratch) as directory:
        staged = Path(directory) / path.name
        staged.write_bytes(payload)
        staged.with_suffix(".json").write_bytes(json_bytes(manifest))
        load_fixture(staged)
        # Recheck mutable sources immediately before publishing the complete pair.
        require(
            manifest["sources"] == source_identity(), "source drift before publication"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, path)
        os.replace(staged.with_suffix(".json"), path.with_suffix(".json"))


def load_fixture(path: Path = FIXTURE) -> tuple[dict[str, Any], dict[str, Array]]:
    path = Path(path)
    require(
        path.is_file() and path.with_suffix(".json").is_file(),
        f"Task 1.4 complete 16-game reference fixture is missing: {path}; "
        "run the approved recorder on the pod if the Mac watchdog cannot fit it",
    )
    manifest = json.loads(path.with_suffix(".json").read_bytes())
    validate_manifest(manifest)
    require(
        path.stat().st_size == manifest["compressed_bytes"],
        "fixture compressed size differs",
    )
    require(
        sha(path.read_bytes()) == manifest["fixture_sha256"], "fixture hash differs"
    )
    template = empty_arrays()
    expanded = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        require(
            archive.namelist() == [key + ".npy" for key in sorted(template)],
            "zip member inventory differs",
        )
        total = sum(info.file_size for info in archive.infolist())
        require(
            total == manifest["expanded_bytes"] and total <= MAX_EXPANDED,
            "expanded size budget differs",
        )
        for info in archive.infolist():
            require(info.date_time == (1980, 1, 1, 0, 0, 0), "zip timestamp differs")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                require(version == (1, 0), "unexpected numpy header version")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                meta = manifest["arrays"][info.filename.removesuffix(".npy")]
                require(
                    list(shape) == meta["shape"]
                    and dtype.str == meta["dtype"]
                    and not dtype.hasobject
                    and not fortran,
                    "numpy header shape/dtype/layout differs",
                )
                count = 1
                for size in shape:
                    require(
                        type(size) is int and 0 <= size <= MAX_EXPANDED,
                        "numpy header dimension exceeds budget",
                    )
                    count *= size
                require(
                    count * dtype.itemsize <= MAX_EXPANDED,
                    "numpy header expanded size exceeds budget",
                )
            with archive.open(info) as stream:
                while block := stream.read(1024**2):
                    expanded.update(block)
    require(
        expanded.hexdigest() == manifest["expanded_sha256"],
        "expanded fixture hash differs",
    )
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in template}
    validate_arrays(manifest, arrays)
    return manifest, arrays


def export_reference(directory: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(reference_archive())) as archive:
        # reference_sources has already admitted path and entry types.
        sources = reference_sources()
        for member in archive:
            target = directory / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                source = archive.extractfile(member)
                assert source is not None
                content = source.read()
                require(
                    sha(content) == sources[member.name],
                    f"export source differs: {member.name}",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)


def verify_export(directory: Path) -> None:
    """Recheck every original exported byte after build and before publication."""
    for relative, expected in reference_sources().items():
        path = directory / relative
        require(
            path.is_file() and not path.is_symlink(), f"export source drift: {relative}"
        )
        require(sha(path.read_bytes()) == expected, f"export source drift: {relative}")


def exchange(process: subprocess.Popen[str], request: dict[str, Any]) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    require(
        bool(line), f"reference recorder closed unexpectedly, exit={process.poll()}"
    )
    return json.loads(line)


def decode_bits(values: Any, bits: str, floats: str) -> Array:
    return np.array(values, dtype=bits).view(floats)


def record_worker(output: Path) -> None:
    identity = source_identity()
    scratch = ROOT / ".codex-tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="env-reference-", dir=scratch) as directory:
        exported = Path(directory)
        export_reference(exported)
        example = exported / "engine_rs/examples/task14_recorder.rs"
        example.parent.mkdir(exist_ok=True)
        example.write_bytes(RUST_RECORDER.read_bytes())
        target = scratch / "env-reference-target"
        subprocess.run(
            [
                "cargo",
                "build",
                "--offline",
                "--locked",
                "--manifest-path",
                str(exported / "engine_rs/Cargo.toml"),
                "--example",
                "task14_recorder",
                "--target-dir",
                str(target),
            ],
            cwd=exported,
            check=True,
        )
        verify_export(exported)
        codec = load_module(
            "pinned_reference_actor_codec",
            exported / "python/owl/kaggriculture/actor_codec.py",
        )
        arrays, packed, coverage = empty_arrays(), [], []
        process = subprocess.Popen(
            [str(target / "debug/examples/task14_recorder")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            for game in range(GAMES):
                reward = reward_config(game)
                reward["mode"] = reward.pop("reward_mode")
                reward["share_weight"] = 0.0
                row = exchange(
                    process, {"op": "start", "seed": 17000 + game, "reward": reward}
                )
                require(
                    row["seeds"] == [str(17000 + game)]
                    and row["next_seed"] == str(17001 + game),
                    "reference initial seeds differ",
                )
                hires, placements = 0, 0
                for step in range(STEPS):
                    public = row["public"]
                    require(
                        public["step"] == step,
                        f"reference clock differs game={game} step={step}",
                    )
                    tokens = np.zeros((2, 252, 12), dtype=np.int64)
                    lengths = []
                    for seat in range(2):
                        action = policy.action(public, seat)
                        encoded = codec.encode_action(
                            action,
                            {
                                "observation": public | {"player": seat},
                                "configuration": {},
                            },
                            hire_limit=241,
                        )
                        active = np.array(
                            [
                                [frame[name] for name in codec.SLOT_NAMES]
                                for frame in encoded["frames"]
                            ],
                            dtype=np.int64,
                        )
                        count = len(active)
                        tokens[seat, :count] = active
                        lengths.append(count)
                        packed.append(active)
                    row = exchange(
                        process,
                        {
                            "op": "step",
                            "tokens": tokens.reshape(-1).tolist(),
                            "lengths": lengths,
                        },
                    )
                    arrays["lengths"][game, step] = lengths
                    arrays["rewards"][game, step] = decode_bits(
                        row["rewards_bits"], "<u4", "<f4"
                    )
                    arrays["dones"][game, step] = row["dones"]
                    for side in ("before", "after"):
                        arrays["banks_" + side][game, step] = decode_bits(
                            row["banks_" + side + "_bits"], "<u8", "<f8"
                        )
                        arrays["econ_" + side][game, step] = np.array(
                            row["econ_" + side], dtype=np.int64
                        ).reshape(2, 32)
                    if step == 0:
                        hires = len(row["public"]["farms"][0]["hands"]) - len(
                            public["farms"][0]["hands"]
                        )
                    if step == 3:
                        placements = int(row["goose_placed"])
                    require(
                        row["dones"] == [step == STEPS - 1] * 2,
                        f"reference horizon differs game={game} step={step}",
                    )
                    if step != STEPS - 1:
                        require(
                            row["terminal"] is None, "early reference terminal record"
                        )
                        require(
                            row["next_seed"] == str(17001 + game),
                            "nonterminal seed consumed",
                        )
                terminal = row["terminal"]
                require(
                    terminal is not None and terminal["episode_steps"] == STEPS,
                    "reference terminal steps differ",
                )
                arrays["terminal_banks"][game] = decode_bits(
                    terminal["banks_bits"], "<u8", "<f8"
                )
                arrays["terminal_econ"][game] = arrays["econ_after"][game, -1]
                arrays["terminal_steps"][game] = int(terminal["episode_steps"])
                a, b = arrays["terminal_banks"][game]
                arrays["terminal_winner"][game] = 0 if a > b else 1 if a < b else -1
                require(
                    decode_bits([terminal["margin_bits"]], "<u8", "<f8")[0] == a - b,
                    "reference margin differs",
                )
                require(
                    row["public"]["step"] == 0 and row["seeds"] == [str(17001 + game)],
                    "reference auto-reset differs",
                )
                arrays["autoreset_seed"][game] = int(row["seeds"][0])
                arrays["next_seed"][game] = int(row["next_seed"])
                econ = arrays["terminal_econ"][game].sum(axis=0)
                coverage.append(
                    dict(
                        starvation=int(econ[0]),
                        drought=int(econ[1]),
                        ineffective=int(econ[2]),
                        hires=hires,
                        animal_placements=placements,
                        sell_units=int(econ[8]),
                        sell_cash=int(econ[9]),
                    )
                )
                print(
                    f"reference game {game + 1}/16: 719 transitions; "
                    f"coverage={coverage[-1]}",
                    flush=True,
                )
            assert process.stdin is not None
            process.stdin.close()
            require(process.wait(timeout=5) == 0, "reference process failed on EOF")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        arrays["tokens"] = np.concatenate(packed)
        arrays["program_offsets"][1:] = np.cumsum(arrays["lengths"].reshape(-1))
        verify_export(exported)
        require(identity == source_identity(), "source drift during recording")
        publish_fixture(output, make_manifest(arrays, coverage, identity), arrays)


def resident_bytes(pid: int) -> int:
    if sys.platform == "darwin":
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        info = ctypes.create_string_buffer(96)
        return (
            int((ctypes.c_uint64 * 2).from_buffer(info)[1])
            if libproc.proc_pidinfo(pid, 4, 0, info, 96) == 96
            else 0
        )
    return int(
        (Path("/proc") / str(pid) / "statm").read_text().split()[1]
    ) * os.sysconf("SC_PAGE_SIZE")


def group_rss(group: int) -> int:
    """Sample the full worker process group, including cargo and native children."""
    total = 0
    if sys.platform == "darwin":
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        pids = (ctypes.c_int * 4096)()
        count = libproc.proc_listpgrppids(group, pids, ctypes.sizeof(pids))
        require(0 <= count < len(pids), "process-group inventory unavailable")
        for pid in pids[:count]:
            total += resident_bytes(pid)
    elif sys.platform == "linux":
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) == group:
                    total += resident_bytes(int(path.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
    else:
        raise RuntimeError("watchdog requires macOS libproc or Linux /proc")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--games", type=int, default=GAMES)
    parser.add_argument("--first-seed", type=int, default=17000)
    parser.add_argument("--max-live-envs", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=115)
    parser.add_argument("--max-rss-mib", type=int, default=960)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(
        (args.reference, args.games, args.first_seed, args.max_live_envs)
        == (REFERENCE, GAMES, 17000, 1),
        "exact pinned 16-game one-live-env recipe required",
    )
    require(
        args.max_seconds > 0 and args.max_rss_mib > 0,
        "positive watchdog budget required",
    )
    if sys.platform == "darwin":
        require(
            args.max_seconds <= 115 and args.max_rss_mib <= 960,
            "Mac execution budget cannot be widened",
        )
    if args.worker:
        record_worker(args.output.resolve())
        return
    scratch = ROOT / ".codex-tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="env-reference-result-", dir=scratch
    ) as directory:
        staged = Path(directory) / args.output.name
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--output",
            str(staged),
            "--worker",
        ]
        supervise(command, args.max_seconds, args.max_rss_mib)
        # The worker may die at any point without publishing into the fixture tree.
        manifest, arrays = load_fixture(staged)
        publish_fixture(args.output.resolve(), manifest, arrays)


def supervise(command: list[str], max_seconds: float, max_rss_mib: int) -> None:
    start = time.monotonic()
    limit = max_seconds
    if "KG_BOUNDED_DEADLINE_UNIX" in os.environ:
        limit = min(
            limit, float(os.environ["KG_BOUNDED_DEADLINE_UNIX"]) - time.time() - 3
        )
    require(limit > 0, "outer watchdog deadline exhausted")
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    peak, reason = 0, None
    try:
        while process.poll() is None:
            peak = max(peak, group_rss(process.pid) + resident_bytes(os.getpid()))
            if peak >= max_rss_mib * 1024**2 or time.monotonic() - start >= limit:
                reason = "memory" if peak >= max_rss_mib * 1024**2 else "wall_time"
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.05)
        code = process.wait()
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    receipt = {
        "argv": command,
        "exit_status": code,
        "stop_reason": reason,
        "wall_seconds": time.monotonic() - start,
        "time_limit_seconds": limit,
        "sampled_process_group_peak_rss_bytes": peak,
        "rss_limit_bytes": max_rss_mib * 1024**2,
        "profile": "debug",
        "max_live_envs": 1,
    }
    (OPS / "reference-recording-attempt.json").write_bytes(json_bytes(receipt))
    print(json.dumps(receipt), flush=True)
    if code != 0 or reason is not None:
        raise SystemExit(
            f"Reference recording failed: exit={code}, reason={reason}; "
            "no partial fixture published. Budget failures are PENDING (pod); "
            "source/runtime failures require diagnosis."
        )


if __name__ == "__main__":
    main()
