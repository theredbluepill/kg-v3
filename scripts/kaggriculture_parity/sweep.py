"""Differential parity sweep: Kaggle's Python engine against the Rust kernel.

Generates N seeded games (rotating random, edge-case, built-in and mixed-seat
policies over several configurations) plus the minimal input probes in an
isolated ``kaggle-environments==1.32.7`` environment, replays every trace in the
Rust kernel, and writes a JSON summary with per-policy counts and each trace's
first divergence (step, field, expected and actual). Single process, CPU only::

    uv run python scripts/kaggriculture_parity/sweep.py --games 40
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts/kaggriculture_parity/generate_traces.py"
DEFAULT_OUT = REPO_ROOT / "ops/rebuild-2026-09-29/1.1b"
DEFAULT_TRACES = REPO_ROOT / "engine_rs/target/kaggriculture-parity/sweep"
KAGGLE_REQUIREMENT = "kaggle-environments==1.32.7"


def isolated_python() -> list[str]:
    return [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        KAGGLE_REQUIREMENT,
        "python",
    ]


def _run(command: list[str], env: dict[str, str] | None = None) -> float:
    started = time.perf_counter()
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return time.perf_counter() - started


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    return []


def _commands(action: Any) -> tuple[list[Any], list[Any]]:
    """(unit commands, market orders) of one submitted action."""
    if not isinstance(action, dict):
        return [], []
    hands = action.get("hands")
    market = action.get("market")
    units = [action.get("farmer"), *(hands if isinstance(hands, list) else [])]
    return units, market if isinstance(market, list) else []


def _unhashable(value: Any) -> bool:
    return isinstance(value, list | dict)


def classify(actions: list[Any]) -> str | None:
    """Name the documented divergence class that a step's inputs belong to.

    D1: a non-ASCII Unicode decimal digit string (Python int() accepts it).
    D2: an unhashable verb or item that reaches one of Python's dict lookups:
    a unit verb, a PLANT/PICKUP/PLACE item, or a BUY_SEED/BUY_ANIMAL item.
    """
    for text in _strings(actions):
        if any(not c.isascii() for c in text) and text.strip().isdecimal():
            return "D1"
    for action in actions:
        units, orders = _commands(action)
        for command in units:
            if not isinstance(command, list) or not command:
                continue
            if _unhashable(command[0]):
                return "D2"
            if (
                command[0] in ("PLANT", "PICKUP", "PLACE")
                and len(command) >= 2
                and _unhashable(command[1])
            ):
                return "D2"
        for order in orders:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] in ("BUY_SEED", "BUY_ANIMAL")
                and _unhashable(order[1])
            ):
                return "D2"
    return None


def _record(path: Path, line: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, text in enumerate(handle):
            if index == line:
                return dict(json.loads(text))
    raise ValueError(f"{path}: no line {line}")


def summarize(
    generated: list[dict[str, Any]],
    report: list[dict[str, Any]],
    traces_dir: Path,
) -> dict[str, Any]:
    by_file = {entry["path"]: entry for entry in generated}
    replayed = {entry["file"]: entry for entry in report}
    missing = sorted(set(by_file) - set(replayed))
    if missing:
        raise RuntimeError(f"Rust did not replay: {missing}")
    policies: dict[str, Counter[str]] = defaultdict(Counter)
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    divergences = []
    for name, entry in sorted(by_file.items()):
        result = replayed[name]
        kind = "probe" if "probe" in entry else "game"
        labels = [
            "script" if kind == "probe" and seat == 0 else policy
            for seat, policy in enumerate(entry["policies"])
        ]
        for label in labels:
            policies[label]["seats"] += 1
        for label in set(labels):
            policies[label]["probes" if kind == "probe" else "games"] += 1
            policies[label]["transitions"] += entry["transitions"]
        variants[entry["config_variant"]]["traces"] += 1
        variants[entry["config_variant"]]["transitions"] += entry["transitions"]
        if result["ok"]:
            continue
        divergence = result["divergence"]
        record = _record(traces_dir / name, divergence["line"])
        actions = record.get("actions", [])
        divergences.append(
            {
                "file": name,
                "kind": kind,
                "seed": entry["seed"],
                "policies": entry["policies"],
                "config_variant": entry["config_variant"],
                **({"probe": entry["probe"]} if "probe" in entry else {}),
                "first_divergence": divergence,
                "known_class": classify(actions if isinstance(actions, list) else []),
            }
        )
    games = [e for e in generated if "probe" not in e]
    probes = [e for e in generated if "probe" in e]
    new = [d for d in divergences if d["known_class"] is None]
    return {
        "games": len(games),
        "probes": len(probes),
        "traces": len(generated),
        "transitions": sum(e["transitions"] for e in generated),
        "game_transitions": sum(e["transitions"] for e in games),
        "python_rejected_steps": sum(e["rejected"] for e in generated),
        "traces_agreeing": sum(1 for r in report if r["ok"]),
        "traces_diverging": len(divergences),
        "divergences_by_class": dict(
            Counter(d["known_class"] or "unclassified" for d in divergences)
        ),
        "new_divergences": len(new),
        "per_policy": {k: dict(v) for k, v in sorted(policies.items())},
        "per_config_variant": {k: dict(v) for k, v in sorted(variants.items())},
        "seeds": sorted({e["seed"] for e in games}),
        "divergences": divergences,
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--base-seed", type=int, default=20260929)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--include-known-divergences", action="store_true")
    args = parser.parse_args(argv)
    if args.traces.exists() and any(args.traces.iterdir()):
        raise SystemExit(f"{args.traces} is not empty; choose a fresh --traces dir")
    args.traces.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "UV_OFFLINE": os.environ.get("UV_OFFLINE", "1")}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    timings: dict[str, float] = {}
    games_summary = args.traces / "generated-games.json"
    command = [
        *isolated_python(),
        str(GENERATOR),
        "--preset",
        "sweep",
        "--games",
        str(args.games),
        "--base-seed",
        str(args.base_seed),
        "--out",
        str(args.traces),
        "--summary",
        str(games_summary),
    ]
    if args.include_known_divergences:
        command.append("--include-known-divergences")
    timings["generate_games_s"] = _run(command, env)
    generated = json.loads(games_summary.read_text(encoding="utf-8"))
    if not args.no_probes:
        probes_summary = args.traces / "generated-probes.json"
        timings["generate_probes_s"] = _run(
            [
                *isolated_python(),
                str(GENERATOR),
                "--preset",
                "probes",
                "--out",
                str(args.traces),
                "--summary",
                str(probes_summary),
            ],
            env,
        )
        generated += json.loads(probes_summary.read_text(encoding="utf-8"))
    report_path = args.traces / "rust-report.json"
    rust_env = {
        **env,
        "KAGG_PARITY_TRACES": str(args.traces),
        "KAGG_PARITY_REPORT": str(report_path),
        "CARGO_BUILD_JOBS": env.get("CARGO_BUILD_JOBS", "3"),
    }
    rust = [
        "cargo",
        "test",
        "--manifest-path",
        "engine_rs/Cargo.toml",
        "--locked",
        "--offline",
        "--test",
        "replay_parity",
        "env_directory_traces",
        "--",
        "--exact",
        "--test-threads=1",
    ]
    started_rust = time.perf_counter()
    print("+", " ".join(rust), flush=True)
    rust_status = subprocess.run(rust, cwd=REPO_ROOT, env=rust_env, check=False)
    timings["rust_replay_s"] = time.perf_counter() - started_rust
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "sweep": "kaggriculture-live-differential-parity",
        "started": started,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty_paths": _git("status", "--porcelain").splitlines(),
        "python_engine": KAGGLE_REQUIREMENT,
        "platform": platform.platform(),
        "arguments": {
            "games": args.games,
            "base_seed": args.base_seed,
            "probes": not args.no_probes,
            "include_known_divergences": args.include_known_divergences,
            "traces_dir": str(args.traces),
        },
        "rust_test_exit_code": rust_status.returncode,
        "timings": {key: round(value, 2) for key, value in timings.items()},
        **summarize(generated, report, args.traces),
    }
    output = args.out / "sweep-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{summary['traces']} traces ({summary['games']} games, {summary['probes']} "
        f"probes), {summary['transitions']} transitions: {summary['traces_agreeing']} "
        f"agree, {summary['traces_diverging']} diverge "
        f"{summary['divergences_by_class']}; summary {output}"
    )
    return 1 if summary["new_divergences"] else 0


if __name__ == "__main__":
    sys.exit(main())
