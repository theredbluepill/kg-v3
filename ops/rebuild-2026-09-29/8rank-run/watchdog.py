"""Watchdog for an 8-rank Kaggriculture PPO run (plan 6.3b main run, DRAFT).

launch.sh starts it beside ``torchrun`` (its own process group). Every poll it:

1. **Copies checkpoints off the pod's disk.** ``run_ppo`` writes checkpoints
   atomically (``.name.tmp`` then rename), so every visible ``checkpoint_*.pt``
   is complete. Each new or changed file is hashed and appended to
   ``<receipts>/checkpoints.sha256``; with ``--durable-dir`` (a network-volume
   mount) it is also copied there and the copy's SHA-256 is verified.
   ``checkpoint_last_best.pt`` is rewritten on promotion, so its copies are
   named by hash. The Mac side pulls the same files with ``pull_from_pod.sh``;
   a pod without a network volume loses its disk when stopped.
2. **Reads the run's W&B history** (entity, project and run id from the run
   directory's ``attempts.jsonl``) through ``wandb.Api`` and applies the stop
   rules of ``run-statement-6.3b.md``:
   - any nonfinite logged metric;
   - ``train/own_bank_mean`` below ``--bank-floor`` (20,000) at
     ``--consecutive`` (2) consecutive game intervals, a game interval being a
     logged iteration with ``train/bank_games > 0``.
   Failure to promote (``eval/promoted == 0``) is never a stop. A failed W&B
   read is logged as a ``TELEMETRY READ OUTAGE`` and the run continues.
   The first-minibatch log-ratio alarm is the trainer's own: it raises, and
   torchrun exits nonzero.
3. On a stop rule it writes ``<receipts>/watchdog_stop.json`` and sends SIGTERM
   to the torchrun process group, then SIGKILL after ``--grace-seconds``.

It exits after the process group is gone and a last checkpoint copy.
``--self-test`` checks the stop rules on synthetic rows without W&B or a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NONFINITE_STRINGS = {"nan", "inf", "-inf", "infinity", "-infinity"}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[watchdog {stamp}] {message}", flush=True)


def is_nonfinite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        return value.strip().lower() in _NONFINITE_STRINGS
    return False


@dataclass
class StopRules:
    bank_floor: float
    consecutive: int
    low_games: list[tuple[int, float]] = field(default_factory=list)
    game_intervals: int = 0

    def observe(self, row: Mapping[str, Any]) -> str | None:
        """Return a stop reason for this history row, or None."""
        step = int(row.get("_step", -1))
        for key, value in row.items():
            if not key.startswith("_") and is_nonfinite(value):
                return f"nonfinite metric {key}={value!r} at step {step}"
        games = row.get("train/bank_games")
        bank = row.get("train/own_bank_mean")
        if isinstance(games, (int, float)) and games > 0 and bank is not None:
            self.game_intervals += 1
            if float(bank) < self.bank_floor:
                self.low_games.append((step, float(bank)))
                if len(self.low_games) >= self.consecutive:
                    series = ", ".join(f"{b:.0f}@{s}" for s, b in self.low_games)
                    return (
                        f"train/own_bank_mean below {self.bank_floor:.0f} at "
                        f"{self.consecutive} consecutive game intervals: {series}"
                    )
            else:
                self.low_games.clear()
        return None


def newest_run_dir(out: Path, since: float) -> Path | None:
    candidates = [
        p.parent for p in out.glob("*/attempts.jsonl") if p.stat().st_mtime >= since - 5
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def wandb_run_path(run_dir: Path) -> str:
    lines = (run_dir / "attempts.jsonl").read_text().splitlines()
    attempt = json.loads(lines[-1])
    if attempt["telemetry_mode"] != "wandb-online":
        raise RuntimeError(f"telemetry_mode is {attempt['telemetry_mode']}")
    return f"{attempt['wandb_entity']}/{attempt['wandb_project']}/{attempt['wandb_run_id']}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointCopier:
    def __init__(self, receipts: Path, durable: Path | None) -> None:
        self.receipts = receipts
        self.durable = durable
        self.seen: dict[str, tuple[int, int]] = {}

    def sync(self, run_dir: Path) -> None:
        for path in sorted(run_dir.glob("checkpoint_*.pt")):
            stat = path.stat()
            key = (stat.st_size, stat.st_mtime_ns)
            if self.seen.get(path.name) == key:
                continue
            digest = sha256(path)
            if path.stat().st_mtime_ns != stat.st_mtime_ns:
                continue  # replaced while hashing; the next poll retries
            with (self.receipts / "checkpoints.sha256").open("a") as out:
                out.write(f"{digest}  {run_dir.name}/{path.name}\n")
            if self.durable is not None:
                name = (
                    f"checkpoint_last_best.{digest[:12]}.pt"
                    if path.name == "checkpoint_last_best.pt"
                    else path.name
                )
                target = self.durable / run_dir.name / name
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f".{target.name}.tmp")
                shutil.copyfile(path, tmp)
                if sha256(tmp) != digest:
                    tmp.unlink()
                    log(f"COPY MISMATCH {path.name}; retrying next poll")
                    continue
                tmp.replace(target)
                log(f"copied {path.name} -> {target} ({digest[:12]})")
            else:
                log(
                    f"hashed {path.name} ({digest[:12]}); no --durable-dir, pull it from the Mac"
                )
            self.seen[path.name] = key
        for extra in ("attempts.jsonl", "warm_start.json", "config.yaml"):
            source = run_dir / extra
            if self.durable is not None and source.is_file():
                target = self.durable / run_dir.name / extra
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Our own group can refuse the probe only once its members are zombies
        # waiting to be reaped (macOS); nothing runs any more.
        return False
    return True


def stop_group(pgid: int, grace: float) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and group_alive(pgid):
        time.sleep(5)
    if group_alive(pgid):
        log("process group still alive after SIGTERM grace; SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def history_rows(run_path: str, min_step: int) -> Iterable[dict[str, Any]]:
    import wandb

    run = wandb.Api(timeout=60).run(run_path)
    return run.scan_history(min_step=min_step, page_size=500)


def watch(args: argparse.Namespace) -> int:
    rules = StopRules(bank_floor=args.bank_floor, consecutive=args.consecutive)
    copier = CheckpointCopier(args.receipts, args.durable_dir)
    started = time.time()
    run_dir: Path | None = None
    run_path: str | None = None
    next_step = 0
    outages = 0
    while True:
        alive = group_alive(args.pgid)
        if run_dir is None:
            run_dir = newest_run_dir(args.out, started)
            if run_dir is not None:
                log(f"run directory {run_dir}")
        if run_dir is not None:
            try:
                copier.sync(run_dir)
            except OSError as error:
                log(f"CHECKPOINT COPY ERROR {error!r}")
            if run_path is None and (run_dir / "attempts.jsonl").is_file():
                try:
                    run_path = wandb_run_path(run_dir)
                    log(f"W&B run {run_path}")
                except (KeyError, ValueError, RuntimeError) as error:
                    log(f"cannot resolve the W&B run: {error!r}")
        if not alive:
            log("torchrun process group has exited; final copy done")
            return 0
        if run_path is not None:
            stop: tuple[str, Any] | None = None
            try:
                for row in history_rows(run_path, next_step):
                    next_step = max(next_step, int(row.get("_step", -1)) + 1)
                    reason = rules.observe(row)
                    if reason is not None:
                        stop = (reason, row.get("_step"))
                        break
                outages = 0
            except Exception as error:  # noqa: BLE001 - any read failure is an outage
                outages += 1
                log(
                    f"TELEMETRY READ OUTAGE #{outages}: {type(error).__name__}: {error}"
                )
            if stop is not None:
                reason, step = stop
                log(f"STOP: {reason}")
                (args.receipts / "watchdog_stop.json").write_text(
                    json.dumps(
                        {
                            "reason": reason,
                            "step": step,
                            "run": run_path,
                            "at": time.time(),
                        },
                        indent=2,
                    )
                )
                stop_group(args.pgid, args.grace_seconds)
                if run_dir is not None:
                    copier.sync(run_dir)
                return 3
        log(
            f"alive; next W&B step {next_step}; game intervals "
            f"{rules.game_intervals}; low streak {len(rules.low_games)}"
        )
        time.sleep(args.poll_seconds)


def self_test() -> int:
    rules = StopRules(bank_floor=20_000, consecutive=2)
    rows = [
        {"_step": 16384, "train/bank_games": 0.0},
        {"_step": 196608, "train/bank_games": 256.0, "train/own_bank_mean": 70_000.0},
        {"_step": 376832, "train/bank_games": 256.0, "train/own_bank_mean": 19_000.0},
        {"_step": 393216, "eval/promoted": 0.0, "eval/win_rate_against_last_best": 0.4},
        {"_step": 557056, "train/bank_games": 256.0, "train/own_bank_mean": 25_000.0},
        {"_step": 737280, "train/bank_games": 256.0, "train/own_bank_mean": 12_000.0},
    ]
    assert all(rules.observe(r) is None for r in rows), "false stop"
    stop = rules.observe(
        {"_step": 917504, "train/bank_games": 256.0, "train/own_bank_mean": 900.0}
    )
    assert stop is not None and "2 consecutive" in stop, stop
    for bad in (float("nan"), float("inf"), "NaN", "-Infinity"):
        reason = StopRules(20_000, 2).observe({"_step": 1, "loss/value_loss": bad})
        assert reason is not None and "nonfinite" in reason, bad
    assert StopRules(20_000, 2).observe({"_step": 1, "_runtime": float("nan")}) is None
    print("watchdog self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pgid", type=int)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--durable-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--bank-floor", type=float, default=20_000.0)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--grace-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.out is None or args.pgid is None or args.receipts is None:
        parser.error("--out, --pgid and --receipts are required")
    args.receipts.mkdir(parents=True, exist_ok=True)
    return watch(args)


if __name__ == "__main__":
    sys.exit(main())
