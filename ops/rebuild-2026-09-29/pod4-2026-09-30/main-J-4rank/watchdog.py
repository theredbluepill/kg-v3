"""Pod watchdog for main run J (4 ranks). Usage: watchdog.py PGID LOG

Every 60 s it reads the new lines of LOG and stops the run (SIGTERM to the
process group PGID, SIGKILL after 60 s) with the reason recorded, on:
- a nonfinite value in any loss/* metric of any rank's [nt-probe] iteration
  record (main_probe.py);
- the replay-drift alarm: the trainer's "first-minibatch PPO log-ratio mean"
  RuntimeError text (rl.first_minibatch_logratio_limit) in the log;
- rank 0's train/own_bank_mean < 20000 on 2 consecutive logged iterations that
  have train/bank_games > 0 (iterations with no finished game are skipped).
It exits when the process group is gone. Log lines go to stdout.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time

PGID = int(sys.argv[1])
LOG = sys.argv[2]
POLL_S = int(os.environ.get("WATCHDOG_POLL_S", "60"))
BANK_FLOOR = 20000.0
ALARM = "first-minibatch PPO log-ratio mean"
TAG = "[nt-probe] "


def say(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def alive() -> bool:
    try:
        os.killpg(PGID, 0)
    except ProcessLookupError:
        return False
    return True


def stop(reason: str) -> None:
    say(f"STOP reason={reason}; SIGTERM to process group {PGID}")
    try:
        os.killpg(PGID, signal.SIGTERM)
    except ProcessLookupError:
        say("process group already gone")
        return
    for _ in range(60):
        time.sleep(1)
        if not alive():
            say("process group exited after SIGTERM")
            return
    say("still alive after 60 s; SIGKILL")
    try:
        os.killpg(PGID, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    say(f"watchdog start pgid={PGID} log={LOG} poll={POLL_S}s floor={BANK_FLOOR}")
    offset = 0
    partial = ""
    low_streak = 0
    last_bank: tuple[int, float, float] | None = None
    iterations = 0
    polls = 0
    while True:
        running = alive()
        try:
            with open(LOG, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset += len(chunk)
        except FileNotFoundError:
            chunk = b""
        text = partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        partial = lines.pop()
        reason = None
        for raw in lines:
            for line in raw.split("\r"):
                if ALARM in line:
                    reason = reason or f"replay-drift alarm: {line.strip()[:400]}"
                start = line.find(TAG)
                if start < 0:
                    continue
                try:
                    record = json.loads(line[start + len(TAG):])
                except json.JSONDecodeError:
                    continue
                if record.get("kind") != "iteration":
                    continue
                metrics = record.get("metrics", {})
                rank = record.get("rank")
                it = record.get("iteration")
                for key, value in metrics.items():
                    if key.startswith("loss/") and isinstance(value, (int, float)):
                        if not math.isfinite(value):
                            reason = reason or f"nonfinite {key}={value} rank={rank} iteration={it}"
                if rank != 0:
                    continue
                iterations = it
                games = metrics.get("train/bank_games", 0.0)
                if games and games > 0:
                    bank = metrics.get("train/own_bank_mean")
                    if bank is None:
                        continue
                    last_bank = (it, games, bank)
                    low_streak = low_streak + 1 if bank < BANK_FLOOR else 0
                    say(f"bank iteration={it} games={games:g} own_bank_mean={bank:.1f} low_streak={low_streak}")
                    if low_streak >= 2:
                        reason = reason or (
                            f"train/own_bank_mean < {BANK_FLOOR:g} on 2 consecutive "
                            f"bank intervals (last iteration={it} own_bank_mean={bank:.1f})"
                        )
        if reason is not None:
            stop(reason)
            say("watchdog exit after stop")
            return
        polls += 1
        if polls % 10 == 0:
            say(f"heartbeat rank0_iteration={iterations} last_bank={last_bank}")
        if not running:
            say(f"process group gone; last rank-0 iteration={iterations} last_bank={last_bank}; watchdog exit")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
