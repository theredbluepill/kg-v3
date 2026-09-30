"""Pod watchdog for run A + LR/2 (4 ranks). Usage: watchdog.py PGID LOG

Trimmed from ../main-J-4rank/watchdog.py (sha256 65050a89...) to the owner's
scope for this run: "no need a watch dog to watch the earn"; a nonfinite-loss
stop is fine. Every 60 s it reads the new lines of LOG and stops the run
(SIGTERM to the process group PGID and its descendants, SIGKILL after 60 s)
with the reason recorded, on exactly one condition:
- a nonfinite value in any loss/* metric of any rank's [nt-probe] iteration
  record (main_probe.py).
Removed: the bank floor stop and the replay-drift text match. The trainer's
own first-minibatch log-ratio RuntimeError is unchanged in run_ppo; it ends
the run by itself and is not a watchdog action.
Observe only: rank 0's train/own_bank_mean, train/reward_bank_mean and
train/return_common_mean are printed at each completed-game interval; they
never stop the run.
The run is the process group PGID plus its descendants (torchrun puts each
rank in its own session). It exits when all of them are gone. Log lines go to stdout.
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
TAG = "[nt-probe] "


def say(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def _proc_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, pgid) from /proc/<pid>/stat."""
    table = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as handle:
                stat = handle.read()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        table[int(name)] = (int(fields[1]), int(fields[2]))
    return table


def run_pids() -> set[int]:
    """The run's process group plus every descendant."""
    table = _proc_table()
    pids = {pid for pid, (_, pgid) in table.items() if pgid == PGID}
    grew = True
    while grew:
        grew = False
        for pid, (ppid, _) in table.items():
            if ppid in pids and pid not in pids:
                pids.add(pid)
                grew = True
    return pids


def _signal_all(pids: set[int], sig: signal.Signals) -> None:
    try:
        os.killpg(PGID, sig)
    except ProcessLookupError:
        pass
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop(reason: str) -> None:
    pids = run_pids()
    say(f"STOP reason={reason}; SIGTERM to process group {PGID} and pids {sorted(pids)}")
    _signal_all(pids, signal.SIGTERM)
    for _ in range(60):
        time.sleep(1)
        if not run_pids():
            say("run exited after SIGTERM")
            return
    pids = run_pids()
    say(f"still alive after 60 s; SIGKILL pids {sorted(pids)}")
    _signal_all(pids, signal.SIGKILL)


_DECODER = json.JSONDecoder()
# A record that does not decode yet may be incomplete (still being written, or
# its ranks' writes interleaved); wait for more text up to this many chars.
_MAX_PENDING = 1 << 16


def main() -> None:
    say(f"watchdog start pgid={PGID} log={LOG} poll={POLL_S}s stop=nonfinite loss/* only")
    offset = 0
    buf = ""
    iterations = 0
    polls = 0
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    skipped = 0
    last_bank: tuple[int, float, float] | None = None
    while True:
        running = bool(run_pids())
        try:
            with open(LOG, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset += len(chunk)
        except FileNotFoundError:
            chunk = b""
        buf += chunk.decode("utf-8", errors="replace")
        reason = None
        pos = 0
        # Records are found by their tag anywhere in the text, not per line:
        # the ranks share the log and a record's newline can land elsewhere.
        while True:
            start = buf.find(TAG, pos)
            if start < 0:
                pos = max(pos, len(buf) - len(TAG))
                break
            try:
                record, end = _DECODER.raw_decode(buf, start + len(TAG))
            except json.JSONDecodeError:
                if len(buf) - start < _MAX_PENDING:
                    pos = start
                    break
                skipped += 1
                pos = start + 1
                continue
            pos = end
            if not isinstance(record, dict) or record.get("kind") != "iteration":
                continue
            metrics = record.get("metrics", {})
            rank = record.get("rank")
            it = record.get("iteration")
            if rank in counts:
                counts[rank] += 1
            for key, value in metrics.items():
                if key.startswith("loss/") and isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        reason = reason or f"nonfinite {key}={value} rank={rank} iteration={it}"
            if rank != 0:
                continue
            iterations = it
            games = metrics.get("train/bank_games", 0.0)
            bank = metrics.get("train/own_bank_mean")
            if games and games > 0 and bank is not None:
                last_bank = (it, games, bank)
                say(
                    f"observe iteration={it} games={games:g} own_bank_mean={bank:.1f} "
                    f"reward_bank_mean={metrics.get('train/reward_bank_mean')} "
                    f"return_common_mean={metrics.get('train/return_common_mean')}"
                )
        buf = buf[pos:]
        if reason is not None:
            stop(reason)
            say("watchdog exit after stop")
            return
        polls += 1
        if polls % 10 == 0 or polls == 1:
            say(
                f"heartbeat rank0_iteration={iterations} records_per_rank={counts} "
                f"skipped={skipped} last_bank={last_bank}"
            )
        if not running:
            say(f"run gone; last rank-0 iteration={iterations} last_bank={last_bank}; watchdog exit")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
