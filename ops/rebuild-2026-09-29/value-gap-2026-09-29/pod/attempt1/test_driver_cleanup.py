"""Local CPU test of driver.py's process-group cleanup (no GPU, no torch).

The driver runs with VGAP_DRIVER_SELFTEST=<tmpdir>: its stages become dummy
stages (this file with --dummy-stage) that record their own and their
grandchild's PID. The driver is started in its own process group and signalled
as GNU `timeout -s TERM` does without --foreground (SIGTERM to the command's
group; the stages are in their own sessions, outside it). Survivors are looked
up by the recorded PIDs with os.kill(pid, 0).

Scenarios:
  midstage   both streams inside long stages (one stream's stage and grandchild
             ignore SIGTERM, so the SIGKILL escalation runs); SIGTERM at 3 s.
  spawnstorm many stages whose leader exits at once while its grandchild
             (same group) sleeps 300 s, so the driver spawns continuously and
             every earlier group stays alive; SIGTERM at 8 offsets. Any group
             spawned but missed by cleanup would survive.
  complete   three quick stages per stream, no signal: exit 0, no group left.

Pass: every signalled scenario exits 128+15 within 20 s of the signal (before
timeout's -k 20 grace) with zero survivors; `complete` exits 0 with zero
survivors; exactly one cleanup ran, and every stage that wrote its PIDs was in
cleanup's snapshot of registered groups. Any leftover is SIGKILLed by the test afterwards and reported.
Usage: python3 test_driver_cleanup.py
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "driver.py"
GRANDCHILD = "import signal,sys,time\nif sys.argv[1]=='1': signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(300)"


def dummy_stage(argv: list[str]) -> None:
    get = lambda k: argv[argv.index(k) + 1]  # noqa: E731
    backends, record, pids, mode = get("--backends"), get("--record"), get("--pids"), get("--mode")
    want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
    Path(record).write_text(json.dumps({"value_at_start": want, "value_at_end": want}))
    ignore = mode == "ignore"
    if ignore:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "quick":
        Path(pids).write_text(f"{os.getpid()}\n")
        time.sleep(0.05)
        return
    child = subprocess.Popen([sys.executable, "-c", GRANDCHILD, "1" if ignore else "0"])
    Path(pids).write_text(f"{os.getpid()}\n{child.pid}\n")
    if mode == "spawnlong":
        return  # leader exits; the grandchild keeps the group alive
    time.sleep(300)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); check its state
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(st.stdout.strip()) and not st.stdout.strip().startswith("Z")


def run(name: str, mode: str, stages: int, signal_at: float | None) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix=f"vgap-{name}-"))
    run_dir = work / "run"
    run_dir.mkdir()
    for f in ("driver.py", "test_driver_cleanup.py"):
        (run_dir / f).write_bytes((HERE / f).read_bytes())
    env = dict(os.environ, VGAP_DRIVER_SELFTEST=str(work), VGAP_SELFTEST_MODE=mode,
               VGAP_SELFTEST_STAGES=str(stages))
    t0 = time.monotonic()
    p = subprocess.Popen([sys.executable, str(run_dir / "driver.py")], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         start_new_session=True)
    signalled = None
    if signal_at is not None:
        time.sleep(signal_at)
        signalled = time.monotonic()
        os.killpg(p.pid, signal.SIGTERM)  # GNU timeout: the command's group
    try:
        rc: int | str = p.wait(timeout=20 if signal_at is not None else 60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        rc = "needed_sigkill"
    exit_after = round(time.monotonic() - (signalled or t0), 2)
    time.sleep(0.5)
    pids = [int(x) for f in work.glob("*.pids") for x in f.read_text().split()]
    survivors = [pid for pid in pids if alive(pid)]
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log = [json.loads(line) for line in (run_dir / "driver.jsonl").read_text().splitlines()]
    launches = sum(1 for e in log if e.get("status") == "launch")
    cleanup = [e for e in log if e.get("status") == "stage_cleanup"]
    want_rc = 0 if signal_at is None else 128 + signal.SIGTERM
    # Every stage that got far enough to write its PIDs was spawned, and a spawn
    # happens only before cleanup's snapshot, so it must be in that snapshot.
    registered = cleanup[0]["registered"] if cleanup else 0
    pid_files = len(list(work.glob("*.pids")))
    ok = (rc == want_rc and not survivors and len(cleanup) == 1
          and pid_files <= registered)
    return {"scenario": name, "mode": mode, "signal_at_s": signal_at, "rc": rc,
            "want_rc": want_rc, "exit_after_signal_or_start_s": exit_after,
            "launches_logged": launches, "pid_files": pid_files,
            "registered_at_cleanup": registered,
            "pids_recorded": len(pids), "survivors": survivors,
            "cleanup_events": [{k: e[k] for k in ("reason", "registered")}
                               | {"groups": len(e["groups"]),
                                  "sigkill": sum(g["sigkill"] for g in e["groups"]),
                                  "survived": sum(g.get("survived", False)
                                                  for g in e["groups"])}
                               for e in cleanup],
            "stderr_tail": p.stderr.read().decode()[-300:] if p.stderr else "",
            "pass": ok}


def main() -> int:
    import hashlib

    print(f"# test_driver_cleanup.py utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"# platform={platform.platform()} python={platform.python_version()}")
    for f in ("driver.py", "test_driver_cleanup.py"):
        print(f"# {f} sha256={hashlib.sha256((HERE / f).read_bytes()).hexdigest()}")
    results = [run("midstage", "long", 3, 3.0)]
    for i, at in enumerate((0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.1)):
        results.append(run(f"spawnstorm_{i}", "spawnlong", 400, at))
    results.append(run("complete", "quick", 3, None))
    for r in results:
        print(json.dumps(r))
    ok = all(r["pass"] for r in results)
    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--dummy-stage" in sys.argv:
        dummy_stage(sys.argv[1:])
    else:
        sys.exit(main())
