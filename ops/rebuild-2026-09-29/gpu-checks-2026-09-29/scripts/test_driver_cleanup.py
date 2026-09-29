"""Local test: an outer timeout firing mid-stage leaves no stage process alive.

Runs driver.py with dummy stages (Python sleep subprocesses; no GPU, no torch,
no /workspace paths) under an emulation of launch.sh's
`timeout -s TERM -k 20 <N>`, then looks for any surviving stage process.

Scenarios:
  new_phase1  revised scripts/driver.py; timeout fires while both Phase 1
              streams run (worker threads). One stage and its grandchild
              ignore SIGTERM, so the SIGKILL escalation is exercised.
  new_phase2  revised driver; Phase 1 passes quickly and the timeout fires
              during a Phase 2 stage (run in the driver's main thread).
  new_complete revised driver; every stage is quick and passes before the
              timeout, so the ordinary exit path (exit 0, no cleanup) is checked.
  old_phase1  control: the driver as run in attempt 2 (pod/attempt2/driver.py,
              no signal handler). Expected to leave survivors, which shows
              the check discriminates. The test kills them afterwards.

Why an emulation: GNU coreutils `timeout` is not installed on this Mac. The
emulator reproduces what GNU timeout does without --foreground: the command
runs in its own process group; on expiry it sends the signal to the command
and to that group, and after the -k grace sends SIGKILL to both.

Usage (from any directory): python3 test_driver_cleanup.py
Exit 0 when every new_* timeout scenario has zero survivors and the driver
exited on its own (rc 143) before the -k grace, new_complete exited 0 without
cleanup, and the old_* control left survivors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
NEW_DRIVER = HERE / "driver.py"
OLD_DRIVER = HERE.parent / "pod" / "attempt2" / "driver.py"
TIMEOUT_S = 3.0
KILL_AFTER_S = 20.0

DUMMY_STAGE = r'''
import json, os, signal, subprocess, sys, time
argv = sys.argv[1:]
backends = argv[argv.index("--backends") + 1]
record = argv[argv.index("--record") + 1]
mode = argv[argv.index("--mode") + 1]
token = argv[argv.index("--token") + 1]
want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
with open(record, "w") as fh:
    json.dump({"value_at_start": want, "value_at_end": want}, fh)
if mode == "quick":
    time.sleep(0.3)
    sys.exit(0)
ignore = mode == "long_ignore_term"
if ignore:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child_code = ("import signal, sys, time; "
              "sys.argv[1] == 'ignore' and signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              "time.sleep(300)")
gc = subprocess.Popen([sys.executable, "-c", child_code,
                       "ignore" if ignore else "default", token])
piddir = os.path.join(os.path.dirname(record), "pids")
os.makedirs(piddir, exist_ok=True)
for role, pid in (("leader", os.getpid()), ("grandchild", gc.pid)):
    open(os.path.join(piddir, f"{role}-{pid}"), "w").close()
time.sleep(300)
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(name: str, role: str, mode: str, token: str) -> tuple[Any, ...]:
    return (name, role, "ATEN", name, "dummy.py", ["--mode", mode, "--token", token])


def child(driver_path: str, scenario: str, workdir: str, token: str) -> None:
    """Run inside the emulated `timeout`: patch the driver and run its main."""
    spec = importlib.util.spec_from_file_location("gpu_checks_driver", driver_path)
    assert spec is not None and spec.loader is not None
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    wd = Path(workdir)
    drv.RUN = wd
    drv.ROOT = wd
    drv.PY = sys.executable
    drv.WRAP = str(wd / "dummy_stage.py")
    for key in list(drv.JUDGES):
        drv.JUDGES[key] = lambda ev: []
    if scenario.endswith("phase1"):
        drv.GPU0_PHASE1 = [stage("c3_long_ignore", "correct", "long_ignore_term", token),
                           stage("c3_after", "correct", "long", token)]
        drv.GPU1_PHASE1 = [stage("c2_long", "correct", "long", token),
                           stage("c1_after", "correct", "long", token)]
        drv.GPU0_PHASE2 = [stage("c4_never", "correct", "long", token)]
    elif scenario == "new_complete":
        drv.GPU0_PHASE1 = [stage("c3_quick", "correct", "quick", token)]
        drv.GPU1_PHASE1 = [stage("c2_quick", "correct", "quick", token)]
        drv.GPU0_PHASE2 = [stage("c4_quick", "correct", "quick", token)]
    else:
        drv.GPU0_PHASE1 = [stage("c3_quick", "correct", "quick", token)]
        drv.GPU1_PHASE1 = [stage("c2_quick", "correct", "quick", token),
                           stage("c2_ctl_quick", "control", "quick", token)]
        drv.GPU0_PHASE2 = [stage("c4_long_ignore", "correct", "long_ignore_term", token),
                           stage("c4_after", "correct", "long", token)]
    sys.exit(drv.Driver().main())


def emulated_timeout(cmd: list[str], duration: float, kill_after: float,
                     log_path: Path) -> dict[str, Any]:
    """GNU `timeout -s TERM -k kill_after duration cmd`, without --foreground."""
    t0 = time.monotonic()
    with open(log_path, "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=duration)
            return {"timed_out": False, "rc": rc,
                    "elapsed_s": round(time.monotonic() - t0, 2)}
        except subprocess.TimeoutExpired:
            pass
        fired = time.monotonic()
        os.kill(p.pid, signal.SIGTERM)
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        needed_kill = False
        try:
            rc = p.wait(timeout=kill_after)
        except subprocess.TimeoutExpired:
            needed_kill = True
            for fn in (os.kill, os.killpg):
                try:
                    fn(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            rc = p.wait()
    return {"timed_out": True, "driver_rc": rc, "timeout_needed_sigkill": needed_kill,
            "driver_exit_after_signal_s": round(time.monotonic() - fired, 2),
            "elapsed_s": round(time.monotonic() - t0, 2)}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie would still answer; report its state from ps.
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                        text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def survivors(token: str) -> list[str]:
    out = subprocess.run(["pgrep", "-fl", token], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def run_scenario(scenario: str, driver: Path) -> dict[str, Any]:
    token = f"kgv3-cleanup-test-{uuid.uuid4().hex[:12]}"
    wd = Path(tempfile.mkdtemp(prefix=f"kgv3-{scenario}-"))
    (wd / "dummy_stage.py").write_text(DUMMY_STAGE)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(driver),
           scenario, str(wd), token]
    res = emulated_timeout(cmd, TIMEOUT_S, KILL_AFTER_S, wd / "driver.out")
    time.sleep(1.0)
    pids = sorted(wd.glob("pids/*"))
    pid_alive = {f.name: alive(int(f.name.split("-")[1])) for f in pids}
    surv = survivors(token)
    records = [json.loads(line) for line in (wd / "driver.jsonl").read_text().splitlines()
               if line] if (wd / "driver.jsonl").exists() else []
    res.update({
        "scenario": scenario, "driver": str(driver.relative_to(HERE.parent)),
        "driver_sha256": sha256(driver), "workdir": str(wd),
        "stage_pids_recorded": len(pids),
        "stage_pids_alive": sorted(k for k, v in pid_alive.items() if v),
        "survivors_by_token": surv,
        "driver_log": [{k: r[k] for k in ("t", "stage", "status", "rc", "signal",
                                           "reason", "groups") if k in r}
                       for r in records],
    })
    if surv or any(pid_alive.values()):
        for f in pids:
            try:
                os.kill(int(f.name.split("-")[1]), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)
        res["survivors_after_test_kill"] = survivors(token)
    shutil.rmtree(wd)
    return res


def main() -> int:
    print(f"# test_driver_cleanup.py  utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"# platform={platform.platform()} python={platform.python_version()}")
    print(f"# emulated `timeout -s TERM -k {KILL_AFTER_S:g} {TIMEOUT_S:g}` "
          "(GNU timeout not installed locally)")
    print(f"# test script sha256={sha256(Path(__file__).resolve())}")
    results = [run_scenario("new_phase1", NEW_DRIVER),
               run_scenario("new_phase2", NEW_DRIVER),
               run_scenario("new_complete", NEW_DRIVER),
               run_scenario("old_phase1", OLD_DRIVER)]
    ok = True
    for r in results:
        print(json.dumps(r, indent=1))
        clean = not r["survivors_by_token"] and not r["stage_pids_alive"]
        if r["scenario"] == "new_complete":
            good = (clean and not r["timed_out"] and r["rc"] == 0
                    and not any(x.get("status") in ("signal", "stage_cleanup")
                                for x in r["driver_log"]))
        elif r["scenario"].startswith("new_"):
            want_pids = 4 if r["scenario"].endswith("phase1") else 2
            good = (clean and r["stage_pids_recorded"] == want_pids and r.get("timed_out")
                    and not r.get("timeout_needed_sigkill")
                    and r.get("driver_rc") == 128 + signal.SIGTERM)
        else:
            good = not clean and r["stage_pids_recorded"] == 4
        print(f"# {r['scenario']}: {'PASS' if good else 'FAIL'} "
              f"(survivors={len(r['survivors_by_token'])}, "
              f"pids recorded={r['stage_pids_recorded']}, "
              f"driver rc={r.get('driver_rc', r.get('rc'))}, "
              f"driver exit after signal={r.get('driver_exit_after_signal_s')} s)")
        ok = ok and bool(good)
    print(f"# overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:6])
    else:
        sys.exit(main())
