"""Value-gap diagnostic driver (run statement
ops/rebuild-2026-09-29/run-statements/value-gap-diagnostic.md).

Two streams in parallel, each in a worker thread: GPU 0 runs the Kaggriculture
stages (kg_gap.py), GPU 1 the Isaiah control (is_gap.py). Every stage is a fresh
subprocess in its own session with its own Inductor/Triton cache. The driver
stops at the FIRST unexpected failure (nonzero exit, timeout, missing/errored
result, a step not ok, a wrong/missing backend record, a non-finite comparison,
wrong use_flash_attn flags, wrong compiled-trunk call count): the other
stream's running stage is terminated and nothing further starts. Gap sizes are
results, never failures.

Exit codes: 0 all non-optional stages ran and passed; 3 a stage failed; 4 the
internal deadline ran out; 128 + signum on SIGTERM/SIGINT/SIGHUP (cleanup ran).

Process-group cleanup. Adapted from kg/rebuild-gpu-checks 6392160 driver.py,
with the spawn/registration gap of Codex review verify-gpu-bundle-r2 finding 3
closed structurally:
  * stages are spawned only from worker threads; CPython runs the Python-level
    signal handler only in the main thread, which never spawns and only joins;
  * Popen and registration happen under self.spawn_lock (non-reentrant), which
    the handler also takes before snapshotting the groups, so a group is either
    registered before cleanup snapshots or is never created (a worker checks
    self.cleaned under the lock before Popen);
  * cleanup: SIGTERM each started group, wait <= TERM_GRACE_S, SIGKILL the
    survivors, wait <= KILL_GRACE_S; total below launch.sh's `-k 20` grace.

VGAP_DRIVER_SELFTEST=<dir> replaces the stages by dummy CPU sleep stages (see
test_driver_cleanup.py); it is for the local cleanup test only.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN = Path(__file__).resolve().parent
SELFTEST = os.environ.get("VGAP_DRIVER_SELFTEST")
ROOT = Path("/workspace/kg-v3-rebuild")
PY = str(ROOT / ".venv/bin/python")
WRAP = str(RUN / "gemm_backend_wrap.py")
BUDGET_S = 42 * 60
MIN_START_S = 90
OPTIONAL_MIN_S = 600
TERM_GRACE_S = 8.0
KILL_GRACE_S = 4.0
CLEANUP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

# name, role ("required" | "optional"), backends, script, args, expectations
Stage = tuple[str, str, str, str, list[str], dict[str, Any]]


def _kg(name: str, role: str, backends: str, precision: str, trunk: str,
        extra: list[str]) -> Stage:
    return (name, role, backends, "kg_gap.py",
            ["--case", name, "--precision", precision, "--trunk", trunk, *extra],
            {"flash": precision == "bf16", "compiled": trunk == "compiled",
             "hidden": "--hidden" in extra, "gain_swap": "--gain-swap" in extra,
             "kind": "kg"})


def _is(name: str, backends: str, trunk: str) -> Stage:
    return (name, "required", backends, "is_gap.py",
            ["--case", name, "--trunk", trunk, "--obs-dir", str(RUN / "obs")],
            {"flash": True, "compiled": trunk == "compiled", "kind": "is"})


GPU0: list[Stage] = [
    _kg("kgA_bf16_comp_aten", "required", "ATEN", "bf16", "compiled",
        ["--hidden", "--gain-swap"]),
    _kg("kgB_bf16_eager", "required", "ATEN", "bf16", "eager",
        ["--hidden", "--gain-swap"]),
    _kg("kgC_fp32_comp_aten", "required", "ATEN", "fp32", "compiled", ["--hidden"]),
    _kg("kgC_fp32_eager", "required", "ATEN", "fp32", "eager", ["--hidden"]),
    _kg("kgA_bf16_comp_default", "required", "default", "bf16", "compiled",
        ["--hidden", "--gain-swap"]),
    _kg("kgC_fp32_comp_default", "optional", "default", "fp32", "compiled",
        ["--hidden"]),
]
GPU1: list[Stage] = [
    _is("isF_eager", "ATEN", "eager"),
    _is("isF_comp_default", "default", "compiled"),
    _is("isF_comp_aten", "ATEN", "compiled"),
]

if SELFTEST:  # dummy stages for the local cleanup test only
    PY = sys.executable
    ROOT = Path(SELFTEST)
    BUDGET_S = int(os.environ.get("VGAP_SELFTEST_BUDGET", "120"))
    MIN_START_S = 0
    OPTIONAL_MIN_S = 0
    _n = int(os.environ.get("VGAP_SELFTEST_STAGES", "3"))
    _mode = os.environ.get("VGAP_SELFTEST_MODE", "long")
    GPU0 = [(f"t0_{i}", "required", "ATEN", "dummy", ["--mode", _mode], {})
            for i in range(_n)]
    GPU1 = [(f"t1_{i}", "required", "ATEN", "dummy",
             ["--mode", "ignore" if _mode == "long" else _mode], {})
            for i in range(_n)]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def nonfinite_paths(obj: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "nonfinite" and isinstance(v, int) and v != 0:
                out.append(f"{path}.nonfinite={v}")
            else:
                out += nonfinite_paths(v, f"{path}.{k}")
    return out


def judge(ev: list[dict[str, Any]], exp: dict[str, Any]) -> list[str]:
    if SELFTEST:
        return []
    res = [e for e in ev if e.get("event") == "result"]
    if len(res) != 1:
        return [f"{len(res)} result records"]
    r = res[0]
    errs: list[str] = []
    if "error" in r:
        errs.append(f"error: {r['error'][:400]}")
    if not any(e.get("event") == "done" for e in ev):
        errs.append("no done record")
    rows = r["rows"]
    if exp["kind"] == "kg":
        want = [f"pair_{n}" for n in rows]
        if exp["hidden"]:
            want += [f"hidden_{n}" for n in rows]
        if exp["gain_swap"]:
            want += [f"gain1_pair_{n}" for n in rows]
    else:
        want = [f"{s}_{n}" for n in rows for s in ("states", "pair", "hidden")]
    if sorted(r["steps"]) != sorted(want) or any(v != "ok" for v in r["steps"].values()):
        errs.append(f"steps {r['steps']} want {want}")
    if r["use_flash_attn_flags"] != [exp["flash"]]:
        errs.append(f"use_flash_attn flags {r['use_flash_attn_flags']} want [{exp['flash']}]")
    calls = r["compiled_trunk_calls"]
    if (calls > 0) != exp["compiled"]:
        errs.append(f"compiled_trunk_calls {calls} (compiled={exp['compiled']})")
    if exp.get("gain_swap") and r.get("gain_swap", {}).get("critic_out_is_swapped"):
        errs.append("gain swap touched the critic head")
    errs += nonfinite_paths(r)[:20]
    return errs


class Driver:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.deadline = self.t0 + BUDGET_S
        self.stop = threading.Event()
        self.spawn_lock = threading.Lock()  # non-reentrant; see module doc
        self.log_lock = threading.RLock()
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.failure: int | None = None
        self.cleaned = False
        # Main-thread only (handler/atexit): set before any lock is taken, so a
        # second signal during cleanup returns instead of re-entering the lock.
        self.cleanup_started = False
        self.log_fh = open(RUN / "driver.jsonl", "a")

    def log(self, rec: dict[str, Any]) -> None:
        rec = {"utc": now(), "t": round(time.monotonic() - self.t0, 1), **rec}
        with self.log_lock:
            self.log_fh.write(json.dumps(rec) + "\n")
            self.log_fh.flush()
            print(json.dumps(rec), flush=True)

    @staticmethod
    def _signal_group(p: subprocess.Popen[bytes], sig: int) -> None:
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    @staticmethod
    def _group_alive(p: subprocess.Popen[bytes]) -> bool:
        p.poll()  # reap an exited leader; members may outlive it
        try:
            os.killpg(p.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def fail(self, code: int) -> None:
        """Record the first failure and stop the other stream (called in workers)."""
        with self.spawn_lock:
            if self.failure is None:
                self.failure = code
            self.stop.set()
            procs = list(self.procs.values())
        for p in procs:
            if p.poll() is None:
                self._signal_group(p, signal.SIGTERM)

    def cleanup(self, reason: str) -> list[dict[str, Any]]:
        """Terminate every started stage group, bounded in time. Idempotent.

        Runs only in the main thread (signal handler or atexit).
        """
        if self.cleanup_started:
            return []
        self.cleanup_started = True
        with self.spawn_lock:
            if self.cleaned:
                return []
            self.cleaned = True
            self.stop.set()
            procs = dict(self.procs)
        live = {n: p for n, p in procs.items() if self._group_alive(p)}
        report: list[dict[str, Any]] = []
        if live:
            for p in live.values():
                self._signal_group(p, signal.SIGTERM)
            end = time.monotonic() + TERM_GRACE_S
            while time.monotonic() < end and any(self._group_alive(p) for p in live.values()):
                time.sleep(0.1)
            for n, p in live.items():
                alive = self._group_alive(p)
                if alive:
                    self._signal_group(p, signal.SIGKILL)
                report.append({"stage": n, "pgid": p.pid, "sigkill": alive})
            end = time.monotonic() + KILL_GRACE_S
            while time.monotonic() < end and any(self._group_alive(p) for p in live.values()):
                time.sleep(0.05)
            for rec in report:
                rec["survived"] = self._group_alive(live[rec["stage"]])
        self.log({"status": "stage_cleanup", "reason": reason,
                  "registered": len(procs), "groups": report})
        return report

    def _on_signal(self, signum: int, _frame: object) -> None:
        if self.cleanup_started:
            return
        name = signal.Signals(signum).name
        if self.failure is None:
            self.failure = 128 + signum
        self.log({"status": "signal", "signal": name})
        self.cleanup(f"signal {name}")
        raise SystemExit(128 + signum)

    def install_cleanup(self) -> None:
        atexit.register(self.cleanup, "atexit")
        for sig in CLEANUP_SIGNALS:
            signal.signal(sig, self._on_signal)

    def _command(self, stage: Stage, record: Path, out: Path) -> list[str]:
        name, _role, backends, script, args, _exp = stage
        if SELFTEST:
            return [PY, str(RUN / "test_driver_cleanup.py"), "--dummy-stage",
                    "--backends", backends, "--record", str(record),
                    "--pids", str(Path(SELFTEST) / f"{name}.pids"), *args]
        return [PY, WRAP, "--backends", backends, "--record", str(record), "--",
                str(RUN / script), *args, "--out", str(out)]

    def run_stage(self, stage: Stage, gpu: int) -> None:
        name, role, backends, _script, _args, exp = stage
        if self.stop.is_set():
            self.log({"stage": name, "gpu": gpu, "status": "skipped_after_stop"})
            return
        remaining = self.deadline - time.monotonic()
        if role == "optional" and remaining < OPTIONAL_MIN_S:
            self.log({"stage": name, "gpu": gpu, "status": "optional_skipped_budget",
                      "remaining_s": round(remaining)})
            return
        if remaining < MIN_START_S:
            self.log({"stage": name, "gpu": gpu, "status": "stop_budget",
                      "remaining_s": round(remaining)})
            self.fail(4)
            return
        env = {k: v for k, v in os.environ.items()
               if k != "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(RUN),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / "inductor_cache" / name),
                   TRITON_CACHE_DIR=str(RUN / "triton_cache" / name),
                   TORCH_LOGS="recompiles")
        record = RUN / f"{name}.backend.json"
        out = RUN / f"{name}.jsonl"
        cmd = self._command(stage, record, out)
        rc: int | str
        with open(RUN / f"{name}.log", "w") as fh:
            with self.spawn_lock:
                if self.cleaned or self.stop.is_set():
                    spawned = None
                else:
                    spawned = subprocess.Popen(
                        cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    self.procs[name] = spawned
            if spawned is None:
                self.log({"stage": name, "gpu": gpu, "status": "skipped_after_stop"})
                return
            p = spawned
            self.log({"stage": name, "gpu": gpu, "role": role, "backends": backends,
                      "cmd": cmd, "pgid": p.pid, "status": "launch",
                      "timeout_s": round(remaining)})
            try:
                rc = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self._signal_group(p, signal.SIGKILL)
                p.wait()
                rc = "timeout"
        if self.stop.is_set() and self.failure is not None and rc != 0:
            self.log({"stage": name, "gpu": gpu, "rc": rc, "status": "terminated_by_stop"})
            return
        errs: list[str] = []
        if record.exists():
            br = json.loads(record.read_text())
            want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
            if br.get("value_at_start") != want or br.get("value_at_end") != want:
                errs.append(f"backend record {br.get('value_at_start')}/"
                            f"{br.get('value_at_end')} != {want}")
        else:
            errs.append("no backend record")
        try:
            errs += judge(events(out), exp)
        except (KeyError, TypeError, ValueError) as exc:
            errs.append(f"judge error {type(exc).__name__}: {exc}")
        if rc != 0:
            errs.insert(0, f"rc={rc}")
        if errs:
            self.log({"stage": name, "gpu": gpu, "rc": rc, "errors": errs[:40],
                      "status": "FAIL_stop"})
            self.fail(4 if rc == "timeout" else 3)
            return
        self.log({"stage": name, "gpu": gpu, "rc": rc, "status": "pass"})

    def stream(self, stages: list[Stage], gpu: int) -> None:
        for stage in stages:
            self.run_stage(stage, gpu)

    def main(self) -> int:
        self.install_cleanup()
        self.log({"status": "driver_start", "budget_s": BUDGET_S,
                  "selftest": bool(SELFTEST)})
        threads = [threading.Thread(target=self.stream, args=(GPU0, 0), daemon=True),
                   threading.Thread(target=self.stream, args=(GPU1, 1), daemon=True)]
        for t in threads:
            t.start()
        # The main thread only waits, so the signal handler never interrupts a spawn.
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
        code = self.failure if self.failure is not None else 0
        self.log({"status": "driver_done", "exit": code,
                  "elapsed_s": round(time.monotonic() - self.t0, 1)})
        return code


if __name__ == "__main__":
    sys.exit(Driver().main())
