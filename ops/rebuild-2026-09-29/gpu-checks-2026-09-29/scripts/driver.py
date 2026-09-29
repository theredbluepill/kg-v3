"""GPU checks bundle driver (run statement run-statements/gpu-checks-bundle.md,
attempt 2 under its Amendment 1).

Phase 1 runs two independent streams in parallel, one per GPU:
  GPU 0: c3 full-model smoke  (mid/dense x ATEN/default)
  GPU 1: c2 trunk backward above the bound (ATEN), its default-backend
         control (expected to fail), then c1 fp32 reference (ATEN, default)
Phase 2 runs alone on GPU 0 (GPU 1 idle): c4 timing, mid then dense, ATEN.
Every stage is a fresh subprocess with its own Inductor/Triton cache (the two
c4 processes share one ATEN cache). The driver stops at the FIRST unexpected
failure (a non-control stage failing its pre-declared criteria, nonzero exit,
timeout, a wrong backend record, or an exception while the driver starts or
judges a stage): the other stream's running subprocess is terminated and
nothing further starts. Control stages never stop it.

Exit codes: 0 every stage ran and every non-control stage passed; 3 a
non-control stage failed; 4 the internal deadline ran out; 5 the driver raised
while starting or judging a stage (for example an unreadable backend record or
a failed process launch); 128 + signum the driver received SIGTERM, SIGINT or
SIGHUP (for example from launch.sh's outer `timeout`) and cleaned up.

Stage exceptions (third post-run revision, after Codex review
verify-merge-gpu-receipts-r1 finding 1): `stream` catches any exception from
a stage at the stream boundary, logs `driver_error`, records exit 5, and runs
the bounded cleanup below. The recorded attempts' driver let such an exception
end the Phase 1 worker thread silently, so Phase 2 still started and the
driver returned 0.

Stage cleanup (post-run revision, after Codex review verify-gpu-bundle-r1
finding 3; the recorded attempts ran the earlier driver, which lacked it):
every stage runs in its own session, so it is outside the process group that
launch.sh's `timeout` signals. On SIGTERM/SIGINT/SIGHUP and at interpreter exit
the driver therefore signals every started stage's process group itself:
SIGTERM, a bounded wait of TERM_GRACE_S, then SIGKILL and a further bounded
wait of KILL_GRACE_S. The total stays below launch.sh's `timeout -k 20` grace,
after which `timeout` would SIGKILL the driver and cleanup could not run.

Spawn/registration is atomic with respect to those signals (second post-run
revision, after Codex review verify-gpu-bundle-r2 finding 3). A stage that
exists but is not yet in `procs` would be missed by cleanup, so while the main
thread (Phase 2) starts and registers a stage, the signal handler only records
the signal; the driver handles it as soon as the stage is registered. Worker
threads (Phase 1) need no deferral: the handler runs in the main thread, and
the worker registers under the same lock that cleanup takes, killing the stage
itself if cleanup already ran. Signals are deferred rather than blocked with
`signal.pthread_sigmask`, because a child started while they are blocked
inherits the blocked mask and would then ignore cleanup's SIGTERM.
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
ROOT = Path("/workspace/kg-v3-rebuild")
PY = str(ROOT / ".venv/bin/python")
WRAP = str(RUN / "gemm_backend_wrap.py")
# Attempt 2 (Amendment 1): attempt 1's driver used 55.4 s of the 60-min
# aggregate limit; `timeout -k 20 3540` is the backstop.
BUDGET_S = 55 * 60
MIN_START_S = 90
CONTROL_CAP_S = 600
REL_MAX = 0.05
L512 = 2**31 // 512  # 4,194,304 packed tokens x 512 = 2**31
TERM_GRACE_S = 8.0
KILL_GRACE_S = 4.0
CLEANUP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

# name, role, backends, cache, script, args
Stage = tuple[str, str, str, str, str, list[str]]
C2_POINTS = f"packed:{L512 + 1};packed:4198400"
GPU0_PHASE1: list[Stage] = [
    (f"c3_{d}_{b.lower()}", "correct", b, f"c3_{d}_{b.lower()}", "c3_smoke.py",
     ["--density", d, "--case", f"c3_{d}_{b.lower()}"])
    for b in ("ATEN", "default") for d in ("mid", "dense")
]
GPU1_PHASE1: list[Stage] = [
    ("c2_aten_bwd", "correct", "ATEN", "c2_aten_bwd", "c2_trunk_bwd.py",
     ["--bypass-guard", "--depth", "1", "--case", "c2_aten_bwd", "--points", C2_POINTS]),
    ("c2_ctl_default_bwd", "control", "default", "c2_ctl_default_bwd",
     "c2_trunk_bwd.py",
     ["--bypass-guard", "--depth", "1", "--case", "c2_ctl_default_bwd",
      "--points", C2_POINTS]),
    ("c1_aten", "correct", "ATEN", "c1_aten", "c1_fp32_ref.py",
     ["--densities", "mid,dense,mixed", "--case", "c1_aten"]),
    ("c1_default", "correct", "default", "c1_default", "c1_fp32_ref.py",
     ["--densities", "mid,dense,mixed", "--case", "c1_default"]),
]
GPU0_PHASE2: list[Stage] = [
    (f"c4_{d}_aten", "correct", "ATEN", "c4_aten", "c4_timing.py",
     ["--density", d, "--case", f"c4_{d}_aten"])
    for d in ("mid", "dense")
]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def judge_c1(ev: list[dict[str, Any]]) -> list[str]:
    errs = []
    res = [e for e in ev if e.get("event") == "result"]
    if len(res) != 3:
        errs.append(f"{len(res)} of 3 density results")
    for r in res:
        tag = r["density"]
        if r["ref_nonfinite"]:
            errs.append(f"{tag}: fp32 reference non-finite {r['ref_nonfinite']}")
        for p, m in r["vs_fp32"].items():
            if m["nonfinite"]:
                errs.append(f"{tag}: {p} non-finite {m['nonfinite']}")
        for p in ("compiled", "eager"):
            if r[f"{p}_pack_calls"] != [r["present_tokens"]]:
                errs.append(f"{tag}: {p} pack calls {r[f'{p}_pack_calls']}")
            if not (r[f"{p}_use_flash"] and all(r[f"{p}_use_flash"])):
                errs.append(f"{tag}: {p} use_flash {r[f'{p}_use_flash']}")
    return errs


def _c2_point(r: dict[str, Any]) -> list[str]:
    errs = []
    if r.get("compiled", {}).get("status") != "ok" or r.get("eager", {}).get("status") != "ok":
        return [f"compiled={r.get('compiled', {}).get('status')} "
                f"eager={r.get('eager', {}).get('status')}"]
    if r["compiled"]["trunk_call_M"] != [r["packed_tokens"]]:
        errs.append(f"trunk_call_M {r['compiled']['trunk_call_M']}")
    o = r["out_compiled_vs_eager"]
    if o["wrong_tokens"] or o["nonfinite"]:
        errs.append(f"out wrong={o['wrong_tokens']} nonfinite={o['nonfinite']}")
    d = r["dx_compiled_vs_eager"]
    if (d["nonfinite"] or d["tokens_rel_gt_0.5"] or d["masked_nonzero"]
            or not (d["rel_max"] is not None and d["rel_max"] <= REL_MAX)):
        errs.append(f"dx nonfinite={d['nonfinite']} tok>0.5={d['tokens_rel_gt_0.5']} "
                    f"masked_nonzero={d['masked_nonzero']} rel_max={d['rel_max']}")
    params = r["params_compiled_vs_eager"]
    for name, m in params.items():
        if name == "_missing":
            errs.append(f"param grads missing {m}")
        elif name.endswith("attn.k.bias"):
            # Amendment 1: the key-bias gradient is analytically zero (softmax
            # shift invariance), so compiled and eager values and their
            # difference are judged against the sibling query-bias scale.
            scale = params[name[: -len("k.bias")] + "q.bias"]["ref_max_abs"]
            worst = max(m["max_abs_diff"], m["out_max_abs"], m["ref_max_abs"])
            if m["nonfinite"] or not worst <= REL_MAX * scale:
                errs.append(f"grad {name} max(|diff|,|c|,|e|)={worst} > "
                            f"{REL_MAX} x |q.bias grad| {scale} "
                            f"nonfinite={m['nonfinite']}")
        elif m["nonfinite"] or not m["rel_max"] <= REL_MAX:
            errs.append(f"grad {name} rel_max={m['rel_max']} nonfinite={m['nonfinite']}")
    return errs


def judge_c2(ev: list[dict[str, Any]]) -> list[str]:
    res = [e for e in ev if e.get("event") == "result"]
    errs = []
    for r in res:
        errs += [f"{r['label']} {r['spec']}: {e}" for e in _c2_point(r)]
    n = sum(1 for r in res if r["label"] == "target")
    if n != 2:
        errs.append(f"{n} of 2 target points produced a result")
    return errs


def judge_c3(ev: list[dict[str, Any]]) -> list[str]:
    res = [e for e in ev if e.get("event") == "result"]
    if len(res) != 1:
        return [f"{len(res)} result records"]
    r = res[0]
    if "error" in r:
        return [f"error: {r['error'][:400]}"]
    errs = []
    if not all(v == "ok" for v in r["steps"].values()) or len(r["steps"]) != 8:
        errs.append(f"steps {r['steps']}")
    if r["sample"]["log_probs"]["nonfinite"]:
        errs.append("sample log-probs non-finite")
    for k in ("replay_256", "replay_1024"):
        lr = r[k]["logratio_per_row"]
        if lr["nonfinite"] or r[k]["new_logp_nonfinite"] or not abs(lr["mean"]) <= 0.05:
            errs.append(f"{k} logratio mean={lr['mean']} nonfinite={lr['nonfinite']}")
    for k in ("teacher_self_combined", "teacher_self_cached"):
        kl = r[k]["kl_per_row"]
        if (r[k]["kl_nonfinite"] or not kl["mean"] <= 1e-3 or not kl["max"] <= 1e-2
                or not r[k]["kl_event_min"] >= -1e-4):
            errs.append(f"{k} {kl} event_min={r[k]['kl_event_min']}")
    for k, self_k in (("teacher_perturbed_combined", "teacher_self_combined"),
                      ("teacher_perturbed_cached", "teacher_self_cached")):
        kl = r[k]["kl_per_row"]
        self_mean = r[self_k]["kl_per_row"]["mean"]
        if (r[k]["kl_nonfinite"] or not kl["mean"] > 0 or not kl["mean"] > self_mean
                or not r[k]["kl_event_min"] >= -1e-4):
            errs.append(f"{k} {kl} self_mean={self_mean} event_min={r[k]['kl_event_min']}")
    vals = [("sample", r["sample"]["values"]), ("replay_256", r["replay_256"]["values"]),
            ("replay_1024", r["replay_1024"]["values"]),
            ("compute_value_256", r["compute_value_256"]["values"])]
    # All four teacher paths (post-run revision, after Codex review
    # verify-merge-gpu-receipts-r1 finding 2; the as-run judge checked only
    # teacher_self_combined and teacher_perturbed_cached).
    vals += [(k, r[k]["student_values"]) for k in ("teacher_self_combined",
                                                   "teacher_self_cached",
                                                   "teacher_perturbed_combined",
                                                   "teacher_perturbed_cached")]
    for k, v in vals:
        if v["nonfinite"] or v["min"] < -1 - 1e-6 or v["max"] > 1 + 1e-6:
            errs.append(f"{k} values {v}")
    lb = r["loss_backward_1024"]
    if lb["grad_nonfinite"] or lb["loss"] != lb["loss"] or abs(lb["loss"]) == float("inf"):
        errs.append(f"loss_backward {lb}")
    if not r["use_flash_attn_all_true"] or r["compiled_trunk_calls"] <= 0:
        errs.append(f"flash={r['use_flash_attn_all_true']} "
                    f"compiled_calls={r['compiled_trunk_calls']}")
    return errs


def judge_c4(ev: list[dict[str, Any]]) -> list[str]:
    errs = []
    timings = [e for e in ev if e.get("event") == "timing"]
    if len(timings) != 12:
        errs.append(f"{len(timings)} of 12 timings")
    for t in timings:
        if not t["use_flash_attn_all_true"]:
            errs.append(f"{t['workload']} flash not all true")
        if t["pack_calls_per_iter"] != t["expected_chunks"]:
            errs.append(f"{t['workload']} chunks {t['pack_calls_per_iter']} != "
                        f"{t['expected_chunks']}")
        if "_B_" in t["workload"] and not t.get("last_scalar_finite"):
            errs.append(f"{t['workload']} loss not finite")
    if not any(e.get("event") == "derived" for e in ev):
        errs.append("no derived record")
    return errs


JUDGES = {"c1": judge_c1, "c2": judge_c2, "c3": judge_c3, "c4": judge_c4}


class Driver:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.deadline = self.t0 + BUDGET_S
        self.stop = threading.Event()
        # Reentrant: the signal handler runs in the main thread and may
        # interrupt it while it already holds the lock (in log or fail).
        self.lock = threading.RLock()
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.failure: int | None = None
        self.cleaned = False
        # Main-thread spawn window (see _spawn); only the main thread, where
        # the signal handler also runs, reads or writes these two.
        self.defer_signals = False
        self.pending_signal: int | None = None
        self.log_fh = open(RUN / "driver.jsonl", "a")

    def log(self, rec: dict[str, Any]) -> None:
        rec = {"utc": now(), "t": round(time.monotonic() - self.t0, 1), **rec}
        with self.lock:
            self.log_fh.write(json.dumps(rec) + "\n")
            self.log_fh.flush()
            print(json.dumps(rec), flush=True)

    def fail(self, code: int) -> None:
        with self.lock:
            if self.failure is None:
                self.failure = code
            self.stop.set()
            for name, p in self.procs.items():
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    @staticmethod
    def _group_alive(p: subprocess.Popen[bytes]) -> bool:
        # Reap the leader if it has exited (so it is not counted as a zombie
        # member), then probe the whole group: grandchildren may outlive it.
        p.poll()
        try:
            os.killpg(p.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _signal_group(p: subprocess.Popen[bytes], sig: int) -> None:
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    def cleanup(self, reason: str) -> list[dict[str, Any]]:
        """Terminate every started stage's process group, bounded in time.

        Signals each group whether or not its leader is still running, because
        a stage's own children stay in its group after the leader exits.
        Idempotent. Returns one record per group that was still alive.
        """
        with self.lock:
            if self.cleaned:
                return []
            self.cleaned = True
            self.stop.set()
            procs = dict(self.procs)
        live = {name: p for name, p in procs.items() if self._group_alive(p)}
        report: list[dict[str, Any]] = []
        if not live:
            return report
        for p in live.values():
            self._signal_group(p, signal.SIGTERM)
        end = time.monotonic() + TERM_GRACE_S
        while time.monotonic() < end and any(self._group_alive(p) for p in live.values()):
            time.sleep(0.1)
        for name, p in live.items():
            killed = self._group_alive(p)
            if killed:
                self._signal_group(p, signal.SIGKILL)
            report.append({"stage": name, "pgid": p.pid, "sigkill": killed})
        end = time.monotonic() + KILL_GRACE_S
        while time.monotonic() < end and any(self._group_alive(p) for p in live.values()):
            time.sleep(0.05)
        for rec in report:
            rec["survived"] = self._group_alive(live[rec["stage"]])
        self.log({"status": "stage_cleanup", "reason": reason, "groups": report})
        return report

    def _on_signal(self, signum: int, _frame: object) -> None:
        if self.cleaned:
            return
        if self.defer_signals:
            # A stage may exist that cleanup cannot see yet; _spawn calls
            # back once it is registered.
            if self.pending_signal is None:
                self.pending_signal = signum
            return
        name = signal.Signals(signum).name
        with self.lock:
            if self.failure is None:
                self.failure = 128 + signum
        self.log({"status": "signal", "signal": name})
        self.cleanup(f"signal {name}")
        raise SystemExit(128 + signum)

    def install_cleanup(self) -> None:
        atexit.register(self.cleanup, "atexit")
        for sig in CLEANUP_SIGNALS:
            signal.signal(sig, self._on_signal)

    def _spawn(self, name: str, cmd: list[str], env: dict[str, str],
               fh: Any) -> subprocess.Popen[bytes]:
        """Start a stage in its own session and register it for cleanup.

        No cleanup signal can unwind the driver between the child existing and
        its registration: in the main thread, the handler defers until the
        stage is in `procs`, including while `Popen` itself is executing.
        """
        main = threading.current_thread() is threading.main_thread()
        if main:
            self.defer_signals = True
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.procs[name] = p
                late = self.cleaned
        finally:
            if main:
                self.defer_signals = False
                pending, self.pending_signal = self.pending_signal, None
                if pending is not None:
                    self._on_signal(pending, None)
        if late:
            # Cleanup already ran and missed this group; nothing in it
            # has done work yet, so kill it outright.
            self._signal_group(p, signal.SIGKILL)
        return p

    def run_stage(self, stage: Stage, gpu: int) -> None:
        name, role, backends, cache, script, args = stage
        if self.stop.is_set():
            self.log({"stage": name, "gpu": gpu, "status": "skipped_after_stop"})
            return
        remaining = self.deadline - time.monotonic()
        if remaining < MIN_START_S:
            self.log({"stage": name, "gpu": gpu, "status": "stop_budget",
                      "remaining_s": round(remaining)})
            self.fail(4)
            return
        timeout = min(remaining, CONTROL_CAP_S) if role == "control" else remaining
        env = {k: v for k, v in os.environ.items()
               if k != "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(RUN),
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / "inductor_cache" / cache),
                   TRITON_CACHE_DIR=str(RUN / "triton_cache" / cache),
                   TORCH_LOGS="recompiles")
        record = RUN / f"{name}.backend.json"
        out = RUN / f"{name}.jsonl"
        cmd = [PY, WRAP, "--backends", backends, "--record", str(record), "--",
               str(RUN / script), *args, "--out", str(out)]
        self.log({"stage": name, "gpu": gpu, "role": role, "backends": backends,
                  "cmd": cmd, "status": "launch", "timeout_s": round(timeout)})
        rc: int | str
        with open(RUN / f"{name}.log", "w") as fh:
            p = self._spawn(name, cmd, env, fh)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                rc = "timeout"
        if self.stop.is_set() and role != "control" and self.failure is not None \
                and rc not in (0,):
            self.log({"stage": name, "gpu": gpu, "rc": rc,
                      "status": "terminated_by_stop"})
            return
        rec_errs = []
        if record.exists():
            br = json.loads(record.read_text())
            want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
            if br.get("value_at_start") != want or br.get("value_at_end") != want:
                rec_errs.append(f"backend record {br.get('value_at_start')}/"
                                f"{br.get('value_at_end')} != {want}")
        else:
            rec_errs.append("no backend record")
        try:
            errs = JUDGES[name[:2]](events(out))
        except (KeyError, TypeError, ValueError) as exc:
            errs = [f"judge error {type(exc).__name__}: {exc}"]
        if role == "control":
            reproduced = rc != 0 or bool(errs)
            self.log({"stage": name, "gpu": gpu, "role": role, "rc": rc,
                      "findings": (errs + rec_errs)[:40],
                      "status": "control_reproduced" if reproduced
                      else "control_not_reproduced"})
            return
        errs = rec_errs + errs
        if rc != 0:
            errs.insert(0, f"rc={rc}")
        if errs:
            self.log({"stage": name, "gpu": gpu, "role": role, "rc": rc,
                      "errors": errs[:40], "status": "FAIL_stop"})
            self.fail(4 if rc == "timeout" else 3)
            return
        self.log({"stage": name, "gpu": gpu, "role": role, "rc": rc, "status": "pass"})

    def stream(self, stages: list[Stage], gpu: int) -> None:
        for stage in stages:
            try:
                self.run_stage(stage, gpu)
            except Exception as exc:  # noqa: BLE001 - the stream boundary
                # An exception here would otherwise end a Phase 1 worker
                # thread silently (Phase 2 would still start, exit 0) or
                # unwind main() in Phase 2. Stop, and terminate every group.
                self.log({"stage": stage[0], "gpu": gpu, "status": "driver_error",
                          "error": f"{type(exc).__name__}: {exc}"[:400]})
                self.fail(5)
                self.cleanup(f"driver_error {stage[0]}")
                return

    def main(self) -> int:
        self.install_cleanup()
        self.log({"status": "driver_start", "budget_s": BUDGET_S})
        threads = [threading.Thread(target=self.stream, args=(GPU0_PHASE1, 0)),
                   threading.Thread(target=self.stream, args=(GPU1_PHASE1, 1))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.log({"status": "phase1_done", "stopped": self.stop.is_set()})
        self.stream(GPU0_PHASE2, 0)
        code = self.failure if self.failure is not None else 0
        self.log({"status": "driver_done", "exit": code,
                  "elapsed_s": round(time.monotonic() - self.t0, 1)})
        return code


if __name__ == "__main__":
    sys.exit(Driver().main())
