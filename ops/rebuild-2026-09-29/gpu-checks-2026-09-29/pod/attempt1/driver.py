"""GPU checks bundle driver (run statement run-statements/gpu-checks-bundle.md).

Phase 1 runs two independent streams in parallel, one per GPU:
  GPU 0: c3 full-model smoke  (mid/dense x ATEN/default)
  GPU 1: c2 trunk backward above the bound (ATEN), its default-backend
         control (expected to fail), then c1 fp32 reference (ATEN, default)
Phase 2 runs alone on GPU 0 (GPU 1 idle): c4 timing, mid then dense, ATEN.
Every stage is a fresh subprocess with its own Inductor/Triton cache (the two
c4 processes share one ATEN cache). The driver stops at the FIRST unexpected
failure (a non-control stage failing its pre-declared criteria, nonzero exit,
timeout or a wrong backend record): the other stream's running subprocess is
terminated and nothing further starts. Control stages never stop it.

Exit codes: 0 every stage ran and every non-control stage passed; 3 a
non-control stage failed; 4 the internal deadline ran out.
"""

from __future__ import annotations

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
BUDGET_S = 55 * 60  # internal deadline; `timeout -k 20 3600` is the backstop
MIN_START_S = 90
CONTROL_CAP_S = 600
REL_MAX = 0.05
L512 = 2**31 // 512  # 4,194,304 packed tokens x 512 = 2**31

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
    for name, m in r["params_compiled_vs_eager"].items():
        if name == "_missing":
            errs.append(f"param grads missing {m}")
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
    vals = [r["sample"]["values"], r["replay_256"]["values"], r["replay_1024"]["values"],
            r["compute_value_256"]["values"]]
    vals += [r[k]["student_values"] for k in ("teacher_self_combined",
                                              "teacher_perturbed_cached")]
    for v in vals:
        if v["nonfinite"] or v["min"] < -1 - 1e-6 or v["max"] > 1 + 1e-6:
            errs.append(f"values {v}")
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
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.failure: int | None = None
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
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.procs[name] = p
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
            self.run_stage(stage, gpu)

    def main(self) -> int:
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
