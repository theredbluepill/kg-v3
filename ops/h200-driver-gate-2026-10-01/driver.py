"""H200 ATEN-only correctness re-probe driver (run statement run-statement.md).

Adapted from the 2026-09-29 A/B driver (sha256 recorded in the run statement):
ROOT/PY point at /root/kg-v3; the run dir is /root/aten-h200; the trunk probe
is probe_trunk_07c8fc9.py with --stack-check-bypass; the four timing stages and
the default-backend trunk control are dropped (07c8fc9's claim would force
"ATEN" in that control anyway); the budget is 25 min. Judging is unchanged.

Original docstring follows.

ATEN-only GEMM A/B driver (run statement run-statements/aten-gemm-ab.md).

Runs every stage in a fresh subprocess on GPU 0, in the fixed order below,
with a hard aggregate deadline. Stops at the FIRST failure of an ATEN-only
correctness stage or of a timing stage. The default-backend control stages
(expected to corrupt or fault) run last, each in its own subprocess; their
outcome is recorded as reproduced / not reproduced and never stops the driver.

Exit codes: 0 all stages ran and every non-control stage passed; 3 a
non-control stage failed (driver stopped there); 4 the budget ran out.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = Path("/root/kg-v3")
PY = str(ROOT / ".venv/bin/python")
WRAP = str(RUN / "gemm_backend_wrap.py")
BUDGET_S = 25 * 60  # the outer `timeout 1560` is the backstop
MIN_START_S = 90
PARAM_GRAD_REL_MAX = 0.05

L512 = 2**31 // 512
L768 = 2**31 // 768

# name, role, backends, cache group, script, script args (out path appended)
STAGES: list[tuple[str, str, str, str, str, list[str]]] = [
    ("aten_trunk_packed_bypass", "correct", "ATEN", "aten_trunk_packed_bypass",
     "probe_trunk_07c8fc9.py",
     ["--bypass-guard", "--stack-check-bypass", "--case", "trunk_packed_bypass",
      "--points", "packed:4194305;packed:4198400;dense:5916;dense:11830"]),
    ("aten_lin_768_256", "correct", "ATEN", "aten_lin_768_256", "probe_linear.py",
     ["--case", "lin:768:256", "--ms", str(L768 + 1)]),
    ("aten_lin_512_256", "correct", "ATEN", "aten_lin_512_256", "probe_linear.py",
     ["--case", "lin:512:256", "--ms", f"{L512 + 4097},{2 * L512}"]),
    ("aten_mlpbwd_256_512_256", "correct", "ATEN", "aten_mlpbwd_256_512_256",
     "probe_linear.py", ["--case", "mlpbwd:256:512:256", "--ms", str(L512 + 1)]),
    ("ctl_default_lin_768_256", "control", "default", "ctl_default_lin_768_256",
     "probe_linear.py", ["--case", "lin:768:256", "--ms", str(L768 + 1)]),
]

EXPECTED_CHUNKS = {"mid": {"A": 1, "B": 1, "C": 2, "D": 1},
                   "dense": {"A": 1, "B": 1, "C": 3, "D": 1}}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def lin_ok(r: dict) -> list[str]:
    errs = []
    if "fwd" in r:  # forward + backward case
        for part in ("fwd", "dx"):
            if r[part]["bad_rows"] or r[part]["nonfinite"]:
                errs.append(f"{part} bad_rows={r[part]['bad_rows']} "
                            f"nonfinite={r[part]['nonfinite']}")
        for name, rel in r["param_grad_rel_max"].items():
            if not rel <= PARAM_GRAD_REL_MAX:
                errs.append(f"grad {name} rel {rel}")
    else:
        if r["bad_rows"] or r["nonfinite"] or r["ref_clobbered"]:
            errs.append(f"bad_rows={r['bad_rows']} nonfinite={r['nonfinite']} "
                        f"ref_clobbered={r['ref_clobbered']}")
    if r["input_clobbered"]:
        errs.append("input_clobbered")
    return errs


def trunk_ok(r: dict) -> list[str]:
    errs = []
    if r["eager"] != "ok" or r["compiled"] != "ok":
        errs.append(f"eager={r['eager']} compiled={r['compiled']}")
        return errs
    if r["tokens_diff_gt_0.25"] or r["nonfinite"]:
        errs.append(f"tokens_diff_gt_0.25={r['tokens_diff_gt_0.25']} "
                    f"nonfinite={r['nonfinite']}")
    if r["trunk_call_M"] != [r["packed_tokens"]]:
        errs.append(f"trunk_call_M {r['trunk_call_M']} != [{r['packed_tokens']}]")
    return errs


def judge_probe(script: str, out: Path, n_points: int) -> tuple[list[str], int]:
    ev = events(out)
    results = [e for e in ev if e.get("event") == "result"]
    errs: list[str] = []
    for r in results:
        bad = trunk_ok(r) if script.startswith("probe_trunk") else lin_ok(r)
        errs += [f"{r['label']} {r.get('spec', r.get('M'))}: {b}" for b in bad]
    targets = [r for r in results if r["label"] == "target"]
    if len(targets) != n_points:
        errs.append(f"{len(targets)} of {n_points} target points produced a result")
    return errs, len(targets)


def judge_bench(outdir: Path, density: str) -> list[str]:
    path = outdir / f"results_{density}.json"
    if not path.exists():
        return [f"missing {path}"]
    res = json.loads(path.read_text())["results"]
    errs = []
    for k, want in EXPECTED_CHUNKS[density].items():
        got = res[k]["pack_sequence_calls_per_iter"]
        if got != want:
            errs.append(f"{k} chunks/iter {got} != {want}")
        if not res[k]["use_flash_attn_all_true"]:
            errs.append(f"{k} use_flash_attn not all true")
    return errs


def main() -> int:
    t0 = time.monotonic()
    deadline = t0 + BUDGET_S
    log_fh = open(RUN / "driver.jsonl", "a")

    def log(rec: dict) -> None:
        rec = {"utc": now(), "t": round(time.monotonic() - t0, 1), **rec}
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()
        print(json.dumps(rec), flush=True)

    log({"status": "driver_start", "budget_s": BUDGET_S})
    for name, role, backends, cache, script, args in STAGES:
        remaining = deadline - time.monotonic()
        if remaining < MIN_START_S:
            log({"stage": name, "status": "stop_budget", "remaining_s": round(remaining)})
            return 4
        env = {k: v for k, v in os.environ.items()
               if k != "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"}
        env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                   TORCHINDUCTOR_CACHE_DIR=str(RUN / "inductor_cache" / cache),
                   TRITON_CACHE_DIR=str(RUN / "triton_cache" / cache),
                   TORCH_LOGS="recompiles")
        record = RUN / f"{name}.backend.json"
        if role == "bench":
            outdir = RUN / f"bench_{backends.lower()}"
            outdir.mkdir(exist_ok=True)
            sargs = [*args, "--out", str(outdir)]
        else:
            out = RUN / f"{name}.jsonl"
            sargs = [*args, "--out", str(out)]
        cmd = [PY, WRAP, "--backends", backends, "--record", str(record), "--",
               str(RUN / script), *sargs]
        log({"stage": name, "role": role, "backends": backends, "cmd": cmd,
             "status": "launch", "timeout_s": round(remaining)})
        with open(RUN / f"{name}.log", "w") as fh:
            try:
                rc: int | str = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                    timeout=remaining).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rec_ok = []
        if record.exists():
            br = json.loads(record.read_text())
            want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
            if br.get("value_at_start") != want or br.get("value_at_end") != want:
                rec_ok.append(f"backend record {br.get('value_at_start')}/"
                              f"{br.get('value_at_end')} != {want}")
        else:
            rec_ok.append("no backend record")
        if role == "bench":
            errs = judge_bench(RUN / f"bench_{backends.lower()}", args[1])
            n_done = None
        else:
            n_points = len(args[-1].replace(",", ";").split(";"))
            errs, n_done = judge_probe(script, RUN / f"{name}.jsonl", n_points)
        if rc == "timeout":
            log({"stage": name, "status": "stop_budget_timeout"})
            return 4
        if role == "control":
            reproduced = rc != 0 or bool(errs)
            log({"stage": name, "role": role, "rc": rc, "targets_done": n_done,
                 "findings": errs + rec_ok,
                 "status": "control_reproduced" if reproduced
                 else "control_not_reproduced"})
            continue
        errs = rec_ok + errs
        if rc != 0:
            errs.insert(0, f"rc={rc}")
        if errs:
            log({"stage": name, "role": role, "rc": rc, "errors": errs,
                 "status": "FAIL_stop"})
            return 3
        log({"stage": name, "role": role, "rc": rc, "targets_done": n_done,
             "status": "pass"})
    log({"status": "driver_done", "elapsed_s": round(time.monotonic() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
