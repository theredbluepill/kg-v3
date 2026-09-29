"""Local test: driver failure propagation, judge guards and the launcher idle gate.

Regressions for Codex review verify-merge-gpu-receipts-r1 findings 1-3. No
GPU; the driver scenarios run dummy stages (Python sleep subprocesses, no
torch, no /workspace paths). The C1 and C2 checks import c1_fp32_ref and
c2_trunk_bwd, so they need the repository's uv environment (torch, pydantic);
the launcher check needs bash.

Driver scenarios (each runs driver.main() in a fresh child process, with
PY/WRAP/RUN/ROOT pointed at a scratch directory and every judge returning no
errors):
  phase1_malformed_record  GPU 0's Phase 1 stage exits 0 but writes a
      malformed backend JSON record while GPU 1's stage is still running.
  phase1_spawn_error       starting GPU 0's Phase 1 stage raises OSError
      (injected into subprocess.Popen) while GPU 1's stage is running; that
      stage and its grandchild ignore SIGTERM, so only the bounded cleanup's
      SIGKILL escalation can end them.
  phase2_malformed_record  Phase 1 passes; the first Phase 2 stage (driver
      main thread) writes a malformed backend record.
  complete                 every stage is quick and valid (exit 0 baseline).
Each failing scenario must exit 5, log a driver_error for the faulty stage,
never launch a later stage, and leave no recorded stage process alive. In the
Phase 1 scenarios GPU 1's running stage must end: terminated_by_stop, or a
stage_cleanup record naming it; phase1_spawn_error must show the latter with
SIGKILL escalation.

Judge checks:
  c3_values  every retained attempt 2 C3 record passes judge_c3; for each of
      the four teacher paths, student values with a non-finite count or a
      value outside [-1, 1] make judge_c3 report an error naming that path.
  c1_verdict every retained attempt 2 C1 verdict regenerates identically from
      its per-path metrics; a single path with positive error while the other
      two are exactly zero (median 0) is flagged path_specific; three zero
      paths stay similar.

Regressions for Codex review verify-merge-gpu-receipts-r2 (both findings):
  launcher_idle launch.sh's gpu_idle, extracted verbatim and run under bash
      with nvidia-smi stubbed. Only the idle case (both queries succeed, no
      compute apps, utilization 0) may pass; busy apps, busy utilization, a
      failed compute-apps query (empty output), a failed utilization query
      (with or without a "0" printed before the failure) and both queries
      failing must each fail the gate.
  c2_nonfinite  c2_trunk_bwd's comparators over 513 valid rows (ROW_CHUNK 512,
      so the cases span both chunks). A non-finite value in either operand
      (compiled or eager reference) at a valid position is counted in
      `nonfinite`, and judge_c2 rejects the retained attempt 2 record carrying
      those metrics, including a reference NaN in the same chunk as a 10 %
      compiled-gradient error; the unmodified all-equal case passes.

Usage: uv run python test_driver_guards.py   (exit 0 when every check passes)
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
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
ATTEMPT2 = HERE.parent / "pod" / "attempt2"
CHILD_WALL_S = 60.0
TEACHER_PATHS = ("teacher_self_combined", "teacher_self_cached",
                 "teacher_perturbed_combined", "teacher_perturbed_cached")

DUMMY_STAGE = r'''
import json, os, signal, subprocess, sys, time
argv = sys.argv[1:]
backends = argv[argv.index("--backends") + 1]
record = argv[argv.index("--record") + 1]
mode = argv[argv.index("--mode") + 1]
if mode == "long_ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
want = "ATEN" if backends == "ATEN" else "ATEN,TRITON,CPP"
with open(record, "w") as fh:
    if mode == "bad_record":
        fh.write("{INVALID")
    else:
        json.dump({"value_at_start": want, "value_at_end": want}, fh)
if mode in ("quick", "bad_record"):
    time.sleep(0.3)
    sys.exit(0)
# The grandchild inherits SIG_IGN for SIGTERM in long_ignore_term mode.
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
piddir = os.path.join(os.path.dirname(record), "pids")
os.makedirs(piddir, exist_ok=True)
for role, pid in (("leader", os.getpid()), ("grandchild", gc.pid)):
    open(os.path.join(piddir, f"{role}-{pid}"), "w").close()
time.sleep(300)
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage(name: str, mode: str) -> tuple[Any, ...]:
    return (name, "correct", "ATEN", name, "dummy.py", ["--mode", mode])


SCENARIOS: dict[str, dict[str, Any]] = {
    "phase1_malformed_record": {
        "gpu0": [stage("c3_bad", "bad_record"), stage("c3_after", "quick")],
        "gpu1": [stage("c2_long", "long"), stage("c1_after", "quick")],
        "phase2": [stage("c4_after", "quick")],
        "faulty": "c3_bad", "running": "c2_long",
        "never": ["c3_after", "c1_after", "c4_after"],
    },
    "phase1_spawn_error": {
        "gpu0": [stage("c3_spawn_fail", "quick"), stage("c3_after", "quick")],
        "gpu1": [stage("c2_long", "long_ignore_term"), stage("c1_after", "quick")],
        "phase2": [stage("c4_after", "quick")],
        "faulty": "c3_spawn_fail", "running": "c2_long", "need_sigkill": True,
        "never": ["c3_after", "c1_after", "c4_after"],
    },
    "phase2_malformed_record": {
        "gpu0": [stage("c3_quick", "quick")],
        "gpu1": [stage("c2_quick", "quick")],
        "phase2": [stage("c4_bad", "bad_record"), stage("c4_after", "quick")],
        "faulty": "c4_bad", "running": None, "never": ["c4_after"],
    },
    "complete": {
        "gpu0": [stage("c3_quick", "quick")],
        "gpu1": [stage("c2_quick", "quick")],
        "phase2": [stage("c4_quick", "quick")],
        "faulty": None, "running": None, "never": [],
    },
}


def child(scenario: str, workdir: str) -> None:
    """Run inside a fresh process: patch the driver and run its main."""
    drv = load("gpu_checks_driver", DRIVER)
    wd = Path(workdir)
    drv.RUN = wd
    drv.ROOT = wd
    drv.PY = sys.executable
    drv.WRAP = str(wd / "dummy_stage.py")
    for key in list(drv.JUDGES):
        drv.JUDGES[key] = lambda ev: []
    sc = SCENARIOS[scenario]
    drv.GPU0_PHASE1, drv.GPU1_PHASE1, drv.GPU0_PHASE2 = sc["gpu0"], sc["gpu1"], sc["phase2"]
    if scenario == "phase1_spawn_error":
        real_popen = subprocess.Popen

        def failing_popen(args: Any, *a: Any, **kw: Any) -> Any:
            if any("c3_spawn_fail" in str(x) for x in args):
                raise OSError("injected spawn failure")
            # GPU 1's long stage must be running when GPU 0's spawn fails.
            return real_popen(args, *a, **kw)

        drv.subprocess.Popen = failing_popen
        drv.GPU0_PHASE1 = [("c3_wait", "correct", "ATEN", "c3_wait", "dummy.py",
                            ["--mode", "quick"]), *sc["gpu0"]]
    sys.exit(drv.Driver().main())


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_driver_scenario(scenario: str) -> dict[str, Any]:
    sc = SCENARIOS[scenario]
    wd = Path(tempfile.mkdtemp(prefix=f"kgv3-guards-{scenario}-"))
    (wd / "dummy_stage.py").write_text(DUMMY_STAGE)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", scenario, str(wd)]
    t0 = time.monotonic()
    with open(wd / "driver.out", "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc: int | str = p.wait(timeout=CHILD_WALL_S)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "wall_timeout"
    elapsed = round(time.monotonic() - t0, 2)
    pids = [int(f.name.split("-")[1]) for f in sorted(wd.glob("pids/*"))]
    end = time.monotonic() + 3.0
    while time.monotonic() < end and any(pid_alive(x) for x in pids):
        time.sleep(0.1)
    alive = [x for x in pids if pid_alive(x)]
    for x in alive:
        try:
            os.kill(x, signal.SIGKILL)
        except ProcessLookupError:
            pass
    records = [json.loads(line) for line in (wd / "driver.jsonl").read_text().splitlines()
               if line] if (wd / "driver.jsonl").exists() else []
    launched = [r["stage"] for r in records if r.get("status") == "launch"]
    errors = [r.get("stage") for r in records if r.get("status") == "driver_error"]
    groups = [g for r in records if r.get("status") == "stage_cleanup" for g in r["groups"]]
    cleaned = [g["stage"] for g in groups]
    stopped = [r["stage"] for r in records if r.get("status") == "terminated_by_stop"]
    fails = []
    if sc["faulty"] is None:
        if rc != 0 or errors:
            fails.append(f"rc={rc} driver_errors={errors}")
        want_launch = [s[0] for s in sc["gpu0"] + sc["gpu1"] + sc["phase2"]]
        if sorted(launched) != sorted(want_launch):
            fails.append(f"launched {launched} != {want_launch}")
    else:
        if rc != 5:
            fails.append(f"rc={rc}, want 5")
        if sc["faulty"] not in errors:
            fails.append(f"no driver_error for {sc['faulty']} (got {errors})")
        late = [n for n in sc["never"] if n in launched]
        if late:
            fails.append(f"launched after failure: {late}")
        run = sc["running"]
        if run is not None and run not in cleaned + stopped:
            fails.append(f"running stage {run} neither cleaned {cleaned} nor stopped {stopped}")
        if sc.get("need_sigkill") and not any(g["stage"] == run and g["sigkill"]
                                              and not g["survived"] for g in groups):
            fails.append(f"no stage_cleanup SIGKILL of {run}: {groups}")
    if sc["running"] is not None and len(pids) != 2:
        fails.append(f"{len(pids)} stage pids recorded, want 2")
    if alive:
        fails.append(f"stage pids alive after the driver exited: {alive}")
    if rc == "wall_timeout":
        fails.append(f"driver did not exit within {CHILD_WALL_S} s")
    log = [{k: r[k] for k in ("t", "stage", "status", "rc", "error", "reason", "groups",
                              "exit") if k in r} for r in records]
    return {"check": f"driver:{scenario}", "pass": not fails, "fails": fails, "rc": rc,
            "elapsed_s": elapsed, "stage_pids_recorded": len(pids), "driver_log": log}


def check_c3_values() -> dict[str, Any]:
    drv = load("gpu_checks_driver_judges", DRIVER)
    fails = []
    base = []
    for f in sorted(ATTEMPT2.glob("c3_*.jsonl")):
        ev = [json.loads(line) for line in f.read_text().splitlines() if line]
        errs = drv.judge_c3(ev)
        base.append({"file": f.name, "errors": errs})
        if errs:
            fails.append(f"retained {f.name} fails judge_c3: {errs}")
    if len(base) != 4:
        fails.append(f"{len(base)} retained C3 records, want 4")
    ev = [json.loads(line) for line in (ATTEMPT2 / "c3_mid_aten.jsonl").read_text()
          .splitlines() if line]
    mutations = []
    for path in TEACHER_PATHS:
        for label, patch in (("nonfinite", {"nonfinite": 1}),
                             ("below_-1", {"min": -1.5}),
                             ("above_1", {"max": 1.5})):
            mut = copy.deepcopy(ev)
            res = next(e for e in mut if e.get("event") == "result")
            res[path]["student_values"].update(patch)
            errs = drv.judge_c3(mut)
            caught = any(path in e for e in errs)
            mutations.append({"path": path, "mutation": label, "caught": caught})
            if not caught:
                fails.append(f"{path} student_values {label} not reported ({errs})")
    return {"check": "c3_values", "pass": not fails, "fails": fails,
            "retained": base, "mutations": mutations}


def check_c1_verdict() -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    c1 = load("gpu_checks_c1", HERE / "c1_fp32_ref.py")
    fails = []
    regenerated = 0
    for f in sorted(ATTEMPT2.glob("c1_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line) if line else {}
            if r.get("event") != "result":
                continue
            again = c1.verdict(r["vs_fp32"])
            regenerated += 1
            if again != r["verdict"]:
                fails.append(f"{f.name} {r['density']}: verdict differs on regeneration")
    if regenerated != 6:
        fails.append(f"{regenerated} retained C1 verdicts, want 6")

    def paths(c: float, e: float, p: float) -> dict[str, dict[str, float]]:
        return {k: {"mean_abs": v, "outside_tol_frac": v, "max_abs": v}
                for k, v in (("compiled", c), ("eager", e), ("padded", p))}

    one = c1.verdict(paths(1.0, 0.0, 0.0))
    if one["result"] != "path_specific" or "compiled:mean_abs" not in one["flagged"] \
            or "compiled:outside_tol_frac" not in one["flagged"]:
        fails.append(f"single erroneous path at median 0 not flagged: {one['flagged']}")
    if one["mean_abs"]["ratio_to_median"]["compiled"] is not None:
        fails.append("ratio_to_median at median 0 should be None")
    zero = c1.verdict(paths(0.0, 0.0, 0.0))
    if zero["result"] != "similar" or zero["flagged"]:
        fails.append(f"three zero paths flagged: {zero['flagged']}")
    return {"check": "c1_verdict", "pass": not fails, "fails": fails,
            "retained_regenerated": regenerated, "median0_single": one["flagged"],
            "all_zero": zero["flagged"]}


LAUNCH_SH = HERE / "launch.sh"
C2_SCRIPT = HERE / "c2_trunk_bwd.py"
NVIDIA_SMI_STUB = r"""
nvidia-smi() {
  case "$*" in
    *--query-compute-apps=*)
      case "$SCENARIO" in
        busy_apps) echo '123, python, 400 MiB';;
        apps_query_fail|both_fail) return 1;;
      esac;;
    *--query-gpu=*)
      case "$SCENARIO" in
        busy_util) echo 10;;
        util_query_fail|both_fail) return 1;;
        util_query_fail_after_zero) echo 0; return 1;;
        *) echo 0;;
      esac;;
  esac
}
"""
IDLE_SCENARIOS = ("idle", "busy_apps", "busy_util", "apps_query_fail", "util_query_fail",
                  "util_query_fail_after_zero", "both_fail")


def check_launcher_idle() -> dict[str, Any]:
    text = LAUNCH_SH.read_text()
    func = text[text.index("gpu_idle()"): text.index("\nsnap()")]
    fails = []
    rcs = {}
    for scenario in IDLE_SCENARIOS:
        p = subprocess.run(["bash", "-c", "set -u\n" + NVIDIA_SMI_STUB + func + "\ngpu_idle 0\n"],
                           env={**os.environ, "SCENARIO": scenario},
                           capture_output=True, text=True, timeout=30)
        rcs[scenario] = p.returncode
        passed_gate = p.returncode == 0
        if passed_gate != (scenario == "idle"):
            fails.append(f"{scenario}: gpu_idle rc={p.returncode} "
                         f"({'passed' if passed_gate else 'failed'} the idle gate)")
    return {"check": "launcher_idle", "pass": not fails, "fails": fails, "rc": rcs}


def check_c2_nonfinite() -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(HERE))
    c2 = load("gpu_checks_c2", C2_SCRIPT)
    drv = load("gpu_checks_driver_c2", DRIVER)
    retained = [json.loads(line) for line in (ATTEMPT2 / "c2_aten_bwd.jsonl").read_text()
                .splitlines() if line]
    fails = []
    if drv.judge_c2(retained):
        fails.append(f"retained attempt 2 C2 record fails judge_c2: {drv.judge_c2(retained)}")
    rows = c2.ROW_CHUNK + 1
    cases: list[dict[str, Any]] = []

    def judged(field: str, metrics: dict[str, Any]) -> list[str]:
        ev = copy.deepcopy(retained)
        point = next(e for e in ev if e.get("event") == "result" and e["label"] == "target")
        point[field] = metrics
        return list(drv.judge_c2(ev))

    def record(comparator: str, name: str, metrics: Any, errs: list[str],
               nonfinite: int, reject: bool) -> None:
        cases.append({"comparator": comparator, "case": name, "nonfinite": nonfinite,
                      "judge_errors": errs, "want_reject": reject})
        if reject and not errs:
            fails.append(f"{comparator} {name}: judge_c2 accepted ({metrics})")
        if not reject and errs:
            fails.append(f"{comparator} {name}: judge_c2 rejected a clean case ({errs})")
        if reject and nonfinite < 1 and name != "compiled_error_only":
            fails.append(f"{comparator} {name}: nonfinite={nonfinite}, want >= 1")

    # dX: [rows, tokens, channels] gradients, all valid.
    mask = torch.ones(rows, 2, dtype=torch.bool)
    dx_cases = {
        "baseline": [],
        "compiled_error_only": [("c", 1, 1.1)],
        "compiled_nan_chunk0": [("c", 0, float("nan"))],
        "eager_nan_chunk0": [("e", 0, float("nan"))],
        "eager_nan_masks_compiled_error": [("e", 0, float("nan")), ("c", 1, 1.1)],
        "eager_nan_chunk1": [("e", c2.ROW_CHUNK, float("nan"))],
        "eager_inf_chunk0": [("e", 3, float("inf"))],
    }
    for name, edits in dx_cases.items():
        comp = torch.ones(rows, 2, 4)
        ref = comp.clone()
        for which, row, value in edits:
            (comp if which == "c" else ref)[row, 0, 0] = value
        metrics = c2.cmp_dx(comp, ref, mask)
        record("dx", name, metrics, judged("dx_compiled_vs_eager", metrics),
               metrics["nonfinite"], name != "baseline")

    # Output: same shapes; a wrong token is |d| > 0.25 on any channel.
    for name, row, value in (("baseline", None, 0.0),
                             ("eager_nan_chunk0", 0, float("nan")),
                             ("eager_nan_chunk1", c2.ROW_CHUNK, float("nan"))):
        comp = torch.ones(rows, 2, 4)
        ref = comp.clone()
        if row is not None:
            ref[row, 0, 0] = value
        metrics = c2.cmp_out(comp, ref, mask)
        record("out", name, metrics, judged("out_compiled_vs_eager", metrics),
               metrics["nonfinite"], row is not None)

    # Parameter gradients: a plain weight and the Amendment 1 key-bias path
    # (its gradient is analytically zero, judged against the query-bias scale).
    names = ("blocks.0.attn.q.bias", "blocks.0.attn.k.bias", "blocks.0.mlp.w")
    for name, target in (("baseline", None), ("eager_nan_weight", "blocks.0.mlp.w"),
                         ("eager_nan_k_bias", "blocks.0.attn.k.bias")):
        comp = {n: torch.zeros(rows) if n.endswith("k.bias") else torch.ones(rows)
                for n in names}
        ref = {n: t.clone() for n, t in comp.items()}
        if target is not None:
            ref[target][0] = float("nan")
        metrics = c2.cmp_params(comp, ref)
        count = sum(m["nonfinite"] for m in metrics.values() if isinstance(m, dict))
        record("params", name, metrics, judged("params_compiled_vs_eager", metrics),
               count, target is not None)
    return {"check": "c2_nonfinite", "pass": not fails, "fails": fails,
            "rows": rows, "row_chunk": c2.ROW_CHUNK, "cases": cases}


def main() -> int:
    print(f"# test_driver_guards.py  utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"# platform={platform.platform()} python={platform.python_version()}")
    print(f"# test script sha256={sha256(Path(__file__).resolve())}")
    print(f"# driver.py sha256={sha256(DRIVER)}")
    print(f"# c1_fp32_ref.py sha256={sha256(HERE / 'c1_fp32_ref.py')}")
    print(f"# c2_trunk_bwd.py sha256={sha256(C2_SCRIPT)}")
    print(f"# launch.sh sha256={sha256(LAUNCH_SH)}")
    results = [run_driver_scenario(s) for s in SCENARIOS]
    results += [check_c3_values(), check_c1_verdict(), check_launcher_idle(),
                check_c2_nonfinite()]
    ok = True
    for r in results:
        print(json.dumps(r, indent=1))
        print(f"# {r['check']}: {'PASS' if r['pass'] else 'FAIL'}")
        ok = ok and r["pass"]
    print(f"# overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
