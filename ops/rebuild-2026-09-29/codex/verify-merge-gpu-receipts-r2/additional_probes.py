"""Independent CPU probes; operates only on a scratch bundle."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

E = Path(__file__).resolve().parent
META = json.loads((E / "scratch.json").read_text())
B = Path(META["scratch"]) / "bundle"
S = B / "scripts"
os.environ["GPUCHK_ROOT"] = META["root"]
os.environ["GPUCHK_DRYRUN"] = "1"
sys.path.insert(0, str(S))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

D = load("driver_additional", S / "driver.py")
C2 = load("c2_additional", S / "c2_trunk_bwd.py")
import torch

result = {}
base = D.events(B / "pod/attempt2/c2_aten_bwd.jsonl")
compiled = torch.ones(513, 1, 1)
reference = compiled.clone()
mask = torch.ones(513, 1, dtype=torch.bool)
for label, array in (("baseline", None), ("compiled_nan", compiled), ("eager_nan", reference)):
    if array is not None:
        array[0] = float("nan")
    metrics = C2.cmp_dx(compiled, reference, mask)
    events = copy.deepcopy(base)
    point = next(r for r in events if r.get("event") == "result" and r["label"] == "target")
    point["dx_compiled_vs_eager"] = metrics
    result[label] = {"metrics": metrics, "errors": D.judge_c2(events)}
    if array is not None:
        array[0] = 1

# Extract exactly the launcher's idle function, stub only nvidia-smi.
launcher = (S / "launch.sh").read_text()
idle = launcher[launcher.index("gpu_idle()") : launcher.index("\nsnap()")]
checks = {}
for scenario in ("idle", "busy_apps", "busy_util", "both_fail", "apps_query_fail", "util_query_fail"):
    shell = r'''
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
        *) echo 0;;
      esac;;
  esac
}
'''
    p = subprocess.run(["bash", "-c", shell + idle + "\ngpu_idle 0\n"],
                       env={**os.environ, "SCENARIO": scenario}, capture_output=True, text=True)
    checks[scenario] = p.returncode
result["launcher_idle"] = checks

# Revert each repaired guard separately in scratch and prove a fault survives.
# Restore each file in finally and compare original hashes.
regressions = []
driver = S / "driver.py"
original = driver.read_bytes()
text = original.decode()
tree = ast.parse(text)
stream = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Driver"
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "stream")
lines = text.splitlines(keepends=True)
mut = "".join(lines[:stream.lineno-1]) + (
    "    def stream(self, stages, gpu):\n"
    "        for stage in stages:\n"
    "            self.run_stage(stage, gpu)\n\n"
) + "".join(lines[stream.end_lineno:])
try:
    driver.write_text(mut)
    M = load("driver_stream_mutant", driver)
    wd = Path(tempfile.mkdtemp(prefix="stream-mutant-", dir=META["scratch"]))
    M.RUN = wd
    obj = M.Driver()
    obj.install_cleanup = lambda: None
    launched = []
    def run(stage, gpu):
        launched.append(stage[0])
        if stage[0] == "bad":
            raise OSError("source-mutation sentinel")
    obj.run_stage = run
    M.GPU0_PHASE1 = [("bad",)]
    M.GPU1_PHASE1 = []
    M.GPU0_PHASE2 = [("later",)]
    rc = obj.main()
    obj.log_fh.close()
    regressions.append({"guard_removed": "stream exception boundary", "bad_rc": rc,
                        "later_launched": "later" in launched,
                        "fault_reproduced": rc == 0 and "later" in launched})
finally:
    driver.write_bytes(original)

try:
    driver.write_text(text.replace('                                                   "teacher_self_cached",\n', "")
                          .replace('                                                   "teacher_perturbed_combined",\n', ""))
    assert driver.read_bytes() != original
    M = load("driver_value_mutant", driver)
    events = D.events(B / "pod/attempt2/c3_mid_aten.jsonl")
    next(r for r in events if r.get("event") == "result")["teacher_self_cached"]["student_values"]["nonfinite"] = 1
    regressions.append({"guard_removed": "two C3 teacher value paths",
                        "fault_reproduced": not M.judge_c3(events),
                        "baseline_rejects": bool(D.judge_c3(events))})
finally:
    driver.write_bytes(original)

c1 = S / "c1_fp32_ref.py"
old = c1.read_bytes()
try:
    c1.write_text(old.decode().replace('if metric != "max_abs":', 'if metric != "max_abs" and med:'))
    M = load("c1_median_mutant", c1)
    paths = {p: {m: float(p == "compiled") for m in ("mean_abs", "outside_tol_frac", "max_abs")}
             for p in ("compiled", "eager", "padded")}
    regressions.append({"guard_removed": "C1 median-zero comparison",
                        "fault_reproduced": M.verdict(paths)["result"] == "similar"})
finally:
    c1.write_bytes(old)
result["source_mutations"] = regressions
result["restored"] = all(hashlib.sha256((B / p).read_bytes()).hexdigest() == h
                         for p, h in META["hashes"].items())
assert all(r["fault_reproduced"] for r in regressions)
assert result["restored"]
(E / "additional-probes.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
