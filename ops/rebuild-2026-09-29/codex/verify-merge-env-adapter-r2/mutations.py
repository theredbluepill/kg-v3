"""Mutate only a HEAD archive; prove each scratch file is restored byte-exact."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SCRATCH = Path(tempfile.mkdtemp(prefix="verify-merge-env-mutations-"))
archive = subprocess.check_output(["git", "archive", "HEAD", "configs", "python", "scripts", "tests", "pyproject.toml"], cwd=ROOT)
with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
    tf.extractall(SCRATCH, filter="data")
shutil.copy2(ROOT / "python/owl/rs.abi3.so", SCRATCH / "python/owl/rs.abi3.so")
ENV = os.environ | {"OMP_NUM_THREADS": "2", "CARGO_BUILD_JOBS": "2", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": f"{SCRATCH / 'python'}:{SCRATCH}"}
PYTHON = str(ROOT / ".venv/bin/python")
rows = []

def run(name, selectors):
    argv = [PYTHON, "-B", "-m", "pytest", "-q", *selectors]
    with (OUT / f"mutation-{name}.log").open("w") as log:
        result = subprocess.run(argv, cwd=SCRATCH, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    return {"argv": argv, "returncode": result.returncode, "log": f"mutation-{name}.log"}

def mutate(name, relative, old, new, selectors):
    path = SCRATCH / relative
    before = path.read_bytes()
    original = before.decode()
    assert original.count(old) == 1, (name, original.count(old))
    baseline = run(name + "-baseline", selectors)
    assert baseline["returncode"] == 0, baseline
    try:
        path.write_text(original.replace(old, new))
        mutant_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        mutant = run(name + "-mutant", selectors)
    finally:
        path.write_bytes(before)
    restored = run(name + "-restored", selectors)
    row = {"name": name, "path": relative, "before_sha256": hashlib.sha256(before).hexdigest(),
           "mutant_sha256": mutant_hash, "restored_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "main_worktree_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
           "baseline": baseline, "mutant": mutant, "restored": restored}
    assert row["before_sha256"] == row["restored_sha256"] == row["main_worktree_sha256"]
    assert restored["returncode"] == 0
    assert mutant["returncode"] == 1
    rows.append(row)
    (OUT / "mutations.json").write_text(json.dumps({"scratch": str(SCRATCH), "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "native_module_sha256": hashlib.sha256((SCRATCH / "python/owl/rs.abi3.so").read_bytes()).hexdigest(), "rows": rows}, indent=2) + "\n")
    print(json.dumps(row), flush=True)

STARTUP = "tests/scripts/test_run_ppo.py::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation[kaggriculture_8rank.yaml-64-4096-1]"
mutate("8rank-missing-cap", "configs/kaggriculture_8rank.yaml", "    econ_ineffective_cap: 0.1\n", "", [
    "tests/kaggriculture/test_configs.py::test_ranked_config_global_workload_equals_scaling_6m[kaggriculture_8rank.yaml-8]", STARTUP])
mutate("explicit-startup-stop", "scripts/run_ppo.py", '''        if isinstance(cfg.env, KaggricultureEnvConfig):
            raise RuntimeError(
                "run_ppo cannot run Kaggriculture yet: Task 3.1 rollout storage "
                "and action mapping are not implemented"
            )
''', "", [STARTUP])
mutate("workload-before-stop", "scripts/run_ppo.py", '''        _check_model_workload(
            cfg.model, n_envs=cfg.env.n_envs, rl=cfg.rl, distributed=distributed
        )
''', "", ["tests/scripts/test_run_ppo.py::test_main_rejects_unserviceable_kaggriculture_workload_before_allocation", STARTUP])
mutate("compile-before-stop", "scripts/run_ppo.py", "        _check_compile_stack(cfg.model, rl=cfg.rl, distributed=distributed)\n", "", ["tests/scripts/test_run_ppo.py::test_main_rejects_an_unprobed_compile_stack_before_allocation"])
mutate("policy-eval-stop", "scripts/run_ppo.py", '''    if isinstance(env, KaggricultureVectorizedEnv):
        raise RuntimeError(
            "Kaggriculture policy evaluation needs Task 3.1 observation "
            "and action mapping"
        )
''', "", ["tests/scripts/test_run_ppo.py::test_kaggriculture_policy_evaluation_names_remaining_mapping_blocker"])
mutate("model-native-table-default", "python/owl/model/kaggriculture.py", "            native_grammar_tables() if grammar_tables is None else grammar_tables,", "            __import__('owl.kaggriculture.gpu_grammar', fromlist=['expected_grammar_tables']).expected_grammar_tables() if grammar_tables is None else grammar_tables,", ["tests/kaggriculture/test_native_tables.py::test_model_default_loads_native_tables_once"])
