"""Independent scratch-only mutation sensitivity check, ca37089 review r2."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
SCRATCH = OUT / "scratch"
assert not SCRATCH.exists(), "Fresh scratch directory required"
SCRATCH.mkdir()
for directory in ("configs", "python/owl"):
    shutil.copytree(ROOT / directory, SCRATCH / directory,
                    ignore=shutil.ignore_patterns("__pycache__"))
for relative in ("tests/conftest.py", "tests/kaggriculture/conftest.py",
                 "tests/kaggriculture/test_configs.py", "tests/scripts/test_run_ppo.py",
                 "scripts/run_ppo.py", "pyproject.toml"):
    target = SCRATCH / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)

def digest(data):
    return hashlib.sha256(data).hexdigest()

original = {str(p.relative_to(SCRATCH)): p.read_bytes()
            for p in SCRATCH.rglob("*") if p.is_file()}
env = dict(os.environ, PYTHONPATH=str(SCRATCH / "python"), OMP_NUM_THREADS="2",
           PYTHONDONTWRITEBYTECODE="1")
python = str(ROOT / ".venv/bin/python")
config_test = "tests/kaggriculture/test_configs.py"
startup_test = "tests/scripts/test_run_ppo.py"
shape = config_test + "::test_ranked_config_per_rank_shapes_are_scaling_6m_divided[kaggriculture_8rank.yaml-8]"
cross = config_test + "::test_ranked_configs_differ_only_in_per_rank_shapes"
head = config_test + "::test_eight_rank_workloads_fit_the_model_chunking"
guard = config_test + "::test_isaiah_per_rank_shape_fails_loudly_when_not_divisible"
partial = config_test + "::test_isaiah_per_rank_shape_rejects_partial_minibatches_and_teacher_chunks"
startup = startup_test + "::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation[kaggriculture_8rank.yaml-64-4096-1]"
yaml = "configs/kaggriculture_8rank.yaml"
workload = "python/owl/model/kaggriculture_workload.py"
factory = "python/owl/model/factory.py"

def run(name, nodes):
    cmd = [python, "-B", "-m", "pytest", "-p", "no:cacheprovider", *nodes, "-q"]
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=SCRATCH, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=180)
    (OUT / (name + ".log")).write_text(proc.stdout)
    result = {"command": cmd, "cwd": str(SCRATCH), "exit_code": proc.returncode,
              "seconds": round(time.monotonic() - started, 3),
              "log": name + ".log", "tail": proc.stdout.splitlines()[-2:]}
    print(json.dumps({"name": name, **result}), flush=True)
    return result, proc.stdout

def mutate(name, path, old, new, node, expected):
    before = original[path]
    assert before.decode().count(old) == 1, (name, "replacement not unique")
    target = SCRATCH / path
    assert target.read_bytes() == before
    changed = before.decode().replace(old, new).encode()
    target.write_bytes(changed)
    try:
        result, output = run(name, [node])
        result.update(name=name, mutation_path=path, old=old, new=new,
                      before_sha256=digest(before), mutant_sha256=digest(changed),
                      expected=expected, detected=result["exit_code"] == 1 and expected in output)
    finally:
        target.write_bytes(before)
        assert target.read_bytes() == before
        assert (ROOT / path).read_bytes() == before
    result.update(restored_sha256=digest(target.read_bytes()), restored_byte_exact=True,
                  live_source_unchanged=True)
    results.append(result)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert result["detected"], (name, "unexpected result", result)

results = []
baseline, _ = run("baseline", [config_test, startup_test + "::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation"])
assert baseline["exit_code"] == 0
mutate("per-rank-spm", yaml, "  segments_per_minibatch: 2\n", "  segments_per_minibatch: 4\n", shape, "segments_per_minibatch: 4 != 2")
mutate("per-rank-envs", yaml, "  n_envs: 32\n", "  n_envs: 64\n", shape, "n_envs: 64 != 32")
mutate("fixed-teacher-setting", yaml, "  teacher_segments_per_minibatch: 128\n", "  teacher_segments_per_minibatch: 64\n", shape, "assert 64 == 128")
mutate("env-divisibility-guard", config_test, "if value % world_size != 0:", "if name != 'n_envs' and value % world_size != 0:", guard, "Regex pattern did not match")
mutate("spm-divisibility-guard", config_test, "if value % world_size != 0:", "if name != 'segments_per_minibatch' and value % world_size != 0:", guard, "ZeroDivisionError")
mutate("partial-minibatch-guard", config_test, "if n_envs % per_step != 0:", "if False and n_envs % per_step != 0:", partial, "DID NOT RAISE")
mutate("partial-teacher-guard", config_test, "if n_envs % teacher_chunk != 0:", "if False and n_envs % teacher_chunk != 0:", partial, "DID NOT RAISE")
mutate("cross-rank-optimizer", yaml, "  muon_lr: 0.002\n", "  muon_lr: 0.003\n", cross, "assert two.optimizer == other.optimizer")
mutate("cross-rank-env", yaml, "  native_threads: 2\n", "  native_threads: 3\n", cross, 'assert two.env.model_copy')
mutate("cross-rank-ppo", yaml, "  ent_coef: 1e-6\n", "  ent_coef: 2e-6\n", cross, 'assert two.rl.model_copy')
model_body = (ROOT / "configs/model/kaggriculture.yaml").read_text().replace("depth: 8", "depth: 7")
model_inline = "model:\n" + "".join("  " + line + "\n" for line in model_body.splitlines())
mutate("cross-rank-model", yaml, "model: kaggriculture\n", model_inline, cross, 'assert two.model == other.model')
mutate("rollout-seat-rows", workload, 'ForwardWorkload("rollout", n_envs * kt.PLAYERS)', 'ForwardWorkload("rollout", n_envs)', head, "'rollout': (32, 1, 1)")
mutate("minibatch-seat-rows", workload, 'ForwardWorkload("minibatch", segments_per_minibatch * horizon * kt.PLAYERS)', 'ForwardWorkload("minibatch", segments_per_minibatch * horizon)', head, "'minibatch': (128, 1, 1)")
mutate("teacher-clamp-and-call-counts", workload, "teacher_segments = min(teacher_segments_per_minibatch, n_envs)", "teacher_segments = teacher_segments_per_minibatch", head, "'teacher_chunk': (16384, 3, 2)")
mutate("evaluation-seat-rows", workload, 'ForwardWorkload("evaluation", n_envs * kt.PLAYERS)', 'ForwardWorkload("evaluation", n_envs)', head, "'evaluation': (32, 1, 1)")
mutate("startup-workload-caller", "scripts/run_ppo.py", "        _check_model_workload(\n            cfg.model, n_envs=cfg.env.n_envs, rl=cfg.rl, distributed=distributed\n        )\n", "", startup, "assert [] ==")
mutate("global-workload", yaml, "  n_envs: 32\n", "  n_envs: 64\n", config_test + "::test_ranked_config_global_workload_equals_scaling_6m[kaggriculture_8rank.yaml-8]", "global_envs: 512 != 256")
mutate("scaling-optimizer", yaml, "  muon_lr: 0.002\n", "  muon_lr: 0.003\n", config_test + "::test_ranked_config_optimizer_and_ppo_equal_scaling_6m[kaggriculture_8rank.yaml]", "assert ours.optimizer == scaling.optimizer")
mutate("reward-recipe", yaml, "    econ_shaping: 0.2\n", "    econ_shaping: 0.1\n", config_test + "::test_config_env_and_cross_section_rules[kaggriculture_8rank.yaml]", "assert ours.env.reward_shaping == _REWARD_SHAPING")
mutate("model-factory", factory, "        case KaggricultureTransformerConfig():\n", "        case KaggricultureTransformerConfig() if False:\n", config_test + "::test_configs_load_through_full_config_and_build_the_model[kaggriculture_8rank.yaml]", "Expected code to be unreachable")
restored, _ = run("restored", [config_test, startup_test + "::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation"])
assert restored["exit_code"] == 0
manifest = {}
for path, data in original.items():
    assert (SCRATCH / path).read_bytes() == data, path
    assert (ROOT / path).read_bytes() == data, path
    manifest[path] = digest(data)
(OUT / "restored-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
summary = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "baseline": baseline, "restored": restored,
           "detected": sum(r["detected"] for r in results), "attempted": len(results),
           "copied_files_byte_identical_to_live_and_original": len(manifest)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
