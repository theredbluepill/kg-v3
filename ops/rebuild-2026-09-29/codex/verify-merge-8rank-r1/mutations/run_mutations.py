from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path('/Users/poonszesen/kg-v3-m-8rank')
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-merge-8rank-r1/mutations'
SCRATCH = ROOT / '.codex-tmp/verify-merge-8rank-r1-mutations'
PYTHON = ROOT / '.venv/bin/python'
assert not SCRATCH.exists(), SCRATCH
SCRATCH.mkdir(parents=True)
for directory in ('configs', 'python'):
    shutil.copytree(ROOT / directory, SCRATCH / directory, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for relative in ('tests/conftest.py', 'tests/kaggriculture/conftest.py', 'tests/kaggriculture/test_configs.py', 'tests/scripts/test_run_ppo.py', 'scripts/run_ppo.py'):
    target = SCRATCH / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)
CFG = 'configs/kaggriculture_8rank.yaml'
TEST = 'tests/kaggriculture/test_configs.py'
RUN = 'scripts/run_ppo.py'
WORK = 'python/owl/model/kaggriculture_workload.py'
CONFIG_TEST = 'tests/kaggriculture/test_configs.py'
STARTUP_TEST = 'tests/scripts/test_run_ppo.py::test_main_loads_kaggriculture_config_and_prints_headroom_before_allocation'
originals = {path: (SCRATCH / path).read_bytes() for path in (CFG, TEST, RUN, WORK)}
hashes = {path: hashlib.sha256(data).hexdigest() for path, data in originals.items()}
env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONPATH=str(SCRATCH / 'python'), PYTHONDONTWRITEBYTECODE='1')
results = []

def run(label, selection):
    command = [str(PYTHON), '-B', '-m', 'pytest', '-c', '/dev/null', '-p', 'no:cacheprovider', '-q', *selection]
    start = time.monotonic()
    proc = subprocess.run(command, cwd=SCRATCH, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout
    (OUT / f'{label}.log').write_text('COMMAND ' + repr(command) + '\nCWD ' + str(SCRATCH) + '\n' + output)
    result = dict(label=label, returncode=proc.returncode, seconds=round(time.monotonic()-start, 3), command=command, output_log=f'{label}.log')
    print(label, proc.returncode, output.strip().splitlines()[-1], flush=True)
    return result, output

def mutate(label, path, old, new, selection, expected_text):
    text = originals[path].decode()
    assert text.count(old) == 1, (label, text.count(old))
    (SCRATCH / path).write_text(text.replace(old, new))
    try:
        result, output = run(label, selection)
        result.update(path=path, original_sha256=hashes[path], mutated_sha256=hashlib.sha256((SCRATCH/path).read_bytes()).hexdigest(), old=old, new=new, expected_text=expected_text)
        result['killed_as_expected'] = result['returncode'] == 1 and expected_text in output
        assert result['killed_as_expected'], (label, output)
    finally:
        (SCRATCH / path).write_bytes(originals[path])
        assert (SCRATCH/path).read_bytes() == originals[path]
    result['restored_byte_exact'] = True
    results.append(result)
    (OUT/'results.json').write_text(json.dumps(dict(scratch=str(SCRATCH), hashes=hashes, results=results), indent=2) + '\n')

baseline_selection = [CONFIG_TEST, STARTUP_TEST, '-k', 'ranked_config or isaiah_per_rank_shape or eight_rank_workloads or main_loads']
result, output = run('baseline', baseline_selection)
assert result['returncode'] == 0, output
results.append(result)
shape = [CONFIG_TEST+'::test_ranked_config_per_rank_shapes_are_scaling_6m_divided', '-k', '8rank']
mutate('shape-spm-4', CFG, '  segments_per_minibatch: 2\n', '  segments_per_minibatch: 4\n', shape, 'Differing attributes:')
mutate('shape-envs-64', CFG, '  n_envs: 32\n', '  n_envs: 64\n', shape, 'Differing attributes:')
mutate('teacher-fixed-64', CFG, '  teacher_segments_per_minibatch: 128\n', '  teacher_segments_per_minibatch: 64\n', shape, 'assert 64 == 128')
guard = [CONFIG_TEST+'::test_isaiah_per_rank_shape_fails_loudly_when_not_divisible']
mutate('remove-envs-divisibility-guard', TEST, '        if value % world_size != 0:\n', '        if name != "n_envs" and value % world_size != 0:\n', guard + ['-k', '3-n_envs'], 'Regex pattern did not match')
mutate('remove-spm-divisibility-guard', TEST, '        if value % world_size != 0:\n', '        if name != "segments_per_minibatch" and value % world_size != 0:\n', guard + ['-k', '32-segments'], 'ZeroDivisionError')
partial = [CONFIG_TEST+'::test_isaiah_per_rank_shape_rejects_partial_minibatches_and_teacher_chunks']
mutate('remove-partial-minibatch-guard', TEST, '    if n_envs % per_step != 0:\n', '    if False:\n', partial, 'DID NOT RAISE')
mutate('remove-partial-teacher-guard', TEST, '    if n_envs % teacher_chunk != 0:\n', '    if False:\n', partial, 'DID NOT RAISE')
only_shape = [CONFIG_TEST+'::test_ranked_configs_differ_only_in_per_rank_shapes']
mutate('cross-rank-optimizer-drift', CFG, '  muon_lr: 0.002\n', '  muon_lr: 0.003\n', only_shape, 'assert two.optimizer == other.optimizer')
mutate('cross-rank-env-drift', CFG, '  native_threads: 2\n', '  native_threads: 3\n', only_shape, 'assert two.env.model_copy')
mutate('cross-rank-ppo-drift', CFG, '  ent_coef: 1e-6\n', '  ent_coef: 2e-6\n', only_shape, 'assert two.rl.model_copy')
# Full model mapping with one legal modified value exercises the expanded model comparison.
model_cfg = (SCRATCH/'configs/model/kaggriculture.yaml').read_text()
mutate('cross-rank-model-drift', CFG, '\nmodel: kaggriculture\n', '\nmodel:\n' + '\n'.join('  '+line for line in model_cfg.splitlines() if not line.startswith('#')) + '\n  depth: 7\n', only_shape, 'assert two.model == other.model')
headroom = [CONFIG_TEST+'::test_eight_rank_workloads_fit_the_model_chunking']
mutate('workload-rollout-seats', WORK, 'ForwardWorkload("rollout", n_envs * kt.PLAYERS)', 'ForwardWorkload("rollout", n_envs)', headroom, "'rollout': (32, 1, 1)")
mutate('workload-minibatch-seats', WORK, 'ForwardWorkload("minibatch", segments_per_minibatch * horizon * kt.PLAYERS)', 'ForwardWorkload("minibatch", segments_per_minibatch * horizon)', headroom, "'minibatch': (128, 1, 1)")
mutate('workload-teacher-unclamped', WORK, 'teacher_segments = min(teacher_segments_per_minibatch, n_envs)', 'teacher_segments = teacher_segments_per_minibatch', headroom, "'teacher_chunk': (16384, 3, 2)")
mutate('workload-evaluation-seats', WORK, 'ForwardWorkload("evaluation", n_envs * kt.PLAYERS)', 'ForwardWorkload("evaluation", n_envs)', headroom, "'evaluation': (32, 1, 1)")
mutate('remove-startup-workload-caller', RUN, '        _check_model_workload(\n            cfg.model, n_envs=cfg.env.n_envs, rl=cfg.rl, distributed=distributed\n        )\n', '', [STARTUP_TEST, '-k', '8rank'], 'assert [line.split(":")[0] for line in headroom]')
result, output = run('restored', baseline_selection)
assert result['returncode'] == 0, output
results.append(result)
restored = {path: hashlib.sha256((SCRATCH/path).read_bytes()).hexdigest() for path in originals}
source = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in originals}
assert restored == hashes == source
(OUT/'results.json').write_text(json.dumps(dict(scratch=str(SCRATCH), hashes=hashes, restored_hashes=restored, source_hashes=source, results=results), indent=2) + '\n')
print('DONE: every mutation killed, scratch byte-exact restored, source hashes unchanged', flush=True)
