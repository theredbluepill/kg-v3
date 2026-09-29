from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/Users/poonszesen/kg-v3-m-8rank')
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-merge-8rank-r1/mutations'
SCRATCH = ROOT / '.codex-tmp/verify-merge-8rank-r1-mutations'
PYTHON = ROOT / '.venv/bin/python'
CFG = 'configs/kaggriculture_8rank.yaml'
FACTORY = 'python/owl/model/factory.py'
originals = {path: (SCRATCH/path).read_bytes() for path in (CFG, FACTORY)}
hashes = {path: hashlib.sha256(value).hexdigest() for path, value in originals.items()}
env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONPATH=str(SCRATCH/'python'), PYTHONDONTWRITEBYTECODE='1')
results = []
T = 'tests/kaggriculture/test_configs.py::'
selection = [T+'test_ranked_config_global_workload_equals_scaling_6m[kaggriculture_8rank.yaml-8]', T+'test_ranked_config_optimizer_and_ppo_equal_scaling_6m[kaggriculture_8rank.yaml]', T+'test_config_env_and_cross_section_rules[kaggriculture_8rank.yaml]', T+'test_configs_load_through_full_config_and_build_the_model[kaggriculture_8rank.yaml]']

def run(label, tests):
    command = [str(PYTHON), '-B', '-m', 'pytest', '-c', '/dev/null', '-p', 'no:cacheprovider', '-q', *tests]
    start = time.monotonic()
    result = subprocess.run(command, cwd=SCRATCH, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT/f'{label}.log').write_text('COMMAND '+repr(command)+'\nCWD '+str(SCRATCH)+'\n'+result.stdout)
    receipt = dict(label=label, returncode=result.returncode, seconds=round(time.monotonic()-start,3), command=command, output_log=f'{label}.log')
    print(label, result.returncode, result.stdout.strip().splitlines()[-1], flush=True)
    return receipt, result.stdout

receipt, output = run('extended-baseline', selection)
assert receipt['returncode'] == 0, output
results.append(receipt)

cases = [
    ('global-workload-envs64', CFG, '  n_envs: 32\n', '  n_envs: 64\n', selection[:1], 'Differing attributes:'),
    ('scaling-optimizer-drift', CFG, '  muon_lr: 0.002\n', '  muon_lr: 0.003\n', selection[1:2], 'assert ours.optimizer == scaling.optimizer'),
    ('env-reward-recipe-drift', CFG, '    econ_shaping: 0.2\n', '    econ_shaping: 0.1\n', selection[2:3], 'assert ours.env.reward_shaping == _REWARD_SHAPING'),
]
factory = originals[FACTORY].decode()
start = factory.index('        case KaggricultureTransformerConfig():\n')
end = factory.index('        case _:\n', start)
cases.append(('remove-model-factory-dispatch', FACTORY, factory[start:end], '', selection[3:4], 'Expected code to be unreachable'))
for label, path, old, new, tests, expected in cases:
    text = originals[path].decode()
    assert text.count(old) == 1
    (SCRATCH/path).write_text(text.replace(old,new))
    try:
        receipt, output = run(label,tests)
        receipt.update(path=path, old=old, new=new, original_sha256=hashes[path], mutated_sha256=hashlib.sha256((SCRATCH/path).read_bytes()).hexdigest(), expected_text=expected)
        receipt['killed_as_expected'] = receipt['returncode'] == 1 and expected in output
        assert receipt['killed_as_expected'], output
    finally:
        (SCRATCH/path).write_bytes(originals[path])
        assert (SCRATCH/path).read_bytes() == originals[path]
    receipt['restored_byte_exact'] = True
    results.append(receipt)
receipt, output = run('extended-restored', selection)
assert receipt['returncode'] == 0, output
results.append(receipt)
restored = {path: hashlib.sha256((SCRATCH/path).read_bytes()).hexdigest() for path in originals}
source = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in originals}
assert hashes == restored == source
(OUT/'extended-results.json').write_text(json.dumps(dict(scratch=str(SCRATCH), hashes=hashes, restored_hashes=restored, source_hashes=source, results=results), indent=2)+'\n')
print('DONE: all extended mutations killed and source hashes unchanged',flush=True)
