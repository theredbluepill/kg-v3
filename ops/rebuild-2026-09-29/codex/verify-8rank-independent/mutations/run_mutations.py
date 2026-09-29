from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/Users/poonszesen/kg-v3-8rank')
SCRATCH = ROOT / '.codex-tmp/verify-8rank-independent-mutations'
LOGS = ROOT / 'ops/rebuild-2026-09-29/codex/verify-8rank-independent/mutations'
CFG = 'configs/kaggriculture_8rank.yaml'
TEST = 'tests/kaggriculture/test_configs.py'
STARTUP = 'tests/scripts/test_run_ppo.py'
RUNNER = 'scripts/run_ppo.py'

def case(name, path, old, new, expression, files=(TEST,)):
    return dict(name=name, path=path, old=old, new=new, expression=expression, files=files)

cases = [
    case('wrong_spm_4', CFG, '  segments_per_minibatch: 2\n', '  segments_per_minibatch: 4\n',
         '(global_workload or per_rank_shapes or differ_only or eight_rank_workloads) and (8rank or differ_only or eight_rank_workloads)'),
    case('wrong_envs_64', CFG, '  n_envs: 32\n', '  n_envs: 64\n',
         '(global_workload or per_rank_shapes or differ_only or eight_rank_workloads or main_loads) and (8rank or differ_only or eight_rank_workloads)', (TEST, STARTUP)),
    case('teacher_constant_64', CFG, '  teacher_segments_per_minibatch: 128\n', '  teacher_segments_per_minibatch: 64\n',
         'per_rank_shapes and 8rank'),
    case('remove_env_divisibility_guard', TEST, '        if value % world_size != 0:\n', '        if name != "n_envs" and value % world_size != 0:\n',
         'fails_loudly_when_not_divisible and 3-n_envs'),
    case('remove_spm_divisibility_guard', TEST, '        if value % world_size != 0:\n', '        if name != "segments_per_minibatch" and value % world_size != 0:\n',
         'fails_loudly_when_not_divisible and 32-segments'),
    case('remove_minibatch_divisibility_guard', TEST, '    if n_envs % per_step != 0:\n', '    if False and n_envs % per_step != 0:\n',
         'rejects_partial_minibatches_and_teacher_chunks'),
    case('remove_teacher_divisibility_guard', TEST, '    if n_envs % teacher_chunk != 0:\n', '    if False and n_envs % teacher_chunk != 0:\n',
         'rejects_partial_minibatches_and_teacher_chunks'),
    case('wrong_optimizer_lr', CFG, '  muon_lr: 0.002\n', '  muon_lr: 0.003\n',
         'differ_only_in_per_rank_shapes'),
    case('wrong_env_threads', CFG, '  native_threads: 2\n', '  native_threads: 3\n',
         'differ_only_in_per_rank_shapes'),
    case('wrong_model', CFG, 'model: kaggriculture\n', 'model: kaggriculture_cpu\n',
         'differ_only_in_per_rank_shapes'),
    case('wrong_ppo_entropy', CFG, '  ent_coef: 1e-6\n', '  ent_coef: 2e-6\n',
         'differ_only_in_per_rank_shapes'),
    case('remove_startup_workload_call', RUNNER,
         '        _check_model_workload(\n            cfg.model, n_envs=cfg.env.n_envs, rl=cfg.rl, distributed=distributed\n        )\n',
         '        # Independent verification mutation: startup call removed.\n',
         'main_loads_kaggriculture and 8rank', (STARTUP,)),
]
source_files = sorted(p for p in SCRATCH.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.toml'))
originals = {str(p.relative_to(SCRATCH)): p.read_bytes() for p in source_files}
def digest(data):
    return hashlib.sha256(data).hexdigest()
manifest = {path: digest(data) for path, data in originals.items()}
(LOGS / 'scratch-source-sha256-before.json').write_text(json.dumps(manifest, indent=2) + '\n')
env = os.environ | {'PYTHONPATH': str(SCRATCH / 'python'), 'PYTHONDONTWRITEBYTECODE': '1'}
results = []
for item in cases:
    path = SCRATCH / item['path']
    before = path.read_bytes()
    old, new = item['old'].encode(), item['new'].encode()
    assert before.count(old) == 1, item['name']
    mutated = before.replace(old, new)
    path.write_bytes(mutated)
    # Delete caches so same-length substitutions cannot be hidden by .pyc timestamps.
    import shutil
    for cache in SCRATCH.rglob('__pycache__'):
        shutil.rmtree(cache)
    command = [str(ROOT / '.venv/bin/python'), '-m', 'pytest', *item['files'], '-k', item['expression'], '-q']
    try:
        run = subprocess.run(command, cwd=SCRATCH, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (LOGS / f"{item['name']}.log").write_text(run.stdout)
        record = item | {'command': command, 'exit_code': run.returncode,
                         'before_sha256': digest(before), 'mutated_sha256': digest(mutated),
                         'summary': '\n'.join(run.stdout.splitlines()[-4:])}
    finally:
        path.write_bytes(before)
        assert path.read_bytes() == before
    record['restored_sha256'] = digest(path.read_bytes())
    record['restored_byte_exact'] = path.read_bytes() == originals[item['path']]
    results.append(record)
    print(json.dumps({k: record[k] for k in ('name', 'exit_code', 'restored_byte_exact', 'summary')}), flush=True)
    (LOGS / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    if run.returncode != 1 or ' failed' not in run.stdout:
        raise SystemExit(f"Mutation did not fail test as expected: {item['name']}")
after = {path: digest((SCRATCH / path).read_bytes()) for path in originals}
assert after == manifest
(LOGS / 'scratch-source-sha256-after.json').write_text(json.dumps(after, indent=2) + '\n')
command = [str(ROOT / '.venv/bin/python'), '-m', 'pytest', TEST, STARTUP, '-k', 'isaiah_per_rank or ranked_config or eight_rank_workload or main_loads_kaggriculture', '-q']
run = subprocess.run(command, cwd=SCRATCH, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(LOGS / 'restored-baseline.log').write_text(run.stdout)
print(run.stdout, flush=True)
assert run.returncode == 0
print(json.dumps({'mutations': len(results), 'all_killed': True, 'source_files_restored': len(originals), 'all_byte_exact': after == manifest}), flush=True)
