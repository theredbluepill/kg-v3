from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/Users/poonszesen/kg-v3-8rank')
SCRATCH = ROOT / '.codex-tmp/verify-8rank-r2-mutations'
LOGS = ROOT / 'ops/rebuild-2026-09-29/codex/verify-8rank-r2-independent/mutations'
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

# Fresh checkout-source snapshot; no prior scratch or result is reused.
import shutil
assert not SCRATCH.exists()
SCRATCH.mkdir()
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
selected = [p for p in tracked if p and (p.startswith(('python/', 'scripts/', 'tests/', 'configs/')) or p == 'pyproject.toml')]
for name in selected:
    src, dst = ROOT / name, SCRATCH / name
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
# Python package import needs the existing local native extension, copied byte-for-byte.
for native in (ROOT / 'python/owl').glob('rs*.so'):
    shutil.copy2(native, SCRATCH / 'python/owl' / native.name)
WORKLOAD = 'python/owl/model/kaggriculture_workload.py'
cases.extend([
    case('independent_headroom_teacher_rows', WORKLOAD,
         '        teacher_segments = min(teacher_segments_per_minibatch, n_envs)\n',
         '        teacher_segments = max(teacher_segments_per_minibatch, n_envs)\n',
         'eight_rank_workloads_fit_the_model_chunking'),
    case('independent_headroom_trunk_calls', WORKLOAD,
         '        return math.ceil(self.rows / self.trunk_rows_per_call)\n',
         '        return 1 + math.ceil(self.rows / self.trunk_rows_per_call)\n',
         'eight_rank_workloads_fit_the_model_chunking'),
    case('independent_headroom_head_calls', WORKLOAD,
         '        return math.ceil(self.rows / self.head_rows_per_call)\n',
         '        return 1 + math.ceil(self.rows / self.head_rows_per_call)\n',
         'eight_rank_workloads_fit_the_model_chunking'),
])
source_files = sorted(p for p in SCRATCH.rglob('*') if p.is_file())
originals = {str(p.relative_to(SCRATCH)): p.read_bytes() for p in source_files}
def digest(data):
    return hashlib.sha256(data).hexdigest()
manifest = {path: digest(data) for path, data in originals.items()}
(LOGS / 'scratch-source-sha256-before.json').write_text(json.dumps(manifest, indent=2) + '\n')
(LOGS / 'source-identity.json').write_text(json.dumps({
    'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'base': subprocess.check_output(['git','rev-parse','b51b0c0'],cwd=ROOT,text=True).strip(),
    'tracked_copies': len(selected),
    'native_extension_sha256': digest((SCRATCH/'python/owl/rs.abi3.so').read_bytes()),
    'source_root': str(ROOT),
    'scratch_root': str(SCRATCH),
}, indent=2) + '\n')
env = os.environ | {'PYTHONPATH': str(SCRATCH / 'python'), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'}
base_command = [str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-p', 'no:cacheprovider']
expression = 'isaiah_per_rank or ranked_config or eight_rank_workload or main_loads_kaggriculture'
baseline_command = [*base_command, TEST, STARTUP, '-k', expression, '-q']
def run_baseline(label):
    run = subprocess.run(baseline_command, cwd=SCRATCH, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (LOGS / f'{label}.log').write_text(run.stdout)
    print(label + ': ' + run.stdout, flush=True)
    assert run.returncode == 0, label
run_baseline('baseline')
results = []
for item in cases:
    path = SCRATCH / item['path']
    before = path.read_bytes()
    old, new = item['old'].encode(), item['new'].encode()
    assert before.count(old) == 1, item['name']
    mutated = before.replace(old, new)
    path.write_bytes(mutated)
    command = [*base_command, *item['files'], '-k', item['expression'], '-q']
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
    if run.returncode != 1 or ' failed' not in run.stdout or 'ERROR collecting' in run.stdout:
        raise SystemExit(f"Mutation did not fail test as expected: {item['name']}")
run_baseline('restored-baseline')
after = {str(p.relative_to(SCRATCH)): digest(p.read_bytes()) for p in sorted(SCRATCH.rglob('*')) if p.is_file()}
assert after == manifest
(LOGS / 'scratch-source-sha256-after.json').write_text(json.dumps(after, indent=2) + '\n')
assert all((SCRATCH / path).read_bytes() == content for path, content in originals.items())
summary = {'mutations': len(results), 'all_killed': True, 'all_restored_files': len(originals), 'all_byte_exact': after == manifest}
(LOGS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary), flush=True)
