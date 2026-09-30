"""Mutation checks of current tests on exact scratch copies; restore all bytes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
out = root / 'ops/rebuild-2026-09-29/codex/verify-3.2-3.3-r2'
scratch = out / 'scratch'
source = root / 'scripts/run_ppo.py'
target = scratch / 'scripts/run_ppo.py'
test = scratch / 'tests/scripts/test_run_ppo.py'
for dst, src in ((target, source), (test, root / 'tests/scripts/test_run_ppo.py')):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
original = target.read_bytes()
original_hash = hashlib.sha256(original).hexdigest()
selection = 'test_kaggriculture_evaluation_decides_winners_by_raw_banks or test_run_training_loop_reports_promotion_only_after_it_completes'
command = ['uv', 'run', 'pytest', str(test), '-k', selection, '-q', '--import-mode=importlib']
results = {}
def run(label):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out / f'mutation-{label}.log').write_text(completed.stdout)
    result = {'exit_code': completed.returncode, 'tail': completed.stdout.splitlines()[-5:]}
    results[label] = result
    print(label, result, flush=True)
    return completed.returncode

try:
    assert run('baseline') == 0
    text = original.decode()
    start = text.index('                # Logged only after refresh, teacher update, promoted checkpoint')
    end = text.index('                next_checkpoint_env_steps', start)
    log_block = text[start:end]
    mutated = text[:start] + text[end:]
    insertion = mutated.index('                if replace_last_best:')
    mutated = mutated[:insertion] + log_block + mutated[insertion:]
    target.write_text(mutated)
    try:
        assert run('premature-promotion') == 1
    finally:
        target.write_bytes(original)
    assert target.read_bytes() == original == source.read_bytes()
    old = 'return banks, _candidate_bank_metrics(banks, assignment)'
    assert text.count(old) == 1
    target.write_text(text.replace(old, 'return returns, _candidate_bank_metrics(banks, assignment)'))
    try:
        assert run('shaped-return-winner') == 1
    finally:
        target.write_bytes(original)
    assert target.read_bytes() == original == source.read_bytes()
    assert run('restored') == 0
finally:
    target.write_bytes(original)
    results['custody'] = {
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sha256_before': original_hash,
        'sha256_restored': hashlib.sha256(target.read_bytes()).hexdigest(),
        'byte_equal_source': target.read_bytes() == source.read_bytes(),
        'test_byte_equal_source': test.read_bytes() == (root / 'tests/scripts/test_run_ppo.py').read_bytes(),
    }
    (out / 'mutation.json').write_text(json.dumps(results, indent=2) + '\n')
    shutil.rmtree(scratch)
