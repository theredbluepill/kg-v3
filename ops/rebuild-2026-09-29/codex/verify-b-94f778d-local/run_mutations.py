from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-b-94f778d-local'
CASES = [
    (
        'constant-critic',
        'python/owl/model/kaggriculture.py',
        b'        return logits.log_softmax(dim=-1)\n',
        b'        return torch.zeros_like(logits).log_softmax(dim=-1)\n',
        ['uv', 'run', 'pytest', 'tests/kaggriculture/test_model_encoder.py', '-q', '-k', 'critic or values_are'],
        '3 failed, 3 passed, 28 deselected',
    ),
    (
        'tile-count-bound',
        'python/owl/kaggriculture/types.py',
        b'        if tile_counts.numel() and int(tile_counts.min()) < 0:\n',
        b'        if tile_counts.numel() and int(tile_counts.min()) < -1:\n',
        ['uv', 'run', 'pytest', 'tests/kaggriculture/test_types.py', '-q'],
        '3 failed, 123 passed',
    ),
]
results = []
for name, rel, old, new, command, expected in CASES:
    target = ROOT / rel
    original = target.read_bytes()
    snapshot = OUT / (rel.replace('/', '__') + '.before')
    assert original == snapshot.read_bytes(), f'{rel} changed since snapshot'
    assert original.count(old) == 1, f'{name}: mutation target must match exactly once'
    result = {'name': name, 'path': rel, 'command': command, 'before_sha256': hashlib.sha256(original).hexdigest()}
    try:
        target.write_bytes(original.replace(old, new))
        result['mutation_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        with (OUT / f'mutation-{name}.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        result['exit_code'] = run.returncode
    finally:
        target.write_bytes(original)
        result['restored_byte_for_byte'] = target.read_bytes() == original
        result['restored_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        results.append(result)
        (OUT / 'mutation-results.json').write_text(json.dumps(results, indent=2) + '\n')
    output = (OUT / f'mutation-{name}.log').read_text()
    print(json.dumps(result), flush=True)
    print('\n'.join(output.splitlines()[-8:]), flush=True)
    assert result['restored_byte_for_byte']
    assert run.returncode == 1, f'{name}: expected assertion failures, got {run.returncode}'
    assert '3 failed' in output, f'{name}: expected three regression tests to fail'
