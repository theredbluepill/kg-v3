"""Independent scratch-only removal probes of each repaired inventory guard."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path('/Users/poonszesen/kg-v3-env')
scratch = Path('/private/tmp/kg-env-r3-oracle')
receipts = root / 'ops/rebuild-2026-09-29/codex/verify-env-r3/oracle'
source = scratch / 'scripts/record_kaggriculture_env_reference.py'
original = source.read_bytes()
text = original.decode()
lines = text.splitlines(keepends=True)
offsets = [0]
for line in lines:
    offsets.append(offsets[-1] + len(line))
tree = ast.parse(text)
names = [
    'program lengths outside 1..252', 'program offset inventory differs',
    'packed token inventory differs', 'transition index inventory differs',
    'seed consumption differs', 'terminal steps differ',
    'terminal done schedule differs', 'terminal values differ',
    'terminal winner differs', 'economic transition continuity differs',
    'bank transition continuity differs', 'nonfinite {name}',
    'economic counters decrease', 'array hash/shape custody differs',
]
statements = {}
for node in ast.walk(tree):
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        continue
    call = node.value
    if not isinstance(call.func, ast.Name) or call.func.id != 'require' or len(call.args) != 2:
        continue
    reason = call.args[1]
    if isinstance(reason, ast.Constant):
        message = reason.value
    elif isinstance(reason, ast.JoinedStr) and ast.unparse(reason) == "f'nonfinite {name}'":
        message = 'nonfinite {name}'
    else:
        continue
    if message in names:
        assert message not in statements
        statements[message] = node
assert set(statements) == set(names)
results = []
try:
    for index, name in enumerate(names):
        node = statements[name]
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        mutated = (text[:start] + 'pass' + text[end:]).encode()
        source.write_bytes(mutated)
        # Avoid import caches with timestamp/size collisions across rapid mutation restores.
        for cached in (source.parent / '__pycache__').glob('record_kaggriculture_env_reference.*.pyc'):
            cached.unlink()
        command = [sys.executable, '-m', 'pytest', '-c', '/dev/null', f'--rootdir={scratch}', f'--confcutdir={scratch}', str(scratch / 'tests/tools/test_record_kaggriculture_env_reference.py'), '-q', '-k', 'semantic or hash_custody', '-p', 'no:cacheprovider']
        start_time = time.monotonic()
        run = subprocess.run(command, cwd=scratch, env=os.environ|{'GIT_DIR':str(root/'.git'),'GIT_WORK_TREE':str(root)}, timeout=12, capture_output=True, text=True)
        output = run.stdout + run.stderr
        (receipts / f'guard-{index:02d}.log').write_text(output)
        result = {'guard': name, 'source_line': node.lineno, 'exit_code': run.returncode,
                  'seconds': time.monotonic() - start_time,
                  'summary': output.strip().splitlines()[-1],
                  'mutant_sha256': hashlib.sha256(mutated).hexdigest(),
                  'log': f'guard-{index:02d}.log'}
        results.append(result)
        print(json.dumps(result), flush=True)
finally:
    source.write_bytes(original)
    restored = source.read_bytes() == original
    source_hash = hashlib.sha256(original).hexdigest()
    report = {'source_sha256': source_hash, 'scratch_restored_byte_exact': restored,
              'main_source_unchanged': (root / 'scripts/record_kaggriculture_env_reference.py').read_bytes() == original,
              'mutations': results}
    (receipts / 'guard-removals.json').write_text(json.dumps(report, indent=2) + '\n')
    assert restored
assert len(results) == 14
assert all(row['exit_code'] == 1 and '26 failed' not in row['summary'] for row in results)
