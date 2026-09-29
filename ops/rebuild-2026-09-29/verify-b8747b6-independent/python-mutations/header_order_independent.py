from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/verify-b8747b6-independent/python-mutations'
SCRATCH = OUT / 'scratch'
SOURCE = 'scripts/kaggriculture_observation_oracle/regenerate.py'
source = SCRATCH / SOURCE
before = source.read_bytes()
repo_before = (ROOT / SOURCE).read_bytes()
test = SCRATCH / 'tests/tools/test_header_order_independent.py'
test.write_text((ROOT / 'tests/tools/test_observation_oracle_custody.py').read_text().replace(
    '        inv[key] = inv.pop(key)\n',
    '        inv[key] = inv.pop(key)\n        manifest["coverage"] = oracle.count_coverage(rows)\n',
))
sha = lambda data: hashlib.sha256(data).hexdigest()
record = {'name': 'header_order_independent', 'source_before_sha256': sha(before),
          'distinction': 'Recompute coverage after header order corruption so the hash guard is the sole rejection.'}

def run(phase):
    cmd = [str(ROOT / '.venv/bin/python'), '-B', '-m', 'pytest', '-c', str(SCRATCH / 'pytest.ini'), '--confcutdir', str(SCRATCH), '-q', str(test) + '::test_custody_rejects_corruption_even_with_rehashed_files[header_order]']
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=SCRATCH, env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    log = 'header_order_independent-' + phase + '.log'
    (OUT / log).write_bytes(proc.stdout)
    return {'exit': proc.returncode, 'argv': cmd, 'log': log, 'wall_seconds': round(time.monotonic() - start, 3)}

try:
    record['baseline'] = run('baseline')
    assert record['baseline']['exit'] == 0
    old = 'digest(meta["header_sha256"], "header hash")\n                    == sha(header_bytes(raw)),'
    assert before.decode().count(old) == 1
    source.write_text(before.decode().replace(old, 'True,'))
    record['mutant'] = run('mutant')
    assert record['mutant']['exit'] == 1
    assert 'DID NOT RAISE' in (OUT / record['mutant']['log']).read_text()
finally:
    source.write_bytes(before)
    record['restored'] = run('restored')
    record['scratch_restored_exact'] = source.read_bytes() == before
    record['repository_source_unchanged'] = (ROOT / SOURCE).read_bytes() == repo_before
    (OUT / 'header_order_independent.json').write_text(json.dumps(record, indent=2) + '\n')
assert record['restored']['exit'] == 0
assert record['scratch_restored_exact'] and record['repository_source_unchanged']
print(json.dumps(record, indent=2))
