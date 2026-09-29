from pathlib import Path
import io
import json
import os
import subprocess
import sys
import tarfile

root = Path.cwd()
evidence = root / 'ops/rebuild-2026-09-29/merge-1.1b/verify-codex-8209193'
scratch = evidence / 'scratch'
scratch.mkdir(exist_ok=True)
paths = ['scripts/check_engine_trim.py', 'tests/tools/test_check_engine_trim.py', 'engine_rs']
archive = subprocess.check_output(['git', 'archive', '82091936df52125e7f4cce687887a8c259b7ee02', *paths])
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
    stream.extractall(scratch, filter='data')
checker = scratch / 'scripts/check_engine_trim.py'
original = checker.read_text()
env = os.environ.copy()
env['GIT_DIR'] = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], text=True).strip()
env['GIT_WORK_TREE'] = str(scratch)
env['PYTHONDONTWRITEBYTECODE'] = '1'
python = root / '.venv/bin/python'
command = [str(python), '-B', '-m', 'pytest', 'tests/tools/test_check_engine_trim.py', '-q', '-p', 'no:cacheprovider']
results = []

def run(label, source, expected_code):
    checker.write_text(source)
    result = subprocess.run(command, cwd=scratch, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (evidence / f'{label}.log').write_text(result.stdout)
    summary = [line for line in result.stdout.splitlines() if ' passed' in line or line.startswith('FAILED ')]
    print(label, 'exit', result.returncode, '\n' + '\n'.join(summary), flush=True)
    results.append({'label': label, 'exit_code': result.returncode, 'summary': summary})
    assert result.returncode == expected_code, result.stdout

run('mutation-baseline', original, 0)
old = 'if path not in EDITABLE:'
new = 'if path.endswith(".rs") and path != "engine_rs/src/lib.rs":'
assert original.count(old) == 1
run('mutation-task1_1-editable', original.replace(old, new), 1)
old = '_require(sha(data) == _digest(entry["sha256"], name), f"{name}: trace hash")'
new = '_digest(entry["sha256"], name)'
assert original.count(old) == 1
run('mutation-task1_1b-trace-hash', original.replace(old, new), 1)
checker.write_text(original)
(evidence / 'mutation-summary.json').write_text(json.dumps(results, indent=2) + '\n')
