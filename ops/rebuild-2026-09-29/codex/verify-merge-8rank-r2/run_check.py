from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-merge-8rank-r2'
if sys.argv[1] == 'snapshot':
    entries = subprocess.check_output(['git', 'ls-files', '-s', '-z']).split(b'\0')
    hashes = {}
    for entry in entries:
        if not entry:
            continue
        meta, raw = entry.split(b'\t', 1)
        name = raw.decode()
        path = ROOT / name
        content = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
        hashes[name] = {'index': meta.decode(), 'sha256': hashlib.sha256(content).hexdigest()}
    (OUT / sys.argv[2]).write_text(json.dumps(hashes, indent=2) + '\n')
    print(f'Snapshot: {len(hashes)} tracked files')
else:
    name, *cmd = sys.argv[1:]
    start = time.time()
    with (OUT / f'{name}.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    info = {'command': cmd, 'exit_code': result.returncode, 'seconds': round(time.time() - start, 3)}
    (OUT / f'{name}.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))
    print('\n'.join((OUT / f'{name}.log').read_text().splitlines()[-15:]))
