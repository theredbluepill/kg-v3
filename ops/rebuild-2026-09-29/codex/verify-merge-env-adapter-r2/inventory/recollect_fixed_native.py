"""Freeze rebuilt HEAD module before repeating collection for exact custody."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

receipts = Path(__file__).resolve().parent
manifest_path = receipts / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
scratch = Path(manifest['scratch'])
frozen = scratch / 'frozen-native' / 'rs.abi3.so'
frozen.parent.mkdir()
shutil.copyfile(manifest['native_collection_dependency']['path'], frozen)
frozen_hash = hashlib.sha256(frozen.read_bytes()).hexdigest()
assert frozen_hash == 'f165fc43da61cedb2ea4d06ee60ec2d3496dd53db498813b8e3c8eea78526efe'
manifest['python_collection_retry'] = {
    'reason': 'Another verifier rebuilt HEAD native extension during the first collections; initial native identity cannot be credited for all three. Repeat collections against this frozen copy.',
    'frozen_native_path': str(frozen),
    'sha256_before': frozen_hash,
    'final_log_suffix': 'pytest-collect-fixed-native',
    'limitation': 'Frozen HEAD native module is used only to collect Python tests; no parent behavioral execution is claimed.',
}
for label, revision in manifest['revisions'].items():
    dest = scratch / label
    link = dest / 'python/owl/rs.abi3.so'
    assert link.is_symlink()
    link.unlink()
    link.symlink_to(frozen)
    env = os.environ.copy()
    env.update(CARGO_BUILD_JOBS='2', OMP_NUM_THREADS='2', PYTHONPATH=str(dest / 'python'))
    for kind, command in [
        ('import-path-fixed-native', [manifest['python'], '-c', 'import owl, owl.rs; from pathlib import Path; print(owl.__file__); print(owl.rs.__file__); print(Path(owl.rs.__file__).resolve())']),
        ('pytest-collect-fixed-native', [manifest['python'], '-m', 'pytest', '--collect-only', '-q', 'tests']),
    ]:
        log = receipts / f'{label}-{kind}.log'
        start = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=dest, env=env, stdout=stream, stderr=subprocess.STDOUT)
        manifest['runs'].append({
            'label': label, 'revision': revision, 'kind': kind, 'command': command, 'cwd': str(dest),
            'env': {key: env[key] for key in ['CARGO_BUILD_JOBS', 'OMP_NUM_THREADS', 'PYTHONPATH']},
            'exit_code': result.returncode, 'seconds': round(time.monotonic() - start, 2), 'log': str(log),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(label, kind, result.returncode, flush=True)
        assert result.returncode == 0
manifest['python_collection_retry']['sha256_after'] = hashlib.sha256(frozen.read_bytes()).hexdigest()
assert manifest['python_collection_retry']['sha256_after'] == frozen_hash
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
