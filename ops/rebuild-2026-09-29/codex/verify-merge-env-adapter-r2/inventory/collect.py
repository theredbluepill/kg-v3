"""Independent r2 parent inventory; only writes receipts and scratch archives."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time

repo = Path('/Users/poonszesen/kg-v3-m-env-adapter-r2')
receipts = repo / 'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r2/inventory'
scratch = Path(tempfile.mkdtemp(prefix='verify-env-adapter-r2-inventory-', dir='/private/tmp'))
revisions = {
    'base': 'faed71773fa9f6414e4379ace904780349979cd8',
    'adapter': '8699ca9eab63d0dd3d951fa9cb58e65b1f1c650a',
    'head': 'bc953e98fd70e08c5c87d67a210f8a970d186b6b',
}
python = repo / '.venv/bin/python'
native = repo / 'python/owl/rs.abi3.so'
manifest = {
    'revisions': revisions,
    'scratch': str(scratch),
    'python': str(python),
    'native_collection_dependency': {
        'path': str(native),
        'sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
        'limitation': 'HEAD native extension is reused for Python collection only; no parent native behavioral execution is credited.',
    },
    'archive_symlinks_omitted': {},
    'runs': [],
}

def record():
    (receipts / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

def run(label, kind, command, cwd, env):
    log = receipts / f'{label}-{kind}.log'
    print(f'START {label} {kind}', flush=True)
    start = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    manifest['runs'].append({
        'label': label, 'revision': revisions[label], 'kind': kind,
        'command': command, 'cwd': str(cwd),
        'env': {key: env[key] for key in ['CARGO_BUILD_JOBS', 'OMP_NUM_THREADS', 'CARGO_TARGET_DIR', 'PYO3_PYTHON', 'PYTHONPATH']},
        'exit_code': result.returncode, 'seconds': round(time.monotonic() - start, 2),
        'log': str(log.relative_to(repo)),
    })
    record()
    print(f'END {label} {kind} rc={result.returncode}', flush=True)
    if result.returncode:
        raise RuntimeError(f'{label} {kind} failed; inspect {log}')

record()
for label, revision in revisions.items():
    dest = scratch / label
    dest.mkdir()
    archive = scratch / f'{label}.tar'
    subprocess.run(['git', 'archive', '-o', str(archive), revision], cwd=repo, check=True)
    with tarfile.open(archive) as handle:
        all_members = handle.getmembers()
        manifest['archive_symlinks_omitted'][label] = [m.name for m in all_members if m.issym() or m.islnk()]
        handle.extractall(dest, members=[m for m in all_members if not m.issym() and not m.islnk()], filter='data')
    archive.unlink()
    (dest / 'python/owl/rs.abi3.so').symlink_to(native)
    target = scratch / f'cargo-target-{label}'
    assert not target.exists(), target
    env = os.environ.copy()
    env.update(CARGO_BUILD_JOBS='2', OMP_NUM_THREADS='2', CARGO_TARGET_DIR=str(target),
               PYO3_PYTHON=str(python), PYTHONPATH=str(dest / 'python'))
    run(label, 'import-path', [str(python), '-c', 'import owl, owl.rs; print(owl.__file__); print(owl.rs.__file__)'], dest, env)
    run(label, 'pytest-collect', [str(python), '-m', 'pytest', '--collect-only', '-q', 'tests'], dest, env)
    run(label, 'cargo-engine-list', ['cargo', 'test', '--manifest-path', 'engine_rs/Cargo.toml', '--locked', '--offline', '--', '--list'], dest, env)
    run(label, 'cargo-root-list', ['cargo', 'test', '--locked', '--offline', '--', '--list'], dest, env)

manifest['native_collection_dependency']['sha256_after'] = hashlib.sha256(native.read_bytes()).hexdigest()
record()
print('DONE', scratch, flush=True)
