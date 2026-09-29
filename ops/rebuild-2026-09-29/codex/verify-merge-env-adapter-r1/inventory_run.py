from pathlib import Path
import hashlib, json, os, subprocess, tarfile, tempfile, time

repo = Path('/Users/poonszesen/kg-v3-m-env-adapter')
receipts = repo / 'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1'
scratch = Path(tempfile.mkdtemp(prefix='verify-env-adapter-inventory-', dir='/private/tmp'))
revisions = {'base': '666deec789b2e75f56afb6dbb7b7acd40171c6b1', 'adapter': '8699ca9eab63d0dd3d951fa9cb58e65b1f1c650a', 'head': '49a48350cdd72d6b546320441d677fbcbae2d893'}
python = repo / '.venv/bin/python'
envbase = os.environ.copy()
envbase.update(CARGO_BUILD_JOBS='2', OMP_NUM_THREADS='2', CARGO_TARGET_DIR=str(scratch / 'cargo-target'))
manifest = {'scratch': str(scratch), 'python': str(python), 'env': {k: envbase[k] for k in ['CARGO_BUILD_JOBS', 'OMP_NUM_THREADS', 'CARGO_TARGET_DIR']}, 'native_collection_dependency': {'path': str(repo/'python/owl/rs.abi3.so'), 'sha256': hashlib.sha256((repo/'python/owl/rs.abi3.so').read_bytes()).hexdigest(), 'limitation': 'HEAD extension reused for collection only; no parent native behavioral test is credited.'}, 'runs': []}
def run(label, kind, cmd, cwd, env):
    path = receipts / f'inventory-{label}-{kind}.log'
    start = time.monotonic()
    print(f'START {label} {kind}', flush=True)
    with path.open('w') as f:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    record = {'revision': revisions[label], 'label': label, 'kind': kind, 'command': cmd, 'cwd': str(cwd), 'PYTHONPATH': env.get('PYTHONPATH'), 'exit_code': result.returncode, 'seconds': round(time.monotonic()-start,2), 'log': str(path.relative_to(repo))}
    manifest['runs'].append(record)
    (receipts/'inventory-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'END {label} {kind} rc={result.returncode} elapsed={record["seconds"]}', flush=True)
    return result.returncode
for label, revision in revisions.items():
    dest = scratch / label
    dest.mkdir()
    archive = scratch / f'{label}.tar'
    subprocess.run(['git','archive','-o',str(archive),revision],cwd=repo,check=True)
    with tarfile.open(archive) as f:
        f.extractall(dest, members=[m for m in f.getmembers() if not m.issym() and not m.islnk()], filter='data')
    archive.unlink()
    (dest/'python/owl/rs.abi3.so').symlink_to(repo/'python/owl/rs.abi3.so')
    env = envbase.copy()
    env['PYTHONPATH'] = str(dest/'python')
    run(label,'import-path',[str(python),'-c','import owl, owl.rs; print(owl.__file__); print(owl.rs.__file__)'],dest,env)
    run(label,'pytest-collect',[str(python),'-m','pytest','--collect-only','-q','tests'],dest,env)
    run(label,'cargo-engine-list',['cargo','test','--manifest-path','engine_rs/Cargo.toml','--locked','--offline','--','--list'],dest,env)
    run(label,'cargo-root-list',['cargo','test','--locked','--offline','--','--list'],dest,env)
print('DONE', str(scratch), flush=True)
