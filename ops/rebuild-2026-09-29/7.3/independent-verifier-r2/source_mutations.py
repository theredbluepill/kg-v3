"""Two scratch-only guard removals; targeted pytest must discriminate each."""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
root = Path.cwd()
out = root / 'ops/rebuild-2026-09-29/7.3/independent-verifier-r2'
scratch = Path(tempfile.mkdtemp(prefix='replay-source-r2-', dir=root / '.codex-tmp'))
for name in ('Cargo.toml', 'Cargo.lock', 'build.rs'):
    shutil.copy2(root / name, scratch / name)
shutil.copytree(root / 'src', scratch / 'src')
(scratch / 'engine_rs').mkdir()
for name in ('Cargo.toml', 'Cargo.lock'):
    shutil.copy2(root / 'engine_rs' / name, scratch / 'engine_rs' / name)
shutil.copytree(root / 'engine_rs/src', scratch / 'engine_rs/src')
shutil.copytree(root / 'python/owl', scratch / 'python/owl', ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.pyc'))
path = scratch / 'src/kaggriculture/replay_export.rs'
original = path.read_bytes()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
base_env = dict(os.environ, PYO3_PYTHON=str(root / '.venv/bin/python'), CARGO_TARGET_DIR=str(root / 'target'), PYTHONPATH=str(scratch / 'python') + os.pathsep + str(root))
byte_test = 'tests/kaggriculture/test_replay_export_oracles.py::test_native_byte_round_trip_and_wide_seed'
completion_test = 'tests/kaggriculture/test_replay_export.py::test_every_successful_native_export_verifies'

def run(label, args, *, env=base_env, timeout=180):
    start = time.monotonic()
    with (out / (label + '.log')).open('w') as f:
        result = subprocess.run(args, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return {'command':args, 'returncode':result.returncode, 'seconds':time.monotonic()-start, 'log':label+'.log'}

def build(label):
    entry = run(label, ['cargo', 'rustc', '--manifest-path', str(scratch/'Cargo.toml'), '--locked', '--offline', '--features', 'extension-module', '--', '-C', 'link-arg=-undefined', '-C', 'link-arg=dynamic_lookup'])
    assert entry['returncode']==0, entry
    shutil.copy2(root / 'target/debug/librs.dylib', scratch / 'python/owl/rs.abi3.so')
    return entry

def tests(label, targets):
    code = 'from owl import rs; import pytest; print("extension:", rs.__file__); raise SystemExit(pytest.main(' + repr(['-q'] + targets) + '))'
    return run(label, [str(root / '.venv/bin/python'), '-c', code])

receipt = {'head':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(), 'scratch':str(scratch), 'original_sha256':sha(original), 'mutations':[]}
try:
    receipt['baseline_build'] = build('source-baseline-build')
    receipt['baseline'] = tests('source-baseline-tests', [byte_test, completion_test])
    assert receipt['baseline']['returncode']==0
    for label, old, new, target in [
        ('byte-guard', b'if canonical_json.as_bytes() != input_canonical.as_bytes() {', b'if false && canonical_json.as_bytes() != input_canonical.as_bytes() {', byte_test),
        ('completion-guard', b'if !tape.complete && done {', b'if false && !tape.complete && done {', completion_test),
    ]:
        assert original.count(old)==1
        changed = original.replace(old,new)
        path.write_bytes(changed)
        try:
            entry = {'name':label, 'mutated_sha256':sha(changed), 'build':build('source-'+label+'-build'), 'pytest':tests('source-'+label+'-tests', [target])}
            assert entry['pytest']['returncode']==1, entry
        finally:
            path.write_bytes(original)
            assert path.read_bytes()==original
        entry['restored_sha256']=sha(path.read_bytes())
        receipt['mutations'].append(entry)
        print(label, 'detected', flush=True)
    receipt['restored_build']=build('source-restored-build')
    receipt['restored']=tests('source-restored-tests',[byte_test,completion_test])
    assert receipt['restored']['returncode']==0
finally:
    path.write_bytes(original)
    receipt['restored_byte_identical']=path.read_bytes()==original
    receipt['live_source_unchanged']=(root/'src/kaggriculture/replay_export.rs').read_bytes()==original
    (out/'source-mutations.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'restored_byte_identical':receipt['restored_byte_identical'],'mutations':len(receipt['mutations'])}),flush=True)
