"""Build a done-bit mutant; the complete reference replay must reject it."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path(__file__).resolve().parents[3]
source = root / 'src/kaggriculture/env.rs'
original = source.read_bytes()
needle = b'transition.dones[i * 2..i * 2 + 2].fill(row.done);'
assert original.count(needle) == 1
assert (root/'tests/fixtures/kaggriculture_env_reference_v1.npz').is_file(), 'Missing full fixture: this mutation is PENDING (pod), never skipped'
receipts = []
def run(name, argv):
    result = subprocess.run(['python3',str(root/'ops/rebuild-2026-09-29/1.4/bounded.py'),'--name',name,'--',*argv],cwd=root,env=os.environ,check=False)
    receipts.append({'name':name,'exit_status':result.returncode})
    return result.returncode
try:
    source.write_bytes(original.replace(needle,b'transition.dones[i * 2..i * 2 + 2].fill(!row.done); // MUTATION'))
    assert run('g-done-mutation-build',['uv','run','--offline','maturin','develop','--locked']) == 0
    assert run('g-done-mutation-red',['uv','run','--offline','pytest','tests/kaggriculture/test_env_reference.py','-q']) == 1
    log=(root/'ops/rebuild-2026-09-29/1.4/g-done-mutation-red.log').read_text()
    assert 'first divergence game=0 seed=17000 step=0 seat=0' in log and 'field=dones' in log
finally:
    source.write_bytes(original)
    restored_build = run('g-done-restored-build',['uv','run','--offline','maturin','develop','--locked'])
    (root/'ops/rebuild-2026-09-29/1.4/reference-mutation.json').write_text(json.dumps({'mutations':receipts,'original_sha256':hashlib.sha256(original).hexdigest(),'restored_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
# Restore the installed extension too, not only its Rust source.
assert restored_build == 0
assert run('g-done-restored-green',['uv','run','--offline','pytest','tests/kaggriculture/test_env_reference.py','-q']) == 0
