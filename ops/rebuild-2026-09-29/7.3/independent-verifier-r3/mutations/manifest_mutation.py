"""Mutate an independent minimal manifest scratch file; restore exact bytes."""
import copy, hashlib, json, tempfile
from pathlib import Path
from tests.tools.test_replay_trim_manifest import baseline, updater
root=Path.cwd()
out=root/'ops/rebuild-2026-09-29/7.3/independent-verifier-r3/mutations'
before=baseline()
original=json.dumps(before,indent=2).encode()
changed=copy.deepcopy(before)
changed['retained'][0]['sha256']='changed-kernel'
mutant=json.dumps(changed,indent=2).encode()
sha=lambda data:hashlib.sha256(data).hexdigest()
assert updater.regenerate(before,json.loads(original),set())['retained']==before['retained']
with tempfile.TemporaryDirectory(prefix='replay-manifest-r2-',dir=root/'.codex-tmp') as d:
    path=Path(d)/'manifest.json'
    path.write_bytes(original)
    try:
        path.write_bytes(mutant)
        try:
            updater.regenerate(before,json.loads(path.read_bytes()),set())
        except ValueError as e:
            error=str(e)
        else:
            raise AssertionError('Frozen kernel mutation survived')
        assert 'unexpected manifest input' in error
    finally:
        path.write_bytes(original)
        assert path.read_bytes()==original
    receipt={'mutation':'retained kernel hash changed','error':error,'baseline_sha256':sha(original),'mutant_sha256':sha(mutant),'restored_sha256':sha(path.read_bytes()),'restored_byte_identical':path.read_bytes()==original}
assert updater.regenerate(before,json.loads(original),set())['retained']==before['retained']
(out/'manifest-mutation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
