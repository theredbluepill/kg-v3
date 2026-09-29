"""Exercise the trunk half of the combined head/trunk teacher chunking oracle."""
import hashlib, json, os, subprocess
from pathlib import Path
ROOT=Path.cwd(); OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-merge-teacher-r2/numerical-mutations'
SCRATCH=Path(json.loads((OUT/'scratch-receipt.json').read_text())['scratch'])
ENV={**os.environ,'PYTHONPATH':str(SCRATCH/'python')+os.pathsep+str(SCRATCH),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
rel='python/owl/model/kaggriculture.py'; path=SCRATCH/rel; original=path.read_bytes()
old='trunk(x[start : start + chunk], token_mask[start : start + chunk], None)'
new='trunk(x[:chunk], token_mask[:chunk], None)'
assert original.decode().count(old)==1
try:
    path.write_text(original.decode().replace(old,new))
    with (OUT/'trunk-chunk-reuses-first.log').open('w') as f:
        p=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','tests/kaggriculture/test_teacher.py::test_head_and_trunk_chunking_match_the_unchunked_kl','-q'],cwd=SCRATCH,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=150)
finally:
    path.write_bytes(original)
result={'name':'trunk-chunk-reuses-first','file':rel,'tests':['tests/kaggriculture/test_teacher.py::test_head_and_trunk_chunking_match_the_unchunked_kl'],'exit_code':p.returncode,'summary':(OUT/'trunk-chunk-reuses-first.log').read_text().splitlines()[-1],'restored':path.read_bytes()==original,'sha256':hashlib.sha256(original).hexdigest()}
(OUT/'trunk-result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
with (OUT/'final-restored-union.log').open('w') as f:
    p=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','tests/kaggriculture/test_teacher.py','tests/kaggriculture/test_merge_teacher_guards.py','tests/kaggriculture/test_teacher_precision_probe.py','-q'],cwd=SCRATCH,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=150)
print('final baseline',p.returncode,(OUT/'final-restored-union.log').read_text().splitlines()[-1])
assert p.returncode==0
copied=json.loads((OUT/'copied-before.json').read_text()); after={rel:hashlib.sha256((SCRATCH/rel).read_bytes()).hexdigest() for rel in copied}
(OUT/'final-full-copy-restoration.json').write_text(json.dumps({'copied_files':len(copied),'sha256_before':copied,'sha256_after':after,'mismatched':[k for k in copied if copied[k]!=after[k]]},indent=2)+'\n')
assert copied==after
