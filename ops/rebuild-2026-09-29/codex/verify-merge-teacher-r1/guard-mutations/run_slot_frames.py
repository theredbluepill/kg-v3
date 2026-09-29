from pathlib import Path
import hashlib, json, os, subprocess
root=Path.cwd()
out=root/'ops/rebuild-2026-09-29/codex/verify-merge-teacher-r1/guard-mutations'
scratch=out/'scratch'
source=scratch/'python/owl/model/kaggriculture_teacher.py'
original=source.read_bytes()
testfile=scratch/'tests/kaggriculture/test_verifier_slot_frames.py'
testfile.write_text('''import pytest
from owl.model.kaggriculture_teacher import slot_frames

@pytest.mark.parametrize("slot", [-1, 0, 2, 11, 12])
def test_invalid_slot_frames_rejected(slot):
    with pytest.raises(ValueError, match="not a policy slot"):
        slot_frames(slot)
''')
env={**os.environ,'PYTHONPATH':str(scratch/'python')+os.pathsep+str(scratch),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
command=[str(root/'.venv/bin/python'),'-m','pytest','-q','-p','no:cacheprovider','--tb=short','tests/kaggriculture/test_verifier_slot_frames.py']
def run(name):
    result=subprocess.run(command,cwd=scratch,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/(name+'.log')).write_text(result.stdout)
    print(name,result.returncode,result.stdout,flush=True)
    return result
assert run('slot_frames_baseline').returncode==0
needle='raise ValueError(f"slot {slot} is not a policy slot {POLICY_SLOTS}")'
assert original.decode().count(needle)==1
try:
    source.write_text(original.decode().replace(needle,'return 0'))
    result=run('slot_frames_guard')
    assert result.returncode==1 and '5 failed' in result.stdout
finally:
    source.write_bytes(original)
    original_source=(root/'python/owl/model/kaggriculture_teacher.py').read_bytes()
    assert original==source.read_bytes()==original_source
    (out/'slot_frames_restoration.json').write_text(json.dumps({'source_sha256':hashlib.sha256(original_source).hexdigest(),'restored_scratch_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
assert run('slot_frames_restored').returncode==0
(out/'slot_frames_tests.py').write_bytes(testfile.read_bytes())
