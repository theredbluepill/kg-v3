from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path.cwd(); scratch=root/'.codex-tmp/verify-e197528-python'; out=root/'ops/rebuild-2026-09-29/verify-e197528-independent/mutations-python'
source=scratch/'scripts/kaggriculture_observation_oracle/regenerate.py'; original=source.read_bytes()
sha=lambda x:hashlib.sha256(x).hexdigest()
shutil.copyfile(scratch/'test_source_gap.py',out/'source-gap-test.py')
cmd=[str(root/'.venv/bin/python'),'-B','-m','pytest','-c',str(scratch/'pytest.ini'),'--confcutdir',str(scratch),'-q',str(scratch/'test_source_gap.py')]
receipts={}
def run(phase):
 p=subprocess.run(cmd,cwd=scratch,env=os.environ|{'PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/('source-gap-'+phase+'.log')).write_bytes(p.stdout)
 receipts[phase]={'exit':p.returncode,'sha256':sha(source.read_bytes())}
 return p.returncode
try:
 assert run('unmodified')==1
 text=original.decode(); anchor='    "src/kaggriculture/oracle_corpus.rs",\n'
 # Only SOURCE_PATHS has this standalone anchor.
 assert text.count(anchor)==1
 source.write_text(text.replace(anchor,anchor+'    "engine_rs/src/py_random.rs",\n    "engine_rs/Cargo.toml",\n'))
 assert run('fixed_scratch')==0
finally:
 source.write_bytes(original)
assert run('restored')==1
receipts['restoration_exact']=source.read_bytes()==original
(out/'source-gap-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(receipts,indent=2))
