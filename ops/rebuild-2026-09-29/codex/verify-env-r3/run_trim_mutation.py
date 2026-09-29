from pathlib import Path
import hashlib, json, os, shutil, subprocess
root=Path.cwd(); out=root/'ops/rebuild-2026-09-29/codex/verify-env-r3'; scratch=root/'.codex-tmp/verify-env-r3-trim'
paths=['tests/tools/test_check_engine_trim.py','scripts/check_engine_trim.py','engine_rs/TRIM_MANIFEST.json','src/kaggriculture/grammar_kernel_tests.rs']
for p in paths:
    dst=scratch/p; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/p,dst)
path=scratch/paths[-1]; original=path.read_bytes(); results=[]
command=[str(root/'.venv/bin/python'),'-m','pytest','--noconftest',str(scratch/paths[0]),'-k','no_authored_grammar_path_include_after_root_engine_edge','-q']
def run(phase):
    proc=subprocess.run(command,cwd=scratch,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/f'trim-guard-{phase}.log').write_text(proc.stdout)
    results.append({'phase':phase,'exit_status':proc.returncode})
try:
    run('baseline')
    path.write_bytes(original+b'\n#[path = "grammar.rs"]\nmod duplicate_grammar;\n')
    run('mutant')
finally:
    path.write_bytes(original)
run('restored')
assert [r['exit_status'] for r in results]==[0,1,0],results
assert all((scratch/p).read_bytes()==(root/p).read_bytes() for p in paths)
record={'command':command,'results':results,'original_sha256':hashlib.sha256(original).hexdigest(),'restored_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'trim-guard-mutation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
