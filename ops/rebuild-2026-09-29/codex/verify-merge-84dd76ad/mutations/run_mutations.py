from pathlib import Path
import difflib, hashlib, json, os, subprocess, sys
ROOT = Path('/Users/poonszesen/kg-v3-merge3')
OUT = Path('/private/tmp/kg-merge84-contracts')
SCRATCH = OUT/'scratch'
env=os.environ.copy()
env['GIT_DIR']=(OUT/'git_dir.txt').read_text()
env['PYTHONDONTWRITEBYTECODE']='1'
checker=SCRATCH/'scripts/check_engine_trim.py'
original=checker.read_bytes()
command=[str(ROOT/'.venv/bin/python'), '-B', '-m', 'pytest', '-c', '/dev/null', '--confcutdir', str(SCRATCH), str(SCRATCH/'tests/tools/test_check_engine_trim.py'), '-q']
results=[]
mutations=[
 ('baseline_checker',None,None),
 ('task11_editable','if path not in EDITABLE:', 'if path.endswith(".rs") and path != "engine_rs/src/lib.rs":'),
 ('task11b_trace_hash','_require(sha(data) == _digest(entry["sha256"], name), f"{name}: trace hash")', '_digest(entry["sha256"], name)'),
 ('task12_authored_grammar', '            "engine_rs/tests/grammar_kernel.rs",\n', ''),
 ('merge_authored_generated', '            GENERATED_MANIFEST,\n', ''),
]
try:
 for label,before,after in mutations:
  source=original.decode()
  if before is not None:
   assert source.count(before)==1,(label,source.count(before))
   source=source.replace(before,after)
   (OUT/f'{label}.diff').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),source.splitlines(True),fromfile='merged',tofile=label)))
  checker.write_text(source)
  proc=subprocess.run(command,cwd=SCRATCH,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  (OUT/f'{label}.log').write_text(proc.stdout)
  tail=[line for line in proc.stdout.splitlines() if 'passed' in line or line.startswith('FAILED') or 'error' in line]
  print(label,'exit',proc.returncode,*tail,sep='\n',flush=True)
  results.append({'mutation':label,'exit':proc.returncode,'summary':tail})
  assert proc.returncode==(0 if before is None else 1),(label,proc.returncode)
finally:
 checker.write_bytes(original)
 assert checker.read_bytes()==original
 (OUT/'checker-results.json').write_text(json.dumps(results,indent=2)+'\n')
