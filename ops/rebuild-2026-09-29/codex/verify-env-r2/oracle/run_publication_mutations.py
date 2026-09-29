import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/Users/poonszesen/kg-v3-env')
SCRATCH=Path('/private/tmp/kg-env-r2-oracle')
OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r2/oracle'
SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py'
BASE=SOURCE.read_bytes();text=BASE.decode();results=[]
cases=[
 ('premature-validation-publication',
  '    validate_arrays(manifest, arrays)\n    payload, expanded_sha, expanded_size = deterministic_npz(arrays)',
  '    path.write_bytes(b"premature fixture")\n    validate_arrays(manifest, arrays)\n    payload, expanded_sha, expanded_size = deterministic_npz(arrays)',
  'failed_validation_publishes_no_partial_fixture'),
 ('premature-worker-publication',
  '        supervise(command, args.max_seconds, args.max_rss_mib)',
  '        args.output.write_bytes(b"premature fixture")\n        supervise(command, args.max_seconds, args.max_rss_mib)',
  'worker_failure_never_publishes_target'),
]
try:
 for name,old,new,selector in cases:
  assert text.count(old)==1
  code=text.replace(old,new).encode();SOURCE.write_bytes(code)
  for cache in (SOURCE.parent/'__pycache__').glob('record_kaggriculture_env_reference.*.pyc'):cache.unlink()
  command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-c','/dev/null',f'--rootdir={SCRATCH}',f'--confcutdir={SCRATCH}','-p','no:cacheprovider',str(SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py'),'-q','-k',selector]
  p=subprocess.run(command,cwd=SCRATCH,env=os.environ|dict(GIT_DIR=str(ROOT/'.git'),GIT_WORK_TREE=str(ROOT)),capture_output=True,text=True,timeout=15)
  output=p.stdout+p.stderr;(OUT/f'{name}.log').write_text(output)
  assert p.returncode==1 and 'assert not' in output,(name,p.returncode,output)
  results.append(dict(name=name,status='killed',exit=p.returncode,mutant_sha256=hashlib.sha256(code).hexdigest(),summary=output.strip().splitlines()[-1]))
  print(name,'killed',flush=True)
finally:
 SOURCE.write_bytes(BASE);assert SOURCE.read_bytes()==BASE
 (OUT/'publication-mutations.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(BASE).hexdigest(),restored_byte_exact=SOURCE.read_bytes()==BASE,mutations=results),indent=2)+'\n')
