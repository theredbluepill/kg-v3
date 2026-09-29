from pathlib import Path
import hashlib, importlib.util, json, shutil
ROOT=Path.cwd()
OUT=ROOT/'ops/rebuild-2026-09-29/verify-b8747b6-independent'
SCRATCH=OUT/'source-scratch'
SCRATCH.mkdir(exist_ok=True)
rel='scripts/kaggriculture_observation_oracle/regenerate.py'
target=SCRATCH/rel
target.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(ROOT/rel,target)
spec=importlib.util.spec_from_file_location('custody_probe',target)
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
paths=(*oracle.SOURCE_PATHS,*(p for _,p in oracle.ENGINE_PATHS),*(f'engine_rs/fixtures/episode-{ep}.jsonl.gz' for ep in oracle.EPISODES))
for path in paths:
 p=SCRATCH/path
 p.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(ROOT/path,p)
sha=lambda b:hashlib.sha256(b).hexdigest()
receipts=[]
for path in ('engine_rs/Cargo.toml','engine_rs/src/py_random.rs','engine_rs/src/econ_attrib.rs'):
 p=SCRATCH/path
 original=p.read_bytes()
 snapshot=oracle.source_snapshot()
 oracle.check_source_snapshot(snapshot)
 entry={'path':path,'captured':path in dict(snapshot[1]),'root_commit':snapshot[0],'before':sha(original),'baseline':'pass'}
 try:
  p.write_bytes(original+(b'\n# custody probe\n' if path.endswith('.toml') else b'\n// custody probe\n'))
  entry['mutant']=sha(p.read_bytes())
  try: oracle.check_source_snapshot(snapshot)
  except ValueError as e: entry['rejected']=True;entry['error']=str(e)
  else: entry['rejected']=False
 finally: p.write_bytes(original)
 oracle.check_source_snapshot(snapshot)
 entry['restored']=sha(p.read_bytes())
 entry['restored_exact']=p.read_bytes()==original
 assert entry['rejected'] and entry['restored_exact']
 receipts.append(entry)
for path in ('engine_rs/src/new_module.rs','engine_rs/build.rs'):
 p=SCRATCH/path
 assert not p.exists()
 snapshot=oracle.source_snapshot()
 entry={'path':path}
 try:
  p.write_text('// new engine input\n')
  for label,call in [('capture',oracle.source_snapshot),('recheck',lambda:oracle.check_source_snapshot(snapshot))]:
   try:call()
   except ValueError as e:entry[label]={'rejected':True,'error':str(e)}
   else:entry[label]={'rejected':False}
 finally:p.unlink()
 oracle.check_source_snapshot(snapshot)
 entry['restored_exact']=not p.exists()
 assert entry['capture']['rejected'] and entry['recheck']['rejected']
 receipts.append(entry)
for path in paths: assert (SCRATCH/path).read_bytes()==(ROOT/path).read_bytes()
(OUT/'source-snapshot-probes.json').write_text(json.dumps({'probes':receipts,'all_scratch_inputs_byte_exact':True,'commands':oracle.COMMAND_RECEIPTS},indent=2)+'\n')
print(json.dumps(receipts,indent=2))
