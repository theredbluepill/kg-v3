from pathlib import Path
import hashlib,importlib.util,json,shutil
ROOT=Path.cwd(); SCRATCH=ROOT/'.codex-tmp/verify-e197528-python'; OUT=ROOT/'ops/rebuild-2026-09-29/verify-e197528-independent/mutations-python'
spec=importlib.util.spec_from_file_location('real_oracle',SCRATCH/'scripts/kaggriculture_observation_oracle/regenerate.py'); oracle=importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)
paths=(*oracle.SOURCE_PATHS,*(p for _,p in oracle.ENGINE_PATHS),*(f'engine_rs/fixtures/episode-{ep}.jsonl.gz' for ep in oracle.EPISODES),'engine_rs/src/py_random.rs','engine_rs/Cargo.toml')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path in paths:
 p=SCRATCH/path;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,p)
root_before={p:sha(ROOT/p) for p in paths}
receipts=[]
for path,comment in [('engine_rs/src/py_random.rs',b'\n// Independent verification: source custody mutation.\n'),('engine_rs/Cargo.toml',b'\n# Independent verification: source custody mutation.\n')]:
 p=SCRATCH/path; before=p.read_bytes(); entry={'path':path,'before_sha256':sha(p),'root_sha256':sha(ROOT/path)}
 snapshot=oracle.source_snapshot();oracle.check_source_snapshot(snapshot)
 entry['baseline_check']='passed';entry['snapshot_root_commit']=snapshot[0];entry['path_in_snapshot']=path in dict(snapshot[1])
 try:
  p.write_bytes(before+comment);entry['mutant_sha256']=sha(p)
  try:
   oracle.check_source_snapshot(snapshot)
  except ValueError as err: entry['mutation_rejected']=True;entry['error']=str(err)
  else: entry['mutation_rejected']=False
 finally:
  p.write_bytes(before)
 entry['restored_sha256']=sha(p);entry['restoration_exact']=p.read_bytes()==before
 oracle.check_source_snapshot(snapshot);entry['restored_check']='passed';receipts.append(entry)
assert all(sha(ROOT/p)==digest for p,digest in root_before.items())
(OUT/'real_source_snapshot_receipts.json').write_text(json.dumps({'probes':receipts,'repository_unchanged':True,'commands':oracle.COMMAND_RECEIPTS},indent=2)+'\n')
print(json.dumps(receipts,indent=2))
