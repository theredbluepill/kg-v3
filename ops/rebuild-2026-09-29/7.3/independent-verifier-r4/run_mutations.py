"""Independent source mutations; actual repository source is never modified."""
import hashlib,json,os,pathlib,resource,subprocess,sys,tempfile,time
OUT=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path.cwd()
SOURCE=ROOT/'python/owl/kaggriculture/replay_export.py'
ORIGINAL=SOURCE.read_bytes(); source=ORIGINAL.decode()
SHA=hashlib.sha256(ORIGINAL).hexdigest()
unit='tests/kaggriculture/test_replay_export.py::'
probes=str(OUT/'test_r4_probes.py')+'::'
fail=unit+'test_publication_failure_never_leaves_successful_custody'
order=unit+'test_episode_is_durable_before_custody_claims_it'
mutations=[
 ('visible_staging_name','f".{path.name}.{secrets.token_hex(8)}.tmp"','f"visible-{path.name}.{secrets.token_hex(8)}.tmp"',[probes+'test_dot_staging_precedes_visible_file']),
 ('skip_file_fsync','            os.fsync(stream.fileno())','            pass  # mutated: skip staged-file fsync',[order]),
 ('skip_directory_fsync','        _fsync_directory(directory)','        pass  # mutated: skip directory fsync',[order]),
 ('replace_existing_path','        os.link(temporary, path)','        os.replace(temporary, path)',[unit+'test_publication_never_replaces_a_path_created_meanwhile']),
 ('custody_before_episode','            files.insert(0, (episode_path, episode_bytes))','            files.append((episode_path, episode_bytes))',[order]),
 ('skip_published_cleanup','            path.unlink(missing_ok=True)','            pass  # mutated: leave published files',[fail]),
 ('skip_error_custody','                self._publish_failure_custody(\n                    replay, sidecar, custody_path, error=error\n                )','                pass  # mutated: suppress failed-publication custody',[fail]),
 ('skip_staging_cleanup','        temporary.unlink(missing_ok=True)','        pass  # mutated: leave staging file',[fail]),
 ('retain_failed_hash','            if key not in ("episode_sha256", "verification")','            if key != "verification"',[fail]),
 ('retain_failed_verification','            if key not in ("episode_sha256", "verification")','            if key != "episode_sha256"',[fail]),
 ('retire_on_error_custody_failure','            return\n        self._retire(replay)','            self._retire(replay)\n            return\n        self._retire(replay)',[unit+'test_unpublishable_error_custody_leaves_no_files_and_the_game_active']),
]
scratch=pathlib.Path(tempfile.mkdtemp(prefix='kg-t73-r4-mutations-', dir='/tmp'))
target=scratch/'replay_export.py'; target.write_bytes(ORIGINAL)
loader=scratch/'run_one.py'
loader.write_text('''import hashlib,importlib.util,pathlib,sys
root=pathlib.Path(sys.argv[1]); source=pathlib.Path(sys.argv[2]); sys.path.insert(0,str(root))
import owl.kaggriculture
name='owl.kaggriculture.replay_export'
spec=importlib.util.spec_from_file_location(name,source)
module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
owl.kaggriculture.replay_export=module
print('MUTATION_SOURCE',module.__file__,hashlib.sha256(source.read_bytes()).hexdigest(),flush=True)
import pytest
raise SystemExit(pytest.main(sys.argv[3:]))
''')
results=[]
def run(name,tests):
 start=time.monotonic()
 command=[str(ROOT/'.venv/bin/python'),str(loader),str(ROOT),str(target),*tests,'-q']
 with (OUT/f'mutation-{name}.log').open('w') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=45,env=os.environ)
 text=(OUT/f'mutation-{name}.log').read_text()
 return {'exit_code':result.returncode,'wall_seconds':round(time.monotonic()-start,3),'log':f'mutation-{name}.log','summary':[line for line in text.splitlines() if ' failed' in line or ' passed' in line or line.startswith('FAILED ')]}
all_tests=list(dict.fromkeys(t for *_,tests in mutations for t in tests))
baseline=run('baseline',all_tests)
assert baseline['exit_code']==0,baseline
for name,old,new,tests in mutations:
 assert source.count(old)==1,(name,source.count(old))
 mutated=source.replace(old,new)
 target.write_text(mutated)
 try:
  outcome=run(name,tests)
 finally:
  target.write_bytes(ORIGINAL)
 restored_sha=hashlib.sha256(target.read_bytes()).hexdigest()
 assert restored_sha==SHA
 restored=run(name+'-restored',tests)
 row={'name':name,'old':old,'new':new,'tests':tests,'mutated_sha256':hashlib.sha256(mutated.encode()).hexdigest(),'source_sha256':SHA,'scratch_restored_sha256':restored_sha,'outcome':outcome,'restored':restored,'detected':outcome['exit_code']==1 and restored['exit_code']==0}
 results.append(row)
 (OUT/'mutations.json').write_text(json.dumps({'scratch':str(scratch),'source':str(SOURCE),'source_sha256':SHA,'baseline':baseline,'mutations':results},indent=2)+'\n')
 print(name, 'DETECTED' if row['detected'] else 'NOT DETECTED',flush=True)
 assert row['detected'],row
assert SOURCE.read_bytes()==ORIGINAL
receipt={'scratch':str(scratch),'source_sha256':SHA,'scratch_restored_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'live_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'detected':len(results),'total':len(mutations)}
(OUT/'mutation-restoration.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
