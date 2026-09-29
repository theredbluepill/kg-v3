"""Inject an OSError on final manifest replacement; no real fixture is touched."""
from pathlib import Path
import importlib.util,os,json,hashlib,tempfile
root=Path('/Users/poonszesen/kg-v3-env');scratch=Path('/private/tmp/kg-env-r3-oracle');out=root/'ops/rebuild-2026-09-29/codex/verify-env-r3/oracle'
os.environ.update(GIT_DIR=str(root/'.git'),GIT_WORK_TREE=str(root))
spec=importlib.util.spec_from_file_location('examples',scratch/'tests/tools/test_record_kaggriculture_env_reference.py');ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex);m=ex.recorder
results=[]
def h(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
for existing in (False,True):
 with tempfile.TemporaryDirectory(dir=scratch) as directory:
  target=Path(directory)/'oracle.npz'
  if existing:ex.publish_example(Path(directory))
  before={str(p.suffix):h(p) for p in (target,target.with_suffix('.json'))}
  manifest,arrays=ex.example();arrays['rewards'][0,0,0]=0.125;manifest['arrays']=m.metadata(arrays)
  original=m.os.replace;calls=[]
  def replace(source,destination):
   calls.append([str(source),str(destination)])
   if Path(destination)==target.with_suffix('.json'):raise OSError('injected second replacement failure')
   return original(source,destination)
  m.os.replace=replace
  try:m.publish_fixture(target,manifest,arrays)
  except OSError as exc:error=str(exc)
  else:raise AssertionError('injection not reached')
  finally:m.os.replace=original
  after={str(p.suffix):h(p) for p in (target,target.with_suffix('.json'))}
  try:m.load_fixture(target)
  except ValueError as exc:loader_error=str(exc)
  else:raise AssertionError('inconsistent pair accepted')
  assert after['.npz'] is not None and after['.npz']!=before['.npz'];assert after['.json']==before['.json']
  result=dict(existing_output=existing,injected_error=error,before_sha256=before,after_sha256=after,replace_calls=calls,loader_error=loader_error)
  results.append(result);print(json.dumps(result),flush=True)
(out/'pair-publication-probe-details.json').write_text(json.dumps(results,indent=2)+'\n')
