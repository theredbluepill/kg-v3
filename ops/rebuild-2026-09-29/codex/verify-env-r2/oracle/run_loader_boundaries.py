import ast,hashlib,importlib.util,json,os,tempfile
from pathlib import Path
ROOT=Path('/Users/poonszesen/kg-v3-env');SCRATCH=Path('/private/tmp/kg-env-r2-oracle');OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r2/oracle';SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py'
BASE=SOURCE.read_bytes();TEXT=BASE.decode();os.environ.update(GIT_DIR=str(ROOT/'.git'),GIT_WORK_TREE=str(ROOT))
def load(code):
 SOURCE.write_bytes(code);spec=importlib.util.spec_from_file_location('boundaries',SOURCE);module=importlib.util.module_from_spec(spec);exec(compile(code,str(SOURCE),'exec'),module.__dict__);return module
def omission(needle):
 offsets=[0]
 for line in TEXT.splitlines(keepends=True):offsets.append(offsets[-1]+len(line))
 hits=[]
 for node in ast.walk(ast.parse(TEXT)):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2 and needle in ast.get_source_segment(TEXT,node.args[1]):
   arg=node.args[0];hits.append((offsets[arg.lineno-1]+arg.col_offset,offsets[arg.end_lineno-1]+arg.end_col_offset))
 assert len(hits)==1,(needle,hits)
 a,b=hits[0];return (TEXT[:a]+'True'+TEXT[b:]).encode()
spec=importlib.util.spec_from_file_location('boundary_examples',SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py');examples=importlib.util.module_from_spec(spec);spec.loader.exec_module(examples)
good,arrays=examples.example();results=[]
try:
 for name,needle in [('compressed-cap','compressed size budget exceeded'),('expanded-cap','expanded size budget exceeded'),('compressed-hash','"fixture hash differs"'),('expanded-hash','expanded fixture hash differs'),('header-layout','numpy header shape/dtype/layout')]:
  outcomes=[]
  for changed in [False,True]:
   code=omission(needle) if changed else BASE;module=load(code)
   manifest=module.make_manifest(arrays,good['coverage'],module.source_identity())
   payload,sha,nbytes=module.deterministic_npz(arrays)
   manifest.update(compressed_bytes=len(payload),fixture_sha256=module.sha(payload),expanded_bytes=nbytes,expanded_sha256=sha)
   if name=='compressed-cap':module.MAX_COMPRESSED=len(payload)-1
   elif name=='expanded-cap':module.MAX_EXPANDED=nbytes-1
   elif name=='compressed-hash':manifest['fixture_sha256']='0'*64
   elif name=='expanded-hash':manifest['expanded_sha256']='0'*64
   elif name=='header-layout':manifest['arrays']['rewards']['dtype']='<f8'
   old_load=module.np.load
   if name=='header-layout':module.np.load=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('numpy-load sentinel'))
   try:
    with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
     path=Path(folder)/'coherent.npz';path.write_bytes(payload);path.with_suffix('.json').write_bytes(module.json_bytes(manifest));module.load_fixture(path)
    outcome='accepted'
   except (ValueError,RuntimeError) as error:outcome=f'{type(error).__name__}: {error}'
   finally:module.np.load=old_load
   outcomes.append(outcome)
  if name=='expanded-cap':assert outcomes==['ValueError: expanded size budget exceeded','ValueError: expanded size budget differs']
  elif name=='header-layout':assert outcomes==['ValueError: numpy header shape/dtype/layout differs','RuntimeError: numpy-load sentinel']
  else:assert outcomes[0].startswith('ValueError:') and outcomes[1]=='accepted',(name,outcomes)
  results.append(dict(name=name,baseline=outcomes[0],mutant=outcomes[1]));print(name,outcomes,flush=True)
finally:
 SOURCE.write_bytes(BASE);assert SOURCE.read_bytes()==BASE
 (OUT/'loader-boundaries.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(BASE).hexdigest(),restored_byte_exact=SOURCE.read_bytes()==BASE,cases=results),indent=2)+'\n')
