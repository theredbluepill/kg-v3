"""Coherent scratch NPZ boundaries, distinguishing every pre-load header guard."""
from pathlib import Path
import ast, hashlib, importlib.util, io, json, os, tempfile, zipfile
import numpy as np
ROOT=Path('/Users/poonszesen/kg-v3-env');SCRATCH=Path('/private/tmp/kg-env-r3-oracle');OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r3/oracle';SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py';BASE=SOURCE.read_bytes();TEXT=BASE.decode()
os.environ.update(GIT_DIR=str(ROOT/'.git'),GIT_WORK_TREE=str(ROOT))
def load(code):
 SOURCE.write_bytes(code)
 spec=importlib.util.spec_from_file_location('loader_probe',SOURCE);m=importlib.util.module_from_spec(spec);exec(compile(code,str(SOURCE),'exec'),m.__dict__);return m
spec=importlib.util.spec_from_file_location('examples',SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py');examples=importlib.util.module_from_spec(spec);spec.loader.exec_module(examples)
def omit(needle):
 offsets=[0]
 for line in TEXT.splitlines(keepends=True): offsets.append(offsets[-1]+len(line))
 nodes=[node.args[0] for node in ast.walk(ast.parse(TEXT)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2 and needle in ast.get_source_segment(TEXT,node.args[1])]
 assert len(nodes)==1,(needle,len(nodes))
 n=nodes[0];a=offsets[n.lineno-1]+n.col_offset;b=offsets[n.end_lineno-1]+n.end_col_offset
 return (TEXT[:a]+'True'+TEXT[b:]).encode()
def probe(module,name):
 manifest,arrays=examples.example();manifest['sources']=module.source_identity()
 payload,_,_=module.deterministic_npz(arrays)
 with zipfile.ZipFile(io.BytesIO(payload)) as old: members={info.filename:(info,old.read(info)) for info in old.infolist()}
 field='rewards.npy';info,content=members[field]
 if name=='zip-timestamp': info.date_time=(1981,1,1,0,0,0)
 elif name=='header-version': content=content[:6]+bytes([2,0])+content[8:]
 elif name=='header-layout': manifest['arrays']['rewards']['shape']=[1]
 elif name in ('header-dimension','header-total'):
  shape=(module.MAX_EXPANDED+1,) if name=='header-dimension' else (module.MAX_EXPANDED,)
  manifest['arrays']['rewards']['shape']=list(shape)
  raw=io.BytesIO();np.lib.format.write_array_header_1_0(raw,dict(descr=np.dtype('float32').str,fortran_order=False,shape=shape));content=raw.getvalue()
 else:raise AssertionError(name)
 members[field]=(info,content)
 new=io.BytesIO();expanded=hashlib.sha256();expanded_size=0
 with zipfile.ZipFile(new,'w',compression=zipfile.ZIP_DEFLATED) as archive:
  for name2,(info,content) in members.items():archive.writestr(info,content);expanded.update(content);expanded_size+=len(content)
 payload=new.getvalue();manifest.update(fixture_sha256=module.sha(payload),compressed_bytes=len(payload),expanded_sha256=expanded.hexdigest(),expanded_bytes=expanded_size)
 with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
  path=Path(folder)/'probe.npz';path.write_bytes(payload);path.with_suffix('.json').write_bytes(module.json_bytes(manifest))
  old=module.np.load;module.np.load=lambda *_a,**_k:(_ for _ in ()).throw(RuntimeError('NumPy load sentinel'))
  try:module.load_fixture(path)
  finally:module.np.load=old
cases=[('zip-timestamp','zip timestamp differs'),('header-version','unexpected numpy header version'),('header-layout','numpy header shape/dtype/layout differs'),('header-dimension','numpy header dimension exceeds budget'),('header-total','numpy header expanded size exceeds budget')]
results=[]
try:
 for name,needle in cases:
  outcomes=[]
  for changed in (False,True):
   m=load(omit(needle) if changed else BASE)
   try:probe(m,name)
   except Exception as exc:outcome=f'{type(exc).__name__}: {exc}'
   else:outcome='accepted'
   outcomes.append(outcome)
  assert outcomes[0]==f'ValueError: {needle}',(name,outcomes)
  assert outcomes[1]!=outcomes[0],(name,outcomes)
  results.append(dict(name=name,guard=needle,baseline=outcomes[0],omission=outcomes[1]));print(name,outcomes,flush=True)
finally:
 SOURCE.write_bytes(BASE);assert SOURCE.read_bytes()==BASE
 (OUT/'loader-guard-probes.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(BASE).hexdigest(),restored_byte_exact=True,probes=results),indent=2)+'\n')
