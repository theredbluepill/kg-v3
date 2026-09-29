"""Scratch source/export/publication custody guard omissions with inert inputs."""
from pathlib import Path
import ast,hashlib,importlib.util,io,json,os,tarfile,tempfile
ROOT=Path('/Users/poonszesen/kg-v3-env');SCRATCH=Path('/private/tmp/kg-env-r3-oracle');OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r3/oracle';SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py';BASE=SOURCE.read_bytes();TEXT=BASE.decode()
os.environ.update(GIT_DIR=str(ROOT/'.git'),GIT_WORK_TREE=str(ROOT))
def load(code):
 SOURCE.write_bytes(code);spec=importlib.util.spec_from_file_location('source_boundary',SOURCE);m=importlib.util.module_from_spec(spec);exec(compile(code,str(SOURCE),'exec'),m.__dict__);return m
def omit(needle):
 offsets=[0]
 for line in TEXT.splitlines(keepends=True):offsets.append(offsets[-1]+len(line))
 nodes=[node.args[0] for node in ast.walk(ast.parse(TEXT)) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2 and needle in ast.get_source_segment(TEXT,node.args[1])]
 assert len(nodes)==1,(needle,len(nodes))
 n=nodes[0];a=offsets[n.lineno-1]+n.col_offset;b=offsets[n.end_lineno-1]+n.end_col_offset;return (TEXT[:a]+'True'+TEXT[b:]).encode()
def probe(m,name):
 if name in ('archive-path','archive-type'):
  output=io.BytesIO()
  with tarfile.open(fileobj=output,mode='w') as a:
   item=tarfile.TarInfo('../escape' if name=='archive-path' else 'link')
   if name=='archive-type':item.type=tarfile.SYMTYPE;item.linkname='target'
   a.addfile(item)
  m.reference_archive=lambda:output.getvalue();m.reference_sources()
 elif name=='resolved-reference':
  original=m.subprocess.check_output;calls=[]
  def fake(*_a,**_k):
   calls.append(1)
   if len(calls)==1:return b'badcommit'
   raise RuntimeError('archive sentinel')
  m.subprocess.check_output=fake
  try:m.reference_archive()
  finally:m.subprocess.check_output=original
 elif name=='publication-source-drift':
  spec=importlib.util.spec_from_file_location('examples',SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py');ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
  manifest,arrays=ex.example();identity=m.source_identity();manifest['sources']=identity;calls=[]
  def changing_source():
   calls.append(1);return identity if len(calls)<3 else identity|dict(reference='drift')
  m.source_identity=changing_source
  with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
   p=Path(d)/'probe.npz'
   try:m.publish_fixture(p,manifest,arrays)
   except ValueError:
    assert not p.exists() and not p.with_suffix('.json').exists();raise
   assert p.exists() and p.with_suffix('.json').exists()
 else:raise AssertionError(name)
results=[]
try:
 for name,needle in [('archive-path','unsafe reference archive path'),('archive-type','reference archive requires regular files'),('resolved-reference','reference source commit differs'),('publication-source-drift','source drift before publication')]:
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
 (OUT/'source-boundary-probes.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(BASE).hexdigest(),restored_byte_exact=True,probes=results),indent=2)+'\n')
