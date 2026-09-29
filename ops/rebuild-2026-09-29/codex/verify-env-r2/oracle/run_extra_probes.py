"""Scratch-only coherent custody probes; workers never run or compile."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import numpy as np

ROOT=Path('/Users/poonszesen/kg-v3-env')
SCRATCH=Path('/private/tmp/kg-env-r2-oracle')
OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r2/oracle'
SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py'
BASE=SOURCE.read_bytes()
TEXT=BASE.decode()
os.environ.update(GIT_DIR=str(ROOT/'.git'),GIT_WORK_TREE=str(ROOT))

def load(code):
    SOURCE.write_bytes(code)
    spec=importlib.util.spec_from_file_location('probe',SOURCE)
    module=importlib.util.module_from_spec(spec)
    # Compile exact current bytes so import timestamp caches cannot affect probes.
    exec(compile(code,str(SOURCE),'exec'),module.__dict__)
    return module

baseline=load(BASE)
spec=importlib.util.spec_from_file_location('example',SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py')
examples=importlib.util.module_from_spec(spec);spec.loader.exec_module(examples)
good,good_arrays=examples.example()

def omission(needle):
    offsets=[0]
    for line in TEXT.splitlines(keepends=True):offsets.append(offsets[-1]+len(line))
    found=[]
    for node in ast.walk(ast.parse(TEXT)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2 and needle in ast.get_source_segment(TEXT,node.args[1]):
            arg=node.args[0];found.append((offsets[arg.lineno-1]+arg.col_offset,offsets[arg.end_lineno-1]+arg.end_col_offset))
    assert len(found)==1,(needle,found)
    start,end=found[0]
    return (TEXT[:start]+'True'+TEXT[end:]).encode()

def arrays_probe(module,name):
    manifest=copy.deepcopy(good);arrays={k:v.copy() for k,v in good_arrays.items()}
    if name=='coverage-counter':manifest['coverage'][0]['sell_cash']+=1
    elif name=='coverage-positive':manifest['coverage'][0]['hires']=0
    elif name=='coverage-keys':manifest['coverage'][0]['extra']=1
    elif name=='coverage-games':manifest['coverage'].pop()
    elif name=='array-dtype':arrays['rewards']=arrays['rewards'].astype(np.float64)
    elif name=='array-shape':arrays['rewards']=arrays['rewards'][:-1].copy()
    manifest['arrays']=module.metadata(arrays)
    module.validate_arrays(manifest,arrays)

def manifest_probe(module,name):
    manifest=module.make_manifest(good_arrays,good['coverage'],module.source_identity())
    manifest.update(compressed_bytes=1,expanded_bytes=1)
    if name=='source-custody':manifest['sources']['reference']='bad'
    elif name=='manifest-keys':manifest['extra']=1
    elif name=='manifest-recipe':manifest['seeds'][0]+=1
    elif name=='compressed-cap':manifest['compressed_bytes']=module.MAX_COMPRESSED+1
    elif name=='expanded-cap':manifest['expanded_bytes']=module.MAX_EXPANDED+1
    module.validate_manifest(manifest)

def load_probe(module,name):
    manifest=module.make_manifest(good_arrays,good['coverage'],module.source_identity())
    payload,expanded_sha,expanded_size=module.deterministic_npz(good_arrays)
    manifest.update(fixture_sha256=module.sha(payload),compressed_bytes=len(payload),expanded_sha256=expanded_sha,expanded_bytes=expanded_size)
    if name=='compressed-size':manifest['compressed_bytes']+=1
    elif name=='compressed-hash':manifest['fixture_sha256']='0'*64
    elif name=='expanded-hash':manifest['expanded_sha256']='0'*64
    with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
        path=Path(folder)/'probe.npz';path.write_bytes(payload)
        path.with_suffix('.json').write_bytes(module.json_bytes(manifest))
        module.load_fixture(path)

cases=[
 ('coverage-counter','coverage counter differs',arrays_probe),
 ('coverage-positive','coverage missing',arrays_probe),
 ('coverage-keys','coverage keys differ',arrays_probe),
 ('coverage-games','coverage game inventory',arrays_probe),
 ('array-dtype','array dtype/layout',arrays_probe),
 ('array-shape','array shape differs',arrays_probe),
 ('source-custody','reference/recorder/policy source custody',manifest_probe),
 ('manifest-keys','manifest key inventory',manifest_probe),
 ('manifest-recipe','manifest recipe differs',manifest_probe),
 ('compressed-cap','compressed size budget exceeded',manifest_probe),
 ('expanded-cap','expanded size budget exceeded',manifest_probe),
 ('compressed-size','fixture compressed size differs',load_probe),
 ('compressed-hash','"fixture hash differs"',load_probe),
 ('expanded-hash','expanded fixture hash differs',load_probe),
]
results=[]
try:
    for name,needle,probe in cases:
        baseline=load(BASE)
        try:probe(baseline,name)
        except ValueError as error:before=str(error)
        else:raise AssertionError(('baseline accepted invalid input',name))
        code=omission(needle);mutant=load(code)
        probe(mutant,name)
        results.append(dict(name=name,baseline=before,mutant='accepted invalid input; omission detected',mutant_sha256=hashlib.sha256(code).hexdigest()))
        print(name,'killed',flush=True)
    for name,needle,args in [
        ('fixed-recipe','exact pinned 16-game',['--games','2']),
        ('positive-budget','positive watchdog budget',['--max-seconds','0']),
        ('mac-budget','Mac execution budget',['--max-seconds','200']),
    ]:
        outcomes=[]
        for changed in [False,True]:
            module=load(omission(needle) if changed else BASE)
            old=module.sys.argv
            module.sys.argv=['recorder','--worker',*args]
            module.record_worker=lambda *_: (_ for _ in ()).throw(RuntimeError('worker sentinel: no work executed'))
            try:module.main()
            except (ValueError,RuntimeError) as error:outcome=f'{type(error).__name__}: {error}'
            finally:module.sys.argv=old
            assert outcome.startswith('RuntimeError' if changed else 'ValueError'),(name,outcome)
            outcomes.append(outcome)
        results.append(dict(name=name,baseline=outcomes[0],mutant=outcomes[1]))
        print(name,'killed with worker sentinel',flush=True)
finally:
    SOURCE.write_bytes(BASE)
    assert SOURCE.read_bytes()==BASE
    (OUT/'extra-probes.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(BASE).hexdigest(),restored_byte_exact=SOURCE.read_bytes()==BASE,main_source_unchanged=(ROOT/'scripts/record_kaggriculture_env_reference.py').read_bytes()==BASE,cases=results),indent=2)+'\n')
