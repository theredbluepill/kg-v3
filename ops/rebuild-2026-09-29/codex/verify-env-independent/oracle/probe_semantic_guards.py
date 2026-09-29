from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os
import numpy as np
ROOT=Path('/Users/poonszesen/kg-v3-env'); SCRATCH=Path('/private/tmp/kg-verify-env-oracle-20260929'); OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-independent/oracle'
SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py'; BASE=SOURCE.read_bytes(); text=BASE.decode()
os.environ.update(GIT_DIR=str(ROOT/'.git'),GIT_WORK_TREE=str(ROOT))
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
tools=load(SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py','probe_examples'); good,arrays=tools.example()
def mutant(needle):
    offsets=[0]
    for line in text.splitlines(keepends=True): offsets.append(offsets[-1]+len(line))
    found=[]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2 and needle in ast.get_source_segment(text,node.args[1]):
            a=node.args[0];found.append((offsets[a.lineno-1]+a.col_offset,offsets[a.end_lineno-1]+a.end_col_offset))
    assert len(found)==1,(needle,found)
    a,b=found[0];return (text[:a]+'True'+text[b:]).encode()
def update(m,a,name):
    if name=='length-range':
        a['lengths'][0,0,0]=253
        a['program_offsets']=np.concatenate(([0],np.cumsum(a['lengths'].reshape(-1)))).astype(np.int64)
        a['tokens']=np.pad(a['tokens'],((0,251),(0,0)))
    elif name=='offsets': a['program_offsets'][1]+=1
    elif name=='indices': a['transition_indices'][0,0]+=1
    elif name=='seeds': a['next_seed'][0]+=1
    elif name=='steps': a['terminal_steps'][0]-=1
    elif name=='done': a['dones'][0,0,0]=True
    elif name=='terminal-values': a['terminal_econ'][0,0,3]+=1
    elif name=='winner': a['terminal_winner'][0]=0
    elif name=='econ-continuity': a['econ_before'][0,1,0,0]=0
    elif name=='bank-continuity': a['banks_before'][0,1,0]=1
    elif name=='nonfinite': a['rewards'][0,0,0]=np.nan
    elif name=='monotonic': a['econ_before'][0,0,0,3]=-1
    elif name=='coverage-counter': m['coverage'][0]['sell_cash']+=1
    elif name=='coverage-positive': m['coverage'][0]['hires']=0
    elif name=='dtype': a['rewards']=a['rewards'].astype(np.float64)
    elif name=='shape': a['rewards']=a['rewards'][:-1].copy()
    else: raise AssertionError(name)
    # Deliberately rebuild metadata so semantic guards, not stale custody, are exercised.
    m['arrays']=tools.recorder.metadata(a)
cases=[('length-range','program lengths outside'),('offsets','program offset inventory'),('indices','transition index inventory'),('seeds','seed consumption differs'),('steps','"terminal steps differ"'),('done','terminal done schedule'),('terminal-values','terminal values differ'),('winner','terminal winner differs'),('econ-continuity','economic transition continuity'),('bank-continuity','bank transition continuity'),('nonfinite','nonfinite'),('monotonic','economic counters decrease'),('coverage-counter','coverage counter differs'),('coverage-positive','coverage missing'),('dtype','array dtype/layout'),('shape','array shape differs')]
results=[]
try:
    for index,(name,needle) in enumerate(cases):
        manifest=copy.deepcopy(good); changed={k:v.copy() for k,v in arrays.items()}; update(manifest,changed,name)
        try: tools.recorder.validate_arrays(manifest,changed)
        except ValueError as error: baseline=str(error)
        else: raise AssertionError(('baseline accepted',name))
        code=mutant(needle);SOURCE.write_bytes(code)
        changed_module=load(SOURCE,f'probe_{index}')
        changed_module.validate_arrays(manifest,changed)
        results.append(dict(name=name,baseline_rejection=baseline,mutant='accepted invalid input; independent probe kills omitted guard',mutant_sha256=hashlib.sha256(code).hexdigest()))
        SOURCE.write_bytes(BASE)
        print(name,baseline,'=> omission detected',flush=True)
finally:
    SOURCE.write_bytes(BASE)
    assert SOURCE.read_bytes()==BASE
(OUT/'semantic-guard-probes.json').write_text(json.dumps({'baseline_sha256':hashlib.sha256(BASE).hexdigest(),'restored_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'cases':results},indent=2)+'\n')
