from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/codex/verify-merge-gpu-receipts-r3/root'
SRC = ROOT / 'ops/rebuild-2026-09-29/gpu-checks-2026-09-29'
SCRATCH = OUT / 'scratch'
shutil.copytree(SRC / 'scripts', SCRATCH / 'scripts', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(SRC / 'pod' / 'attempt2', SCRATCH / 'pod' / 'attempt2', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
original = {p.relative_to(SCRATCH): p.read_bytes() for p in SCRATCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
sys.path.insert(0, str(SCRATCH / 'scripts'))

def load(name, path):
    # Compile source directly so mutation tests never reuse bytecode.
    import types
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), mod.__dict__)
    return mod

g = load('guards_r3', SCRATCH / 'scripts/test_driver_guards.py')
drv = load('driver_r3', SCRATCH / 'scripts/driver.py')
c2 = load('c2_r3', SCRATCH / 'scripts/c2_trunk_bwd.py')
results = {'head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'baseline': [g.check_launcher_idle(), g.check_c2_nonfinite()], 'tensor_faults': [], 'source_regressions': []}
assert all(r['pass'] for r in results['baseline'])
retained = [json.loads(x) for x in (SCRATCH/'pod/attempt2/c2_aten_bwd.jsonl').read_text().splitlines()]

def reject(field, metrics):
    ev = copy.deepcopy(retained)
    next(x for x in ev if x.get('event') == 'result' and x['label'] == 'target')[field] = metrics
    return drv.judge_c2(ev)

# All three nonfinite types, both operands, both chunks; parameter mutation
# also exercises the special key-bias scale exception.
rows = c2.ROW_CHUNK + 1
mask = torch.ones(rows, 2, dtype=torch.bool)
for operand in ['compiled','reference']:
    for label, value in [('nan',float('nan')),('inf',float('inf')),('neginf',-float('inf'))]:
        for row in [0, c2.ROW_CHUNK]:
            for comp_name, func, field in [('out',c2.cmp_out,'out_compiled_vs_eager'),('dx',c2.cmp_dx,'dx_compiled_vs_eager')]:
                a = torch.ones(rows,2,4); b = a.clone()
                (a if operand == 'compiled' else b)[row,0,0] = value
                # Keep a genuine finite 10% error in the same first chunk.
                a[1,0,1] = 1.1
                met = func(a,b,mask); errors = reject(field,met)
                assert met['nonfinite'] == 1 and errors, (comp_name,operand,label,row,met)
                results['tensor_faults'].append({'comparator':comp_name,'operand':operand,'value':label,'row':row,'nonfinite':met['nonfinite'],'rejected':bool(errors)})
            for target in ['blocks.0.mlp.w','blocks.0.attn.k.bias']:
                a = {'blocks.0.mlp.w':torch.ones(rows),'blocks.0.attn.q.bias':torch.ones(rows),'blocks.0.attn.k.bias':torch.zeros(rows)}
                b = {k:v.clone() for k,v in a.items()}
                (a if operand == 'compiled' else b)[target][row] = value
                met = c2.cmp_params(a,b); errors = reject('params_compiled_vs_eager',met)
                assert met[target]['nonfinite'] == 1 and errors
                results['tensor_faults'].append({'comparator':'params','parameter':target,'operand':operand,'value':label,'row':row,'nonfinite':met[target]['nonfinite'],'rejected':bool(errors)})

# Restore the exact old implementations on scratch; the new regression must fail.
for name, check in [('launch.sh',g.check_launcher_idle),('c2_trunk_bwd.py',g.check_c2_nonfinite)]:
    path = SCRATCH / 'scripts' / name
    before = path.read_bytes()
    prior = subprocess.check_output(['git','show',f'3f26e49:{(SRC / "scripts" / name).relative_to(ROOT)}'])
    try:
        path.write_bytes(prior)
        # g.load loads bytecode; delete caches to force current scratch source.
        shutil.rmtree(SCRATCH/'scripts/__pycache__',ignore_errors=True)
        failure = check()
        assert not failure['pass'], name
        results['source_regressions'].append({'file':name,'mutation':'restore pre-r2 implementation','killed':True,'failures':failure['fails']})
    finally:
        path.write_bytes(before)
        shutil.rmtree(SCRATCH/'scripts/__pycache__',ignore_errors=True)
    assert check()['pass']

# Independently remove each query status check: both must be necessary.
path = SCRATCH / 'scripts/launch.sh'
for query in ['--query-compute-apps=', '--query-gpu=']:
    before = path.read_bytes()
    try:
        lines = before.decode().splitlines(True)
        index = next(i for i,l in enumerate(lines) if query in l and '|| return 1' in l)
        lines[index] = lines[index].replace(' || return 1','')
        path.write_text(''.join(lines))
        result = g.check_launcher_idle()
        assert not result['pass']
        results['source_regressions'].append({'file':'launch.sh','mutation':'remove status check '+query,'killed':True,'failures':result['fails']})
    finally:
        path.write_bytes(before)

# Remove each comparator's newly added reference nonfinite count separately.
path = SCRATCH / 'scripts/c2_trunk_bwd.py'
for func_name in ['cmp_out','cmp_dx','cmp_params']:
    before = path.read_bytes()
    try:
        text = before.decode(); start = text.index('def '+func_name+'('); end = text.find('\ndef ', start+1)
        if end == -1: end = len(text)
        chunk = text[start:end]
        addition = {'cmp_out':' + int((~torch.isfinite(r[m])).sum())', 'cmp_dx':' + int((~torch.isfinite(b[m])).sum())', 'cmp_params':' + int((~torch.isfinite(r)).sum())'}[func_name]
        assert chunk.count(addition) == 1
        path.write_text(text[:start]+chunk.replace(addition,'')+text[end:])
        shutil.rmtree(SCRATCH/'scripts/__pycache__',ignore_errors=True)
        result = g.check_c2_nonfinite()
        assert not result['pass']
        results['source_regressions'].append({'file':path.name,'mutation':'remove reference count in '+func_name,'killed':True,'failures':result['fails']})
    finally:
        path.write_bytes(before)
        shutil.rmtree(SCRATCH/'scripts/__pycache__',ignore_errors=True)

assert g.check_launcher_idle()['pass'] and g.check_c2_nonfinite()['pass']
restoration = {str(p): hashlib.sha256(content).hexdigest() for p,content in original.items() if (SCRATCH/p).read_bytes() == content}
assert len(restoration) == len(original)
results['restoration'] = {'files':len(original),'all_equal':True,'sha256':restoration}
(OUT / 'r2-probes.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'baseline_groups':len(results['baseline']),'tensor_faults_rejected':len(results['tensor_faults']),'source_regressions_killed':len(results['source_regressions']),'scratch_files_restored':len(original)},indent=2))
