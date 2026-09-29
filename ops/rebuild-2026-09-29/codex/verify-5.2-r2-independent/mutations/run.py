from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, os, shutil, subprocess, tempfile, time

EVIDENCE=Path(__file__).resolve().parent
REPO=Path.cwd()
PYTHON=str(REPO/'.venv/bin/python')
CASES=json.loads((EVIDENCE/'cases.json').read_text())
for c in CASES:
    if c['name']=='winner_ce_sign':
        c['old']='value_ce = -(targets * evaluation.winner_log_probabilities.float()).sum(dim=-1)'
        c['new']='value_ce = (targets * evaluation.winner_log_probabilities.float()).sum(dim=-1)'
for c in CASES:
    assert (REPO/c['file']).read_text().count(c['old']) == 1, c['name']
for i,c in enumerate(CASES,1): c['index']=i
(EVIDENCE/'cases.json').write_text(json.dumps(CASES,indent=2)+'\n')
paths={c['file'] for c in CASES}
original={p:(REPO/p).read_bytes() for p in paths}
sha=lambda raw:hashlib.sha256(raw).hexdigest()
env=os.environ.copy()
env['PYTHONDONTWRITEBYTECODE']='1'

def worker(worker_id):
    scratch=Path(tempfile.mkdtemp(prefix=f'kg-bc-r2-w{worker_id}-',dir='/private/tmp'))
    for part in ['python','scripts','configs','tests']:
        shutil.copytree(REPO/part,scratch/part,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    shutil.copy2(REPO/'pyproject.toml',scratch/'pyproject.toml')
    worker_env=env|{'PYTHONPATH':f'{scratch}/python:{scratch}'}
    custody={'worker':worker_id,'scratch':str(scratch),'python':PYTHON,'source':{p:sha(b) for p,b in original.items()}}
    (EVIDENCE/f'worker-{worker_id}.json').write_text(json.dumps(custody,indent=2)+'\n')
    probe=subprocess.run([PYTHON,'-B','-c','import owl.train.bc as b, owl.kaggriculture.bc_data as d, tests.kaggriculture.test_bc as t; print(b.__file__); print(d.__file__); print(t.__file__); print(t.train_bc_script.__file__)'],cwd=scratch,env=worker_env,capture_output=True,text=True)
    (EVIDENCE/f'worker-{worker_id}-source.log').write_text(probe.stdout+probe.stderr)
    assert probe.returncode==0 and all(str(scratch) in line for line in probe.stdout.splitlines()),probe.stdout+probe.stderr
    baseline=subprocess.run([PYTHON,'-B','-m','pytest','tests/kaggriculture/test_bc.py','-q'],cwd=scratch,env=worker_env,capture_output=True,text=True,timeout=90)
    (EVIDENCE/f'worker-{worker_id}-baseline.log').write_text(baseline.stdout+baseline.stderr)
    assert baseline.returncode==0,baseline.stdout+baseline.stderr
    results=[]
    try:
        for c in CASES[worker_id::3]:
            p=scratch/c['file']; raw=original[c['file']]
            assert p.read_bytes()==raw
            p.write_bytes(raw.decode().replace(c['old'],c['new']).encode())
            started=time.monotonic()
            cmd=[PYTHON,'-B','-m','pytest','tests/kaggriculture/test_bc.py','-q']
            try:
                r=subprocess.run(cmd,cwd=scratch,env=worker_env,capture_output=True,text=True,timeout=90)
                output=r.stdout+r.stderr
                status='SURVIVED' if r.returncode==0 else 'KILLED' if r.returncode==1 else 'SETUP_ERROR'
                result={'index':c['index'],'name':c['name'],'status':status,'exit_code':r.returncode,'seconds':round(time.monotonic()-started,3),'summary':'\n'.join(output.splitlines()[-5:])}
                (EVIDENCE/f'{c["index"]:03d}-{c["name"]}.log').write_text(output)
            except subprocess.TimeoutExpired:
                result={'index':c['index'],'name':c['name'],'status':'TIMEOUT'}
            finally:
                p.write_bytes(raw)
                assert p.read_bytes()==raw
            result['restored_sha256']=sha(p.read_bytes())
            results.append(result)
            (EVIDENCE/f'worker-{worker_id}-results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:result[k] for k in ('index','name','status')}),flush=True)
    finally:
        for p,b in original.items():
            (scratch/p).write_bytes(b)
            assert (scratch/p).read_bytes()==b
            assert (REPO/p).read_bytes()==b
        custody['all_byte_exact']=True
        (EVIDENCE/f'worker-{worker_id}-restoration.json').write_text(json.dumps(custody,indent=2)+'\n')
    post=subprocess.run([PYTHON,'-B','-m','pytest','tests/kaggriculture/test_bc.py','-q'],cwd=scratch,env=worker_env,capture_output=True,text=True,timeout=90)
    (EVIDENCE/f'worker-{worker_id}-restored-baseline.log').write_text(post.stdout+post.stderr)
    assert post.returncode==0
    return results

with ThreadPoolExecutor(max_workers=3) as pool:
    results=[item for group in pool.map(worker,range(3)) for item in group]
results.sort(key=lambda r:r['index'])
(EVIDENCE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
counts={s:sum(r['status']==s for r in results) for s in ('KILLED','SURVIVED','SETUP_ERROR','TIMEOUT')}
(EVIDENCE/'summary.json').write_text(json.dumps({'total':len(results),'counts':counts},indent=2)+'\n')
print(json.dumps(counts),flush=True)
