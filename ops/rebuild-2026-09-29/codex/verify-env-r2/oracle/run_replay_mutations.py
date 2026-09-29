from pathlib import Path
import hashlib,json,os,subprocess,time
ROOT=Path('/Users/poonszesen/kg-v3-env'); SCRATCH=Path('/private/tmp/kg-env-r2-oracle'); OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r2/oracle'
SOURCE=SCRATCH/'tests/kaggriculture/test_env_reference.py'; BASE=SOURCE.read_bytes(); text=BASE.decode(); results=[]
mutations={}
for field in ['rewards','dones','transition_banks_before','transition_banks_after','transition_econ_before','transition_econ_after']:
    change=f'            out["{field}"].flat[0] '+('^= True' if field=='dones' else '+= 1')+'\n'
    old='            metrics = env.step(tokens, lengths, **out)\n'
    mutations[field]=text.replace(old,old+change)
mutations['mathematical-reward']=text.replace('        dtype=np.float32,\n    )\n\n\ndef _same', '        dtype=np.float32,\n    ) + np.float32(0.01)\n\n\ndef _same')
try:
    for name,mutant in mutations.items():
        assert mutant!=text
        SOURCE.write_text(mutant)
        for cache in (SOURCE.parent/'__pycache__').glob('test_env_reference.*.pyc'): cache.unlink()
        command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-c','/dev/null',f'--rootdir={SCRATCH}',f'--confcutdir={SCRATCH}','-p','no:cacheprovider',str(SOURCE),'-q']
        started=time.monotonic();proc=subprocess.run(command,cwd=SCRATCH,env=os.environ|{'GIT_DIR':str(ROOT/'.git'),'GIT_WORK_TREE':str(ROOT),'PYTHONPATH':str(ROOT/'python')},capture_output=True,text=True,timeout=20)
        log=proc.stdout+proc.stderr;(OUT/f'replay-mutant-{name}.log').write_text(log)
        assert proc.returncode==1,(name,proc.returncode)
        assert ('first divergence game=0 seed=17000 step=0' in log if name!='mathematical-reward' else 'mathematical reward' in log),name
        results.append(dict(name=name,status='killed',exit=proc.returncode,wall_seconds=time.monotonic()-started,mutant_sha256=hashlib.sha256(mutant.encode()).hexdigest()))
        SOURCE.write_bytes(BASE);print(name,'killed',flush=True)
finally:
    SOURCE.write_bytes(BASE); assert SOURCE.read_bytes()==BASE
(OUT/'replay-mutations.json').write_text(json.dumps({'baseline_sha256':hashlib.sha256(BASE).hexdigest(),'restored_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'mutations':results},indent=2)+'\n')
