import importlib.util,json,os,subprocess,tempfile,time
from pathlib import Path
from unittest.mock import patch
P=Path('/private/tmp/gpu-r3-mut-z43e6hsa/bundle/scripts/driver.py')
spec=importlib.util.spec_from_file_location('driver',P);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
results=[]
for fault in ('baseline','stop','deadline','missing_backend','backend_start','backend_end','exit','timeout','judge_throws','judge_error','control_failure'):
    wd=Path(tempfile.mkdtemp(prefix='gpu-driver-guard-',dir='/private/tmp/gpu-r3-mut-z43e6hsa'));D.RUN=wd;D.ROOT=wd
    d=D.Driver();role='control' if fault=='control_failure' else 'correct';calls=[];timeouts=[]
    if fault=='stop':d.stop.set()
    if fault=='deadline':d.deadline=time.monotonic()
    class Fake:
        pid=98765432
        def wait(self,timeout=None):
            if timeout is not None:timeouts.append(timeout)
            if fault=='timeout' and timeout is not None:raise subprocess.TimeoutExpired('stub',timeout)
            return 1 if fault in ('exit','control_failure') else 0
    def spawn(name,cmd,env,fh):
        calls.append(name)
        assert 'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS' not in env
        if fault!='missing_backend':
            (wd/f'{name}.backend.json').write_text(json.dumps({'value_at_start':'WRONG' if fault=='backend_start' else 'ATEN','value_at_end':'WRONG' if fault=='backend_end' else 'ATEN'}))
        return Fake()
    d._spawn=spawn
    def judge(ev):
        if fault=='judge_throws':raise ValueError('injected')
        return ['injected'] if fault=='judge_error' else []
    D.JUDGES={'c3':judge}
    with patch.dict(os.environ,{'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS':'contaminated'}),patch.object(os,'killpg',return_value=None):
        d.run_stage(('c3_guard',role,'ATEN','cache','noop.py',[]),0)
    d.log_fh.close()
    logs=[json.loads(l) for l in (wd/'driver.jsonl').read_text().splitlines()]
    if fault=='baseline':ok=d.failure is None and logs[-1]['status']=='pass'
    elif fault=='stop':ok=not calls and logs[-1]['status']=='skipped_after_stop'
    elif fault=='deadline':ok=not calls and d.failure==4
    elif fault=='control_failure':ok=d.failure is None and timeouts[0]<=600 and logs[-1]['status']=='control_reproduced'
    else:ok=d.failure==(4 if fault=='timeout' else 3) and d.stop.is_set()
    results.append({'probe':fault,'passed':ok,'failure':d.failure,'status':logs[-1]['status']})
print(json.dumps(results,indent=2));print('SUMMARY',len(results),'passed',sum(r['passed'] for r in results))
Path('/private/tmp/gpu-r3-mut-z43e6hsa/driver-guard-matrix.json').write_text(json.dumps(results,indent=2))
