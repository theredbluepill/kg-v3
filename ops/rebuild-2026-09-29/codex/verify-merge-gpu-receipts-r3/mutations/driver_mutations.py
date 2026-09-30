import importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
P=Path('/private/tmp/gpu-r3-mut-z43e6hsa/bundle/scripts/driver.py')
spec=importlib.util.spec_from_file_location('driver',P);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
results=[]
def run(label,corruption):
    wd=Path(tempfile.mkdtemp(prefix='gpu-driver-mutation-',dir='/private/tmp/gpu-r3-mut-z43e6hsa'));D.RUN=wd;D.ROOT=wd
    d=D.Driver();d.install_cleanup=lambda:None
    calls=[]
    class Fake:
        pid=-1
        def wait(self,timeout): return 0
    def spawn(name,cmd,env,fh):
        calls.append(name)
        if corruption=='popen' and name=='c3_mutated':raise OSError('injected spawn failure')
        if name=='c3_mutated' and corruption=='malformed':(wd/f'{name}.backend.json').write_text('{INVALID')
        else:(wd/f'{name}.backend.json').write_text(json.dumps({'value_at_start':'ATEN','value_at_end':'ATEN'}))
        return Fake()
    d._spawn=spawn
    D.JUDGES={key:lambda ev:[] for key in ('c1','c2','c3','c4')}
    D.GPU0_PHASE1=[('c3_mutated','correct','ATEN','cache','noop.py',[])]
    D.GPU1_PHASE1=[]
    D.GPU0_PHASE2=[('c4_after','correct','ATEN','cache','noop.py',[])]
    rc=d.main();d.log_fh.close()
    result={'probe':label,'rc':rc,'launched':calls,'guard_rejected':rc!=0 and 'c4_after' not in calls,'evidence':str(wd/'driver.jsonl')};results.append(result);print(json.dumps(result),flush=True)
run('valid_baseline','none')
run('malformed_backend_record','malformed')
run('spawn_exception','popen')
Path('/private/tmp/gpu-r3-mut-z43e6hsa/driver-mutations.json').write_text(json.dumps(results,indent=2))
