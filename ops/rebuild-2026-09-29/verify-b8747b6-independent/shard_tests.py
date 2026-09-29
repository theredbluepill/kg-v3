from pathlib import Path
import json,os,subprocess,sys
out=Path(__file__).resolve().parent
root=out.parents[2]
env=os.environ | {'CARGO_BUILD_JOBS':'2','CARGO_NET_OFFLINE':'true','UV_OFFLINE':'true','RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RUST_TEST_THREADS':'1'}
files=sorted(p for folder in ('kaggriculture','owl','scripts','tools') for p in (root/'tests'/folder).rglob('test_*.py'))
results=[]
def run(name,args):
    p=subprocess.run([sys.executable,str(out/'bounded.py'),'--name',name,'--','uv','run','pytest',*args,'-m','not slow','-q','--junitxml',str(out/(name+'.xml'))],env=env,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    result=json.loads((out/(name+'.json')).read_text());results.append(result|{'name':name});(out/'shard-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name, 'exit',p.returncode, 'stop',result['stop_reason'],flush=True)
    return result
for index,p in enumerate(files):
    rel=str(p.relative_to(root));name=f'shard-{index:02}-{p.stem}'
    r=run(name,[rel])
    if r['stop_reason']:
        c=subprocess.run(['uv','run','pytest',rel,'-m','not slow','--collect-only','-q'],env=env,cwd=root,text=True,capture_output=True,timeout=120)
        if c.returncode: print(c.stdout,c.stderr);sys.exit(1)
        ids=[line for line in c.stdout.splitlines() if line.startswith(rel+'::')]
        (out/(name+'-nodes.json')).write_text(json.dumps(ids,indent=2)+'\n')
        for i in range(0,len(ids),8):
            sub=run(f'{name}-nodes-{i:03}',ids[i:i+8])
            if sub['exit_status']!=0:sys.exit(1)
    elif r['exit_status']!=0:sys.exit(1)
print('ALL SHARDS COMPLETE',len(files),flush=True)
