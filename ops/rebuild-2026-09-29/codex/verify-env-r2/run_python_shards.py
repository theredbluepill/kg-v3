from pathlib import Path
import os, subprocess, json, re
root=Path(__file__).resolve().parents[4]
out=Path(__file__).resolve().parent
bounded=out/'bounded.py'
env=os.environ | {'CARGO_BUILD_JOBS':'2','CARGO_NET_OFFLINE':'true','UV_OFFLINE':'true','UV_NO_SYNC':'true','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
files=subprocess.check_output(['rg','--files','tests/kaggriculture','tests/owl','tests/scripts','tests/tools','-g','test*.py'],cwd=root,text=True).splitlines()
results=[]
counter=0

def run(nodes):
    global counter
    counter+=1
    name=f'python-shard-{counter:03}'
    p=subprocess.run(['python3',str(bounded),'--name',name,'--','uv','run','pytest',*nodes,'-m','not slow','-q'],cwd=root,env=env,stdout=subprocess.DEVNULL)
    receipt=json.loads((out/(name+'.json')).read_text())
    log=(out/(name+'.log')).read_text()
    receipt['name']=name
    receipt['nodes']=nodes
    receipt['summary']=log.splitlines()[-1] if log.splitlines() else ''
    receipt['accepted']=p.returncode==0
    results.append(receipt)
    (out/'python-shards.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,receipt['exit_status'],receipt['stop_reason'],receipt['summary'],flush=True)
    if receipt['stop_reason']:
        if len(nodes)==1 and '::' not in nodes[0]:
            collected=subprocess.run(['uv','run','pytest',nodes[0],'-m','not slow','--collect-only','-q'],cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout
            (out/(name+'-collection.log')).write_text(collected)
            nodes=[line for line in collected.splitlines() if line.startswith('tests/') and '::' in line]
        if len(nodes)>1:
            midpoint=len(nodes)//2
            run(nodes[:midpoint]);run(nodes[midpoint:])
        else: print('UNRESOLVED single test resource limit',nodes,flush=True)
for file in sorted(files):run([file])
passed=skipped=deselected=0
for result in results:
    if result['accepted']:
        for label in ('passed','skipped','deselected'):
            found=re.search(r'(\d+) '+label,result['summary'])
            if found:
                if label=='passed':passed+=int(found[1])
                elif label=='skipped':skipped+=int(found[1])
                else:deselected+=int(found[1])
print(json.dumps({'passed':passed,'skipped':skipped,'deselected':deselected,'attempts':len(results)}),flush=True)
