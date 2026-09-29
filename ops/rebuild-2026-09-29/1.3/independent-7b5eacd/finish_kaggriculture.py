import json
from pathlib import Path
import subprocess
import sys
out=Path('ops/rebuild-2026-09-29/1.3/independent-7b5eacd')

def run(name,command):
    with (out/f'{name}-runner.log').open('w') as log:
        code=subprocess.run([sys.executable,'ops/rebuild-2026-09-29/1.3/bounded.py','--name',f'independent-7b5eacd/{name}','--seconds','120','--',*command],stdout=log,stderr=subprocess.STDOUT).returncode
    info=json.loads((out/f'{name}.json').read_text())
    data=(out/f'{name}.log').read_text()
    print(json.dumps({'name':name,'exit':code,'stop':info['stop_reason'],'tail':data[-1800:]}),flush=True)
    return code,data

code,data=run('heads-collection',['uv','run','--no-sync','pytest','tests/kaggriculture/test_model_heads.py','--collect-only','-q'])
assert code==0
nodes=[s for s in data.splitlines() if s.startswith('tests/kaggriculture/test_model_heads.py::')]
assert nodes
results=[]
for group in range(0,len(nodes),12):
    group_nodes=nodes[group:group+12]
    name=f'heads-{group:03d}'
    code,_=run(name,['uv','run','--no-sync','pytest',*group_nodes,'-q','--tb=short'])
    results.append({'name':name,'nodes':group_nodes,'exit':code})
(out/'head-node-groups.json').write_text(json.dumps(results,indent=2)+'\n')
code,data=run('observe-collection',['uv','run','--no-sync','pytest','tests/kaggriculture/test_observe.py','--collect-only','-q'])
assert code==0
nodes=[s for s in data.splitlines() if s.startswith('tests/kaggriculture/test_observe.py::')]
pinned=[s for s in nodes if 'test_native_writer_uses_real_schema_and_keeps_all_pointers[True-' in s]
assert len(pinned)==2,pinned
code,_=run('observe-all-unpinned',['uv','run','--no-sync','pytest','tests/kaggriculture/test_observe.py','-q','--tb=short',*[v for node in pinned for v in ['--deselect',node]]])
(out/'observed-pinned-nodeids.json').write_text(json.dumps(pinned,indent=2)+'\n')
