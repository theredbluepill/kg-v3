import json
from pathlib import Path
import subprocess
import sys
out=Path('ops/rebuild-2026-09-29/1.3/independent-8247ee2')
cases=[
 ('heads-noncompiled',['uv','run','--no-sync','pytest','tests/kaggriculture/test_model_heads.py','-q','-k','not fullgraph_captured_core','--tb=short']),
 ('heads-compiled',['uv','run','--no-sync','pytest','tests/kaggriculture/test_model_heads.py::test_fullgraph_captured_core_matches_eager_density_and_gradients','-q','--tb=short']),
 ('custody',['uv','run','--no-sync','pytest','tests/tools/test_observation_oracle_custody.py','tests/tools/test_check_engine_trim.py','-q','--tb=short']),
]
results=[]
for name,command in cases:
    with (out/f'{name}-runner.log').open('w') as log:
        code=subprocess.run([sys.executable,'ops/rebuild-2026-09-29/1.3/bounded.py','--name',f'independent-8247ee2/{name}','--seconds','120','--',*command],stdout=log,stderr=subprocess.STDOUT).returncode
    info=json.loads((out/f'{name}.json').read_text())
    summary=(out/f'{name}.log').read_text()[-2000:]
    results.append({'name':name,**info,'summary':summary})
    (out/'remaining.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'name':name,'exit':code,'stop':info['stop_reason'],'tail':summary}),flush=True)
subprocess.run([sys.executable,str(out/'run_shards.py'),'tests/owl'],check=True)
