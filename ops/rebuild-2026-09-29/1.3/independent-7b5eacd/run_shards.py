import json
from pathlib import Path
import subprocess
import sys

root=Path.cwd()
out=root/'ops/rebuild-2026-09-29/1.3/independent-7b5eacd'
paths=sorted((root/sys.argv[1]).rglob('test_*.py'))
results=[]
for path in paths:
    name=path.relative_to(root).with_suffix('').as_posix().replace('/','-')
    command=[sys.executable,'ops/rebuild-2026-09-29/1.3/bounded.py','--name',f'independent-7b5eacd/{name}','--seconds','120','--','uv','run','--no-sync','pytest',str(path.relative_to(root)),'-q','--tb=short']
    with (out/f'{name}-runner.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    info=json.loads((out/f'{name}.json').read_text())
    tail=(out/f'{name}.log').read_text()[-1600:]
    results.append({'path':str(path.relative_to(root)),**info,'summary':tail})
    (out/('shards-'+sys.argv[1].replace('/','-')+'.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'path':str(path.relative_to(root)),'exit':result.returncode,'stop':info['stop_reason'],'tail':tail}),flush=True)
