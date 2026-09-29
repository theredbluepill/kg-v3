import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();out=root/'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r2'
env=os.environ|{'CARGO_BUILD_JOBS':'2','OMP_NUM_THREADS':'2','RUST_TEST_THREADS':'2','UV_OFFLINE':'1'}
rows=[]
for name,cmd in [
 ('pytest-oracle-shard',['uv','run','pytest','tests/tools/test_observation_oracle_custody.py','-m','not slow','-q']),
 ('pytest-rest-shard',['uv','run','pytest','tests','--ignore=tests/tools/test_observation_oracle_custody.py','-m','not slow','-q'])]:
 print('START',name,flush=True); start=time.monotonic()
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'name':name,'command':cmd,'exit_code':p.returncode,'seconds':round(time.monotonic()-start,3)})
 (out/'shards.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('END',name,rows[-1],flush=True)
