import hashlib,json,os,platform,subprocess,time
from pathlib import Path
ROOT=Path.cwd(); OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r2'
ENV=os.environ|{'CARGO_BUILD_JOBS':'2','OMP_NUM_THREADS':'2','RUST_TEST_THREADS':'2','UV_OFFLINE':'1'}
rows=[]
def run(name,cmd):
 print('START',name,flush=True); start=time.monotonic()
 with (OUT/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'name':name,'command':cmd,'exit_code':p.returncode,'seconds':round(time.monotonic()-start,3)})
 (OUT/'checks.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('END',name,p.returncode,rows[-1]['seconds'],flush=True)
 return p.returncode
native=ROOT/'python/owl/rs.abi3.so'
meta={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'platform':platform.platform(),'env':{k:ENV[k] for k in ['CARGO_BUILD_JOBS','OMP_NUM_THREADS','RUST_TEST_THREADS','UV_OFFLINE']},'native_before_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}
assert run('native-rebuild',['uv','run','maturin','develop','--locked','--offline','--skip-install'])==0
meta['native_after_sha256']=hashlib.sha256(native.read_bytes()).hexdigest()
(OUT/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
run('engine-tests',['cargo','test','--manifest-path','engine_rs/Cargo.toml','--locked','--offline'])
run('root-tests',['cargo','test','--locked','--offline'])
run('engine-trim',['uv','run','python','scripts/check_engine_trim.py'])
run('pytest',['uv','run','pytest','tests','-m','not slow','-q'])
run('mypy',['uv','run','mypy','python/owl','scripts'])
run('docs-freshness',['uv','run','python','scripts/check_doc_freshness.py'])
