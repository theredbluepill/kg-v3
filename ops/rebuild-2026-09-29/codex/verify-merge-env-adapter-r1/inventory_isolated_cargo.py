from pathlib import Path
import json, os, subprocess, time
repo=Path('/Users/poonszesen/kg-v3-m-env-adapter')
receipts=repo/'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1'
mpath=receipts/'inventory-manifest.json'
manifest=json.loads(mpath.read_text())
scratch=Path(manifest['scratch'])
manifest['isolated_cargo_reason']='Shared cargo target reused the BASE test binary across git archives with historical file mtimes; earlier cargo lists are NOT credited. Rebuild into independent initially absent target directories per revision.'
for label in ['base','adapter','head']:
    env=os.environ.copy()
    env.update(CARGO_BUILD_JOBS='2',OMP_NUM_THREADS='2',CARGO_TARGET_DIR=str(scratch/f'cargo-isolated-{label}'),PYO3_PYTHON=manifest['python'])
    for kind,cmd in [('cargo-engine-list-isolated',['cargo','test','--manifest-path','engine_rs/Cargo.toml','--locked','--offline','--','--list']),('cargo-root-list-isolated',['cargo','test','--locked','--offline','--','--list'])]:
        log=receipts/f'inventory-{label}-{kind}.log'
        dest=scratch/label
        start=time.monotonic()
        print('START',label,kind,flush=True)
        with log.open('w') as f:
            result=subprocess.run(cmd,cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT)
        manifest['runs'].append({'label':label,'kind':kind,'command':cmd,'cwd':str(dest),'env':{k:env[k] for k in ['CARGO_BUILD_JOBS','OMP_NUM_THREADS','CARGO_TARGET_DIR','PYO3_PYTHON']},'exit_code':result.returncode,'seconds':round(time.monotonic()-start,2),'log':str(log.relative_to(repo))})
        mpath.write_text(json.dumps(manifest,indent=2)+'\n')
        print('END',label,kind,result.returncode,flush=True)
