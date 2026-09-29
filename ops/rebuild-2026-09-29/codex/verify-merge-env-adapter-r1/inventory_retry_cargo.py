from pathlib import Path
import json, os, subprocess, time
repo = Path('/Users/poonszesen/kg-v3-m-env-adapter')
receipts = repo/'ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1'
mpath = receipts/'inventory-manifest.json'
manifest = json.loads(mpath.read_text())
scratch = Path(manifest['scratch'])
env = os.environ.copy()
env.update(manifest['env'], PYO3_PYTHON=manifest['python'])
manifest['retry_cargo_reason'] = 'Archive snapshots lack .venv/bin/python required by .cargo/config.toml; set PYO3_PYTHON explicitly to shared existing interpreter.'
for label in ['base', 'adapter', 'head']:
    log = receipts/f'inventory-{label}-cargo-root-list-retry.log'
    cmd = ['cargo','test','--locked','--offline','--','--list']
    dest = scratch/label
    start=time.monotonic()
    print('START',label,flush=True)
    with log.open('w') as f:
        result=subprocess.run(cmd,cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT)
    manifest['runs'].append({'label':label,'kind':'cargo-root-list-retry','command':cmd,'cwd':str(dest),'PYO3_PYTHON':env['PYO3_PYTHON'],'exit_code':result.returncode,'seconds':round(time.monotonic()-start,2),'log':str(log.relative_to(repo))})
    mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    print('END',label,result.returncode,flush=True)
