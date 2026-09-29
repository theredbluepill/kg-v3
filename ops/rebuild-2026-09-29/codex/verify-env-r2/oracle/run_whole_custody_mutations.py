from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
ROOT=Path('/Users/poonszesen/kg-v3-env')
SCRATCH=Path('/private/tmp/kg-env-r2-oracle')
OUT=ROOT/'ops/rebuild-2026-09-29/codex/verify-env-r2/oracle'
SOURCE=SCRATCH/'scripts/record_kaggriculture_env_reference.py'
BASE=SOURCE.read_bytes(); text=BASE.decode()
def digest(b): return hashlib.sha256(b).hexdigest()
def guard_mutant(needle):
    lines=text.splitlines(keepends=True); offsets=[0]
    for line in lines: offsets.append(offsets[-1]+len(line))
    found=[]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2:
            reason=ast.get_source_segment(text,node.args[1])
            if needle in reason:
                arg=node.args[0]; found.append((offsets[arg.lineno-1]+arg.col_offset, offsets[arg.end_lineno-1]+arg.end_col_offset))
    assert len(found)==1,(needle,found)
    a,b=found[0]; return (text[:a]+'True'+text[b:]).encode()
mutants=[
('whole-compressed-cap','compressed size budget exceeded','not frozen_fixture'),
('whole-compressed-hash','"fixture hash differs"','not frozen_fixture'),
('whole-expanded-hash','expanded fixture hash differs','not frozen_fixture'),
]
# export guard reason includes same prefix twice; select byte-check statement explicitly.
results=[]
for name,needle,selector in mutants:
    if name=='export-drift':
        mutant=text.replace('require(sha(path.read_bytes()) == expected, f"export source drift: {relative}")','require(True, f"export source drift: {relative}")').encode()
        assert mutant!=BASE
    else: mutant=guard_mutant(needle)
    SOURCE.write_bytes(mutant)
    for cache in (SOURCE.parent/'__pycache__').glob('record_kaggriculture_env_reference.*.pyc'): cache.unlink()
    command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-c','/dev/null',f'--rootdir={SCRATCH}',f'--confcutdir={SCRATCH}','-p','no:cacheprovider',str(SCRATCH/'tests/tools/test_record_kaggriculture_env_reference.py'),'-q','-k',selector]
    started=time.monotonic()
    try:
        proc=subprocess.run(command,cwd=SCRATCH,env=os.environ|{'GIT_DIR':str(ROOT/'.git'),'GIT_WORK_TREE':str(ROOT),'PYTHONPATH':str(ROOT/'python')},capture_output=True,text=True,timeout=12)
        output=proc.stdout+proc.stderr
        result={'name':name,'exit':proc.returncode,'status':'killed' if proc.returncode==1 else 'survived' if proc.returncode==0 else 'harness-error','command':command,'wall_seconds':time.monotonic()-started,'mutant_sha256':digest(mutant)}
    except subprocess.TimeoutExpired as exc:
        output=str(exc);result={'name':name,'status':'timeout','wall_seconds':time.monotonic()-started}
    finally:
        SOURCE.write_bytes(BASE)
        assert SOURCE.read_bytes()==BASE
    (OUT/f'mutant-{name}.log').write_text(output)
    results.append(result);print(name,result['status'],flush=True)
(OUT/'whole-custody-mutations.json').write_text(json.dumps({'baseline_sha256':digest(BASE),'restored_sha256':digest(SOURCE.read_bytes()),'mutations':results},indent=2)+'\n')
