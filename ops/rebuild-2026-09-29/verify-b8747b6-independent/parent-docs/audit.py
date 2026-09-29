from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT=Path.cwd()
OUT=ROOT/'ops/rebuild-2026-09-29/verify-b8747b6-independent/parent-docs'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)
def tree(rev):
    result={}
    for record in git('ls-tree','-r','-z',rev).split(b'\0'):
        if record:
            meta,path=record.split(b'\t',1)
            result[path.decode()]=meta.split()[2].decode()
    return result
def blob(rev,path):
    return git('show',f'{rev}:{path}')
revs=['b51b0c0','dc6b200','e197528','HEAD']
trees={rev:tree(rev) for rev in revs}
def inventory(rev):
    tests=[]
    for path in trees[rev]:
        if not (path.startswith(('src/','engine_rs/src/','engine_rs/tests/','tests/')) and path.endswith(('.rs','.py'))):continue
        data=blob(rev,path).decode()
        if path.endswith('.py'):
            if not Path(path).name.startswith('test_'):continue
            parsed=ast.parse(data)
            for node in ast.walk(parsed):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                    tests.append((path,node.name))
        else:
            tests += [(path,n) for n in re.findall(r'#\[test\](?:\s*#\[[^\n]*\])*\s*(?:pub\s+)?fn\s+(\w+)',data)]
    return sorted(tests)
inv={rev:inventory(rev) for rev in revs}
def normalized(entry):
    path,name=entry
    if path=='engine_rs/tests/grammar_kernel.rs':path='src/kaggriculture/grammar_kernel_tests.rs'
    if name=='test_task_authored_inventory_accepts_two_tests_and_generated_manifest':name='test_task_authored_inventory_accepts_replay_test_and_generated_manifest'
    return path,name
head=set(inv['HEAD'])
parents={}
for rev in revs[:-1]:
    prior=set(map(normalized,inv[rev]))
    parents[rev]={'test_declarations':len(inv[rev]),'missing_after_relocation_and_authored_rename':sorted(prior-head),'new_in_head':sorted(head-prior)}
protected=[p for p in trees['b51b0c0'] if p.startswith(('python/owl/model/','python/owl/train/','python/owl/kaggriculture/','configs/','tests/kaggriculture/','tests/owl/')) or p in ('scripts/run_ppo.py','scripts/benchmark_checkpoints.py','tests/scripts/test_run_ppo.py')]
identities={p:{rev:trees[rev].get(p) for rev in ('b51b0c0','HEAD')} for p in protected}
changed=[p for p,row in identities.items() if row['b51b0c0']!=row['HEAD']]
relocated_old=blob('b51b0c0','engine_rs/tests/grammar_kernel.rs').decode()
relocated_new=blob('HEAD','src/kaggriculture/grammar_kernel_tests.rs').decode()
old_tests=[n for p,n in inv['b51b0c0'] if p=='engine_rs/tests/grammar_kernel.rs']
new_tests=[n for p,n in inv['HEAD'] if p=='src/kaggriculture/grammar_kernel_tests.rs']
log_head=blob('HEAD','cookbook/log.md').decode()
log_checks={rev:{'headings':len(re.findall(r'^## .+',blob(rev,'cookbook/log.md').decode(),re.M)),'missing_headings':[h for h in re.findall(r'^## .+',blob(rev,'cookbook/log.md').decode(),re.M) if h not in log_head]} for rev in ('b51b0c0','dc6b200')}
manifest=blob('HEAD','engine_rs/TRIM_MANIFEST.json')
with tempfile.TemporaryDirectory(prefix='parent-docs-trim-') as tmp:
    scratch=Path(tmp)/'TRIM_MANIFEST.json'
    original=blob('b51b0c0','engine_rs/TRIM_MANIFEST.json')
    scratch.write_bytes(original)
    spec=importlib.util.spec_from_file_location('retire_bridge',ROOT/'ops/rebuild-2026-09-29/1.3/retire_grammar_bridge.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MANIFEST=scratch
    module.main()
    reproduced=scratch.read_bytes()
    module.main()
    idempotent=scratch.read_bytes()==reproduced
    scratch.write_bytes(original)
    generator={'generator':'ops/rebuild-2026-09-29/1.3/retire_grammar_bridge.py','input':'b51b0c0:engine_rs/TRIM_MANIFEST.json','matches_head':reproduced==manifest,'sha256':hashlib.sha256(reproduced).hexdigest(),'idempotent':idempotent,'scratch_restored_byte_exact':scratch.read_bytes()==original,'matches_task_parent':manifest==blob('dc6b200','engine_rs/TRIM_MANIFEST.json')}
result={'head':git('rev-parse','HEAD').decode().strip(),'head_test_declarations':len(head),'parent_tests':parents,'relocated_kernel_tests':{'old':old_tests,'new':new_tests,'same_names':old_tests==new_tests},'protected_code_test_config_files':len(protected),'protected_changed':changed,'log_headings_preserved':log_checks,'manifest_generator':generator}
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'test-inventories.json').write_text(json.dumps(inv,indent=2)+'\n')
(OUT/'protected-blobs.json').write_text(json.dumps(identities,indent=2)+'\n')
print(json.dumps(result,indent=2))
