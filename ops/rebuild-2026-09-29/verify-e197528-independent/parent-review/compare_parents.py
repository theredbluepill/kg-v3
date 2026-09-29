from pathlib import Path
import subprocess, ast, re, json, hashlib, sys
OUT=Path(__file__).resolve().parent

def git(*args):
    return subprocess.check_output(['git',*args])

def files(rev):
    return {line.split('\t')[1]:line.split()[2] for line in git('ls-tree','-r',rev).decode().splitlines() if line.split()[1]=='blob'}

def contents(rev,path):
    return git('show',f'{rev}:{path}')

def test_names(rev, paths):
    result={}
    for path in paths:
        if path.endswith('.py') and path.startswith('tests/'):
            tree=ast.parse(contents(rev,path))
            result[path]=[node.name for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_')]
        elif path.endswith('.rs') and path.startswith(('src/','engine_rs/')):
            result[path]=re.findall(r'#\[test\](?:\s*#\[[^\n]+\])*\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)',contents(rev,path).decode())
    return {p:n for p,n in result.items() if n}

HEAD=files('HEAD')
summary={}
head_tests=test_names('HEAD',HEAD)
(OUT/'tests-HEAD.json').write_text(json.dumps(head_tests,indent=2)+'\n')
for rev in ('b51b0c0','dc6b200'):
    parent=files(rev)
    tests=test_names(rev,parent)
    (OUT/f'tests-{rev}.json').write_text(json.dumps(tests,indent=2)+'\n')
    lost=[]
    moved=[]
    for path,names in tests.items():
        dst='src/kaggriculture/grammar_kernel_tests.rs' if path=='engine_rs/tests/grammar_kernel.rs' else path
        for name in names:
            if name not in head_tests.get(dst,[]): lost.append([path,name])
            elif dst!=path: moved.append([path,dst,name])
    changed_existing=[p for p,sha in parent.items() if p in HEAD and HEAD[p]!=sha]
    scope=lambda p: p.startswith(('python/','configs/','tests/kaggriculture/','tests/owl/','tests/scripts/')) or p in ('scripts/run_ppo.py','scripts/benchmark_checkpoints.py')
    summary[rev]={
      'python_test_functions':sum(len(n) for p,n in tests.items() if p.endswith('.py')),
      'rust_test_functions':sum(len(n) for p,n in tests.items() if p.endswith('.rs')),
      'missing_test_functions':lost,
      'relocated_test_functions':moved,
      'deleted_paths':[p for p in parent if p not in HEAD],
      'existing_python_config_trainer_paths_changed':[p for p in changed_existing if scope(p)],
      'existing_python_config_trainer_paths_identical':sum(scope(p) and HEAD.get(p)==sha for p,sha in parent.items()),
      'existing_task13_paths_changed':[p for p in changed_existing if p.startswith(('src/kaggriculture/','scripts/kaggriculture_observation_oracle/','tests/fixtures/kaggriculture/observation-v3/')) or p in ('Cargo.toml','Cargo.lock','src/rules_engine/generation.rs','src/lib.rs','scripts/check_engine_trim.py','tests/tools/test_observation_oracle_custody.py','tests/tools/test_check_engine_trim.py','tests/kaggriculture/test_observe.py','python/owl/rs.pyi','engine_rs/TRIM_MANIFEST.json')],
      'engine_changes':[p for p in changed_existing if p.startswith('engine_rs/')],
    }
summary['HEAD']={'python_test_functions':sum(len(n) for p,n in head_tests.items() if p.endswith('.py')),'rust_test_functions':sum(len(n) for p,n in head_tests.items() if p.endswith('.rs'))}
# Re-execute committed retirement generator over the pre-retirement parent manifest.
scratch=OUT/'manifest-rebuild'
script=scratch/'ops/rebuild-2026-09-29/1.3/retire_grammar_bridge.py'
manifest=scratch/'engine_rs/TRIM_MANIFEST.json'
script.parent.mkdir(parents=True,exist_ok=True)
manifest.parent.mkdir(parents=True,exist_ok=True)
script.write_bytes(contents('HEAD','ops/rebuild-2026-09-29/1.3/retire_grammar_bridge.py'))
manifest.write_bytes(contents('b51b0c0','engine_rs/TRIM_MANIFEST.json'))
subprocess.run([sys.executable,str(script)],check=True)
expected=contents('HEAD','engine_rs/TRIM_MANIFEST.json')
summary['manifest_regeneration']={'generator':'ops/rebuild-2026-09-29/1.3/retire_grammar_bridge.py','input':'b51b0c0:engine_rs/TRIM_MANIFEST.json','matches_HEAD':manifest.read_bytes()==expected,'output_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'HEAD_sha256':hashlib.sha256(expected).hexdigest(),'identical_to_Task13':expected==contents('dc6b200','engine_rs/TRIM_MANIFEST.json')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
