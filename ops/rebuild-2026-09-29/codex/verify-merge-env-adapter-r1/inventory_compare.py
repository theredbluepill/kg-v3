from pathlib import Path
import ast, json, re, subprocess
receipts = Path('ops/rebuild-2026-09-29/codex/verify-merge-env-adapter-r1')
manifest=json.loads((receipts/'inventory-manifest.json').read_text())
scratch=Path(manifest['scratch'])
labels=['base','adapter','head']
results={}
for kind,suffix in [('pytest','pytest-collect'),('cargo_engine','cargo-engine-list-isolated'),('cargo_root','cargo-root-list-isolated')]:
    sets={}
    for label in labels:
        lines=(receipts/f'inventory-{label}-{suffix}.log').read_text().splitlines()
        if kind=='pytest':
            names=[x for x in lines if x.startswith('tests/') and '::' in x]
        else:
            names=[]
            binary=''
            for line in lines:
                m=re.match(r'\s+Running (.+?) \(',line)
                if m: binary=m[1]
                if line.endswith(': test'): names.append(binary+'::'+line.removesuffix(': test'))
        (receipts/f'inventory-{label}-{kind}-names.txt').write_text('\n'.join(sorted(names))+'\n')
        sets[label]=set(names)
        assert len(sets[label])==len(names), (kind,label,'duplicate names')
    results[kind]={'counts':{k:len(v) for k,v in sets.items()},'missing_from_head':{k:sorted(sets[k]-sets['head']) for k in labels[:2]},'added_at_head':{k:sorted(sets['head']-sets[k]) for k in labels[:2]},'union_count':len(sets['base']|sets['adapter']),'union_missing_count':len((sets['base']|sets['adapter'])-sets['head'])}

# Mechanical Python declarations ignores parametrization and follows nested test classes.
def declarations(root):
    out=set()
    for p in (root/'tests').rglob('test_*.py'):
        tree=ast.parse(p.read_text())
        def visit(nodes, scope=()):
            for node in nodes:
                if isinstance(node,ast.ClassDef): visit(node.body,scope+(node.name,))
                elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                    out.add(str(p.relative_to(root))+'::'+'::'.join(scope+(node.name,)))
        visit(tree.body)
    return out
sets={label:declarations(scratch/label) for label in labels}
results['python_declarations']={'counts':{k:len(v) for k,v in sets.items()},'missing_from_head':{k:sorted(sets[k]-sets['head']) for k in labels[:2]}}
results['pytest_renames']=[
 {'old':'tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native','new':'tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture','origin':'adapter parent 8699ca9','assessment':'Orbit constructor argument assertions retained; obsolete NotImplementedError for native Kaggriculture replaced by actual environment creation/type/pinning assertion, plus argument, replay and independent environment tests.'},
 {'old':'tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration','new':'tests/tools/test_check_engine_trim.py::test_no_authored_grammar_path_include_after_root_engine_edge','origin':'adapter parent 8699ca9','assessment':'Retains no engine grammar bridge file; replaces root grammar test file existence with read_text (implicit existence check), asserts native root import/engine edge, no #[path], and no authored grammar manifest entries.'}]
(receipts/'inventory-results.json').write_text(json.dumps(results,indent=2)+'\n')
for kind, data in results.items():
 if isinstance(data,dict): print(kind,data.get('counts'), 'missing',data.get('missing_from_head'))
print('union', {k:(v['union_count'],v['union_missing_count']) for k,v in results.items() if isinstance(v,dict) and 'union_count' in v})
