import ast, collections, json, re, subprocess
from pathlib import Path
OUT=Path('/private/tmp/kg-merge84-inventory-docs')
def git(ref,path): return subprocess.check_output(['git','show',f'{ref}:{path}'],text=True)
def headings(src): return re.findall(r'^## (.+)$',src,re.M)
logs={r:headings(git(r,'cookbook/log.md')) for r in ['HEAD^1','HEAD^2','HEAD']}
unique12=[x for x in logs['HEAD^2'] if x not in logs['HEAD^1']]
assert len(logs['HEAD'])==71
assert len(set(logs['HEAD']))==71
assert logs['HEAD'][1:3]==unique12
assert logs['HEAD'][3:]==logs['HEAD^1']
assert set(logs['HEAD^1'])|set(logs['HEAD^2'])<=set(logs['HEAD'])
report={'log':{'integration':len(logs['HEAD^1']),'task12':len(logs['HEAD^2']),'task12_unique':unique12,'merge':71,'exact_order':'new merge entry; two unique Task1.2 entries in their original order; full integration sequence in original order','duplicate_headings':[]}}
# Top-level #[test] function segments for this known file, through next top-level item.
a=git('HEAD^2','engine_rs/tests/replay_parity.rs');b=git('HEAD','engine_rs/tests/replay_parity.rs')
def rusttests(src):
    return {m.group(1):m.group(0) for m in re.finditer(r'(?ms)^#\[test\]\s*(?:#\[[^\n]*\]\s*)*fn\s+(\w+)\s*\(.*?^}',src)}
at,bt=rusttests(a),rusttests(b)
assert len(at)==9 and len(bt)==19
assert set(at)<=set(bt)
report['old_replay_test_bodies_changed']=[k for k,v in at.items() if v!=bt[k]]
assert not report['old_replay_test_bodies_changed']
# Preserve Task1.1b full Live Differential / Known Divergences block exactly.
p='docs/rules-parity-coverage.md'
a=git('HEAD^1',p);b=git('HEAD',p)
start='### Kaggriculture Live Differential Parity\n'
assert a[a.index(start):].strip()==b[b.index(start):b.index('## Kaggriculture Native Grammar')].strip()
report['live_parity_sections_byte_equivalent']=True
# Literal testcase counts, separate from parametrized runtime invocations.
p='tests/tools/test_check_engine_trim.py'
params={}
for ref in ['HEAD^1','HEAD^2','HEAD']:
    tree=ast.parse(git(ref,p)); found=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name.startswith('test_task_authored_inventory_requires_exact'):
            case_list=node.decorator_list[0].args[1]
            found.append({'name':node.name,'cases':len(case_list.elts)})
    params[ref]=found
report['authored_exact_set_cases']=params
report['checks_claimed_in_committed_prepare_log']={'root':'164 passed; 2 ignored','engine':'41+18+9+19=87 passed; 0 ignored','python':'1337 passed; 4 skipped'}
report['finding']={'priority':'P3','issue':'Remaining current heads docs still say synthetic tables stand in until Task1.2 after Task1.2 has landed. Native grammar and coverage correctly state binding remains Task1.4. Reconcile current index/heads Reference description/model architecture wording with that boundary.','locations':['cookbook/references/index.md:23','cookbook/references/kaggriculture-grammar-heads-sit-behind-isaiahs-actor-projection.md:4','docs/model-architecture.md:752'],'scope':'Documentation-only; binding absence remains explicit and correct; no production change needed.'}
(OUT/'docs-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
