import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

OUT = Path('ops/rebuild-2026-09-29/verify-merge-1.3/parent-preservation')
REVS = {'integration':'b51b0c0382a13a72854860ae17d7798f0875e4ee','task13':'dc6b200','merged':'HEAD'}
def git(*args):
    return subprocess.check_output(['git', *args])

def blob_map(rev):
    result = {}
    for row in git('ls-tree','-r',rev).decode().splitlines():
        metadata,path = row.split('\t',1)
        mode,kind,oid = metadata.split()
        result[path] = oid
    return result

def sources(blobs):
    selected = {p:o for p,o in blobs.items() if p.endswith(('.py','.rs')) and p.startswith(('tests/','src/','engine_rs/src/','engine_rs/tests/'))}
    stream = subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    output,_=stream.communicate(('\n'.join(selected.values())+'\n').encode())
    offset=0
    result={}
    for path in selected:
        end=output.index(b'\n',offset)
        size=int(output[offset:end].split()[-1])
        offset=end+1
        result[path]=output[offset:offset+size].decode()
        offset+=size+1
    return result

def pytests(path,code):
    found={}
    def descend(node,scope):
        for child in node.body:
            if isinstance(child,ast.ClassDef):
                descend(child,scope+[child.name])
            elif isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and child.name.startswith('test_'):
                name='::'.join([path,*scope,child.name])
                found[name]={'line':child.lineno,'sha256_ast':hashlib.sha256(ast.dump(child).encode()).hexdigest(),'decorators':[ast.unparse(d) for d in child.decorator_list]}
    descend(ast.parse(code),[])
    return found

def rusttests(path,code):
    found={}
    pattern=r'#\[test\](?:(?!#\[test\]).)*?\bfn\s+(\w+)\s*\('
    for match in re.finditer(pattern,code,re.S):
        name=f'{path}::{match.group(1)}'
        found[name]={'line':code[:match.start()].count('\n')+1}
    return found

all_data={}
blobs={}
for name,rev in REVS.items():
    blobs[name]=blob_map(rev)
    result={'revision':git('rev-parse',rev).decode().strip(),'python':{},'rust':{}}
    for path,code in sources(blobs[name]).items():
        if path.endswith('.py') and path.startswith('tests/'):
            result['python'].update(pytests(path,code))
        elif path.endswith('.rs'):
            result['rust'].update(rusttests(path,code))
    all_data[name]=result
    (OUT/f'{name}-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
summary={'counts':{name:{kind:len(data[kind]) for kind in ('python','rust')} for name,data in all_data.items()},'diffs':{}}
for parent in ('integration','task13'):
    summary['diffs'][parent]={}
    for kind in ('python','rust'):
        original=all_data[parent][kind]; merged=all_data['merged'][kind]
        summary['diffs'][parent][kind]={'removed':sorted(original.keys()-merged.keys()),'added':sorted(merged.keys()-original.keys())}
        if kind=='python':
            summary['diffs'][parent][kind]['ast_changed']=[name for name in sorted(original.keys() & merged.keys()) if original[name]['sha256_ast']!=merged[name]['sha256_ast']]
# List authored non-ops merge results that equal neither parent: these are precisely merge-specific changes.
summary['equals_neither_parent_non_ops']=[p for p,oid in blobs['merged'].items() if not p.startswith('ops/') and oid not in (blobs['integration'].get(p),blobs['task13'].get(p))]
summary['both_parents_different_non_ops']=[p for p,oid in blobs['merged'].items() if not p.startswith('ops/') and p in blobs['integration'] and p in blobs['task13'] and blobs['integration'][p]!=blobs['task13'][p]]
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({**summary,'diffs':{p:{k:{key:(value if key!='added' else len(value)) for key,value in d.items()} for k,d in kinds.items()} for p,kinds in summary['diffs'].items()}},indent=2))
