import ast, collections, hashlib, io, json, re, subprocess, tarfile
from pathlib import Path
OUT=Path('/private/tmp/kg-merge84-inventory-docs')
refs={'integration':'HEAD^1','task12':'HEAD^2','merge':'HEAD'}
data={}
# Whole tracked trees; no checkout or writes to repository.
for label, ref in refs.items():
    buf=subprocess.check_output(['git','archive',ref])
    with tarfile.open(fileobj=io.BytesIO(buf)) as tar:
        texts={x.name:tar.extractfile(x).read().decode() for x in tar.getmembers() if x.isfile() and x.name.endswith(('.py','.rs','.md'))}
    tests={}; errors=[]
    for path, src in texts.items():
        if path.endswith('.py'):
            try: tree=ast.parse(src)
            except SyntaxError as exc:
                errors.append((path,str(exc))); continue
            def walk(nodes, prefix=''):
                for node in nodes:
                    if isinstance(node,ast.ClassDef): walk(node.body,prefix+node.name+'.')
                    elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                        key=f'{path}::{prefix}{node.name}'
                        tests[key]={'language':'python','line':node.lineno,'ast':ast.dump(node,include_attributes=False),'text':ast.get_source_segment(src,node)}
            walk(tree.body)
        elif path.endswith('.rs'):
            # Production uses ordinary #[test]. Any unfamiliar test annotations are recorded for inspection below.
            for match in re.finditer(r'(?m)^\s*#\[test\]\s*(?:#\[[^\n]*\]\s*)*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(',src):
                name=match.group(1); key=f'{path}::{name}'
                # Paths and function names are unique in this source set.
                assert key not in tests,key
                tests[key]={'language':'rust','line':src.count('\n',0,match.start())+1}
    log=texts['cookbook/log.md']; headings=re.findall(r'^## (.+)$',log,re.M)
    data[label]={'blobs':{parts[3]:parts[2] for line in subprocess.check_output(['git','ls-tree','-r',ref],text=True).splitlines() if len(parts:=line.split(maxsplit=3))==4},'ref':subprocess.check_output(['git','rev-parse',ref],text=True).strip(),'tests':tests,'parse_errors':errors,'log_headings':headings,'test_annotations':[(p,line.strip()) for p,s in texts.items() if p.endswith('.rs') for line in s.splitlines() if re.search(r'#\[.*(?:rstest|test_case|::test)',line)]}
    (OUT/f'{label}-tests.txt').write_text('\n'.join(sorted(tests))+'\n')
    (OUT/f'{label}-log-headings.txt').write_text('\n'.join(headings)+'\n')
merge=data['merge']['tests']; report={}
for label,d in data.items():
    counts=collections.Counter((v['language'],'ops' if k.startswith('ops/') else 'active') for k,v in d['tests'].items())
    report[label]={'commit':d['ref'],'counts':{':'.join(k):v for k,v in sorted(counts.items())},'parse_errors':d['parse_errors'],'unfamiliar_annotations':d['test_annotations'],'log_heading_count':len(d['log_headings']),'duplicate_log_headings':[h for h,c in collections.Counter(d['log_headings']).items() if c>1]}
    if label!='merge':
        report[label]['missing_names']=sorted(set(d['tests'])-set(merge))
        report[label]['changed_python_tests']=[k for k,v in d['tests'].items() if k in merge and v['language']=='python' and v['ast']!=merge[k]['ast']]
        report[label]['missing_log_headings']=sorted(set(d['log_headings'])-set(data['merge']['log_headings']))
        # Every same-path Rust source with tests compared byte-for-byte, or listed for diff inspection.
        names=[k.split('::')[0] for k,v in d['tests'].items() if v['language']=='rust']
        report[label]['changed_rust_test_files']=sorted({p for p in names if d['blobs'][p]!=data['merge']['blobs'][p]})
(OUT/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
