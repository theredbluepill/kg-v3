import ast
import json
from pathlib import Path
import re
import subprocess

OUT = Path('ops/rebuild-2026-09-29/codex/verify-merge-8rank-r1')
REVS = ('HEAD^1', 'HEAD^2', 'HEAD')
PREFIXES = ('python/', 'scripts/', 'src/', 'engine_rs/', 'tests/', 'configs/')

def git(*args):
    return subprocess.check_output(['git', *args])

def tree(rev):
    result = {}
    for line in git('ls-tree', '-rz', '--full-tree', rev).split(b'\0'):
        if not line:
            continue
        meta, name = line.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        if kind == 'blob':
            result[name.decode()] = oid
    return result

trees = {rev: tree(rev) for rev in REVS}
blobs = {}
for t in trees.values():
    for p, oid in t.items():
        if ((p.startswith('tests/') and p.endswith('.py')) or p.endswith('.rs')) and oid not in blobs:
            blobs[oid] = git('cat-file', 'blob', oid).decode()

class PythonNames(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.names = []
    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
    def visit_FunctionDef(self, node):
        if node.name.startswith('test_'):
            self.names.append('.'.join([*self.scope, node.name]))
    visit_AsyncFunctionDef = visit_FunctionDef

inventory = {}
for rev, t in trees.items():
    tests = {'python': [], 'rust': []}
    files = {'python': [], 'rust': []}
    for p, oid in t.items():
        if p.startswith('tests/') and p.endswith('.py'):
            visitor = PythonNames()
            visitor.visit(ast.parse(blobs[oid]))
            if visitor.names:
                files['python'].append(p)
                tests['python'].extend(p + '::' + n for n in visitor.names)
        elif p.startswith(('src/', 'engine_rs/')) and p.endswith('.rs'):
            names = re.findall(r'#\[test\]\s*(?:#\[[^\]]*\]\s*)*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', blobs[oid])
            if names:
                files['rust'].append(p)
                tests['rust'].extend(p + '::' + n for n in names)
    inventory[rev] = {'test_counts': {lang: len(v) for lang, v in tests.items()}, 'test_file_counts': {lang: len(v) for lang, v in files.items()}, 'tests': tests, 'test_files': files}

comparisons = {}
for rev in REVS[:2]:
    comparisons[rev] = {}
    for lang in ('python', 'rust'):
        before = set(inventory[rev]['tests'][lang])
        after = set(inventory['HEAD']['tests'][lang])
        missing = sorted(before - after)
        added = sorted(after - before)
        missing_names = sorted({v.rsplit('::', 1)[1] for v in missing} - {v.rsplit('::', 1)[1] for v in after})
        comparisons[rev][lang] = {'missing_path_names': missing, 'added_path_names': added, 'missing_names_at_any_path': missing_names}

scope = sorted(p for p in set.union(*(set(v) for v in trees.values())) if p.startswith(PREFIXES))
no_parent_match = []
from_parent = {'parent1_only': [], 'parent2_only': [], 'both': [], 'deleted_from_parent2': [], 'deleted_from_parent1': []}
for p in scope:
    h = trees['HEAD'].get(p)
    p1 = trees['HEAD^1'].get(p)
    p2 = trees['HEAD^2'].get(p)
    if h is None:
        if p1 is not None:
            from_parent['deleted_from_parent1'].append(p)
        if p2 is not None:
            from_parent['deleted_from_parent2'].append(p)
    elif h == p1 == p2:
        from_parent['both'].append(p)
    elif h == p1:
        from_parent['parent1_only'].append(p)
    elif h == p2:
        from_parent['parent2_only'].append(p)
    else:
        no_parent_match.append(p)
result = {'revisions': {r: git('rev-parse', r).decode().strip() for r in REVS}, 'inventory': inventory, 'comparisons': comparisons, 'scope_blob_comparison': {'prefixes': PREFIXES, 'count': len(scope), 'no_parent_match': no_parent_match, 'groups': from_parent}}
(OUT / 'parent_inventory.json').write_text(json.dumps(result, indent=2) + '\n')
summary = {'revisions': result['revisions'], 'counts': {r: {k: v for k, v in inv.items() if k.endswith('counts')} for r, inv in inventory.items()}, 'missing': {r: {lang: {k:v for k,v in d.items() if k.startswith('missing')} for lang,d in langs.items()} for r,langs in comparisons.items()}, 'new_tests_since_integration': comparisons['HEAD^1'], 'scope_blob_comparison': {'count': len(scope), 'no_parent_match': no_parent_match, 'group_counts': {k:len(v) for k,v in from_parent.items()}, 'parent2_only': from_parent['parent2_only'], 'deleted': {k:v for k,v in from_parent.items() if k.startswith('deleted')}}}
print(json.dumps(summary, indent=2))
