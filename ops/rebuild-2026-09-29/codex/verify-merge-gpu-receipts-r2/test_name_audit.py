"""Reproduce the original broad regex scan and reconcile its scope with review r1.

Run from the repository root using Python 3.9 to reproduce r1's AST skips, or
Python 3.12+ to show those newer-syntax files parsing. This performs no tests.
The original 1,424 count includes historical test copies under ops/. The live
source inventory is 1,257 declarations, excluding ops/: 930 Python, 327 Rust.
Names are inventory entries, not parameterized pytest test-case counts.
"""
import ast
import collections
import json
from pathlib import Path
import re
import subprocess as sp
import sys

REFS = ['ca37089', '24380f3', 'd793d16', '3f26e49']
PY_RE = re.compile(r'^\s*(?:async )?def (test_\w+)\s*\(', re.M)
RS_RE = re.compile(r'#\[(?:test|rstest)\][\s\S]*?\bfn (\w+)\s*\(')
R1_RS_RE = re.compile(
    r'#\[(?:test|rstest|test_case(?:\([^\n]*\))?)\]'
    r'(?:(?!\bfn\b)[\s\S])*?\bfn\s+(\w+)'
)


def blobs(ref):
    paths = sp.check_output(
        ['git', 'ls-tree', '-r', '--name-only', ref], text=True
    ).splitlines()
    paths = [p for p in paths if p.endswith(('.py', '.rs'))]
    raw = sp.check_output(
        ['git', 'cat-file', '--batch'],
        input=('\n'.join(ref + ':' + p for p in paths) + '\n').encode(),
    )
    pos = 0
    for path in paths:
        end = raw.index(b'\n', pos)
        size = int(raw[pos:end].split()[-1])
        content = raw[end + 1:end + 1 + size].decode()
        pos = end + size + 2
        yield path, content


def extract(ref):
    broad = set()
    current = set()
    prior_ast = set()
    skipped = []
    historical = {}
    language_counts = collections.Counter()
    for path, content in blobs(ref):
        # EXACT selection/extraction used by original test-name-audit.json.
        selected = path.startswith(('tests/', 'src/', 'engine_rs/')) or '/test_' in path
        pattern = PY_RE if path.endswith('.py') else RS_RE
        names = {path + ':' + m[1] for m in pattern.finditer(content)}
        if selected:
            broad.update(names)
            if not path.startswith('ops/'):
                current.update(names)
                language_counts[Path(path).suffix] += len(names)
            elif names:
                historical[path] = len(names)
        # R1 included all non-ops Python/Rust and AST-qualified Python names.
        if path.startswith('ops/'):
            continue
        if path.endswith('.py'):
            try:
                root = ast.parse(content)
            except SyntaxError as error:
                skipped.append({'path': path, 'regex_test_count': len(names),
                                'error': str(error)})
                continue
            def walk(node, parts):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith('test_'):
                            prior_ast.add(path + ':' + '.'.join(parts + [child.name]))
                        walk(child, parts + [child.name])
                    else:
                        walk(child, parts)
            walk(root, [])
        else:
            prior_ast.update(path + ':' + m[1] for m in R1_RS_RE.finditer(content))
    return broad, current, {
        'original_broad_count': len(broad),
        'live_source_count_excluding_ops': len(current),
        'live_source_counts_by_language': dict(language_counts),
        'historical_ops_copies': historical,
        'r1_ast_method_count_on_this_python': len(prior_ast),
        'r1_ast_skipped_files_on_this_python': skipped,
    }


report = {'python': sys.version, 'refs': {}}
sets = {}
for ref in REFS:
    broad, current, details = extract(ref)
    report['refs'][ref] = details
    sets[ref] = current
report['current_source_conservation'] = {
    'missing_from_base': sorted(sets['ca37089'] - sets['3f26e49']),
    'missing_from_gpu_parent': sorted(sets['24380f3'] - sets['3f26e49']),
    'missing_from_gpu_already_absent_in_base': sorted(
        (sets['24380f3'] - sets['3f26e49']) - sets['ca37089']
    ),
    'added_since_base': sorted(sets['3f26e49'] - sets['ca37089']),
}
output = Path(__file__).with_name('test-name-count-reconciliation.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
