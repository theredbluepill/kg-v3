"""Compare separately collected inventories and source declarations."""
from pathlib import Path
import ast
from collections import Counter
import hashlib
import json
import re
import subprocess

receipts = Path(__file__).resolve().parent
manifest = json.loads((receipts / 'manifest.json').read_text())
scratch = Path(manifest['scratch'])
labels = ['base', 'adapter', 'head']
results = {}

for kind, suffix in [('pytest', 'pytest-collect-fixed-native'), ('cargo_engine', 'cargo-engine-list'), ('cargo_root', 'cargo-root-list')]:
    names_by_label = {}
    for label in labels:
        lines = (receipts / f'{label}-{suffix}.log').read_text().splitlines()
        if kind == 'pytest':
            names = [line for line in lines if line.startswith('tests/') and '::' in line]
        else:
            names = []
            binary = ''
            for line in lines:
                match = re.match(r'\s+Running (.+?) \(', line)
                if match:
                    binary = match[1]
                if line.endswith(': test'):
                    names.append(binary + '::' + line.removesuffix(': test'))
        assert len(set(names)) == len(names), (kind, label, 'duplicate collection identities')
        (receipts / f'{label}-{kind}-names.txt').write_text('\n'.join(sorted(names)) + '\n')
        names_by_label[label] = set(names)
    parent_union = names_by_label['base'] | names_by_label['adapter']
    results[kind] = {
        'counts': {label: len(names) for label, names in names_by_label.items()},
        'missing_from_head': {label: sorted(names_by_label[label] - names_by_label['head']) for label in labels[:2]},
        'added_at_head': {label: sorted(names_by_label['head'] - names_by_label[label]) for label in labels[:2]},
        'parent_union_count': len(parent_union),
        'parent_union_missing_count': len(parent_union - names_by_label['head']),
    }

def python_declarations(root):
    declarations = Counter()
    for path in (root / 'tests').rglob('*.py'):
        tree = ast.parse(path.read_text())
        def visit(nodes, scope=()):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, scope + (node.name,))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                    declarations[str(path.relative_to(root)) + '::' + '::'.join(scope + (node.name,))] += 1
        visit(tree.body)
    return declarations

def rust_declarations(root):
    declarations = Counter()
    for directory in ['src', 'tests', 'engine_rs']:
        for path in (root / directory).rglob('*.rs'):
            for name in re.findall(r'#\[test\]\s*(?:#\[[^\]]+\]\s*)*(?:pub\s+)?fn\s+(\w+)', path.read_text()):
                declarations[str(path.relative_to(root)) + '::' + name] += 1
    return declarations

for kind, collector in [('python_declarations', python_declarations), ('rust_test_declarations', rust_declarations)]:
    inventories = {label: collector(scratch / label) for label in labels}
    results[kind] = {
        'counts': {label: sum(items.values()) for label, items in inventories.items()},
        'duplicate_declarations': {label: {key: count for key, count in items.items() if count > 1} for label, items in inventories.items()},
        'missing_from_head': {label: sorted((inventories[label] - inventories['head']).elements()) for label in labels[:2]},
    }

test_files = {label: set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', manifest['revisions'][label], '--', 'tests'], cwd=receipts, text=True).splitlines()) for label in labels}
results['test_file_retention'] = {label: sorted(test_files[label] - test_files['head']) for label in labels[:2]}
results['pytest_replacements'] = [
    {
        'old': 'tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_rejects_kaggriculture_until_native',
        'replacement': 'tests/scripts/test_run_ppo.py::test_create_eval_env_keeps_orbit_env_and_builds_kaggriculture',
        'reason': 'Adapter replaces obsolete missing-native expectation with real native environment construction, retaining Orbit constructor assertions and adding new evaluation tests.',
    },
    {
        'old': 'tests/tools/test_check_engine_trim.py::test_grammar_bridge_is_retired_to_root_integration',
        'replacement': 'tests/tools/test_check_engine_trim.py::test_no_authored_grammar_path_include_after_root_engine_edge',
        'reason': 'Adapter retains old bridge absence and strengthens root-file-existence assertion to read/import/path-include/manifest assertions.',
    },
]
(receipts / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
for kind, result in results.items():
    if isinstance(result, dict):
        print(kind, result.get('counts'), result.get('missing_from_head', result))

hashes = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(receipts.iterdir()) if path.is_file() and path.name not in ['sha256.json', 'progress.log']}
(receipts / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
