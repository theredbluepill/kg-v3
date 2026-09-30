"""Independent merge/receipt audit; writes only beside this untracked script."""
from __future__ import annotations

import ast
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'ops/rebuild-2026-09-29/gpu-checks-2026-09-29'
BASE = 'ca370893eff25f98a1a00a0bbcde82750eff1c4c'
MERGE = 'd793d16f23ee529f9fc49120558c8e9731b843f1'
GPU = '24380f3eb0e9c7090d874f8f1a45c0d4e3477f40'

def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], text=True)

def tree(ref: str) -> dict[str, str]:
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0]
            for line in git('ls-tree', '-r', ref).splitlines()}

def emit(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

HEAD = git('rev-parse', 'HEAD').strip()
COMMON = git('merge-base', BASE, GPU).strip()
trees = {r: tree(r) for r in (BASE, GPU, MERGE, HEAD, COMMON)}
all_paths = set().union(*[set(t) for t in trees.values()])
left = {p for p in all_paths if trees[BASE].get(p) != trees[COMMON].get(p)}
right = {p for p in all_paths if trees[GPU].get(p) != trees[COMMON].get(p)}
expected = {p: trees[GPU].get(p) if p in right else trees[BASE].get(p)
            for p in all_paths}
bad_merge = [p for p in sorted(all_paths) if trees[MERGE].get(p) != expected[p]]
delta = git('diff', '--numstat', BASE, MERGE).splitlines()
as_run = [p for p in trees[GPU] if '/gpu-checks-2026-09-29/pod/' in p]
source_prefixes = ('tests/', 'src/', 'python/', 'engine_rs/', 'scripts/', 'configs/', 'docs/')
source_changed = [p for p in sorted(all_paths)
                  if p.startswith(source_prefixes) and trees[BASE].get(p) != trees[HEAD].get(p)]
emit('merge.json', {
    'head': HEAD, 'base': BASE, 'merge': MERGE, 'gpu_parent': GPU,
    'merge_base': COMMON, 'left_right_overlap': sorted(left & right),
    'unexpected_original_merge_paths': bad_merge,
    'incoming_commits': git('rev-list', GPU, '^' + BASE).splitlines(),
    'original_merge_changed_files': len(delta),
    'original_merge_added_lines': sum(int(x.split('\t')[0]) for x in delta),
    'original_merge_deleted_lines': sum(int(x.split('\t')[1]) for x in delta),
    'current_changed_files': len(git('diff', '--name-only', BASE + '...' + HEAD).splitlines()),
    'source_config_docs_changed_from_base': source_changed,
    'as_run_files': len(as_run),
    'as_run_changed_from_gpu_parent': [p for p in as_run if trees[GPU][p] != trees[HEAD].get(p)],
})

def inventory(ref: str) -> tuple[set[str], dict[str, int]]:
    names: set[str] = set()
    counts: collections.Counter[str] = collections.Counter()
    paths = [p for p in trees[ref] if p.startswith(('tests/', 'src/', 'engine_rs/'))
             and p.endswith(('.py', '.rs'))]
    for path in paths:
        content = git('show', ref + ':' + path)
        if path.endswith('.py'):
            parsed = ast.parse(content)
            def visit(node: ast.AST, parents: list[str]) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith('test_'):
                            names.add(path + ':' + '.'.join(parents + [child.name]))
                            counts['python'] += 1
                        visit(child, parents + [child.name])
                    else:
                        visit(child, parents)
            visit(parsed, [])
        else:
            found = re.findall(r'#\[(?:test|rstest|test_case(?:\([^\n]*\))?)\](?:(?!\bfn\b)[\s\S])*?\bfn\s+(\w+)', content)
            counts['rust'] += len(found)
            names.update(path + ':' + n for n in found)
    return names, dict(counts)

inv = {r: inventory(r) for r in (BASE, GPU, MERGE, HEAD, COMMON)}
emit('test-names.json', {
    'python': sys.version,
    'method': 'AST-qualified Python test_ declarations under tests/; regex Rust #[test], #[rstest], #[test_case] declarations under src/ and engine_rs/. No ops snapshots. Not parameterized execution counts.',
    'counts': {r: {'unique_path_names': len(n), 'declarations_by_language': c} for r, (n, c) in inv.items()},
    'lost_from_base': sorted(inv[BASE][0] - inv[HEAD][0]),
    'added_from_base': sorted(inv[HEAD][0] - inv[BASE][0]),
    'lost_from_gpu_parent': sorted(inv[GPU][0] - inv[HEAD][0]),
    'gpu_parent_losses_already_absent_in_base': sorted((inv[GPU][0] - inv[HEAD][0]) - inv[BASE][0]),
    'lost_during_original_merge': sorted((inv[BASE][0] | (inv[GPU][0] - inv[COMMON][0])) - inv[MERGE][0]),
    'lost_after_original_merge': sorted(inv[MERGE][0] - inv[HEAD][0]),
})

def manifest(path: Path, prefix: Path) -> dict:
    entries = []
    bad = []
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        expected_hash, name = line.split(None, 1)
        file = prefix / name.removeprefix('*')
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        entries.append(str(file.relative_to(BUNDLE)))
        if actual != expected_hash:
            bad.append(str(file))
    return {'entries': len(entries), 'bad': bad, 'paths': entries}

whole = manifest(BUNDLE / 'MANIFEST.sha256', BUNDLE)
actual_files = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob('*') if p.is_file()
                and '__pycache__' not in p.parts and p.name != 'MANIFEST.sha256'}
whole['unlisted_files'] = sorted(actual_files - set(whole['paths']))
whole.pop('paths')
receipts = {}
for attempt in ('attempt1', 'attempt2'):
    p = BUNDLE / 'pod' / attempt
    receipts[attempt] = manifest(p / 'receipts/scripts.sha256', p)

spec = importlib.util.spec_from_file_location('receipt_summarizer', BUNDLE / 'summarize.py')
assert spec and spec.loader
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
summary.HERE = OUT
summary.main()
generated = (OUT / 'summary.json').read_bytes()
retained = (BUNDLE / 'summary.json').read_bytes()
timing_checks = []
for density in ('mid', 'dense'):
    p = BUNDLE / 'pod/attempt2'
    recs = [json.loads(line) for line in (p / f'c4_{density}_aten.jsonl').read_text().splitlines()]
    times = [r for r in recs if r['event'] == 'timing']
    derived = json.loads((p / f'c4_{density}_aten.derived.json').read_text())
    for r in times:
        assert len(r['cuda_event_ms']) == r['timed_iters'] == 20
        assert statistics.median(r['cuda_event_ms']) == r['median_ms']
        assert sorted(r['cuda_event_ms'])[17] == r['p90_ms']
        assert len(r['packed_tokens_per_chunk_first_iter']) == r['expected_chunks'] == r['pack_calls_per_iter']
        assert r['use_flash_attn_all_true']
    for ranks, split in derived['splits'].items():
        rt = {r['workload'].split('_')[1]: r for r in times if r['ranks'] == int(ranks)}
        for kind, r in rt.items():
            assert r['median_ms'] == split['median_ms'][kind]
            assert r['p90_ms'] == split['p90_ms'][kind]
        wall = (64*rt['A']['median_ms'] + rt['C']['median_ms'] + 16*rt['B']['median_ms'] + rt['D']['median_ms'])/1000
        assert math.isclose(wall, split['update_wall_s_median'], rel_tol=1e-14)
        assert math.isclose(16384/wall, split['global_sps'], rel_tol=1e-14)
        assert math.isclose(16384/int(ranks)/wall, split['per_rank_sps'], rel_tol=1e-14)
    timing_checks.append({'density': density, 'timing_records': len(times), 'rank_splits': len(derived['splits']), 'all_checks_pass': True})

emit('custody.json', {
    'manifest': whole, 'as_run_script_receipts': receipts,
    'summary_regenerated_byte_exact': generated == retained,
    'summary_sha256': hashlib.sha256(retained).hexdigest(),
    'independent_raw_timing_reduction': timing_checks,
})

results = (ROOT / 'ops/rebuild-2026-09-29/results.md').read_text()
headings = [x for x in results.splitlines() if x.startswith('## ')]
emit('results-order.json', {
    'headings': headings,
    'duplicate_headings': [x for x, n in collections.Counter(headings).items() if n > 1],
    'base_results_is_prefix': results.startswith(git('show', BASE + ':ops/rebuild-2026-09-29/results.md')),
    'original_merge_results_equal_gpu_parent': trees[MERGE]['ops/rebuild-2026-09-29/results.md'] == trees[GPU]['ops/rebuild-2026-09-29/results.md'],
})
print('merge, test-name, custody, summary and raw timing audits completed')
