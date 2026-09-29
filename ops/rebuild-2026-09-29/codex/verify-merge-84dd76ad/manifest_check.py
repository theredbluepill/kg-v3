import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
out = root / 'ops/rebuild-2026-09-29/codex/verify-merge-84dd76ad'
manifest_path = 'engine_rs/TRIM_MANIFEST.json'
updater_path = 'ops/rebuild-2026-09-29/merge-1.2/update_trim_manifest.py'
revs = ['90ed86c', 'e1458d2', '7877c46', '84dd76ad']
def blob(rev, path):
    return subprocess.check_output(['git', 'show', f'{rev}:{path}'])
def sha(data):
    return hashlib.sha256(data).hexdigest()
raws = [blob(r, manifest_path) for r in revs]
ms = [json.loads(b) for b in raws]
base, ours, theirs, merged = ms
assert (root / updater_path).read_bytes() == blob('84dd76ad', updater_path)
assert (root / manifest_path).read_bytes() == raws[-1]
for key in ('schema_version', 'reference_commit', 'retained', 'excluded'):
    assert all(m[key] == merged[key] for m in ms), key
for key, next_key in [('retained', 'excluded'), ('excluded', 'authored')]:
    sections = [b.split(f'  "{key}": '.encode(), 1)[1].split(f'  "{next_key}": '.encode(), 1)[0] for b in raws]
    assert len(set(sections)) == 1, key
retained_hashes = {}
for entry in merged['retained']:
    path = entry['path']
    contents = [blob(r, path) for r in revs]
    contents.append((root / path).read_bytes())
    assert len(set(contents)) == 1, path
    assert sha(contents[-1]) == entry['sha256'], path
    retained_hashes[path] = sha(contents[-1])
for entry in merged['excluded']:
    assert not (root / entry['path']).exists(), entry['path']
expected_authored = {'engine_rs/tests/replay_parity.rs', 'engine_rs/tests/grammar_kernel.rs', 'engine_rs/fixtures/generated/MANIFEST.json'}
assert {e['path'] for e in merged['authored']} == expected_authored
for entry in merged['authored']:
    assert sha((root / entry['path']).read_bytes()) == entry['sha256']
reason_maps = [{e['path']: e['reason'] for e in m['non_engine_changes']} for m in ms]
b, o, t, m = reason_maps
both_changed = {p for p in o.keys() & t.keys() if o[p] != t[p] and o[p] != b.get(p) and t[p] != b.get(p)}
assert both_changed == {'scripts/check_engine_trim.py', 'tests/tools/test_check_engine_trim.py', 'docs/rules-parity-coverage.md', 'cookbook/log.md', 'cookbook/references/index.md'}
assert m.keys() == o.keys() | t.keys()
for p in m:
    if p in both_changed:
        continue
    expected = o.get(p) if o.get(p) == t.get(p) or p not in t or t.get(p) == b.get(p) else t.get(p)
    assert m[p] == expected, p
original = (root / manifest_path).read_bytes()
digests = [sha(original)]
try:
    for _ in range(2):
        result = subprocess.run([sys.executable, updater_path], capture_output=True, check=True)
        assert not result.stdout and not result.stderr
        generated = (root / manifest_path).read_bytes()
        digests.append(sha(generated))
        assert generated == original, 'updater output differs'
finally:
    if (root / manifest_path).read_bytes() != original:
        (root / manifest_path).write_bytes(original)
assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']) == b''
report = {'merge': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'manifest_sha256_before_after_twice': digests, 'updater_sha256': sha(blob('84dd76ad', updater_path)), 'retained_count': len(merged['retained']), 'excluded_count': len(merged['excluded']), 'authored_count': len(merged['authored']), 'non_engine_count': len(merged['non_engine_changes']), 'new_non_engine_paths_vs_integration': sorted(m.keys() - o.keys()), 'both_changed_reasons': sorted(both_changed), 'retained_sha256_all_four_revisions_and_worktree_equal': retained_hashes, 'tracked_status_empty': True}
(out / 'manifest-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
