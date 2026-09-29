from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

root = Path.cwd()
evidence = root / 'ops/rebuild-2026-09-29/merge-1.1b/verify-codex-8209193'
scratch = evidence / 'scratch'
manifest_path = 'engine_rs/TRIM_MANIFEST.json'

def blob(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'])

expected = blob('8209193', manifest_path)
second_parent = blob('16e56b6', manifest_path)
assert second_parent == expected
spec = importlib.util.spec_from_file_location('trim_updater', root / 'ops/rebuild-2026-09-29/1.1b/update_trim_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = scratch
module.MANIFEST = scratch / manifest_path
module.MANIFEST.write_bytes(blob('69397da', manifest_path))
module.main()
actual = module.MANIFEST.read_bytes()
assert actual == expected
module.main()
assert module.MANIFEST.read_bytes() == expected
retained = json.loads(expected)['retained']
for entry in retained:
    assert blob('69397da', entry['path']) == blob('8209193', entry['path'])
for path in ['engine_rs/tests/replay_parity.rs', 'engine_rs/fixtures/generated/MANIFEST.json']:
    assert blob('16e56b6', path) == blob('8209193', path)
result = {
    'merge': subprocess.check_output(['git', 'rev-parse', '8209193'], text=True).strip(),
    'manifest_sha256': hashlib.sha256(actual).hexdigest(),
    'manifest_identical_to_second_parent': True,
    'updater_from_first_parent_reproduces_merge_byte_for_byte': True,
    'updater_idempotent': True,
    'retained_files_identical_to_first_parent': len(retained),
    'replay_test_and_generated_manifest_identical_to_second_parent': True,
}
(evidence / 'manifest-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
