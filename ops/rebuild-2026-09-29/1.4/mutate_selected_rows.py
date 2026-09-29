"""Prove selected-row test rejects the forbidden whole-batch publish path."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path(__file__).resolve().parents[3]
source = root / 'src/kaggriculture/env.rs'
original = source.read_bytes()
needle = b'commit_selected_rows(&mut pending.staging.buffers_mut(), out, selected);'
assert original.count(needle) == 1
status = None
try:
    source.write_bytes(original.replace(needle, b'let _ = selected; pending.staging.publish(out)?; // MUTATION'))
    result = subprocess.run(['python3', str(root/'ops/rebuild-2026-09-29/1.4/bounded.py'), '--name', 'e-selected-mutation', '--', 'cargo', 'test', '--offline', '--locked', '--lib', 'truncate_commits_only_selected_rows'], cwd=root, env=os.environ | {'RUST_TEST_THREADS':'1'}, check=False)
    status = result.returncode
    log = (root/'ops/rebuild-2026-09-29/1.4/e-selected-mutation.log').read_text()
    assert status == 101 and 'unselected observation bytes changed' in log
finally:
    source.write_bytes(original)
    (root/'ops/rebuild-2026-09-29/1.4/selected-mutation.json').write_text(json.dumps({'exit_status':status, 'original_sha256':hashlib.sha256(original).hexdigest(), 'restored_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
