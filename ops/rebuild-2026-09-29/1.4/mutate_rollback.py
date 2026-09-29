"""Move a live seed write before admission; both real-core rollback tests must fail."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path(__file__).resolve().parents[3]
source = root / 'src/kaggriculture/env.rs'
original = source.read_bytes()
receipts = []
try:
    for method, test, name in [
        ('prepare_step', 'batch_failure_preserves_every_published_byte', 'd-mutation-step-live-seed'),
        ('prepare_reset', 'reset_and_truncate_failures_preserve_every_published_byte', 'd-mutation-reset-live-seed'),
    ]:
        text = original.decode()
        start = text.index(f'    pub fn {method}(')
        point = text.index('        caught(|| {', start)
        text = text[:point] + '        self.stream.next += self.stream.stride; // MUTATION: premature live commit\n' + text[point:]
        source.write_text(text)
        result = subprocess.run(['python3', str(root/'ops/rebuild-2026-09-29/1.4/bounded.py'), '--name', name,
                                 '--','cargo','test','--offline','--locked','--lib',test], cwd=root,
                                env=os.environ | {'RUST_TEST_THREADS':'1'}, check=False)
        receipts.append({'method':method,'test':test,'exit_status':result.returncode})
        if result.returncode != 101:
            raise RuntimeError(f'expected assertion failure, got {result.returncode}')
        log = (root/f'ops/rebuild-2026-09-29/1.4/{name}.log').read_text()
        if 'assertion `left == right` failed' not in log:
            raise RuntimeError('mutation did not reach rollback assertion')
        source.write_bytes(original)
finally:
    source.write_bytes(original)
    (root/'ops/rebuild-2026-09-29/1.4/rollback-mutations.json').write_text(json.dumps({
        'mutations':receipts,'restored_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_sha256':hashlib.sha256(original).hexdigest()},indent=2)+'\n')
