import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from owl.kaggriculture import gpu_grammar as g

root = Path.cwd()
out = root / 'ops/rebuild-2026-09-29/codex/verify-merge-84dd76ad'
receipt = root / 'ops/rebuild-2026-09-29/merge-1.2/native-tables-crosscheck'
with tempfile.TemporaryDirectory(prefix='kg-merge84-tables-', dir='/private/tmp') as directory:
    scratch = Path(directory)
    (scratch / 'src').mkdir()
    shutil.copyfile(receipt / 'Cargo.toml.txt', scratch / 'Cargo.toml')
    shutil.copyfile(receipt / 'main.rs.txt', scratch / 'src/main.rs')
    env = os.environ.copy()
    env['CARGO_BUILD_JOBS'] = '2'
    result = subprocess.run(['cargo', 'run', '--offline', '--quiet', '--manifest-path', str(scratch / 'Cargo.toml')], capture_output=True, check=True, env=env)
    (out / 'native-tables.json').write_bytes(result.stdout)
    (out / 'native-tables-build.log').write_bytes(result.stderr)
raw = json.loads(result.stdout)
tables = g.grammar_tables_from_arrays({k: np.array(v, dtype=bool) for k, v in raw.items()}).as_dict()
expected = g.expected_grammar_tables().as_dict()
assert tables.keys() == expected.keys()
report = {k: {'shape': list(v.shape), 'bits': v.numel(), 'mismatches': int((v != expected[k]).sum())} for k, v in tables.items()}
assert sum(v['bits'] for v in report.values()) == 964
assert sum(v['mismatches'] for v in report.values()) == 0
assert raw == json.loads((receipt / 'native-tables.json').read_bytes())
(out / 'native-tables-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print('8 fields, 964 bits, 0 mismatches; emitted values also equal the committed receipt')
