import hashlib
import json
import os
from pathlib import Path
import subprocess

OUT = Path(__file__).parent
MUTATIONS = [
    ('unsafe-replay-index', 'python/owl/model/kaggriculture_actor.py',
     'safe = raw.clamp(0, width - 1)', 'safe = raw',
     'tests/kaggriculture/test_model_heads.py::test_out_of_range_indices_are_rejected_without_a_fault'),
    ('inclusive-raw-hire', 'python/owl/model/kaggriculture_actor.py',
     'prior = raw_hires.cumsum(dim=-1) - raw_hires',
     'prior = raw_hires.cumsum(dim=-1)',
     'tests/kaggriculture/test_model_heads.py::test_hire_coupling_enumeration_equals_replayed_density'),
    ('head-nonstrict-bound', 'python/owl/model/kaggriculture.py',
     'rows = (_GEMM_ELEMENT_LIMIT - 1) // (kt.MAX_FRAMES * head_gemm_width(config))',
     'rows = _GEMM_ELEMENT_LIMIT // (kt.MAX_FRAMES * head_gemm_width(config))',
     'tests/kaggriculture/test_model_heads.py::test_head_rows_per_chunk_follows_the_limit'),
]
results = []
for name, filename, before, after, test in MUTATIONS:
    path = Path(filename)
    original = path.read_bytes()
    assert original.count(before.encode()) == 1
    checksum = hashlib.sha256(original).hexdigest()
    (OUT / (name + '.original')).write_bytes(original)
    try:
        path.write_bytes(original.replace(before.encode(), after.encode()))
        with (OUT / ('mutation-' + name + '.log')).open('w') as log:
            run = subprocess.run(['uv', 'run', 'pytest', test, '-q'],
                                 stdout=log, stderr=subprocess.STDOUT,
                                 env=os.environ | {'OMP_NUM_THREADS': '2'}, timeout=180)
    finally:
        path.write_bytes(original)
        assert path.read_bytes() == original
    result = dict(name=name, file=filename, test=test, exit_code=run.returncode,
                  sha256_before=checksum, sha256_after=hashlib.sha256(path.read_bytes()).hexdigest())
    results.append(result)
    (OUT / 'mutations.json').write_text(json.dumps(results, indent=2) + '\n')
    print(name, 'exit', run.returncode, 'restored', flush=True)
    assert run.returncode == 1, result
