from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

import c2_trunk_bwd as c2
import driver

HERE = Path(__file__).resolve().parent
BUNDLE = Path('/Users/poonszesen/kg-v3-m-gpu-receipts/ops/rebuild-2026-09-29/gpu-checks-2026-09-29')
FILES = ('c2_trunk_bwd.py', 'common.py', 'driver.py')
before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}
events = driver.events(BUNDLE / 'pod/attempt2/c2_aten_bwd.jsonl')
point = next(q for q in events if q.get('event') == 'result')
probes = []
for case in ('baseline', 'eager_nan', 'eager_nan_masks_relative_error'):
    compiled = torch.ones(513, 1, 1)
    eager = compiled.clone()
    mask = torch.ones(513, 1, dtype=torch.bool)
    if case != 'baseline':
        eager[0, 0, 0] = float('nan')
    if case == 'eager_nan_masks_relative_error':
        compiled[1, 0, 0] = 1.1  # >0.05 global relative bound, <0.5 per-token bound
    metrics = c2.cmp_dx(compiled, eager, mask)
    record = dict(point, dx_compiled_vs_eager=metrics)
    errors = driver._c2_point(record)
    probes.append({'case': case, 'dx_metrics': metrics, 'judge_errors': errors,
                   'should_reject': case != 'baseline', 'rejected': bool(errors)})
actual_nonfinite_fields = []
def visit(obj, location):
    if isinstance(obj, dict):
        for key, value in obj.items():
            visit(value, f'{location}.{key}')
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            visit(value, f'{location}[{index}]')
    elif isinstance(obj, float) and not math.isfinite(obj):
        actual_nonfinite_fields.append(location)
for attempt in ('attempt1', 'attempt2'):
    for index, record in enumerate(driver.events(BUNDLE / 'pod' / attempt / 'c2_aten_bwd.jsonl')):
        if record.get('event') == 'result':
            visit(record, f'{attempt}.result[{index}]')
after = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}
output = {'probes': probes, 'retained_nonfinite_numeric_fields': actual_nonfinite_fields,
          'scratch_hashes_before': before, 'scratch_hashes_after': after,
          'scratch_equals_source': all((HERE / name).read_bytes() == (BUNDLE / 'scripts' / name).read_bytes() for name in FILES)}
(HERE / 'repro_c2_eager_nan.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
