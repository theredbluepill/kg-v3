"""Run all unchanged test assertions in memory-bounded file-sized processes."""
from pathlib import Path
from run_check import ROOT, run

failures = []
for path in sorted((ROOT / 'tests').rglob('test_*.py')):
    relative = path.relative_to(ROOT)
    name = 'batched/' + str(relative.with_suffix('')).replace('/', '-')
    command = ['uv', 'run', '--offline', 'pytest', str(relative), '-q', '-rs']
    if path.name == 'test_base_generics_typing.py':
        # This pure mypy test uses no project fixtures or runtime compile state.
        # Avoid importing torch solely via the global autouse compile fixture.
        command.append('--noconftest')
    code = run(name, command)
    if code:
        failures.append(str(relative))
print('FAILED/BUDGET-LIMITED FILES:', failures, flush=True)
raise SystemExit(bool(failures))
