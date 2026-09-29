"""Run bounded offline Stage 1 verification and preserve unmasked exit codes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

root = Path(__file__).resolve().parents[3]
artifacts = Path(__file__).resolve().parent
env = os.environ | {'OMP_NUM_THREADS': '2', 'CARGO_BUILD_JOBS': '2',
                    'CARGO_NET_OFFLINE': 'true', 'UV_OFFLINE': 'true'}
checks = [
    ('rewards-configs-final', ['uv', 'run', '--offline', 'pytest', 'tests/kaggriculture/test_rewards.py', 'tests/kaggriculture/test_configs.py', '-q']),
    ('focused-final', ['uv', 'run', '--offline', 'pytest',
                      'tests/kaggriculture/test_rewards.py', 'tests/kaggriculture/test_configs.py',
                      'tests/kaggriculture/test_game.py', 'tests/kaggriculture/test_env.py',
                      'tests/kaggriculture/test_env_cuda_fence.py', 'tests/kaggriculture/test_codec.py', '-q']),
    ('env-contract-final', ['uv', 'run', '--offline', 'pytest', 'tests/kaggriculture/test_env.py', '-k', 'checks_contract', '-q']),
    ('env-fence-final', ['uv', 'run', '--offline', 'pytest', 'tests/kaggriculture/test_env.py', '-k', 'fence', '-q']),
    ('py-prepare-final', ['uvx', '--offline', '--from', 'rust-just', 'just', 'py-prepare']),
    ('abi-check-final', ['uv', 'run', '--offline', 'python', str(artifacts / 'check_contract.py')]),
    ('docs-lint-final', ['uvx', '--offline', 'pymarkdownlnt', 'scan', 'docs/rl-api-specs.md']),
    ('orbit-diff-final', ['git', 'diff', '--stat', '--', 'tests/owl', 'tests/scripts', 'tests/tools']),
    ('diff-check-final', ['git', 'diff', '--check']),
]
results = []
for name, command in checks:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=root, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed.stdout
    (artifacts / f'{name}.log').write_text(
        'COMMAND: ' + shlex.join(command) + '\n' + output +
        f'\nEXIT_CODE={completed.returncode}\n')
    record = {'name': name, 'command': shlex.join(command),
              'exit_code': completed.returncode,
              'wall_seconds': round(time.monotonic()-started, 3),
              'log': f'{name}.log'}
    results.append(record)
    (artifacts / 'final-checks.json').write_text(json.dumps(results, indent=2)+'\n')
    print(name, 'exit', completed.returncode, flush=True)
    print('\n'.join(output.splitlines()[-6:]), flush=True)
raise SystemExit(int(any(result['exit_code'] for result in results)))
