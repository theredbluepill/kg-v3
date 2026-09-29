"""Read-only custody checks for the Stage 1 stub and protected paths."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
brief = (root / 'ops/rebuild-2026-09-29/briefs/1.4.md').read_text()
block = brief.split('Full new surface for `python/owl/rs.pyi`')[1].split('```python\n')[1].split('```')[0]
expected = block[block.index('class KaggricultureRewardDict'):]
stub = (root / 'python/owl/rs.pyi').read_text()
actual = stub.split('# fmt: off\n')[1].split('# fmt: on')[0]
assert actual == expected, 'reviewed native stub differs'
print('Verbatim Task 1.4 declarations: match; imports deduplicated')

def methods(text, class_name):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

native = methods(block, 'KaggricultureEnv')
fake = methods((root / 'tests/kaggriculture/fake_env.py').read_text(), 'FakeKaggricultureEnv')
for name, method in native.items():
    assert ast.dump(method.args) == ast.dump(fake[name].args), name
print(f'Exact fake signatures: {len(native)} matched, no kwargs catch-all')

protected = ['python/owl/rl.py', 'python/owl/train/config.py', 'python/owl/train/ppo.py',
             'python/owl/train/distributed.py', 'scripts/run_ppo.py',
             'python/owl/kaggriculture/gpu_grammar.py', 'src', 'engine_rs',
             'tests/owl', 'tests/scripts', 'tests/tools', 'pyproject.toml', 'uv.lock',
             'Cargo.toml', 'Cargo.lock']
changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *protected], cwd=root, text=True)
assert not changed, changed
print('Protected Rust/engine/trainer/Orbit/tests/dependencies: byte unchanged vs HEAD')
print('Stub additions SHA256:', hashlib.sha256(expected.encode()).hexdigest())
