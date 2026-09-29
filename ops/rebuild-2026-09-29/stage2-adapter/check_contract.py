"""Stage 2 declaration, fake-signature and protected-path custody checks."""
from __future__ import annotations
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
brief = (ROOT / 'ops/rebuild-2026-09-29/briefs/1.4.md').read_text()
block = brief.split('Full new surface for `python/owl/rs.pyi`')[1].split('```python\n')[1].split('```')[0]
expected = {n.name: n for n in ast.parse(block).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
actual_tree = ast.parse((ROOT / 'python/owl/rs.pyi').read_text())
for name, declaration in expected.items():
    candidates = [n for n in actual_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
    assert len(candidates) == 1, name
    assert ast.dump(candidates[0]) == ast.dump(declaration), name
print(f'Brief 1.4 stub declarations: {len(expected)} AST-identical, unique; formatting/order ignored')

def methods(text: str, class_name: str) -> dict[str, ast.FunctionDef]:
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

native = methods(block, 'KaggricultureEnv')
fake = methods((ROOT / 'tests/kaggriculture/fake_env.py').read_text(), 'FakeKaggricultureEnv')
for name, method in native.items():
    assert ast.dump(method.args) == ast.dump(fake[name].args), name
print(f'Exact fake signatures: {len(native)} matched')

protected = ['python/owl/rl.py', 'python/owl/train/config.py', 'python/owl/train/ppo.py',
             'python/owl/train/distributed.py', 'python/owl/rs.pyi', 'engine_rs',
             'tests/owl', 'tests/tools', 'pyproject.toml', 'uv.lock', 'Cargo.toml', 'Cargo.lock',
             'cookbook']
changed = subprocess.check_output(['git', 'diff', '--name-only', '558ac3c', '--', *protected], cwd=ROOT, text=True)
assert not changed, changed
print('Protected engine/trainer/Orbit/dependencies/stub/cookbook: unchanged vs 558ac3c')
for path in (ROOT / 'tests/scripts').glob('test_*.py'):
    old = subprocess.check_output(['git', 'show', f'558ac3c:{path.relative_to(ROOT)}'], cwd=ROOT, text=True)
    new = path.read_text()
    if path.name != 'test_run_ppo.py':
        assert old == new, str(path)
        continue
    old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
    new_nodes = {n.name: n for n in ast.parse(new).body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    count = 0
    for node in ast.parse(old).body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if 'kaggriculture' in node.name or node.name == 'test_resume_startup_checks_the_runtime_adapted_workload':
            continue
        start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
        other = new_nodes[node.name]
        new_start = min([other.lineno, *[d.lineno for d in other.decorator_list]]) - 1
        assert old_lines[start:node.end_lineno] == new_lines[new_start:other.end_lineno], node.name
        count += 1
    print(f'test_run_ppo.py: {count} non-Kaggriculture definitions byte-identical')
