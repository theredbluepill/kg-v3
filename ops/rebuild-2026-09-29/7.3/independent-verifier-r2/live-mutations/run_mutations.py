from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/7.3/independent-verifier-r2/live-mutations'
SOURCE = ROOT / 'python/owl/kaggriculture/native_evaluation.py'
TEST = ROOT / 'tests/kaggriculture/test_replay_export_integration.py'
original = SOURCE.read_bytes()
test_original = TEST.read_bytes()
sha = lambda data: hashlib.sha256(data).hexdigest()
mutations = [
    ('consumed_seed_plus_one', 'seeds[env_index] = env.seed_state()[1][env_index]', 'seeds[env_index] = env.seed_state()[1][env_index] + 1'),
    ('terminal_snapshot_after_reset', 'env.terminal_snapshot(env_index)', 'env.state_snapshot(env_index)'),
    ('seven_selected_instead_of_eight', 'selected = recorder.selected_games if recorder is not None else frozenset()', 'selected = frozenset(sorted(recorder.selected_games)[:-1]) if recorder is not None else frozenset()'),
    ('swap_decoded_action_seats', 'actions[env_index],', 'list(reversed(actions[env_index])),') ,
    ('remove_schedule_length_guard', '    if len(seat_assignments) != n_games:\n        raise ValueError("seat_assignments length must equal n_games")\n', ''),
]
plugin_template = '''import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("_live_mutant", Path(__file__).with_name("native_evaluation.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name == "test_replay_export_integration.py":
            item.module.evaluate_native_games = module.evaluate_native_games
'''
results = []
with tempfile.TemporaryDirectory(prefix='live-mutants-', dir=ROOT/'.codex-tmp') as name:
    scratch = Path(name)
    copy = scratch / 'native_evaluation.py'
    plugin = scratch / 'mutation_plugin.py'
    plugin.write_text(plugin_template)
    for label, before, after in mutations:
        copy.write_bytes(original)
        text = original.decode()
        assert text.count(before) == 1, (label, text.count(before))
        copy.write_text(text.replace(before, after))
        mutant_sha = sha(copy.read_bytes())
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(scratch))
        run = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', str(TEST), '-p', 'mutation_plugin'], cwd=ROOT, env=env, text=True, capture_output=True)
        (OUT/f'{label}.log').write_text(run.stdout + run.stderr)
        copy.write_bytes(original)
        restored = copy.read_bytes() == original
        record = dict(mutation=label, exit_code=run.returncode, original_sha256=sha(original), mutant_sha256=mutant_sha, restored_sha256=sha(copy.read_bytes()), restored_byte_for_byte=restored, summary=[line for line in run.stdout.splitlines() if ' failed' in line or ' passed' in line][-1:])
        results.append(record)
        print(json.dumps(record), flush=True)
        assert run.returncode == 1, (label, run.returncode)
        assert restored
    # Exercise the independent direct terminal_snapshot API oracle. The proxy
    # returns the correct None until a terminal record exists, then incorrectly
    # supplies the reset game's current state instead of the saved terminal state.
    proxy_plugin = '''from types import SimpleNamespace
from owl import rs
class BrokenTerminalEnv:
    def __init__(self, *args, **kwargs):
        self.env = rs.KaggricultureEnv(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.env, name)
    def terminal_snapshot(self, index):
        terminal = self.env.terminal_snapshot(index)
        return None if terminal is None else self.env.state_snapshot(index)
def pytest_collection_modifyitems(items):
    for item in items:
        if item.name == "test_live_terminal_snapshot_is_captured_before_auto_reset":
            item.module.rs = SimpleNamespace(KaggricultureEnv=BrokenTerminalEnv)
'''
    plugin.write_text(proxy_plugin)
    before_proxy_sha = sha(plugin_template.encode())
    proxy_sha = sha(plugin.read_bytes())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(scratch))
    run = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', str(TEST)+'::test_live_terminal_snapshot_is_captured_before_auto_reset', '-p', 'mutation_plugin'], cwd=ROOT, env=env, text=True, capture_output=True)
    (OUT/'terminal_api_returns_reset_state.log').write_text(run.stdout + run.stderr)
    plugin.write_text(plugin_template)
    record = dict(mutation='terminal_api_returns_reset_state', exit_code=run.returncode, original_sha256=before_proxy_sha, mutant_sha256=proxy_sha, restored_sha256=sha(plugin.read_bytes()), restored_byte_for_byte=plugin.read_text()==plugin_template, summary=[line for line in run.stdout.splitlines() if ' failed' in line or ' passed' in line][-1:])
    results.append(record)
    print(json.dumps(record), flush=True)
    assert run.returncode == 1
    assert record['restored_byte_for_byte']
assert SOURCE.read_bytes() == original
assert TEST.read_bytes() == test_original
receipt = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=sha(original), test=str(TEST.relative_to(ROOT)), test_sha256=sha(test_original), tracked_source_unchanged=True, tracked_test_unchanged=True, mutations=results)
(OUT/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
