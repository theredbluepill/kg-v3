"""R3 source mutants run through targeted tests without changing tracked modules."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/7.3/independent-verifier-r3/mutations'
INTEGRATION = 'tests/kaggriculture/test_replay_export_integration.py'
RECORDER = 'tests/kaggriculture/test_replay_export.py'
SOURCES = {
 'native_evaluation': ROOT / 'python/owl/kaggriculture/native_evaluation.py',
 'replay_export': ROOT / 'python/owl/kaggriculture/replay_export.py',
}
originals = {name:path.read_bytes() for name,path in SOURCES.items()}
sha = lambda data: hashlib.sha256(data).hexdigest()
error_targets = [INTEGRATION + '::' + name for name in (
 'test_selected_decoder_rejection_writes_error_custody_for_every_active_game',
 'test_native_transaction_rejection_writes_error_custody',
 'test_error_custody_publication_failure_keeps_the_original_exception')]
cases = [
 ('consumed_seed_plus_one', 'native_evaluation', 'seeds[env_index] = env.seed_state()[1][env_index]', 'seeds[env_index] = env.seed_state()[1][env_index] + 1', [INTEGRATION + '::test_live_env_consumed_seed_custody']),
 ('terminal_snapshot_after_reset', 'native_evaluation', 'env.terminal_snapshot(env_index)', 'env.state_snapshot(env_index)', [INTEGRATION + '::test_evaluate_games_exports_eight_complete_episodes']),
 ('seven_selected_instead_of_eight', 'native_evaluation', 'selected = recorder.selected_games if recorder is not None else frozenset()', 'selected = frozenset(sorted(recorder.selected_games)[:-1]) if recorder is not None else frozenset()', [INTEGRATION + '::test_evaluate_games_exports_eight_complete_episodes']),
 ('swap_decoded_action_seats', 'native_evaluation', 'actions[env_index],', 'list(reversed(actions[env_index])),', [INTEGRATION + '::test_evaluate_games_exports_eight_complete_episodes']),
 ('remove_schedule_length_guard', 'native_evaluation', '    if len(seat_assignments) != n_games:\n        raise ValueError("seat_assignments length must equal n_games")\n', '', [INTEGRATION + '::test_native_evaluation_rejects_inconsistent_schedule']),
 ('abort_handler_publishes_nothing', 'native_evaluation', '            _publish_error_custody(recorder, error)', '            pass', error_targets),
 ('publication_failure_escapes', 'replay_export', '            except Exception as publication_error:\n                failures.append((game_ordinal, publication_error))', '            except Exception as publication_error:\n                raise publication_error', error_targets),
 ('action_copy_is_shallow', 'replay_export', 'transition: dict[str, Any] = {"actions": deepcopy(list(actions))}', 'transition: dict[str, Any] = {"actions": list(actions)}', [RECORDER + '::test_recorder_retains_consumed_seed_actions_tokens_and_custody']),
 ('token_copy_is_shallow', 'replay_export', 'transition["tokens"] = deepcopy(list(tokens))', 'transition["tokens"] = list(tokens)', [RECORDER + '::test_recorder_retains_consumed_seed_actions_tokens_and_custody']),
]
results=[]
with tempfile.TemporaryDirectory(prefix='r3-python-mutants-', dir=ROOT/'.codex-tmp') as directory:
 scratch=Path(directory)
 plugin=scratch/'mutation_plugin.py'
 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(scratch)+os.pathsep+str(ROOT))
 def run(label,targets):
  completed=subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *targets, '-p', 'mutation_plugin'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
  (OUT/(label+'.log')).write_text(completed.stdout+completed.stderr)
  summary=[line for line in completed.stdout.splitlines() if ' failed' in line or ' passed' in line][-1:]
  return completed.returncode,summary
 for label,module,old,new,targets in cases:
  raw=originals[module]
  text=raw.decode()
  assert text.count(old)==1,(label,text.count(old))
  path=scratch/(module+'.py')
  path.write_bytes(raw)
  plugin.write_text('''import importlib.util
import sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("_r3_mutant", Path(__file__).with_name("'''+module+'''.py"))
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in ("test_replay_export.py", "test_replay_export_integration.py"):
'''+('            item.module.evaluate_native_games=module.evaluate_native_games\n' if module=='native_evaluation' else '            item.module.replay_export=module\n'))
  baseline,summary=run(label+'-baseline',targets)
  assert baseline==0,(label,baseline,summary)
  changed=text.replace(old,new).encode()
  try:
   path.write_bytes(changed)
   exit_code,summary=run(label,targets)
   assert exit_code==1,(label,exit_code,summary)
  finally:
   path.write_bytes(raw)
  assert path.read_bytes()==raw
  restored_code,restored_summary=run(label+'-restored',targets)
  assert restored_code==0,(label,restored_code,restored_summary)
  record={'name':label,'source':str(SOURCES[module].relative_to(ROOT)),'exit_code':exit_code,'summary':summary,'baseline_exit_code':baseline,'restored_exit_code':restored_code,'restored_summary':restored_summary,'original_sha256':sha(raw),'mutated_sha256':sha(changed),'restored_sha256':sha(path.read_bytes()),'restored_byte_identical':path.read_bytes()==raw}
  results.append(record)
  print(json.dumps(record),flush=True)
 # Native terminal API oracle: a proxy returns the reset state when a terminal record exists.
 original_plugin='def pytest_collection_modifyitems(items):\n    pass\n'
 plugin.write_text(original_plugin)
 targets=[INTEGRATION+'::test_live_terminal_snapshot_is_captured_before_auto_reset']
 assert run('terminal_api_baseline',targets)[0]==0
 proxy='''from types import SimpleNamespace
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
 try:
  plugin.write_text(proxy)
  code,summary=run('terminal_api_returns_reset_state',targets)
  assert code==1
 finally:
  plugin.write_text(original_plugin)
 assert plugin.read_text()==original_plugin
 assert run('terminal_api_restored',targets)[0]==0
 results.append({'name':'terminal_api_returns_reset_state','exit_code':code,'summary':summary,'original_sha256':sha(original_plugin.encode()),'mutated_sha256':sha(proxy.encode()),'restored_sha256':sha(plugin.read_bytes()),'restored_byte_identical':plugin.read_text()==original_plugin})
assert all(path.read_bytes()==originals[name] for name,path in SOURCES.items())
receipt={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mutations':results,'mutations_rejected':len(results),'tracked_sources_unchanged':True,'source_hashes':{name:sha(raw) for name,raw in originals.items()}}
(OUT/'python-mutations.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'mutations_rejected':len(results),'tracked_sources_unchanged':True}),flush=True)
