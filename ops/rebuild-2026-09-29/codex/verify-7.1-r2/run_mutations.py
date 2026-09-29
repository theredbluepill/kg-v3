from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import time

root = Path(__file__).resolve().parent
scratch = Path.cwd() / '.codex-tmp/verify-7-1-r2/scratch'
crate = scratch / 'opponents_rs'
env = dict(os.environ, CARGO_TARGET_DIR=str(Path.cwd() / '.codex-tmp/verify-7-1/mutation-target'))
base_command = ['cargo', 'test', '--manifest-path', str(crate / 'Cargo.toml'), '--locked', '--offline']
parity = ['--test', 'oracle_parity', 'original_python_oracles_match_native_actions_and_states', '--', '--nocapture']
inventory = {str(p.relative_to(scratch)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in scratch.rglob('*') if p.is_file()}
results = []

def run(name, extra):
    previous = root / f'mutations-{name}.log'
    if os.environ.get('MUTATION_RESUME') == '1' and previous.exists():
        output = previous.read_text()
        if 'test result:' in output and 'could not compile' not in output:
            code = 101 if 'test result: FAILED' in output else 0
            print(name, code, 'previous completed receipt', flush=True)
            return subprocess.CompletedProcess(base_command + extra, code, output)
    start = time.monotonic()
    completed = subprocess.run(base_command + extra, cwd=scratch, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (root / f'mutations-{name}.log').write_text(completed.stdout)
    print(name, completed.returncode, f'{time.monotonic() - start:.2f}s', flush=True)
    return completed

baseline = run('baseline', parity)
assert baseline.returncode == 0
manifest = json.loads((crate / 'fixtures/oracle/MANIFEST.json').read_text())
for index, entry in enumerate(manifest['traces']):
    path = crate / 'fixtures/oracle' / entry['path']
    original = path.read_bytes()
    step, seat = 37, index % 2
    rows = [json.loads(line) for line in gzip.decompress(original).decode().splitlines()]
    row = next(row for row in rows if row.get('from_step') == step)
    before = row['actions'][seat]['farmer']
    row['actions'][seat]['farmer'] = ['INDEPENDENT_VERIFIER_MUTATION']
    try:
        path.write_bytes(gzip.compress(('\n'.join(json.dumps(row, separators=(',', ':')) for row in rows) + '\n').encode(), mtime=0))
        result = run(f'oracle-{index:02}', parity)
        expected = f'step {step} seat {seat}'
        assert result.returncode != 0, entry['path']
        assert entry['path'] in result.stdout and expected in result.stdout and 'INDEPENDENT_VERIFIER_MUTATION' in result.stdout
        assert 'could not compile' not in result.stdout
        results.append({'kind':'oracle-action', 'path':entry['path'], 'step':step, 'seat':seat,
                        'before':before, 'after':row['actions'][seat]['farmer'], 'exit_code':result.returncode})
    finally:
        path.write_bytes(original)
        assert path.read_bytes() == original
    results[-1]['restored_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()

replay_test = ['--test', 'oracle_parity', 'original_python_mid_episode_replays_match_native_resumed_actions', '--', '--nocapture']
path = crate / 'fixtures/replay/REPLAY.json.gz'
original = path.read_bytes()
document = json.loads(gzip.decompress(original))
for index, case in enumerate(document['cases']):
    case['resumed_actions'][5][index % 2]['farmer'] = ['INDEPENDENT_REPLAY_MUTATION']
try:
    path.write_bytes(gzip.compress((json.dumps(document,separators=(',',':'))+'\n').encode(),mtime=0))
    result = run('replay-all-24-cases', replay_test)
    assert result.returncode != 0 and 'could not compile' not in result.stdout
    for index, case in enumerate(document['cases']):
        label = f"{case['source']}@{case['reconstruct_step']} resume step {case['reconstruct_step']+5} seat {index%2}"
        assert label in result.stdout, label
        results.append({'kind':'replay-action','case':index,'source':case['source'],'point':case['reconstruct_step'],'seat':index%2,'exit_code':result.returncode})
finally:
    path.write_bytes(original)
    assert path.read_bytes() == original
print('Replay fixture: all 24 altered cases detected; restored byte-exact',flush=True)

changes = {
    'starter': ('Ok(json!({"farmer": rule.verb, "hands": [], "market": rule.orders}))',
                'Ok(json!({"farmer": if step == 700 { json!(["INDEPENDENT_VERIFIER_MUTATION"]) } else { rule.verb }, "hands": [], "market": rule.orders}))'),
    'r04': ('Ok(json!({"farmer": farmer, "hands": hands, "market": orders}))',
            'Ok(json!({"farmer": if world.step == 700 { json!(["INDEPENDENT_VERIFIER_MUTATION"]) } else { farmer }, "hands": hands, "market": orders}))'),
    'ecobot': ('Ok(json!({"farmer": farmer, "hands": hands, "market": decision.market_orders}))',
               'Ok(json!({"farmer": if game.step_index() == 700 { json!(["INDEPENDENT_VERIFIER_MUTATION"]) } else { Value::Array(farmer) }, "hands": hands, "market": decision.market_orders}))'),
    'e776': ('Ok(action.into_value())',
             'let mut value = action.into_value();\n        if step == 700 { value["farmer"] = json!(["INDEPENDENT_VERIFIER_MUTATION"]); }\n        Ok(value)'),
}
for bot, (before, after) in changes.items():
    path = crate / 'src/native_agents' / f'{bot}.rs'
    original = path.read_bytes()
    text = original.decode()
    assert text.count(before) == 1, bot
    try:
        path.write_text(text.replace(before, after))
        result = run(f'controller-{bot}', replay_test)
        assert result.returncode != 0, bot
        assert 'resume step 700 seat' in result.stdout and f'{bot} action.farmer' in result.stdout
        assert 'could not compile' not in result.stdout
        results.append({'kind':'controller-action', 'path':str(path.relative_to(scratch)),
                        'step':700, 'before':before, 'after':after, 'exit_code':result.returncode})
    finally:
        path.write_bytes(original)
        assert path.read_bytes() == original
    results[-1]['restored_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()

restored = {str(p.relative_to(scratch)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in scratch.rglob('*') if p.is_file()}
assert restored == inventory
final = run('restored-full-suite', [])
assert final.returncode == 0
(root / 'mutations-results.json').write_text(json.dumps({'results':results,
    'restored_all_source_and_fixture_files':True, 'inventory':inventory,
    'baseline_exit':baseline.returncode, 'final_exit':final.returncode}, indent=2)+'\n')
print(f'DONE: {len(results)} mutations rejected, all {len(inventory)} files restored byte-exact; full suite green', flush=True)
