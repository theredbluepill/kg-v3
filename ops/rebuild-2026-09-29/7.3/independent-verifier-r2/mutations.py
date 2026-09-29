"""Independent scratch-file mutations; tracked sources and fixtures stay untouched."""
from __future__ import annotations
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from owl import rs
from tests.kaggriculture.test_replay_export_oracles import (
    FIXTURE_HASHES, FRAMEWORK_HASHES, SEED, _run_framework_game, _fixture_episode,
    _header_tape, _captured, _dump, _ordered, _semantic_payload,
)

ROOT = Path.cwd()
OUT = ROOT / 'ops/rebuild-2026-09-29/7.3/independent-verifier-r2'
START = time.monotonic()
results: list[dict[str, Any]] = []
baselines: list[dict[str, Any]] = []

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def baseline(name: str, text: str, evidence: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    report = json.loads(rs.verify_kaggriculture_episode(text, evidence))
    assert report['ok'], report
    entry = {'label': name, 'mode': report['mode'], 'transitions': report['transitions'],
             'captured': report['captured'], 'seconds': time.monotonic() - started}
    baselines.append(entry)
    print('BASELINE ' + json.dumps(entry), flush=True)
    return report

def mutation(name: str, original: str, mutated: str, expected: str, *, episode: str | None = None,
             error_prefix: str | None = None) -> None:
    """Mutate one real scratch file, check rejection, then restore exact saved bytes."""
    raw, altered = original.encode(), mutated.encode()
    assert raw != altered, name
    with tempfile.TemporaryDirectory(prefix='replay-mutation-', dir=ROOT / '.codex-tmp') as directory:
        path = Path(directory) / 'oracle-input.json'
        path.write_bytes(raw)
        saved = path.read_bytes()
        started = time.monotonic()
        error = None
        try:
            path.write_bytes(altered)
            assert sha(path.read_bytes()) != sha(saved)
            try:
                if episode is None:
                    rs.verify_kaggriculture_episode(path.read_text())
                else:
                    rs.verify_kaggriculture_episode(episode, path.read_text())
            except ValueError as exc:
                error = str(exc)
            assert error is not None, f'{name}: mutation unexpectedly survived'
            assert expected in error, (name, expected, error)
            assert 'transition' in error, (name, error)
            if error_prefix:
                assert error.startswith(error_prefix), (name, error)
        finally:
            path.write_bytes(saved)
            restored = path.read_bytes()
            assert restored == raw, name
        entry = {'label': name, 'expected_pointer': expected, 'error': error,
                 'seconds': time.monotonic() - started, 'original_sha256': sha(raw),
                 'mutated_sha256': sha(altered), 'restored_sha256': sha(restored),
                 'restored_byte_identical': restored == raw}
        results.append(entry)
        print('MUTATION ' + json.dumps(entry), flush=True)

runtime = _run_framework_game()
framework = runtime['episode']
assert framework['info']['seed'] == SEED
header, tape = _header_tape(framework)
native = rs.export_kaggriculture_episode(_dump(header), _dump(tape))
report = baseline('pinned live framework', _dump(framework), _dump(_captured(framework)))
assert _ordered(_semantic_payload(json.loads(report['canonical_json']))) == _ordered(_semantic_payload(framework))
assert _ordered(_semantic_payload(json.loads(native))) == _ordered(_semantic_payload(framework))
report = baseline('native canonical byte round trip', native)
assert report['canonical_json'].encode() == native.encode()
assert json.loads(native)['info']['seed'] == SEED

altered = copy.deepcopy(framework)
altered['info']['seed'] += 1
mutation('framework resolved seed incremented', _dump(framework), _dump(altered), '/steps/3/0/observation/farms/0/tiles/0/1')
altered = copy.deepcopy(framework)
assert type(altered['steps'][1][0]['reward']) is int
altered['steps'][1][0]['reward'] = 0.0
mutation('framework ACTIVE reward integer to float', _dump(framework), _dump(altered), '/steps/1/0/reward')
# Preserve numeric value and JSON float kind, so only the canonical-byte guard differs.
assert '3000.0' in native
altered_text = native.replace('3000.0', '3.0e3', 1)
assert json.loads(native) == json.loads(altered_text)
mutation('native float equivalent exponent bytes', native, altered_text,
         '/steps/0/0/observation/farms/0/money', error_prefix='canonical number bytes differ')
# The number-kind test separately establishes the semantic comparison.
altered = json.loads(native)
for seat in altered['steps'][0]:
    seat['observation']['farms'][0]['money'] = 3000
mutation('native float money to integer', native, _dump(altered), '/steps/0/0/observation/farms/0/money')

evidence = _captured(framework)
baseline('captured evidence against independent framework', native, _dump(evidence))
altered = copy.deepcopy(evidence)
altered['banks'][1][0] += 1
mutation('captured transition bank incremented', _dump(evidence), _dump(altered), '/captured/banks/1/0', episode=native)

shared_omitted = copy.deepcopy(framework)
shared_keys = [key for key, field in shared_omitted['specification']['observation'].items() if field.get('shared')]
for step in shared_omitted['steps']:
    for key in shared_keys:
        step[1]['observation'].pop(key, None)
baseline('specification shared fields restored', _dump(shared_omitted))
altered = copy.deepcopy(shared_omitted)
altered['steps'][1][1]['observation']['private'] = copy.deepcopy(altered['steps'][1][0]['observation']['private'])
mutation('seat 0 private state leaked into seat 1', _dump(shared_omitted), _dump(altered), '/steps/1/1/observation/private/shed/WHEAT')
altered = copy.deepcopy(framework)
seeds = altered['steps'][0][0]['observation']['private']['seeds']
assert len(seeds) > 1
altered['steps'][0][0]['observation']['private']['seeds'] = dict(reversed(list(seeds.items())))
mutation('initial private seed key order reversed', _dump(framework), _dump(altered), '/steps/0/0/observation/private/seeds')

for fixture in FIXTURE_HASHES:
    episode = _fixture_episode(fixture, framework)
    original = _dump(episode)
    report = baseline(f'official fixture {fixture}', original, _dump(_captured(episode)))
    assert report['transitions'] == 719
    assert report['captured'] == {'initial': True, 'terminal': True, 'banks': 719, 'snapshots': 719}
    assert _ordered(_semantic_payload(json.loads(report['canonical_json']))) == _ordered(_semantic_payload(episode))
    altered = copy.deepcopy(episode)
    altered['steps'][-1][0]['reward'] = 0.8
    altered['rewards'][0] = 0.8
    for seat in altered['steps'][-1]:
        seat['observation']['farms'][0]['money'] = 0.8
    mutation(f'official fixture {fixture} raw reward to shaped reward', original, _dump(altered), '/steps/719/0/reward')
    if fixture == '95324500':
        altered = copy.deepcopy(episode)
        inventory = altered['steps'][22][1]['observation']['private']['inventories'][3]
        assert list(inventory) == ['COW', 'WHEAT']
        altered['steps'][22][1]['observation']['private']['inventories'][3] = dict(reversed(list(inventory.items())))
        mutation('official fixture actor inventory order reversed', original, _dump(altered), '/steps/22/1/observation/private/inventories/3')

receipt = {'target': 'For each new replay oracle, a valid independent input passes and one materially altered scratch input fails at a localizable pointer; all altered inputs restored byte-for-byte. No tracked source mutation.',
           'stopping_condition': 'All 8 baselines and 12 mutation controls complete, each restoration hash equals baseline hash.',
           'framework_hashes': FRAMEWORK_HASHES, 'fixture_hashes': FIXTURE_HASHES,
           'framework_runtime_seconds': runtime['seconds'], 'framework_peak_rss_bytes': runtime['peak_rss_bytes'],
           'seed': SEED, 'baselines': baselines, 'mutations': results,
           'baseline_count': len(baselines), 'rejected_mutations': len(results),
           'seconds': time.monotonic() - START}
assert len(baselines) == 8
assert len(results) == 12
(OUT / 'mutations-results.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('COMPLETE ' + json.dumps({'baseline_count': len(baselines), 'rejected_mutations': len(results), 'seconds': receipt['seconds']}), flush=True)
