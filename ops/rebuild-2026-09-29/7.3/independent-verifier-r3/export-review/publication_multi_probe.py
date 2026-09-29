from pathlib import Path
from unittest.mock import patch
import json
import hashlib
from tests.kaggriculture.test_replay_export_integration import _run

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'publication-multi-output'
watched = Path('python/owl/kaggriculture/replay_export.py')
before = hashlib.sha256(watched.read_bytes()).hexdigest()
original = Path.open

def broken_open(path, mode='r', *args, **kwargs):
    if path == OUT / 'game_000000.json' and mode == 'xb':
        raise PermissionError('injected episode publish failure')
    return original(path, mode, *args, **kwargs)

try:
    with patch.object(Path, 'open', broken_open):
        _run(OUT, n_envs=2, n_games=2, count=2, configuration={'episodeSteps': 3})
except PermissionError as error:
    result = {'exception': str(error), 'notes': error.__notes__}
else:
    raise AssertionError('expected injected error')
sidecar = json.loads((OUT / 'game_000000.custody.json').read_text())
result.update(sidecar_status=sidecar['status'], sidecar_complete=sidecar['complete'],
              verification=sidecar.get('verification'), episode_exists=(OUT / 'game_000000.json').exists(),
              checksum_before=before, checksum_after=hashlib.sha256(watched.read_bytes()).hexdigest())
assert result['checksum_before'] == result['checksum_after']
assert result['sidecar_status'] == 'complete' and not result['episode_exists']
result['other_custody'] = json.loads((OUT / 'game_000001.custody.json').read_text())['status']
assert result['other_custody'] == 'error'
(ROOT / 'publication-multi-result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
