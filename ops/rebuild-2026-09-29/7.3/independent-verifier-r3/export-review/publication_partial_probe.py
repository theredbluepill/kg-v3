from pathlib import Path
from unittest.mock import patch
import hashlib
import json
from tests.kaggriculture.test_replay_export_integration import _run

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'publication-partial-output'
watched = Path('python/owl/kaggriculture/replay_export.py')
before = hashlib.sha256(watched.read_bytes()).hexdigest()
original = Path.open

class PartialWriter:
    def __init__(self, stream):
        self.stream = stream
    def __enter__(self):
        self.stream.__enter__()
        return self
    def __exit__(self, *args):
        return self.stream.__exit__(*args)
    def write(self, value):
        self.stream.write(value[:20])
        self.stream.flush()
        raise OSError('injected partial episode write failure')

def broken_open(path, mode='r', *args, **kwargs):
    stream = original(path, mode, *args, **kwargs)
    if path == OUT / 'game_000000.json' and mode == 'xb':
        return PartialWriter(stream)
    return stream

try:
    with patch.object(Path, 'open', broken_open):
        _run(OUT, n_envs=2, n_games=2, count=2, configuration={'episodeSteps': 3})
except OSError as error:
    result = {'exception': str(error), 'notes': error.__notes__}
else:
    raise AssertionError('expected injected error')
sidecar = json.loads((OUT / 'game_000000.custody.json').read_text())
other = json.loads((OUT / 'game_000001.custody.json').read_text())
payload = (OUT / 'game_000000.json').read_bytes()
try:
    json.loads(payload)
except json.JSONDecodeError as error:
    parse_error = str(error)
else:
    raise AssertionError('expected invalid JSON')
result.update(sidecar_status=sidecar['status'], sidecar_complete=sidecar['complete'],
              verification=sidecar.get('verification'), episode_exists=True,
              episode_bytes=len(payload), episode_text=payload.decode(), parse_error=parse_error,
              recorded_episode_sha256=sidecar['episode_sha256'], actual_episode_sha256=hashlib.sha256(payload).hexdigest(),
              other_custody=other['status'], checksum_before=before,
              checksum_after=hashlib.sha256(watched.read_bytes()).hexdigest())
assert result['checksum_before'] == result['checksum_after']
assert result['sidecar_status'] == 'complete' and result['other_custody'] == 'error'
assert result['recorded_episode_sha256'] != result['actual_episode_sha256']
(ROOT / 'publication-partial-result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
