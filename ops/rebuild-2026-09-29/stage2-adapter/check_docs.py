"""Assert the Stage 2 documented boundary, without claiming pending runtime work."""
from pathlib import Path
root = Path(__file__).resolve().parents[3]
api = (root / 'docs/rl-api-specs.md').read_text()
assert '**Stage 2 status: real native binding wired.**' in api
assert 'Stage 1 status: native binding pending' not in api
assert 'and pending Task 1.4 exports' not in api
assert 'Task 3.1 rollout storage and action mapping' in api
assert 'pod DMA' in api and '2-rank smoke' in api
readme = (root / 'README.md').read_text()
assert 'the native evaluation env that consumes it is pending rebuild Tasks 1.4/1.5' not in readme
print('Stage 2 API/README status assertions pass')
