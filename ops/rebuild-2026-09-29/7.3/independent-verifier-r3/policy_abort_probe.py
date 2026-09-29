import json
from pathlib import Path
from tests.kaggriculture.test_replay_export_integration import _recorder, market_policy, HASHES, VERSIONS, HIRE_LIMIT
from tests.kaggriculture.test_native_env import REWARD, buffers
from owl import rs
from owl.kaggriculture.native_evaluation import evaluate_native_games

out = Path(__file__).resolve().parent / 'policy-abort'
recorder = _recorder(out, 2, 2, [0, 1])
env = rs.KaggricultureEnv(2, 91, 3, '{"episodeSteps":4}', REWARD, 1, hire_limit=HIRE_LIMIT)
arrays = buffers(2)
original_error = RuntimeError('independent verifier policy failure')
calls = 0

def policy(arrays, seats):
    global calls
    calls += 1
    if calls == 2:
        raise original_error
    return market_policy(arrays, seats)

try:
    evaluate_native_games(env, arrays, policy, n_games=2, seat_assignments=[0, 1], configuration={'episodeSteps':4}, hire_limit=HIRE_LIMIT, recorder=recorder, checkpoint_hashes=HASHES, versions=VERSIONS)
except RuntimeError as error:
    assert error is original_error
else:
    raise AssertionError('policy failure was swallowed')
records = [json.loads(p.read_text()) for p in sorted(out.glob('*.custody.json'))]
assert len(records) == 2
assert not recorder.active_games
assert all(r['status'] == 'error' and r['complete'] is False for r in records)
assert all(len(r['action_tape']['transitions']) == 1 for r in records)
assert not list(out.glob('game_??????.json'))
result = {'records':len(records), 'statuses':[r['status'] for r in records], 'committed_transitions':[len(r['action_tape']['transitions']) for r in records], 'active_games':sorted(recorder.active_games), 'original_exception_identity_preserved':True, 'published_episodes':0}
(out.parent / 'policy-abort-result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
