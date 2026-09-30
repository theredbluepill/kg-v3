"""Mac load test: packaged agent loads 08bc strictly with its run config and plays 5 turns (both seats, strict)."""
import hashlib, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, "scripts")
from extract_model_weights import extract_model_weights
from kaggle_environments import make
from owl.kaggriculture.kaggle_agent import KaggricultureAgent

ckpt = Path(sys.argv[1]); cfg = Path(sys.argv[2]); root = Path(sys.argv[3])
sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
assert sha == "08bc19aed8c0647002ea60e281f39b217f59b82deba54f45f53b839b2e574600", sha
if root.exists(): shutil.rmtree(root)
root.mkdir(parents=True)
extract_model_weights(ckpt, root / "checkpoint.pt")
shutil.copy2(cfg, root / "config.yaml")
agents = [KaggricultureAgent(root, deterministic=True, strict=True, min_overage_time=2.0) for _ in range(2)]
agents[0].warm_up()
env = make("kaggriculture", configuration={"seed": 20260930}, debug=True)
env.reset()
times = []
for turn in range(5):
    actions = []
    for seat in range(2):
        obs = json.loads(json.dumps({**env.state[0].observation, **env.state[seat].observation}))  # kaggle merges shared keys from seat 0
        t = time.perf_counter(); a = agents[seat].act(obs, dict(env.configuration)); times.append(time.perf_counter() - t)
        actions.append(a)
    env.step(actions)
    statuses = [s.status for s in env.state]
    assert all(s == "ACTIVE" for s in statuses), statuses
    print(f"turn {turn} statuses={statuses} action0={json.dumps(actions[0])[:120]}")
print(json.dumps({"checkpoint_sha256": sha, "slim_sha256": hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest(),
  "turns": 5, "calls": sum(a.calls for a in agents), "caught_errors": sum(a.caught_errors for a in agents),
  "budget_passes": sum(a.budget_passes for a in agents), "max_act_s": round(max(times), 4)}))
