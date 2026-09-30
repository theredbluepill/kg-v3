"""X1d: checkpoint self-play on CPU with per-seat econ counters and daily farm composition.

Usage (from /root/kg-v3-anchor): x1d_selfplay.py LABEL CKPT [N_ENVS=8] [THREADS=4]
Same eval worlds for every checkpoint: configs/kaggriculture_4rank_vs_cha22.yaml env.seed,
run_ppo._evaluation_seed(env_steps=0), n_envs=N, no opponent mix (two-model self-play,
both seats the same checkpoint). Sampled actions; torch.manual_seed(20260930).
"""
import collections, importlib.util, json, sys, time
from pathlib import Path
import torch
LABEL, CKPT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
torch.manual_seed(20260930)
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("run_ppo", "scripts/run_ppo.py")
rp = importlib.util.module_from_spec(spec); sys.modules["run_ppo"] = rp; spec.loader.exec_module(rp)
from owl.train.config import FullConfig
from owl.kaggriculture.env import KaggricultureActions
E = ['starv','drought','ineff','cmds','pass','harvest','water','feed','sell_units','sell_cash','expiry_units','overflow','care_lost','fert_wasted','weeds','unsold_end','death_held','clipped','missed_growth','dug','redundant_fert','care_wasted','term_flags','seeds_unused','unused_land_cash','hire_wasted_cash','idle_hand_steps','mkt_unfilled','malformed','sale_shortfall_cash','floor_sale_units','buy_premium_cash']
cfg = FullConfig.from_file(Path("configs/kaggriculture_4rank_vs_cha22.yaml"))
cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"n_envs": N}), "rl": cfg.rl.model_copy(update={"dtype": "float32"})})
dev = torch.device("cpu")
model = rp._create_eval_model_for_config(cfg, device=dev, roundtrip_lora_base=False)
ckpt = torch.load(CKPT, map_location=dev, weights_only=False)
rp.load_model_state_dict_allowing_lora(model, ckpt["model"]); model.eval()
env = rp._create_eval_env(cfg, n_envs=N, device=dev, env_steps=0, opponent_mix=None)
try:
    _ss = env.seed_state(); seed_state = [_ss[0], list(_ss[1])]
except Exception as ex:
    seed_state = f'unavailable: {ex!r}'
obs = env.reset(); finished = [False]*N; term = {}
tile_days = {i: [collections.Counter(), collections.Counter()] for i in range(N)}
money = {i: [[], []] for i in range(N)}; quads = {i: [[], []] for i in range(N)}; hands = {i: [[], []] for i in range(N)}
def comp(farm):
    c = collections.Counter()
    for r in farm['tiles']:
        for x in r:
            if isinstance(x, dict): c[x.get('animal') or x.get('crop') or x['kind']] += 1
            elif x is None: c['empty'] += 1
    return c
t = time.time(); step = 0
with torch.no_grad():
    while not all(finished):
        learner = env.learner_mask.clone()
        acts = rp.forward_learner_rows(model, rp._obs_to_device(obs, dev, non_blocking=False), learner).actions
        if step % 24 == 23 or step == 0:
            for i in range(N):
                if finished[i]: continue
                s = env.state_snapshot(i)
                for p in (0, 1):
                    f = s['public']['farms'][p]
                    tile_days[i][p].update(comp(f)); money[i][p].append(round(f['money']))
                    quads[i][p].append(len(f['unlocked_quadrants'])); hands[i][p].append(len(f['hands']))
        obs, _r, dones, _m = env.step(KaggricultureActions(tokens=acts.tokens.cpu().contiguous(), lengths=acts.lengths.cpu().contiguous()))
        step += 1
        for i in torch.nonzero(dones.all(dim=1)).flatten().tolist():
            if not finished[i]:
                tm = env.terminal_metrics(i); term[i] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in tm.items()}; finished[i] = True
seats = []
for i in range(N):
    for p in (0, 1):
        e = dict(zip(E, term[i][f'econ_{p}'])); assert len(term[i][f'econ_{p}']) == 32
        e.update(env=i, seat=p, bank=term[i][f'bank_{p}'], winner=term[i]['winner'], episode_steps=term[i]['episode_steps'],
                 tile_days=dict(tile_days[i][p]), money_by_day=money[i][p], quads_by_day=quads[i][p], hands_by_day=hands[i][p])
        seats.append(e)
out = {"label": LABEL, "ckpt": CKPT, "n_envs": N, "seed_state": seed_state, "eval_base_seed": rp._evaluation_seed(base_seed=cfg.env.seed, env_steps=0), "torch_seed": 20260930,
       "wall_s": time.time() - t, "steps": step, "seats": seats}
Path(f"/root/x1/games_{LABEL}.json").write_text(json.dumps(out))
print("done", LABEL, round(time.time() - t), "s", [round(s['bank']) for s in seats])
