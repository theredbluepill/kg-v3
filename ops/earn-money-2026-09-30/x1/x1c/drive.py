"""X1c driver: one checkpoint in mirror self-play on CPU, N envs (one game each).

Actions sampled exactly as run_ppo's evaluation (model(obs, deterministic=False)),
float32, CPU. Dumps per game: seed, game config JSON, per-step joint action JSON
(decoded with owl.kaggriculture.codec.decode_actions), terminal banks.
Usage: drive.py CKPT LABEL BASE_SEED N OUT.jsonl
"""
import importlib.util, json, sys, time
from pathlib import Path
import torch

torch.set_num_threads(4)
ckpt_path, label, base_seed, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.manual_seed(base_seed)
spec = importlib.util.spec_from_file_location("run_ppo", "scripts/run_ppo.py")
rp = importlib.util.module_from_spec(spec); sys.modules["run_ppo"] = rp; spec.loader.exec_module(rp)
from owl.train.config import FullConfig
from owl.game import create_env
from owl.kaggriculture.codec import decode_actions, encode_actions
import os
FIX = os.environ.get('X1C_FIX') == '1'
fixed_cmds = [0]
from owl.kaggriculture.types import KaggricultureActions

cfg = FullConfig.from_file(Path("configs/kaggriculture_4rank_margin.yaml"))
cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"n_envs": N, "opponent_mix": None, "native_threads": 1}),
                             "rl": cfg.rl.model_copy(update={"dtype": "float32"})})
dev = torch.device("cpu")
model = rp._create_eval_model_for_config(cfg, device=dev, roundtrip_lora_base=False)
ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
rp.load_model_state_dict_allowing_lora(model, ck["model"]); model.eval()
env = create_env(cfg.env, n_envs=N, base_seed=base_seed, rank=0, world_size=1, pin_memory=False, transfer_device=dev)
obs = env.reset()
_, seeds = env.seed_state()
gconf = json.loads(cfg.env.config.to_native_json()) if isinstance(cfg.env.config.to_native_json(), str) else cfg.env.config.to_native_json()
steps = [[] for _ in range(N)]; banks = [None] * N; finished = [False] * N
t0 = time.time()
with torch.no_grad():
    while not all(finished):
        o = model(obs, deterministic=False)
        acts = KaggricultureActions(tokens=o.actions.tokens.cpu().contiguous(), lengths=o.actions.lengths.cpu().contiguous())
        decoded = decode_actions(acts, obs, action_spec=cfg.env.action_spec)
        if FIX:
            # counterfactual engine-rule repair: keep the first s PLANT X commands (farmer, then hands) when the
            # seat submits more PLANT X than it holds X seeds (s > 0); the excess becomes PASS instead of all.
            changed = False
            new = []
            for i in range(N):
                pair = [json.loads(json.dumps(decoded[i][0])), json.loads(json.dumps(decoded[i][1]))]
                if not finished[i]:
                    snap = env.state_snapshot(i)
                    for p in (0, 1):
                        prog = pair[p]
                        units = [prog.get("farmer")] + list(prog.get("hands") or [])
                        seedc = snap["privates"][p]["seeds"]
                        dem = {}
                        for u, c in enumerate(units):
                            if isinstance(c, list) and c and c[0] == "PLANT" and len(c) > 1:
                                dem.setdefault(c[1], []).append(u)
                        for crop, us in dem.items():
                            sd = int(seedc.get(crop, 0))
                            if 0 < sd < len(us):
                                for u in us[sd:]:
                                    if u == 0: prog["farmer"] = ["PASS"]
                                    else: prog["hands"][u - 1] = ["PASS"]
                                    fixed_cmds[0] += 1; changed = True
                new.append((pair[0], pair[1]))
            if changed:
                acts = encode_actions(new, obs, action_spec=cfg.env.action_spec)
                decoded = new
        for i in range(N):
            if not finished[i]:
                steps[i].append([decoded[i][0], decoded[i][1]])
        obs, _r, dones, _m = env.step(acts)
        for i in torch.nonzero(dones.all(dim=1)).flatten().tolist():
            if not finished[i]:
                tm = env.terminal_metrics(i)
                banks[i] = [float(tm["bank_0"]), float(tm["bank_1"])]; finished[i] = True
with open(out, "w") as f:
    for i in range(N):
        f.write(json.dumps({"label": label, "ckpt": ckpt_path, "env": i, "seed": str(seeds[i]), "config": gconf,
                            "banks": banks[i], "steps": steps[i]}) + "\n")
print(json.dumps({"label": label, "wall_s": round(time.time() - t0, 1), "banks": banks, "seeds": [str(s) for s in seeds],
                  "lens": [len(s) for s in steps], "fix": FIX, "fixed_cmds": fixed_cmds[0]}))
