"""BC best vs Cha22 through run_ppo._evaluate_against_bot, on CPU (observe only).

Same function as the trainer's eval/*_vs_bot (sampled actions, both seats by
env parity), with env.n_envs overridden to N (default 16) and env_steps=0 for
the evaluation worlds. CPU, float32 (rl.dtype override), 4 torch threads, no
compile. Usage: bc_vs_cha22_eval.py [N]
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
spec = importlib.util.spec_from_file_location("run_ppo", "scripts/run_ppo.py")
rp = importlib.util.module_from_spec(spec)
sys.modules["run_ppo"] = rp
spec.loader.exec_module(rp)
from owl.train.config import FullConfig  # noqa: E402

cfg = FullConfig.from_file(Path("configs/kaggriculture_4rank_vs_cha22.yaml"))
cfg = cfg.model_copy(
    update={
        "env": cfg.env.model_copy(update={"n_envs": N}),
        "rl": cfg.rl.model_copy(update={"dtype": "float32"}),
    }
)
dev = torch.device("cpu")
model = rp._create_eval_model_for_config(cfg, device=dev, roundtrip_lora_base=False)
ckpt = torch.load("/root/bc-best/checkpoint_bc_best.pt", map_location=dev, weights_only=False)
rp.load_model_state_dict_allowing_lora(model, ckpt["model"])
model.eval()
t = time.time()
m = rp._evaluate_against_bot(current_model=model, cfg=cfg, device=dev, env_steps=0)
m["wall_s"] = time.time() - t
m["n_envs"] = N
print(json.dumps(m, indent=1, sort_keys=True))
