"""Load the Task 3.5 CLI run's final checkpoint through run_ppo's own loader."""

import importlib.util
import sys
from pathlib import Path

import torch
from owl.train import FullConfig

run_dir = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("run_ppo", "scripts/run_ppo.py")
assert spec is not None
assert spec.loader is not None
run_ppo = importlib.util.module_from_spec(spec)
sys.modules["run_ppo"] = run_ppo
spec.loader.exec_module(run_ppo)
cfg = FullConfig.from_file(run_dir / "config.yaml")
model = run_ppo._create_eval_model_for_config(
    cfg, device=torch.device("cpu"), roundtrip_lora_base=False
)
metadata = run_ppo._load_model_from_checkpoint(
    model, path=run_dir / "checkpoint_final.pt", device=torch.device("cpu")
)
finite = all(torch.isfinite(v).all() for v in model.state_dict().values())
params = sum(p.numel() for p in model.parameters())
print(f"env_steps={metadata.env_steps} player_step_total={metadata.player_step_total}")
print(f"total_games_played={metadata.total_games_played}")
print(f"wandb_run_id={metadata.wandb_run_id}")
print(f"parameters={params} all_finite={finite}")
print(f"n_envs={cfg.env.n_envs} horizon={cfg.rl.horizon} model={cfg.model.model_arch}")
print(f"eval_replay_games={cfg.rl.eval_replay_games}")
print(f"teacher_mode={cfg.rl.teacher_mode}")
assert metadata.env_steps == 64
assert finite
