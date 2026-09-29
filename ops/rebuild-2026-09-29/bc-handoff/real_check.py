import json, importlib.util, sys
from pathlib import Path
import torch
from owl.kaggriculture.bc_data import BCManifest, _load_split
from owl.train import FullConfig
from owl.train.optimizer import create_optimizer
from owl.train.ppo import PPOTrainer
REPO=Path.cwd()
spec=importlib.util.spec_from_file_location("run_ppo", REPO/"scripts/run_ppo.py"); run_ppo=importlib.util.module_from_spec(spec); sys.modules["run_ppo"]=run_ppo; spec.loader.exec_module(run_ppo)
torch.set_num_threads(2)
best=Path("/tmp/kg-v3-bc-best/checkpoint_bc_best.pt")
root=Path("/tmp/kg-v3-bc-best/shards-top1")
m=BCManifest.model_validate_json((root/"manifest.json").read_bytes())
one=m.model_copy(update={"episodes": tuple(e for e in m.episodes if e.episode_id=="112208626")})
split=_load_split(root,"validation",one,rank=0,world_size=1)
import numpy as np
batch=split.gather(np.arange(0, split.rank_rows[0], 45, dtype=np.int64))
print("rows", batch.obs.still_playing.shape)
outs={}
for cfgp in [sys.argv[1], *sys.argv[2:]]:
    cfg=FullConfig.from_file(Path(cfgp))
    cfg=cfg.model_copy(update={"model": cfg.model.model_copy(update={"force_flash_attn": False})})
    torch.manual_seed(0)
    model=run_ppo._create_model(cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec); model.reset_parameters()
    tr=PPOTrainer.__new__(PPOTrainer); tr.model=model; tr.optimizer=create_optimizer(model,cfg.optimizer); tr.device=torch.device("cpu")
    tr.player_step_total=tr.total_games_played=tr.total_active_entities=0
    md=tr.load_model_weights(best, load_optimizer=False)
    model.eval()
    with torch.no_grad():
        ev=model.evaluate_actions(batch.obs, batch.actions)
    nll=-(ev.log_probs.per_player_entity.float().sum(-1))/batch.actions.lengths.float()
    pol=batch.policy_seat
    print(cfgp, "meta", md)
    print(" values", ev.values.tolist())
    print(" winner_p", ev.winner_probabilities[...,0].tolist() if ev.winner_probabilities.ndim==3 else ev.winner_probabilities.tolist())
    print(" final_banks", batch.final_banks[0].tolist())
    print(" policy-seat turn NLL mean", float(nll[pol].mean()), "finite", all(bool(torch.isfinite(t).all()) for t in (ev.values, ev.winner_probabilities, ev.log_probs.per_player_entity)))
    outs[cfgp]=(ev.values.clone(), ev.log_probs.per_player_entity.clone())
ks=list(outs)
for k in ks[1:]:
    print(k, "equal to", ks[0], torch.equal(outs[k][0],outs[ks[0]][0]), torch.equal(outs[k][1],outs[ks[0]][1]))
