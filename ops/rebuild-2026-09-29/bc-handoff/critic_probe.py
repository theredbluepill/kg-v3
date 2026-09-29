import sys, importlib.util
from pathlib import Path
import numpy as np, torch
from owl.kaggriculture.bc_data import BCManifest, _load_split
from owl.train import FullConfig
from owl.model import create_model
torch.set_num_threads(2)
best=Path("/tmp/kg-v3-bc-best/checkpoint_bc_best.pt"); root=Path("/tmp/kg-v3-bc-best/shards-top1")
m=BCManifest.model_validate_json((root/"manifest.json").read_bytes())
one=m.model_copy(update={"episodes": tuple(e for e in m.episodes if e.episode_id=="112208626")})
split=_load_split(root,"validation",one,rank=0,world_size=1)
batch=split.gather(np.arange(split.rank_rows[0], dtype=np.int64))
cfg=FullConfig.from_file(Path("configs/kaggriculture_2rank.yaml"))
cfg=cfg.model_copy(update={"model": cfg.model.model_copy(update={"force_flash_attn": False})})
ck=torch.load(best, map_location="cpu", weights_only=False)["model"]
def build(fresh_head):
    torch.manual_seed(0)
    mdl=create_model(cfg.model, obs_spec=cfg.env.obs_spec, action_spec=cfg.env.action_spec); mdl.reset_parameters()
    keep={k:v.clone() for k,v in mdl.state_dict().items() if k.startswith("critic_head.")}
    mdl.load_state_dict(ck)
    if fresh_head: mdl.load_state_dict(keep, strict=False)
    return mdl.eval()
turn=split.turn.numpy()
for fresh in (False, True):
    mdl=build(fresh)
    with torch.no_grad():
        ev=mdl.evaluate_actions(batch.obs, batch.actions)
    p=ev.winner_probabilities[:,1].float() if ev.winner_probabilities.ndim==2 else ev.winner_probabilities[:,1,0].float()
    v=ev.values.float()
    sat=(v.abs()>1-2e-6).float().mean().item()
    print("fresh_head" if fresh else "bc_head", "winner_probabilities shape", tuple(ev.winner_probabilities.shape), "values shape", tuple(v.shape))
    print("  frac |value|>1-2e-6:", round(sat,4), " |value| mean", round(v.abs().mean().item(),4))
    for q in (0,10,50,100,200,359):
        print("   row",q,"turn",int(turn[q]),"values",[round(x,6) for x in v[q].tolist()])
    # MSE gradient w.r.t. critic head if the true outcome were the opposite
    mdl.train(); mdl.zero_grad()
    rows=np.arange(0,360,12,dtype=np.int64); b=split.gather(rows)
    ev=mdl.evaluate_actions(b.obs,b.actions)
    target=-torch.sign(ev.values.detach()); target[target==0]=1.0
    loss=((ev.values.float()-target)**2).mean(); loss.backward()
    g=torch.sqrt(sum((q.grad**2).sum() for n,q in mdl.named_parameters() if n.startswith("critic_head.") and q.grad is not None))
    print("  MSE loss vs flipped outcome", round(loss.item(),4), "critic_head grad norm", float(g))
    # actor unchanged check
    mdl.eval()
    with torch.no_grad(): lp=mdl.evaluate_actions(batch.obs,batch.actions).log_probs.per_player_entity
    if not fresh: lp0=lp
    else: print("  actor log-probs identical to bc_head model:", torch.equal(lp,lp0))
