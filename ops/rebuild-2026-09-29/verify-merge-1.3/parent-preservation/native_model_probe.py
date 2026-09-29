import gzip
import importlib.util
import json
from pathlib import Path
import torch
from owl import rs
from owl.kaggriculture import types as kt
from owl.model import KaggricultureTransformerConfig,create_model

torch.set_num_threads(1)
torch.manual_seed(101)
p=Path('tests/kaggriculture/test_observe.py')
spec=importlib.util.spec_from_file_location('observe_test_helpers',p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with gzip.open('tests/fixtures/kaggriculture/observation-v3/states.jsonl.gz','rt') as f:
 records=[json.loads(line) for line in f]
selected={0,len(records)-1,max(range(len(records)),key=lambda i:max(len(farm['hands']) for farm in records[i]['header']['initial']['public']['farms'])),max(range(len(records)),key=lambda i:records[i]['header']['initial']['public']['step'])}
config=KaggricultureTransformerConfig(embed_dim=16,depth=1,n_heads=2,n_scratch_tokens=1,force_flash_attn=False)
model=create_model(config,obs_spec=kt.KaggricultureObsConfig(),action_spec=kt.KaggricultureActionConfig()).eval()
results=[]
for i in sorted(selected):
 obs=m._allocate(1)
 m._write(json.dumps([records[i]['header']]),m._arrays(obs))
 obs.check_contract()
 with torch.inference_mode():
  encoded=model.encode_observations(obs)
  values=model.compute_value(obs)
  logs=model.winner_log_probabilities(obs)
 assert tuple(values.shape)==(1,2)
 assert tuple(logs.shape)==(1,2,2)
 assert torch.isfinite(encoded.hidden).all()
 assert torch.isfinite(values).all()
 assert torch.isfinite(logs).all()
 torch.testing.assert_close(logs.exp().sum(-1),torch.ones((1,2)))
 if (~obs.still_playing).any():
  torch.testing.assert_close(values[~obs.still_playing],torch.zeros_like(values[~obs.still_playing]))
 results.append({'record_id':records[i]['record_id'],'step':records[i]['header']['initial']['public']['step'],'actor_counts':obs.actor_mask[...,:kt.MAX_ACTORS].sum(-1).tolist(),'live':obs.still_playing.tolist(),'hidden_shape':list(encoded.hidden.shape),'value_shape':list(values.shape)})
print(json.dumps({'native_module':rs.__file__,'records':results,'passed':len(results)},indent=2))
