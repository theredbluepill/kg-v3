import json, time, statistics
from pathlib import Path
import numpy as np
import torch
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.actor_codec import SLOT_NAMES, encode_action
from owl.kaggriculture.types import KaggricultureActions
source=Path('python/owl/kaggriculture/env.py').read_text()
source=source.replace('self._actor_indices < counts[..., None]', 'torch.arange(MAX_ACTORS)[None, None, :] < counts[..., None]').replace('self._frame_indices < envelope[..., None]', 'torch.arange(MAX_FRAMES)[None, None, :] < envelope[..., None]')
source=source.replace('np.copyto(self._frames, actions.tokens.numpy(), casting="unsafe")', 'self._frames = np.ascontiguousarray(actions.tokens.numpy(), dtype=np.int16)').replace('np.copyto(self._lengths, actions.lengths.numpy(), casting="unsafe")', 'self._lengths = np.ascontiguousarray(actions.lengths.numpy(), dtype=np.int32)').replace('self._batch.econ(self._econ_array)', 'self._batch.econ(self._econ.numpy())')
source=source.replace('if self.reward_shaping.enabled:', 'if True:')
namespace={'__name__':'baseline_env'}
exec(compile(source,'baseline_env.py','exec'),namespace)
Baseline=namespace['KaggricultureVectorizedEnv']
torch.set_num_threads(1)
result={'method':'Interleaved same-process baseline semantics versus optimized Python env; native wheel identical; order alternates each batch','n_envs':8,'threads':1,'repetitions':5,'steps_per_repetition':800,'includes_terminal_autoreset':True,'results':[]}
for work in ('pass','move_and_market'):
 action={'farmer':['PASS' if work=='pass' else 'EAST'],'hands':[],'market':[] if work=='pass' else [['BUY_SEED','WHEAT',1],['SELL','WHEAT',1]]}
 rows=encode_action(action,hire_limit=241)['frames']
 tokens=torch.zeros((8,2,252,12),dtype=torch.int64)
 tokens[:,:,:len(rows)]=torch.tensor([[row[s] for s in SLOT_NAMES] for row in rows])
 a=KaggricultureActions(tokens,torch.full((8,2),len(rows),dtype=torch.int64))
 with Baseline(n_envs=8,pin_memory=False,seed=2301) as old, KaggricultureVectorizedEnv(n_envs=8,pin_memory=False,seed=2301) as new:
  times=[[],[]]; checks=0
  for rep in range(5):
   old.reset();new.reset()
   for i in range(800):
    for k in ((0,1) if i%2 else (1,0)):
     env=(old,new)[k]
     start=time.perf_counter_ns();env.step(a);times[k].append((time.perf_counter_ns()-start)/1e6)
    if i%100==0 or old.dones.any():
     assert torch.equal(old.observations.features,new.observations.features)
     assert torch.equal(old.rewards,new.rewards)
     assert torch.equal(old.dones,new.dones)
     checks+=1
  result['results'].append({'workload':work,'equal_output_checkpoints':checks,'baseline_ms_median':float(np.median(times[0])),'optimized_ms_median':float(np.median(times[1])),'baseline_game_steps_per_sec':8000/statistics.mean(times[0]),'optimized_game_steps_per_sec':8000/statistics.mean(times[1]),'baseline_total_ms':sum(times[0]),'optimized_total_ms':sum(times[1])})
print(json.dumps(result,indent=2))
