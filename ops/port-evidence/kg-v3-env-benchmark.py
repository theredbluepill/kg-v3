import json, os, platform, statistics, sys, time
import numpy as np
import torch
from owl.kaggriculture.env import KaggricultureVectorizedEnv
from owl.kaggriculture.actor_codec import SLOT_NAMES, encode_action
from owl.kaggriculture.types import KaggricultureActions

torch.set_num_threads(1)
N=8
REPEATS=5
STEPS=400
out={'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'n_envs':N,'repetitions':REPEATS,'steps_per_repetition':STEPS,'unit':'one transition of one two-seat game','results':[]}
for workload in ('pass','move_and_market'):
    action={'farmer':['PASS' if workload=='pass' else 'EAST'],'hands':[],'market':[] if workload=='pass' else [['BUY_SEED','WHEAT',1],['SELL','WHEAT',1]]}
    rows=encode_action(action,hire_limit=241)['frames']
    tokens=torch.zeros((N,2,252,12),dtype=torch.int64)
    tokens[:,:,:len(rows)]=torch.tensor([[row[s] for s in SLOT_NAMES] for row in rows])
    actions=KaggricultureActions(tokens,torch.full((N,2),len(rows),dtype=torch.int64))
    frames=np.ascontiguousarray(tokens.numpy(),dtype=np.int16)
    lengths=np.ascontiguousarray(actions.lengths.numpy(),dtype=np.int32)
    for threads in (1,2,4):
      for layer in ('native','python'):
        samples=[]
        resets=[]
        with KaggricultureVectorizedEnv(n_envs=N,pin_memory=False,seed=2301,threads=threads) as env:
          b=env._buffers
          def step():
            if layer=='native':
              env._batch._native.step_frames(frames,lengths,b.features,b.context,b.banks,b.done,b.observation_version)
            else:
              env.step(actions)
          for _ in range(20):step()
          for rep in range(REPEATS):
            started=time.perf_counter_ns();env.reset();resets.append((time.perf_counter_ns()-started)/1e6)
            elapsed=[]
            for i in range(STEPS):
              started=time.perf_counter_ns();step();elapsed.append(time.perf_counter_ns()-started)
            samples.append(elapsed)
        flat=np.asarray(samples,dtype=np.float64).reshape(-1)/1e6
        result={'workload':workload,'threads':threads,'layer':layer,'active_frames_per_seat':len(rows),'batch_latency_ms_median':float(np.median(flat)),'batch_latency_ms_p95':float(np.quantile(flat,.95)),'raw_game_steps_per_sec':N/(float(flat.mean())/1000),'reset_batch_ms_median':statistics.median(resets),'repeat_mean_ms':[sum(s)/len(s)/1e6 for s in samples]}
        out['results'].append(result)
print(json.dumps(out,indent=2))
