#!/bin/bash
# Owner: halve the win/loss reward (0.5 -> 0.25) and split the spared 0.25 equally: own bank 0.375 (/150k), cash difference 0.375 (/100k), terminal 0.25. Start from pcy5knet 50M checkpoint (0cc80065..., owner-promoted), model_only; 720 window, 12 envs/rank, lambda 1, Muon 1e-4 / AdamW 5e-6 unchanged.
#   setsid nohup /root/earnC/run.sh > /root/runs/earn720-r375-from-c50-4rank-20260930.log 2>&1 < /dev/null & disown
# No stop limit.
set -u
NAME=earn720-r375-from-c50-4rank-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/earnC/main_probe.py
CFG=configs/kaggriculture_4rank_margin.yaml
SRC=/root/promoted-C/checkpoint.pt
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3-anchor
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum $CFG configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py src/kaggriculture/reward.rs src/kaggriculture/env.rs python/owl/rs.abi3.so uv.lock Cargo.lock $SRC $P /root/earnC/watchdog.py "$0" > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ]; then echo "GPU busy; refusing" | tee -a $R/times.txt; exit 3; fi
timeout 900 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_early.csv 2>&1 < /dev/null &
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 60 > $R/nvsmi_60s.csv 2>&1 < /dev/null &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_ENTITY=spoon KG_NT_NUMA=cpu
export TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
.venv/bin/torchrun --nproc-per-node 4 $P scripts/run_ppo.py $CFG $RUN \
  --log-mode wandb --wandb-mode online --experiment-id $NAME --load-model-weights $SRC --load-model-weights-mode model_only -o env.n_envs=12 rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0 env.reward_shaping.econ_bank_weight=0.375 env.reward_shaping.econ_bank_scale=150000 env.reward_shaping.econ_bank_cap=0.375 env.reward_shaping.econ_margin_weight=0.375 env.reward_shaping.econ_margin_scale=100000 env.reward_shaping.econ_margin_cap=0.375
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
