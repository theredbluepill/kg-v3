#!/bin/bash
# Owner: safe to increase the games? 16 envs/rank used 91 of 96 GB per GPU, too tight unattended; 12 envs/rank: 12 optimizer steps of 1,440 rows per iteration, 34,560 env steps per iteration, 12 eval games. Horizon 720 (one game per segment), promoted 10M start (fc6b123c...), model_only.
#   setsid nohup /root/earn720c/run.sh > /root/runs/earn720-12env-from-promoted-4rank-20260930.log 2>&1 < /dev/null & disown
# No stop limit.
set -u
NAME=earn720-12env-from-promoted-4rank-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/earn720c/main_probe.py
CFG=configs/kaggriculture_4rank_margin.yaml
SRC=/root/promoted-10M/checkpoint_last_best.pt
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3-anchor
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum $CFG configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py src/kaggriculture/reward.rs src/kaggriculture/env.rs python/owl/rs.abi3.so uv.lock Cargo.lock $SRC $P /root/earn720c/watchdog.py "$0" > $R/hashes.sha256
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
  --log-mode wandb --wandb-mode online --experiment-id $NAME --load-model-weights $SRC --load-model-weights-mode model_only -o env.n_envs=12 rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0 env.reward_shaping.econ_bank_weight=0.25 env.reward_shaping.econ_bank_scale=150000 env.reward_shaping.econ_bank_cap=0.25 env.reward_shaping.econ_margin_weight=0.25 env.reward_shaping.econ_margin_scale=100000 env.reward_shaping.econ_margin_cap=0.25
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
