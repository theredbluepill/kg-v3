#!/bin/bash
# Owner: Sure please do (stop the collapsing 2e-3 run pz3xhg9e, relaunch at 1e-4). From cmwjclbe promoted 20M (/root/promoted-B, weights = checkpoint_00_020_033_024), model_only, 720 window, 12 envs/rank, Muon 1e-4 / AdamW 5e-6 (preset).
#   setsid nohup /root/earnB/run.sh > /root/runs/earn720-lr1e4-from-promoted2-4rank-20260930.log 2>&1 < /dev/null & disown
# No stop limit.
set -u
NAME=earn720-lr1e4-from-promoted2-4rank-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/earnB/main_probe.py
CFG=configs/kaggriculture_4rank_margin.yaml
SRC=/root/promoted-B/checkpoint_last_best.pt
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3-anchor
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum $CFG configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py src/kaggriculture/reward.rs src/kaggriculture/env.rs python/owl/rs.abi3.so uv.lock Cargo.lock $SRC $P /root/earnB/watchdog.py "$0" > $R/hashes.sha256
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
