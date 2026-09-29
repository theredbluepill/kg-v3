#!/bin/bash
# Task B: plan 6.4 Orbit end-to-end. Watchdog: timeout 1800 (+60 s kill).
R=/root/receipts/6.4-orbit
RUN=/root/runs/kg-v3-6.4-orbit-scaling6m-2it-20260930
mkdir -p $R /root/runs
cd /root/kg-v3
git rev-parse HEAD > $R/git_state.txt; git status --porcelain | wc -l >> $R/git_state.txt
sha256sum configs/scaling_6m.yaml configs/model/stateless_transformer_6m.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/rs.abi3.so /root/launch_run_ppo_v3.py > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 1900 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 1000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_ENTITY=spoon WANDB_RUN_GROUP=kg-v3-6.4-orbit \
  WANDB_TAGS=kg-v3,task-6.4,orbit-regression,pod-aki4vy8kpfldpa,src-994818b \
  WANDB_NOTES="Plan 6.4 Orbit end-to-end smoke; scaling_6m, 1 GPU, 2 iterations, forced eval via checkpoint_freq=32768; integration 994818b"
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
timeout --kill-after=60 1800 .venv/bin/python /root/launch_run_ppo_v3.py scripts/run_ppo.py configs/scaling_6m.yaml $RUN \
  --log-mode wandb --max-env-steps 32768 -o rl.checkpoint_freq=32768 > $R/run.log 2>&1
RC=$?
E=$(date +%s)
echo "exit=$RC wall_s=$((E-S))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -la $RUN > $R/run_dir_listing.txt 2>&1
find $RUN -maxdepth 3 -type f -name "*.pt" -exec sha256sum {} \; > $R/checkpoints.sha256 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
