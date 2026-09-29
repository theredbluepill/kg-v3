#!/bin/bash
# Plan 6.2 complete-work run on 2 ranks from the BC best (pre-landing, pending Codex review).
set -u
R=/root/receipts/prelanding/6.2
RUN=/root/runs/kg-v3-prelanding-6.2-bcbest-2rank-30min-20260930
E=ops/rebuild-2026-09-29/pod-6000-2026-09-29/early-smoke
SPLIT_OVERRIDES="${SPLIT_OVERRIDES:-}"
mkdir -p $R /root/runs
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "split_overrides=$SPLIT_OVERRIDES"; } > $R/git_state.txt
sha256sum configs/kaggriculture_2rank.yaml configs/model/kaggriculture.yaml $E/launch_kg_run_ppo.py scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so /root/bc-best/checkpoint_bc_best.pt > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 2000 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon KG_PROBE_STOP_ON_NONFINITE=1
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
timeout --kill-after=60 1860 .venv/bin/torchrun --nproc-per-node 2 $E/launch_kg_run_ppo.py scripts/run_ppo.py configs/kaggriculture_2rank.yaml $RUN \
  --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_fresh_critic_head \
  --log-mode wandb --wandb-mode online --max-runtime-hours 0.47 -o rl.eval_replay_games=0 $SPLIT_OVERRIDES > $R/run.log 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
find $RUN -type f -name "*.pt" -exec sha256sum {} \; > $R/checkpoints.sha256 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
