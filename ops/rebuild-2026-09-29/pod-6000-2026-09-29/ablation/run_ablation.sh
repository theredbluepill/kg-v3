#!/bin/bash
# PPO collapse ablation: 46 complete iterations (4 full games) from the BC best on 2 ranks.
# Usage: run_ablation.sh NAME [-o key=value ...]   (pre-landing, pending Codex review)
# The pod checkout stays at e74d67e (the 6.2 control's code); this script lives outside it.
set -u
NAME=$1; shift
OVERRIDES="$*"
R=/root/receipts/prelanding/ablation/$NAME
RUN=/root/runs/$NAME
E=ops/rebuild-2026-09-29/pod-6000-2026-09-29/early-smoke
mkdir -p $R /root/runs
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME"; echo "overrides=$OVERRIDES"; } > $R/git_state.txt
sha256sum configs/kaggriculture_2rank.yaml configs/model/kaggriculture.yaml $E/launch_kg_run_ppo.py scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so /root/bc-best/checkpoint_bc_best.pt "$0" > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 700 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon KG_PROBE_STOP_ON_NONFINITE=1
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
timeout --kill-after=30 600 .venv/bin/torchrun --nproc-per-node 2 $E/launch_kg_run_ppo.py scripts/run_ppo.py configs/kaggriculture_2rank.yaml $RUN \
  --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_fresh_critic_head \
  --log-mode wandb --wandb-mode online --max-env-steps 753664 -o rl.eval_replay_games=0 $OVERRIDES > $R/run.log 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
find $RUN -type f -name "*.pt" -exec sha256sum {} \; > $R/checkpoints.sha256 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
