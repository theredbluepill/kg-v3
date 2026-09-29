#!/bin/bash
# PPO collapse ablation C (critic warm-up), DIAGNOSTIC-ONLY hook; pre-landing, pending Codex review.
# Same as ../run_ablation.sh except: the torchrun entry is critic_warmup_launcher.py, which loads the
# unchanged shared launcher ($E/launch_kg_run_ppo.py) and adds the warm-up loss mask and grad audit.
# Usage: run_c.sh NAME WARMUP_ITERS MAX_ENV_STEPS GRAD_AUDIT LOG_MODE
#   main run: run_c.sh ablate-C-critic-warmup 22 753664 first wandb
#   dry run:  run_c.sh ablate-C-dryrun 1 32768 all debug
set -u
NAME=$1 WARMUP=$2 MAX_STEPS=$3 AUDIT=$4 LOG_MODE=$5
R=/root/receipts/prelanding/ablation/$NAME
RUN=/root/runs/$NAME
E=ops/rebuild-2026-09-29/pod-6000-2026-09-29/early-smoke
HOOK=/root/ablation/critic_warmup_launcher.py
if [ "$LOG_MODE" = wandb ]; then LOG_ARGS="--log-mode wandb --wandb-mode online"; else LOG_ARGS="--log-mode debug"; fi
mkdir -p $R /root/runs
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME warmup_iters=$WARMUP max_env_steps=$MAX_STEPS grad_audit=$AUDIT log=$LOG_ARGS"; } > $R/git_state.txt
sha256sum configs/kaggriculture_2rank.yaml configs/model/kaggriculture.yaml $E/launch_kg_run_ppo.py scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so /root/bc-best/checkpoint_bc_best.pt "$0" $HOOK > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 700 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon KG_PROBE_STOP_ON_NONFINITE=1
export KG_BASE_LAUNCHER=$E/launch_kg_run_ppo.py KG_DIAG_CRITIC_WARMUP_ITERS=$WARMUP KG_DIAG_GRAD_AUDIT=$AUDIT
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
timeout --kill-after=30 600 .venv/bin/torchrun --nproc-per-node 2 $HOOK scripts/run_ppo.py configs/kaggriculture_2rank.yaml $RUN \
  --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_fresh_critic_head \
  $LOG_ARGS --max-env-steps $MAX_STEPS -o rl.eval_replay_games=0 > $R/run.log 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
find $RUN -type f -name "*.pt" -exec sha256sum {} \; > $R/checkpoints.sha256 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
