#!/bin/bash
# PPO collapse ablation D (LR / 10 with the BC critic head); pre-landing, pending Codex review.
# Same as ../run_ablation.sh except:
#  - load mode model_only (keeps the BC critic head) instead of model_fresh_critic_head;
#  - the torchrun entry is trunk_audit_launcher.py, a DIAGNOSTIC-ONLY, telemetry-only hook that
#    loads the unchanged shared launcher ($E/launch_kg_run_ppo.py) and adds a read-only trunk audit;
#  - it waits (bounded) for an exclusive GPU lock and idle GPUs, since other ablation arms share the pod.
# Usage: run_d.sh NAME MAX_ENV_STEPS AUDIT_ITERS LOG_MODE [-o key=value ...]
#   main run: run_d.sh ablate-D-lr10-bccritic 753664 12,23,34,46 wandb -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001
#   dry run:  run_d.sh ablate-D-dryrun 32768 1,2 debug -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001
set -u
SELF=$(readlink -f "$0")
NAME=$1 MAX_STEPS=$2 AUDIT_ITERS=$3 LOG_MODE=$4; shift 4
OVERRIDES="$*"
R=/root/receipts/prelanding/ablation/$NAME
RUN=/root/runs/$NAME
E=ops/rebuild-2026-09-29/pod-6000-2026-09-29/early-smoke
HOOK=/root/ablation/ablate-D-lr10-bccritic/trunk_audit_launcher.py
if [ "$LOG_MODE" = wandb ]; then LOG_ARGS="--log-mode wandb --wandb-mode online"; else LOG_ARGS="--log-mode debug"; fi
mkdir -p $R /root/runs
exec 9>/root/ablation/gpu.lock
date -u +"lock_wait_start %FT%TZ" > $R/times.txt
flock -w 3600 9 || { echo "lock timeout" >> $R/times.txt; exit 3; }
for i in $(seq 1 360); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 10
done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "gpus busy" >> $R/times.txt; exit 4; }
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME max_env_steps=$MAX_STEPS audit_iters=$AUDIT_ITERS log=$LOG_ARGS overrides=$OVERRIDES load_mode=model_only"; } > $R/git_state.txt
sha256sum configs/kaggriculture_2rank.yaml configs/model/kaggriculture.yaml $E/launch_kg_run_ppo.py scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so /root/bc-best/checkpoint_bc_best.pt "$SELF" $HOOK > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 700 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon KG_PROBE_STOP_ON_NONFINITE=1
export KG_BASE_LAUNCHER=$E/launch_kg_run_ppo.py KG_DIAG_TRUNK_AUDIT_ITERS=$AUDIT_ITERS KG_DIAG_GRAD_AUDIT=first
date -u +"launch %FT%TZ" >> $R/times.txt
S=$(date +%s)
timeout --kill-after=30 600 .venv/bin/torchrun --nproc-per-node 2 $HOOK scripts/run_ppo.py configs/kaggriculture_2rank.yaml $RUN \
  --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only \
  $LOG_ARGS --max-env-steps $MAX_STEPS -o rl.eval_replay_games=0 $OVERRIDES > $R/run.log 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
find $RUN -type f -name "*.pt" -exec sha256sum {} \; > $R/checkpoints.sha256 2>&1
sleep 20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_after.csv
echo DONE >> $R/times.txt
