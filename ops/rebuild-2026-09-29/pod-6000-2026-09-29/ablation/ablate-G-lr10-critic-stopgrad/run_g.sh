#!/bin/bash
# PPO collapse ablation G (LR / 10, fresh critic head, critic STOP-GRADIENT); pre-landing, pending Codex review.
# Same as ../ablate-E-lr10-vf05/run_e.sh (itself ../run_ablation.sh plus D's telemetry-only trunk audit and
# the GPU lock) except:
#  - the torchrun entry is critic_stopgrad_launcher.py, a DIAGNOSTIC-ONLY hook that CHANGES TRAINING:
#    with STOPGRAD=on the critic head reads the trunk's critic-token features through .detach(), so the
#    value loss and teacher value distillation train critic_head only. It loads D's unchanged
#    trunk_audit_launcher.py, which loads the unchanged shared launcher ($E/launch_kg_run_ppo.py);
#  - two extra positional arguments: STOPGRAD (on|off) and CHECK_ITERS (iterations whose first PPO-loss
#    call runs the value-only gradient check; comma list, or "none").
# Usage: run_g.sh NAME MAX_ENV_STEPS AUDIT_ITERS LOG_MODE STOPGRAD CHECK_ITERS [-o key=value ...]
#   main run:     run_g.sh ablate-G-lr10-critic-stopgrad 753664 12,23,34,46 wandb on 1 -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001
#   dry run:      run_g.sh ablate-G-dryrun-on 32768 1,2 debug on 1,2 -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001
#   control dry:  run_g.sh ablate-G-dryrun-off 32768 1,2 debug off 1,2 -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001
set -u
SELF=$(readlink -f "$0")
NAME=$1 MAX_STEPS=$2 AUDIT_ITERS=$3 LOG_MODE=$4 STOPGRAD=$5 CHECK_ITERS=$6; shift 6
[ "$CHECK_ITERS" = none ] && CHECK_ITERS=
case "$STOPGRAD" in on|off) ;; *) echo "STOPGRAD must be on|off" >&2; exit 2;; esac
OVERRIDES="$*"
R=/root/receipts/prelanding/ablation/$NAME
RUN=/root/runs/$NAME
E=ops/rebuild-2026-09-29/pod-6000-2026-09-29/early-smoke
AUDIT_HOOK=/root/ablation/ablate-D-lr10-bccritic/trunk_audit_launcher.py
HOOK=/root/ablation/ablate-G-lr10-critic-stopgrad/critic_stopgrad_launcher.py
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
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME max_env_steps=$MAX_STEPS audit_iters=$AUDIT_ITERS log=$LOG_ARGS overrides=$OVERRIDES load_mode=model_fresh_critic_head critic_stopgrad=$STOPGRAD stopgrad_check_iters=$CHECK_ITERS"; } > $R/git_state.txt
sha256sum configs/kaggriculture_2rank.yaml configs/model/kaggriculture.yaml $E/launch_kg_run_ppo.py scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so /root/bc-best/checkpoint_bc_best.pt "$SELF" $AUDIT_HOOK $HOOK > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 700 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon KG_PROBE_STOP_ON_NONFINITE=1
export KG_BASE_LAUNCHER=$E/launch_kg_run_ppo.py KG_TRUNK_AUDIT_HOOK=$AUDIT_HOOK KG_DIAG_TRUNK_AUDIT_ITERS=$AUDIT_ITERS KG_DIAG_GRAD_AUDIT=first
export KG_DIAG_CRITIC_STOPGRAD=$STOPGRAD KG_DIAG_STOPGRAD_CHECK_ITERS=$CHECK_ITERS
date -u +"launch %FT%TZ" >> $R/times.txt
S=$(date +%s)
timeout --kill-after=30 600 .venv/bin/torchrun --nproc-per-node 2 $HOOK scripts/run_ppo.py configs/kaggriculture_2rank.yaml $RUN \
  --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_fresh_critic_head \
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
