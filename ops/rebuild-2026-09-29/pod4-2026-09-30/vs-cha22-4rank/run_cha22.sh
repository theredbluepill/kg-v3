#!/bin/bash
# The cha22 anchor (owner: "Sure you can just launch the same reward on CHA22 run now?"),
# PRE-LANDING launch: kg/rebuild-opponent-mix at 0f70773 (review still running), pod abl4mvr5w1mmn4.
# PPO vs the FIXED bot Cha22 under term M, 4 ranks, from the BC best (model_only; the
# last_best teacher is built from the loaded weights, so teacher = BC best).
# Launched as a session leader:
#   setsid nohup /root/vs-cha22/run_cha22.sh > /root/runs/vs-cha22-4rank-20260930.log 2>&1 < /dev/null & disown
# No stop limit: no --max-env-steps, no --max-runtime-hours, no -o overrides.
set -u
NAME=vs-cha22-4rank-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/vs-cha22/main_probe.py
CFG=configs/kaggriculture_4rank_vs_cha22.yaml
SRC=/root/bc-best/checkpoint_bc_best.pt
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3-anchor
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum $CFG configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py src/kaggriculture/reward.rs src/kaggriculture/opponents.rs src/kaggriculture/env.rs python/owl/rs.abi3.so uv.lock Cargo.lock $SRC $P /root/vs-cha22/watchdog.py "$0" > $R/hashes.sha256
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
  --load-model-weights $SRC --load-model-weights-mode model_only \
  --log-mode wandb --wandb-mode online --experiment-id $NAME
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
