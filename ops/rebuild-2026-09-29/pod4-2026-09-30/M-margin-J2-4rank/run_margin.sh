#!/bin/bash
# Owner term M (0.5 x cash difference + 0.5 x terminal sign), 4 ranks, pod abl4mvr5w1mmn4.
# Fresh warm start from J/2's checkpoint_final.pt with model_and_optimizer, as hz4bpjnq did.
# Checkout /root/kg-v3-M at 87beaf0 (kg/rebuild-reward-margin, pre-landing core), clean.
# Launched as a session leader:
#   setsid nohup /root/M-margin/run_margin.sh > /root/runs/M-margin-J2-4rank-20260930.log 2>&1 < /dev/null & disown
# No stop limit: no --max-env-steps, no --max-runtime-hours, no -o overrides.
set -u
NAME=M-margin-J2-4rank-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/M-margin/main_probe.py
CFG=configs/kaggriculture_4rank_margin.yaml
SRC=/root/runs/control-J2-4rank-20260930/20260930-010131/checkpoint_final.pt
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3-M
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum $CFG configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py src/kaggriculture/reward.rs python/owl/rs.abi3.so uv.lock Cargo.lock $SRC $P /root/M-margin/watchdog.py "$0" > $R/hashes.sha256
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
  --load-model-weights $SRC --load-model-weights-mode model_and_optimizer \
  --log-mode wandb --wandb-mode online --experiment-id $NAME
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
