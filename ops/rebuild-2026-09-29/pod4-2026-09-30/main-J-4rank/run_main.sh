#!/bin/bash
# Main run J, 4 ranks, pod abl4mvr5w1mmn4. Launched as a session leader:
#   setsid nohup /root/main-J/run_main.sh > /root/runs/main-J-4rank-20260930.log 2>&1 < /dev/null &
# The watchdog stops the run by signalling this script's process group.
# Lives outside the checkout (/root/main-J); the checkout stays at 7e87f54, clean.
set -u
NAME=main-J-4rank-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/main-J/main_probe.py
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum configs/kaggriculture_4rank.yaml configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so /root/bc-best/checkpoint_bc_best.pt $P "$0" > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ]; then echo "GPU busy; refusing" | tee -a $R/times.txt; exit 3; fi
# 2 s samples for the first 15 min (early peak memory), then 60 s samples.
timeout 900 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_early.csv 2>&1 &
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 60 > $R/nvsmi_60s.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_ENTITY=spoon KG_NT_NUMA=cpu
export TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
.venv/bin/torchrun --nproc-per-node 4 $P scripts/run_ppo.py configs/kaggriculture_4rank.yaml $RUN \
  --load-model-weights /root/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only \
  --log-mode wandb --wandb-mode online --experiment-id $NAME --max-runtime-hours 20 \
  -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001 rl.checkpoint_freq=10000000 env.native_threads=4
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
