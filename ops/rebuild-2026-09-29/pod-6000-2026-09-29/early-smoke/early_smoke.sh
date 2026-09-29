#!/bin/bash
# Early 2-rank smoke + isolated failure-status probe (pre-landing, pending Codex review).
set -u
R=/root/receipts/prelanding/early-smoke
RUN=/root/runs/kg-v3-prelanding-early-smoke-2rank-20260930
RUNF=/root/runs/kg-v3-prelanding-early-smoke-failprobe-20260930
E=ops/rebuild-2026-09-29/pod-6000-2026-09-29/early-smoke
mkdir -p $R /root/runs
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; } > $R/git_state.txt
sha256sum $E/config.yaml $E/launch_kg_run_ppo.py scripts/run_ppo.py python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py python/owl/rs.abi3.so > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
timeout 700 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 1000 > $R/nvsmi_samples.csv 2>&1 &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=spoon
date -u +"smoke_launch %FT%TZ" > $R/times.txt
S=$(date +%s)
timeout --kill-after=30 360 .venv/bin/torchrun --nproc-per-node 2 $E/launch_kg_run_ppo.py scripts/run_ppo.py $E/config.yaml $RUN \
  --log-mode wandb --wandb-mode online --max-env-steps 32768 --max-runtime-hours 0.1666667 > $R/run.log 2>&1
RC=$?
echo "smoke_exit=$RC wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
if grep -q '"kind": "iteration", "rank": 0, "t": [0-9.]*, "iteration": 2' $R/run.log && [ $RC -eq 0 ]; then
  date -u +"probe_launch %FT%TZ" >> $R/times.txt
  S=$(date +%s)
  KG_PROBE_FAIL_BEFORE_UPDATE=1 timeout --kill-after=30 150 .venv/bin/torchrun --nproc-per-node 2 $E/launch_kg_run_ppo.py scripts/run_ppo.py $E/config.yaml $RUNF \
    --log-mode wandb --wandb-mode online --max-env-steps 16384 > $R/probe.log 2>&1
  echo "probe_exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
else
  echo "probe_skipped (smoke did not pass)" >> $R/times.txt
fi
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -la $RUN $RUNF > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
