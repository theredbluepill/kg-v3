#!/bin/bash
# J/2 continuation under the 0.2/0.8 reward, 4 ranks, pod abl4mvr5w1mmn4 (owner: "reactivate this run?
# ppo-20260930-010131 ... 0.2 reward sahping, + 0.8 own-opp bank+terminal loss (1/-1/0)").
# run_ppo's resume path refuses this: resume launches reject -o overrides ("resume launches cannot use
# config overrides") and the J/2 run dir has no checkpoint_last_best.pt. So this is a fresh warm start
# from J/2's checkpoint_final.pt with model_and_optimizer (weights + optimizer moments; env_steps,
# player_step_total, games and entities continue; LR scheduler and optimizer step counter are fresh).
# Checkout /root/kg-v3 at 7e87f54, clean (the code J/2 trained on).
# Launched as a session leader:
#   setsid nohup /root/J2-resume/run_resume.sh > /root/runs/J2-resume-r0208-20260930.log 2>&1 < /dev/null & disown
# The watchdog (nonfinite loss only) stops the run by signalling this script's process group.
# No stop limit (owner: "you should not set any time cap"): no --max-env-steps, no --max-runtime-hours;
# run_ppo then stops only at its rollout seed budget.
set -u
NAME=J2-resume-r0208-20260930
R=/root/receipts/$NAME
RUN=/root/runs/$NAME
P=/root/J2-resume/main_probe.py
CFG=configs/kaggriculture_4rank.yaml
SRC=/root/runs/control-J2-4rank-20260930/20260930-010131/checkpoint_final.pt
mkdir -p $R /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cd /root/kg-v3
{ git rev-parse HEAD; git status --porcelain | wc -l; echo "name=$NAME native_threads=4 bind=cpu pgid=$(ps -o pgid= $$)"; } > $R/git_state.txt
sha256sum $CFG configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py src/kaggriculture/reward.rs python/owl/rs.abi3.so uv.lock Cargo.lock $SRC $P /root/J2-resume/watchdog.py "$0" > $R/hashes.sha256
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_prelaunch.csv
nvidia-smi --query-compute-apps=pid,name --format=csv >> $R/idle_prelaunch.csv
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ]; then echo "GPU busy; refusing" | tee -a $R/times.txt; exit 3; fi
# 2 s samples for the first 15 min (early peak memory), then 60 s samples.
timeout 900 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > $R/nvsmi_early.csv 2>&1 < /dev/null &
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 60 > $R/nvsmi_60s.csv 2>&1 < /dev/null &
SAMPLER=$!
export OMP_NUM_THREADS=1 WANDB_ENTITY=spoon KG_NT_NUMA=cpu
export TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton
date -u +"launch %FT%TZ" > $R/times.txt
S=$(date +%s)
.venv/bin/torchrun --nproc-per-node 4 $P scripts/run_ppo.py $CFG $RUN \
  --load-model-weights $SRC --load-model-weights-mode model_and_optimizer \
  --log-mode wandb --wandb-mode online --experiment-id $NAME \
  -o optimizer.muon_lr=0.0001 optimizer.adamw_lr=0.000005 rl.checkpoint_freq=10000000 env.native_threads=4 env.reward_shaping.econ_cap=0.2
echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> $R/times.txt
date -u +"end %FT%TZ" >> $R/times.txt
kill $SAMPLER 2>/dev/null
ls -laR $RUN > $R/run_dir_listing.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > $R/idle_after.csv
echo DONE >> $R/times.txt
