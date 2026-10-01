#!/bin/bash
# usage: sps_train.sh <name> <on|off> ; runs under taskset 48-59, stops after N_ITERS probe iteration records
set -u
NAME=$1 MODE=$2 N_ITERS=${N_ITERS:-7}
RUN=/root/runs/$NAME LOG=/root/sps-diag/$NAME.log
cd /root/kg-v3
export OMP_NUM_THREADS=1 WANDB_ENTITY=spoon KG_NT_NUMA=cpu KG_NT_NATIVE_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton
SW=()
if [ "$MODE" = actor ]; then SW=(rl.compile_actor_heads=true); fi
if [ "$MODE" = on ]; then SW=(rl.compile_actor_heads=true rl.rollout_packing=true rl.pinned_action_d2h=true env.skip_reward_telemetry_validation=true); fi
grep throttled /sys/fs/cgroup/cpu.stat > /root/sps-diag/$NAME.cpustat.before
setsid taskset -c 48-57 .venv/bin/torchrun --nproc-per-node 1 /root/sprint-kit/main_probe_auto.py scripts/run_ppo.py configs/kaggriculture_4rank_margin.yaml $RUN --log-mode wandb --wandb-mode online --experiment-id $NAME --load-model-weights /root/sps-diag/ckpt/checkpoint_00_070_041_344.pt --load-model-weights-mode model_only -o env.n_envs=20 rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0 env.reward_shaping.econ_bank_weight=0.3 env.reward_shaping.econ_bank_scale=150000 env.reward_shaping.econ_bank_cap=0.3 env.reward_shaping.econ_margin_weight=0.3 env.reward_shaping.econ_margin_scale=100000 env.reward_shaping.econ_margin_cap=0.3 env.reward_shaping.econ_shaping=0.01 env.reward_shaping.econ_cap=0.1 rl.checkpoint_freq=1000000000 "${SW[@]}" > $LOG 2>&1 < /dev/null &
PID=$!
while kill -0 $PID 2>/dev/null; do
  n=$(grep -c '"kind": "iteration"' $LOG)
  if [ "$n" -ge "$N_ITERS" ]; then
    kill -TERM -- -$(ps -o pgid= $PID | tr -d ' ') 2>/dev/null
    sleep 20; kill -KILL -- -$(ps -o pgid= $PID | tr -d ' ') 2>/dev/null
    break
  fi
  sleep 5
done
wait $PID 2>/dev/null
grep throttled /sys/fs/cgroup/cpu.stat > /root/sps-diag/$NAME.cpustat.after
echo "DONE $NAME iters=$(grep -c '"kind": "iteration"' $LOG)" >> $LOG
