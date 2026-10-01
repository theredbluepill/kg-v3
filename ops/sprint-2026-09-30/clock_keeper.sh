#!/bin/bash
# Clock keeper used on the 8x H200 training pod (7gbzus3pufmik6) during the
# 2026-09-30/10-01 sprint. The spinner command line is verbatim from the pod's
# /root/sps-diag/clock_keeper.sh (as recorded in sprint-facts.md); the CPU-range
# loop follows the archived throughput/sps-diag-evidence/keepers_start.sh.
#
# Why: the host ran intel_cpufreq (passive) + schedutil (800 MHz - 4.0 GHz).
# Rayon env workers sampled at a median 800 MHz while the main thread ran at
# 3.2 GHz. One SCHED_IDLE busy loop per CPU keeps every core clocked up (and
# out of deep C-states) without competing with real work.
# Measured (sprint-facts.md): rollout 27 s -> 18 s, iteration 31.8 -> 21.9 s,
# 3,620 -> ~5,250 env steps/s, no code change.
#
# Caveats from the sprint:
# - The container cgroup quota was 81.6 CPUs; spinners count against it
#   (throttling ~15%, main-thread run-queue wait 11.6%).
# - Spinning only 14 CPUs per node (56 total) was WORSE (3,800-4,450 env
#   steps/s): workers landed on unspun slow cores. Spin every CPU.
# - Stop the keepers before a restart/compile and restart them after the first
#   iteration (a shell bug once skipped the restart: ~3 min at 3,700 env steps/s).
#
# Usage: clock_keeper.sh [FIRST-LAST]   (default: every online CPU)
# Stop:  pkill -f kg-clock-keeper-spin
set -euo pipefail
RANGE=${1:-0-$(($(nproc --all) - 1))}
for c in $(seq "${RANGE%-*}" "${RANGE#*-}"); do
  setsid taskset -c $c chrt -i 0 bash -c 'exec -a kg-clock-keeper-spin bash -c "while :; do :; done"' </dev/null >/dev/null 2>&1 &
done
sleep 1
pgrep -fc kg-clock-keeper-spin
