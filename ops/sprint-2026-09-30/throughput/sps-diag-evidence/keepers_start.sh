#!/bin/bash
# SCHED_IDLE busy-loop per CPU in $1 (e.g. 48-59)
for c in $(seq ${1%-*} ${1#*-}); do
  setsid taskset -c $c chrt -i 0 bash -c 'exec -a kg-clock-keeper-spin bash -c "while :; do :; done"' </dev/null >/dev/null 2>&1 &
done
sleep 1; pgrep -fc kg-clock-keeper-spin
