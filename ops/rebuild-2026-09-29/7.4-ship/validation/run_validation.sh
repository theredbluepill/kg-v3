#!/bin/bash
# Kaggle-mode validation of the unpacked 7.4 submission on the pod (CPU only).
# Runs in /root/ship/validate only; nice -n 19; no GPU; OMP/MKL threads 1.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset KAGGRICULTURE_AGENT_STRICT
V=/root/ship/validate
PY=$V/venv-kaggle/bin/python
A=$V/agent
mkdir -p $V/out $V/work
cd $V/work
date -u +%FT%TZ
for seat in 0 1; do
  seed=$((20261000 + seat))
  nice -n 19 $PY $V/kaggle_mode_episode.py --agent-dir $A --agent-seat $seat \
    --opponent starter --seed $seed --replay $V/out/replay-starter-seat$seat-seed$seed.json \
    --receipt $V/out/episode-starter-seat$seat-seed$seed.json > $V/out/episode-starter-seat$seat.log 2>&1
  echo "seat$seat exit=$?"; date -u +%FT%TZ
done
# The env's own CLI path: kaggle-environments run, self-play (Kaggle's validation-episode shape).
nice -n 19 $V/venv-kaggle/bin/kaggle-environments run --environment kaggriculture \
  --agents $A/main.py $A/main.py --configuration '{"seed": 20261002}' \
  --out $V/out/cli-run-self-seed20261002.json --log $V/out/cli-run-self-seed20261002-logs.json \
  > $V/out/cli-run-self.log 2>&1 &
pid=$!
hwm=0
while kill -0 $pid 2>/dev/null; do
  h=$(awk '/^VmHWM:/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$h" ] && hwm=$h
  sleep 1
done
wait $pid; rc=$?
echo "cli VmHWM_kib=$hwm (last 1 s poll before exit)"
echo "cli exit=$rc"; date -u +%FT%TZ
