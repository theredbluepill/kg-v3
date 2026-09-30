#!/bin/bash
# usage: run_all.sh POLICY(candidate|bc) ; 4 anchors x 8 seeds x 2 seats, 2 parallel, nice 10, 1 thread.
set -u
W=/Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/anchor-games
PY=/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/venv-kaggle/bin/python
POLICY=$1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W/games/$POLICY/{receipts,logs,replays,work}
jobs() {
  for anchor in smaller_market_shock; do
    for seed in 93001 93002 93003 93004 93005 93006 93007 93008; do
      for seat in 0 1; do echo "$anchor $seed $seat"; done
    done
  done
}
one() {
  anchor=$1 seed=$2 seat=$3
  label=$POLICY-$anchor-s$seed-seat$seat
  [ -f $W/games/$POLICY/receipts/$label.json ] && exit 0
  if [ $anchor = starter ]; then opp=starter; else opp=$W/anchors/$anchor/main.py; fi
  cd $W/games/$POLICY/work
  nice -n 10 $PY $W/run_game.py --agent-dir $W/pkg-$POLICY --seed $seed --opponent $opp \
    --agent-seat $seat --replay-dir $W/games/$POLICY/replays \
    --receipt $W/games/$POLICY/receipts/$label.json --label $label \
    > $W/games/$POLICY/logs/$label.out 2> $W/games/$POLICY/logs/$label.err
  rc=$?
  echo "$(date -u +%FT%TZ) $label exit=$rc"
}
export -f one; export W PY POLICY
date -u +%FT%TZ
jobs | xargs -P 3 -L 1 bash -c 'one $0 $1 $2'
date -u +%FT%TZ
