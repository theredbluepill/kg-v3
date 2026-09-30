#!/bin/bash
# usage: run_fixedshop_r12.sh OUT_NAME LIQ(0|1) BLOCK(0|1) [ANCHOR...]
# pkg-c50-r12 (kg/submit-08bc 31c99619 = rule 1 + rule 2) vs fixed-shop anchors,
# 8 seeds x 2 seats, 5 parallel, nice 10, 1 thread, through run_game_logged.py
# (logs every rule-2 block). KAGGRICULTURE_FINAL_TURN_LIQUIDATION=LIQ,
# KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS=BLOCK.
set -u
W=/Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/anchor-games
PY=/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/venv-kaggle/bin/python; export PYTHONPATH=/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/kenv-fixedshop
OUT=$1; LIQ=$2; BLOCK=$3; shift 3
ANCHORS=${*:-smaller_market_shock cha22 v56}
SEEDS=${SEEDS:-93001 93002 93003 93004 93005 93006 93007 93008}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export KAGGRICULTURE_FINAL_TURN_LIQUIDATION=$LIQ KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS=$BLOCK
D=$W/games-fixedshop/$OUT
mkdir -p $D/{receipts,logs,replays,work,blocks}
jobs() {
  for anchor in $ANCHORS; do
    for seed in $SEEDS; do
      for seat in 0 1; do echo "$anchor $seed $seat"; done
    done
  done
}
one() {
  anchor=$1 seed=$2 seat=$3
  label=c50-$anchor-s$seed-seat$seat
  [ -f $D/receipts/$label.json ] && exit 0
  cd $D/work
  nice -n 10 $PY $W/endgame/run_game_logged.py --blocks $D/blocks/$label.json \
    --agent-dir $W/pkg-c50-r12 --seed $seed --opponent $W/anchors/$anchor/main.py \
    --agent-seat $seat --replay-dir $D/replays \
    --receipt $D/receipts/$label.json --label $label \
    > $D/logs/$label.out 2> $D/logs/$label.err
  rc=$?
  echo "$(date -u +%FT%TZ) $label exit=$rc"
}
export -f one; export W PY D
date -u +%FT%TZ
jobs | xargs -P 5 -L 1 bash -c 'one $0 $1 $2'
date -u +%FT%TZ
