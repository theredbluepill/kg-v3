#!/bin/bash
# usage: run_fixedshop_70M.sh OUT_NAME LIQ(0|1) [ANCHOR...]   (env SEEDS, SEATS override)
# Copy of eval-60M/run_fixedshop_60M.sh with the package switched to pkg-70M
# (kg/submit-08bc 4c99768a = rule 1 only, no rule-2 code; checkpoint 70M 4c8b9483...),
# labels prefixed 70M-, and the game run through eval-70M/run_game_prerule.py, which
# records the policy's own step-718 action before rule 1 rewrites it (prerule/<label>.json).
# Fixed-shop anchors, 8 seeds x 2 seats, 5 parallel, nice 10, 1 thread;
# KAGGRICULTURE_FINAL_TURN_LIQUIDATION=LIQ; rule-2 switch unset.
set -u
W=/Users/poonszesen/kg-v3-int/ops/earn-money-2026-09-30/anchor-games
PY=/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/venv-kaggle/bin/python; export PYTHONPATH=/private/tmp/claude-501/-Users-poonszesen-kg-v3/3f7ce81c-f27d-4f24-bb54-c9e0de2d998a/scratchpad/kenv-fixedshop
OUT=$1; LIQ=$2; shift 2
ANCHORS=${*:-smaller_market_shock cha22 v56}
SEEDS=${SEEDS:-93001 93002 93003 93004 93005 93006 93007 93008}
SEATS=${SEATS:-0 1}
unset KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 KAGGRICULTURE_FINAL_TURN_LIQUIDATION=$LIQ
D=$W/games-fixedshop/$OUT
mkdir -p $D/{receipts,logs,replays,work,prerule}
jobs() {
  for anchor in $ANCHORS; do
    for seed in $SEEDS; do
      for seat in $SEATS; do echo "$anchor $seed $seat"; done
    done
  done
}
one() {
  anchor=$1 seed=$2 seat=$3
  label=70M-$anchor-s$seed-seat$seat
  [ -f $D/receipts/$label.json ] && exit 0
  cd $D/work
  nice -n 10 $PY $W/eval-70M/run_game_prerule.py --prerule $D/prerule/$label.json \
    --agent-dir $W/pkg-70M --seed $seed --opponent $W/anchors/$anchor/main.py \
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
