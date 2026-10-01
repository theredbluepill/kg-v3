#!/bin/bash
# Pod-side driver: run the 48-game fixed-shop anchor panel for one unpacked package.
# usage: run_games.sh AGENT_DIR OUT_NAME LABEL_PREFIX [ANCHOR...]
# Linux port of anchor-games/eval-60M/run_fixedshop_60M.sh: same seeds (93001-93008),
# seats (0,1), anchors, labels (<PREFIX>-<anchor>-s<seed>-seat<seat>), run_game.py and
# receipt format; strict agent, nice 10, 1 torch thread per game; PARALLEL (default 30)
# games at once. Rule 1 is whatever the package bakes in (the env var is also set to 1
# for env-switch packages); rule 2 is forced off (KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS=0).
set -u
R=/root/anchor-eval
PY=$R/venv/bin/python
export PYTHONPATH=$R/kenv-fixedshop
AGENT=$(cd "$1" && pwd); OUT=$2; PREFIX=$3; shift 3
ANCHORS=${*:-smaller_market_shock cha22 v56}
SEEDS=${SEEDS:-93001 93002 93003 93004 93005 93006 93007 93008}
PARALLEL=${PARALLEL:-30}
unset V92_SELL_LIB
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export KAGGRICULTURE_FINAL_TURN_LIQUIDATION=${LIQ:-1} KAGGRICULTURE_AGENT_BLOCK_LATE_INVESTMENTS=0
D=$R/games/$OUT
mkdir -p $D/{receipts,logs,replays,work}
list() {
  for anchor in $ANCHORS; do
    for seed in $SEEDS; do
      for seat in 0 1; do echo "$anchor $seed $seat"; done
    done
  done
}
one() {
  anchor=$1 seed=$2 seat=$3
  label=$PREFIX-$anchor-s$seed-seat$seat
  [ -f $D/receipts/$label.json ] && exit 0
  cd $D/work
  t0=$(date +%s)
  nice -n 10 $PY $R/harness/run_game.py --agent-dir $AGENT --seed $seed --opponent $R/anchors/$anchor/main.py \
    --agent-seat $seat --replay-dir $D/replays \
    --receipt $D/receipts/$label.json --label $label \
    > $D/logs/$label.out 2> $D/logs/$label.err
  rc=$?
  echo "$(date -u +%FT%TZ) $label exit=$rc wall=$(( $(date +%s) - t0 ))s"
}
export -f one; export R PY D AGENT PREFIX
start=$(date +%s)
echo "start $(date -u +%FT%TZ) agent=$AGENT out=$OUT parallel=$PARALLEL"
list | xargs -P $PARALLEL -L 1 bash -c 'one $0 $1 $2'
end=$(date +%s)
echo "end $(date -u +%FT%TZ) wall_s=$((end - start))"
$PY - $D <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in sorted((d / "receipts").glob("*.json"))]
eng = sorted({r["runtime"]["kaggriculture_py_sha256"] for r in rows})
man = sorted({r["manifest_sha256"] for r in rows})
print(json.dumps({"receipts": len(rows), "qualified": sum(r["qualified"] for r in rows),
                  "engine_sha256": eng, "manifest_sha256": man}))
PY
