#!/bin/bash
# Owner (2026-09-30): "let eval complete, I pretty much sure it will promote, but let's run 2e-3 /1e-4 on the new run".
# Waits for cmwjclbe's first 10M checkpoint + promotion eval; if promoted, stops that run and launches the same
# 720-window / 12-env recipe from the new last_best with Muon 2e-3 / AdamW 1e-4 (Isaiah's LRs). If not promoted,
# leaves the run going and exits (the main agent reports to the owner).
set -u
OLD=earn720-12env-from-promoted-4rank-20260930
NEW=earn720-lr2e3-from-promoted2-4rank-20260930
OLDLOG=/root/runs/$OLD.log
D=$(ls -d /root/runs/$OLD/*/ | head -1)
log(){ echo "$(date -u +%FT%TZ) $*"; }
log "waiting for the 10M checkpoint in $D"
until ls $D/checkpoint_00_0*.pt >/dev/null 2>&1; do sleep 30; done
CK=$(ls $D/checkpoint_00_0*.pt | head -1); CKT=$(stat -c %Y $CK); log "checkpoint $CK"
IT0=$(grep -c '"kind": "iteration", "rank": 0' $OLDLOG)
PROMOTED=no
while true; do
  LBT=$(stat -c %Y $D/checkpoint_last_best.pt 2>/dev/null || echo 0)
  if [ "$LBT" -ge "$CKT" ]; then PROMOTED=yes; break; fi
  IT=$(grep -c '"kind": "iteration", "rank": 0' $OLDLOG)
  if [ $((IT - IT0)) -ge 4 ]; then break; fi
  sleep 15
done
log "promoted=$PROMOTED"
if [ $PROMOTED != yes ]; then log "not promoted; leaving $OLD running"; exit 0; fi
sleep 20
mkdir -p /root/promoted-B; cp $D/checkpoint_last_best.pt $D/config.yaml /root/promoted-B/
log "copied last_best sha $(sha256sum /root/promoted-B/checkpoint_last_best.pt | cut -c1-16)"
R=$(ps -eo pid=,args= | awk '/earn720c\/main_probe/ && !/torchrun/ && !/awk/ {print $1}')
W=$(ps -eo pid=,args= | awk '/earn720c\/watchdog/ && !/awk/ {print $1}')
kill -TERM -67658; kill -TERM $R 2>/dev/null; sleep 12; kill $W 2>/dev/null; sleep 2
log "stopped old run; gpu apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
mkdir -p /root/earnlr; cp /root/earn720c/main_probe.py /root/earn720c/watchdog.py /root/earnlr/
f=/root/earnlr/run.sh
sed -e "s#$OLD#$NEW#g" -e "s#/root/earn720c/#/root/earnlr/#g" -e "s#^SRC=.*#SRC=/root/promoted-B/checkpoint_last_best.pt#" \
    -e "s#env.reward_shaping.econ_margin_cap=0.25\$#env.reward_shaping.econ_margin_cap=0.25 optimizer.muon_lr=0.002 optimizer.adamw_lr=0.0001#" /root/earn720c/run.sh > $f
sed -i "2s#.*#\# Owner: let eval complete ... let's run 2e-3 /1e-4 on the new run. From cmwjclbe's promoted 10M last_best (model_only), 720 window, 12 envs/rank, Muon 2e-3 / AdamW 1e-4 (Isaiah). Previous full-LR arms from BC collapsed (ablation); this run tests it from the promoted policy with the 720 window.#" $f
chmod +x $f; grep -c "muon_lr=0.002" $f
cd /root/kg-v3-anchor; setsid nohup $f > /root/runs/$NEW.log 2>&1 < /dev/null &
sleep 5; P=$(ps -eo pid=,args= | awk '/earnlr\/run.sh/ && !/awk/ {print $1}' | head -1); log "launched pgid=$P"
until grep -q '"kind": "iteration", "rank": 0' /root/runs/$NEW.log || grep -q Traceback /root/runs/$NEW.log; do sleep 10; done
setsid nohup .venv/bin/python -u /root/earnlr/watchdog.py $P /root/runs/$NEW.log > /root/runs/earnlr-watchdog.log 2>&1 < /dev/null &
sleep 3; log "watchdog: $(tail -1 /root/runs/earnlr-watchdog.log)"; log DONE
