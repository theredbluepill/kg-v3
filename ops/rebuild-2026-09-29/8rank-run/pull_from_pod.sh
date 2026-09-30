#!/usr/bin/env bash
# Pull checkpoints and receipts from the pod to the Mac's durable storage (DRAFT).
# Run ON THE MAC while the main run is going, and once after it ends:
#   bash ops/rebuild-2026-09-29/8rank-run/pull_from_pod.sh <pod-ssh-target> <pod OUT_DIR> [interval-seconds]
#
# A pod without a network volume loses its disk when stopped, so every
# checkpoint must leave the pod. With an interval it repeats until interrupted.
# Destination: $KG_DURABLE_LOCAL (default ~/kg-v3-runs/<basename of OUT_DIR>),
# outside any repository. After each pull the checkpoint hashes are compared
# with the pod's two hash lists (paths relative to OUT_DIR):
#   checkpoints.sha256        the watchdog's append-only list; every version of
#                             checkpoint_last_best.pt must exist here as
#                             checkpoint_last_best.<digest12>.pt, and for other
#                             files the newest line counts;
#   checkpoints_final.sha256  launch.sh's end-of-run list (kg_post).
# Each list prints "verified N / mismatched M / missing K". A mismatch fails the
# pull. Without an interval (the final pull after the run) a missing file or an
# absent watchdog list fails it too; with an interval the pull only warns about
# them and repeats.
set -euo pipefail

POD=${1:?usage: pull_from_pod.sh <pod-ssh-target> <pod OUT_DIR> [interval-seconds]}
REMOTE_OUT=${2:?usage: pull_from_pod.sh <pod-ssh-target> <pod OUT_DIR> [interval-seconds]}
INTERVAL=${3:-0}
NAME=$(basename "$REMOTE_OUT")
DEST=${KG_DURABLE_LOCAL:-$HOME/kg-v3-runs/$NAME}
REMOTE_RECEIPTS=${KG_POD_RECEIPTS:-/workspace/kg-v3-receipts}/main/$NAME
mkdir -p "$DEST/run" "$DEST/receipts"

pull_once() {
  # Skip the atomic writer's .checkpoint_*.tmp files; keep every version of
  # checkpoint_last_best.pt by hash, since promotion rewrites it in place.
  rsync -a --exclude '.*.tmp' --exclude 'wandb/' "$POD:$REMOTE_OUT/" "$DEST/run/" || return 1
  rsync -a "$POD:$REMOTE_RECEIPTS/" "$DEST/receipts/" || true
  find "$DEST/run" -name 'checkpoint_last_best.pt' | while read -r best; do
    digest=$(shasum -a 256 "$best" | cut -c1-12)
    # Not cp -n: macOS's cp -n exits 1 when the target exists.
    [ -e "${best%.pt}.$digest.pt" ] || cp "$best" "${best%.pt}.$digest.pt"
  done
  (cd "$DEST/run" && find . -name 'checkpoint_*.pt' ! -name 'checkpoint_last_best*' -print0 \
    | xargs -0 shasum -a 256) > "$DEST/checkpoints.local.sha256" || true
  echo "$(date -u +%FT%TZ) pulled to $DEST: $(find "$DEST/run" -name 'checkpoint_*.pt' | wc -l) checkpoint files"
  local failed=0
  if [ -f "$DEST/receipts/checkpoints.sha256" ]; then
    verify_list "$DEST/receipts/checkpoints.sha256" || failed=1
  else
    echo "no watchdog checkpoints.sha256 in $DEST/receipts" >&2
    [ "$INTERVAL" -gt 0 ] || failed=1
  fi
  if [ -f "$DEST/receipts/checkpoints_final.sha256" ]; then
    verify_list "$DEST/receipts/checkpoints_final.sha256" || failed=1
  fi
  return "$failed"
}

# verify_list FILE: compare "<sha256>  <path relative to OUT_DIR>" lines with the
# local copy.
verify_list() {
  local list=$1 verified=0 mismatched=0 missing=0 digest path target
  while read -r digest path; do
    if [ "$(basename "$path")" = checkpoint_last_best.pt ]; then
      target=$DEST/run/${path%.pt}.${digest:0:12}.pt
    else
      target=$DEST/run/$path
    fi
    if [ ! -f "$target" ]; then
      missing=$((missing + 1))
      echo "MISSING $path (${digest:0:12})" >&2
    elif [ "$(shasum -a 256 "$target" | cut -d' ' -f1)" = "$digest" ]; then
      verified=$((verified + 1))
    else
      mismatched=$((mismatched + 1))
      echo "HASH MISMATCH $path" >&2
    fi
  done < <(awk 'NF == 2 { if ($2 ~ /(^|\/)checkpoint_last_best\.pt$/) print; else last[$2] = $1 }
    END { for (p in last) print last[p], p }' "$list")
  echo "$(basename "$list"): verified $verified / mismatched $mismatched / missing $missing"
  [ "$mismatched" = 0 ] || return 1
  [ "$missing" = 0 ] || [ "$INTERVAL" -gt 0 ] || return 1
}

if [ "$INTERVAL" -eq 0 ]; then
  pull_once
  exit
fi
while true; do
  pull_once || echo "pull or verification failed; retrying in $INTERVAL s" >&2
  sleep "$INTERVAL"
done
