#!/usr/bin/env bash
# Pull checkpoints and receipts from the pod to the Mac's durable storage (DRAFT).
# Run ON THE MAC while the main run is going, and once after it ends:
#   bash ops/rebuild-2026-09-29/8rank-run/pull_from_pod.sh <pod-ssh-target> <pod OUT_DIR> [interval-seconds]
#
# A pod without a network volume loses its disk when stopped, so every
# checkpoint must leave the pod. With an interval it repeats until interrupted.
# Destination: $KG_DURABLE_LOCAL (default ~/kg-v3-runs/<basename of OUT_DIR>),
# outside any repository. After each pull the checkpoint hashes are compared
# with the pod watchdog's checkpoints.sha256.
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
  rsync -a --exclude '.*.tmp' --exclude 'wandb/' "$POD:$REMOTE_OUT/" "$DEST/run/"
  rsync -a "$POD:$REMOTE_RECEIPTS/" "$DEST/receipts/" || true
  find "$DEST/run" -name 'checkpoint_last_best.pt' | while read -r best; do
    digest=$(shasum -a 256 "$best" | cut -c1-12)
    cp -n "$best" "${best%.pt}.$digest.pt"
  done
  (cd "$DEST/run" && find . -name 'checkpoint_*.pt' ! -name 'checkpoint_last_best*' -print0 \
    | xargs -0 shasum -a 256) > "$DEST/checkpoints.local.sha256" || true
  if [ -f "$DEST/receipts/checkpoints.sha256" ]; then
    while read -r digest path; do
      [ -f "$DEST/run/$path" ] || continue
      local_digest=$(shasum -a 256 "$DEST/run/$path" | cut -d' ' -f1)
      [ "$local_digest" = "$digest" ] || [ "$(basename "$path")" = checkpoint_last_best.pt ] \
        || echo "HASH MISMATCH $path" >&2
    done < "$DEST/receipts/checkpoints.sha256"
  fi
  echo "$(date -u +%FT%TZ) pulled to $DEST: $(find "$DEST/run" -name 'checkpoint_*.pt' | wc -l) checkpoint files"
}

pull_once
while [ "$INTERVAL" -gt 0 ]; do
  sleep "$INTERVAL"
  pull_once || echo "pull failed; retrying in $INTERVAL s" >&2
done
