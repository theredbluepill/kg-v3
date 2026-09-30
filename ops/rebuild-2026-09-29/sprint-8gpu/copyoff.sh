#!/bin/bash
# Mac copy-off loop for a sprint run. Usage:
#   copyoff.sh HOST PORT NAME [DEST]     e.g. copyoff.sh root@1.2.3.4 12345 earn720-...-8gpu
#   (DEST defaults to /Users/poonszesen/kg-v3-runs/NAME)
# Run detached: nohup copyoff.sh ... >> DEST/copyoff.log 2>&1 &   (mac_side.md step 7)
# Generalised from ../pod4-2026-09-30/earnB/copyoff.sh. Every INTERVAL s (default 600):
# copy the run log, the watchdog log and the receipts; copy each finished checkpoint_*.pt
# (unchanged for 60 s; the trainer writes .tmp then renames) that is new or whose size OR
# mtime changed (earnB compared size only, so an in-place last_best promotion of the same
# size was never recopied), verify it against the pod's sha256, keep every distinct
# checkpoint_last_best.pt as last_best-history/<pod mtime>-<sha12>.pt, and rewrite
# DEST/SHA256SUMS. Needs rsync on the pod (bootstrap.sh installs it).
# ONCE=1 does a single pass and exits (the final copy after stop.sh).
set -u
HOST=${1:?usage: copyoff.sh HOST PORT NAME [DEST]}
PORT=${2:?usage: copyoff.sh HOST PORT NAME [DEST]}
NAME=${3:?usage: copyoff.sh HOST PORT NAME [DEST]}
DEST=${4:-/Users/poonszesen/kg-v3-runs/$NAME}
INTERVAL=${INTERVAL:-600}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "PORT must be numeric" >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "bad NAME" >&2; exit 2; }
KEY_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$HOME/.ssh/known_hosts.runpod" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)
SSH=(ssh "${KEY_OPTS[@]}" -p "$PORT" "$HOST")
RSYNC_E="ssh ${KEY_OPTS[*]} -p $PORT"
mkdir -p "$DEST/checkpoints" "$DEST/receipts" "$DEST/last_best-history"
LOCK="$DEST/.copyoff.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then echo "another copyoff (pid $(cat "$LOCK/pid")) runs for $DEST" >&2; exit 1; fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
echo "$(date -u +%FT%TZ) copyoff start host=$HOST port=$PORT name=$NAME dest=$DEST pid=$$"
while true; do
  echo "$(date -u +%FT%TZ) pass start"
  rsync -a --partial -e "$RSYNC_E" "$HOST:/root/runs/$NAME.log" "$HOST:/root/runs/$NAME-watchdog.log" "$DEST/" 2>&1 | sed 's/^/  rsync-log: /'
  rsync -a -e "$RSYNC_E" "$HOST:/root/receipts/$NAME/" "$DEST/receipts/" 2>&1 | sed 's/^/  rsync-receipts: /'
  LIST=$("${SSH[@]}" "cd /root/runs/$NAME 2>/dev/null && find . -name 'checkpoint_*.pt' -mmin +1 -printf '%P\t%s\t%T@\n'" 2>&1) || { echo "  list failed: $LIST"; LIST=""; }
  while IFS=$'\t' read -r REL SIZE MTIME; do
    [ -z "$REL" ] && continue
    [[ "$SIZE" =~ ^[0-9]+$ ]] || { echo "  skip unparsable line: $REL"; continue; }
    MTIME=${MTIME%%.*}
    LOCAL="$DEST/checkpoints/$REL"
    if [ -f "$LOCAL" ] && [ "$(stat -f %z "$LOCAL")" = "$SIZE" ] && [ "$(stat -f %m "$LOCAL")" = "$MTIME" ]; then continue; fi
    mkdir -p "$(dirname "$LOCAL")"
    if rsync -a -e "$RSYNC_E" "$HOST:/root/runs/$NAME/$REL" "$LOCAL"; then
      REMOTE_SHA=$("${SSH[@]}" "sha256sum /root/runs/$NAME/$REL" | awk '{print $1}')
      LOCAL_SHA=$(shasum -a 256 "$LOCAL" | awk '{print $1}')
      if [ "$REMOTE_SHA" = "$LOCAL_SHA" ]; then
        echo "  copied $REL $SIZE B mtime $MTIME sha256 $LOCAL_SHA (matches pod)"
        if [ "$(basename "$REL")" = checkpoint_last_best.pt ]; then
          H="$DEST/last_best-history/$MTIME-${LOCAL_SHA:0:12}.pt"
          [ -f "$H" ] || { cp -p "$LOCAL" "$H" && echo "  kept promoted last_best as $(basename "$H")"; }
        fi
      else
        # The file changed during the copy (or corruption): drop it and retry next pass.
        echo "  MISMATCH $REL pod=$REMOTE_SHA local=$LOCAL_SHA; removing local copy for retry"; rm -f "$LOCAL"
      fi
    else echo "  copy failed $REL"; fi
  done <<< "$LIST"
  (cd "$DEST" && { find checkpoints last_best-history -name '*.pt' -type f | sort | xargs -r shasum -a 256; shasum -a 256 "$NAME.log" 2>/dev/null; } > SHA256SUMS.tmp; mv SHA256SUMS.tmp SHA256SUMS)
  echo "$(date -u +%FT%TZ) pass end: $(grep -c '\.pt$' "$DEST/SHA256SUMS" 2>/dev/null) checkpoint files"
  [ "${ONCE:-0}" = 1 ] && break
  sleep "$INTERVAL"
done
