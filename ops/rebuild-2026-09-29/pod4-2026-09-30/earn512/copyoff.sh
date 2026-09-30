#!/bin/bash
# Mac copy-off loop for scratch-bank-4rank (from-scratch mirror self-play, own-bank reward). Every 10 min: copy new or changed
# checkpoint_*.pt files (finished files only; the trainer writes .tmp then
# renames), the run log, the watchdog log and the pod receipts from pod
# abl4mvr5w1mmn4 to $DEST, check each new .pt against the pod's sha256, and
# rewrite $DEST/SHA256SUMS. Run: nohup copyoff.sh >> $DEST/copyoff.log 2>&1 &
set -u
DEST=/Users/poonszesen/kg-v3-runs/earn512-from-promoted-4rank-20260930
NAME=earn512-from-promoted-4rank-20260930
KEY_OPTS=(-i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes -o UserKnownHostsFile="$HOME/.ssh/known_hosts.runpod" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)
HOST=root@157.157.221.177
PORT=11014
SSH=(ssh "${KEY_OPTS[@]}" -p $PORT $HOST)
RSYNC_E="ssh ${KEY_OPTS[*]} -p $PORT"
mkdir -p "$DEST/checkpoints" "$DEST/receipts"
while true; do
  echo "$(date -u +%FT%TZ) pass start"
  rsync -a --partial -e "$RSYNC_E" "$HOST:/root/runs/$NAME.log" "$HOST:/root/runs/earn512-watchdog.log" "$DEST/" 2>&1 | sed 's/^/  rsync-log: /'
  rsync -a -e "$RSYNC_E" "$HOST:/root/receipts/$NAME/" "$DEST/receipts/" 2>&1 | sed 's/^/  rsync-receipts: /'
  # Finished checkpoints only: name checkpoint_*.pt, not a dotfile, unchanged for 60 s.
  LIST=$("${SSH[@]}" "cd /root/runs/$NAME 2>/dev/null && find . -name 'checkpoint_*.pt' -mmin +1 -printf '%P\t%s\n'" 2>&1) || { echo "  list failed: $LIST"; LIST=""; }
  while IFS=$'\t' read -r REL SIZE; do
    [ -z "$REL" ] && continue
    LOCAL="$DEST/checkpoints/$REL"
    if [ -f "$LOCAL" ] && [ "$(stat -f %z "$LOCAL")" = "$SIZE" ]; then continue; fi
    mkdir -p "$(dirname "$LOCAL")"
    if rsync -a -e "$RSYNC_E" "$HOST:/root/runs/$NAME/$REL" "$LOCAL"; then
      REMOTE_SHA=$("${SSH[@]}" "sha256sum /root/runs/$NAME/$REL" | awk '{print $1}')
      LOCAL_SHA=$(shasum -a 256 "$LOCAL" | awk '{print $1}')
      if [ "$REMOTE_SHA" = "$LOCAL_SHA" ]; then echo "  copied $REL $SIZE B sha256 $LOCAL_SHA (matches pod)"
      else echo "  MISMATCH $REL pod=$REMOTE_SHA local=$LOCAL_SHA; removing local copy for retry"; rm -f "$LOCAL"; fi
    else echo "  copy failed $REL"; fi
  done <<< "$LIST"
  (cd "$DEST" && find checkpoints -name '*.pt' -type f | sort | xargs -r shasum -a 256 > SHA256SUMS.tmp; shasum -a 256 "$NAME.log" >> SHA256SUMS.tmp 2>/dev/null; mv SHA256SUMS.tmp SHA256SUMS)
  echo "$(date -u +%FT%TZ) pass end: $(grep -c '\.pt$' "$DEST/SHA256SUMS" 2>/dev/null) checkpoints"
  sleep 600
done
