#!/bin/bash
# Stop a sprint run started by launch.sh:  bash /root/sprint-kit/stop.sh NAME [REASON]
# Reads the real pgid from /root/receipts/NAME/pgid, sends SIGTERM to that process
# group and every descendant (torchrun gives each rank its own session, so the
# group alone misses the ranks), waits up to 60 s, then SIGKILLs what is left,
# stops the watchdog and records it all in /root/receipts/NAME/stop.txt.
# Checkpoints already written stay in /root/runs/NAME; the Mac copy-off keeps running
# until you stop it (mac_side.md step 8).
set -euo pipefail
NAME=${1:?usage: stop.sh NAME [REASON]}
REASON=${2:-operator stop}
R=/root/receipts/$NAME
[ -s "$R/pgid" ] || { echo "no $R/pgid: was $NAME started by launch.sh?" >&2; exit 2; }
PGID=$(cat "$R/pgid")
[[ "$PGID" =~ ^[0-9]+$ ]] || { echo "bad pgid '$PGID'" >&2; exit 2; }

run_pids() {
  ps -e -o pid=,ppid=,pgid= | awk -v g="$PGID" '
    { pid[NR]=$1; ppid[NR]=$2; pg[NR]=$3; n=NR }
    END {
      for (i = 1; i <= n; i++) if (pg[i] == g) s[pid[i]] = 1
      grew = 1
      while (grew) { grew = 0
        for (i = 1; i <= n; i++) if ((ppid[i] in s) && !(pid[i] in s)) { s[pid[i]] = 1; grew = 1 }
      }
      for (p in s) print p
    }'
}

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/stop.txt"; }
PIDS=$(run_pids | sort -n | tr '\n' ' ')
if [ -z "$PIDS" ]; then log "nothing running for pgid $PGID"; else
  log "STOP $NAME reason='$REASON' pgid=$PGID pids=[$PIDS]"
  kill -TERM -- "-$PGID" 2>/dev/null || true
  # shellcheck disable=SC2086
  kill -TERM $PIDS 2>/dev/null || true
  for _ in $(seq 1 60); do [ -z "$(run_pids)" ] && break; sleep 1; done
  LEFT=$(run_pids | sort -n | tr '\n' ' ')
  if [ -n "$LEFT" ]; then
    log "still alive after 60 s; SIGKILL [$LEFT]"
    kill -KILL -- "-$PGID" 2>/dev/null || true
    # shellcheck disable=SC2086
    kill -KILL $LEFT 2>/dev/null || true
  else
    log "run exited after SIGTERM"
  fi
fi
if [ -s "$R/watchdog.pid" ] && kill -0 "$(cat "$R/watchdog.pid")" 2>/dev/null; then
  kill -TERM "$(cat "$R/watchdog.pid")" && log "watchdog $(cat "$R/watchdog.pid") stopped"
fi
sleep 2
log "gpu compute apps now: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
