#!/usr/bin/env bash
# The 8-rank MAIN RUN (DRAFT; see run-statement-6.3b.md). Launch only after
# every 6.3b qualification check passed and the owner approved the pod and the
# run. Run on the pod:
#
#   bash ops/rebuild-2026-09-29/8rank-run/launch.sh OUT_DIR RUNTIME_HOURS [EXPERIMENT_ID]
#
# Environment:
#   KG_NATIVE_THREADS  native_threads chosen by the 6.3b sweep (default: the
#                      preset's 2); passed as -o env.native_threads=N.
#   KG_DURABLE_DIR     a network-volume mount for checkpoint copies (optional;
#                      without it, run pull_from_pod.sh on the Mac).
#   KG_BC_BEST, KG_RECEIPTS as in common.sh.
#
# It runs recipe J (configs/kaggriculture_8rank_bc_finetune.yaml: both LRs / 10,
# checkpoint_freq 10M) from the BC best with --load-model-weights-mode
# model_only and online W&B. RUNTIME_HOURS goes to run_ppo's
# --max-runtime-hours, which stops after a complete iteration and writes
# checkpoint_final.pt; an outer timeout (cap + 45 min) kills a hung run.
# watchdog.py copies checkpoints off the pod disk and applies the stop rules.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT=${1:?usage: launch.sh OUT_DIR RUNTIME_HOURS [EXPERIMENT_ID]}
HOURS=${2:?usage: launch.sh OUT_DIR RUNTIME_HOURS [EXPERIMENT_ID]}
EXPERIMENT=${3:-kg-v3-8rank-recipe-j-$(date -u +%Y%m%d)}
THREADS=${KG_NATIVE_THREADS:-2}
R=$KG_RECEIPTS_ROOT/main/$(basename "$OUT")
awk -v h="$HOURS" 'BEGIN { exit !(h > 0) }' || kg_fail "RUNTIME_HOURS must be > 0"
HARD_LIMIT=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 3600 + 2700 }')
mkdir -p "$R" "$OUT"
[ -z "$(ls -A "$OUT")" ] || kg_fail "$OUT is not empty; a resume is a different command (see the run statement)"

kg_preflight "$R"
cmd=("$KG_TORCHRUN" --nproc-per-node 8 scripts/run_ppo.py "$KG_PRESET" "$OUT"
  --load-model-weights "$KG_BC_BEST" --load-model-weights-mode model_only
  --log-mode wandb --wandb-mode online --experiment-id "$EXPERIMENT"
  --max-runtime-hours "$HOURS" -o "env.native_threads=$THREADS")
printf '%q ' "${cmd[@]}" > "$R/command.txt"
echo >> "$R/command.txt"
{
  echo "experiment_id=$EXPERIMENT"
  echo "runtime_hours=$HOURS hard_limit_s=$HARD_LIMIT native_threads=$THREADS"
  echo "durable_dir=${KG_DURABLE_DIR:-none}"
} > "$R/launch.txt"
kg_sampler_start "$R" "$HARD_LIMIT" 10000

date -u +"launch %FT%TZ" > "$R/times.txt"
START=$(date +%s)
# setsid gives torchrun and its workers one process group for the watchdog.
setsid timeout --kill-after=120 "$HARD_LIMIT" "${cmd[@]}" > "$R/run.log" 2>&1 &
PGID=$!
echo "pgid=$PGID" >> "$R/launch.txt"

watchdog_args=(--out "$OUT" --pgid "$PGID" --receipts "$R")
[ -n "${KG_DURABLE_DIR:-}" ] && watchdog_args+=(--durable-dir "$KG_DURABLE_DIR")
"$KG_PY" "$KG_PKG/watchdog.py" "${watchdog_args[@]}" > "$R/watchdog.log" 2>&1 &
WATCHDOG=$!

set +e
wait "$PGID"
STATUS=$?
wait "$WATCHDOG"
WATCHDOG_STATUS=$?
set -e
echo "exit=$STATUS watchdog_exit=$WATCHDOG_STATUS wall_s=$(( $(date +%s) - START ))" >> "$R/times.txt"
date -u +"end %FT%TZ" >> "$R/times.txt"
kg_post "$R" "$OUT"
RUN_DIR=$(dirname "$(ls -t "$OUT"/*/attempts.jsonl | head -1)")
"$KG_PY" "$KG_PKG/summarize_run.py" "$RUN_DIR" --out "$R" > /dev/null \
  || echo "summary failed" >> "$R/times.txt"
sha256sum "$R"/* > "$R.SHA256SUMS" 2>/dev/null || true
cat "$R/times.txt"
[ -f "$R/watchdog_stop.json" ] && cat "$R/watchdog_stop.json"
exit "$STATUS"
