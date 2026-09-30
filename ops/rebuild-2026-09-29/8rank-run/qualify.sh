#!/usr/bin/env bash
# Plan 6.3b 8-rank qualification steps (DRAFT). Run on the pod after setup.sh,
# one step at a time, in the order of run-statement-6.3b.md. Each step writes
# receipts to $KG_RECEIPTS/6.3b/<step> and runs from the BC best with the
# recipe-J preset and --load-model-weights-mode model_only, W&B online.
#
#   bash ops/rebuild-2026-09-29/8rank-run/qualify.sh seeds
#   bash ops/rebuild-2026-09-29/8rank-run/qualify.sh memory-smoke
#   bash ops/rebuild-2026-09-29/8rank-run/qualify.sh threads-sweep            # 2, 4, 8 (8 only with >= 72 vCPUs)
#   bash ops/rebuild-2026-09-29/8rank-run/qualify.sh allreduce
#   bash ops/rebuild-2026-09-29/8rank-run/qualify.sh complete-work MINUTES THREADS [nsys]
#
# Nothing here is the main run (launch.sh). Each launch has a hard timeout.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STEP=${1:?usage: qualify.sh seeds|memory-smoke|threads-sweep|allreduce|complete-work}
R=$KG_RECEIPTS_ROOT/6.3b/$STEP
RUNS=${KG_RUNS:-/workspace/kg-v3-runs/6.3b}
ITER=16384
mkdir -p "$R" "$RUNS"
KG_LAUNCHER=("$KG_TORCHRUN")
RC=0

# run_ppo_8 OUT SECONDS EXPERIMENT_ID [extra run_ppo args...]
run_ppo_8() {
  local out=$1 limit=$2 experiment=$3
  shift 3
  date -u +"launch %FT%TZ" >> "$R/times.txt"
  local start
  start=$(date +%s)
  set +e
  timeout --kill-after=120 "$limit" "${KG_LAUNCHER[@]}" --nproc-per-node 8 scripts/run_ppo.py \
    "$KG_PRESET" "$out" \
    --load-model-weights "$KG_BC_BEST" --load-model-weights-mode model_only \
    --log-mode wandb --wandb-mode online --experiment-id "$experiment" "$@" \
    > "$R/run-$(basename "$out").log" 2>&1
  local status=$?
  set -e
  echo "$(basename "$out") exit=$status wall_s=$(( $(date +%s) - start ))" >> "$R/times.txt"
  return $status
}

latest_run_dir() { dirname "$(ls -t "$1"/*/attempts.jsonl | head -1)"; }

case "$STEP" in
  seeds)
    # CPU: the probe at world size 8 plus the native seed-partition tests.
    "$KG_PY" "$KG_PKG/seed_probe.py" "$KG_PRESET" --world-size 8 | tee "$R/seed_probe.json"
    "$KG_PY" -m pytest -q tests/kaggriculture/test_native_env.py -k seed_partition \
      2>&1 | tail -3 | tee "$R/seed_tests.log"
    ;;
  memory-smoke)
    # One full iteration with the teacher on and a forced last-best evaluation
    # (checkpoint_freq = one iteration), as plan 6.1 did at 2 ranks. The
    # evaluation starts from game start, not from dense BC positions, and the
    # rollout and update see only turns 0-63; complete-work's nvidia-smi peak
    # covers late-game iterations (see run-statement-6.3b.md, step 2).
    kg_preflight "$R"
    kg_sampler_start "$R" 1500 1000
    run_ppo_8 "$RUNS/memory-smoke" 1320 kg-v3-6.3b-memory-smoke \
      --max-env-steps "$ITER" -o "rl.checkpoint_freq=$ITER" || RC=$?
    kg_post "$R" "$RUNS/memory-smoke"
    cat "$R/peak_memory.txt"
    ;;
  threads-sweep)
    # native_threads 2 / 4 / 8 at 32 envs per rank: 12 iterations each (one
    # 720-step game spans about 11.25 iterations), same seed. Compare
    # time/rollout_seconds and complete-work SPS over iterations 2-12.
    # 8 threads needs 8 x 8 native threads + 8 trainer processes = 72 vCPUs;
    # with fewer (nproc; kg_preflight also records cgroup cpu.max) it is skipped.
    kg_preflight "$R"
    vcpus=$(nproc)
    sweep=(2 4 8)
    if [ "$vcpus" -lt 72 ]; then
      sweep=(2 4)
      echo "skipped native_threads 8: nproc $vcpus < 72 (8 x 8 + 8)" | tee -a "$R/times.txt"
    fi
    kg_sampler_start "$R" 3600 2000
    for threads in "${sweep[@]}"; do
      out=$RUNS/threads-$threads
      run_ppo_8 "$out" 1200 kg-v3-6.3b-threads \
        --max-env-steps $(( 12 * ITER )) -o "env.native_threads=$threads" || { RC=$?; break; }
      "$KG_PY" "$KG_PKG/summarize_run.py" "$(latest_run_dir "$out")" --out "$R/threads-$threads" \
        || echo "summary failed for threads-$threads" >> "$R/times.txt"
    done
    kg_post "$R" "$RUNS"
    ;;
  allreduce)
    kg_preflight "$R"
    nvidia-smi topo -m > "$R/topo.txt"
    "$KG_TORCHRUN" --nproc-per-node 8 "$KG_PKG/allreduce_bench.py" --out "$R" \
      2>&1 | tail -20 | tee "$R/allreduce.log"
    ;;
  complete-work)
    minutes=${2:?complete-work MINUTES THREADS [nsys]}
    threads=${3:?complete-work MINUTES THREADS [nsys]}
    hours=$(awk -v m="$minutes" 'BEGIN { printf "%.4f", m / 60 }')
    kg_preflight "$R"
    kg_sampler_start "$R" $(( minutes * 60 + 900 )) 2000
    extra=(--max-runtime-hours "$hours" -o "env.native_threads=$threads")
    if [ "${4:-}" = nsys ]; then
      # Canonical timeline profiler: capture a bounded post-warm-up window
      # (starting after about 5 minutes, 60 s long) across all ranks.
      # --kill=none keeps the learner running when the capture window ends
      # (the profiling workflow never kills a learner for profiling);
      # --wait=all is added only where this nsys offers it.
      command -v nsys >/dev/null || kg_fail "nsys requested but absent"
      nsys --version > "$R/nsys_version.txt"
      nsys profile --help > "$R/nsys_profile_help.txt" 2>&1 || true
      grep -q -- '--kill' "$R/nsys_profile_help.txt" \
        || kg_fail "this nsys profile has no --kill option; see $R/nsys_profile_help.txt"
      nsys_wait=()
      grep -q -- '--wait' "$R/nsys_profile_help.txt" && nsys_wait=(--wait=all)
      KG_LAUNCHER=(nsys profile --trace=cuda,nvtx,osrt --delay=300 --duration=60
        --kill=none "${nsys_wait[@]}"
        --trace-fork-before-exec=true "--output=$R/complete-work" --force-overwrite=true
        "$KG_TORCHRUN")
    fi
    run_ppo_8 "$RUNS/complete-work" $(( minutes * 60 + 600 )) kg-v3-6.3b-complete-work \
      "${extra[@]}" || RC=$?
    kg_post "$R" "$RUNS/complete-work"
    "$KG_PY" "$KG_PKG/summarize_run.py" "$(latest_run_dir "$RUNS/complete-work")" --out "$R" \
      || echo "summary failed" >> "$R/times.txt"
    if [ -f "$R/complete-work.nsys-rep" ]; then
      nsys stats --report cuda_gpu_kern_sum --format csv --output "$R/kern_sum" \
        "$R/complete-work.nsys-rep" > /dev/null
      grep -i nccl "$R"/kern_sum*.csv > "$R/nccl_kernels.csv" || true
    fi
    ;;
  *)
    kg_fail "unknown step $STEP"
    ;;
esac
sha256sum "$R"/* > "$R.SHA256SUMS" 2>/dev/null || true
[ "$RC" = 0 ] || kg_fail "6.3b $STEP: run_ppo exited $RC; see $R"
echo "6.3b $STEP done; receipts in $R"
