# Shared helpers for the 8-rank run package (sourced by qualify.sh and
# launch.sh on the pod; DRAFT). Nothing here launches training.
set -euo pipefail

KG_REPO=$(git rev-parse --show-toplevel)
KG_PKG=$KG_REPO/ops/rebuild-2026-09-29/8rank-run
KG_BC_BEST=${KG_BC_BEST:-/workspace/bc-best/checkpoint_bc_best.pt}
KG_BC_SHA256=fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51
KG_RECEIPTS_ROOT=${KG_RECEIPTS:-/workspace/kg-v3-receipts}
KG_PRESET=configs/kaggriculture_8rank_bc_finetune.yaml
KG_PY=$KG_REPO/.venv/bin/python
KG_TORCHRUN=$KG_REPO/.venv/bin/torchrun
export OMP_NUM_THREADS=1
export WANDB_ENTITY=${WANDB_ENTITY:-spoon}
cd "$KG_REPO"

kg_fail() { printf 'STOP: %s\n' "$1" >&2; exit 1; }

# kg_preflight RECEIPT_DIR: clean tree, idle GPUs, BC best, credential, hashes.
kg_preflight() {
  local r=$1
  mkdir -p "$r"
  { git rev-parse HEAD; git status --porcelain | wc -l; } > "$r/git_state.txt"
  [ "$(sed -n 2p "$r/git_state.txt" | tr -d ' ')" = 0 ] \
    || kg_fail "the checkout has uncommitted changes; launch from a committed tree"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,driver_version --format=csv > "$r/idle_prelaunch.csv"
  nvidia-smi --query-compute-apps=pid,name --format=csv,noheader > "$r/compute_apps_prelaunch.txt"
  [ ! -s "$r/compute_apps_prelaunch.txt" ] || kg_fail "GPUs are busy: $(cat "$r/compute_apps_prelaunch.txt")"
  echo "$KG_BC_SHA256  $KG_BC_BEST" | sha256sum -c - > "$r/bc_best.check" \
    || kg_fail "BC best SHA-256 mismatch at $KG_BC_BEST"
  [ "$(stat -c %a "$HOME/.netrc")" = 600 ] || kg_fail "~/.netrc missing or not mode 600"
  [ -z "${WANDB_MODE:-}" ] || kg_fail "WANDB_MODE is set; unset it"
  sha256sum "$KG_PRESET" configs/model/kaggriculture.yaml scripts/run_ppo.py \
    python/owl/train/ppo.py python/owl/train/logging.py python/owl/kaggriculture/env.py \
    python/owl/rs.abi3.so "$KG_BC_BEST" "$KG_PKG"/*.sh "$KG_PKG"/*.py > "$r/hashes.sha256"
  { nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true; } > "$r/vcpus.txt"
}

# kg_sampler_start RECEIPT_DIR SECONDS INTERVAL_MS: bounded nvidia-smi sampler.
kg_sampler_start() {
  timeout "$2" nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv -lms "$3" > "$1/nvsmi_samples.csv" 2>&1 &
  KG_SAMPLER_PID=$!
}

# kg_post RECEIPT_DIR OUT_DIR: run listing, checkpoint hashes, idle check, peaks.
kg_post() {
  local r=$1 out=$2
  kill "${KG_SAMPLER_PID:-0}" 2>/dev/null || true
  ls -laR "$out" > "$r/run_dir_listing.txt" 2>&1 || true
  find "$out" -type f -name 'checkpoint_*.pt' -exec sha256sum {} \; > "$r/checkpoints.sha256" 2>&1 || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$r/idle_after.csv"
  # Peak nvidia-smi memory per GPU (MiB). Includes the allocator's reserve, so it
  # bounds the allocated peak from above.
  awk -F', ' 'NR > 1 { split($3, m, " "); if (m[1] + 0 > peak[$2]) peak[$2] = m[1] + 0 }
    END { for (g in peak) print "gpu " g " peak_used_MiB " peak[g] }' \
    "$r/nvsmi_samples.csv" | sort -n -k2 > "$r/peak_memory.txt" || true
}
