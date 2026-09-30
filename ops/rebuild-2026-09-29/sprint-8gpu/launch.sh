#!/bin/bash
# Launch the current recipe (live run pcy5knet) on N GPUs of the sprint pod.
#
#   bash /root/sprint-kit/launch.sh --checkpoint /root/start/checkpoint_last_best.pt \
#        --name earn720-lr1e4-8gpu-sprint-YYYYMMDD [--gpus 8] [--n-envs 12] \
#        [--repo /root/kg-v3] [--numa cpu|1|off] [--extra 'k=v k2=v2'] [--allow-tight-memory]
#        [--allow-recipe-drift] [--dry-run]
#
# CHECKPOINT, NAME and N_GPUS may also come from the environment; flags win.
# Recipe (unchanged from pcy5knet except the world size): configs/kaggriculture_4rank_margin.yaml
# with -o env.n_envs=12 rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0 and reward
# econ_bank 0.25/150000/0.25, econ_margin 0.25/100000/0.25 (econ_shaping 0 in the preset);
# Muon 1e-4 / AdamW 5e-6, native_threads 4 and the 10M checkpoint/eval come from the preset;
# --load-model-weights CHECKPOINT --load-model-weights-mode model_only; W&B online (spoon/kg-v3).
# The "4rank" preset name is only its origin: run_ppo takes the world size from torchrun.
# --extra appends further -o overrides (a recipe change: record the owner's yes).
# Source pin: refuses unless sha256 of the preset is b7fa7f9d57363869... (the preset of pcy5knet and of
# the pinned sprint source 07c8fc99); --allow-recipe-drift overrides (owner's yes). A HEAD other
# than 07c8fc99 is warned about and recorded.
#
# Effects: /root/sprint/NAME/ (frozen copies of main_probe_auto.py, watchdog.py and the
# generated run.sh), /root/receipts/NAME/ (receipts), /root/runs/NAME{,.log,-watchdog.log}.
# Refuses to reuse a NAME. --dry-run prints the resolved plan and command, touches nothing.
set -euo pipefail

KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT=${CHECKPOINT:-}
NAME=${NAME:-}
N_GPUS=${N_GPUS:-8}
N_ENVS=12
REPO=/root/kg-v3
NUMA=cpu
EXTRA=""
DRY_RUN=0
ALLOW_TIGHT=0
NATIVE_THREADS=4   # the preset's env.native_threads; used for the topology report only
CFG=configs/kaggriculture_4rank_margin.yaml
# sha256 of $CFG in the live run pcy5knet's receipts (earn720-lr1e4-from-promoted2-4rank-20260930/
# receipts/hashes.sha256) and at the pinned sprint source 07c8fc99.
EXPECTED_CFG_SHA256=b7fa7f9d57363869546a076b20184d16fd97583c38c8541dbb8d5789e520fcae
PINNED_SHA=07c8fc9972297f28e0414da752700194a90d4fc3
ALLOW_DRIFT=0

usage() { sed -n '2,23p' "$0"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --name) NAME=${2:?}; shift 2 ;;
    --gpus) N_GPUS=${2:?}; shift 2 ;;
    --n-envs) N_ENVS=${2:?}; shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --numa) NUMA=${2:?}; shift 2 ;;
    --extra) EXTRA=${2?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-tight-memory) ALLOW_TIGHT=1; shift ;;
    --allow-recipe-drift) ALLOW_DRIFT=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

err() { echo "launch.sh: $*" >&2; exit 2; }
[ -n "$CHECKPOINT" ] || err "--checkpoint (or CHECKPOINT) is required"
[[ "$CHECKPOINT" = /* ]] || err "--checkpoint must be an absolute path (run.sh runs from the repo), got '$CHECKPOINT'"
[ -n "$NAME" ] || err "--name (or NAME) is required"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || err "NAME may hold only letters, digits, . _ - (got '$NAME')"
[[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]] || err "--gpus must be a positive integer (got '$N_GPUS')"
[[ "$N_ENVS" =~ ^[1-9][0-9]*$ ]] || err "--n-envs must be a positive integer (got '$N_ENVS')"
case "$NUMA" in cpu|1|off) ;; *) err "--numa must be cpu, 1 or off (got '$NUMA')" ;; esac
read -r -a EXTRA_ARR <<< "$EXTRA"
for kv in "${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}"; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_.]*=[^[:space:]]+$ ]] || err "--extra item '$kv' is not key=value"
done
EAGER=0
for kv in "${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}"; do [ "$kv" = rl.model_compile=none ] && EAGER=1; done

R=/root/receipts/$NAME
RUN=/root/runs/$NAME
LOG=/root/runs/$NAME.log
WLOG=/root/runs/$NAME-watchdog.log
FROZEN=/root/sprint/$NAME
OVERRIDES=(env.n_envs="$N_ENVS" rl.horizon=720 rl.segments_per_minibatch=1 rl.gae_lambda=1.0
  env.reward_shaping.econ_bank_weight=0.25 env.reward_shaping.econ_bank_scale=150000 env.reward_shaping.econ_bank_cap=0.25
  env.reward_shaping.econ_margin_weight=0.25 env.reward_shaping.econ_margin_scale=100000 env.reward_shaping.econ_margin_cap=0.25
  "${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}")
CMD=(.venv/bin/torchrun --nproc-per-node "$N_GPUS" "$FROZEN/main_probe_auto.py" scripts/run_ppo.py "$CFG" "$RUN"
  --log-mode wandb --wandb-mode online --experiment-id "$NAME"
  --load-model-weights "$CHECKPOINT" --load-model-weights-mode model_only -o "${OVERRIDES[@]}")
ENV_EXPORTS=(OMP_NUM_THREADS=1 WANDB_ENTITY=spoon KG_NT_NUMA="$NUMA" KG_NT_NATIVE_THREADS="$NATIVE_THREADS"
  CUDA_DEVICE_ORDER=PCI_BUS_ID TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton)

plan() {
  echo "name=$NAME gpus=$N_GPUS n_envs=$N_ENVS global_games_per_iteration=$((N_GPUS * N_ENVS)) env_steps_per_iteration=$((720 * N_ENVS * N_GPUS))"
  echo "optimizer_steps_per_iteration=$N_ENVS (each over $N_GPUS game-segments, one per rank) numa=$NUMA eager=$EAGER"
  echo "checkpoint=$CHECKPOINT repo=$REPO pinned_source=$PINNED_SHA expected_preset_sha256=$EXPECTED_CFG_SHA256 allow_recipe_drift=$ALLOW_DRIFT"
  echo "run_dir=$RUN log=$LOG watchdog_log=$WLOG receipts=$R frozen=$FROZEN"
  echo "env: ${ENV_EXPORTS[*]}"
  printf 'cmd (in %s):' "$REPO"; printf ' %q' "${CMD[@]}"; echo
}

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY RUN (nothing checked on the host, nothing written)"
  plan
  exit 0
fi

# ---- pre-launch checks (fail fast, nothing written yet) ----
fail() { echo "launch.sh REFUSED: $*" >&2; exit 3; }
[ ! -e "$R" ] && [ ! -e "$RUN" ] && [ ! -e "$LOG" ] && [ ! -e "$FROZEN" ] || fail "NAME $NAME already used (one of $R $RUN $LOG $FROZEN exists); pick a new --name"
[ -x "$REPO/.venv/bin/torchrun" ] || fail "$REPO/.venv/bin/torchrun missing; run bootstrap.sh first"
[ -f "$REPO/$CFG" ] || fail "$REPO/$CFG missing"
CFG_SHA256=$(sha256sum "$REPO/$CFG" | awk '{print $1}')
if [ "$CFG_SHA256" != "$EXPECTED_CFG_SHA256" ]; then
  [ "$ALLOW_DRIFT" = 1 ] || fail "preset $CFG sha256 $CFG_SHA256 differs from pcy5knet's $EXPECTED_CFG_SHA256 (source $(git -C "$REPO" rev-parse --short HEAD), pinned 07c8fc99); bootstrap at the pinned --sha, or pass --allow-recipe-drift with the owner's yes"
  echo "WARNING: preset $CFG sha256 $CFG_SHA256 != $EXPECTED_CFG_SHA256 (--allow-recipe-drift given)"
fi
HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
[ "$HEAD_SHA" = "$PINNED_SHA" ] || echo "WARNING: $REPO HEAD $HEAD_SHA is not the pinned sprint source $PINNED_SHA; code outside the preset may differ (review the diff against 07c8fc99 before launching)"
[ -r "$CHECKPOINT" ] || fail "checkpoint $CHECKPOINT not readable"
for f in main_probe_auto.py watchdog.py stop.sh; do [ -f "$KIT/$f" ] || fail "kit file $KIT/$f missing"; done
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || fail "$REPO has tracked changes; the run must start from a clean commit"
[ -f /root/.netrc ] && [ "$(stat -c %a /root/.netrc)" = 600 ] || fail "/root/.netrc missing or not mode 600 (mac_side.md step 3)"
[ -z "${WANDB_MODE:-}" ] || fail "WANDB_MODE is set ($WANDB_MODE); unset it"
HOST_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$HOST_GPUS" -ge "$N_GPUS" ] || fail "--gpus $N_GPUS but the host has $HOST_GPUS GPU(s)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || fail "GPUs busy (nvidia-smi lists compute apps)"
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
MEM_GIB=$((MEM_MIB / 1024))
# ~5 GB per env at horizon 720 plus ~10 GB base (12 envs: 68.5 GB measured on 96 GB GPUs,
# 16 envs: 90.96 GB, judged too tight for rank 0's 10M evaluation). Keep >= ~25 GB headroom.
EST=$((10 + 5 * N_ENVS)); FIT=$(( (MEM_GIB - 35) / 5 ))
echo "GPU memory ${MEM_GIB} GiB; estimated use at n_envs=$N_ENVS ~${EST} GiB; n_envs with >=25 GiB headroom: <= $FIT"
[ "$EST" -le "$((MEM_GIB - 5))" ] || fail "n_envs=$N_ENVS needs ~${EST} GiB of ${MEM_GIB} GiB; use --n-envs $FIT (a batch change: tell the owner)"
if [ "$N_ENVS" -gt "$FIT" ]; then
  [ "$ALLOW_TIGHT" = 1 ] || fail "n_envs=$N_ENVS leaves < 25 GiB headroom (16 envs at 91 of 96 GB was stopped as too tight); use --n-envs $FIT (tell the owner) or pass --allow-tight-memory"
  echo "WARNING: n_envs=$N_ENVS leaves < 25 GiB headroom (--allow-tight-memory given)"
fi
cd "$REPO"
if [ "$EAGER" = 0 ]; then
  .venv/bin/python -c 'from owl.model.compile_gemm import check_compile_stack as c, installed_compile_stack as i; r=c(i()); print("compile stack ok:", r.torch, r.triton, r.nvidia_driver)' \
    || fail "compile-stack gate refused (see bootstrap.sh step 3 options A/B/C)"
fi
if [ "$NUMA" = off ]; then
  TOPO=$(KG_NT_NATIVE_THREADS=$NATIVE_THREADS .venv/bin/python "$KIT/main_probe_auto.py" --topology "$N_GPUS" 2>/dev/null) \
    || TOPO='{"error": "topology discovery failed; --numa off, so the ranks run unbound"}'
else
  TOPO=$(KG_NT_NATIVE_THREADS=$NATIVE_THREADS .venv/bin/python "$KIT/main_probe_auto.py" --topology "$N_GPUS") \
    || fail "topology discovery failed; fix it or relaunch with --numa off"
  if echo "$TOPO" | grep -q '"oversubscribed": true'; then
    echo "WARNING: a NUMA node has fewer CPUs than ranks x (native_threads + 2); see receipts topology.json"
  fi
fi

# ---- freeze the kit and write the run script ----
mkdir -p "$R" "$FROZEN" /root/runs /root/sweep-cache/inductor /root/sweep-cache/triton
cp "$KIT/main_probe_auto.py" "$KIT/watchdog.py" "$FROZEN/"
echo "$TOPO" > "$R/topology.json"
{
  echo '#!/bin/bash'
  echo "# Generated by $KIT/launch.sh at $(date -u +%FT%TZ). Session leader: writes its pgid first."
  echo 'set -u'
  printf 'NAME=%q R=%q RUN=%q REPO=%q SRC=%q CFG=%q\n' "$NAME" "$R" "$RUN" "$REPO" "$CHECKPOINT" "$CFG"
  echo 'ps -o pgid= $$ | tr -d " " > "$R/pgid.tmp" && mv "$R/pgid.tmp" "$R/pgid"'
  echo 'trap '"'"'echo "SIGTERM $(date -u +%FT%TZ)" >> "$R/times.txt"'"'"' TERM'
  echo 'cd "$REPO"'
  printf '{ git rev-parse HEAD; git status --porcelain --untracked-files=no | wc -l; echo "name=$NAME gpus=%s n_envs=%s numa=%s eager=%s pgid=$(cat "$R/pgid")"; } > "$R/git_state.txt"\n' "$N_GPUS" "$N_ENVS" "$NUMA" "$EAGER"
  printf 'sha256sum "$CFG" configs/model/kaggriculture.yaml scripts/run_ppo.py python/owl/train/ppo.py python/owl/kaggriculture/env.py python/owl/kaggriculture/rewards.py python/owl/model/compile_gemm.py src/kaggriculture/reward.rs src/kaggriculture/env.rs python/owl/rs.abi3.so uv.lock Cargo.lock "$SRC" %q %q "$0" > "$R/hashes.sha256"\n' "$FROZEN/main_probe_auto.py" "$FROZEN/watchdog.py"
  echo 'nvidia-smi > "$R/nvidia-smi.txt"'
  echo 'nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version --format=csv > "$R/idle_prelaunch.csv"'
  echo 'nvidia-smi --query-compute-apps=pid,name --format=csv >> "$R/idle_prelaunch.csv"'
  echo 'if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ]; then echo "GPU busy; refusing" | tee -a "$R/times.txt"; exit 3; fi'
  echo 'timeout 900 nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 2000 > "$R/nvsmi_early.csv" 2>&1 < /dev/null &'
  echo 'nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 60 > "$R/nvsmi_60s.csv" 2>&1 < /dev/null &'
  echo 'SAMPLER=$!'
  printf 'export'; printf ' %q' "${ENV_EXPORTS[@]}"; echo
  echo 'env | grep -E "^(OMP_NUM_THREADS|WANDB_ENTITY|KG_NT_|CUDA_DEVICE_ORDER|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR)=" | sort > "$R/env.txt"'
  echo 'date -u +"launch %FT%TZ" > "$R/times.txt"'
  echo 'S=$(date +%s)'
  printf '%q' "${CMD[0]}"; printf ' %q' "${CMD[@]:1}"; echo
  echo 'echo "exit=$? wall_s=$(( $(date +%s) - S ))" >> "$R/times.txt"'
  echo 'date -u +"end %FT%TZ" >> "$R/times.txt"'
  echo 'kill $SAMPLER 2>/dev/null'
  echo 'ls -laR "$RUN" > "$R/run_dir_listing.txt" 2>&1'
  echo 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$R/idle_after.csv"'
  echo 'echo DONE >> "$R/times.txt"'
} > "$FROZEN/run.sh"
chmod +x "$FROZEN/run.sh"
bash -n "$FROZEN/run.sh" || fail "generated $FROZEN/run.sh does not parse"
{ echo "launch.sh $(date -u +%FT%TZ)"; plan; echo "kit=$KIT head=$HEAD_SHA preset_sha256=$CFG_SHA256"; } > "$R/launch.txt"
sha256sum "$KIT"/*.py "$KIT"/*.sh "$FROZEN"/* >> "$R/launch.txt"

# ---- launch as a session leader; read the REAL pgid from the file it writes ----
setsid nohup "$FROZEN/run.sh" > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 60); do [ -s "$R/pgid" ] && break; sleep 1; done
[ -s "$R/pgid" ] || fail "run.sh wrote no pgid in 60 s; check $LOG (the run may still be starting: inspect before relaunching)"
PGID=$(cat "$R/pgid")
[[ "$PGID" =~ ^[0-9]+$ ]] || fail "bad pgid '$PGID' in $R/pgid"
[ "$(ps -o pgid= -p "$PGID" | tr -d ' ')" = "$PGID" ] || fail "process $PGID is not its own group leader (run exited already? see $LOG)"

setsid nohup "$REPO/.venv/bin/python" "$FROZEN/watchdog.py" "$PGID" "$LOG" > "$WLOG" 2>&1 < /dev/null &
WPID=$!
echo "$WPID" > "$R/watchdog.pid"
sleep 2
kill -0 "$WPID" 2>/dev/null || fail "watchdog exited at once; see $WLOG (run pgid $PGID is still going: stop it or restart the watchdog)"
{ echo "pgid=$PGID watchdog_pid=$WPID"; echo "stop: bash $KIT/stop.sh $NAME"; } | tee -a "$R/launch.txt"

cat <<EOF

LAUNCHED $NAME
  pgid $PGID (launcher session; torchrun ranks are its descendants), watchdog pid $WPID
  log       tail -f $LOG
  watchdog  tail -f $WLOG
  check     python3 $KIT/first_iters.py $LOG $N_GPUS $N_ENVS   (after ~3 iterations)
  stop      bash $KIT/stop.sh $NAME      (SIGTERM to the group and all descendants, SIGKILL after 60 s)
EOF
