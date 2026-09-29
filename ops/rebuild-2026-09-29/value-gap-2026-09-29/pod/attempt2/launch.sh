#!/bin/bash
# Value-gap diagnostic launcher (run statement run-statements/value-gap-diagnostic.md).
# Attempt 2 (Amendment 1). Idle gate on GPUs 0 and 1, receipts, the driver under
# a hard 42-min timeout (45-min aggregate with attempt 1's 96 s; internal deadline 40 min), post receipts. Never
# stops, restarts or deletes the pod.
set -u
R=/workspace/kg-v3-rebuild/runs/value-gap-2026-09-29
REPO=/workspace/kg-v3-rebuild
RC=$R/receipts
mkdir -p "$RC"
cd "$REPO"

gpu_idle() {  # $1 = index
  local apps util
  apps=$(nvidia-smi -i "$1" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  util=$(nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  [ -z "$apps" ] && [ "$util" = "0" ]
}

snap() {  # $1 = tag
  { date -u +%FT%TZ; nvidia-smi; echo "# query";
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,pstate --format=csv;
    echo "# compute apps"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv; } > "$RC/idle_nvidia_smi_$1.txt" 2>&1
  { date -u +%FT%TZ; ps -eo pid,etime,pcpu,rss,args | grep -E "python|torchrun|run_ppo|cargo|pip|maturin" | grep -v grep; } > "$RC/idle_ps_$1.txt" 2>&1
}

echo "=== launch.sh start $(date -u +%FT%TZ)"
{ date -u +%FT%TZ; git rev-parse HEAD; git rev-parse 'HEAD^{tree}'; echo "# status --porcelain"; git status --porcelain;
  echo "# check-ignore runs/"; git check-ignore -v runs/value-gap-2026-09-29/driver.py;
  echo "# owl.rs extension"; sha256sum python/owl/rs*.so; } > "$RC/git_pre.txt" 2>&1
{ date -u +%FT%TZ; echo "# which nsys"; which nsys; echo "which rc=$?"; } > "$RC/nsys_check.txt" 2>&1
{ .venv/bin/python -c "import torch, triton, flash_attn, torch._inductor.config as c; print(torch.__version__, torch.version.git_version, triton.__version__, flash_attn.__version__, torch.version.cuda, repr(c.max_autotune_gemm_backends))";
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; env | grep -E '^TORCHINDUCTOR|^TRITON' ; echo "env-grep done"; } > "$RC/versions.txt" 2>&1
( cd "$R" && sha256sum *.py *.sh ) > "$RC/scripts.sha256"
snap pre

waited=0
until gpu_idle 0 && gpu_idle 1; do
  if [ $waited -ge 600 ]; then
    snap notidle
    echo "=== GPUs not idle after 600 s; not running $(date -u +%FT%TZ)"
    echo ALLDONE
    exit 2
  fi
  sleep 15; waited=$((waited + 15))
done
snap prelaunch
echo "=== idle gate passed after ${waited}s; driver start $(date -u +%FT%TZ)"

# -s TERM: the driver gets SIGTERM first; its handler terminates every stage
# group (own sessions, outside timeout's group) within 12 s, before the -k 20 SIGKILL.
timeout -s TERM -k 20 2520 .venv/bin/python "$R/driver.py" > "$R/driver.out" 2>&1
DRC=$?
echo "=== driver exit $DRC $(date -u +%FT%TZ)"

sleep 5
snap post
{ date -u +%FT%TZ; git rev-parse HEAD; git rev-parse 'HEAD^{tree}'; echo "# status --porcelain"; git status --porcelain;
  echo "# owl.rs extension"; sha256sum python/owl/rs*.so; } > "$RC/git_post.txt" 2>&1
( cd "$R" && find inductor_cache triton_cache obs -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum ) > "$RC/caches_and_obs.sha256" 2>/dev/null
echo "=== launch.sh end $(date -u +%FT%TZ)"
echo ALLDONE
