#!/bin/bash
# GPU checks bundle launcher (run statement run-statements/gpu-checks-bundle.md).
# Attempt 2 (Amendment 1). Idle gate on GPUs 0 and 1, receipts, the driver
# under a hard 59-minute timeout (60-min aggregate with attempt 1's 55 s), generated-code census (text only), post receipts. Never stops,
# restarts or deletes the pod.
set -u
R=/workspace/kg-v3-rebuild/runs/gpu-checks-2026-09-29
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
  echo "# check-ignore runs/"; git check-ignore -v runs/gpu-checks-2026-09-29/driver.py;
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

# -s TERM: on expiry the driver gets SIGTERM first. Its stages run in their own
# sessions, outside the group that timeout signals, so the driver's handler
# terminates each stage group itself (SIGTERM, bounded wait, SIGKILL; <= 12 s)
# before timeout's SIGKILL at +20 s. (Post-run revision; the recorded attempts
# ran `timeout -k 20 3540` with the earlier driver, which had no handler.)
timeout -s TERM -k 20 3540 .venv/bin/python "$R/driver.py" > "$R/driver.out" 2>&1
DRC=$?
echo "=== driver exit $DRC $(date -u +%FT%TZ)"

"$REPO/.venv/bin/python" "$R/check_templates.py" "$R" > "$R/template_census.txt" 2>&1; echo "=== check_templates exit $?"

sleep 5
snap post
{ date -u +%FT%TZ; git rev-parse HEAD; git rev-parse 'HEAD^{tree}'; echo "# status --porcelain"; git status --porcelain;
  echo "# owl.rs extension"; sha256sum python/owl/rs*.so; } > "$RC/git_post.txt" 2>&1
( cd "$R" && find inductor_cache triton_cache -type f -print0 | sort -z | xargs -0 sha256sum ) > "$RC/compile_caches.sha256" 2>/dev/null
echo "=== launch.sh end $(date -u +%FT%TZ)"
echo ALLDONE
