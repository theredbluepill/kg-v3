#!/bin/bash
# ATEN-only GEMM A/B launcher (run statement run-statements/aten-gemm-ab.md).
# Idle gate on GPU 0, receipts, the driver under a hard 45-minute timeout,
# generated-code analysis (text only), post receipts. Never stops the pod.
set -u
R=/workspace/kg-v3-rebuild/runs/aten-gemm-ab-2026-09-29
REPO=/workspace/kg-v3-rebuild
RC=$R/receipts
mkdir -p "$RC"
cd "$REPO"

gpu0_idle() {
  local apps util
  apps=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  util=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  [ -z "$apps" ] && [ "$util" = "0" ]
}

snap() {  # $1 = tag
  { date -u +%FT%TZ; nvidia-smi; echo "# gpu0 query";
    nvidia-smi -i 0 --query-gpu=index,memory.used,utilization.gpu,pstate --format=csv;
    echo "# gpu0 compute apps"; nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$RC/idle_nvidia_smi_$1.txt" 2>&1
  { date -u +%FT%TZ; ps -eo pid,etime,pcpu,rss,args | grep -E "python|torchrun|run_ppo|cargo|pip|maturin" | grep -v grep; } > "$RC/idle_ps_$1.txt" 2>&1
}

echo "=== launch.sh start $(date -u +%FT%TZ)"
{ date -u +%FT%TZ; git rev-parse HEAD; git rev-parse 'HEAD^{tree}'; echo "# status --porcelain"; git status --porcelain;
  echo "# check-ignore runs/"; git check-ignore -v runs/aten-gemm-ab-2026-09-29/driver.py; } > "$RC/git_pre.txt" 2>&1
{ date -u +%FT%TZ; echo "# which nsys"; which nsys; echo "which rc=$?"; ls -d /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems* 2>&1; } > "$RC/nsys_check.txt" 2>&1
{ .venv/bin/python -c "import torch, triton, flash_attn, torch._inductor.config as c; print(torch.__version__, torch.version.git_version, triton.__version__, flash_attn.__version__, torch.version.cuda, repr(c.max_autotune_gemm_backends))";
  nvidia-smi --query-gpu=index,name,driver_version --format=csv; env | grep -E '^TORCHINDUCTOR|^TRITON' ; echo "env-grep done"; } > "$RC/versions.txt" 2>&1
( cd "$R" && sha256sum *.py *.sh ) > "$RC/scripts.sha256"
snap pre

waited=0
until gpu0_idle; do
  if [ $waited -ge 600 ]; then
    snap notidle
    echo "=== GPU 0 not idle after 600 s; not running $(date -u +%FT%TZ)"
    echo ALLDONE
    exit 2
  fi
  sleep 15; waited=$((waited + 15))
done
snap prelaunch
echo "=== idle gate passed after ${waited}s; driver start $(date -u +%FT%TZ)"

timeout -k 20 2700 .venv/bin/python "$R/driver.py" > "$R/driver.out" 2>&1
DRC=$?
echo "=== driver exit $DRC $(date -u +%FT%TZ)"

"$REPO/.venv/bin/python" "$R/analyze_kernels.py" "$R" > "$R/kernel_analysis.txt" 2>&1; echo "=== analyze_kernels exit $?"
"$REPO/.venv/bin/python" "$R/check_templates.py" "$R" > "$R/template_census.txt" 2>&1; echo "=== check_templates exit $?"

snap post
{ date -u +%FT%TZ; git rev-parse HEAD; echo "# status --porcelain"; git status --porcelain; } > "$RC/git_post.txt" 2>&1
( cd "$R" && find inductor_cache triton_cache -type f -print0 | sort -z | xargs -0 sha256sum ) > "$RC/compile_caches.sha256" 2>/dev/null
echo "=== launch.sh end $(date -u +%FT%TZ)"
echo ALLDONE
