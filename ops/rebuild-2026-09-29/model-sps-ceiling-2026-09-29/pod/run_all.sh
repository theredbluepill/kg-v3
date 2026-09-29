#!/bin/bash
set -u
R=/workspace/kg-v3-rebuild/runs/model-sps-ceiling-2026-09-29
cd /workspace/kg-v3-rebuild
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$R/inductor_cache TRITON_CACHE_DIR=$R/triton_cache PYTHONUNBUFFERED=1
for d in sparse mid dense; do
  echo "=== $d start $(date -u +%FT%TZ)"
  timeout 1500 .venv/bin/python $R/bench_model_sps.py --density $d --out $R > $R/log_$d.txt 2>&1
  echo "=== $d exit $? $(date -u +%FT%TZ)"
done
echo ALLDONE
