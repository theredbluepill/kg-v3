#!/bin/bash
cd /root/kg-v3
export TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton
nohup .venv/bin/python /root/parity/gpu_head_parity2.py > /root/parity/run2.log 2>&1 &
echo $! > /root/parity/pid
