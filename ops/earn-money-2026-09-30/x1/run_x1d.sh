#!/bin/bash
# X1d launcher on the pod: two concurrent CPU processes x 4 threads = 8 threads, nice 19, no GPU.
cd /root/kg-v3-anchor
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4
PY=/root/kg-v3-anchor/.venv/bin/python; X=/root/x1/x1d_selfplay.py
run() { nice -n 19 $PY $X "$1" "$2" 8 4 > /root/x1/log_$1.txt 2>&1; }
( run BC /root/bc-best/checkpoint_bc_best.pt; run hz4_10M /root/runs/J2-resume-r0208-20260930/20260930-020138/checkpoint_00_010_010_624.pt; run M_20M /root/runs/M-margin-J2-4rank-20260930/20260930-033319/checkpoint_00_020_004_864.pt ) &
( run J2_final /root/runs/control-J2-4rank-20260930/20260930-010131/checkpoint_final.pt; run M_10M /root/runs/M-margin-J2-4rank-20260930/20260930-033319/checkpoint_00_010_010_624.pt ) &
wait
