cd /root/kg-v3
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TORCHINDUCTOR_CACHE_DIR=/root/sweep-cache/inductor TRITON_CACHE_DIR=/root/sweep-cache/triton
B="scripts/bench_rollout_step.py --config configs/sps_diag_bench.yaml --device cuda --seat-rows 40 --steps 150 --warmup 20 --threads 1 --with-env"
pgrep -fc "kg-clock-keeper-spi[n]" > /root/sps-diag/bench2.keepers
grep throttled /sys/fs/cgroup/cpu.stat > /root/sps-diag/bench-keeper-native.cpustat.before
taskset -c 48-57 .venv/bin/python $B --output /root/sps-diag/bench-keeper-native.json > /root/sps-diag/bench-keeper-native.log 2>&1
grep throttled /sys/fs/cgroup/cpu.stat > /root/sps-diag/bench-keeper-native.cpustat.after
pkill -f "kg-clock-keeper-spi[n]"; sleep 1; pgrep -fc "kg-clock-keeper-spi[n]" >> /root/sps-diag/bench2.keepers
taskset -c 48-57 .venv/bin/python $B --output /root/sps-diag/bench-nokeeper-native.json > /root/sps-diag/bench-nokeeper-native.log 2>&1
echo BENCH2DONE > /root/sps-diag/bench2.done
