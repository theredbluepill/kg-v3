#!/bin/bash
cd /root/kg-v3-bc
R=/workspace/kg-v3-bc-2026-09-29/run
date -u +%Y-%m-%dT%H:%M:%SZ > $R/launch.start
exec .venv/bin/python scripts/train_bc.py configs/bc/kaggriculture_1gpu_eager.yaml --data /workspace/kg-v3-bc-2026-09-29/shards-top1 --output-dir $R --wandb-mode offline --max-runtime-hours 2
