#!/bin/bash
cd /root/kg-v3-bc
mkdir -p /workspace/kg-v3-bc-2026-09-29/logs
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/kg-v3-bc-2026-09-29/logs/prepare-top1.start
exec .venv/bin/python scripts/kaggriculture_prepare_bc.py /root/bc-archives /workspace/kg-v3-bc-2026-09-29/shards-top1 --days 22-28 --workers 12 --turn-stride 2 --team "M & M & P & Q"
