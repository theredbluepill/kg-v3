# BC training receipts: A100 pod, 2026-09-29 (top-1 team, 1 GPU, eager)

Run statement: `../../run-statements/bc-a100.md` (committed in `f0b7a38` before
launch). Compact copies only; weights and optimizer state stay on the pod's
network volume.

## Identity

- Pod `23f7xiz368x9jf`, 1x A100 80GB PCIe, NVIDIA driver 580.159.03, torch
  2.9.0+cu128, triton 3.5.0, flash-attn 2.8.3 (checkout venv `/root/kg-v3-bc/.venv`).
- Source `f0b7a3877eb16657f973aaf55c70779b040e5b46` (clean checkout; recorded by
  `train_bc.py` in `bc_attempts.jsonl`). Trainer code as at `89ca39c`; `owl_rs`
  build reused.
- Data `/workspace/kg-v3-bc-2026-09-29/shards-top1`, manifest SHA-256
  `ba5fe1c417741c587473b5696a6ca55227240b394236948cb4f4f5c4ed9a3036`
  (507 train episodes / 182,273 turn rows; 16 validation episodes / 5,748 rows).
- Config `configs/bc/kaggriculture_1gpu_eager.yaml` -> `configs/kaggriculture_1gpu_eager.yaml`:
  512 turn rows per step (the 2-rank config's global batch), accumulation 1,
  eager model (`Compiled model regions: 0`), BF16 autocast.
- Command: `run-bc-a100.sh` (nohup, plain python, world size 1,
  `--wandb-mode offline --max-runtime-hours 2`).
- Pod run dir: `/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/`.
- W&B: launched with `--wandb-mode offline` (no key on the pod), then synced
  to [spoon/kg-v3/kvl4rfda](https://wandb.ai/spoon/kg-v3/runs/kvl4rfda).
  The pod's `wandb/offline-run-20260929_142222-kvl4rfda/run-kvl4rfda.wandb.synced`
  marker has mtime `2026-09-29T15:01:13Z`; the orchestrator reports the final
  sync at `2026-09-29T15:01:14Z`.

## Result

- Stop: `no_held_out_improvement` (the L9 rule: 10 scheduled evaluations,
  200 steps apart, without a new minimum) at step 5,200 (14.6 epochs,
  356 steps/epoch). The 2 h cap was not reached.
- Best: step 3,200 (9.0 epochs), held-out policy NLL 0.48018 per turn.
  Final (step 5,200): 0.52824. Initial train NLL 3.7106 (preflight replay).
- Best checkpoint: `/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/checkpoint_bc_best.pt`,
  51,911,591 bytes, SHA-256
  `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`
  (matches `checkpoint_bc_best.json` and `bc_result.json`).
- Wall: launch 14:19:23Z, data loaded and run dir created 14:22:16Z, result
  written 14:58:01Z (38.6 min total; 2,138 s inside `train_bc`, including the
  117 s preflight replay of all train rows and 27 held-out evaluations of
  about 3 s each).

### Held-out NLL curve (every 200 steps; train NLL is the interval mean)

| step | val NLL | train NLL | val value CE |
| ---: | ---: | ---: | ---: |
| 0 | 3.7321 | - | 0.7243 |
| 400 | 2.3982 | 2.8242 | 0.0269 |
| 800 | 1.1154 | 1.3877 | 0.0405 |
| 1200 | 0.7507 | 0.7528 | 0.0446 |
| 1600 | 0.6461 | 0.6016 | 0.0700 |
| 2000 | 0.5665 | 0.5055 | 0.0613 |
| 2400 | 0.5231 | 0.4454 | 0.0529 |
| 2800 | 0.4939 | 0.4030 | 0.0345 |
| **3200** | **0.4802** | 0.3667 | 0.0848 |
| 3800 | 0.4816 | 0.3018 | 0.0950 |
| 4400 | 0.5026 | 0.2617 | 0.0787 |
| 5200 | 0.5282 | 0.2119 | 0.0742 |

Full curve: `train/bc_history.jsonl`. Train NLL kept falling (0.37 -> 0.21)
while held-out NLL rose after step 3,200: overfitting on 507 episodes, the L9
pattern. Held-out value CE also rose from its 0.027-0.035 low (steps 400,
2800) to 0.07-0.13.

## Throughput and memory

- Learner seat rows/s over training steps only (`perf/learner_seat_rows_per_second`,
  26 intervals): mean 2,739, min 2,483, max 3,052, i.e. about 1,370 turn rows/s
  at 512 turn rows per step, about 0.37 s per step.
- Complete work inside `train_bc` (preflight, evaluations and checkpoint writes
  included): 5,200 x 512 = 2,662,400 turn rows in 2,138 s = 1,245 turn rows/s.
- GPU memory: nvidia-smi samples every 30 s (`train/gpu-samples.csv`) peaked at
  34,003 MiB used (process-wide, allocator reserve included) of 80 GB;
  utilization samples 32-99 %. `torch.cuda.max_memory_allocated` was not logged.
- Eager only; no compiled comparison and no nsys capture on this pod, so this
  is not a throughput ceiling.

## Scope and gaps

- Selection only: the checkpoint is chosen on the same 16 validation episodes
  it is reported on; no separate held-out qualification. Playing strength,
  legality in live play and PPO warm-start benefit are untested.
- One seed, one learning-rate setting (Muon 0.002 / AdamW 1e-4, LR still at its
  warmup plateau when stopped; cosine decays over 100 k steps), turn stride 2.
- The PPO `config.yaml` saved with the checkpoint is the eager 1-GPU config;
  the model section is identical to `kaggriculture_2rank.yaml`, so
  `--load-model-weights` into the 2-rank config is expected to work but was not
  run.

## Files

`train/`: `bc_history.jsonl`, `bc_result.json`, `checkpoint_bc_best.json`,
`bc_attempts.jsonl`, `config.yaml`, `bc_config.yaml`, `train.log`,
`gpu-samples.csv`, `run-bc-a100.sh`, `launch.start`, `SHA256SUMS` (pod-side
hashes, including the checkpoint and `bc_state.pt`).
