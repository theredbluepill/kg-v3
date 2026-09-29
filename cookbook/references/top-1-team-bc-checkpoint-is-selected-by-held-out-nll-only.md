---
type: "Reference"
title: "Top-1 team BC checkpoint is selected by held-out NLL only"
description: "First real BC run (A100, 1 GPU, eager, source f0b7a38): imitating the leaderboard #1 team's winning seats over 2026-09-22..28 (507 train / 16 validation episodes, stride 2), held-out policy NLL fell from 3.73 to a best of 0.480 at step 3,200 (9 epochs) and the L9 stop fired at step 5,200 while train NLL kept falling. Checkpoint fd854587... on the pod volume. Selection only; playing strength and PPO warm-start benefit are untested."
tags: ["kaggriculture-v3", "bc", "checkpoint"]
status: "verified-scoped"
generated: {"by": "anthropic/claude-opus-5-5", "at": "2026-09-29"}
sources: [{"resource": "repository:ops/rebuild-2026-09-29/run-statements/bc-a100.md"}, {"resource": "repository:ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/README.md"}, {"resource": "repository:ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/bc_history.jsonl"}, {"resource": "repository:ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/bc_result.json"}, {"resource": "repository:ops/rebuild-2026-09-29/bc-a100-2026-09-29/train/SHA256SUMS"}, {"resource": "repository:ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md"}, {"resource": "repository:configs/bc/kaggriculture_1gpu_eager.yaml"}, {"resource": "repository:configs/kaggriculture_1gpu_eager.yaml"}, {"resource": "pod-artifact:23f7xiz368x9jf:/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/"}]
---

# Top-1 team BC checkpoint is selected by held-out NLL only

The first real run of the [[kaggriculture-bc-trainer-warm-starts-ppo-from-the-held-out-best|BC trainer]] produced a warm-start checkpoint. It was **selected** by the lowest held-out policy NLL. Nothing here shows that it plays well.

## What was run

- Data: the leaderboard #1 team's winning seats, days 2026-09-22..28, turn stride 2; 507 train episodes (182,273 turn rows) and 16 validation episodes (5,748 rows). Shard manifest SHA-256 `ba5fe1c4…3036` (`ops/rebuild-2026-09-29/bc-a100-2026-09-29/receipts.md`).
- Source `f0b7a38` on `kg/rebuild-bc-now`, `scripts/train_bc.py`, one A100 80GB, `configs/bc/kaggriculture_1gpu_eager.yaml`: the 2-rank config's global batch (512 turn rows per step), same optimizer, seed and L9 patience. The model runs **eager**, because this pod's NVIDIA driver (580.159.03) is outside the probed compile stack and `train_bc.py` refuses it.

## Result

- Held-out NLL 3.732 → **0.480 at step 3,200** (9.0 epochs), then 0.48–0.53 while train NLL fell from 0.37 to 0.21. The L9 stop (`no_held_out_improvement`) fired at step 5,200, after 36 min inside the trainer. The 2 h cap was not reached.
- Checkpoint `/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/checkpoint_bc_best.pt` on pod `23f7xiz368x9jf`'s network volume, SHA-256 `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51`. It uses run_ppo's checkpoint schema.
- Throughput, eager: 2,739 learner seat rows/s mean over training steps (about 1,370 turn rows/s). Complete work including preflight and evaluations was 1,245 turn rows/s. nvidia-smi peak was 34,003 MiB.

## Limits and consequence

- **Selection, not qualification.** The same 16 episodes select the checkpoint and report its NLL. No separate held-out set, no game play, no legality check in live play and no PPO warm-start comparison exists.
- There is one seed. The LR was still at its post-warmup plateau when the run stopped. The small train set (507 episodes) overfit after about 9 epochs at these settings. Other data, stride or regularization were not tried.
- W&B telemetry is offline only (no key on the pod; run `kvl4rfda`, not synced).
- The saved `config.yaml` is the eager 1-GPU PPO config. The weights load through `run_ppo`'s `--load-model-weights` path into the eager, 2- and 8-rank models with equal outputs (CPU); the 4-rank model follows by equal model sections. Its critic is saturated, so PPO starts with a fresh critic head: see [[bc-best-starts-ppo-with-a-fresh-critic-head|BC-to-PPO handoff Reference]].

Reopen this note when the checkpoint is evaluated in play or used to start PPO.
