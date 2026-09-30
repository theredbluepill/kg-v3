# 8-rank run package (plan 6.3b and the recipe-J main run) — DRAFT

The owner said on 2026-09-30: "cut the interval into half. thanks. also stop any running J/K. We will prepare the run for 8-rank". This directory prepares that run. It holds docs and scripts only. **Nothing in it has run on a GPU, and no pod was created.** Creating the 8-GPU pod needs the owner's explicit approval after seeing the live price (`cost.md`).

## Durable BC checkpoint (outside any repository)

`/Users/poonszesen/kg-v3-runs/bc-best/` on the owner's Mac. It was copied on 2026-09-30 from `/tmp/kg-v3-bc-best/`, which the OS can wipe. It holds `checkpoint_bc_best.pt`, its `.json` record, `bc_result.json`, `bc_config.yaml`, `config.yaml` and `shards-top1/` (the manifest and one validation shard). Every file's SHA-256 matched the `/tmp` source, and the list is in `/Users/poonszesen/kg-v3-runs/bc-best.SHA256SUMS`:

- `checkpoint_bc_best.pt` `fd8545872aca59c70e273e9655055e1104cd719588364f463ae87b0d488e6f51` (equal to its record's `sha256`);
- `shards-top1/manifest.json` `ba5fe1c417741c587473b5696a6ca55227240b394236948cb4f4f5c4ed9a3036` (equal to the record's `dataset_manifest_sha256`).

The original is on network volume `4llk4uaf20` (EU-RO-1) at `/workspace/kg-v3-bc-2026-09-29/run/bc-20260929-142216/checkpoint_bc_best.pt`. Only a pod created in EU-RO-1 with that volume attached can read it.

## Order of operations

| # | Where | What | File |
|---|---|---|---|
| 0 | Mac | Owner approves the pod after a fresh live price read | `cost.md` |
| 1 | Mac | Bundle the source, clone it on the pod, copy and verify the BC best, install the W&B credential (stdin, api.wandb.ai only, mode 600) | `stage_from_mac.sh`, `copy_bc_best.sh` |
| 2 | Pod | Toolchain, frozen sync with flash-attn, release build, driver gate (595.91.07), flash-attn on sm_120 on all 8 GPUs, W&B and BC checks | `setup.sh`, `check_flash_all_gpus.py` |
| 3 | Pod | 6.3b checks: seeds, memory smoke, `native_threads` sweep 2/4/8, all-reduce, 30-minute complete-work | `qualify.sh`, `seed_probe.py`, `allreduce_bench.py`, `summarize_run.py`, `run-statement-6.3b.md` Part A |
| 4 | Pod + Mac | Main run with the watchdog; keep pulling checkpoints to the Mac | `launch.sh`, `watchdog.py`, `pull_from_pod.sh`, `run-statement-6.3b.md` Part B |

`common.sh` holds the pod-side preflight (clean tree, idle GPUs, BC SHA-256, credential mode, input hashes) and the nvidia-smi sampler.

## Recipe and cadence

- `configs/kaggriculture_8rank_bc_finetune.yaml` is recipe J: the 8-rank config with both LRs / 10. Launch it with `--load-model-weights <bc best> --load-model-weights-mode model_only` (cookbook: `references/bc-fine-tune-presets-divide-both-learning-rates-by-ten.md`).
- `checkpoint_freq` is 10,000,000 by the owner's decision: a checkpoint and a last-best promotion check every 610–611 iterations (cookbook: `decisions/halve-the-kaggriculture-checkpoint-interval-to-10m-steps.md`).

## Receipts of this change

- `local-checks.md`: what was checked on the Mac (seed probe, watchdog scenarios, summarizer, shell syntax, lint).
- `local-seed-probe.json`: the probe output at world size 8.
- `prepare.log`: `just prepare` on this change.
- `mutations.log`: the config-test mutations.

## Open items for the owner

- Approve an 8-GPU pod at a live price. As of the 2026-09-30 read, 8x RTX PRO 6000 is out of stock on every listed CUDA version.
- Set the main run's `RUNTIME_HOURS`.
- Choose the region: EU-RO-1 with volume `4llk4uaf20`, or any region with the Mac copy.
