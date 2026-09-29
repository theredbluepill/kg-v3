# Result: early 2-rank Kaggriculture PPO smoke (pre-landing, pending Codex review)

**PASS** on every check in `run-statement.md`. Pre-landing diagnostic on the
unverified merge `kg/pod-ppo-prelanding` `90a1fb0` (0 porcelain lines on the
pod); not Codex-verified. Receipts: `pod-receipts/`.

| Check | Observation |
|---|---|
| Completion | exit 0, 2 iterations, 32,768 global env steps, wall 14 s (launch to exit, including startup) |
| Optimizer steps | `optimizer/steps` 16 after iteration 1, 32 after iteration 2 (16 minibatches per update), both ranks |
| Finite metrics | 0 nonfinite of 63 logged keys, both iterations, both ranks (losses, entropy, grad norm, advantages, explained variance) |
| Replay drift (L6) | no alarm; `policy/logratio_mean` -5.6e-7 / +3.6e-6 nats, `logratio_abs_max` 1.3e-3 / 2.4e-3 (limit 0.05 on the first minibatch) |
| Rank seeds | rank 0 constructed seed 23000 stride 2, rank 1 seed 23001 stride 2 (disjoint residues); `episode_steps` 33 |
| Denominators | per iteration 16,384 game transitions, 32,768 learner-seat turns (`train/player_step_total` 32,768 then 65,536), 512 games (`train/total_games_played` 512 then 1,024) |
| Terminal records | 128 per rank at native step calls 32 and 64 of each rollout (every env ends at transition 32), fields `terminal_bank_0/1`, `terminal_margin_0`, `total_games_played` from the native completed-game banks; 512 records per rank in total |
| W&B | smoke `https://wandb.ai/spoon/kg-v3/runs/yfppat21` state `finished`, project `kg-v3`, job_type/group `ppo`, tags `kaggriculture-v3, ppo`, 2 history rows; online (launcher refused offline) |
| Failure probe (L7) | `KG_PROBE_FAIL_BEFORE_UPDATE=1`, exit 1 after 8 s, rank-tagged tracebacks on both ranks, W&B `https://wandb.ai/spoon/kg-v3/runs/uf1lyaak` state `failed` (not `finished`) |

Other numbers: training throughput 4,924 / 4,990 env steps/s (tiny model,
32-step games; a denominator check, not a speed claim); native step 33.5–34.4
ms per call for 128 envs at `native_threads` 1 (fence + native step +
publish); CUDA peak per rank 3.19 GB allocated in rollout, 7.58 GB allocated /
8.18 GB reserved in update; nvidia-smi peak 8,837 MiB per GPU; GPUs back to 0
MiB after. Cost: about 22 s of launched work, under $0.05 at $4.18/h.

Limits: no evaluation ran (checkpoint_freq null), so the raw-bank
evaluation winner branch is covered only by the CPU tests; the stale
reset-bank outcome check is limited to observing that terminal records arrive
exactly at game ends. The terminal-sample margins shown are small because the
random policy barely trades. Hashes: `pod-receipts/hashes.sha256`.
