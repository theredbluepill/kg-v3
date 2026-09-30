# Launch: 720 window, 12 games per GPU, from the promoted 10M checkpoint (2026-09-30)

- **Owner:** asked whether it is safe to increase the games per GPU (the 6-env 720 run pw6qjsz3 ran at ~1,900 env steps/s).
- **Sequence:** pw6qjsz3 (6 envs/rank) stopped at iteration 90; a 16-env/rank launch (W&B 4h9c3d6g) reached 2,615 env steps/s but used 90.96 GB of 96 GB per GPU, too tight for an unattended run with a 10M evaluation on rank 0, so it was stopped at iteration ~3; relaunched at 12 envs/rank. Logs on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/cmwjclbe, experiment `earn720-12env-from-promoted-4rank-20260930`.
- **Config:** as pw6qjsz3 (promoted 10M start fc6b123c..., model_only; reward 0.25 own bank /150k + 0.25 cash diff /100k + 0.5 terminal sign; gae_lambda 1.0; Muon 1e-4 / AdamW 5e-6; horizon 720 = one whole game per segment) with `env.n_envs=12`: 12 optimizer steps of 1,440 rows per iteration (per-step batch unchanged: one game-segment per rank), 34,560 env steps per iteration, 12 games in each last_best evaluation.
- **Checks:** 48 game ends in each of iterations 1-3 (alignment holds); GPU memory 68.5 GB per GPU (27 GB headroom); ~2,410-2,460 env steps/s; own bank 95.2k / 97.3k / 87.7k.
- **Processes:** launcher pgid 67658; watchdog `/root/runs/earn720c-watchdog.log`; Mac copy-off pid 29144.
