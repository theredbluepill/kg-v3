# Launch: full-game (720) credit window from the promoted 10M checkpoint (2026-09-30)

- **Owner:** "theorectically 720 is better or 512?" then "ok go with 720 and relaunch the run".
- **Replaces:** earn512-from-promoted-4rank-20260930 (W&B ssoc84zg), stopped at iteration 41; log on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/pw6qjsz3, experiment `earn720-from-promoted-4rank-20260930`.
- **Start:** `/root/promoted-10M/checkpoint_last_best.pt` (sha256 prefix fc6b123c5e5f76dc, h3lpxy6q's promoted 10M), `model_only` (fresh optimizer, LR re-warm; teacher and eval opponent = the promoted weights).
- **Config:** as h3lpxy6q except `rl.horizon=720 env.n_envs=6` per rank: 17,280 env steps per iteration, 6 optimizer steps of 1,440 rows. Reward 0.25 own bank /150k + 0.25 cash diff /100k + 0.5 terminal sign, gae_lambda 1.0, Muon 1e-4 / AdamW 5e-6, 10M eval, /root/kg-v3-anchor 0f70773.
- **Alignment check (passed):** iterations 1-4 each logged `train/bank_games` 24 (= 6 envs x 4 ranks), i.e. every 720-step segment is exactly one whole game: pure Monte Carlo returns, the critic only a baseline, every update spans all game phases.
- **First iterations:** own bank per iteration 103.2k / 96.9k / 102.4k / 93.4k (24 games each); EV 0.05-0.15 (critic is only a baseline now); SPS ~1,900 (vs ~2,700 at 256); GPU memory 40.9 / 48.6 / 40.9 / 41.0 GB.
- **Processes:** launcher pgid 64260; watchdog `/root/runs/earn720-watchdog.log`; Mac copy-off pid 27072.
