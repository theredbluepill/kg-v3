# Relaunch: own bank 0.375 + cash difference 0.375 + econ shaping 0.10 + win/loss 0.15, from pcy5knet's 50M checkpoint (2026-09-30)

- **Owner:** "split the 0.25 of win /lose loss, 0.10 to econ shaping." then "and relaucn hthe current run."
- **Replaces:** earn720-r375-from-c50-4rank-20260930 (W&B j4t2z0hn; 0.375/0.375/0.25), stopped at iteration 29. Log on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/jk0nu7an, experiment `earn720-r375e10-from-c50-4rank-20260930`.
- **Reward (saved config.yaml confirms):** econ_bank 0.375/150000/0.375; econ_margin 0.375/100000/0.375; econ_shaping 0.1 with econ_cap 0.1 (starvation weight 4, drought 1, as in earlier runs; one starvation or drought event already reaches the 0.1 cap); terminal_scale = 1 - 0.375 - 0.375 - 0.1 = 0.15.
- **Unchanged:** start `/root/promoted-C/checkpoint.pt` (pcy5knet 50M, sha 0cc80065, owner-promoted) model_only; horizon 720, gae_lambda 1.0, n_envs 12/rank, Muon 1e-4 / AdamW 5e-6, 10M eval. Launcher pgid 78074; watchdog `/root/runs/earnD-watchdog.log`; Mac copy-off pid 83080.
- **Plan:** evaluate its first 10M checkpoint locally against smaller_market_shock, cha22 and v56 (fixed-shop harness).
