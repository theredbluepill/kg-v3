# Relaunch: per-event econ shaping (0.01, cap 0.10) + own bank 0.375 + cash difference 0.375 + win/loss 0.15 (2026-09-30)

- **Owner:** agreed ("Sure go ahead.") to the main agent's proposal: econ_shaping 0.1 with cap 0.1 saturates at the first starvation (0.4) or drought (0.1) event, so use econ_shaping 0.01 with cap 0.1 (starvation 0.04, drought 0.01 each; saturates after ~2-3 starvations or 10 droughts).
- **Replaces:** earn720-r375e10-from-c50-4rank-20260930 (W&B jk0nu7an; econ_shaping 0.1), stopped at iteration 10. Log on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/yei0cild, experiment `earn720-r375e01-from-c50-4rank-20260930`.
- **Reward (saved config.yaml confirms econ_shaping 0.01, econ_cap 0.1):** econ_bank 0.375/150000/0.375; econ_margin 0.375/100000/0.375; econ shaping 0.01 (starvation 4, drought 1), cap 0.10; terminal_scale 0.15.
- **Unchanged:** start `/root/promoted-C/checkpoint.pt` (pcy5knet 50M, sha 0cc80065, owner-promoted) model_only; horizon 720, gae_lambda 1.0, n_envs 12/rank, Muon 1e-4 / AdamW 5e-6, 10M eval. Launcher pgid 78904; watchdog `/root/runs/earnE-watchdog.log`; Mac copy-off pid 83289.
- **Starting point, local fixed-shop anchors (c50 = this checkpoint, 8 seeds x 2 seats):** vs smaller_market_shock 4/16 wins, margin -4.4k (c50-08bc +7.2k +/- 3.2k, 7/8 seeds); vs cha22 2/16 wins, -9.5k (+1.7k +/- 2.5k, 4/8); vs v56 0/16, -10.4k (+2.7k +/- 2.4k, 5/8).
