# Relaunch: own bank 0.30 + cash difference 0.30 + econ shaping 0.10 (per event) + win/loss 0.30 (2026-09-30)

- **Owner:** "actually give back 0.075 from [own bank 0.375 / cash difference 0.375] to win/lose loss to make win/lose to be 0.3, and relaunch the current run again. Let's do that."
- **Replaces:** earn720-r375e01-from-c50-4rank-20260930 (W&B yei0cild; 0.375/0.375/0.10/0.15), stopped at iteration 23. Log on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/48gyi9m5, experiment `earn720-r30e01w30-from-c50-4rank-20260930`.
- **Reward (saved config.yaml confirms):** econ_bank 0.3/150000/0.3; econ_margin 0.3/100000/0.3; econ_shaping 0.01 (starvation 4, drought 1) cap 0.1; terminal_scale = 1 - 0.3 - 0.3 - 0.1 = 0.30.
- **Unchanged:** start `/root/promoted-C/checkpoint.pt` (pcy5knet 50M, sha 0cc80065, owner-promoted) model_only; horizon 720, gae_lambda 1.0, n_envs 12/rank, Muon 1e-4 / AdamW 5e-6, 10M eval. Launcher pgid 79722; watchdog `/root/runs/earnF-watchdog.log`; Mac copy-off pid 83575.
- **Plan:** first 10M checkpoint -> local fixed-shop games vs smaller_market_shock, cha22, v56; compare with c50 (4/16, 2/16, 0/16 wins; margins -4.4k, -9.5k, -10.4k).
