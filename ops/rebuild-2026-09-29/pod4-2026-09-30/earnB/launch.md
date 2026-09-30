# Relaunch at 1e-4 from the promoted 20M checkpoint (2026-09-30)

- **Owner:** "Sure please do" (after the main agent recommended stopping the 2e-3 run and relaunching at 1e-4 from the same promoted checkpoint).
- **Stopped:** earn720-lr2e3-from-promoted2-4rank-20260930 (W&B pz3xhg9e) at iteration 81, LR rising to its 2e-3 peak. Iterations 51-69 vs 1-25: approx_kl 0.144 vs 0.017, clip fraction 0.55 vs 0.19, entropy 8.92 vs 6.49, teacher KL 5.56 vs 0.36, own bank 83.8k vs 94.5k (p10 59.6k vs 64.6k): the full-LR collapse signature seen in the BC-start ablation arms. Log on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/pcy5knet, experiment `earn720-lr1e4-from-promoted2-4rank-20260930`.
- **Config:** start `/root/promoted-B/checkpoint_last_best.pt` (cmwjclbe's promoted 20M; weights bit-identical to its checkpoint_00_020_033_024) model_only; 720 window, 12 envs/rank, gae_lambda 1.0; reward 0.25 own bank /150k + 0.25 cash diff /100k + 0.5 terminal sign; Muon 1e-4 / AdamW 5e-6 (saved config confirms). Launcher pgid 72804; watchdog `/root/runs/earnB-watchdog.log`; Mac copy-off pid 32294.
