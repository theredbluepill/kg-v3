# Launch: from-scratch mirror self-play at Isaiah's LRs (pod abl4mvr5w1mmn4, 2026-09-30)

- **Owner:** "20× below the 2e-3 then use this Learning Rate then?" (after the main agent noted Muon 1e-4 is 20x below Isaiah's from-scratch 2e-3).
- **Replaces:** scratch-bank-4rank-20260930 (W&B d6sh0akf, Muon 1e-4), stopped at iteration 132: own bank 0 and draw rate ~1.0 at every game end, entropy oscillating 40-45, no learning signal visible. Final log on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/uujuarkx, experiment `scratch-bank-lr2e3-4rank-20260930`.
- **Config:** as scratch-bank-4rank (0.25 own bank /150k + 0.25 cash difference /100k + 0.5 terminal sign; econ shaping 0; random init; teacher seeded from the independent random-init checkpoint) plus `optimizer.muon_lr=0.002 optimizer.adamw_lr=0.0001` (Isaiah `scaling_6m`). Saved config.yaml confirms. Warm-up 1,000 optimizer steps (~iteration 63). No step/time cap.
- **Processes:** launcher pgid 54243; nonfinite-only watchdog log `/root/runs/scratch-lr-watchdog.log`; Mac copy-off `copyoff.sh` to `/Users/poonszesen/kg-v3-runs/scratch-bank-lr2e3-4rank-20260930/`.
- **Question / observation:** does PPO from a random init, at Isaiah's step size, learn to keep and then earn money? Discriminating: own_bank_mean at game ends leaving 0 and trending up. Loss: still 0 with draw ~1.0 after several 10M checkpoints, or nonfinite/entropy collapse with no bank.
