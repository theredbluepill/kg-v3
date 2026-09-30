# Launch: 512-turn credit window from the promoted 10M checkpoint (pod abl4mvr5w1mmn4, 2026-09-30)

- **Owner:** "no need second pod, just start with previous promoted ckpt, and launcch it with 512 window."
- **Replaces:** earn-bank-credit-4rank-20260930 (W&B h3lpxy6q, horizon 256 / lambda 1), stopped at iteration 1,007 (~16.5M env steps), before its 20M evaluation. Its 10M evaluation beat BC 16/16 (own 117.6k vs 90.4k) and promoted. Final log and checkpoints are on the Mac.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/ssoc84zg, experiment `earn512-from-promoted-4rank-20260930`.
- **Start:** `/root/promoted-10M/checkpoint_last_best.pt` (sha256 prefix fc6b123c5e5f76dc; copy of h3lpxy6q's promoted last_best) with `--load-model-weights-mode model_only`: fresh optimizer and a new 1,000-step LR warm-up; the last_best teacher and eval opponent are the promoted weights.
- **Config:** as h3lpxy6q (configs/kaggriculture_4rank_margin.yaml + reward 0.25 own bank /150k + 0.25 cash diff /100k + 0.5 terminal sign, econ shaping 0, gae_lambda 1.0, Muon 1e-4 / AdamW 5e-6, 10M checkpoint/eval, code /root/kg-v3-anchor 0f70773) except `rl.horizon=512 env.n_envs=8` per rank: 16,384 env steps per iteration, but 8 optimizer steps of 1,024 rows (was 16 of 512), and 32 games per game-end window (was 64). Script `run_earn512.sh` here.
- **Processes:** launcher pgid 62653; nonfinite-only watchdog `/root/runs/earn512-watchdog.log`; Mac copy-off pid 26447.
- **First checks:** GPU memory 38.9 GB per GPU (unchanged); first game-end window (iteration 3, 32 games) own bank 95.0k.
- **Confounds:** window 256 -> 512 changes the batch structure too (half as many, twice as large optimizer steps); the start is the promoted policy, not BC, and the LR re-warms. Comparison to h3lpxy6q is therefore not a clean one-change test.
