# Result: PPO collapse ablation B, "ablate-B-lr10" (both LRs / 10)

**Execution: PASS. Prediction: met.** With both learning rates divided by 10,
per-update approx KL fell about 5–10x (0.001–0.018 against the control's
0.003–0.09), teacher KL at iteration 46 was 0.64 against 5.28, and the
economy held much better: final banks 73k → 66k → 64k → 62k against the
control's 73k → 55k → 7k → 0 (and A's 41k at game 4). Banks still declined
slightly each game. Pre-landing diagnostic on `kg/pod-ppo-prelanding`
`e74d67e` (0 porcelain lines on the pod), not Codex-verified. Receipts are
in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/32pahqok`, state `finished`,
  online, renamed through the API after the run to `ablate-B-lr10` in group
  `kg-v3-ppo-collapse-ablation` (original name `ppo-20260929-193207`, group
  `ppo`; job_type and tags unchanged). Summary `_step` 753,664.
- **Run:** exit 0 at `--max-env-steps 753664`; 46 complete iterations on both
  ranks, 16 optimizer steps each (736 in total); no nonfinite metric (the
  launcher stop was armed and never fired); no traceback. Launch 19:32:05Z,
  end 19:40:08Z, 483 s wall, about $0.56 at $4.18/h. Overrides reached the
  trainer as `['rl.eval_replay_games=0', 'optimizer.muon_lr=0.0002',
  'optimizer.adamw_lr=0.00001']`; the logged (Muon) LR is exactly 0.1x the
  control's at every iteration (3.2e-6 → 1.472e-4). The AdamW LR is not
  logged separately. `idle_after.csv` caught GPU 0 at 79 % utilisation with
  0 MiB during teardown; `idle_recheck.csv` 47 s later shows both GPUs at
  0 % / 0 MiB with no compute apps.
- **Code identity:** the nine input hashes in `pod-receipts/hashes.sha256`
  equal ablation A's (and so the 6.2 control's); `run_ablation.sh`
  `e5acaf10c2f7…` equals the committed copy.
- **Checkpoint (stays on the pod):** `checkpoint_final.pt` `3671e150…45ab`.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | B (LR / 10) | A (anchor 0.1) | Control 6.2 |
|---|---|---|---|---|
| 1 | 12 | 73,360 / 73,208 | 71,943 / 70,933 | 73,673 / 72,690 |
| 2 | 23 | 65,621 / 65,125 | 59,073 / 61,208 | 55,496 / 54,423 |
| 3 | 34 | 63,689 / 63,913 | 51,212 / 49,184 | 7,324 / 7,644 |
| 4 | 45 | 62,104 / 62,321 | 40,408 / 42,457 | 89 / 94 |

## Per-iteration metrics at the requested iterations (B / control)

Full table: `pod-receipts/iterations.tsv` (from `../extract_iterations.py`);
control: `../ablate-A-anchor/control-6.2-iterations-1-46.tsv`.

| Iter | Entropy | Approx KL | Clipfrac | Adv std | Expl var | Teacher KL | Unit kind KL | Market kind KL | LR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.41 / 4.33 | 0.0014 / 0.0026 | 0.003 / 0.020 | 0.204 / 0.195 | -0.06 / -0.05 | 0.0013 / 0.0026 | 0.0007 / 0.0011 | 0.0003 / 0.0009 | 3.2e-6 / 3.2e-5 |
| 5 | 6.09 / 6.05 | 0.0027 / 0.015 | 0.016 / 0.18 | 0.175 / 0.149 | 0.42 / 0.26 | 0.008 / 0.071 | 0.004 / 0.048 | 0.002 / 0.011 | 1.6e-5 / 1.6e-4 |
| 13 | 4.38 / 4.54 | 0.0044 / 0.042 | 0.043 / 0.36 | 0.165 / 0.088 | 0.17 / 0.44 | 0.059 / 0.56 | 0.033 / 0.35 | 0.005 / 0.089 | 4.16e-5 / 4.16e-4 |
| 23 | 5.31 / 5.38 | 0.0066 / 0.085 | 0.082 / 0.45 | 0.237 / 0.216 | 0.00 / 0.10 | 0.20 / 2.37 | 0.12 / 1.77 | 0.027 / 0.26 | 7.36e-5 / 7.36e-4 |
| 34 | 6.16 / 5.45 | 0.0105 / 0.092 | 0.14 / 0.50 | 0.223 / 0.211 | 0.03 / 0.11 | 0.56 / 3.70 | 0.42 / 2.75 | 0.068 / 0.55 | 1.09e-4 / 1.09e-3 |
| 46 | 4.77 / 5.56 | 0.0114 / 0.043 | 0.15 / 0.39 | 0.077 / 0.043 | -0.36 / 0.10 | 0.64 / 5.28 | 0.34 / 3.08 | 0.11 / 1.82 | 1.47e-4 / 1.47e-3 |

Within-game peaks in B (games reset together, about 11.25 iterations per
game): entropy 7.9 (it 10), 8.0 (21), 8.1 (33), 8.6 (44); teacher KL 0.04
(12), 0.24 (21), 0.66 (33), 1.36 (45). Approx KL rose steadily with the LR
warm-up, 0.0014 → 0.018 (it 45), so the predicted 0.003–0.01 band held
through about iteration 33 and was exceeded after.

## Reading

- **Prediction check:** approx KL about 0.003–0.01 — yes through iteration
  33, then 0.010–0.018 as the LR kept warming. Entropy rising more slowly —
  **not in the way predicted**: B's entropy at iteration 5 (6.09) equals the
  control's (6.05) and its within-game swings (4.4 → 8.0) match the
  control's, with teacher KL of only 0.008–0.04 in game 1. Banks declining
  later than the control's — yes (62k at game 4 against 0), with a small
  continued decline.
- **Correction to the premise:** the entropy rise to about 8 within each game
  is present at a tenth of the step size with negligible policy change, so it
  is a game-phase property of the BC policy's action distribution (late-game
  states are higher-entropy), not drift. Teacher KL, approx KL and banks are
  the drift signals; per-iteration entropy is not, unless compared at the
  same game phase.
- **What it supports:** the update size is LR-bound (approx KL and clipfrac
  scale down roughly with the LR), and smaller steps delay the collapse
  strongly. Teacher KL in B at iteration 46 (0.64, cumulative LR ≈ 0.1x of
  the control's at iteration 46) is close to the control's at iteration 13
  (0.56, cumulative LR ≈ 0.08x), which is consistent with drift tracking the
  integrated step size rather than a signal the policy is climbing. Inferred,
  not shown: B does not tell whether a still smaller LR, or the same LR over
  more iterations, stops the drift or only delays it; the small continuing
  bank decline and teacher KL still rising (1.36 peak at iteration 45) point
  toward delay.
- **Unresolved attribution:** explained variance still swings from about
  -0.4 at game start to 0.5 late in each game with advantage std about
  0.05–0.24 under `normalize_advantages`; B does not separate step size from
  critic noise (C). A drift that tracks cumulative LR regardless of
  direction is what a noisy, near-zero signal would produce; a real
  improving signal would show banks rising, which none of the three runs
  show in 4 games.
- **Throughput note (not an ablation claim):** iterations 2–46 ran at 1,639
  game SPS (rollout 6.35 s, update 2.74 s per iteration), close to A's 1,626
  and below the control's 2,016; rank 0's native step mean 82.8 ms (rank 1
  71.2 ms). Cause not measured.

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../final-report.md`.

- **Outcome:** Execution PASS. No collapse, slow decline: banks 73,284 → 62,212 by game 4 (−11,072).
- **Denominators:** 46 complete iterations on both ranks; 736 optimizer steps (16 per iteration); 753,664 global env steps (46 × 16,384); 1,024 completed games (4 game phases × 256 envs, `train/total_games_played` 1024); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/32pahqok
- **Spend:** about $0.56 at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; all 46 iterations lie inside the 1,000-step LR warm-up (LR at iteration 46 is 0.736 of peak), so nothing here was measured at peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
