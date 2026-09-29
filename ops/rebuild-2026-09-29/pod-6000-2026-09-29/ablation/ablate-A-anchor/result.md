# Result: PPO collapse ablation A, "ablate-A-anchor" (teacher_kl_coef 0.1)

**Execution: PASS. Prediction: not met.** The 20x anchor slowed the economy's
decline (game 4 banks about 41k against the control's about 90) but did not
hold the policy near the BC best: teacher KL passed 0.5 at iteration 16 and
reached 2–3 nats, and entropy rose to about 10.9. Pre-landing diagnostic on
`kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod), not
Codex-verified. Receipts are in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/y8j6vzky`, state `finished`,
  online, renamed through the API after the run to `ablate-A-anchor` in group
  `kg-v3-ppo-collapse-ablation` (original name `ppo-20260929-192113`, group
  `ppo`; job_type and tags unchanged). Summary `_step` 753,664.
- **Run:** exit 0 at `--max-env-steps 753664`; 46 complete iterations on both
  ranks, 16 optimizer steps each (736 in total); no nonfinite metric (the
  launcher stop was armed and never fired); no traceback. Launch 19:21:10Z,
  end 19:29:20Z, 490 s wall, about $0.57 at $4.18/h. GPUs idle afterwards
  (`idle_after.csv`, no compute apps).
- **Code identity:** the nine input hashes in `pod-receipts/hashes.sha256`
  (config, model config, launcher, `run_ppo.py`, `ppo.py`, `logging.py`,
  `env.py`, `rs.abi3.so`, BC best `fd854587…6f51`) equal the 6.2 control's.
  `run_ablation.sh` `e5acaf10c2f7…` equals the committed copy. Logged
  `teacher/kl_coef` is 0.1 in every iteration.
- **Checkpoint (stays on the pod):** `checkpoint_final.pt` `01d52b5c…0c52`.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | A (coef 0.1) | Control 6.2 (coef 0.005) |
|---|---|---|---|
| 1 | 12 | 71,943 / 70,933 | 73,673 / 72,690 |
| 2 | 23 | 59,073 / 61,208 | 55,496 / 54,423 |
| 3 | 34 | 51,212 / 49,184 | 7,324 / 7,644 |
| 4 | 45 | 40,408 / 42,457 | 89 / 94 |

## Per-iteration metrics at the requested iterations (A, then control)

Full tables: `pod-receipts/iterations.tsv` (A) and
`control-6.2-iterations-1-46.tsv`, both from `../extract_iterations.py`.

| Iter | Entropy | Approx KL | Clipfrac | Adv std | Expl var | Teacher KL | Unit kind KL | Market kind KL | LR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.33 / 4.33 | 0.0022 / 0.0026 | 0.013 / 0.020 | 0.190 / 0.195 | -0.37 / -0.05 | 0.0022 / 0.0026 | 0.0011 / 0.0011 | 0.0005 / 0.0009 | 3.2e-5 |
| 5 | 6.03 / 6.05 | 0.014 / 0.015 | 0.18 / 0.18 | 0.145 / 0.149 | 0.25 / 0.26 | 0.049 / 0.071 | 0.034 / 0.048 | 0.005 / 0.011 | 1.6e-4 |
| 13 | 4.63 / 4.54 | 0.036 / 0.042 | 0.32 / 0.36 | 0.084 / 0.088 | -0.13 / 0.44 | 0.29 / 0.56 | 0.19 / 0.35 | 0.029 / 0.089 | 4.2e-4 |
| 23 | 5.77 / 5.38 | 0.27 / 0.085 | 0.57 / 0.45 | 0.215 / 0.216 | 0.09 / 0.10 | 0.82 / 2.37 | 0.57 / 1.77 | 0.11 / 0.26 | 7.4e-4 |
| 34 | 8.16 / 5.45 | 0.40 / 0.092 | 0.70 / 0.50 | 0.212 / 0.211 | 0.13 / 0.11 | 1.88 / 3.70 | 1.46 / 2.75 | 0.14 / 0.55 | 1.09e-3 |
| 46 | 6.43 / 5.56 | 0.28 / 0.043 | 0.65 / 0.39 | 0.056 / 0.043 | 0.04 / 0.10 | 2.14 / 5.28 | 1.43 / 3.08 | 0.35 / 1.82 | 1.47e-3 |

Within-game swings are large (all envs reset together, about 11.25
iterations per game). In A, entropy peaks late in each game: 7.7 (it 10),
8.4 (it 21), 9.6 (it 33), 10.9 (it 44); teacher KL peaks 0.25, 1.09, 2.21,
3.01 at the same points. `teacher_kl_loss` (coef x KL) grew from 0.029 (it
13) to 0.21 (it 46), above the policy loss (0.024 → 0.089).

## Reading

- **Prediction check:** teacher KL below about 0.5 — no (above 0.5 from
  iteration 16, 2–3 nats late). Entropy near 4–5 — no (up to 10.9). Banks
  near 70k through game 4 — no, but the decline is much slower (72k → 59k →
  50k → 41k against 73k → 55k → 7k → 0).
- **What it supports:** the anchor strength matters (a 20x coefficient cut
  late teacher KL by about 2.5x and kept an economy through game 4), so a
  weak anchor is part of the collapse. It does not stop the drift: the
  policy still moves away at a steady rate as the LR warms up.
- **What it points at next:** with the stronger anchor the per-update
  approx KL (0.2–0.4) and clip fraction (0.55–0.70) became *larger* than the
  control's, while the advantage std late in each game fell to 0.02–0.04.
  Larger steps per update despite a larger restoring term is consistent
  with the step-size candidate (B): the update size is set by the optimizer
  and LR, not by the size of the PPO signal. This is inferred, not shown.
- **Unresolved attribution:** A cannot separate step size (B) from critic
  noise under `normalize_advantages` with a fresh critic head (C); explained
  variance swings from about -0.4 at game start to 0.9 late in each game in
  both runs. The KL direction and per-head weighting of `teacher/kl` were not
  re-inspected for this run.
- **Throughput note (not an ablation claim):** A's iterations 2–46 ran at
  1,626 game SPS against the control's 2,016 over the same iterations;
  rollout 6.4 s against 4.5 s per iteration, update 2.7 s in both. Rank 0's
  native step mean was 84 ms (control 38 ms over its whole run). The cause
  (pod contention, higher-entropy action mix, or other) was not measured.
