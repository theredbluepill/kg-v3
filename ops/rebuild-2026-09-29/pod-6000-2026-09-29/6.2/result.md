# Result: plan 6.2 complete-work run on 2 ranks from the BC best (pre-landing, pending Codex review)

**Execution: PASS. Learning: a red flag.** The run completed with every
execution check in the run statement met. The policy, however, moved far
from the BC best and the game economy collapsed within about three games
(see "Losses, entropy and banks"). This is a pre-landing diagnostic on the
unverified merge `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the
pod); it is not Codex-verified. Receipts are in `pod-receipts/`. The
iteration metrics come from the launcher's `[kg-probe]` records in
`run.log`, which carry the same reduced values the trainer logs.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/7k07gp7c`, state `finished`,
  online, project `kg-v3`, job_type and group `ppo`, `lastHistoryStep`
  3,850,240. A full `scan_history` pull was stopped after about 7 minutes;
  `wandb_7k07gp7c_state.json` holds the state and summary.
- **Run:** exit 0. Launch was 16:33:04Z and end 17:01:28Z, 1,704 s wall.
  The runtime stop came at 0.47 h. Cost is about $1.98 at $4.18/h.
- **Checkpoint (stays on the pod):** `checkpoint_final.pt`
  `19cd0e6a…256e`.

## Execution checks

| Check | Observation |
|---|---|
| Complete iterations | 235 on both ranks. Global env steps 3,850,240 = 235 x 16,384. |
| Optimizer steps | Exactly 16 per iteration in all 235 iterations (`optimizer/steps` deltas), 3,760 in total. |
| Finite metrics | No nonfinite value in any iteration. The launcher's nonfinite stop was armed and never fired. |
| L6 replay drift | The first-minibatch alarm (|mean log-ratio| > 0.05, checked before any optimizer step) never fired, so every iteration's first-minibatch mean was within ±0.05 nats. The exact per-iteration first-minibatch value is not logged. `policy/logratio_mean`, which is logged, averages all 16 minibatches after the parameters change and reached at most 0.160. |
| Native exceptions | None. There is no traceback in `run.log`. |
| Seeds | Rank 0 seed 0 and rank 1 seed 1, both stride 2 (disjoint residues). 2,560 terminal records per rank (20 games x 128 envs). |

## Throughput over complete iterations (equivalent complete work)

| Window | Global game SPS | Learner-seat SPS | Seconds | Env steps / seat turns |
|---|---|---|---|---|
| All 235 iterations | 2,272.5 | 4,545.0 | 1,694.3 | 3,850,240 / 7,700,480 |
| Iterations 2–235 (excludes iteration 1, 24.9 s with warm compile) | 2,296.6 | 4,593.2 | 1,669.4 | 3,833,856 / 7,667,712 |

- **Phase seconds,** averaged over iterations 46–235: rollout 3.2–3.4,
  teacher precompute 0.89, update 2.7, iteration 6.9.
- **Native step:** 38.0 ms (rank 0) and 36.8 ms (rank 1) per step for 128
  envs at `native_threads` 2. That is about 2.4 s of the roughly 3.3 s
  rollout, so the native env is the largest single cost in an iteration.
  `native_threads` was not varied.
- **Iteration 1 compile:** iteration 1 reused the compile cache from 6.1 on
  the same pod, so its compile time does not represent a cold start.

## Memory, dense states included (235 iterations span 20 full 720-step games)

The peak allocated per phase is the largest on either rank. The target is
83,204 MiB allocated.

| Phase | Peak allocated | Share of target |
|---|---|---|
| Teacher precompute | 38,457 MiB | 46 % |
| Update | 25,970 MiB | 31 % |
| Rollout | 3,446 MiB | 4 % |

Peak reserved was 64,182 MiB. The nvidia-smi peak was 65,776 MiB per GPU.
This confirms the 6.1 split of spm 8 / accum 1 on dense states.

## Teacher telemetry

- `teacher/cache_bytes` was 1,674,575,872 B in every iteration.
- `teacher/kl` rose steadily. Block means over iterations: 2.32 (1–45),
  4.64 (46–90), 5.17 (91–135), 4.71 (136–180), 5.11 (181–225). The first
  iteration was 0.0026 and single iterations peaked at 12.5.
- `teacher/value_cross_entropy` went from 0.656 to 0.692 (≈ ln 2). The
  teacher's fresh critic head predicts about 50/50.
- There was no promotion and no evaluation (`checkpoint_freq` 20M), so the
  teacher stayed the loaded BC model with its fresh critic head throughout.

## Losses, entropy and banks (trend)

The block means below are over 45 iterations. Per-iteration values swing
with the game phase, because all 128 envs per rank reset together, so each
720-step game spans about 11.25 iterations.

| Iterations | Entropy | Teacher KL | Value loss | Policy loss | Approx KL | Clip fraction | Explained variance | Grad norm |
|---|---|---|---|---|---|---|---|---|
| 1–45 | 6.29 | 2.32 | 5.0e-3 | 0.041 | 0.069 | 0.41 | 0.55 | 12.9 |
| 46–90 | 7.86 | 4.64 | 4.7e-4 | 0.012 | 0.027 | 0.28 | 0.51 | 3.25 |
| 91–135 | 8.71 | 5.17 | 4.6e-4 | 0.003 | 0.012 | 0.16 | 0.55 | 1.98 |
| 136–180 | 7.03 | 4.71 | 3.7e-5 | 0.002 | 0.008 | 0.10 | 0.58 | 1.59 |
| 181–225 | 7.12 | 5.11 | 8.7e-5 | 0.002 | 0.007 | 0.10 | 0.60 | 1.48 |

The learning rate warmed up from 3.2e-5 to 2.0e-3 over the first 1,000
optimizer steps (about 63 iterations).

The completed-game raw final banks, rank-reduced means of seat 0 / seat 1,
were:

| Game | Iteration | Seat 0 bank | Seat 1 bank |
|---|---|---|---|
| 1 | 12 | 73,673 | 72,690 |
| 2 | 23 | 55,496 | 54,423 |
| 3 | 34 | 7,324 | 7,644 |
| 4 | 45 | 89 | 94 |
| 5 | 57 | 21 | 21 |

From game 6 on, the banks are between 0 and 54, and at or below 2 from game
11 on. For comparison, the BC policy's evaluation games in 6.1 ended near
70,000 per seat.

**Reading:** the PPO updates carried the self-play policy away from the BC
behaviour (teacher KL of about 5 nats, higher entropy) into a no-economy
equilibrium. Both seats end near zero, the value target becomes nearly
constant (value loss about 1e-5), and the clip fraction was 0.4–0.5 early.

**Unresolved attribution.** The candidates are:

- a decision or architecture error, such as advantage noise from the fresh
  critic head under `normalize_advantages`;
- a teacher KL coefficient of 0.005 that is too weak to anchor the policy;
- the reward shaping or win_loss objective favouring inaction once both
  seats decline;
- an execution error.

The execution-error candidate is weakest: the alarm, the finite metrics and
the 16 steps per iteration all held. This run cannot tell the other
candidates apart. The next discriminating runs would vary the teacher KL
coefficient, keep the BC critic head, or freeze the actor while the critic
warms up. Each is a new run statement, and none has been started.

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../ablation/final-report.md`.

- **Outcome:** Execution PASS; learning FAIL. The self-play economy collapsed: game-end banks 73,182, 54,960, 7,484 and 92 at games 1–4, near zero afterwards.
- **Denominators:** 235 complete iterations on both ranks; 3,760 optimizer steps (16 per iteration); 3,850,240 global env steps (235 × 16,384); 5,120 completed games (20 × 256 envs); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/7k07gp7c
- **Spend:** about $1.98 (1,704 s wall at $4.18/h) at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** the collapse's cause is attributed in `../ablation/final-report.md`, not here; single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; the LR warm-up ended at iteration 63, so iterations 1–62 ran below the peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
