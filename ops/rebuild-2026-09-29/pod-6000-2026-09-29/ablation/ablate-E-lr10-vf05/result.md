# Result: PPO collapse ablation E, "ablate-E-lr10-vf05" (LR / 10, vf_coef 0.5)

**Execution: PASS. Prediction: met on all three clauses by the letter, but
two of them only by margins inside the round's noise; the mechanism reading
does not support "the value weight drives the drift".**

- **Banks:** 75k, 74k, 73k, 67k over games 1–4 (B: 73k, 66k, 64k, 62k). Above
  B in every game, but still declining, and already 2k above B in game 1,
  when both policies were within teacher KL 0.04 of BC.
- **Trunk movement:** shared 7.86e-3 at iteration 46 (B 8.02e-3, −2 %),
  actor-only 5.10e-3 (B 5.12e-3). Equal to within the D-vs-B spread (5 %).
- **Teacher KL:** 0.63 at iteration 46 (B 0.64). At matched game ends
  (iterations 12, 23, 34, 45) it was 0.030 / 0.15 / 0.45 / 1.14, against B's
  0.039 / 0.20 / 0.56 / 1.36, so 0.76x, 0.75x, 0.81x and 0.83x of B's.
- **Critic:** mean explained variance over iterations 1–46 was −0.10
  (B 0.29, D 0.84). The fresh critic learned worse with a 4x smaller value
  weight in the shared trunk.

Pre-landing diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain
lines on the pod). One seed, not Codex-verified. Receipts in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/hi2lqsrx`, state `finished`,
  online. Renamed through the API after the run (`../ablate-E-lr10-vf05/wb_rename_e.py`)
  to `ablate-E-lr10-vf05` in group `kg-v3-ppo-collapse-ablation` (original
  name `ppo-20260929-221548`, group `ppo`; job_type and tags unchanged).
  Summary `_step` 753,664 (`pod-receipts/wandb_hi2lqsrx_state.json`).
- **Run:**
  - Exit 0 at `--max-env-steps 753664`; 46 complete iterations on both
    ranks; 736 optimizer steps. No nonfinite metric (stop armed, never
    fired), no traceback.
  - The lock was free and the GPUs idle at 22:15:45Z; ended 22:23:12Z,
    447 s wall. End-of-run W&B summary: 2,355 game SPS (rollout 4,818,
    teacher 18,474, update 6,141), comparable to D; not investigated further.
  - After the run: both GPUs 0 MiB / 0 %, no compute apps (`idle_after.csv`).
- **Code identity:**
  - The nine shared input hashes in `pod-receipts/hashes.sha256` equal B's;
    the two extra lines are `run_e.sh` `2fed2fe3…7a20` (equal to the
    committed copy, commit `6d1aa9f`) and D's unchanged hook `f98720bd…78f9`.
  - Overrides reached the trainer as `['rl.eval_replay_games=0',
    'optimizer.muon_lr=0.0002', 'optimizer.adamw_lr=0.00001',
    'rl.vf_coef=0.5']`; the logged LR equals B's at every iteration.
  - A fresh critic head was loaded: its ||θ0|| is 22.6495 on both ranks
    (the BC head's is 22.0106), and the actor-only and shared ||θ0|| equal
    the BC weights'.
- **Checkpoint:** `checkpoint_final.pt` `4fcf9de6…280a`, kept on the pod.
- **Spend:** 447 s of GPU run, about $0.52 at $4.18/h (no dry run; W&B and
  receipt calls are CPU-only and under a minute).

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | E (LR / 10, vf 0.5) | B (LR / 10, vf 2.0) | D (LR / 10, BC critic) | Control 6.2 |
|---|---|---|---|---|---|
| 1 | 12 | 75,427 / 74,969 | 73,360 / 73,208 | 73,073 / 71,405 | 73,673 / 72,690 |
| 2 | 23 | 74,130 / 73,792 | 65,621 / 65,125 | 76,902 / 75,924 | 55,496 / 54,423 |
| 3 | 34 | 73,160 / 72,706 | 63,689 / 63,913 | 80,549 / 78,593 | 7,324 / 7,644 |
| 4 | 45 | 66,869 / 67,109 | 62,104 / 62,321 | 85,232 / 82,347 | 89 / 94 |

## Per-iteration metrics at the key iterations (E / B)

Full table: `pod-receipts/iterations.tsv` (`../extract_iterations.py`).
`value_loss` is the unweighted `loss/value_loss` from the rank-0 iteration
records, so E and B compare directly.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0014 / 0.0013 | 0.0007 / 0.0007 | 0.0003 / 0.0003 | 0.0014 / 0.0014 | 0.003 / 0.003 | 0.185 / 0.204 | −0.32 / −0.06 | 0.017 / 0.021 |
| 13 | 0.049 / 0.059 | 0.023 / 0.033 | 0.008 / 0.005 | 0.0044 / 0.0044 | 0.046 / 0.043 | 0.161 / 0.165 | −0.30 / 0.17 | 0.013 / 0.013 |
| 23 | 0.15 / 0.20 | 0.099 / 0.12 | 0.021 / 0.027 | 0.0063 / 0.0066 | 0.075 / 0.082 | 0.254 / 0.237 | −0.03 / 0.00 | 0.032 / 0.028 |
| 34 | 0.45 / 0.56 | 0.33 / 0.42 | 0.041 / 0.068 | 0.0108 / 0.0105 | 0.140 / 0.14 | 0.239 / 0.223 | 0.05 / 0.03 | 0.028 / 0.025 |
| 46 | 0.63 / 0.64 | 0.37 / 0.34 | 0.055 / 0.11 | 0.0118 / 0.0114 | 0.140 / 0.15 | 0.104 / 0.077 | −0.31 / −0.36 | 0.005 / 0.003 |

- Mean teacher KL over iterations 1–46: E 0.32, B 0.36, D 0.48.
- Mean explained variance: E −0.10, B 0.29, D 0.84. E's was negative
  through most of game 1 and game 2 (−0.42 to −0.02 in iterations 1–9 and
  13–20), where B's was mostly 0.2–0.5.
- Approx KL and clip fraction track B's closely throughout: per-update
  policy change is the same size.
- Total `optimizer/grad_norm` was 12.1–17.6 (B 12.4–15.4), above
  `max_grad_norm` 10 in every iteration, so the clip bound in both arms. The
  confound named in the statement (the clip ceasing to bind) did not occur.

## Trunk audit (telemetry only, ||θ − θ0|| / ||θ0|| from the loaded weights)

In-run records after the listed iteration's last optimizer step
(`pod-receipts/trunk_audit.tsv`, D's `trunk_audit_summary.py`). Both ranks
equal to every printed digit.

| After iteration | Actor-only | Critic-only | Shared trunk |
|---|---|---|---|
| 12 | 6.21e-4 | 1.02e-3 | 1.00e-3 |
| 23 | 1.76e-3 | 3.37e-3 | 2.75e-3 |
| 34 | 3.23e-3 | 6.84e-3 | 4.98e-3 |
| 46 | 5.10e-3 | 1.16e-2 | 7.86e-3 |

References at iteration 46: B (checkpoint vs BC, fresh head so no critic
θ0) actor-only 5.12e-3, shared 8.02e-3; D (in-run) 5.07e-3 / 7.50e-3
critic / 8.43e-3 shared.

**Gradient audit** (before the first optimizer step of each iteration,
rank 0):

- Shared-trunk gradient norm: 11.95 at iteration 1, 10.38–16.21 over all
  46 iterations. D (vf 2.0, BC head) had 12.96 at iteration 1 and
  10.55–16.24 over all 46 (recomputed here from D's `trunk_audit.tsv`). Cutting the
  value weight 4x barely changed the shared gradient norm, so the trunk
  gradient at these points is dominated by the policy and teacher terms, not
  by the value term.
- Critic-only: 0.41 at iteration 1, 0.32–0.72 over all 46 (D 1.19 at
  iteration 1, 0.35–1.97).
- Actor-only: 2.76–5.05 over all 46 (D 2.85–5.53).

## Reading

**Prediction check** (hypothesis: the value-loss weight through the trunk
drives the drift):

- Less trunk movement than B: **met by the letter only.** 7.86e-3 against
  8.02e-3 is −2 %, smaller than the 5 % D-vs-B difference. As expected before
  launch, movement magnitude is set by the step size.
- Teacher KL at iteration 46 below 0.64: **met by the letter only.** 0.634
  against 0.64. At matched game ends E was a consistent 0.75–0.83x of B,
  which is the more informative comparison.
- Game-4 banks at least 62k: **met.** 67k / 67k.

**Supported (one seed):**

1. **The value term is not the dominant part of the shared-trunk gradient
   at LR / 10.** The shared gradient norm hardly moved when the value weight
   fell 4x, and trunk movement was unchanged. The round-1 premise, built on
   the synthetic handoff probe (12.4 against 0.118), again does not describe
   real rollouts; D found the same from the other side.
2. **A smaller value weight slowed the drift from BC modestly**
   (teacher KL 0.75–0.83x B's at game ends) **and the bank decline
   modestly** (game 4: 67k against 62k). The economy still fell 11 % from
   game 1 to game 4, against D's 17 % rise with the BC critic.
3. **It did so while the critic got worse** (mean explained variance −0.10
   against 0.29). So E's gain did not come through better advantages, which
   is D's route. The more likely route is that the policy and teacher terms
   now set a larger share of each (clipped, normalised) trunk step, so the
   teacher KL anchor at 0.005 carries relatively more weight; that is an
   inference, not measured.

**Unresolved attribution and limits:**

- The bank gains (2–9k) are of the order of the game-1 difference (2k) that
  arose while both policies were still within teacher KL 0.04 of BC. No
  second seed exists to size seed noise; the ranking E > B is not
  established.
- Self-play banks are not strength; no held-out opponent ran
  (`rl.eval_replay_games=0`), so there is no win rate or bank margin.
- B has no in-run grad audit, so the E-vs-B shared-gradient composition is
  compared with D (a different head), not with B directly.
- The critic-only group moved more in E (1.16e-2) than in D (7.50e-3); B's
  critic movement is unknown (no θ0 for a fresh head in B's checkpoint).
- The mechanism behind E's lower teacher KL (relative teacher-term weight
  versus a changed clip scale) was not isolated.

**Consequence:** round 2 so far points at the critic's *quality* (D: BC
critic, informative advantages, rising banks) rather than the value loss's
*weight* in the trunk (E: small, mixed gain with a worse critic) as the
stronger lever at LR / 10.

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../final-report.md`.

- **Outcome:** Execution PASS. Decline: banks 75,198 → 66,989 by game 4 (−8,209), within the unsized spread of B; critic worse (EV mean −0.10).
- **Denominators:** 46 complete iterations on both ranks; 736 optimizer steps (16 per iteration); 753,664 global env steps (46 × 16,384); 1,024 completed games (4 game phases × 256 envs, `train/total_games_played` 1024); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/hi2lqsrx
- **Spend:** about $0.52 at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; all 46 iterations lie inside the 1,000-step LR warm-up (LR at iteration 46 is 0.736 of peak), so nothing here was measured at peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
