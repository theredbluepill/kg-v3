# Result: PPO collapse ablation F, "ablate-F-lr10-anchor01" (LR / 10, teacher_kl_coef 0.1)

**Execution: PASS. Prediction: met** (teacher KL 0.21 at iteration 46 against
B's 0.64; game-4 banks 67k against B's 62k). **But the reading is narrower
than "the levers combine":** the anchor held the policy 3x closer to BC than
B at every game end, yet the banks still fell 9 % over 4 games and ended
equal to E's (67k), whose teacher KL was 0.63.

- **Banks:** 73k, 72k, 71k, 67k over games 1–4 (B: 73k, 65k, 64k, 62k;
  E: 75k, 74k, 73k, 67k).
- **Teacher KL at game ends** (iterations 12, 23, 34, 45): 0.018 / 0.072 /
  0.159 / 0.218, against B's 0.039 / 0.20 / 0.56 / 1.36 (0.47x, 0.35x, 0.28x,
  0.16x of B's). 0.21 at iteration 46. Mean over 1–46: F 0.098, B 0.36,
  E 0.32.
- **Trunk movement:** shared 7.78e-3 at iteration 46 (B 8.02e-3, E 7.86e-3),
  actor-only 5.14e-3 (B 5.12e-3). The anchor changed the direction of the
  update, not its size.
- **Critic:** mean explained variance 0.23 over iterations 1–46 (B 0.29,
  E −0.10, D 0.84).

Pre-landing diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain
lines on the pod). One seed, not Codex-verified. Receipts in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/nur61v3a`, state `finished`,
  online. Renamed through the API after the run (`wb_rename_f.py`, this
  folder) to `ablate-F-lr10-anchor01` in group `kg-v3-ppo-collapse-ablation`
  (original name `ppo-20260929-222809`, group `ppo`; job_type and tags
  unchanged). Summary `_step` 753,664 (`pod-receipts/wandb_nur61v3a_state.json`).
- **Run:**
  - Exit 0 at `--max-env-steps 753664`. 46 complete iterations on both ranks,
    736 optimizer steps. No nonfinite metric (the stop was armed and never
    fired), no traceback.
  - The lock was free and the GPUs idle at 22:28:06Z. The run ended at
    22:36:09Z after 483 s wall. End-of-run W&B summary: 2,089 game SPS
    (rollout 3,904, teacher 18,471, update 5,944). Slower than E (2,355 game
    SPS, rollout 4,818). Not investigated; the pod was not shared during the
    run (lock held, no other compute apps before launch).
  - After the run, both GPUs showed 0 MiB and 0 % with no compute apps
    (`idle_after.csv`).
- **Code identity:**
  - The nine shared input hashes in `pod-receipts/hashes.sha256` equal B's.
  - The two extra lines are E's launcher `run_e.sh` `2fed2fe3…7a20`, reused
    unchanged and equal to the committed copy, and D's hook `f98720bd…78f9`,
    also unchanged.
  - The overrides reached the trainer as `['rl.eval_replay_games=0',
    'optimizer.muon_lr=0.0002', 'optimizer.adamw_lr=0.00001',
    'rl.teacher_kl_coef=0.1']`. Logged `teacher/kl_coef` is 0.1 in every
    iteration, and the logged LR equals B's.
  - A fresh critic head was loaded: its ||θ0|| is 22.6495 on both ranks, as
    in E. The actor-only and shared ||θ0|| equal the BC weights'.
- **Checkpoint:** `checkpoint_final.pt` `c80828b4…8822`, kept on the pod.
- **Spend:** 483 s of GPU run, about $0.56 at $4.18/h. There was no dry
  run. The W&B and receipt calls ran on the CPU in under a minute.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | F (LR / 10, anchor 0.1) | B (LR / 10, anchor 0.005) | E (LR / 10, vf 0.5) | A (full LR, anchor 0.1) |
|---|---|---|---|---|---|
| 1 | 12 | 73,184 / 73,736 | 73,360 / 73,208 | 75,427 / 74,969 | 71,943 / 70,933 |
| 2 | 23 | 70,793 / 73,094 | 65,621 / 65,125 | 74,130 / 73,792 | 59,073 / 61,208 |
| 3 | 34 | 70,481 / 71,115 | 63,689 / 63,913 | 73,160 / 72,706 | 51,212 / 49,184 |
| 4 | 45 | 67,282 / 66,337 | 62,104 / 62,321 | 66,869 / 67,109 | 40,408 / 42,457 |

## Per-iteration metrics at the key iterations (F / B)

Full table: `pod-receipts/iterations.tsv` (`../extract_iterations.py`).
`value_loss` is the unweighted `loss/value_loss` from the rank-0 iteration
records.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0014 / 0.0013 | 0.0007 / 0.0007 | 0.0003 / 0.0003 | 0.0014 / 0.0014 | 0.004 / 0.003 | 0.203 / 0.204 | 0.03 / −0.06 | 0.021 / 0.021 |
| 13 | 0.025 / 0.059 | 0.016 / 0.033 | 0.004 / 0.005 | 0.0044 / 0.0044 | 0.045 / 0.043 | 0.173 / 0.165 | 0.16 / 0.17 | 0.015 / 0.013 |
| 23 | 0.072 / 0.20 | 0.049 / 0.12 | 0.007 / 0.027 | 0.0060 / 0.0066 | 0.072 / 0.082 | 0.246 / 0.237 | 0.00 / 0.00 | 0.030 / 0.028 |
| 34 | 0.16 / 0.56 | 0.11 / 0.42 | 0.012 / 0.068 | 0.0109 / 0.0105 | 0.147 / 0.140 | 0.231 / 0.223 | 0.02 / 0.03 | 0.027 / 0.025 |
| 46 | 0.21 / 0.64 | 0.12 / 0.34 | 0.030 / 0.11 | 0.0118 / 0.0114 | 0.150 / 0.153 | 0.088 / 0.077 | −0.08 / −0.36 | 0.004 / 0.003 |

- **Per-update policy change is unchanged.** Approx KL and clip fraction
  track B's closely throughout. The anchor did not shrink the steps; it bent
  them back toward the teacher.
- **Loss terms.** The weighted `teacher_kl_loss` (0.1 × KL) was 0.0025,
  0.0072, 0.016 and 0.021 at iterations 13, 23, 34 and 46. It was 2.9–3.0x
  the policy loss at each of those points (0.0009, 0.0025, 0.0051, 0.0072).
- **Clipping.** Total `optimizer/grad_norm` was 12.0–15.6 (B 12.4–15.4),
  above `max_grad_norm` 10 in every iteration, so the clip bound in both
  arms.
- **Entropy.** It rose within each game to 7.7–8.2 late in the game and
  reset near 4.2–4.4 at game start, the same pattern as B (7.3–8.6). This
  swing is not driven by the anchor. A's full-LR climb to 10.9 did not
  occur.

## Trunk audit (telemetry only, ||θ − θ0|| / ||θ0|| from the loaded weights)

These are in-run records after the listed iteration's last optimizer step
(`pod-receipts/trunk_audit.tsv`, from D's `trunk_audit_summary.py`). Both
ranks are equal to every printed digit. The hook is DIAGNOSTIC-ONLY and
telemetry-only (`diagnostic_hook` record in the log).

| After iteration | Actor-only | Critic-only | Shared trunk |
|---|---|---|---|
| 12 | 6.27e-4 | 1.02e-3 | 1.03e-3 |
| 23 | 1.77e-3 | 3.26e-3 | 2.80e-3 |
| 34 | 3.23e-3 | 6.38e-3 | 5.00e-3 |
| 46 | 5.14e-3 | 1.08e-2 | 7.78e-3 |

The two arms that share this load have in-run audits: E (same fresh-head
load) had 5.10e-3 / 1.16e-2 / 7.86e-3 at iteration 46, and D (BC critic)
had 5.07e-3 / 7.50e-3 / 8.43e-3. B has only its checkpoint-vs-BC change:
actor-only 5.12e-3 and shared 8.02e-3.

**Gradient audit.** Norms were read before the first optimizer step of
each iteration, on rank 0:

- Shared trunk: 11.42 at iteration 1 and 10.61–14.70 over all 46 (E
  10.38–16.21, D 10.55–16.24).
- Actor-only: 3.48 at iteration 1 and 2.70–5.32 over all 46.
- Critic-only: 0.48 at iteration 1 and 0.33–0.64 over all 46.

## Reading

**Prediction check.** The hypothesis was that the anchor and the smaller
steps combine.

- **Teacher KL at iteration 46 below 0.64: met,** at 0.21, and below B's
  at every game end by a widening factor (0.47x to 0.16x).
- **Game-4 banks at least 62k: met,** at 67k / 66k.
- **Discriminating observation 1 holds by the letter.** The more
  informative comparison is with E. E reached the same game-4 banks at
  3x F's teacher KL, and E was about 2k higher than F in games 1–3.

**Supported (one seed):**

1. **At LR / 10, a 0.1 anchor holds the policy much closer to the BC
   teacher without changing the step size.** Trunk movement, approx KL and
   clip fraction equal B's. Teacher KL, unit-kind KL and market-kind KL are
   3.5x, 3.8x and 5.6x lower at iteration 34. At the full LR (A), the same coefficient could not
   hold teacher KL below 2 nats, so the two levers do interact.
2. **Staying close to the teacher does not by itself stop the bank decline
   at this horizon.** F's economy still fell from 73k to 67k (−9 %). The
   decline between games 3 and 4 (−4k) came while teacher KL was only 0.16
   to 0.22. E fell by the same amount with 3x the teacher KL. So the size
   of the policy's KL distance from BC is not what sets the banks here;
   which actions moved matters more than how far the policy moved. This is
   an inference from two arms, not a measured mechanism.
3. **The critic was B-like, not D-like.** Mean explained variance was 0.23
   against B's 0.29 and D's 0.84. D, the only arm whose banks rose, differs
   from F in the critic, not in closeness to BC (D's mean teacher KL was
   0.48, the highest of B, D, E and F). This favours D's reading: advantage
   quality is the stronger lever at LR / 10.

**Unresolved attribution and limits:**

- **Seed noise.** There is one seed per arm. The F-vs-E and F-vs-B bank
  differences (0–9k) are of the order of the game-1 spread across arms
  (72–75k), and no second seed exists to size it.
- **Anchor versus learning.** A low teacher KL is partly true by
  construction, since the teacher is the anchor target. F cannot separate
  "held near BC" from "learned less"; BC's own self-play banks are the
  baseline.
- **Strength.** Self-play banks are not strength. No held-out opponent ran
  (`rl.eval_replay_games=0`), so there is no win rate or bank margin.
- **Late-game decline.** Which action families carry the decline between
  games 3 and 4 at low KL was not measured. The market-kind KL tripled
  between iterations 45 and 46 (0.012 to 0.030), but that is a single
  boundary point.
- **Throughput.** The lower rollout SPS than E (3.9k against 4.8k) is
  unexplained. It does not affect the per-iteration metrics.

**Consequence.** The anchor at LR / 10 is a reliable brake on distance
from BC, but not a repair of the economy. The next discriminating arm is
probably D's BC critic combined with this anchor, which would test whether
the anchor adds anything once the advantages are informative.

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../final-report.md`.

- **Outcome:** Execution PASS. Decline: banks 73,460 → 66,810 by game 4 (−6,650) with teacher KL held at 0.22 at the game-4 end.
- **Denominators:** 46 complete iterations on both ranks; 736 optimizer steps (16 per iteration); 753,664 global env steps (46 × 16,384); 1,024 completed games (4 game phases × 256 envs, `train/total_games_played` 1024); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/nur61v3a
- **Spend:** about $0.56 at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; all 46 iterations lie inside the 1,000-step LR warm-up (LR at iteration 46 is 0.736 of peak), so nothing here was measured at peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
