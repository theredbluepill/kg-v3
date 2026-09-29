# Result: PPO collapse ablation G, "ablate-G-lr10-critic-stopgrad" (LR / 10, critic stop-gradient)

**Execution: PASS. Prediction: not met.** The prediction assumed the value
path through the trunk was the main driver. It failed on all three counts:

- **Trunk:** it did not move far less than in B. Shared change was 7.81e-3
  at iteration 46, against B's 8.02e-3.
- **Teacher KL at iteration 46:** 0.58, not well below B's 0.64.
- **Banks:** not flat at about 70k. They went 70k, 65k, 67k, 63k over
  games 1–4, against B's 73k, 65k, 64k, 62k.

**One real effect:** at the same trunk step size, blocking the critic's
gradient roughly **halved the policy's drift from BC at every game end**.
Teacher KL was 0.021 / 0.13 / 0.31 / 0.69, against B's 0.039 / 0.20 / 0.56 /
1.36. That drift reduction did not raise the banks.

This arm used a **DIAGNOSTIC-ONLY launcher hook that changes training**
(`critic_stopgrad_launcher.py`). It is not a recipe. The run is a pre-landing
diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the
pod), with one seed, not Codex-verified. Run statement:
`run-statement.md`. Receipts: `pod-receipts/` and `dryrun-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/yz34n7i6`, state `finished`,
  online.
  - Renamed through the API after the run (`wb_rename_g.py`) to
    `ablate-G-lr10-critic-stopgrad` in group `kg-v3-ppo-collapse-ablation`.
  - The original name was `ppo-20260929-225057` in group `ppo`; job_type
    and tags are unchanged.
  - Summary `_step` 753,664 (`pod-receipts/wandb_yz34n7i6_state.json`).
- **Run:**
  - Exit 0 at `--max-env-steps 753664`, with 46 complete iterations on both
    ranks and 736 optimizer steps.
  - No nonfinite metric (the stop was armed), no traceback.
  - Launch 22:50:54Z, end 22:58:34Z, 460 s wall.
  - W&B end-of-run SPS: rollout 3,996, teacher 18,472, update 6,167 (B:
    3,868 / 18,447 / 6,154).
  - After the run both GPUs showed 0 MiB and 0 % with no compute apps
    (`idle_after.csv`).
- **Code identity:**
  - The nine shared input hashes in `pod-receipts/hashes.sha256` equal B's.
  - The extra lines are `run_g.sh` `a2fb7965…81b2`, D's unchanged audit hook
    `f98720bd…78f9` and the G hook `ee5af67b…d621`. All three are
    hash-equal to the committed copies and to the hooks verified in the
    final dry runs.
  - The overrides reached the trainer in argv as `rl.eval_replay_games=0`,
    `optimizer.muon_lr=0.0002` and `optimizer.adamw_lr=0.00001`. The logged
    LR equals B's at every iteration, and `teacher/kl_coef` is 0.005.
  - A fresh critic head was loaded: `critic_head` ||θ0|| is 22.6495 on both
    ranks, as in E and F.
  - The `diagnostic_hook` record shows `critic_stopgrad: on` and
    `donated_buffer_disabled: false`. The patched critic function was
    called 3,772 times by iteration 46 on rank 0.
- **Stop-gradient check in the main run** (iteration 1, both ranks,
  `pod-receipts/stopgrad_check.tsv`): **passed.**
  - Shared trunk: 157 of 157 tensors had no gradient from the value side.
  - Actor-only (48 tensors) and `critic_value_tokens`: none either.
  - `critic_head` gradient norm: 0.28 on rank 0 and 0.20 on rank 1.
- **Dry-run check** (`run-statement.md`; `dryrun-receipts/stopgrad_check_dryrun.tsv`):
  - `on`: exactly zero value-side gradient on the shared, actor-only and
    `critic_value_tokens` groups at iterations 1 and 2 on both ranks, and a
    non-zero gradient on `critic_head`.
  - `off` control: shared-trunk value-side gradient norm 1.06–1.29.
- **Checkpoint:** `checkpoint_final.pt` `5eb02765…730a`, kept on the pod.
- **Spend:**
  - Dry runs (26 + 39 + 43 + 40 s) and the main run (460 s) total 608 s,
    about $0.71 at $4.18/h.
  - The W&B rename and receipt extraction ran on the CPU in under a minute.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | G (LR / 10, critic stop-grad) | B (LR / 10) | E (LR / 10, vf 0.5) | F (LR / 10, anchor 0.1) |
|---|---|---|---|---|---|
| 1 | 12 | 69,745 / 70,492 | 73,360 / 73,208 | 75,427 / 74,969 | 73,184 / 73,736 |
| 2 | 23 | 64,807 / 64,358 | 65,621 / 65,125 | 74,130 / 73,792 | 70,793 / 73,094 |
| 3 | 34 | 67,432 / 65,656 | 63,689 / 63,913 | 73,160 / 72,706 | 70,481 / 71,115 |
| 4 | 45 | 63,701 / 63,126 | 62,104 / 62,321 | 66,869 / 67,109 | 67,282 / 66,337 |

## Per-iteration metrics at the key iterations (G / B)

Full table: `pod-receipts/iterations.tsv` (`../extract_iterations.py`).
`value_loss` is the unweighted `loss/value_loss` from the iteration records;
the trainer reduces metrics across ranks.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0014 / 0.0013 | 0.0007 / 0.0007 | 0.0003 / 0.0003 | 0.0015 / 0.0014 | 0.004 / 0.003 | 0.192 / 0.204 | 0.02 / −0.06 | 0.019 / 0.021 |
| 13 | 0.033 / 0.059 | 0.020 / 0.033 | 0.007 / 0.005 | 0.0044 / 0.0044 | 0.045 / 0.043 | 0.185 / 0.165 | 0.12 / 0.17 | 0.017 / 0.013 |
| 23 | 0.13 / 0.20 | 0.095 / 0.12 | 0.010 / 0.027 | 0.0070 / 0.0066 | 0.085 / 0.082 | 0.282 / 0.237 | −0.12 / 0.00 | 0.040 / 0.028 |
| 34 | 0.31 / 0.56 | 0.20 / 0.42 | 0.073 / 0.068 | 0.0094 / 0.0105 | 0.123 / 0.140 | 0.267 / 0.223 | 0.03 / 0.03 | 0.036 / 0.025 |
| 46 | 0.58 / 0.64 | 0.31 / 0.34 | 0.15 / 0.11 | 0.0114 / 0.0114 | 0.138 / 0.153 | 0.174 / 0.077 | −0.13 / −0.36 | 0.015 / 0.003 |

- **Teacher KL at game ends** (iterations 12, 23, 34, 45): G 0.021 / 0.13 /
  0.31 / 0.69 and B 0.039 / 0.20 / 0.56 / 1.36. Other arms without the
  anchor: E 0.029 / 0.15 / 0.45 / 1.14, D 0.046 / 0.29 / 0.81 / 1.41.
  Means over iterations 1–46: G 0.24, B 0.36, E 0.32, D 0.48, F 0.10.
  - Iteration 46 is the first update of game 5, where KL resets downward,
    so it understates the gap. At iteration 45, G's 0.69 is about half of
    B's 1.36.
  - Market-kind KL was not reduced, and at 34 and 46 it was slightly higher
    than B's. The reduction is in unit-kind KL.
- **Per-update policy change is unchanged.** Approx KL and clip fraction
  track B's.
- **Critic quality is worse than B's.** Mean explained variance over
  iterations 1–46 was G 0.16, B 0.29, F 0.23, E −0.10 and D 0.84. Late-run
  value loss was 5x B's (0.015 against 0.003 at 46).
- **Grad norms.** Total `optimizer/grad_norm` ranged 12.0–13.9 (B
  12.4–15.1), above `max_grad_norm` 10 in every iteration, as in B/E/F.
  The teacher value loss stayed at 0.0032–0.0033 throughout, as in B.

## Trunk audit (telemetry only, ||θ − θ0|| / ||θ0|| from the loaded weights)

In-run records are taken after the listed iteration's last optimizer step
(`pod-receipts/trunk_audit.tsv`, from D's unchanged `trunk_audit_summary.py`).
Both ranks are equal to every printed digit. The `critic_head` and
`critic_value_tokens` split comes from G's `critic_split_audit` records.

| After iteration | Actor-only | Critic-only | (critic_head / critic_value_tokens) | Shared trunk |
|---|---|---|---|---|
| 12 | 6.21e-4 | 9.76e-4 | 9.78e-4 / 2.19e-4 | 9.92e-4 |
| 23 | 1.76e-3 | 3.41e-3 | 3.42e-3 / 5.11e-4 | 2.72e-3 |
| 34 | 3.24e-3 | 7.19e-3 | 7.21e-3 / 7.93e-4 | 4.92e-3 |
| 46 | 5.17e-3 | 1.27e-2 | 1.27e-2 / 1.36e-3 | 7.81e-3 |

References at iteration 46:

- E (same fresh-head load): 5.10e-3 / 1.16e-2 / 7.86e-3;
- F: 5.14e-3 / 1.08e-2 / 7.78e-3;
- D (BC critic): 5.07e-3 / 7.50e-3 / 8.43e-3;
- B (checkpoint vs BC only): actor-only 5.12e-3, shared 8.02e-3.

G's shared-trunk trajectory (9.9e-4, 2.72e-3, 4.92e-3, 7.81e-3) is within
4 % of F's at every audited iteration. `critic_value_tokens` moved about
10x less than the head; they now move only under the policy gradient.

**Gradient audit.** Norms were read on rank 0 before the first optimizer
step of each iteration, after DDP:

- Shared trunk: 10.59 at iteration 1 and 10.11–15.45 over all 46. F had
  10.61–14.70 and E 10.38–16.21.
- Actor-only: 2.34–5.82.
- Critic-only: 0.34–0.80.

Removing the critic's contribution (the value-side shared norm of about
1.2 in the dry runs) did not lower the shared-trunk gradient norm range
noticeably.

## Reading

**Prediction check** (hypothesis: the value path through the trunk is the
main driver):

- **Trunk movement far less than B's: not met.** It was 7.81e-3 against
  8.02e-3 (and E's 7.86e-3). As the run statement expected, the Muon step
  size, not the gradient mix, sets the trunk's movement at a given LR.
- **Teacher KL at 46 well below 0.64: not met** at iteration 46 (0.58).
  It was met at the game ends: about 0.5x B's at 23, 34 and 45.
- **Banks about 70k and flat: not met.** Banks fell 70k to 63k (−9 %),
  against B's 73k to 62k (−15 %).
- **Discriminating observation 2 holds for teacher KL, not for banks.** The
  trunk changed like B's and teacher KL was clearly lower, but the banks
  were only marginally different: +1.5k at game 4, −3k at game 1.

**Supported (one seed):**

1. **The critic's gradient through the shared trunk accounts for roughly
   half of the policy's distance from BC at LR / 10, without changing how
   far the trunk moves.** Game-end teacher KL was 0.5–0.6x B's at 23, 34
   and 45. G's 0.69 at game 4 is outside the 1.14–1.41 range of the three
   other arms without the anchor (B, D, E). The effect is on direction:
   with the value term gone, the same-size trunk steps disturb the actor's
   input features less. This is inferred from equal trunk change, equal
   approx KL and lower teacher KL.
2. **Less drift from BC again did not hold the economy.** This is the
   same outcome as F (anchor 0.1: teacher KL 0.22 at game 4, banks 67k).
   Across B, E, F and G, game-4 banks are 62–67k, while game-4 teacher KL
   spans 0.22–1.36. The size of the KL distance from BC does not predict
   the bank decline at this horizon.
3. **A critic head on value-blind trunk features is a weaker critic.** Mean
   explained variance was 0.16 against B's 0.29, and late value loss was
   5x B's. D, the only arm whose banks rose, had the best critic
   (explained variance 0.84). Here, cutting the critic's influence on the
   trunk bought closeness to BC at the price of advantage quality. The
   banks did not move either way, which fits D's reading that advantage
   quality, not drift magnitude, is the lever that mattered.

**Unresolved attribution and limits:**

- **Seed noise.** There is one seed. G's game-1 banks (70k) are about 3k
  below B's before any meaningful policy change (teacher KL 0.02). The
  game-1 spread across arms is 70–75k, so G-vs-B bank differences of 1–3k
  are within the unsized noise. The teacher-KL halving is larger than the
  spread among the unanchored arms, but it is not sized by a second seed
  either.
- **Teacher value distillation.** Its trunk gradient was blocked along
  with the value loss (same critic function). Its coefficient is 0.005
  against 2.0, but it was not separated.
- **Frozen-feature critic versus trunk protection.** G changes two things
  at once by construction: the trunk loses the value gradient, and the
  critic loses adaptable features. The worse advantages may have cancelled
  a real benefit of the protected trunk. A stop-grad arm with the BC critic
  head (D's load) would separate them.
- **Which actions drive the decline.** As in F, the per-family cause of the
  game 3 to game 4 decline was not measured. Market-kind KL was not
  reduced by the stop-gradient. It was slightly above B's at 34 and 46,
  and it peaked at 0.17 at iteration 45.
- **Strength.** Self-play banks are not strength. No held-out opponent ran
  (`rl.eval_replay_games=0`).
- **Dry-run nondeterminism.** Iteration-1 grad norms differed between the
  `on` and `off` dry runs, although forward values are identical. This was
  not investigated. It does not affect the main run's check or audit.

**Consequence.** At LR / 10, the fresh critic's trunk gradient is a real
but secondary contributor: it doubles drift from BC and leaves the banks
unchanged. It is not the collapse driver that arm C's full-LR value-only
warm-up suggested. The next discriminating arm is the BC critic head (D's
load) with this stop-gradient. That would keep D's informative advantages
while removing the critic's pull on the trunk, and test whether the two
together hold or raise the banks.

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../final-report.md`.

- **Outcome:** Execution PASS. Decline: banks 70,118 → 63,414 by game 4 (−6,704); critic weaker (EV mean 0.16).
- **Denominators:** 46 complete iterations on both ranks; 736 optimizer steps (16 per iteration); 753,664 global env steps (46 × 16,384); 1,024 completed games (4 game phases × 256 envs, `train/total_games_played` 1024); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/yz34n7i6
- **Spend:** about $0.71 (main run and dry runs) at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; all 46 iterations lie inside the 1,000-step LR warm-up (LR at iteration 46 is 0.736 of peak), so nothing here was measured at peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
