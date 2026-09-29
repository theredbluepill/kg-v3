# PPO collapse ablations A, B, C against the 6.2 control, iterations 1–46

**Pre-landing diagnostic; single seed per arm.** Every number and reading
below comes from one 2-rank run per arm (rank seeds 0/1) on the unverified
merge `kg/pod-ppo-prelanding` `e74d67e`. None of it is Codex-verified. It
supports no selection, ranking or submission. The measurements are in the
first half. The interpretation is in the second half and is marked as such.

## Arms

All four arms use the same nine input hashes: config, model config,
launcher, `run_ppo.py`, `ppo.py`, `logging.py`, `env.py`, `rs.abi3.so` and
BC best `fd854587…6f51` with `model_fresh_critic_head`. Each run's result.md
records this. All four also use 2 ranks, `-o rl.eval_replay_games=0`, 16
optimizer steps per iteration, and no teacher promotion inside 46
iterations.

| Arm | Single change versus the control | W&B | Result |
|---|---|---|---|
| Control 6.2 | None: Muon 0.002 / AdamW 1e-4, `teacher_kl_coef` 0.005. Runtime stop at 235 iterations; only iterations 1–46 are compared here. | `spoon/kg-v3/7k07gp7c` | `../6.2/result.md` |
| A `ablate-A-anchor` | `rl.teacher_kl_coef=0.1` | `spoon/kg-v3/y8j6vzky` | `ablate-A-anchor/result.md` |
| B `ablate-B-lr10` | `optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001` (both LRs / 10, same schedule) | `spoon/kg-v3/32pahqok` | `ablate-B-lr10/result.md` |
| C `ablate-C-critic-warmup` | For iterations 1–22, advantages are 0, `teacher_kl_coef` 0 and `ent_coef` 0, so only the value loss and teacher value distillation train. The shared trunk stays trainable. From iteration 23 the loss is control PPO. This is a diagnostic-only hook outside `scripts/run_ppo.py`. | `spoon/kg-v3/lpt11y9j` | `ablate-C-critic-warmup/result.md` |

## Measurement

### Sources and checks made for this comparison

- **Control:** `ablation/extract_iterations.py ../6.2/pod-receipts/run.log 46`
  (rank-0 `[kg-probe]` iteration records, which hold rank-reduced metrics)
  reproduces `ablate-A-anchor/control-6.2-iterations-1-46.tsv` byte for
  byte. The control banks below match `../6.2/result.md`.
- **A, B, C:** the same extractor on each `pod-receipts/run.log` reproduces
  that arm's `pod-receipts/iterations.tsv` byte for byte.
- **Completeness:** every table has iterations 1–46 and 0 nonfinite
  metrics, and `optimizer/steps` is 736 at iteration 46 in all four.
- **Derived values:** the approx-KL mean is the arithmetic mean of the 46
  per-iteration values, iterations 1–46. Cumulative LR is the sum of 16 x
  the logged (Muon) LR per iteration. The AdamW LR is not logged; its
  configured base is 1/20 of Muon's in every arm.

### Comparison table

Banks are the completed-game raw final banks (`train/terminal_bank_0/1`,
rank-reduced means, seat 0 / seat 1) at iterations 12, 23, 34 and 45. All
other columns are the per-iteration values at the stated iteration.

| Arm | Game 1 bank | Game 2 bank | Game 3 bank | Game 4 bank | Entropy it 5 / 13 / 46 | Teacher KL it 13 / 46 | Unit kind KL it 46 | Approx KL mean 1–46 | Adv std it 46 | Expl var it 46 |
|---|---|---|---|---|---|---|---|---|---|---|
| Control 6.2 | 73,673 / 72,690 | 55,496 / 54,423 | 7,324 / 7,644 | 89 / 94 | 6.05 / 4.54 / 5.56 | 0.56 / 5.28 | 3.08 | 0.068 | 0.043 | 0.10 |
| A anchor 0.1 | 71,943 / 70,933 | 59,073 / 61,208 | 51,212 / 49,184 | 40,408 / 42,457 | 6.03 / 4.63 / 6.43 | 0.29 / 2.14 | 1.43 | 0.130 | 0.056 | 0.04 |
| B LR / 10 | 73,360 / 73,208 | 65,621 / 65,125 | 63,689 / 63,913 | 62,104 / 62,321 | 6.09 / 4.38 / 4.77 | 0.059 / 0.64 | 0.34 | 0.0081 | 0.077 | -0.36 |
| C critic warm-up | 70,867 / 70,632 | 4 / 3 | 0 / 0 | 0 / 0 | 6.14 / 13.7 / 43.2 | 11.5 / 35.2 | 18.3 | 0.057 | 0.044 | 0.15 |

### Supporting measurements

- **Approx KL split:**
  - Control: mean 0.049 over iterations 1–22 and 0.086 over 23–46; maximum 0.159.
  - A: 0.044 and 0.208; maximum 0.397.
  - B: 0.0045 and 0.011; maximum 0.018.
  - C: 0.073 during the warm-up and 0.042 after it; maximum 0.179.
- **Clip fraction mean, iterations 1–46:** control 0.41, A 0.47, B 0.099, C 0.38.
- **Game-end teacher KL** at iterations 12 / 23 / 34 / 45:
  - Control: 0.44 / 2.37 / 3.70 / 3.25.
  - A: 0.25 / 0.82 / 1.88 / 2.49.
  - B: 0.039 / 0.20 / 0.56 / 1.36.
  - C: 10.1 / 22.3 / 19.0 / 8.26.
- **Cumulative LR** at iterations 12 / 23 / 34 / 46:
  - Control, A and C: 0.040 / 0.141 / 0.305 / 0.554.
  - B: exactly 1/10 of that.
  - The control reaches B's iteration-46 cumulative LR (0.055) at iteration 15, where its teacher KL is 0.96.
- **Parameter movement in C** (grad audit, relative to the loaded weights):
  - At warm-up end (after iteration 22): actor-only 1.2e-3 (Muon weight decay only), critic-only 1.8e-2, shared trunk 3.4e-2.
  - Actor-only gradients were exactly 0 on every audited warm-up step.
  - No trunk audit exists for the control, A or B.
- **Game-phase swing:** all envs reset together, so one game spans about
  11.25 iterations. Within-game entropy peaks were 7.9–8.6 in B, 7.7–10.9
  in A and up to 43 in C. Single-iteration values at 5, 13 and 46 fall at
  different game phases, so they are compared across arms at equal
  iterations only.
- **Throughput, not an ablation claim:** over iterations 2–46, game SPS was
  2,016 for the control, 1,626 for A, 1,639 for B and 2,065 for C. The cause
  of A's and B's lower rate was not measured.

## Interpretation (inferred from the measurements above)

### Which predictions held

- **A (anchor x20):** the prediction was **not met**.
  - Teacher KL stayed below 0.5 only until iteration 16.
  - Entropy did not stay near 4–5.
  - Banks did not stay near 70k: they reached 41k at game 4.
  - The decline was much slower than the control's (41k against 0 at game 4).
- **B (LRs / 10):** the prediction was **largely met**.
  - Approx KL shrank about 5–10x. It stayed in the predicted 0.003–0.01 band through iteration 33 and reached 0.018 as the warm-up continued.
  - Teacher KL stayed far below the control's.
  - Banks declined later and far less: 62k at game 4, still falling about 1–2k per game after game 2.
  - The "entropy rises more slowly" part failed. B's own reading corrected it: within-game entropy swings are a game-phase property of the BC policy, not drift.
- **C (critic warm-up):** the prediction was **not met; the result was the opposite**.
  - Teacher KL grew to 18.6 by iteration 22 with zero policy gradient.
  - Games 1–2 did not hold: game 2 ended at a bank of 4.
  - The post-warm-up comparison could not be tested, because the policy had already collapsed.
  - Explained variance rose as predicted, but on a degenerate zero-bank economy.

### What the evidence supports about the mechanism

1. **The drift is mostly set by step size, not by the anchor.** The update
   size tracks the LR: approx KL and clip fraction fell about 10x in B. The
   20x anchor in A cut late teacher KL about 2.5x, yet its per-update approx
   KL was *larger* than the control's. Of the three levers, only the smaller
   step kept the economy near the BC level for 4 games.
2. **Drift grows with cumulative step size, but not in strict proportion.**
   B's teacher KL at iteration 46 (0.64, cumulative LR 0.055) is the same
   order as the control's at equal cumulative LR (0.96 at iteration 15). Per
   unit of cumulative LR, B drifted somewhat more at its game ends
   (1.36 / 0.053) than the control did at game 2 (2.37 / 0.141). So LR / 10
   *delays* the drift. There is no evidence that it stops it: B's game-end
   teacher KL rose 0.04 → 0.20 → 0.56 → 1.36 and its banks kept falling.
3. **The value loss moves the policy through the shared trunk.** In C, with
   actor-only gradients exactly 0 and no anchor, value-only updates at the
   control LR moved the trunk 3.4 % and destroyed the BC policy faster than
   full PPO did. This is a coupling effect between architecture and
   optimizer. Because the value loss (`vf_coef` 2.0, fresh critic head) is
   present in every arm, it is a candidate contributor to the drift in the
   control, A and B as well.
4. **No arm shows a PPO signal that improves on the BC policy.** No arm's
   banks rose in any game. All observed movement away from the BC policy
   lowered the self-play economy.

### What stays unresolved

- **How much of the control's drift comes from the value gradient through
  the trunk rather than the policy gradient.** The control, A and B have no
  trunk audit, and C also removed the anchor (confound).
- **Muon's per-matrix scale invariance against gradient magnitude.** B
  shows the step is LR-bound. It does not show why a near-zero advantage
  signal (advantage std about 0.04–0.08 late) still produces full-size
  steps.
- **Whether keeping the BC critic head** (instead of `model_fresh_critic_head`)
  **or a lower `vf_coef` removes the coupling.** Neither was run.
- **Whether LR / 10 plus anchor 0.1 compounds or interferes.** The
  combination was not run. A showed a larger per-update approx KL at full LR.
- **Whether a still smaller LR stops the drift or only delays it.**
- **Seed variance.** There is one seed per arm, so no arm-to-arm difference
  has an uncertainty estimate. The ordering B > A > control > C on banks is
  large (62k, 41k, 0, 0 at game 4) but unreplicated.
- **The per-head weighting and KL direction of `teacher/kl`** were not
  re-inspected.

## Recommended recipe change for the next longer run

This is a recommendation for the owner. Nothing has been launched or
approved.

**Change:** carry forward B's step size as the single recipe change:
`optimizer.muon_lr=0.0002`, `optimizer.adamw_lr=0.00001`, with the same
schedule (1,000-step warm-up, cosine decay to 400k, min ratio 0.01).
Everything else stays as in the control: `teacher_kl_coef` 0.005, fresh
critic head, no critic warm-up.

- **Why this change:** it is the only lever these runs support.
- **Why not C:** a naive critic warm-up is harmful.
- **Why not add the anchor now:** anchor 0.1 is not added in the same run,
  so its effect stays attributable. Its interaction with LR / 10 is
  unmeasured.
- **Suggested length:** 235 iterations, matching 6.2 (3,850,240 env steps).
  - B's schedule reaches the control's cumulative LR at game-2 end around iteration 75, at game-3 end (the control's 7k collapse) around iteration 126, and at game-4 end around iteration 197.
  - So a 235-iteration run tests whether drift keeps tracking cumulative LR past the control's collapse point.
  - This projection assumes the relation in mechanism point 2. It is not measured.
  - At B's measured 1,639 game SPS this is about 39 minutes (about $2.7 at $4.18/h). That estimate is not measured for this length.
- **Add a trunk audit:** add the relative parameter-movement audit (actor,
  critic, shared trunk) as telemetry only. This lets the run attribute drift
  between value and policy paths. It is not a training change.

**Loss conditions.** Any one of these means the recipe loses as a basis for
further scaling. The thresholds are proposed and are anchored to values
these four runs observed; they are not validated.

1. **Economy:** any completed game's final bank (rank-reduced mean, either
   seat) falls below 55k. This is the control's game-2 level, the last game
   before its collapse.
2. **Anchor distance:** game-end `teacher/kl` exceeds 2.4 nats. This is the
   control's game-2 end value (2.37), after which its next game ended at
   7k.
3. **Step size:** after warm-up, per-iteration approx KL stays above 0.05
   (the control's iteration 1–22 mean) for a full game. That would mean the
   update is no longer LR-bound as it was in B.
4. **Delay only, no improvement:** by the end, game-end banks show no upward
   trend and game-end teacher KL is still rising monotonically. LR / 10 is
   then only a delay, and the next discriminating runs target the
   trunk/value coupling: keep the BC critic head, or lower `vf_coef`, or
   stop the value gradient into the shared trunk, each with the trunk
   audit. Anchor 0.1 at LR / 10 is the other candidate.
5. **Execution:** any nonfinite metric, trainer fault, or iteration with
   other than 16 optimizer steps.

**Win condition.** None of the loss conditions fires over 235 iterations.
Even then, the result shows only that the self-play economy stays healthy.
Improvement over the BC policy needs a separate evaluation against the BC
best and other opponents, reporting win rate and bank margin by seat, with
denominators.
