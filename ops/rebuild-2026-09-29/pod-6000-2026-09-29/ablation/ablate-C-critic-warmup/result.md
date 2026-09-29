# Result: PPO collapse ablation C, "ablate-C-critic-warmup" (critic warm-up, DIAGNOSTIC-ONLY hook)

**Execution: PASS. Prediction: not met, and the opposite of the prediction.**
With the policy gradient, teacher-KL anchor and entropy bonus zeroed for
iterations 1–22, the actor-only gradients were exactly 0 on every audited
step, yet the policy drifted **faster** than in the control. Teacher KL
reached 10.3 nats at iteration 11 (control 0.46) and 18.6 at iteration 22
(control 3.06). The economy collapsed a game earlier than the control: final
banks were 70.9k, then 4, 0 and 0 (control 73.7k, 55k, 7.3k, 0). Critic-only
updates through the shared trunk are enough on their own to destroy the BC
policy under this optimizer and LR. This is a pre-landing diagnostic on
`kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod) and has not
been Codex-verified. Receipts are in `pod-receipts/`.

The warm-up is a **diagnostic-only hook** (`critic_warmup_launcher.py`
`3addc8e6…7e90`), not a training feature. It is not in `scripts/run_ppo.py`
or `python/owl`. It loads the shared launcher `b393ad31…92df` unchanged.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/lpt11y9j`, state `finished`,
  online. It was renamed through the API after the run to
  `ablate-C-critic-warmup` in group `kg-v3-ppo-collapse-ablation` (original
  name `ppo-20260929-195208`, group `ppo`; job_type and tags unchanged).
  Summary `_step` is 753,664 (`pod-receipts/wandb_lpt11y9j_state.json`).
- **Run:**
  - Exit 0 at `--max-env-steps 753664`.
  - 46 complete iterations and 736 optimizer steps.
  - No nonfinite metric: the stop was armed and never fired. No traceback.
  - Launch 19:52:05Z, end 19:58:34Z, 389 s wall.
  - Iterations 2–46 ran at 7.93 s each, which is 2,065 game SPS (rollout 3.68 s, teacher 0.89 s, update 2.71 s).
  - After the run: GPUs 0 MiB / 0 %, no compute apps.
- **Code identity:** the nine input hashes in `pod-receipts/hashes.sha256`
  equal ablation B's, so they equal A's and the control's. `run_c.sh` is
  `87386d14…1ab8` and the hook is `3addc8e6…7e90`; both equal the committed
  copies (commit `f05c9b8`).
- **Hook verification:**
  - The dry run is in `dryrun-receipts/` and the run statement.
  - The in-run audit (`pod-receipts/grad_audit.tsv`, first optimizer step of each iteration, both ranks) shows actor-only gradient norm and max |grad| exactly 0 at iterations 1–22.
  - The loss wrapper masked 352 calls (22 x 16) and passed through the rest.
  - Actor-only gradient norms were 2.0–11.3 from iteration 23.
- **Spend:** about 486 s of GPU runs: the 389 s main run plus three dry runs of 28 + 28 + 41 s. That is about $0.56 at $4.18/h.
- **Checkpoint:** `checkpoint_final.pt` `bce62cea…adef`, kept on the pod.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | C (critic warm-up) | B (LR / 10) | A (anchor 0.1) | Control 6.2 |
|---|---|---|---|---|---|
| 1 | 12 | 70,867 / 70,632 | 73,360 / 73,208 | 71,943 / 70,933 | 73,673 / 72,690 |
| 2 | 23 | 4 / 3 | 65,621 / 65,125 | 59,073 / 61,208 | 55,496 / 54,423 |
| 3 | 34 | 0 / 0 | 63,689 / 63,913 | 51,212 / 49,184 | 7,324 / 7,644 |
| 4 | 45 | 0 / 0 | 62,104 / 62,321 | 40,408 / 42,457 | 89 / 94 |

Games 1 and 2 were played entirely inside the warm-up.

## Per-iteration metrics at the key iterations (C / control)

The full table is `pod-receipts/iterations.tsv`. The control is
`../ablate-A-anchor/control-6.2-iterations-1-46.tsv`.

| Iter | Entropy | Approx KL | Clipfrac | Adv std | Expl var | Teacher KL | Unit kind KL | Market kind KL | LR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.37 / 4.33 | 0.0019 / 0.0026 | 0.010 / 0.020 | 0.224 / 0.195 | 0.43 / -0.05 | 0.0019 / 0.0026 | 0.0010 / 0.0011 | 0.0004 / 0.0009 | 3.2e-5 |
| 5 | 6.14 / 6.05 | 0.056 / 0.015 | 0.35 / 0.18 | 0.110 / 0.149 | 0.79 / 0.26 | 0.53 / 0.071 | 0.31 / 0.048 | 0.098 / 0.011 | 1.6e-4 |
| 13 | 13.7 / 4.54 | 0.036 / 0.042 | 0.33 / 0.36 | 0.055 / 0.088 | 0.66 / 0.44 | 11.5 / 0.56 | 6.87 / 0.35 | 2.91 / 0.089 | 4.16e-4 |
| 22 | 15.9 / 7.21 | 0.020 / 0.104 | 0.21 / 0.53 | 0.0066 / 0.038 | 0.99 / 0.80 | 18.6 / 3.06 | 4.34 / 2.45 | 8.94 / 0.16 | 7.04e-4 |
| 23 | 24.5 / 5.38 | 0.145 / 0.085 | 0.63 / 0.45 | 0.140 / 0.216 | 0.10 / 0.10 | 22.3 / 2.37 | 7.97 / 1.77 | 7.27 / 0.26 | 7.36e-4 |
| 34 | 22.7 / 5.45 | 0.040 / 0.092 | 0.37 / 0.50 | 0.014 / 0.211 | 0.63 / 0.11 | 19.0 / 3.70 | 7.17 / 2.75 | 5.24 / 0.55 | 1.09e-3 |
| 46 | 43.2 / 5.56 | 0.056 / 0.043 | 0.47 / 0.39 | 0.044 / 0.043 | 0.15 / 0.10 | 35.2 / 5.28 | 18.3 / 3.08 | 5.65 / 1.82 | 1.47e-3 |

The LR is identical in both runs at every iteration.

## Parameter movement (grad audit, relative change from the loaded weights)

The audit runs before the first optimizer step of the next iteration, so
"after iteration N" is the audit row for iteration N+1.

| After iteration | Actor-only | Critic-only | Shared trunk |
|---|---|---|---|
| 11 (game 1 end) | 3.0e-4 | 8.9e-3 | 1.46e-2 |
| 22 (warm-up end) | 1.19e-3 | 1.77e-2 | 3.40e-2 |
| 45 (run end) | 2.82e-2 | 4.00e-2 | 7.14e-2 |

- Actor-only drift during warm-up came only from Muon's decoupled weight decay.
- The drift that mattered went through the shared trunk. It moved 3.4 % under the value loss alone in 352 steps.
- No trunk-movement audit exists for the control, so the control's trunk change is not known.

## Reading

**Prediction check.** Each part of the prediction failed:

- Teacher KL and approx KL near 0 during warm-up: **no**. Teacher KL was 0.53 at iteration 5, 10.3 at 11 and 18.6 at 22. Approx KL within an update was up to 0.18.
- Entropy on the BC game-phase curve: **no**. It rose to 13–22, far above the BC's in-game 4.4–8 swing in B and the control.
- Explained variance up: **yes**. It was 0.43 at iteration 1 and 0.96–0.99 late in each warm-up game; the control's mean over iterations 1–22 was 0.50 and C's was 0.84.
- Banks near BC for games 1–2: **no**. Game 2 ended at 4.
- Slower drift after warm-up: **not testable**. The policy was already collapsed when PPO started at iteration 23.

**Supported (discriminating observation 1).** Critic-only updates through the shared trunk, at the control's optimizer and LR, disturb the policy by themselves. They do so more than full PPO with the 0.005 anchor did over the same iterations. The actor-only gradients were exactly 0, so this is a coupling effect through the shared trunk (an architecture and optimizer effect), not a PPO-signal effect.

**The high explained variance does not show an informative critic.** Advantage std fell to 0.003–0.007 late in games. From game 2 onward both seats ended near 0 bank, so the outcomes are nearly constant and easy to predict. The fit tracks a degenerate economy, not useful credit assignment.

**Confounds and unresolved attribution:**

- The warm-up also removed the 0.005 teacher-KL anchor (by design; owner brief). C therefore combines "value-only gradient" with "no anchor on the policy features". The control has both the value gradient and a weak anchor plus a policy gradient, and it drifted about 6–22x less in teacher KL over iterations 5–22.
- Inferred mechanism, not established: Muon orthogonalises each matrix update to a size set by the LR. With only the value gradient present, the whole per-matrix step goes to value-fitting directions in the shared trunk, with nothing preserving the actor's features. A fresh critic head (`model_fresh_critic_head`) makes the early value gradient large and arbitrary. B's result, that the drift tracks the integrated LR, fits this.
- C does not separate these effects:
  - Muon scale-invariance versus value-gradient magnitude.
  - The fresh critic head versus the value loss itself.
  - The removal of the anchor.
- **Possible follow-ups (not run, no spend approved):**
  - Warm-up with the teacher-KL anchor kept.
  - Warm-up with the shared trunk frozen (critic head and critic tokens only).
  - Warm-up at LR / 10.
  - A trunk-movement audit on the control.
- **Consequence for the ablation set:** a naive critic warm-up (value loss only, shared trunk trainable) is harmful here, not a fix. Of A, B and C, only the smaller step (B) kept the economy near the BC level over 4 games.
