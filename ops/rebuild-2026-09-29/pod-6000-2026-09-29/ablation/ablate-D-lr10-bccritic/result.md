# Result: PPO collapse ablation D, "ablate-D-lr10-bccritic" (LR / 10, BC critic head)

**Execution: PASS. Prediction: not met as stated.**

- **Banks:** the self-play economy rose in every game: 73k, 77k, 81k, 85k (B: 73k, 66k, 64k, 62k). D is the first arm in which banks rose above the BC level.
- **Trunk movement:** the trunk did **not** move less than in B. At iteration 46 the shared trunk had changed 8.43e-3 (B 8.02e-3) and actor-only 5.07e-3 (B 5.12e-3).
- **Teacher KL:** teacher KL at iteration 46 was 1.45, against B's 0.64. That gap is partly a game-phase artefact. Iteration 46 is the first iteration of game 5, where B's teacher KL dropped from 1.36 and D's did not. At game ends (iterations 12, 23, 34, 45), D's teacher KL was 1.2x, 1.4x, 1.4x and 1.04x B's: 0.046 / 0.29 / 0.81 / 1.41 against 0.039 / 0.20 / 0.56 / 1.36.

With the BC critic, the parameters moved as far as in B at equal LR. The policy moved somewhat further from the BC teacher, and the drift went in a direction that raised self-play banks. The critic explained a mean 0.84 of return variance over iterations 1–46 (B: 0.29).

This is a pre-landing diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod). It is one seed and has not been Codex-verified. Receipts are in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/hftcr4xa`, state `finished`,
  online. It was renamed through the API after the run to
  `ablate-D-lr10-bccritic` in group `kg-v3-ppo-collapse-ablation` (original
  name `ppo-20260929-220315`, group `ppo`; job_type and tags unchanged).
  Summary `_step` is 753,664 (`pod-receipts/wandb_hftcr4xa_state.json`).
- **Run:**
  - Exit 0 at `--max-env-steps 753664`.
  - 46 complete iterations on both ranks, 736 optimizer steps.
  - No nonfinite metric (the stop was armed and never fired). No traceback.
  - Lock acquired and launched at 22:03:10Z with the GPUs idle; ended 22:10:29Z, 439 s wall.
  - Iterations 2–46 took 8.91 s each, 1,839 game SPS (rollout 5.10 s, teacher 0.91 s, update 2.74 s). B ran at 1,639 SPS. The difference was not investigated.
  - After the run: both GPUs 0 MiB / 0 %, no compute apps (`idle_after.csv`).
- **Code identity:**
  - The nine shared input hashes in `pod-receipts/hashes.sha256` equal B's.
  - `run_d.sh` is `ed87c412…1b22` and the hook is `f98720bd…a32`; both equal the committed copies (commit `9c62cae`).
  - Overrides reached the trainer as `['rl.eval_replay_games=0', 'optimizer.muon_lr=0.0002', 'optimizer.adamw_lr=0.00001']`, and the logged LR equals B's at every iteration.
  - The BC critic head was loaded: its ||θ0|| on both ranks is 22.0106, equal to the `critic_head.*` norm in `checkpoint_bc_best.pt`.
- **Checkpoint:** `checkpoint_final.pt` `1013d3c9…e130`, kept on the pod.
- **Spend:** about 481 s of GPU runs: the 439 s run plus the 42 s dry run.
  CPU-only checkpoint comparisons took about 1 minute more. The total is about
  $0.60 at $4.18/h.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | D (LR / 10, BC critic) | B (LR / 10, fresh critic) | Control 6.2 |
|---|---|---|---|---|
| 1 | 12 | 73,073 / 71,405 | 73,360 / 73,208 | 73,673 / 72,690 |
| 2 | 23 | 76,902 / 75,924 | 65,621 / 65,125 | 55,496 / 54,423 |
| 3 | 34 | 80,549 / 78,593 | 63,689 / 63,913 | 7,324 / 7,644 |
| 4 | 45 | 85,232 / 82,347 | 62,104 / 62,321 | 89 / 94 |

## Per-iteration metrics at the key iterations (D / B)

The full table is `pod-receipts/iterations.tsv`, produced by `../extract_iterations.py`. `value_loss` is
`loss/value_loss` from the rank-0 iteration records.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0013 / 0.0013 | 0.0007 / 0.0007 | 0.0003 / 0.0003 | 0.0013 / 0.0014 | 0.003 / 0.003 | 0.537 / 0.204 | 0.56 / -0.06 | 0.144 / 0.021 |
| 13 | 0.051 / 0.059 | 0.029 / 0.033 | 0.011 / 0.005 | 0.0055 / 0.0044 | 0.062 / 0.043 | 0.426 / 0.165 | 0.73 / 0.17 | 0.089 / 0.013 |
| 23 | 0.29 / 0.20 | 0.18 / 0.12 | 0.031 / 0.027 | 0.0079 / 0.0066 | 0.103 / 0.082 | 0.520 / 0.237 | 0.48 / 0.00 | 0.132 / 0.028 |
| 34 | 0.81 / 0.56 | 0.53 / 0.42 | 0.084 / 0.068 | 0.0123 / 0.0105 | 0.163 / 0.14 | 0.400 / 0.223 | 0.64 / 0.03 | 0.078 / 0.025 |
| 46 | 1.45 / 0.64 | 0.78 / 0.34 | 0.25 / 0.11 | 0.0127 / 0.0114 | 0.169 / 0.15 | 0.224 / 0.077 | 0.76 / -0.36 | 0.024 / 0.003 |

- **Explained variance:**
  - In D it was 0.72–0.97 within games, apart from 0.56 at iteration 1. It dipped to 0.38–0.64 only at game boundaries (iterations 12, 23, 34, 45–46), where every env resets together. The mean over iterations 1–46 was 0.84.
  - In B it was −0.36 to 0.58, with a mean of 0.29.
- The advantage std was about 2–3x B's throughout.
- The value loss is 5–8x larger than B's even though the critic explains far more variance. The scale and composition of the value loss were not inspected: the BC head starts saturated at ±1, while a fresh head starts near 0.

## Trunk audit (telemetry only, relative change ||θ − θ0|| / ||θ0|| from the loaded weights)

These are in-run records after the listed iteration's last optimizer step (`pod-receipts/trunk_audit.tsv`). Both ranks are equal to every printed digit.

| After iteration | Actor-only | Critic-only | Shared trunk |
|---|---|---|---|
| 12 | 6.21e-4 | 8.32e-4 | 1.04e-3 |
| 23 | 1.73e-3 | 2.50e-3 | 2.94e-3 |
| 34 | 3.18e-3 | 4.67e-3 | 5.38e-3 |
| 46 | 5.07e-3 | 7.50e-3 | 8.43e-3 |

**The fresh-head arms have no in-run audit at these iterations.** For them,
each arm's kept `checkpoint_final.pt` (iteration 46) was compared on the CPU
with the BC best (`pod-receipts/final_checkpoint_change_vs_bc.tsv`).

- **Method check:** on D's dry-run and main-run checkpoints the method reproduces the in-run audit exactly: dry run 2.7504e-5 / 4.6277e-5, main run 5.0734e-3 / 8.4336e-3.
- **Critic head:** a fresh head's θ0 is not in the BC checkpoint, so its change is omitted. The ~1.43 in the file is fresh-versus-BC distance, not movement.

| Arm (iteration 46) | Actor-only | Shared trunk | Critic value tokens |
|---|---|---|---|
| D (LR / 10, BC critic) | 5.07e-3 | 8.43e-3 | 3.24e-3 |
| B (LR / 10, fresh critic) | 5.12e-3 | 8.02e-3 | 1.87e-3 |
| A (anchor 0.1, full LR) | 4.98e-2 | 7.41e-2 | 7.06e-3 |
| C (critic warm-up, full LR) | 2.89e-2 | 7.26e-2 | 7.64e-3 |

**Gradient audit** (before the first optimizer step of each iteration, rank 0; `pod-receipts/trunk_audit.tsv`):

- D's gradient norms:
  - Critic-only: 0.47–1.19 (iteration 1: 1.19; iteration 46: 0.47).
  - Shared: 10.6–13.4.
  - Actor-only: 3.1–5.0.
- C's gradient norms, with a fresh head and value-only loss:
  - Critic-only: 0.21 at iteration 1 and 0.01–0.20 afterwards.
  - Shared: 1.05 at iteration 1.
- On real rollouts, then, the BC head's value gradient was *larger* than the fresh head's, not 100x smaller as the synthetic handoff probe (0.118 against 12.4) suggested.
- The total `optimizer/grad_norm` was similar in D and B: 12.6–13.9 in D and 12.6–14.7 in B.

## Reading

**Prediction check** (the hypothesis was that the fresh head's large value gradients drive the drift):

- Trunk moves less than in a fresh-head arm: **no**. The shared change is 8.43e-3 against B's 8.02e-3, and actor-only 5.07e-3 against 5.12e-3. At equal LR the parameter movement is the same to within about 5 %.
- Teacher KL at iteration 46 below 0.64: **no**. It was 1.45. D's teacher KL was at or above B's from iteration 15 on. The gap at iteration 46 is inflated by B's game-start drop, and at the game-4 end the two are 1.41 against 1.36.
- Banks at game 4 at least 62k: **yes**. They were 85k / 82k, and they rose every game.

**Supported:**

1. **Parameter movement is set by the step size, not by the critic head.**
   - Two arms at equal LR, with value-gradient magnitudes that differ in the opposite direction to the premise, moved the trunk and the actor equally.
   - The arms at 10x the LR (A and C) moved it about 9x more.
   - This agrees with round 1's reading that the drift is LR-bound. It rules out the fresh head's gradient magnitude as the driver of trunk *movement* at LR / 10.
2. **The critic head sets the direction of the drift and the sign of its economic effect.**
   - At equal movement, D's PPO policy moved at least as far from the BC teacher as B's (mean teacher KL over iterations 1–46: 0.48 against 0.36) and raised self-play banks each game.
   - B's policy lowered them.
   - D's critic explained most of the return variance (mean 0.84; 0.72–0.97 within games), where B's explained much less (mean 0.29, at most 0.58). This is consistent with the BC critic giving PPO informative advantages, while B's fresh critic gave noise-dominated ones.
   - This is D's discriminating observation. It was not among the anticipated outcomes: more teacher drift with better banks.
3. **Teacher KL is not a collapse measure on its own.** At similar teacher KL, B's banks fell and D's rose (game-4 end: 1.36 and 1.41). "Distance from BC" cannot by itself rank arms. Single-iteration comparisons also need a matched game phase.

**Confounds and unresolved attribution:**

- The load mode changed three things at once:
  - the critic's initial function (saturated, seat-biased BC winner head);
  - the teacher's value target, which follows the loaded critic;
  - the initial value-gradient magnitude.
  D cannot say which of these produced the better advantages. The high explained variance is the only measured link.
- **Self-play banks are not strength.** Both seats run the same policy, and banks rose for both. This may be a joint economic gain, not a policy that beats other opponents. No evaluation against other opponents ran (`rl.eval_replay_games=0`), so there is no win rate or bank margin against a different opponent.
- The value loss is larger than B's even though explained variance is higher. The value-loss scale was not reconciled.
- The BC critic's saturation (97 % of seat values at ±1 on a held-out game in the handoff note) was not re-measured on these rollouts, and its seat bias was not tested.
- One seed. Banks were still rising at game 4, so 46 iterations cannot show whether this holds, plateaus or reverses over a longer run.
- **Consequence for the handoff choice.** The handoff note (`cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md`) chose `model_fresh_critic_head` on the synthetic gradient probe. It named reopening "if a PPO run from the BC best shows the fresh critic's early explained variance or value loss behaving worse than a comparison start". This run shows a worse explained variance for the fresh head (B: mean 0.29) than for the BC head (D: mean 0.84) at equal LR. The reopening condition is met, subject to the one-seed caveat.
- **Possible follow-ups (not run):**
  - D at the control's full LR, to see whether informative advantages survive the larger step.
  - A second seed of D.
  - A longer D, to see whether the bank rise persists.
  - A held-out evaluation of D's final checkpoint against the BC best and other opponents.

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../final-report.md`.

- **Outcome:** Execution PASS. Banks rose 72,239 → 83,790 by game 4 (+11,551), the only arm whose banks rose; the rise at game 4 did not reproduce at a second seed (J).
- **Denominators:** 46 complete iterations on both ranks; 736 optimizer steps (16 per iteration); 753,664 global env steps (46 × 16,384); 1,024 completed games (4 game phases × 256 envs, `train/total_games_played` 1024); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/hftcr4xa
- **Spend:** about $0.60 (main run and dry run) at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; all 46 iterations lie inside the 1,000-step LR warm-up (LR at iteration 46 is 0.736 of peak), so nothing here was measured at peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
