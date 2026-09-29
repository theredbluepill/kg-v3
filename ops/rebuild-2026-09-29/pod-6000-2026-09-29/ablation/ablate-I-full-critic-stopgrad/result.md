# Result: PPO collapse ablation I, "ablate-I-full-critic-stopgrad" (full LR, critic stop-gradient)

**Execution: PASS. Prediction: not met. Discriminating observations 2 and 4.**

- **Banks:** the economy collapsed as in the control: 70k, 50k, 4.8k, 0.10k over games 1–4. The control went 73k, 55k, 7.3k, 0.09k. Blocking the value gradient from the shared trunk did not delay the collapse at full LR.
- **Trunk movement:** at iteration 46 the shared trunk had changed 8.59e-2 and actor-only 4.70e-2. H (full LR, value gradient on) had 8.14e-2 / 4.88e-2. Movement is again set by the step size.
- **Teacher KL:** at game ends it was 0.47 / 2.10 / 3.38 / 2.31, against the control's 0.44 / 2.37 / 3.70 / 3.25. That is 0.7–1.06x the control's. At LR / 10, G's stop-gradient gave about 0.5x B's.
- **Critic:** much weaker. Mean explained variance over iterations 1–46 was 0.06, against the control's 0.54 and H's 0.79.

This arm used a **DIAGNOSTIC-ONLY launcher hook that changes training**: G's unchanged `../ablate-G-lr10-critic-stopgrad/critic_stopgrad_launcher.py`. It is not a recipe. The run is a pre-landing diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod before and after). It is one seed and has not been Codex-verified. Run statement: `run-statement.md`. Receipts: `pod-receipts/` and `dryrun-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/o9c5ji4v`, state `finished`, online.
  - Renamed through the API after the run (`wb_rename_i.py`, exit 0) to `ablate-I-full-critic-stopgrad` in group `kg-v3-ppo-collapse-ablation`.
  - The original name was `ppo-20260929-231710` in group `ppo`. Job type `ppo` and tags `kaggriculture-v3`, `ppo` are unchanged.
  - Summary `_step` 753,664. State and summary are in `pod-receipts/wandb_o9c5ji4v_state.json`.
- **Run:**
  - Exit 0 at `--max-env-steps 753664`. The lock was acquired immediately with the GPUs idle. Launch 23:17:07Z, end 23:23:48Z, 401 s wall.
  - 46 complete iterations on both ranks, 736 optimizer steps, 0 nonfinite metrics (the stop was armed and never fired), no traceback.
  - After the run both GPUs showed 0 MiB and 0 %, with no compute apps (`idle_after.csv`, rechecked after receipt extraction).
- **Code identity:**
  - The 12 input hashes in `pod-receipts/hashes.sha256` equal G's main run. They are the nine shared inputs plus `run_g.sh` `a2fb7965…81b2`, D's audit hook `f98720bd…78f9` and the G hook `ee5af67b…d621`. All are hash-equal to the committed copies.
  - The trainer received `['rl.eval_replay_games=0', 'rl.teacher_kl_coef=0.005']`, with no LR override.
  - Logged LR was 3.2e-5 at iteration 1 and 1.472e-3 at 46, the control's schedule. `teacher/kl_coef` was 0.005.
  - A fresh critic head was loaded: `critic_head` ||θ0|| is 22.6495 on both ranks, as in G.
  - The `diagnostic_hook` record shows `critic_stopgrad: on`, `changes_training: true` and `donated_buffer_disabled: false`.
- **Stop-gradient check:**
  - **Main run** (iteration 1, both ranks, `pod-receipts/stopgrad_check.tsv`): passed.
    - Shared trunk: 157 of 157 tensors had no value-side gradient path.
    - Actor-only (48 tensors) and `critic_value_tokens`: none either.
    - `critic_head` gradient norm: 0.23 on rank 0, 0.19 on rank 1.
  - **Dry run** (`ablate-I-dryrun-on`, 2 iterations at full LR, iterations 1 and 2 on both ranks, `dryrun-receipts/stopgrad_check_dryrun.tsv`): passed at every point.
    - Shared, actor-only and `critic_value_tokens` gradients were exactly zero.
    - `critic_head` gradient norm was 0.21–0.29.
    - Non-vacuity rests on G's `off` control dry run (shared value-side gradient norm 1.06–1.29).
  - Correction to the run statement's aside: the fourth `critic_head` tensor was not all-zero on rank 0 in the main run (0 of 4 all-zero). The "output bias always cancels" guess is therefore not universal. It remains unchecked by name.
- **Checkpoint:** `checkpoint_final.pt` `e6836ab0…8ade`, kept on the pod.
- **Spend:** 40 s dry run + 401 s main run = 441 s of 2-GPU pod time, about $0.51 at $4.18/h. The rename and extraction ran on the CPU in under a minute.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | I (full LR, stop-grad) | Control 6.2 (full LR) | H (full LR, BC critic) | G (LR / 10, stop-grad) |
|---|---|---|---|---|---|
| 1 | 12 | 70,476 / 71,263 | 73,673 / 72,690 | 75,825 / 77,652 | 69,745 / 70,492 |
| 2 | 23 | 49,826 / 50,161 | 55,496 / 54,423 | 60,048 / 60,084 | 64,807 / 64,358 |
| 3 | 34 | 4,813 / 4,789 | 7,324 / 7,644 | 20,712 / 20,520 | 67,432 / 65,656 |
| 4 | 45 | 104 / 102 | 89 / 94 | 71 / 76 | 63,701 / 63,126 |

## Per-iteration metrics at the key iterations (I / control)

Full table: `pod-receipts/iterations.tsv` (`../extract_iterations.py`). `value_loss` is `loss/value_loss` from the rank-0 `[kg-probe]` iteration records. The rank-0 records for iterations 4, 6, 22 and 25 were interleaved with other output and are not decodable; the key iterations are all present.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0025 / 0.0026 | 0.0011 / 0.0011 | 0.0007 / 0.0009 | 0.0025 / 0.0026 | 0.017 / 0.020 | 0.183 / 0.195 | −0.51 / −0.05 | 0.017 / 0.019 |
| 13 | 0.56 / 0.56 | 0.36 / 0.35 | 0.084 / 0.089 | 0.039 / 0.042 | 0.35 / 0.36 | 0.170 / 0.088 | −0.14 / 0.44 | 0.014 / 0.0037 |
| 23 | 2.10 / 2.37 | 1.39 / 1.77 | 0.35 / 0.26 | 0.072 / 0.085 | 0.41 / 0.45 | 0.250 / 0.216 | −0.07 / 0.10 | 0.031 / 0.023 |
| 34 | 3.38 / 3.70 | 1.89 / 2.75 | 1.20 / 0.55 | 0.035 / 0.092 | 0.27 / 0.50 | 0.220 / 0.211 | −0.03 / 0.11 | 0.024 / 0.021 |
| 46 | 3.21 / 5.28 | 1.31 / 3.08 | 1.65 / 1.82 | 0.018 / 0.043 | 0.20 / 0.39 | 0.017 / 0.043 | −0.19 / 0.10 | 0.00015 / 0.0010 |

- **Means over iterations 1–46.** Approx KL 0.050 (control 0.068, H 0.089). Clip fraction 0.33 (control 0.41, H 0.45). Teacher KL 2.19 (control 2.39, H 3.30). Explained variance 0.06 (control 0.54, H 0.79).
- **The policy collapsed onto fewer actions.** Entropy fell from 4.4 at iteration 1 to 2.19–2.83 over iterations 33–46, against the control's 5.56 and H's 7.26 at iteration 46. Late approx KL and clip fraction are lower because the policy is nearly deterministic, not because it is healthier.
- **Market-kind KL was larger than the control's at iteration 34** (1.20 against 0.55). Unit-kind KL was smaller from game 2 on. The teacher-KL reduction is again in unit kinds, as in G.
- **Grad norms.** `optimizer/grad_norm` ranged 2.46–21.9 over the 42 decodable rank-0 records (control 3.7–47.9).

## Trunk audit (telemetry only, ||θ − θ0|| / ||θ0|| from the loaded weights)

In-run records are taken after the listed iteration's last optimizer step (`pod-receipts/trunk_audit.tsv`, from D's unchanged `trunk_audit_summary.py`). Both ranks are equal to every printed digit. The `critic_head` / `critic_value_tokens` split comes from G's `critic_split_audit` records. This audit is DIAGNOSTIC-ONLY and telemetry-only (D's hook).

| After iteration | Actor-only | Critic-only | (critic_head / critic_value_tokens) | Shared trunk | H shared | G shared (LR / 10) |
|---|---|---|---|---|---|---|
| 12 | 6.30e-3 | 1.03e-2 | 1.03e-2 / 1.57e-3 | 9.59e-3 | 1.00e-2 | 9.92e-4 |
| 23 | 1.77e-2 | 3.32e-2 | 3.33e-2 / 3.32e-3 | 2.66e-2 | 2.73e-2 | 2.72e-3 |
| 34 | 3.35e-2 | 6.08e-2 | 6.09e-2 / 4.71e-3 | 5.41e-2 | 5.01e-2 | 4.92e-3 |
| 46 | 4.70e-2 | 8.22e-2 | 8.23e-2 / 5.05e-3 | 8.59e-2 | 8.14e-2 | 7.81e-3 |

- The control has no in-run audit. Its end-of-run change against BC was about 7.4e-2 (comparison.md).
- **Gradient audit** (rank 0, before the first optimizer step of each iteration, after DDP):
  - actor-only 0.64–4.56;
  - critic-only 0.038–0.57;
  - shared 1.14–14.85.

  At iteration 1 they were 3.94 / 0.40 / 10.70; at iteration 46, 1.42 / 0.044 / 1.81.

## Reading

**Prediction check** (the value path is the main driver even at full LR, so the banks do far better than the control's, with the remaining drift from the policy path only): **not met.** Game-4 banks were 104 / 102, the control's level, and games 2–3 were slightly below the control's.

- Observation 2 of the run statement holds: the policy and teacher path at this step size is enough to collapse the economy with no value gradient in the trunk.
- Observation 4 also holds: explained variance was far below the control's and the banks were no better.
- My pre-launch expectation (shared change about 7–8e-2, game-4 banks under 20k) held. The expected teacher-KL reduction to 0.3–0.6x the control's at game ends did not: it was 0.7–1.06x.

**Supported (one seed):**

1. **The value gradient through the shared trunk is not necessary for the full-LR collapse.** I and the control differ only in the detach, and both reach about 0.1k by game 4. With H (BC critic head, same outcome), this is the third full-LR arm to collapse whatever the critic setup.
2. **Trunk movement is set by the step size, not by the loss mix.** With the value term removed, the shared trunk moved 8.59e-2, against H's 8.14e-2 with it and about 7.4e-2 for the control, A and C. This matches G's result at LR / 10 (7.81e-3 against 7.8–8.4e-3).
3. **The stop-gradient's drift reduction does not scale to full LR.** At LR / 10 it about halved game-end teacher KL (G against B). At full LR it cut the control's by 0–30 %, and not at game 1.
4. **A critic head on value-blind features fails at full LR.** Mean explained variance was 0.06 (G 0.16 at LR / 10). Advantages were therefore close to raw returns minus a poor baseline. The policy then lost most of its entropy (2.2 at iteration 45), unlike the control and H.

**Unresolved attribution and limits:**

- **Two changes by construction.** The detach both protects the trunk and starves the critic of adaptable features. The weaker critic may have cancelled a real trunk-protection benefit, or caused the entropy collapse. This run cannot separate them. A stop-gradient arm with the BC critic head (D's load) at full LR would.
- **Seed noise.** There is one seed. The 3–5k gaps from the control at games 1–3 are within the unsized arm-to-arm spread (game-1 banks range 70–78k across arms before any meaningful policy change).
- **Teacher value distillation.** Its trunk gradient was blocked along with the value loss (coefficient 0.005 against 2.0). It was not separated.
- **Which actions drive the collapse.** The per-family cause was not measured. Market-kind KL rose above the control's in game 3 while unit-kind KL fell.
- **Strength.** Self-play banks are not strength. No held-out opponent ran (`rl.eval_replay_games=0`).

**Consequence.** Across the control, H and I, the full step size collapses the economy by game 4 whatever the critic head or the critic's access to the trunk. At full LR the lever is the step size (B, D, E, F, G all survive at LR / 10), not the value gradient's path. The critic's trunk gradient remains a secondary contributor to drift at LR / 10 only (G).

## Receipt close (2026-09-30)

Closing summary for the frozen receipt; it restates facts recorded above and in `../final-report.md`.

- **Outcome:** Execution PASS. Collapse as in the control: banks 70,870 → 103 by game 4.
- **Denominators:** 46 complete iterations on both ranks; 736 optimizer steps (16 per iteration); 753,664 global env steps (46 × 16,384); 1,024 completed games (4 game phases × 256 envs, `train/total_games_played` 1024); 0 nonfinite metrics.
- **W&B:** https://wandb.ai/spoon/kg-v3/runs/o9c5ji4v
- **Spend:** about $0.51 (main run and dry run) at $4.18/h on pod `aki4vy8kpfldpa`.
- **Gaps:** single seed at env seed 0 (rank seeds 0/1), shared with every other arm, so seed variance is unknown; no held-out evaluation (`rl.eval_replay_games=0`), so there is no win rate or bank margin against any opponent and self-play banks are not strength; not Codex-verified; the final checkpoint stays on the pod only; all 46 iterations lie inside the 1,000-step LR warm-up (LR at iteration 46 is 0.736 of peak), so nothing here was measured at peak LR.
- **Frozen:** this file is covered by `ablation/SHA256SUMS`; later corrections go in a new file.
