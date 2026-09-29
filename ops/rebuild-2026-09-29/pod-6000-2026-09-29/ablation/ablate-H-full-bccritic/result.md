# Result: PPO collapse ablation H, "ablate-H-full-bccritic" (full LR, BC critic head)

**Execution: PASS (second attempt). Prediction: not met. Discriminating observation 2.**

- **Banks:** the economy collapsed as in the control: 76k, 60k, 21k, 0.07k over 4 games (control 73k, 55k, 7k, 0.09k). The BC critic head delayed game 3's fall somewhat (21k against 7k) and did not prevent collapse by game 4.
- **Trunk movement:** at iteration 46 the shared trunk had changed 8.14e-2 and actor-only 4.88e-2. That is 9.6x D's (8.43e-3 / 5.07e-3) at 10x D's LR, and equal to A's and C's end-of-run change vs BC (shared 7.4e-2 / 7.3e-2). Movement is again set by the step size.
- **Teacher KL:** higher than the control's through game 3 (game-end 0.93 / 2.59 / 5.91 / 4.66 against 0.44 / 2.37 / 3.70 / 3.25) and 4.92 at iteration 46 (control 5.28).
- **Critic:** explained a mean 0.79 of return variance over iterations 1–46 (control 0.54, D 0.84), 0.88–0.97 within games 1–3, yet the policy still destroyed the economy.

This is a pre-landing diagnostic on `kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod). It is one seed and has not been Codex-verified. Receipts are in `pod-receipts/`.

- **W&B:** `https://wandb.ai/spoon/kg-v3/runs/ksqdtvm0`, state `finished`, online. Renamed through the API after the run (`wb_rename_h.py`, exit 0) to `ablate-H-full-bccritic` in group `kg-v3-ppo-collapse-ablation` (original name `ppo-20260929-230417`, group `ppo`; job_type and tags unchanged). State and summary in `pod-receipts/wandb_ksqdtvm0_state.json`.
- **Attempt 1 (failed before any GPU work, kept in `attempt1-dup-override-receipts/`):** the command in the run statement passed `-o rl.teacher_kl_coef=0.005 rl.eval_replay_games=0`, but `run_d.sh` already passes `-o rl.eval_replay_games=0`, and `scripts/run_ppo.py` rejects the duplicate (`ValueError: Duplicate override field 'rl.eval_replay_games'`) on both ranks during argument resolution. Exit 1 after 3 s. The relaunch dropped the duplicate: `run_d.sh ablate-H-full-bccritic 753664 12,23,34,46 wandb -o rl.teacher_kl_coef=0.005`. The trainer received `['rl.eval_replay_games=0', 'rl.teacher_kl_coef=0.005']`, the intended set.
- **Run:**
  - Exit 0 at `--max-env-steps 753664`; lock acquired and launched at 23:04:14Z with the GPUs idle; ended 23:10:59Z, 405 s wall.
  - 46 complete iterations on both ranks, 736 optimizer steps, 0 nonfinite metrics (stop armed, never fired), no traceback.
  - Iteration 1 took 23.7 s; iterations 2–46 averaged 8.26 s (about 1,984 game SPS). Not a throughput claim.
  - After the run: both GPUs 0 MiB / 0 %, no compute apps (`idle_after.csv`).
- **Code identity:** the nine shared input hashes in `pod-receipts/hashes.sha256` equal D's (and so the control's); `run_d.sh` `ed87c412…1b22` and the hook `f98720bd…78f9` equal the committed copies. Logged LR 3.2e-5 at iteration 1 and 1.472e-3 at 46, equal to the control's schedule (10x D's). `teacher/kl_coef` 0.005. The BC critic head was loaded: its ||θ0|| on both ranks is 22.0106.
- **Checkpoint:** `checkpoint_final.pt` `5322af8f…f1a`, kept on the pod.
- **Spend:** 405 s + 3 s of GPU time, about $0.47 at $4.18/h.

## Completed-game final banks (rank-reduced means, seat 0 / seat 1)

| Game | Iteration | H (full LR, BC critic) | Control 6.2 (full LR, fresh critic) | D (LR / 10, BC critic) |
|---|---|---|---|---|
| 1 | 12 | 75,825 / 77,652 | 73,673 / 72,690 | 73,073 / 71,405 |
| 2 | 23 | 60,048 / 60,084 | 55,496 / 54,423 | 76,902 / 75,924 |
| 3 | 34 | 20,712 / 20,520 | 7,324 / 7,644 | 80,549 / 78,593 |
| 4 | 45 | 71 / 76 | 89 / 94 | 85,232 / 82,347 |

## Per-iteration metrics at the key iterations (H / control)

Full table: `pod-receipts/iterations.tsv` (`../extract_iterations.py`). `value_loss` is `loss/value_loss` from the rank-0 iteration records.

| Iter | Teacher KL | Unit kind KL | Market kind KL | Approx KL | Clipfrac | Adv std | Expl var | Value loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0022 / 0.0026 | 0.0010 / 0.0011 | 0.0004 / 0.0009 | 0.0022 / 0.0026 | 0.014 / 0.020 | 0.560 / 0.195 | 0.50 / -0.05 | 0.156 / 0.019 |
| 13 | 1.24 / 0.56 | 0.57 / 0.35 | 0.25 / 0.089 | 0.048 / 0.042 | 0.39 / 0.36 | 0.233 / 0.088 | 0.89 / 0.44 | 0.024 / 0.0037 |
| 23 | 2.59 / 2.37 | 1.79 / 1.77 | 0.35 / 0.27 | 0.103 / 0.085 | 0.50 / 0.45 | 0.259 / 0.216 | 0.46 / 0.10 | 0.032 / 0.023 |
| 34 | 5.91 / 3.70 | 4.72 / 2.75 | 0.41 / 0.55 | 0.127 / 0.092 | 0.57 / 0.50 | 0.225 / 0.211 | 0.05 / 0.11 | 0.025 / 0.021 |
| 46 | 4.92 / 5.28 | 3.72 / 3.08 | 0.71 / 1.82 | 0.059 / 0.043 | 0.42 / 0.39 | 0.061 / 0.043 | 0.09 / 0.10 | 0.0018 / 0.0010 |

- Iterations 23, 34 and 46 sit at or just after game boundaries (all envs reset together), where explained variance dips in every arm; within games 3–4, H's was 0.78–0.94.
- Means over iterations 1–46: approx KL 0.089 (control 0.068, D 0.0093), max 0.231 (control 0.159); clip fraction 0.45 (control 0.41, D 0.12); teacher KL 3.30 (control 2.39, D 0.48).
- `optimizer/grad_norm` 4.5–20.1 (control 3.7–47.9).

## Trunk audit (telemetry only, ||θ − θ0|| / ||θ0|| from the loaded weights)

In-run records after the listed iteration's last optimizer step (`pod-receipts/trunk_audit.tsv`; both ranks equal to every printed digit). DIAGNOSTIC-ONLY hook, unchanged from D.

| After iteration | Actor-only | Critic-only | Shared trunk | D shared (LR / 10) |
|---|---|---|---|---|
| 12 | 6.17e-3 | 7.44e-3 | 1.00e-2 | 1.04e-3 |
| 23 | 1.77e-2 | 2.17e-2 | 2.73e-2 | 2.94e-3 |
| 34 | 3.29e-2 | 4.13e-2 | 5.01e-2 | 5.38e-3 |
| 46 | 4.88e-2 | 6.60e-2 | 8.14e-2 | 8.43e-3 |

Gradient audit (before the first optimizer step of each iteration, rank 0): actor-only 1.78–5.69, critic-only 0.087–1.67, shared 3.03–15.8 (iteration 1: 5.46 / 0.76 / 13.45; iteration 46: 1.78 / 0.13 / 3.03).

## Reading

**Prediction check** (the fresh head is the main driver even at full LR, so game-4 banks >> 0): **not met.** Game-4 banks were 71 / 76, the control's level. Observation 2 of the run statement holds: at the control's step size the economy collapses whichever critic head PPO starts from.

**Supported (one seed):**

1. **The step size is necessary for the collapse; the BC critic head is not sufficient to prevent it.** D and H differ only in LR (10x). D's banks rose to 85k; H's fell to 0.07k. The control and H differ only in the head; both collapse by game 4. Across D, H, B and the control, LR decides whether the economy survives; the head decides the direction only at the small LR (D rose, B fell).
2. **Parameter movement scales with LR regardless of the head** (H/D shared 9.7x, actor 9.6x at 10x LR), consistent with D's reading and with A and C.
3. **An informative critic does not stop the collapse at full LR.** H's critic explained 0.88–0.97 of return variance within games 1–3 (control 0.84–0.90 at matched points in games 3–4 too), yet teacher KL grew faster than the control's in games 1–3 and banks fell. Explained variance of the self-play return is therefore not a sufficient health signal; the value target itself follows the collapsing self-play economy.

**Confounds and unresolved attribution:**

- The load mode changes three things at once (critic initial function, teacher value target, initial value-gradient magnitude); H cannot separate them, but none of them was enough at full LR.
- Game 3 banks were higher than the control's (21k against 7k) and game 1–2 banks slightly higher; one seed cannot say whether that partial delay is real.
- Teacher KL rose faster than the control's early (1.24 against 0.56 at 13) while banks were higher, again showing teacher KL alone does not rank arms.
- Self-play banks are not strength; `rl.eval_replay_games=0`, no win rate or margin against another opponent.
- Which LR between LR / 10 and full LR keeps D's rise is not measured.
