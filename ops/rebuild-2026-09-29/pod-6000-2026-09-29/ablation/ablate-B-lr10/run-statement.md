# Run statement: PPO collapse ablation B, "ablate-B-lr10" (smaller steps)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner-approved as one of the three collapse ablations
(A anchor, B smaller steps, C critic warm-up; about $2 in total). No selection,
ranking or submission follows from it.

- **Question:** in the 6.2 control (W&B `spoon/kg-v3/7k07gp7c`) the self-play
  economy collapsed from the BC best (final banks 73k, 55k, 7.3k, 0 over games
  1–4) while approx KL stayed at 0.03–0.1 and clip fraction at 0.3–0.5 even
  as the raw advantage std fell to 0.002–0.02. Hypothesis (inferred, not
  established): scale-invariant Muon/Adam steps on a near-zero, noisy PPO
  signal random-walk the sharp BC policy away. Ablation A (teacher coef 0.1)
  slowed the decline (41k at game 4) but per-update approx KL grew to
  0.2–0.4. If the update size is set by the optimizer LR rather than by the
  signal, a 10x smaller LR should shrink per-update drift about 10x. Does it
  slow the drift and the economic decline over 4 games?
- **Single change versus the control:** both optimizer learning rates divided
  by 10: `-o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001` (control
  0.002 / 0.0001). Schedule unchanged (linear warm-up over 1,000 optimizer
  steps, cosine decay to 400k, min ratio 0.01), so the logged (Muon) LR runs
  from 3.2e-6 at iteration 1 to about 1.47e-4 at iteration 46.
  `teacher_kl_coef` stays 0.005 and `teacher_value_coef` 0.005. Everything
  else matches 6.2: pod checkout `e74d67e` (clean),
  `configs/kaggriculture_2rank.yaml`, BC best `fd854587…6f51` with
  `model_fresh_critic_head`, `-o rl.eval_replay_games=0`, 2 ranks, seeds
  0/1, the same launcher with `KG_PROBE_STOP_ON_NONFINITE=1`, the same
  `--max-env-steps 753664` stop as ablation A (46 x 16,384; the control's
  first 46 iterations are the comparison window).
- **Command:** `ablation/run_ablation.sh ablate-B-lr10 -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`
  (the committed script, `e5acaf10c2f7…`, already on the pod outside the
  checkout; its hash is recorded in the receipts).
- **Measurements, iterations 1–46:** entropy, approx KL, clip fraction,
  advantage std, explained variance, `teacher/kl`, `teacher/unit_kind_kl`,
  `teacher/market_kind_kl`, learning rate, and the completed-game final banks
  (`train/terminal_bank_0/1` at iterations 12, 23, 34, 45), extracted by
  `ablation/extract_iterations.py` and compared with the control's table
  (`ablate-A-anchor/control-6.2-iterations-1-46.tsv`) and A's.
- **Prediction if the hypothesis holds:** drift is slower — approx KL about
  0.003–0.01 (control 0.03–0.1), entropy rises more slowly than the
  control's 4.3 → 7.9 by iteration 5, teacher KL stays well below the
  control's 2–5 nats — and banks decline later than the control's (7k at
  game 3, about 0 at game 4) but may still decline.
  **Discriminating observation:** if approx KL scales down about 10x and banks
  hold, update size is the main lever; if approx KL does not shrink (the step
  size is not LR-bound, e.g. through Muon normalisation interacting with the
  schedule) or banks still collapse on schedule despite small KL, the critic
  or reward signal (C) gains weight.
- **Stopping condition and budget:** 46 complete iterations
  (`--max-env-steps 753664`), the first nonfinite metric or trainer fault, or
  the external watchdog `timeout --kill-after=30 600` (10 min). About 8
  minutes of pod time is about $0.55 at $4.18/h. W&B is online; the run is
  renamed `ablate-B-lr10` in group `kg-v3-ppo-collapse-ablation` through the
  W&B API after it finishes (the trainer hardcodes group `ppo` and the name
  `ppo-<run dir>`; the code is left unchanged for comparability). GPUs are
  left idle afterwards.
