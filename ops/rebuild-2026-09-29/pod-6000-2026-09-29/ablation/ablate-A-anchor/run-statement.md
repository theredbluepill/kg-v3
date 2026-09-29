# Run statement: PPO collapse ablation A, "ablate-A-anchor" (stronger BC anchor)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner-approved as one of the three collapse ablations
(A anchor, B smaller steps, C critic warm-up; about $2 in total). No selection,
ranking or submission follows from it.

- **Question:** in the 6.2 control (W&B `spoon/kg-v3/7k07gp7c`) the self-play
  economy collapsed from the BC best (final banks 73k, 55k, 7.3k, 0 over games
  1–4) while teacher KL grew to 2–5 nats. Hypothesis (inferred, not
  established): scale-invariant Muon/Adam steps on a near-zero, noisy PPO
  signal random-walk the sharp BC policy away, and the 0.005 teacher anchor is
  too weak to hold it. Does a 20x stronger anchor keep the policy near the BC
  best and keep the economy alive over 4 games?
- **Single change versus the control:** `-o rl.teacher_kl_coef=0.1` (control
  0.005). `teacher_value_coef` stays 0.005. The teacher is the loaded BC model
  (`last_best`; `checkpoint_freq` 20M, so no promotion or evaluation inside
  46 iterations). Everything else matches 6.2: pod checkout `e74d67e` (clean),
  `configs/kaggriculture_2rank.yaml`, BC best `fd854587…6f51` with
  `model_fresh_critic_head`, `-o rl.eval_replay_games=0`, 2 ranks, seeds 0/1,
  the same launcher with `KG_PROBE_STOP_ON_NONFINITE=1`. The stop differs only
  in kind: `--max-env-steps 753664` (46 x 16,384) instead of the runtime stop,
  so the run covers the same first 46 iterations the control logged. The LR
  warm-up (1,000 optimizer steps) is still in progress at iteration 46 (736
  steps), as it was in the control.
- **Command:** `ablation/run_ablation.sh ablate-A-anchor -o rl.teacher_kl_coef=0.1`
  (copied to the pod outside the checkout; its hash is in the receipts).
- **Measurements, iterations 1–46:** entropy, approx KL, clip fraction,
  advantage std, explained variance, `teacher/kl`, `teacher/unit_kind_kl`,
  `teacher/market_kind_kl`, learning rate, logged `teacher/kl_coef`, and the
  completed-game final banks (`train/terminal_bank_0/1` at iterations 12, 23,
  34, 45), extracted by `ablation/extract_iterations.py` from the `[kg-probe]`
  records and compared with the same extraction of the 6.2 `run.log`.
- **Prediction if the hypothesis holds:** teacher KL stays below about 0.5,
  entropy stays near 4–5, and banks stay near 70k through game 4.
  **Discriminating observation:** if teacher KL stays low but banks still
  fall, the anchor holds the policy yet the collapse has another cause
  (reward/critic signal); if teacher KL still climbs past about 1 nat with a
  20x coefficient, the step-size or critic candidates gain weight.
- **Stopping condition and budget:** 46 complete iterations
  (`--max-env-steps 753664`), the first nonfinite metric or trainer fault, or
  the external watchdog `timeout --kill-after=30 600` (10 min). About 7 minutes
  of pod time is about $0.50 at $4.18/h. W&B is online; the run is renamed
  `ablate-A-anchor` in group `kg-v3-ppo-collapse-ablation` through the W&B API
  after it finishes (the trainer hardcodes group `ppo` and the name
  `ppo-<run dir>`; the code is left unchanged for comparability).
