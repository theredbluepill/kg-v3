# Run statement: plan 6.2 complete-work run on 2 ranks from the BC best (pre-landing)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified; Codex is at its usage limit until Oct 6). Results:
pending Codex review. No selection, ranking or submission follows from it.

- **Question:** does the canonical trainer run PPO on Kaggriculture from the
  BC best for a bounded 30 minutes on 2 ranks? That covers complete
  iterations, 16 optimizer steps each, live teacher distillation, online W&B,
  and no L6 replay-drift fault. What game and learner-seat SPS does complete
  work reach, and how do the losses and entropy move?
- **Inputs and code path:** as in the 6.1 run statement (same commit
  family, config `configs/kaggriculture_2rank.yaml` unchanged, spm 8 / accum
  1 per the 6.1 decision, BC best `fd854587…6f51` with
  `model_fresh_critic_head`, last-best teacher copied from the loaded model).
  The only override is `-o rl.eval_replay_games=0`. `checkpoint_freq` stays
  20M, so no periodic evaluation runs within 30 minutes. The run uses
  `--max-runtime-hours 0.47`: the trainer stops after the first complete
  iteration past 28.2 minutes of training and then writes its final
  checkpoint. The external watchdog is `timeout --kill-after=60 1860` (31
  min). `KG_PROBE_STOP_ON_NONFINITE=1` makes the launcher stop all ranks
  after an iteration with any nonfinite metric. The trainer itself raises on
  the first-minibatch alarm and on native exceptions. The command is
  `6.2/complete_work.sh`, with `OMP_NUM_THREADS=1 WANDB_MODE=online
  WANDB_ENTITY=spoon`.
- **Measurements:** game SPS and learner-seat SPS over complete iterations
  only. They come from `iteration` probe records: global env steps 16,384 per
  iteration, `train/player_step_total` deltas, and `time/iteration_seconds`.
  They are reported with and without iteration 1, which includes compile.
  Also recorded: optimizer steps per iteration (from
  `optimizer/steps` deltas); teacher telemetry (`teacher/kl`,
  `teacher/value_cross_entropy`, `teacher/cache_bytes`,
  `time/teacher_seconds`); `policy/logratio_mean` per iteration; per-phase
  peak memory per rank (dense late-game rollouts appear after about 11
  iterations); native step time; loss, value, entropy, KL and explained
  variance trends; W&B URL and final state; wall time and cost.
- **Expected discriminating observation:** exit 0 at the runtime stop, every
  iteration with 16 optimizer steps, no alarm and finite metrics throughout,
  W&B `finished`. Every phase peak allocated stays <= 83,204 MiB. An
  exception, a nonfinite value, an alarm or an OOM is a fail and is reported
  with its iteration.
- **Stopping condition and budget:** the runtime stop, the first fault, or
  the 31-minute watchdog. About 32 minutes of pod time is about $2.25. The
  total for steps 2–4 stays inside 1.5 hours.
