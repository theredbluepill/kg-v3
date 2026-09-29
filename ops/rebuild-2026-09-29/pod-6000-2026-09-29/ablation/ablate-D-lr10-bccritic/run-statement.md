# Run statement: PPO collapse ablation D, "ablate-D-lr10-bccritic" (LR / 10, BC critic head)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

- **Question:** round 1 (`../comparison.md`) showed that in C value-only
  updates with a fresh critic head moved the shared trunk 3.4 % in 352 steps
  and destroyed the BC policy. The handoff probe
  (`cookbook/references/bc-best-starts-ppo-with-a-fresh-critic-head.md`) gave
  a critic-head gradient norm of 12.4 for a fresh head against 0.118 for the
  BC head, with opposite-sign targets. `vf_coef` is 2.0. If the fresh head's
  large value gradients drive the trunk drift, does keeping the BC critic head
  at B's step size move the trunk less and keep the policy closer to the BC
  policy than B?
- **Single change versus B (`ablate-B-lr10`, W&B `spoon/kg-v3/32pahqok`):**
  `--load-model-weights-mode model_only`, which loads the whole BC model
  including `critic_head.*`, instead of `model_fresh_critic_head`. The initial
  last-best teacher is copied from the loaded model, so teacher value
  distillation (0.005) now targets the BC critic rather than the fresh head.
  That follows from the load mode and is part of the same change. Everything
  else matches B: pod checkout `e74d67e` (clean),
  `configs/kaggriculture_2rank.yaml`, `-o rl.eval_replay_games=0
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`, BC best
  `fd854587…6f51`, 2 ranks, seeds 0/1, `KG_PROBE_STOP_ON_NONFINITE=1`,
  `--max-env-steps 753664` (46 x 16,384).
- **Telemetry-only trunk audit (DIAGNOSTIC-ONLY hook, not a training
  feature):** `trunk_audit_launcher.py` (this folder) loads the shared
  launcher `b393ad31…92df` unchanged and only reads parameters and gradients.
  It snapshots the loaded weights at the start of iteration 1 (the hook fails
  if an optimizer step already ran). It then emits the relative parameter
  change ||θ − θ0|| / ||θ0|| for the actor-only (`actor_input_proj.*`,
  `actor.*`), critic-only (`critic_head.*`, `critic_value_tokens`) and shared
  trunk groups after iterations 12, 23, 34 and 46. It also emits per-group
  gradient norms before the first optimizer step of every iteration, the same
  record as C's audit. It is outside `scripts/run_ppo.py` and `python/owl`,
  and nothing imports it. The groups are C's.
- **Hook verification (done before this statement):** a 2-iteration dry run
  (`run_d.sh ablate-D-dryrun 32768 1,2 debug -o optimizer.muon_lr=0.0002
  optimizer.adamw_lr=0.00001`, W&B off, exit 0, 42 s; `dryrun-receipts/`).
  - Overrides reached the trainer as `['rl.eval_replay_games=0',
    'optimizer.muon_lr=0.0002', 'optimizer.adamw_lr=0.00001']`, as in B.
    Logged LR 3.2e-6 at iteration 1, as in B.
  - The BC critic head was loaded: its ||θ0|| on both ranks is 22.0106,
    equal to the `critic_head.*` norm computed directly from
    `checkpoint_bc_best.pt` (22.0106).
  - Group sizes are 873,406 actor-only, 66,561 critic-only and 5,312,256
    shared, the same as C's. Both ranks agree to every printed digit.
  - Trunk audit after iterations 1 and 2: shared 1.20e-5 and 4.63e-5,
    actor-only 6.7e-6 and 2.75e-5, critic-only 8.6e-6 and 3.41e-5.
  - The dry run's `hashes.sha256` lacks `run_d.sh`, because `$0` was relative
    after the `cd`. It is fixed in the launched script (`SELF`), which is
    `ed87c412…1b22`.
  - **Dry-run observation against the premise (not yet a result):** before
    the first optimizer step, the critic-only gradient norm with the BC head
    was 0.90 (max |grad| 0.12). C's audit, with a fresh head, value-only loss
    and 10x the LR, had 0.21 at the same point. On real rollouts, then, the BC
    head does not receive the smaller value gradient that the synthetic
    handoff probe implied. At iteration 1 the advantage std was 0.54 (B 0.20)
    and explained variance 0.56 (B −0.06). The BC critic's saturated ±1
    values change the advantages as well as the value gradient.
- **Command:** `run_d.sh ablate-D-lr10-bccritic 753664 12,23,34,46 wandb -o
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`, started with
  `nohup`. The script takes an exclusive `flock` on `/root/ablation/gpu.lock`
  and waits for the GPUs to have no compute apps, because other round-2 arms
  share the pod. The 10-minute `timeout` starts only after that wait.
- **Measurements, iterations 1–46:** `../extract_iterations.py` (entropy,
  approx KL, clip fraction, advantage std, explained variance, `teacher/kl`,
  `teacher/unit_kind_kl`, `teacher/market_kind_kl`, LR, completed-game banks
  at iterations 12, 23, 34 and 45) plus `value_loss` from the iteration
  records. `trunk_audit_summary.py` gives the trunk audit and the grad audit.
  The comparisons are with B's table (`../ablate-B-lr10/pod-receipts/iterations.tsv`)
  and the control's.
- **Prediction if the fresh head's large value gradients drive the drift:**
  the trunk moves less than in a fresh-head arm, teacher KL at iteration 46
  is below B's 0.64, and banks at game 4 are at least B's 62k.
  **Discriminating observations:**
  1. Teacher KL at 46 below 0.64, banks at least 62k and a shared-trunk
     change below the fresh-head reference: the fresh head is a lever.
  2. Teacher KL and banks at or worse than B's: the fresh head's gradient
     magnitude is not the driver at this LR. Given the dry run, that is the
     more likely outcome.

  **Fresh-head trunk reference:** B has no in-run audit. C's audit (3.4 %
  shared after iteration 22) ran at 10x the LR with value-only updates. I
  will also compute B's end-of-run actor-only and shared change on the CPU
  from its kept `checkpoint_final.pt` against the BC best. The critic group is
  excluded for B, because a fresh head's θ0 is not in the BC checkpoint.
- **Confounds, named in advance:**
  - The BC critic is saturated and seat-biased: every BC episode was a win
    for the imitated seat. It changes the value baseline, and so the
    advantages and the PPO policy signal, not only the value-gradient size.
  - The teacher's value target changes with it.
  - D cannot separate these effects from gradient magnitude.
  - One seed.
- **Stopping condition and budget:** 46 complete iterations, the first
  nonfinite metric or trainer fault, or the watchdog
  `timeout --kill-after=30 600` (10 minutes). B took 483 s, about $0.56 at
  $4.18/h; the dry run took 42 s, about $0.05. W&B is online. After the run,
  it is renamed through the API to `ablate-D-lr10-bccritic` in group
  `kg-v3-ppo-collapse-ablation`. The GPUs are left idle afterwards.
