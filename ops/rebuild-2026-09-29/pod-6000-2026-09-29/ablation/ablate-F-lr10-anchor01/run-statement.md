# Run statement: PPO collapse ablation F, "ablate-F-lr10-anchor01" (LR / 10, teacher_kl_coef 0.1)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

- **Question:** round 1 (`../comparison.md`) had two partial repairs of the
  economy collapse. A (teacher-KL anchor 0.1 at the full LR) slowed the bank
  decline to 41k at game 4 but let teacher KL reach 2–3 nats. B (LR / 10 at
  anchor 0.005) kept teacher KL at 0.64 at iteration 46 and banks at 62k,
  still falling (73k to 62k over 4 games). Do the two levers combine: at B's
  step size, does a 20x stronger pull toward the teacher keep the policy
  closer to BC than B and hold the economy at or above B's?
- **Single change versus B (`ablate-B-lr10`, W&B `spoon/kg-v3/32pahqok`):**
  `-o rl.teacher_kl_coef=0.1` (B: 0.005 from `configs/kaggriculture_2rank.yaml`).
  Everything else matches B: pod checkout `e74d67e` (clean),
  `configs/kaggriculture_2rank.yaml`, `-o rl.eval_replay_games=0
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`, BC best
  `fd854587…6f51` with `--load-model-weights-mode model_fresh_critic_head`,
  2 ranks, seeds 0/1, `KG_PROBE_STOP_ON_NONFINITE=1`, `--max-env-steps
  753664` (46 x 16,384). `teacher_value_coef` 0.005, `vf_coef` 2.0 and the
  teacher (`teacher_mode: last_best`) are unchanged. A ran the same
  coefficient at the full LR, so F is also A with LR / 10.
- **Launcher:** `../ablate-E-lr10-vf05/run_e.sh` reused **unchanged**
  (`2fed2fe3…7a20`, pod copy hash-equal to the committed one). It is
  parameterised by name, steps, audit iterations, log mode and overrides; its
  load mode `model_fresh_critic_head` is B's. Its only differences from
  `../run_ablation.sh` are the DIAGNOSTIC-ONLY, telemetry-only hook
  `../ablate-D-lr10-bccritic/trunk_audit_launcher.py` (`f98720bd…78f9`,
  unchanged) and the exclusive `flock` on `/root/ablation/gpu.lock` with an
  idle-GPU wait, because other round-2 arms share the pod. The hook only
  reads parameters and gradients: it snapshots θ0 at the start of iteration
  1 (after the fresh head is initialised) and emits ||θ − θ0|| / ||θ0|| for
  actor-only, critic-only and shared-trunk groups after iterations 12, 23, 34
  and 46, plus per-group gradient norms before the first optimizer step of
  each iteration. Verified in D's dry run and in D's and E's main runs; no
  new dry run for F.
- **Command:** `nohup /root/ablation/ablate-E-lr10-vf05/run_e.sh
  ablate-F-lr10-anchor01 753664 12,23,34,46 wandb -o optimizer.muon_lr=0.0002
  optimizer.adamw_lr=0.00001 rl.teacher_kl_coef=0.1`. The 10-minute
  `timeout` starts after the lock and idle wait. The first poll checks the
  override list, logged `teacher/kl_coef` 0.1 and the hook's θ0 record; the
  run is killed if the override did not reach the trainer.
- **Measurements, iterations 1–46:** `../extract_iterations.py` (teacher KL,
  unit/market kind KL, approx KL, clip fraction, advantage std, explained
  variance, LR, kl_coef, completed-game banks at iterations 12, 23, 34, 45)
  plus unweighted `loss/value_loss` and `loss/teacher_kl_loss` from the
  rank-0 iteration records; `../ablate-D-lr10-bccritic/trunk_audit_summary.py`
  for the trunk and grad audits. References: B's table (teacher KL at game
  ends 0.039 / 0.20 / 0.56 / 1.36, 0.64 at iteration 46), B's end-of-run
  change vs BC (shared 8.02e-3, actor-only 5.12e-3), E's in-run audit
  (same fresh-head load, shared 7.86e-3 at 46) and A's table.
- **Prediction if the anchor and smaller steps combine:** teacher KL at
  iteration 46 below B's 0.64 and game-4 banks at least B's 62k.
- **Discriminating observations:**
  1. Teacher KL below 0.64 at 46 (and below B's at each game end) with
     game-4 banks at least 62k: the two levers combine.
  2. Teacher KL lower but banks at or below B's: staying closer to the
     teacher does not by itself hold the economy at LR / 10.
  3. Teacher KL not lower than B's: at LR / 10 the anchor's share of the
     clipped, normalised trunk step is too small to matter.
- **Expectation, stated before launch:** E showed that at LR / 10 the
  shared-trunk gradient is dominated by the policy and teacher terms and
  that trunk movement is set by the step size. I therefore expect trunk
  movement magnitude within about 10 % of B's/E's, with the anchor changing
  its *direction*: teacher KL clearly below B's (plausibly 0.2–0.4 at 46),
  and game-4 banks near the game-1 level (about 70k). A 20x coefficient
  makes `teacher_kl_loss` comparable to or larger than the policy loss from
  early on, so the anchor could also slow useful learning; this run cannot
  separate "held near BC" from "learned less", since BC's own banks are the
  baseline.
- **Confounds, named in advance:**
  - Total grad norm was above `max_grad_norm` 10 in every B/E iteration; a
    larger teacher term changes what the clip scales.
  - The teacher is the anchor target, so a lower teacher KL is partly by
    construction; banks are the substantive outcome.
  - One seed; self-play banks are not strength (no held-out opponent,
    `rl.eval_replay_games=0`).
- **Stopping condition and budget:** 46 complete iterations, the first
  nonfinite metric or trainer fault, or the watchdog
  `timeout --kill-after=30 600` (10 minutes). B took 483 s and E 447 s,
  about $0.55 at $4.18/h. W&B is online. After the run it is renamed through
  the API to `ablate-F-lr10-anchor01` in group `kg-v3-ppo-collapse-ablation`.
  The GPUs are left idle afterwards.
