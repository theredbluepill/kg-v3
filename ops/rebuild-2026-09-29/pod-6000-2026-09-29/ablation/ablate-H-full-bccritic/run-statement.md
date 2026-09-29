# Run statement: PPO collapse ablation H, "ablate-H-full-bccritic" (full LR, BC critic head)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

- **Question:** D (`../ablate-D-lr10-bccritic`) kept the BC critic head at
  B's LR / 10 and banks rose every game (73k to 85k), while B (fresh head,
  LR / 10) fell to 62k and the control (fresh head, full LR) collapsed to
  about 0 by game 4. D's trunk moved as far as B's, so the head changed the
  direction of the drift, not its size (D's critic explained a mean 0.84 of
  return variance; B's 0.29). Is the fresh critic head the main driver of the
  collapse even at the control's full step size, or does the full LR collapse
  the economy whichever critic head PPO starts from?
- **Single change versus the control (6.2, W&B `spoon/kg-v3/7k07gp7c`,
  iterations 1–46):** `--load-model-weights-mode model_only`, which loads the
  whole BC model including `critic_head.*`, instead of
  `model_fresh_critic_head`. As in D, the initial last-best teacher is copied
  from the loaded model, so teacher value distillation (0.005) targets the BC
  critic; that follows from the load mode and is part of the same change.
  Everything else matches the control: pod checkout `e74d67e` (clean, 0
  porcelain lines), `configs/kaggriculture_2rank.yaml` with its full LRs
  (Muon 0.002, AdamW 1e-4), `teacher_kl_coef` 0.005 (passed explicitly),
  `vf_coef` 2.0, `rl.eval_replay_games=0`, BC best `fd854587…6f51`, 2 ranks,
  seeds 0/1, `KG_PROBE_STOP_ON_NONFINITE=1`, `--max-env-steps 753664`
  (46 x 16,384). H is also D at the full LR.
- **Launcher:** `../ablate-D-lr10-bccritic/run_d.sh` reused **unchanged**
  (`ed87c412…1b22`, pod copy hash-equal to the committed one). Its load mode
  `model_only` is D's. Its only differences from `../run_ablation.sh` are the
  load mode, the DIAGNOSTIC-ONLY, telemetry-only hook
  `../ablate-D-lr10-bccritic/trunk_audit_launcher.py` (`f98720bd…78f9`,
  unchanged) and the exclusive `flock` on `/root/ablation/gpu.lock` with an
  idle-GPU wait, because other round-2 arms share the pod. The hook only
  reads parameters and gradients: it snapshots θ0 at the start of iteration 1
  and emits ||θ − θ0|| / ||θ0|| for the actor-only (`actor_input_proj.*`,
  `actor.*`), critic-only (`critic_head.*`, `critic_value_tokens`) and
  shared-trunk groups after iterations 12, 23, 34 and 46, plus per-group
  gradient norms before the first optimizer step of each iteration. It is
  outside `scripts/run_ppo.py` and `python/owl`. Verified in D's dry run and
  in the D–G main runs; no new dry run for H.
- **Command:** `nohup /root/ablation/ablate-D-lr10-bccritic/run_d.sh
  ablate-H-full-bccritic 753664 12,23,34,46 wandb -o rl.teacher_kl_coef=0.005
  rl.eval_replay_games=0`. The 10-minute `timeout` starts after the lock and
  idle wait. The first poll checks the override list, the logged LR at
  iteration 1 (must equal the control's, 10x D's 3.2e-6) and the hook's
  critic θ0 norm (22.0106, the BC head); the run is killed if either is wrong.
- **Measurements, iterations 1–46:** `../extract_iterations.py` (teacher KL,
  unit/market kind KL, approx KL, clip fraction, advantage std, explained
  variance, LR, completed-game banks at iterations 12, 23, 34, 45) plus
  `loss/value_loss` from the rank-0 iteration records;
  `../ablate-D-lr10-bccritic/trunk_audit_summary.py` for the trunk and grad
  audits. References: the control's table
  (`../ablate-A-anchor/control-6.2-iterations-1-46.tsv`; banks 74k / 55k /
  7k / 0.1k, teacher KL 0.56 at 13 and 5.28 at 46), A's and C's end-of-run
  change vs BC at full LR (shared 7.41e-2 and 7.26e-2), and D's in-run audit
  (shared 8.43e-3 at 46).
- **Prediction if the fresh head is the main driver even at full LR:** banks
  at game 4 far above the control's (>> 0; at least A's 41k would make the
  head a stronger lever than a 20x anchor), with explained variance staying
  high as in D.
- **Discriminating observations:**
  1. Game-4 banks >> 0 (tens of thousands) and not falling steeply game over
     game: the fresh head is the main driver; the full LR is survivable with
     the BC critic.
  2. Banks fall as in the control (near 0 by game 3–4): the full step size
     collapses the economy regardless of the head; D's rise needs the small
     LR.
  3. Banks fall but clearly slower than the control (e.g. 20–50k at game 4):
     both levers contribute; neither alone attributes the collapse.
- **Expectation, stated before launch:** from D, trunk movement is set by
  the step size, so I expect shared-trunk change about 10x D's (roughly 7e-2
  at 46, like A and C). Whether informative advantages survive a 10x larger
  step is the open question; I lean to outcome 3 but hold no strong prior.
- **Confounds, named in advance:**
  - The load mode changes three things at once (critic initial function,
    teacher value target, initial value-gradient magnitude), as in D.
  - Self-play banks are not strength; `rl.eval_replay_games=0`, so no win
    rate or bank margin against another opponent.
  - Single-iteration values need a matched game phase (all envs reset
    together; one game is about 11.25 iterations).
  - One seed.
- **Stopping condition and budget:** 46 complete iterations, the first
  nonfinite metric or trainer fault, or the watchdog
  `timeout --kill-after=30 600` (10 minutes). D took 439 s, about $0.51 at
  $4.18/h. W&B is online. After the run it is renamed through the API to
  `ablate-H-full-bccritic` in group `kg-v3-ppo-collapse-ablation`. The GPUs
  are left idle afterwards.
