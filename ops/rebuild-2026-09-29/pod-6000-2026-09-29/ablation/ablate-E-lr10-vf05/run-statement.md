# Run statement: PPO collapse ablation E, "ablate-E-lr10-vf05" (LR / 10, vf_coef 0.5)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

- **Question:** round 1 (`../comparison.md`) showed that in C value-only
  updates with a fresh critic head moved the shared trunk 3.4 % and destroyed
  the BC policy. The fresh critic head's synthetic handoff gradient norm was
  12.4 against 0.118 for the BC head, and `vf_coef` is 2.0. If the weight of
  the value loss flowing through the shared trunk drives the drift that
  lowered B's banks (73k to 62k over 4 games), does cutting that weight 4x at
  B's step size keep the policy closer to BC and the economy at or above B's?
- **Single change versus B (`ablate-B-lr10`, W&B `spoon/kg-v3/32pahqok`):**
  `-o rl.vf_coef=0.5` (B: 2.0 from `configs/kaggriculture_2rank.yaml`).
  Everything else matches B: pod checkout `e74d67e` (clean),
  `configs/kaggriculture_2rank.yaml`, `-o rl.eval_replay_games=0
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`, BC best
  `fd854587…6f51` with `--load-model-weights-mode model_fresh_critic_head`,
  2 ranks, seeds 0/1, `KG_PROBE_STOP_ON_NONFINITE=1`, `--max-env-steps
  753664` (46 x 16,384). `teacher_kl_coef` 0.005 and `teacher_value_coef`
  0.005 are unchanged; `vf_coef` multiplies only `value_loss` in
  `python/owl/train/ppo.py` (lines 1849, 1954, 1984 at `e74d67e`).
- **Launcher:** `run_e.sh` (this folder) is D's `run_d.sh` with the load mode
  set back to `model_fresh_critic_head` (the diff is the header comments, the
  `load_mode` receipt label and the `--load-model-weights-mode` value). It
  reuses D's DIAGNOSTIC-ONLY, telemetry-only hook
  `../ablate-D-lr10-bccritic/trunk_audit_launcher.py` (`f98720bd…78f9`,
  unchanged). The hook only reads parameters and gradients; it snapshots
  θ0 at the start of iteration 1 (after the fresh head is initialised, so the
  critic-only group has a real θ0 in this arm), and emits the relative change
  ||θ − θ0|| / ||θ0|| for actor-only, critic-only and shared-trunk groups
  after iterations 12, 23, 34 and 46, plus per-group gradient norms before
  the first optimizer step of every iteration. The hook was verified in D's
  dry run and main run; no new dry run for E. The override list and the
  hook's θ0 record are checked in the first poll, and the run is killed if
  `rl.vf_coef=0.5` did not reach the trainer.
- **Command:** `nohup run_e.sh ablate-E-lr10-vf05 753664 12,23,34,46 wandb -o
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001 rl.vf_coef=0.5`. It
  takes the exclusive `flock` on `/root/ablation/gpu.lock` and waits for idle
  GPUs, because other round-2 arms share the pod; the 10-minute `timeout`
  starts after that wait.
- **Measurements, iterations 1–46:** `../extract_iterations.py` (teacher KL,
  unit/market kind KL, approx KL, clip fraction, advantage std, explained
  variance, LR, completed-game banks at iterations 12, 23, 34, 45) plus
  `loss/value_loss` from the rank-0 iteration records;
  `../ablate-D-lr10-bccritic/trunk_audit_summary.py` for the trunk and grad
  audits. The reference is B's table and B's end-of-run checkpoint change
  (shared 8.02e-3, actor-only 5.12e-3 at iteration 46,
  `../ablate-D-lr10-bccritic/pod-receipts/final_checkpoint_change_vs_bc.tsv`).
  Note that E's value loss is reported unweighted, so it compares with B's
  directly.
- **Prediction if the value-loss weight through the trunk drives the drift:**
  less trunk movement than B, teacher KL at iteration 46 below B's 0.64, and
  banks at game 4 at least B's 62k.
- **Discriminating observations:**
  1. Teacher KL at 46 below 0.64 and game-4 banks at least 62k: the value
     share of the trunk gradient is a lever at this LR.
  2. Teacher KL and banks at or worse than B's: the value-loss weight is not
     the driver at LR / 10.
  3. Explained variance clearly below B's (mean 0.29): the critic learned
     less, and any policy effect is confounded by a worse baseline.
- **Expectation against the premise (stated before launch):** D found that
  trunk movement is set by the step size, not by the value gradient
  (D 8.43e-3 vs B 8.02e-3 shared at equal LR). The trainer's Muon and AdamW
  updates are close to scale-invariant in the loss weight, so I expect E's
  trunk *magnitude* to be within about 10 % of B's; "less trunk movement" is
  therefore not expected to be the discriminating part of the prediction.
  What vf_coef changes is the *composition* of the shared-trunk gradient
  (value versus policy share) and, through `max_grad_norm` 10 (B's total
  grad norm was 12.6–14.7, so clipped), the effective scaling of the policy
  terms. For parameters reached only by the value loss (the critic head), a
  loss weight mostly cancels in the normalised updates, so the critic head's
  own learning speed should change little.
- **Confounds, named in advance:**
  - The clip at `max_grad_norm` 10 may stop binding with a smaller value
    term, changing the effective policy-gradient scale for AdamW parameters
    as well as the value share.
  - Explained variance may change through the trunk features the critic
    reads, which changes the advantages.
  - One seed; self-play banks are not strength (no held-out opponent,
    `rl.eval_replay_games=0`).
- **Stopping condition and budget:** 46 complete iterations, the first
  nonfinite metric or trainer fault, or the watchdog
  `timeout --kill-after=30 600` (10 minutes). B took 483 s and D 439 s, about
  $0.55 at $4.18/h. W&B is online. After the run it is renamed through the
  API to `ablate-E-lr10-vf05` in group `kg-v3-ppo-collapse-ablation`. The
  GPUs are left idle afterwards.
