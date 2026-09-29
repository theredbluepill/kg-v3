# Run statement: PPO collapse ablation I, "ablate-I-full-critic-stopgrad" (full LR, critic stop-gradient)

Written before the 46-iteration launch (after the hook's dry run, which is
recorded below). **Pre-landing diagnostic** on `kg/pod-ppo-prelanding` (not
Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

**This arm uses a DIAGNOSTIC-ONLY hook that CHANGES TRAINING.** It is G's
unchanged `../ablate-G-lr10-critic-stopgrad/critic_stopgrad_launcher.py`
(`ee5af67b…d621`). It is not a training feature or a proposed recipe. It lives
only in the launcher's monkeypatch seam, outside `scripts/run_ppo.py` and
`python/owl`, and nothing in the repository imports it. No new code was
written for I.

- **Question:** G blocked the critic's gradient from the shared trunk at
  LR / 10. The trunk still moved as much as B's, but game-end teacher KL was
  about half of B's, and the banks were about B's. H showed that at the
  control's full LR the economy collapses whichever critic head PPO starts
  from. Round 1's C showed that value-only updates from a fresh critic head
  at full LR moved the trunk 3.4 % and destroyed the policy.

  At the control's full step size, is the value-loss gradient through the
  shared trunk what drives the collapse? If it is blocked, do the banks
  survive, with any remaining drift coming from the policy path only?
- **Single change versus the control (6.2, W&B `spoon/kg-v3/7k07gp7c`):**
  the value-loss gradient is blocked from the shared trunk by
  `KG_DIAG_CRITIC_STOPGRAD=on`, at the control's FULL learning rates (Muon
  0.002 / AdamW 1e-4; no LR override).
  - The hook replaces `KaggricultureTransformer._winner_log_probabilities`
    with a copy that applies `critic_head` to
    `encoded.critic_value_hidden.detach()`. Forward values are unchanged.
  - In backward, `vf_coef` × value loss and `teacher_value_coef` × teacher
    value CE reach `critic_head.*` only.
  - They do not reach the trunk blocks, stems, final norm, the actor or the
    `critic_value_tokens` input embeddings. The embeddings sit upstream of
    the detach and still get policy gradients through attention.

  Everything else matches the control:
  - pod checkout `e74d67e` (clean, 0 porcelain lines);
  - `configs/kaggriculture_2rank.yaml`;
  - `-o rl.eval_replay_games=0 rl.teacher_kl_coef=0.005`;
  - BC best `fd854587…6f51` with `--load-model-weights-mode model_fresh_critic_head`;
  - 2 ranks, `KG_PROBE_STOP_ON_NONFINITE=1`;
  - `--max-env-steps 753664` (46 × 16,384);
  - `vf_coef` 2.0 and `teacher_value_coef` 0.005.
- **Launcher:** G's `run_g.sh` (`a2fb7965…81b2`), unchanged and invoked from
  G's pod folder.
  - It already passes `-o rl.eval_replay_games=0`, so I passes only
    `-o rl.teacher_kl_coef=0.005`. H's attempt 1 shows that a duplicate is
    rejected.
  - It loads D's unchanged, telemetry-only
    `../ablate-D-lr10-bccritic/trunk_audit_launcher.py` (`f98720bd…78f9`).
    That hook provides the trunk audit: ||θ − θ0|| / ||θ0|| from the loaded
    weights for actor-only, critic-only and shared parameters, at
    iterations 12, 23, 34 and 46. It also provides the per-iteration grad
    audit.
  - G's hook adds `critic_split_audit`, which separates `critic_head` from
    `critic_value_tokens`.
- **Hook verification (dry run, done before this statement;
  `dryrun-receipts/`).** `ablate-I-dryrun-on` ran 2 iterations at 32,768
  steps with W&B off. It used the same launcher, hooks and overrides as the
  main run, including full LR. Exit 0 after 40 s. All 12 input hashes equal
  G's main run.
  - Logged LR was 3.2e-5 at iteration 1 and 6.4e-5 at iteration 2. These
    equal the control's schedule and are 10x G's. `teacher/kl_coef` was
    0.005.
  - The value-only gradient check ran on both ranks at iterations 1 and 2
    (optimizer steps 0 and 16). It is summarised in
    `dryrun-receipts/stopgrad_check_dryrun.tsv`. It passed at every point:
    - shared trunk: 157 of 157 tensors had no gradient path, max |grad|
      exactly 0;
    - actor-only (48 tensors) and `critic_value_tokens`: exactly 0;
    - `critic_head`: gradient norm 0.21 / 0.29 at iteration 1 and
      0.24 / 0.26 at iteration 2 (ranks 0 / 1), non-zero;
    - policy-side loss graph: absent.
  - One of the 4 `critic_head` tensors is all-zero, as in G (presumably the
    output bias, which cancels in the two-way winner softmax; inferred, not
    checked by name).
  - Non-vacuity comes from G's `off` control dry run (same hook and check,
    detach not installed). It found a value-side shared-trunk gradient of
    norm 1.06–1.29. It was not repeated at full LR, because the check at
    iteration 1 runs before any optimizer step.
- **Command:**

  ```
  nohup /root/ablation/ablate-G-lr10-critic-stopgrad/run_g.sh ablate-I-full-critic-stopgrad 753664 12,23,34,46 wandb on 1 -o rl.teacher_kl_coef=0.005
  ```

  The check runs again at iteration 1 of the main run, and the launcher
  raises if it fails. The first poll checks three things: the argv
  overrides, the `diagnostic_hook` record (`critic_stopgrad: on`) and a
  passing `stopgrad_check`. The run is killed if any is missing.
- **Measurements, iterations 1–46:**
  - `../extract_iterations.py` for teacher KL, unit and market kind KL,
    approx KL, clip fraction, advantage std, explained variance, LR and
    completed-game banks;
  - `loss/value_loss` from the rank-0 iteration records;
  - the trunk, grad and critic-split audits from the `[kg-probe]` records.
  - References:
    - control: `../ablate-A-anchor/control-6.2-iterations-1-46.tsv`; banks
      74k / 55k / 7.3k / 0.09k; teacher KL 5.28 at iteration 46; no in-run
      trunk audit, but its end-of-run change vs BC was about 7.4e-2
      (comparison.md);
    - H (full LR, BC head): in-run shared 8.14e-2 at iteration 46;
    - G (LR / 10, stop-grad): shared 7.81e-3.
- **Prediction** (from the orchestrator, for the case where the value path
  is the main driver even at full LR): banks far better than the control's,
  with the remaining drift coming from the policy path only.
- **Discriminating observations:**
  1. Game-4 banks far above the control's 0.09k (at least A's 41k) and
     teacher KL well below 5.28: the value gradient through the trunk is the
     main driver at full LR.
  2. Banks collapse as in the control (under about 10k by game 4): the
     policy and teacher path at this step size is enough to collapse the
     economy without any value gradient in the trunk. This matches H, where
     a well-fitted critic did not prevent collapse.
  3. Banks fall clearly more slowly than the control's (roughly 10–40k at
     game 4), with teacher KL lower at every game end: the value path
     contributes but is not the whole cause.
  4. Explained variance clearly below the control's, with banks no better:
     a critic head on value-blind features gives worse advantages.
- **Expectation, stated before launch (mine, distinct from the prediction
  above):** I expect observation 2, possibly with a delay as in observation
  3.
  - Movement follows the step size. B, E, F and G moved about 8e-3 at
    LR / 10, and H, A and C moved about 7.4–8.1e-2 at full LR, whatever the
    loss mix. The critic path is about 1.2 of a shared-gradient norm of
    about 11–14 (G's dry runs; this dry run's iteration-1 shared norm was
    13.7).
  - I therefore expect a shared change of about 7–8e-2 at iteration 46.
  - I expect game-end teacher KL below the control's. G halved B's; here
    perhaps 0.3–0.6x the control's at game ends 1–3.
  - I expect the banks still to fall far, under 20k by game 4, because H
    collapsed even with a strong critic.
  - This is an expectation, not a threshold; the run decides.
- **Confounds, named in advance:**
  - The detach also blocks the teacher value-distillation gradient from the
    trunk. Its coefficient is 0.005, against `vf_coef` 2.0.
  - `critic_value_tokens` still move under the policy gradient.
  - The total grad norm often exceeds `max_grad_norm` 10 at this LR (the
    control's range was 3.7–47.9). Removing the critic part changes what the
    clip scales.
  - The rollout is nondeterministic from run to run (G's on/off dry runs
    differed at iteration 1).
  - One seed. Self-play banks are not strength: no held-out opponent ran
    (`rl.eval_replay_games=0`).
- **Stopping condition and budget:** the run stops at the first of:
  - 46 complete iterations;
  - the first nonfinite metric or trainer fault;
  - a failed stop-grad check;
  - the watchdog `timeout --kill-after=30 600` (10 minutes).

  H took 405 s at this LR, about $0.47 at $4.18/h. The dry run took 40 s,
  about $0.05. W&B is online. After the run it is renamed through the API
  to `ablate-I-full-critic-stopgrad` in group `kg-v3-ppo-collapse-ablation`.
  The GPUs are left idle afterwards.
