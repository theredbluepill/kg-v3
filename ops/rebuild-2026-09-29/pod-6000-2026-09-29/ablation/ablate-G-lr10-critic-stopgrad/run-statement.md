# Run statement: PPO collapse ablation G, "ablate-G-lr10-critic-stopgrad" (LR / 10, critic stop-gradient)

Written before the 46-iteration launch (after the hook's dry runs, which are
recorded below). **Pre-landing diagnostic** on `kg/pod-ppo-prelanding` (not
Codex-verified). Owner approval for the round-2 attribution runs, in the
owner's words: "can you just use your subagents and massively attrivite the
learening? you get my go for all" (budget set by the orchestrator: $12 in
total across the arms). No selection, ranking or submission follows from it.

**This arm uses a DIAGNOSTIC-ONLY hook that CHANGES TRAINING**
(`critic_stopgrad_launcher.py`, this folder). It is not a training feature or
a proposed recipe. It lives only in the launcher's monkeypatch seam, outside
`scripts/run_ppo.py` and `python/owl`, and nothing in the repository imports it.

- **Question:** round 1 arm C showed that value-only updates from a fresh
  critic head moved the shared trunk 3.4 % and destroyed the policy. The
  fresh head's gradient norm was 12.4, against 0.118 for the BC head, with
  `vf_coef` 2.0. At B's step size (LR / 10), is the critic's gradient
  through the shared trunk what drives the remaining drift and bank decline?
  If the value loss can no longer reach the trunk, does the trunk move less,
  does the policy stay closer to BC, and do the banks hold?
- **Single change versus B (`ablate-B-lr10`, W&B `spoon/kg-v3/32pahqok`):**
  the value-loss gradient is blocked from the shared trunk. The hook replaces
  `KaggricultureTransformer._winner_log_probabilities`, the one function that
  turns encoder output into critic output, with a copy that applies
  `critic_head` to `encoded.critic_value_hidden.detach()`. Forward values are
  unchanged. In backward, both critic-output losses reach `critic_head.*`
  only: `vf_coef` × value loss and `teacher_value_coef` × teacher value CE,
  which is the student side of the value distillation (coefficient 0.005).
  These gradients do not reach:
  - the trunk blocks, stems and final norm;
  - the actor;
  - the `critic_value_tokens` input embeddings. These are trunk inputs
    upstream of the detach, and they still receive policy gradients through
    attention.

  So "the critic's own parameters" trained by the value loss means
  `critic_head` alone.

  Everything else matches B:
  - pod checkout `e74d67e` (clean) and `configs/kaggriculture_2rank.yaml`;
  - `-o rl.eval_replay_games=0 optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`;
  - BC best `fd854587…6f51` with `--load-model-weights-mode model_fresh_critic_head`;
  - 2 ranks, seeds 0/1, `KG_PROBE_STOP_ON_NONFINITE=1`;
  - `--max-env-steps 753664` (46 × 16,384);
  - `vf_coef` 2.0, `teacher_kl_coef` 0.005 and `teacher_value_coef` 0.005.
- **Launcher:** `run_g.sh` (this folder) is `../ablate-E-lr10-vf05/run_e.sh`
  with three changes:
  - the torchrun entry is `critic_stopgrad_launcher.py`;
  - two positional arguments, `STOPGRAD on|off` and `CHECK_ITERS`, are
    added;
  - both hooks are hashed into `hashes.sha256`.

  The G hook loads D's unchanged, telemetry-only
  `../ablate-D-lr10-bccritic/trunk_audit_launcher.py` (`f98720bd…78f9`).
  That hook loads the unchanged shared launcher (`b393ad31…`). The run keeps
  D's trunk audit (||θ − θ0|| / ||θ0|| for actor-only, critic-only and
  shared parameters at iterations 12, 23, 34 and 46) and D's per-iteration
  grad audit. G adds a `critic_split_audit` record at the same iterations,
  splitting critic-only into `critic_head` and `critic_value_tokens`.
- **Hook verification (dry runs, done before this statement):** each dry run
  was 2 iterations at 32,768 steps with W&B off. Receipts are in
  `dryrun-receipts/`, summarised in
  `dryrun-receipts/stopgrad_check_dryrun.tsv` by `stopgrad_check_summary.py`.
  - **Check method.** The check runs on the first PPO-loss call of
    iterations 1 and 2, on both ranks. It re-evaluates the eager
    `owl.train.ppo._ppo_loss` on the real minibatch inputs with `new_logp`,
    `entropy` and `teacher_kl` detached and a local reduction, so the only
    differentiable part of the loss is `vf_coef` × value loss +
    `teacher_value_coef` × teacher value CE. It then calls
    `torch.autograd.grad` over every parameter. `.grad` is not touched, so
    training is unaffected.
  - **`on` (`ablate-G-dryrun-on`, final hook `ee5af67b…d621`, exit 0,
    40 s):** the check passed on both ranks at iterations 1 and 2.
    - Shared trunk: 157 of 157 tensors had no gradient path (None), norm
      exactly 0.
    - Actor-only (48 tensors) and `critic_value_tokens`: None, exactly 0.
    - `critic_head`: norm 0.14–0.23, non-zero. One of its 4 tensors was
      all-zero in both `on` and `off`. It is presumably the output bias,
      which cancels in the two-way winner softmax; this is inferred, not
      checked by name.
    - The policy-side loss had no graph (`policy_loss_has_grad` false).
  - **`off` control (`ablate-G-dryrun-off`, same hook, detach not
    installed, exit 0, 43 s):** the same check found a value-side gradient
    in the shared trunk of norm 1.06–1.29 and in `critic_value_tokens` of
    0.11–0.13. So the zero in `on` is the detach, not a vacuous check. This
    run needed `KG_DIAG_DISABLE_DONATED_BUFFER=1`
    (`torch._functorch.config.donated_buffer=False`). Its first attempt
    (`ablate-G-dryrun-off-attempt1-donated-buffer-fail`) failed: a
    `retain_graph` backward through the compiled trunk is not allowed with
    donated buffers. The `on` check never reaches the compiled trunk, so
    the main run leaves this unset.
  - The first `on` attempt (`ablate-G-dryrun-on-attempt1-prefinal-hook`,
    hook `00c7bc87…`) passed identically. It was rerun only so that the
    verified hook is byte-identical to the one launched.
  - **Grad audit.** Total gradients after DDP were read before the first
    step of each iteration. The shared-trunk norm was 10.92 / 12.05 with
    `on` and 10.80 / 11.45 with `off`, so the critic path is a small part
    of the shared-trunk gradient at this LR (see Expectation).
- **Command:** `nohup /root/ablation/ablate-G-lr10-critic-stopgrad/run_g.sh
  ablate-G-lr10-critic-stopgrad 753664 12,23,34,46 wandb on 1 -o
  optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`.
  - The check runs again at iteration 1 of the main run, and the launcher
    raises if it fails.
  - The first poll checks the overrides, the `diagnostic_hook` record
    (`critic_stopgrad: on`) and the passing `stopgrad_check`. The run is
    killed if either is missing.
- **Measurements, iterations 1–46:**
  - `../extract_iterations.py` for teacher KL, unit and market kind KL,
    approx KL, clip fraction, advantage std, explained variance, LR and
    completed-game banks. Unweighted `loss/value_loss` comes from the
    rank-0 iteration records.
  - `../ablate-D-lr10-bccritic/trunk_audit_summary.py` for the trunk and
    grad audits, plus `critic_split_audit`.
  - References: B's table (teacher KL 0.64 at 46; banks 73k / 65k / 64k /
    62k), and B's checkpoint-vs-BC change (shared 8.02e-3, actor-only
    5.12e-3). E has the same fresh-head load and an in-run audit (shared
    7.86e-3 at 46); F has 7.78e-3.
- **Prediction if the value path through the trunk is the main driver:**
  - the trunk moves far less than in B;
  - teacher KL at iteration 46 is well below B's 0.64;
  - banks stay about 70k and flat over games 1–4.
- **Discriminating observations:**
  1. Shared-trunk change well below about 8e-3, teacher KL well below 0.64
     and flat banks of about 70k: the critic path was the main driver.
  2. Trunk change like B's, but banks or teacher KL clearly better than B's:
     the value gradient's direction, not the step size, drove part of the
     damage.
  3. Trunk change, teacher KL and banks all like B's: the critic's gradient
     through the trunk is not the driver at LR / 10. The drift comes from
     the policy and teacher terms, the step size, or advantage quality.
  4. Banks worse than B's with explained variance clearly lower: a critic
     head on frozen features gives worse advantages, and that cost
     outweighs any trunk protection.
- **Expectation, stated before launch (mine, distinct from the prediction
  above):** I expect the trunk-movement part of the prediction to fail.
  - Muon normalises its update, so the step magnitude is set by the LR,
    not by the gradient norm. B, E and F all moved the shared trunk by
    7.8–8.0e-3 whatever their loss mix.
  - The dry runs show the critic path is about 1.2 of a shared-gradient
    norm of about 11, so removing it changes the update direction slightly
    and its size hardly at all.
  - I therefore expect shared change within about 10 % of B's/E's and
    teacher KL near B's (0.4–0.8).
  - I expect the critic to be weaker than B's: explained variance lower,
    since only a 66k-parameter head learns on trunk features that no longer
    adapt to value. Banks should be near B/E (62–67k at game 4), and not
    flat at 70k.
  - This is an expectation, not a threshold; the run decides.
- **Confounds, named in advance:**
  - The detach also blocks the teacher value-distillation gradient from
    the trunk (both use `_winner_log_probabilities`). It cannot be separated
    here, but its coefficient is 0.005 against `vf_coef` 2.0.
  - `critic_value_tokens` still move under the policy gradient, so the
    critic's input embedding is not frozen, only value-blind.
  - Total grad norm was above `max_grad_norm` 10 in B/E/F; removing the
    critic part changes what the clip scales.
  - Iteration-1 grad norms differed between the `on` and `off` dry runs
    (actor-only 3.90 against 3.69), although forward values are identical.
    This points to run-to-run rollout nondeterminism. It is not
    investigated here.
  - One seed. Self-play banks are not strength: no held-out opponent ran
    (`rl.eval_replay_games=0`).
- **Stopping condition and budget:** the run stops at the first of:
  - 46 complete iterations;
  - the first nonfinite metric or trainer fault;
  - a failed stop-grad check;
  - the watchdog `timeout --kill-after=30 600` (10 minutes).

  B took 483 s, about $0.56 at $4.18/h. The dry runs took 26 + 39 + 43 +
  40 s, about $0.17. W&B is online. After the run it is renamed through the
  API to `ablate-G-lr10-critic-stopgrad` in group
  `kg-v3-ppo-collapse-ablation`. The GPUs are left idle afterwards.
