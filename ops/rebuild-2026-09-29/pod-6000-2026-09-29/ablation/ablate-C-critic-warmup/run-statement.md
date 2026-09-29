# Run statement: PPO collapse ablation C, "ablate-C-critic-warmup" (critic warm-up)

Written before launch. **Pre-landing diagnostic** on `kg/pod-ppo-prelanding`
(not Codex-verified). Owner-approved as one of the three collapse ablations
(A anchor, B smaller steps, C critic warm-up; about $2 in total). No selection,
ranking or submission follows from it.

**DIAGNOSTIC-ONLY HOOK.** The warm-up has no config knob. It is implemented
outside `scripts/run_ppo.py` and `python/owl` in
`critic_warmup_launcher.py` (this folder), which loads the shared early-smoke
launcher unchanged (`b393ad31…92df`, the same bytes as the control, A and B)
and adds a loss mask and a gradient audit through the same `PPOTrainer`
monkeypatch seam. Nothing in the repository imports it; it is not a training
feature and makes no claim about how a warm-up should be built.

- **Question:** in the 6.2 control (W&B `spoon/kg-v3/7k07gp7c`) the economy
  collapsed from the BC best (banks 73k, 55k, 7.3k, 0 over games 1–4). B
  showed the drift is LR-bound (LR / 10: banks 73k → 62k) but left open
  whether the PPO signal itself is noise because the critic starts fresh
  (`model_fresh_critic_head`; control explained variance swings from about
  -0.4 to 0.5 within each game). If the critic is trained first while the
  policy is held, do advantages become informative, and does PPO from
  iteration 23 then drift more slowly and keep banks longer than the
  control? And, separately: do critic-only updates through the shared trunk
  by themselves move the policy?
- **Single change versus the control:** for iterations 1–22 (2 full games;
  352 optimizer steps) every PPO loss call has advantages set to 0 (the
  clipped policy-gradient term is exactly 0), `teacher_kl_coef` 0 and
  `ent_coef` 0; the value loss (`vf_coef` 2.0) and teacher value distillation
  (`teacher_value_coef` 0.005) are unchanged, so only the critic side trains.
  The shared trunk still moves under the value loss (by design; measured).
  From iteration 23 the loss is passed through untouched: control PPO. The
  optimizer step count and so the LR schedule advance during warm-up exactly
  as in the control (iteration 23 LR 7.36e-4 in both). Everything else
  matches 6.2/A/B: pod checkout `e74d67e` (clean),
  `configs/kaggriculture_2rank.yaml` (Muon 0.002 / AdamW 1e-4, teacher KL
  0.005), BC best `fd854587…6f51` with `model_fresh_critic_head`,
  `-o rl.eval_replay_games=0`, 2 ranks, `KG_PROBE_STOP_ON_NONFINITE=1`,
  `--max-env-steps 753664` (46 x 16,384).
- **Hook verification (done, before this launch):** a 2-iteration dry run
  (`run_c.sh ablate-C-dryrun 1 32768 all debug`, W&B off, exit 0, 41 s;
  receipts in `dryrun-receipts/`, table `grad_audit.tsv`). With warm-up = 1
  iteration and the audit on every optimizer step, after backward and the
  DDP all-reduce: actor-only parameters (`actor_input_proj.*`, `actor.*`;
  873,406 values, reached only by policy terms) had gradient L2 norm 0 and
  max |grad| 0 on all 16 warm-up steps on both ranks, with no None grads;
  on the 16 PPO steps of iteration 2 their norm was 2.75–5.46. Critic-only
  (66,561) and shared (5,312,256) gradients were nonzero throughout. Logged
  `loss/policy_loss`, `loss/teacher_kl_loss`, `loss/entropy_loss` were 0 in
  iteration 1 and nonzero in 2. Actor-only weights still changed by a
  relative 2.1e-6 over the 16 warm-up steps (Muon decoupled weight decay,
  0.01 x LR; control code path, not removed) against 1.9e-4 after 16 PPO
  steps. Two earlier dry-run attempts failed fast in the audit's parameter
  grouping (`_ddp.module.model.` prefix; receipts `dryrun-receipts/failed-attempt1..2`)
  before any update; fixed in the committed hook `3addc8e6…7e90`.
  Dry-run observation, not yet a result: in iteration 1, value-only updates
  already moved the policy (approx KL 0.0022, teacher KL 0.0021, against
  the control's 0.0026 / 0.0026 with full PPO), and explained variance was
  0.62 (control -0.05).
- **Command:** `run_c.sh ablate-C-critic-warmup 22 753664 first wandb`
  (`87386d14…1ab8`; a copy of `../run_ablation.sh` with the hook as the
  torchrun entry, the hook hash added to the receipts, and the audit on the
  first optimizer step of each iteration).
- **Measurements, iterations 1–46:** entropy, approx KL, clip fraction,
  advantage std, explained variance, `teacher/kl`, `teacher/unit_kind_kl`,
  `teacher/market_kind_kl`, learning rate, and the completed-game final banks
  (`train/terminal_bank_0/1` at iterations 12, 23, 34, 45), via
  `../extract_iterations.py`; per-iteration grad audit via
  `audit_summary.py`. Compared with the control's table
  (`../ablate-A-anchor/control-6.2-iterations-1-46.tsv`), A and B. Entropy is
  compared at the same game phase only (B showed it swings 4.4 → 8 within
  every game from the BC policy itself).
- **Prediction if the hypothesis holds:** during warm-up teacher KL and
  approx KL stay near 0 (well below the control's 0.56 at iteration 13 and
  2.37 at iteration 23), entropy follows the BC game-phase curve, explained
  variance rises and stays higher than the control's; banks of games 1–2
  stay near the BC level (about 73k). After warm-up, teacher KL grows more
  slowly than the control's from iteration 23, and games 3–4 banks stay well
  above the control's 7.3k / 0.
  **Discriminating observations:** (1) if teacher KL grows materially
  during warm-up, critic-only trunk updates by themselves disturb the policy
  (an architecture/coupling finding, not a PPO-signal one); (2) if after
  warm-up drift and collapse proceed at the control's pace despite a
  trained critic, the fresh-critic noise is not the main cause and step
  size (B) remains the lever; (3) if banks hold after warm-up, critic
  quality matters.
- **Stopping condition and budget:** 46 complete iterations
  (`--max-env-steps 753664`), the first nonfinite metric or trainer fault, or
  the external watchdog `timeout --kill-after=30 600` (10 min). About 8
  minutes of pod time is about $0.55 at $4.18/h (the dry runs used about
  2 minutes, $0.14). W&B is online; the run is renamed
  `ablate-C-critic-warmup` in group `kg-v3-ppo-collapse-ablation` through
  the W&B API after it finishes (the trainer hardcodes group `ppo`). GPUs are
  left idle afterwards.
