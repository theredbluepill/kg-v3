# Design: optimizer-first self-play (update only on real signal)

Status: a design, Claude-authored and not Codex-verified. Nothing has been
launched, edited or run for it. Tree: `/Users/poonszesen/kg-v3-int` @ 3e89425.
Inputs: `evidence-runs.md`, `evidence-econ.md`, `evidence-ppo.md` and
`evidence-v2.md` in this folder, plus the code seams cited below, which I
re-read for this design.

Owner constraints kept:
- Self-play only.
- Reward is term M, unchanged: 0.5 x cash difference plus 0.5 x terminal sign.
- `scripts/run_ppo.py` stays the only trainer.
- No adaptive anchor.
- The policy stays stateless: no memory and no opponent identity anywhere.
- No run without a stated mechanism, discriminating observation and stopping
  condition.

## 1. Diagnosis

### Thesis

There are two mechanisms, A and B. The run evidence cannot yet tell their
effects apart.

**A. The step size does not depend on the signal (supported).**
- Muon updates every 2-D trunk matrix (`optimizer.py:231-262`). Muon
  orthogonalizes the momentum gradient, so each step has a fixed spectral size.
  At lr 1e-4 on the 256-wide trunk, that is about 6e-6 RMS per element
  (lr x sqrt(max(1, A/B)) x sqrt(rank)/sqrt(A·B)), whether the gradient is
  signal or noise. A near-zero gradient still produces a full-size step: Muon
  has no way to stand still.
- AdamW updates the heads. Its step is also scale-free.
- Per-minibatch advantage normalization (`ppo.py:1476`, `:3616-3644`) scales a
  raw advantage std of 0.02 back up to 1.
- Each optimizer step sees 16 games of one synchronized game phase, which is
  2,048 player-turns (16 segments x 64 turns x 2 seats). It takes 16 such steps
  per iteration, with momentum 0.95 correlating consecutive steps.
- So the policy moves the same distance every iteration: approx_kl is about
  0.012 at lr 1e-4 in every run, across advantage std from 0.029 to 0.39.
  When the signal is weak, that movement is mostly noise.
- BC sits at a narrow optimum of a precise chain: buy, plant, water daily, feed,
  harvest, sell. From such an optimum, most displacement directions lose money.
  That explains a slow, persistent slide at every learning rate, with the rate
  set by LR.

**B. A weak signal that points the wrong way on long-payback investments (open).**
- Under term M a purchase is an immediate margin loss. Its payback arrives
  100-300 turns later.
- lambda 0.9 credits about 10 turns, and the critic is a bounded zero-sum
  winner head.
- v2 measured exactly this inversion. Buy-9 beat buy-5 in 16 of 16
  continuations, yet its advantage was negative in 0 of 16 pairs, and training
  moved the logits the wrong way. The fix was lambda 1 with 256-turn segments.
- M@20M dropped strawberry tile-days from 200 to 5 and sheep from 133 to 23, and
  7 of 8 seats never buy land. That is a *directed* pattern, which fits B at
  least as well as A.

### Why A comes first even though B is open

- A is necessary to fix under either reading. While step size ignores the
  signal, no signal-side fix can be judged, because every run drifts.
- The fix for A is config-only (section 2).
- The first run carries its own noise-null twin, so it tests A directly.
- Diagnostic D4 tests B offline before any pod time is spent.

### Evidence

| Direction | Observation | Source |
|---|---|---|
| For A | approx_kl follows LR, not advantage scale. J/2: adv std 0.34 -> 0.06 at a constant approx_kl of 0.013. vs-cha22 with a saturated margin has the same KL (0.010-0.013) as M. | evidence-ppo C1 |
| For A | Grad norm 13-15 is clipped to 10 in 66-100% of iterations. Under Muon that cannot change the step size. | evidence-ppo C5 |
| For A | Two same-recipe seeds (ablations D and J) end 16.6k apart by game 4. Trajectory divergence of this size is what a random walk produces. | evidence-runs |
| For A | Teacher KL grows without bound (M 4.9 -> 8.3; J2-resume 0.8 -> 6.5) with explained variance 0.92-0.98. The critic has little left to explain, yet the policy keeps moving. | evidence-ppo C4, C6 |
| For A | Lower LR slows the slide but never stops it: 5 of 5 full-LR arms collapsed, and 4 of 4 runs at 1e-4 slid after iteration 150. | evidence-runs |
| For A | v2 gained only under tiny, bounded steps: LR 1e-6 to 3e-6, clip 1.0 and a KL stop at 0.05. | evidence-v2 |
| Against pure A | vs-cha22 slid faster than M (to 3k versus about 40k) at the *same* per-update KL. So the direction matters and was worse there. | evidence-runs |
| Against pure A | M lost the long-payback assets specifically, not a uniform mix of behaviours. | evidence-econ |
| Against pure A | Ablation F held teacher KL at 0.22 and still lost 6.7k. That is within the 16.6k seed noise, so it is weak evidence. | evidence-runs |
| Untested | No noise-null run exists and no gradient-coherence measurement exists. | evidence-ppo "Open" |

### Why the drift concentrates in unit_kind

This is an inference, tested by D1.

- `ppo_clip_mode: per_player` sums the log-probs of all K unit frames of a
  player-turn, and applies one shared advantage to every frame.
- Each unit head therefore receives K noisy gradient terms per turn, while its
  own share of the credit is about 1/K.
- That fits unit_kind carrying 60-80% of the teacher KL.

### Attribution, split as CLAUDE.md requires

- **Information: not implicated.** The current observation holds every economic
  lever (evidence-econ).
- **Decision: the primary suspect.** The optimizer and credit rules in A and B
  map a weak signal onto large, possibly misdirected moves.
- **Execution: a secondary source of loss.** Parallel unit frames have no
  cross-unit conditioning, a PLANT is voided when seed demand exceeds the seeds
  held, and grammar masks do not check whether an action will have an effect.
  Together these produce BC's 565 ineffective commands. This is a baseline
  inefficiency, not the cause of the slide. Its share of M's rise to 996 is
  unmeasured.
- **Architecture: suspected, not attributed.** The zero-sum critic cannot
  represent the part of the return both seats share. Under term M that part is
  zero, so this matters only if the reward changes.

### Whether self-play can pay for real money at all

- The per-seat gradient of E[margin] does reward a seat that earns more than its
  mirror. So self-play is not blind to money, but the per-seat signal is the
  difference of two noisy banks and has low SNR. Raising the usable SNR is
  exactly the optimizer's job.
- The market caps the target. Town drain is shared and is about 195-211k per
  game at base prices for both seats combined.
- Observed joint banks:

  | Pairing | Joint bank | Split |
  |---|---|---|
  | BC self-play | about 152k | 2 x 76k |
  | cha22 vs BC | about 179k | 129k + 50k |

- cha22's 143-179k comes partly from demand that a weak rival leaves unclaimed.
- A realistic self-play goal is therefore a joint bank of about 180k or more
  (about 90k per seat). That is +15k per seat over BC. Earning is measured by
  **joint bank** as well as own bank.

## 2. The change: run O1, "movement proportional to evidence"

### Config

New file `configs/kaggriculture_2rank_margin_adamw.yaml`. It uses the term-M
recipe of `configs/kaggriculture_4rank_margin.yaml` with the 2-rank batch split
of `configs/kaggriculture_2rank.yaml`: 128 envs and 8 segments per rank, for a
global batch of 256 games. The 2-rank split lets two arms share one 4-GPU pod
at the same global batch.

Deltas from the M recipe:

```yaml
optimizer:
  optimizer: adamw            # was muon (trunk) + adamw (heads)
  learning_rate: 5.0e-06      # = the current head LR; trunk 6e-6/elt Muon -> AdamW 5e-6
  betas: [0.9, 0.999]
  adamw_eps: 1.0e-05
  weight_decay: 0.0           # Muon's wd 0.01 (signal-free shrink) disappears
  lr_schedule:
    schedule: linear_warmup_cosine_decay
    warmup_steps: 250         # 4 steps/iter -> same ~62 iterations as today
    decay_steps: 100_000      # effectively constant, as today
    lr_min_ratio: 0.01
env:
  n_envs: 128                 # per rank, 2 ranks -> 256 global (unchanged global)
rl:
  segments_per_minibatch: 8   # per rank (proven 2-rank memory split)
  gradient_accumulation_steps: 4   # -> 4 optimizer steps/iter, 64 games = 8,192 player-turns per step
  target_kl: 0.005            # guardrail; code checks per minibatch, breaks after the window's step
  # unchanged: normalize_advantages true, clip 0.2 per_player, vf_coef 2, ent 1e-6,
  # gamma 1, gae_lambda 0.9, horizon 64, max_grad_norm 10, teacher last_best 0.005/0.005,
  # checkpoint_freq 10M (owner), reward term M exactly.
```

**Why AdamW makes the step depend on the signal.**
- AdamW's per-coordinate step is lr x m/sqrt(v).
- For a consistent signal, |m|/sqrt(v) is about 1.
- For pure noise it is about sqrt((1-beta1)/(1+beta1)), which is 0.23 at
  beta1 0.9.
- Muon's orthogonalization removes this contrast: the momentum average shrinks
  the noise, then the rescale restores it.

**The movement budget, per iteration (rough, per element).**

| | Current (Muon) | O1 (AdamW) |
|---|---|---|
| Noise | about 16 x 6e-6, correlated by momentum 0.95 | about 4 x 1.2e-6 |
| Consistent signal | about 16 x 6e-6 | about 4 x 5e-6 |

Measured in movement per iteration, the signal-to-noise contrast rises about 4x,
and the per-step gradient SNR doubles (4x the batch per step). The price is
slower learning on a true signal, which D0 calibrates.

**Other choices.**
- **LR 5e-6.** It keeps the heads' LR exactly where it is today, so only the
  trunk rule and the batch change. It lies between v2's successful 1e-6 to 3e-6
  and today's equivalent step. The v3 reference run used AdamW at 1e-4 from BC
  and reached KL 1.89 in its first iteration; that is the upper bound to stay
  far below.
- **target_kl 0.005.** Banks rose only while approx_kl was 0.002-0.004
  (warm-up). This caps a KL spike from a synchronized game phase (v2 saw 19.7).
  With 4 steps, the break (`ppo.py:1361-1365`) fires after the step of the
  exceeding window, so at most one step overshoots.
- **Normalization, teacher and anchor are unchanged in O1.** Under AdamW the SNR
  of the policy term does not depend on its scale. Keeping them fixed keeps O1
  attributable to one mechanism family.

### Code seams, all in the shared `scripts/run_ppo.py` / `ppo.py` path

- **S1 (required): noise-null flag.**
  - Add `rl.advantage_null: Literal["none", "permute"] = "none"` to
    `PPOConfig` (`ppo.py:140`).
  - In `PPOTrainer.update`, after advantages and returns are computed and before
    `_minibatch_indices` (`ppo.py:1325`), when set to `permute`: permute
    `advantages[policy_mask]` with one rank-local `randperm`. Returns, values and
    the teacher are left untouched, so only the advantage-to-action correlation
    is destroyed.
  - Log `diag/advantage_null`.
  - Tests: the permutation preserves the multiset of advantages and the mask;
    the config rejects unknown values.
  - Docs: `docs/rl-api-specs.md` and the training config tests.
  - This is a diagnostic knob. It exposes no identity and does not change the
    reward.
- **S2 (required): economic telemetry.**
  - `run_ppo.py:1822-1835` keeps only bank, margin, steps and winner from
    `env.terminal_metrics`. It drops the counter vectors ("Native diagnostics
    also carry counter vectors").
  - Reduce the engine's existing econ counters per completed-game interval
    (rank-mean) and log them under `econ/*` in nt-probe and W&B, in both the
    training and eval paths. Counters: units sold per item, sale revenue,
    ineffective commands by kind, plant and animal deaths, strawberry
    tile-days, animal-days, land buys, hires per day.
  - Add `econ/joint_bank_mean` (bank_0 + bank_1).
  - Without S2, "which link of the chain breaks first" stays unattributable.
- **S3 (stage 2 only, not in O1): advantage scale floor.**
  - Add `rl.advantage_scale_floor: float | None`.
  - When set, normalize by max(per-update RMS over all ranks, floor), with no
    centering, instead of the per-minibatch mean and std. This is v2's fix
    (evidence-v2 (a)).
  - Take the floor from D1's null distribution, not a guess.

Each implemented seam and the new config need a cookbook adaptation record (note,
folder index and log), plus `just py-prepare`.

### Stage-2 levers, each tied to the diagnostic that would trigger it

| # | Trigger | Lever |
|---|---|---|
| L1 | D1 says B_simple > 8,192 player-turns | Raise `gradient_accumulation_steps` until a step covers B_simple, up to 16 (1 step per iteration, 32,768 player-turns). target_kl then needs `ppo_epochs` 2 to bite. |
| L2 | O1 still drifts in the null twin | S3 advantage floor. |
| L3 | L1 and L2 fail | Fixed BC anchor: `teacher_mode: fixed`, `teacher_init` set to the BC best, `teacher_kl_coef` 0.05-0.1 (ablation arm A kept 41k against 0.1k). The coefficient is not adaptive, but it needs owner confirmation in light of "we dont want adaptive anchor". |
| L4 | D4 shows investment-credit inversion | `gae_lambda` 0.97-1.0 with `horizon` 128-256 (v2: lambda 1 with 256-turn segments worked; 0.98 with 128 stayed flat). This is credit, not reward. Pair it with L1 because variance rises. |
| L5 | O1's target_kl hits or KL spikes cluster by game phase | Desynchronize phases with the existing `rl.truncation_prob` / `truncation_step` (v2 (d)). |

Not proposed: gradient-clip changes (inert under the normalized optimizers),
critic warm-up (arm C collapsed fastest), value clip, and reward changes (the
owner's call).

## 3. Cheap diagnostics, before any long run

The three diagnostics D0, D1 and D4 share one working script,
`ops/earn-money-2026-09-30/probe_signal.py`, and one stored rollout set.

**Where they run.** On one pod GPU that no learner is using: after vs-cha22
stops, or on a free GPU. The CPU fallback is 32 envs, niced, and only with the
owner's OK, because no heavy local runs are allowed.

**Rollout set R.**
- 13 consecutive rollout iterations of 256 games in self-play from
  `/Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt`, collected
  through the trainer's own collection path with no update.
- 720/64 means iterations 1-12 cover one full synchronized game cycle. Iteration
  13 is held out.
- Store the segments, logp, values, rewards, the term-M advantages and the
  teacher targets.
- Repeat the set for M@20M (`checkpoint_00_020_004_864`) as a contrast.

### D0: optimizer response (is movement signal-blind, and which LR?)

- **Question.** Does policy movement depend on whether the advantages carry
  signal, for Muon today and for AdamW in O1?
- **Code path.** Replay R's iterations 1-12 in order through `_update_minibatch`
  / `_step_optimizer` (`ppo.py:1325-1370`, `1654-1661`) from the BC weights with
  a fresh optimizer.
- **Arms.**
  - {Muon 1e-4 / AdamW-heads 5e-6, 16 steps per iteration} and {AdamW 5e-6,
    1e-5, 2e-5, 4 steps per iteration},
  - each with {real, permuted} advantages.
- **Measure.** After each iteration: KL(BC || updated) on iteration 13's
  observations, per head (unit_kind, market_kind, unit_item and so on), and
  approx_kl per update.
- **Discriminating readout.** R_move = KL_real / KL_permuted after 12
  iterations.
  - Mechanism A predicts R_move of about 1 for Muon.
  - O1 is viable at the largest LR with R_move >= 3 and per-update approx_kl
    <= 0.004. That LR replaces the 5e-6 default in O1.
- **Limit.** After the first iteration, the replay is slightly off-policy.

### D1: gradient signal-to-noise (is the per-step direction mostly noise?)

- **Code path.** R, BC weights, no stepping.
- **Method.** For each of the 12 game phases:
  - split the 256 games into 16 disjoint 16-game minibatches (today's step) and
    4 disjoint 64-game minibatches (O1's step);
  - compute the policy, value and teacher gradients separately, flattened per
    group (Muon trunk matrices; AdamW heads, with unit_kind and market_kind
    reported separately).
- **Report.**
  - The mean pairwise cosine between minibatch gradients, real against the
    permuted null (500 permutations, 95th percentile).
  - B_simple = tr(Sigma)/|G|^2 (McCandlish) from the two batch sizes, in
    player-turns.
  - The cosine between the policy and value gradients on the trunk
    (interference).
  - The same readouts at M@20M.
- **Readout.**
  - If the 16-game policy cosines are inside the null band in most phases, the
    current step is noise-driven, which supports A.
  - B_simple sets the batch for L1.
  - If the signal appears only in the phase that contains terminals, record it;
    it bears on L5 and on an analogue of v2's `--require-outcomes`.
  - If unit_kind's cosine sits nearer the null than market_kind's, that
    supports the per_player credit-dilution inference.
- **Abandon trigger.** If even the full 256-game policy gradient cannot be told
  apart from the null (split-half cosine inside the null band) in every phase,
  no optimizer can extract signal. The design then stops and returns to
  signal-side options, which are the owner's call.

### D4: investment-credit sign (is mechanism B real?)

- **Code path.** R (BC weights, term M) together with the stored action tensors.
- **Method.** Tag the player-turns whose market orders include BUY_LAND,
  BUY_ANIMAL or BUY_SEED STRAWBERRY. Match them with same-day player-turns that
  have no investment.
- **Compare**, per family, as mean and 95% CI with denominators:
  - the GAE advantage at lambda 0.9, as trained;
  - the Monte Carlo advantage at lambda 1 (return-to-go minus V).
- **Readout.**
  - If the lambda 0.9 advantage on investment turns is significantly below 0
    while the lambda 1 advantage is >= 0 (or significantly higher), credit is
    inverted and B is supported. O1 then becomes necessary but not sufficient,
    and L4 joins the first run as a third arm (O1 plus lambda).
  - If both are about 0 or positive, B is not supported.
- **Limit.** 256 games per set gives a wide lambda 1 CI.

### D2: a Muon real-vs-null probe on the pod (optional)

Run it only if D0 and D1 disagree. It is the current M recipe, 2 + 2 ranks, S1
on and off, 150 iterations, and it attributes the past slide. The null twin in
O1 already answers the forward question.

## 4. The first run: the O1 pair, both self-play

- **Mechanism tested.** With a signal-sensitive optimizer (AdamW, 4x batch per
  step, target_kl 0.005), the policy moves only when the advantage carries
  signal. Held away from noise, the self-play signal raises own and joint bank
  above BC.
- **Arms**, concurrent on the 4-GPU pod, same seed, same start:
  - O1-real: `configs/kaggriculture_2rank_margin_adamw.yaml`, ranks on GPUs 0-1.
  - O1-null: the same config plus `-o rl.advantage_null=permute`, on GPUs 2-3.
- **Inputs.**
  - Start: `--load-model-weights /Users/poonszesen/kg-v3-runs/bc-best/checkpoint_bc_best.pt --load-model-weights-mode model_only`.
    This keeps the BC critic head, which is the only critic that produced early
    rises, and a fresh optimizer.
  - Teacher: last_best, which is the BC best until a promotion.
  - Code: `scripts/run_ppo.py` shared path with S1 and S2 landed and
    `just py-prepare` green.
  - W&B: project kg-v3, one group, both runs.
  - The LR comes from D0. The default is 5e-6.
- **Baseline.** B0 is each arm's own-bank mean over its first 2 completed-game
  intervals. An interval is about 11 iterations: one synchronized game cycle,
  with 128 games on rank 0.

**Expected discriminating observations** (by 610 iterations, which is 10M global
env steps, and at the first eval):

| # | O1-real | O1-null | Reading and next step |
|---|---|---|---|
| 1 | Own and joint bank at or above B0 and rising, teacher KL growing, eval >= 0.55 against BC | Bank within B0 ± 8k; teacher KL <= 25% of real | A was the block and movement now follows the signal. Continue to 20M. |
| 2 | Flat, KL < 0.2 | Flat | Steps too small. Raise the LR x2, once. |
| 3 | Slides | Slides at a similar rate | Drift survives AdamW (value/trunk interference?). Try S3/L2, then L3. |
| 4 | Slides | Holds | Signal anti-aligned: B dominates. Use L4 (credit), not the optimizer. |
| 5 | Holds | Slides | Noise remains but the real signal protects. Use L1 (batch to B_simple). |

No v3 PPO run has yet been above its start for more than 4 intervals, or won an
eval against its start; the best was 0.141. So outcome 1 is unambiguous.

- **Loss condition, either arm.** Any of:
  - rank-0 own-bank mean < B0 - 15k for 3 consecutive intervals (15k sits just
    under the 16.6k seed gap and exceeds every in-run rise seen);
  - teacher/kl > 2 nats while own bank <= B0 (movement without earning);
  - O1-real's eval at 10M against BC has a win rate < 0.4.
- **Stopping condition.** The first of: the loss condition on O1-real (stop both
  arms and record the outcome row), 10M global env steps plus the eval, or a
  crash or watchdog. The null arm stops at 10M in every case. O1-real continues
  to 20M on 4 ranks only in outcome 1.
- **Metrics to watch**, per interval:
  - own_bank_mean and `econ/joint_bank_mean`;
  - the S2 counters: units sold per item, strawberry tile-days, animal-days,
    land buys, deaths, ineffective commands. The first counter to move locates
    the break in the chain;
  - teacher/kl in total and per head (unit_kind share);
  - policy/approx_kl, `policy/target_kl_exceeded` rate by game phase, and
    optimizer steps per iteration;
  - raw advantage std, clipfrac, explained variance, grad_norm (informational);
  - eval win rate and candidate against last_best bank, with 64 games, seat and
    seed recorded;
  - the same set for O1-null.
- **Custody.** The receipt goes under `ops/earn-money-2026-09-30/o1/`, with the
  config hash, tree SHA, W&B ids and checkpoint paths. Weights stay out of git.

## 5. When to abandon this design

1. **D0.** At no LR in {5e-6, 1e-5, 2e-5} does AdamW reach R_move >= 3. The
   optimizer cannot separate signal from noise at this batch: go to L1 first,
   and drop optimizer-first if L1 (up to 1 step per iteration) also fails.
2. **D1.** The full-batch policy gradient is inside the null band in every
   phase. There is nothing to extract; this is a signal-side problem and the
   owner's reward call.
3. **O1 outcome 4, or D4 inversion that O1 plus L4 does not cure.** The block is
   credit or reward direction, not the optimizer.
4. **O1 outcome 3 after L2 and L3.** Drift is not optimizer noise. Suspect
   value/trunk interference through the shared trunk (D1's policy-value cosine)
   and redesign as architecture.
5. **Outcome 2 twice** (two LR rungs up, still frozen at 10M each). The
   optimizer holds the policy but finds no money; the signal is too weak for
   self-play under term M at this batch.

## 6. Limits and unknowns

- Every prior run is a single seed. The only replicate pair differs by 16.6k.
  O1 is also a single seed per arm, and its null twin is the control, not a
  replicate.
- The per-element step equivalence between Muon and AdamW (about 6e-6 against
  5e-6) is an estimate from the matrix shapes. D0 measures the real value.
- target_kl in this code checks KL per minibatch and breaks after the window's
  step, so it overshoots by up to one step.
- AdamW for the trunk departs from Isaiah's Muon recipe. The justification is the
  regime: a fine-tune at a narrow BC optimum instead of training from scratch.
  If the owner prefers Isaiah alignment, L1 plus S3 under Muon is the fallback.
  It still cannot stand still.
- A self-play ceiling of about 90k per seat is an inference from the shared
  town drain. cha22 against cha22 is unmeasured.
- The per_player credit dilution across K unit frames is an inference. D1's
  per-head readout tests it; no fix is proposed yet.
