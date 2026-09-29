# PPO collapse ablation study: final report (control, A–J)

**Scope.** This report covers the attribution study only: the 6.2 control
(iterations 1–46) and ablations A–J. The main run that is going in parallel
is not part of it. Every arm is a pre-landing diagnostic on
`kg/pod-ppo-prelanding` `e74d67e` (0 porcelain lines on the pod), 2 ranks,
`configs/kaggriculture_2rank.yaml`, BC best `fd854587…6f51`. None of it is
Codex-verified. It supports no selection, ranking or submission.
Self-play banks are not strength; no held-out evaluation ran in any arm
(`rl.eval_replay_games=0`, and the in-trainer last-best evaluation runs only
at `checkpoint_freq` 20M env steps).

**Arm K.** The brief asks for J and K and a J-vs-K seed-variance read. No K
receipt exists at the time of writing: there is no `ablate-K*` directory in
this worktree or any other worktree of the repository, and no commit on any
branch mentions an ablation K. The seed-variance read below therefore uses
the one pair that exists: D (env seed 0) and J (env seed 1,000,000), which
run the same recipe. If K lands later, it goes in a new file (see
`README.md`, frozen receipts).

Sources: `comparison.md` (round 1), `attribution.md` (`d7bc061`, control and
A–I), `ablate-J-combined/result.md` (`298c73e`), and each arm's
`pod-receipts/iterations.tsv`. The numbers in the table below were
recomputed for this report from the committed tables (`extract_iterations.py`
output, which reproduces byte for byte from each `run.log`; J's was
rechecked here) and equal `attribution.md` and J's result.md.

## Measurement

### Denominators common to every arm

- 46 complete iterations on both ranks; 736 optimizer steps (16 per
  iteration); 0 nonfinite metrics in every arm.
- 753,664 global env steps (46 × 16,384); 256 envs globally (128 per rank).
- 1,024 completed self-play games (`train/total_games_played` 1024: four
  game phases of 256 games, ending at iterations 12, 23, 34 and 45). The
  6.2 control ran 235 iterations (3,850,240 env steps, 5,120 games); only its
  iterations 1–46 enter this comparison.
- Banks are the completed-game final banks at iterations 12, 23, 34 and 45,
  as the mean of the two seats' rank-reduced means. Both seats run the same
  policy.

### All arms

LR "full" is Muon 0.002 / AdamW 1e-4; "/ 10" is Muon 0.0002 / AdamW 1e-5.
Head "BC" is `--load-model-weights-mode model_only`; "fresh" is
`model_fresh_critic_head`. Teacher KL is `teacher/kl` at iteration 45
(game-4 end) / 46. EV is the mean of `train/explained_variance` over
iterations 1–46. Trunk is ||θ − θ0|| / ||θ0|| at iteration 46, actor-only /
shared (A–C from `checkpoint_final.pt` against the BC best; D–J in-run; the
control's iteration-46 weights were not kept).

| Arm | LR | Head | Other change | Game 1 | Game 2 | Game 3 | Game 4 | G4 − G1 | Teacher KL 45 / 46 | EV | Approx KL mean | Trunk actor / shared | W&B | Spend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Control 6.2 | full | fresh | — | 73,182 | 54,960 | 7,484 | 92 | −73,090 | 3.25 / 5.28 | 0.54 | 0.068 | n/a | [7k07gp7c](https://wandb.ai/spoon/kg-v3/runs/7k07gp7c) | $1.98 (235 it) |
| A | full | fresh | `teacher_kl_coef` 0.1 | 71,438 | 60,140 | 50,198 | 41,432 | −30,006 | 2.49 / 2.14 | 0.37 | 0.130 | 4.98e-2 / 7.41e-2 | [y8j6vzky](https://wandb.ai/spoon/kg-v3/runs/y8j6vzky) | $0.57 |
| B | / 10 | fresh | — | 73,284 | 65,373 | 63,801 | 62,212 | −11,072 | 1.36 / 0.64 | 0.29 | 0.0081 | 5.12e-3 / 8.02e-3 | [32pahqok](https://wandb.ai/spoon/kg-v3/runs/32pahqok) | $0.56 |
| C | full | fresh | value-only critic warm-up, it 1–22 | 70,750 | 4 | 0 | 0 | −70,750 | 8.26 / 35.2 | 0.69 | 0.057 | 2.89e-2 / 7.26e-2 | [lpt11y9j](https://wandb.ai/spoon/kg-v3/runs/lpt11y9j) | $0.56 |
| D | / 10 | BC | — (seed 0) | 72,239 | 76,413 | 79,571 | 83,790 | +11,551 | 1.41 / 1.45 | 0.84 | 0.0093 | 5.07e-3 / 8.43e-3 | [hftcr4xa](https://wandb.ai/spoon/kg-v3/runs/hftcr4xa) | $0.60 |
| E | / 10 | fresh | `vf_coef` 0.5 | 75,198 | 73,961 | 72,933 | 66,989 | −8,209 | 1.14 / 0.63 | −0.10 | 0.0083 | 5.10e-3 / 7.86e-3 | [hi2lqsrx](https://wandb.ai/spoon/kg-v3/runs/hi2lqsrx) | $0.52 |
| F | / 10 | fresh | `teacher_kl_coef` 0.1 | 73,460 | 71,944 | 70,798 | 66,810 | −6,650 | 0.22 / 0.21 | 0.23 | 0.0079 | 5.14e-3 / 7.78e-3 | [nur61v3a](https://wandb.ai/spoon/kg-v3/runs/nur61v3a) | $0.56 |
| G | / 10 | fresh | critic stop-gradient | 70,118 | 64,582 | 66,544 | 63,414 | −6,704 | 0.69 / 0.58 | 0.16 | 0.0077 | 5.17e-3 / 7.81e-3 | [yz34n7i6](https://wandb.ai/spoon/kg-v3/runs/yz34n7i6) | $0.71 |
| H | full | BC | — | 76,738 | 60,066 | 20,616 | 74 | −76,664 | 4.66 / 4.92 | 0.79 | 0.089 | 4.88e-2 / 8.14e-2 | [ksqdtvm0](https://wandb.ai/spoon/kg-v3/runs/ksqdtvm0) | $0.47 |
| I | full | fresh | critic stop-gradient | 70,870 | 49,994 | 4,801 | 103 | −70,767 | 2.31 / 3.21 | 0.06 | 0.050 | 4.70e-2 / 8.59e-2 | [o9c5ji4v](https://wandb.ai/spoon/kg-v3/runs/o9c5ji4v) | $0.51 |
| **J** | / 10 | BC | — (env seed 1,000,000) | 72,033 | 74,644 | 79,219 | 67,204 | −4,829 | 1.45 / 1.28 | 0.84 | 0.0093 | 5.13e-3 / 8.45e-3 | [pkz85wlw](https://wandb.ai/spoon/kg-v3/runs/pkz85wlw) | $0.60 |
| K | — | — | no receipt | | | | | | | | | | | |

Spend from the receipts: A–J $5.66 in total (round 1 A–C $1.69; rounds 2–3
D–J $3.97, which matches the orchestrator's $3.98 study figure to rounding).
With the control's $1.98, $7.64. All at $4.18/h on pod `aki4vy8kpfldpa`.

### Seed-variance read (D against J, same recipe, disjoint games)

| | Game 1 | Game 2 | Game 3 | Game 4 | G3 − G1 | G4 − G1 |
|---|---|---|---|---|---|---|
| D (seed 0) | 72,239 | 76,413 | 79,571 | 83,790 | +7,332 | +11,551 |
| J (seed 1,000,000) | 72,033 | 74,644 | 79,219 | 67,204 | +7,186 | −4,829 |
| \|D − J\| | 206 | 1,769 | 352 | **16,586** | 146 | 16,380 |

- Trunk movement agreed within 1 % at every audited iteration (shared 8.43e-3
  against 8.45e-3 at iteration 46). EV mean was 0.84 in both. Teacher KL at
  the game-4 end was 1.41 and 1.45.
- Games 1–3 agree within 1.8k. Game 4 differs by 16.6k, 3.3x the 5.1k
  game-1 spread that `attribution.md` used as its noise floor.
- This is two draws, not a variance estimate. It shows that the game-1
  spread understates game-4 variance for at least this recipe.

### LR schedule position (measured for this report)

The `lr` column of every table is the Muon learning rate. The schedule warms
up linearly over 1,000 optimizer steps, which is 62.5 iterations at 16 steps
per iteration. **Every arm's 46 iterations lie inside the warm-up.** At
iteration 46 the LR was 0.736 of its peak: 1.472e-4 in the LR / 10 arms and
1.472e-3 at full LR. The 6.2 control reached its 0.002 peak at iteration 63
(`../6.2/pod-receipts/run.log`). At full LR the control's game-3 bank was
already 7.5k at iteration 34, where the Muon LR was 1.09e-3. No arm has run
at the LR / 10 recipe's peak of 2e-4.

## Interpretation (inferred from the measurements above)

### Causes ranked by evidence

1. **Step size: strong.** LR / 10 separates survival from collapse in all
   three matched pairs (control → B, H → D, I → G: game-4 banks from about
   0.1k to 62–84k). Every full-LR arm except A was below 1k by game 4 (C
   by game 2). Parameter movement scales with LR (0.09–0.11x at 0.1x LR)
   and not with the head, the value weight or the value path. D and J
   (LR / 10, BC head) also survived 46 iterations at two seeds. **Limit:**
   the comparison is between warm-up ramps, since no arm reached peak LR.
   The collapse is shown for LRs from about 7e-4 to 1.5e-3 at full LR.
   Survival is shown only up to 1.47e-4.
2. **Critic quality from the BC head: moderate, and only through game 3.**
   At LR / 10 the BC head gives EV 0.84 at both seeds, against 0.29 or less
   for every fresh-head arm. With the same trunk movement and matched-phase
   teacher KL, the bank trend through game 3 was +7.3k and +7.2k at two
   seeds, where B lost 9.5k and E, F and G lost 2.3–3.6k. D's game-4 rise
   did not reproduce in J. So "the BC head turns the trend upward" holds
   through game 3 at two seeds and fails at game 4 at one of two. The
   head's three coupled changes (initial critic function, teacher value
   target, initial value-gradient size) are still not separated. At full LR
   the head did not prevent collapse (H).
3. **Anchor strength (`teacher_kl_coef` 0.1): helps at full LR only.** A
   delayed the collapse (+41.3k at game 4 over the control) but still fell
   30k. At LR / 10 (B → F) the bank effect, +4.6k, is under both the 5.1k
   game-1 spread and the 16.6k same-recipe game-4 gap. F held teacher KL at
   0.22, a sixth of B's, and its banks still fell 6.7k. Closeness to BC
   does not by itself hold the economy.
4. **Value-side levers: no effect shown.** Stop-gradient (B → G +1.2k,
   control → I 0) and `vf_coef` 0.5 (B → E +4.8k) are inside the spread.
   Both weakened the critic (EV 0.16 and −0.10). The BC-handoff hypothesis
   that the fresh head's large value gradient drags the BC actor is
   contradicted. Trunk and actor movement at equal LR were the same with
   the fresh head (B) and the BC head (D, J). On real rollouts the BC head's
   critic gradient norm (D 0.47–1.19, J 0.33–2.68) exceeded the fresh
   head's (C 0.01–0.21). The synthetic probe had predicted the reverse
   (0.118 against 12.4).
5. **Critic warm-up (C): harmful.** It collapsed fastest of all arms.

### What is still unresolved

- **Post-warm-up behaviour.** No arm ran at its peak LR. A J-recipe run
  reaches 2e-4 at iteration 63 and stays near it: the cosine decay runs over
  400k steps, 25,000 iterations. J's game-4 drop (−12.0k from game 3)
  happened while the LR was rising from 1.09e-4 (iteration 34) to 1.44e-4
  (iteration 45). Whether the
  recipe survives at 2e-4 is untested.
- **Whether J's game-4 drop starts a slower collapse or is noise.** Teacher
  KL at game ends rose monotonically in D (0.046, 0.29, 0.81, 1.41) and J
  (0.036, 0.27, 0.63, 1.45). Unit-kind KL grew faster than market-kind KL
  (J at game 4: 1.20 against 0.11).
- **Seed variance.** Only two same-recipe draws exist, and no arm besides
  D/J has a replicate. So no single-seed game-4 delta below about 17k
  ranks: B → E, B → F, B → G and B → J all fall under it, and B → D's
  +21.6k is only 1.3x the gap.
- **The BC head's three coupled changes**, and the interactions of the BC
  head with the anchor or stop-gradient. None was run.
- **Strength.** There is no win rate or bank margin against BC or any other
  opponent, by seat, with denominators.
- **Per-family attribution** of the bank changes (unit, market and other
  action families) was not measured.
- **Rollout time.** J's rollout was 6.06 s per iteration against D's
  5.10 s. This was not investigated.

## Recommended recipe

**Recipe J = BC best loaded with `model_only` (BC critic head kept) plus
both learning rates / 10. Everything else stays at config default.** It is
recommended as the survival recipe: it is the only configuration that kept
the self-play economy at or above the BC level through game 3 at two seeds
with a critic that explains most of the return variance. It is **not** shown
to improve on BC. Its game-4 result diverged between the seeds.

- `--load-model-weights <bc best, fd854587…6f51> --load-model-weights-mode
  model_only`. `model_only` is already `run_ppo`'s default. The README
  and the BC-handoff Reference currently prescribe `model_fresh_critic_head`
  for the BC launch, and this study contradicts that at LR / 10.
- `-o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001`. The same
  schedule applies: warm-up 1,000 steps, cosine to 400k, `lr_min_ratio`
  0.01. `vf_coef` 2.0, `teacher_kl_coef` 0.005, `teacher_value_coef` 0.005
  and `ent_coef` 1e-6 are unchanged. No stop-gradient, warm-up or anchor
  change.
- **Alignment note.** This departs from Isaiah's `scaling_6m` LRs for a
  case Isaiah never had: he never started PPO from imitation
  (`docs/write-up.md`). Under the recipe-alignment Decision it is a
  residual difference with a measured reason, the full-LR collapse from
  the BC start in the control, H, I and C.

### Loss conditions (proposed, anchored to observed values, not validated)

1. Completed-game banks below the run's own game-1 bank for two consecutive
   games. J met this once, at game 4; two in a row has not been seen at
   LR / 10.
2. Game-end teacher KL above 2.3, the lowest game-4-end value among the
   full-LR arms that collapsed (I 2.31). D and J were 1.41 and 1.45.
3. In the first two completed games after the warm-up ends (iteration 63
   and later at this global batch), banks fall more than 16.6k (the
   same-recipe game-4 gap) below the last pre-peak game.
4. Any nonfinite metric, or an iteration with other than 16 optimizer
   steps.
5. A held-out evaluation of the final checkpoint against the BC best and at
   least one other opponent shows no gain in win rate or bank margin (by
   seat, with denominators). Then the recipe has not produced strength,
   whatever the self-play banks show.

### At 8 ranks

`configs/kaggriculture_8rank.yaml` keeps the same **global** shape as the
2-rank config: 256 envs (32 per rank), 16 optimizer steps per iteration, 16
global segments per optimizer step (2 per rank, accumulation 1) and 16,384
env steps per iteration. Its optimizer, schedule and PPO coefficients are
identical. The per-step learning rate therefore carries over unchanged.
Recipe J at 8 ranks is:

```
torchrun --nproc-per-node 8 scripts/run_ppo.py configs/kaggriculture_8rank.yaml <out> \
  --load-model-weights <bc best> --load-model-weights-mode model_only \
  -o optimizer.muon_lr=0.0002 optimizer.adamw_lr=0.00001
```

On `e74d67e` the 8-rank config still sets `eval_replay_games: 8`, so a
pre-landing launch must add `-o rl.eval_replay_games=0`; the integration
tip's config already sets 0. The recipe exists at 8 ranks in form only:

- **No 8-rank PPO run has happened.** Plan task 6.3b (8-rank memory smoke
  and qualification) has not started. Creating the 8-GPU pod needs a stated
  live price and the owner's approval.
- **Different games.** Training streams draw `base_seed + rank + k ×
  world_size`, so 8 ranks play different games from any 2-rank arm.
- **Different code.** The integration tip adds W&B gating, bank telemetry
  and checkpoint-key validation over `e74d67e`. The `ppo.py` diff reads as
  telemetry and validation only. That is from inspection, not from a run.
- **Launcher.** The ablation launcher `run_d.sh` hard-codes 2 ranks, the
  2-rank config and `timeout 600`, so an 8-rank or long run needs its own
  launcher.

## Code to make the recipe first-class (not implemented here)

The recipe is two CLI overrides plus a load mode that the docs currently
advise against. To make it a named, tested preset, on a branch from the
integration tip with its own cookbook adaptation record:

1. **Fine-tune presets:** `configs/kaggriculture_2rank_finetune.yaml` and
   `configs/kaggriculture_8rank_finetune.yaml`. Each is the ranked config
   with `optimizer.muon_lr: 0.0002` and `optimizer.adamw_lr: 1.0e-05`, and
   a header naming this report, `attribution.md` and the warm-up limit.
2. **Test:** each preset loads through `FullConfig` and equals its base
   config except those two keys. This follows the existing pin that equates
   `kaggriculture_1gpu_eager.yaml` to the 2-rank config. Kill it with two
   mutations: changing any other key, and dropping an LR change.
3. **Warm-start mode as documented:** README's BC launch and the SLURM
   example use `model_only`, the default. `model_fresh_critic_head` stays
   available as the diagnostic comparison, and its help text says so. Test
   (already present for mode wiring): `main`'s fresh launch with
   `model_only` passes an empty `fresh_state_keys`, and `warm_start.json`
   records mode `model_only`. Add one assertion that a BC-trainer checkpoint
   loaded with `model_only` keeps `critic_head.*` bit-identical.
4. **Optional run guard:** `run_ppo` records the preset name, or both LRs,
   in `warm_start.json`, so a run's receipt shows it used the fine-tune LRs.
5. **Docs and cookbook:** `docs/rl-api-specs.md` or README (warm-start
   section), and the configs Reference, cite the preset. The BC-handoff
   Reference's critic decision is corrected, which the cookbook finding in
   this change does.

## Receipts

Every result.md in this study (`../6.2/result.md`, `ablate-*/result.md`)
ends with a closing section: outcome, denominators, W&B link, spend and
gaps. `SHA256SUMS` in this directory covers every receipt file, including
`../6.2/`, and `README.md` marks them frozen.
